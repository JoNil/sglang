# Jetson Thor development image.
#
# NVIDIA's Jetson-compatible SGLang image already contains the CUDA 13.3,
# PyTorch, FlashInfer CuTeDSL, and aarch64 binary stack.  Overlay this fork's
# Python sources so the SM110 kernel remains JIT-compiled for the actual Thor
# instead of baking an SM121 cubin on another machine.
ARG BASE_IMAGE=nvcr.io/nvidia/sglang:26.06-py3@sha256:f1e23b1c96d7e04d061c76b179f81aa32fcef367590a90f6079a0bf899dc4300
FROM ${BASE_IMAGE}

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="JoNil SGLang for Jetson Thor" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.description="DeepSeek-V4 MXFP4 SM110 W4A16 kernel overlay"

COPY python /opt/sglang-thor/python

# Add an indexed SM100-family grouped-GEMM entry point. Decode routing passes
# a compact list of active expert IDs, so CUTLASS no longer constructs and
# schedules 256 expert groups when only a small subset has tokens.
COPY flashinfer-indexed/group_gemm_mxfp4_groupwise_sm100.cuh \
    /usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/gemm/group_gemm_mxfp4_groupwise_sm100.cuh
COPY flashinfer-indexed/group_gemm_mxfp4_groupwise_sm100.cu \
    flashinfer-indexed/group_gemm_mxfp4_groupwise_sm100_kernel_inst.jinja \
    flashinfer-indexed/group_gemm_sm100_binding.cu \
    flashinfer-indexed/group_gemm_mxfp4_indexed_thor.cu \
    flashinfer-indexed/group_gemm_mxfp4_indexed_thor_binding.cu \
    /usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/

# FlashInfer 0.6.15 ships the SM100-family blockwise FP8 CUTLASS source and
# already generates sm_110a code, but its Python capability gates omit Thor.
# Enable the source-JIT path without changing the checkpoint's FP8 values or
# FP32 128x128 scales.
COPY flashinfer-indexed/flashinfer-gemm-sm110.patch /tmp/flashinfer-gemm-sm110.patch
RUN patch --batch --forward -p1 \
      -d /usr/local/lib/python3.12/dist-packages/flashinfer/gemm \
      < /tmp/flashinfer-gemm-sm110.patch && \
    rm /tmp/flashinfer-gemm-sm110.patch

# The generic W8 schedule consumes 230.4 KiB of dynamic shared memory on Thor.
# Keep the arithmetic tile but cap its pipeline at five stages (101.38 KiB),
# allowing two CTAs per SM. The SGLang runtime selects this source-JIT module
# only when SGLANG_THOR_W8_PIPELINE_STAGES=5; zero retains FlashInfer's AOT
# module for an exact same-image rollback/control arm.
COPY flashinfer-indexed/flashinfer-w8-sm110-stage5.patch \
    /tmp/flashinfer-w8-sm110-stage5.patch
RUN patch --batch --forward -p1 \
      -d /usr/local/lib/python3.12/dist-packages/flashinfer/data/include \
      < /tmp/flashinfer-w8-sm110-stage5.patch && \
    rm /tmp/flashinfer-w8-sm110-stage5.patch

ENV PYTHONPATH=/opt/sglang-thor/python \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    SGLANG_THOR_CUDA_GRAPH_MAX_BS=2

# Syntax-check the overlay without importing CUDA or creating a GPU context.
RUN PYTHONPYCACHEPREFIX=/tmp/sglang-thor-pyc \
    python -m compileall -q /opt/sglang-thor/python/sglang && \
    rm -rf /tmp/sglang-thor-pyc

WORKDIR /opt/sglang-thor

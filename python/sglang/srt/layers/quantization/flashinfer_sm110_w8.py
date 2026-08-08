"""Thor-specific low-M FlashInfer W8 module selection.

FlashInfer's generic SM100-family W8 kernel lets automatic shared-memory
carveout select twelve pipeline stages on Thor.  That consumes 230.4 KiB per
CTA and limits the kernel to one resident CTA per SM.  The measured five-stage
variant uses 101.38 KiB and permits two resident CTAs without changing tile or
arithmetic semantics.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_THOR_W8_PIPELINE_STAGES"
_MODULE_NAMES = {
    5: "gemm_sm100_w8_sm110_stage5",
    6: "gemm_sm100_w8_sm110_stage6",
}
_STAGE6_SHAPES_ENV = "SGLANG_THOR_W8_STAGE6_SHAPES"
_REQUIRED_SOURCES = {
    "gemm_groupwise_e4m3_bf16_majorfalse_mma1_sm100.cu",
    "gemm_groupwise_e4m3_bf16_majorfalse_mma2_sm100.cu",
    "gemm_groupwise_sm100.cu",
    "gemm_sm100_binding.cu",
}


@lru_cache(maxsize=1)
def get_thor_w8_pipeline_stages() -> int:
    value = int(os.environ.get(_ENV_NAME, "0"))
    if value not in (0, 5, 6, 56):
        raise ValueError(f"{_ENV_NAME} must be 0, 5, 6, or 56, got {value}")
    return value


def use_thor_stage5_w8() -> bool:
    return get_thor_w8_pipeline_stages() in (5, 6, 56) and (
        torch.cuda.get_device_capability() == (11, 0)
    )


@lru_cache(maxsize=1)
def get_thor_w8_stage6_shapes() -> frozenset[tuple[int, int, int]]:
    """Parse exact M:N:K graph shapes selected for the six-stage module."""

    raw = os.environ.get(_STAGE6_SHAPES_ENV, "").strip()
    if not raw:
        return frozenset()
    shapes = set()
    for item in raw.split(","):
        fields = item.strip().split(":")
        if len(fields) != 3:
            raise ValueError(
                f"{_STAGE6_SHAPES_ENV} entries must be M:N:K, got {item!r}"
            )
        shape = tuple(int(field) for field in fields)
        if any(dimension <= 0 for dimension in shape):
            raise ValueError(
                f"{_STAGE6_SHAPES_ENV} dimensions must be positive, got {item!r}"
            )
        shapes.add(shape)
    return frozenset(shapes)


def select_thor_w8_pipeline_stages(
    q_input: torch.Tensor, weight: torch.Tensor
) -> int:
    mode = get_thor_w8_pipeline_stages()
    if mode in (5, 6):
        return mode
    if mode != 56:
        raise RuntimeError("Thor W8 stage selection was requested while disabled")
    shape = (q_input.shape[0], weight.shape[0], weight.shape[1])
    return 6 if shape in get_thor_w8_stage6_shapes() else 5


@lru_cache(maxsize=2)
def get_thor_w8_module(stages: int):
    if not use_thor_stage5_w8():
        raise RuntimeError("Thor's tuned W8 module was requested while disabled")
    if stages not in _MODULE_NAMES:
        raise ValueError(f"Unsupported Thor W8 stage count: {stages}")

    from flashinfer.compilation_context import CompilationContext
    from flashinfer.jit.gemm import gen_gemm_sm100_module

    base = gen_gemm_sm100_module()
    sources = [source for source in base.sources if source.name in _REQUIRED_SOURCES]
    found = {source.name for source in sources}
    if found != _REQUIRED_SOURCES:
        raise RuntimeError(
            "Could not isolate FlashInfer's W8 sources: "
            f"missing={sorted(_REQUIRED_SOURCES - found)}"
        )
    cuda_flags = list(base.extra_cuda_cflags or [])
    if stages == 6:
        cuda_flags.append("-DSGLANG_THOR_W8_STAGE6=1")
    spec = replace(
        base,
        name=_MODULE_NAMES[stages],
        sources=sources,
        extra_cuda_cflags=cuda_flags,
    )

    # FlashInfer 0.6.12's generic compilation context lists SM100 and SM120
    # families for this source set but omits the binary-compatible SM110
    # family.  Narrowly add it while resolving this module's cache/build path.
    original_get_nvcc_flags_list = CompilationContext.get_nvcc_flags_list

    def get_nvcc_flags_list_with_sm110(self, supported_major_versions=None):
        if (
            supported_major_versions is not None
            and 10 in supported_major_versions
            and 12 in supported_major_versions
            and 11 not in supported_major_versions
        ):
            supported_major_versions = [*supported_major_versions, 11]
        return original_get_nvcc_flags_list(self, supported_major_versions)

    CompilationContext.get_nvcc_flags_list = get_nvcc_flags_list_with_sm110
    try:
        logger.info("Loading Thor %s-stage W8 module %s", stages, _MODULE_NAMES[stages])
        return spec.build_and_load()
    finally:
        CompilationContext.get_nvcc_flags_list = original_get_nvcc_flags_list


def thor_stage5_gemm_fp8_nt_groupwise(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Invoke only the reduced W8 binding without replacing W4's module getter."""

    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, _get_cache_buf

    output = torch.empty(
        (q_input.shape[0], weight.shape[0]),
        device=q_input.device,
        dtype=out_dtype,
    )
    workspace = _get_cache_buf(
        "gemm_fp8_nt_groupwise_workspace", DEFAULT_WORKSPACE_SIZE, q_input.device
    )
    stages = select_thor_w8_pipeline_stages(q_input, weight)
    get_thor_w8_module(stages).gemm_fp8_nt_groupwise(
        workspace,
        q_input,
        weight,
        x_scale,
        weight_scale,
        output,
        1,
        128,
        128,
        "MN",
        1,
    )
    return output

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
_MODULE_NAME = "gemm_sm100_w8_sm110_stage5"
_REQUIRED_SOURCES = {
    "gemm_groupwise_e4m3_bf16_majorfalse_mma1_sm100.cu",
    "gemm_groupwise_e4m3_bf16_majorfalse_mma2_sm100.cu",
    "gemm_groupwise_sm100.cu",
    "gemm_sm100_binding.cu",
}


@lru_cache(maxsize=1)
def get_thor_w8_pipeline_stages() -> int:
    value = int(os.environ.get(_ENV_NAME, "0"))
    if value not in (0, 5):
        raise ValueError(f"{_ENV_NAME} must be 0 or 5, got {value}")
    return value


def use_thor_stage5_w8() -> bool:
    return get_thor_w8_pipeline_stages() == 5 and torch.cuda.get_device_capability() == (
        11,
        0,
    )


@lru_cache(maxsize=1)
def get_thor_stage5_w8_module():
    if not use_thor_stage5_w8():
        raise RuntimeError("Thor's five-stage W8 module was requested while disabled")

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
    spec = replace(base, name=_MODULE_NAME, sources=sources)

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
        logger.info("Loading Thor five-stage W8 module %s", _MODULE_NAME)
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
    get_thor_stage5_w8_module().gemm_fp8_nt_groupwise(
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

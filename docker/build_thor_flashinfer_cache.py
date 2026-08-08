"""Build and package the two Thor-only FlashInfer JIT cache entries."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from flashinfer.gemm import gemm_base
from flashinfer.jit import env as jit_env
from flashinfer.jit.gemm import gen_gemm_sm100_module

SEED_ROOT = Path("/opt/sglang-thor/flashinfer-cache-seed")
W8_SOURCES = {
    "gemm_groupwise_e4m3_bf16_majorfalse_mma1_sm100.cu",
    "gemm_groupwise_e4m3_bf16_majorfalse_mma2_sm100.cu",
    "gemm_groupwise_sm100.cu",
    "gemm_sm100_binding.cu",
}


def get_w4_spec():
    spec = gemm_base.gen_gemm_sm100_module()
    wanted = {
        "group_gemm_mxfp4_groupwise_e4m3_bf16_mma1_swaptrue_sm100.cu",
        "group_gemm_mxfp4_groupwise_e4m3_bf16_mma2_swaptrue_sm100.cu",
    }
    spec.sources = [source for source in spec.sources if source.name in wanted]
    spec.sources.extend(
        [
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_mxfp4_indexed_thor.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "group_gemm_mxfp4_indexed_thor_binding.cu",
        ]
    )
    spec.name = "gemm_sm100_thor_indexed_v2"
    return spec


def get_w8_spec():
    base = gen_gemm_sm100_module()
    sources = [source for source in base.sources if source.name in W8_SOURCES]
    found = {source.name for source in sources}
    if found != W8_SOURCES:
        raise RuntimeError(f"Missing W8 sources: {sorted(W8_SOURCES - found)}")
    return replace(base, name="gemm_sm100_w8_sm110_stage5", sources=sources)


def main() -> None:
    for spec in (get_w4_spec(), get_w8_spec()):
        spec.build(verbose=True)
        print(f"Built {spec.name}: {spec.jit_library_path}", flush=True)

    relative_workspace = jit_env.FLASHINFER_WORKSPACE_DIR.relative_to(
        jit_env.FLASHINFER_CACHE_DIR
    )
    destination = SEED_ROOT / relative_workspace
    shutil.copytree(
        jit_env.FLASHINFER_WORKSPACE_DIR,
        destination,
        copy_function=shutil.copy2,
        dirs_exist_ok=True,
    )
    print(f"Packaged FlashInfer cache seed: {destination}", flush=True)


if __name__ == "__main__":
    main()

"""Jetson Thor (SM110) MXFP4 expert backend.

This adapts FlashInfer's CuTeDSL B12x W4A16 fused MoE kernel, originally
tuned for the SM121 GB10 used by DGX Spark, to DeepSeek-V4's OCP MXFP4
checkpoint format on SM110.  The packed E2M1 weights stay four bits; only the
E8M0 scale metadata is repeated from one scale per 32 weights to the kernel's
one-scale-per-16 representation.

The FlashInfer W4A16 implementation uses BF16 ``mma.sync`` after software FP4
dequantization, so it does not depend on SM121's native FP4 tensor-core
instruction.  We deliberately call the internal launch API: FlashInfer's
public B12x wrapper currently gates the same portable kernel to SM120/SM121.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.utils import is_flashinfer_available, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)

_MXFP4_GROUP_SIZE = 32
_NVFP4_GROUP_SIZE = 16


def expand_mxfp4_e8m0_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert logical OCP-MXFP4 scales to unswizzled NVFP4 scale metadata.

    OCP MXFP4 stores one exact power-of-two E8M0 scale for 32 weights.  The
    Spark W4A16 kernel consumes E4M3 scales for groups of 16. Repeating each
    scale twice is lossless for finite values in E4M3's range and preserves
    the original dequantized weight exactly.
    """
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None or scales.dtype != e8m0_dtype:
        raise TypeError(
            "Thor MXFP4 requires native torch.float8_e8m0fnu block scales; "
            f"got {scales.dtype}."
        )
    if scales.ndim != 3:
        raise ValueError(f"MXFP4 block scales must be 3D, got {tuple(scales.shape)}")

    expanded = scales.float().repeat_interleave(
        _MXFP4_GROUP_SIZE // _NVFP4_GROUP_SIZE, dim=-1
    )
    if not bool(torch.isfinite(expanded).all().item()):
        raise ValueError("MXFP4 block scales contain non-finite values.")
    if bool((expanded.abs() > 448.0).any().item()):
        raise ValueError(
            "MXFP4 block scale exceeds E4M3 range required by FlashInfer W4A16."
        )
    return expanded.to(torch.float8_e4m3fn)


def _prepare_scale_storage(scales: torch.Tensor) -> torch.Tensor:
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_fp4_helpers import (
        swizzle_block_scale,
    )

    return swizzle_block_scale(expand_mxfp4_e8m0_scales(scales))


def map_global_topk_to_local(
    topk_ids: torch.Tensor,
    *,
    moe_ep_rank: int,
    num_global_experts: int,
    num_local_experts: int,
    num_fused_shared_experts: int,
) -> torch.Tensor:
    """Map global routed-expert IDs to one EP rank's packed weight indices.

    FlashInfer's B12x W4A16 route pack treats ``-1`` as a dropped route.  This
    lets the ordinary FusedMoE output all-reduce combine disjoint expert
    partitions without an additional token-dispatch backend.  Fused shared
    experts occupy the tail of both the global and each rank-local layout, so
    they are replicated and must not shift the routed-expert partition.
    """
    num_shared = int(num_fused_shared_experts)
    num_local_routed = int(num_local_experts) - num_shared
    num_global_routed = int(num_global_experts) - num_shared
    local_begin = int(moe_ep_rank) * num_local_routed
    local_end = local_begin + num_local_routed

    local_ids = torch.full_like(topk_ids, -1)
    routed = (topk_ids >= local_begin) & (topk_ids < local_end)
    local_ids = torch.where(routed, topk_ids - local_begin, local_ids)
    if num_shared:
        shared = (topk_ids >= num_global_routed) & (
            topk_ids < num_global_routed + num_shared
        )
        local_ids = torch.where(
            shared, num_local_routed + topk_ids - num_global_routed, local_ids
        )
    return local_ids.contiguous()


class Mxfp4FlashinferThorMoEMethod:
    """DeepSeek-V4 MXFP4 W4A16 fused MoE for Jetson Thor SM110."""

    def __init__(self, fp8_method, prefix: str):
        if not is_flashinfer_available():
            raise RuntimeError("The Thor MXFP4 backend requires FlashInfer.")
        self._fp8 = fp8_method
        self.prefix = prefix
        self._graph_max_tokens = max(
            0, int(os.environ.get("SGLANG_THOR_CUDA_GRAPH_MAX_BS", "0"))
        )
        self._native_enabled = os.environ.get(
            "SGLANG_THOR_NATIVE_MXFP4", "1"
        ).lower() not in {"0", "false", "no", "off"}
        self._decode_workspace = None
        self._native_decode_workspace = None
        self._native_decode_workspaces = []

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer's preparation helper receives [up; gate] and repacks the
        # fused projection into the kernel's [gate; up] activation layout.
        return True

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype,
        **extra_weight_attrs,
    ) -> None:
        if hidden_size % 128 != 0 or intermediate_size_per_partition % 128 != 0:
            raise ValueError(
                "Thor W4A16 requires hidden and intermediate sizes divisible by "
                f"128; got {hidden_size} and {intermediate_size_per_partition}."
            )
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            fp4_scale_dtype=torch.float8_e8m0fnu,
            **extra_weight_attrs,
        )

    def create_moe_runner(self, layer: Module, moe_runner_config) -> None:
        # This backend calls FlashInfer's fused B12x path directly. The unified
        # MoeRunner does not yet expose that W4A16 entry point.
        self.moe_runner_config = moe_runner_config

    def _register_ep_expert_map(
        self,
        layer: Module,
        *,
        num_experts: int,
        num_global_experts: int,
        moe_ep_size: int,
    ) -> None:
        if moe_ep_size <= 1:
            return
        num_shared = int(self.moe_runner_config.num_fused_shared_experts or 0)
        num_local_routed = num_experts - num_shared
        num_global_routed = num_global_experts - num_shared
        local_begin = int(layer.moe_ep_rank) * num_local_routed
        local_end = local_begin + num_local_routed
        expert_map = torch.full(
            (num_global_experts,),
            -1,
            dtype=torch.int32,
            device=layer.w13_weight.device,
        )
        expert_map[local_begin:local_end].copy_(
            torch.arange(
                num_local_routed,
                dtype=torch.int32,
                device=layer.w13_weight.device,
            )
        )
        if num_shared:
            expert_map[num_global_routed:num_global_experts].copy_(
                torch.arange(
                    num_local_routed,
                    num_experts,
                    dtype=torch.int32,
                    device=layer.w13_weight.device,
                )
            )
        layer.register_buffer("_thor_ep_expert_map", expert_map, persistent=False)

    def _process_native_weights(
        self,
        layer: Module,
        *,
        num_experts: int,
        num_global_experts: int,
        moe_ep_size: int,
    ) -> None:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_fp4_helpers import (
            swizzle_block_scale,
        )
        from sglang.srt.layers.quantization.mxfp4_flashinfer_thor_native import (
            allocate_native_mxfp4_workspace,
        )

        log_info_on_rank0(
            logger,
            "Preparing DeepSeek-V4 MXFP4 experts for the Thor SM110 native "
            f"MXFP8xMXFP4 tensor-core kernel (layer: {self.prefix})...",
        )
        hidden = int(layer.w13_weight.shape[-1] * 2)
        intermediate = int(layer.w2_weight.shape[-1] * 2)
        w13_scale = (
            swizzle_block_scale(layer.w13_weight_scale_inv.data)
            .contiguous()
            .view(torch.uint8)
            .flatten(1)
        )
        w2_scale = (
            swizzle_block_scale(layer.w2_weight_scale_inv.data)
            .contiguous()
            .view(torch.uint8)
            .flatten(1)
        )
        layer.w13_weight = Parameter(
            layer.w13_weight.data.view(torch.uint8).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight = Parameter(
            layer.w2_weight.data.view(torch.uint8).contiguous(),
            requires_grad=False,
        )
        layer.w13_weight_scale_inv = Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_scale, requires_grad=False)
        layer._thor_native_mxfp4_hidden = hidden
        layer._thor_native_mxfp4_intermediate = intermediate

        self._register_ep_expert_map(
            layer,
            num_experts=num_experts,
            num_global_experts=num_global_experts,
            moe_ep_size=moe_ep_size,
        )
        if self._graph_max_tokens > 0:
            # Batch-1 and batch-2 DSpark graphs have different static route
            # ceilings. CUTLASS receives the workspace's static group count,
            # so reusing the batch-2 allocation makes every batch-1 grouped
            # GEMM scan twice as many empty groups.
            small_graph_tokens = (self._graph_max_tokens + 1) // 2
            graph_tiers = sorted(
                {small_graph_tokens, self._graph_max_tokens}
            )
            self._native_decode_workspaces = [
                (
                    max_tokens,
                    allocate_native_mxfp4_workspace(
                        num_experts=num_experts,
                        top_k=int(self.moe_runner_config.top_k),
                        hidden=hidden,
                        intermediate=intermediate,
                        max_tokens=max_tokens,
                        device=layer.w13_weight.device,
                    ),
                )
                for max_tokens in graph_tiers
            ]
            self._native_decode_workspace = self._native_decode_workspaces[-1][
                1
            ]
        layer._dsv4_mxfp4_backend = "flashinfer_native_mxfp4_sm110"
        torch.cuda.empty_cache()

    def process_weights_after_loading(self, layer: Module) -> None:
        self._fp8.process_weights_after_loading(layer)
        if getattr(layer, "_mega_moe_weights_built", False):
            return

        num_experts = int(layer.num_local_experts)
        num_global_experts = int(getattr(layer, "num_experts", num_experts))
        moe_ep_size = int(getattr(layer, "moe_ep_size", 1))
        if self._native_enabled:
            self._process_native_weights(
                layer,
                num_experts=num_experts,
                num_global_experts=num_global_experts,
                moe_ep_size=moe_ep_size,
            )
            return

        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            allocate_sm120_moe_workspace,
        )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
            prepare_w4a16_packed_weights,
        )

        log_info_on_rank0(
            logger,
            "Preparing DeepSeek-V4 MXFP4 experts for the Thor SM110 "
            f"FlashInfer W4A16 kernel (layer: {self.prefix})...",
        )

        w13_scales = _prepare_scale_storage(layer.w13_weight_scale_inv.data)
        w2_scales = _prepare_scale_storage(layer.w2_weight_scale_inv.data)
        global_scale = torch.ones(
            num_experts, dtype=torch.float32, device=layer.w13_weight.device
        )

        prepared = prepare_w4a16_packed_weights(
            layer.w13_weight.data.view(torch.uint8),
            w13_scales,
            global_scale,
            layer.w2_weight.data.view(torch.uint8),
            w2_scales,
            global_scale,
            activation="silu",
            params_dtype=torch.bfloat16,
            source_format="modelopt",
        )

        # Replace the checkpoint layout immediately, one layer at a time, so
        # the full raw and packed model are never resident simultaneously.
        layer.w13_weight = Parameter(prepared.w13, requires_grad=False)
        layer.w2_weight = Parameter(prepared.w2, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(
            prepared.w13_scale, requires_grad=False
        )
        layer.w2_weight_scale_inv = Parameter(prepared.w2_scale, requires_grad=False)
        layer.register_buffer(
            "_thor_w4a16_kernel_workspace", prepared.workspace, persistent=False
        )
        layer._thor_w4a16_prepared = replace(
            prepared,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            w13_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            workspace=layer._thor_w4a16_kernel_workspace,
        )

        self._register_ep_expert_map(
            layer,
            num_experts=num_experts,
            num_global_experts=num_global_experts,
            moe_ep_size=moe_ep_size,
        )

        if self._graph_max_tokens > 0:
            top_k = int(self.moe_runner_config.top_k)
            self._decode_workspace = allocate_sm120_moe_workspace(
                state_E=num_experts,
                weight_E=(num_global_experts if moe_ep_size > 1 else num_experts),
                routed_rows=self._graph_max_tokens * top_k,
                k=int(prepared.hidden_size),
                n=int(prepared.intermediate_size),
                num_topk=top_k,
                device=layer.w13_weight.device,
                quant_mode="w4a16",
                activation="silu",
            )

        layer._dsv4_mxfp4_backend = "flashinfer_w4a16_sm110"
        del w13_scales, w2_scales, global_scale, prepared
        torch.cuda.empty_cache()

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(
                "Thor MXFP4 requires standard top-k output, got "
                f"{topk_output.format}."
            )

        x = dispatch_output.hidden_states.contiguous()
        topk_ids = topk_output.topk_ids.contiguous()
        moe_ep_size = int(getattr(layer, "moe_ep_size", 1))
        topk_weights = topk_output.topk_weights.contiguous()
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.float()
        # With EP, a rank may own none of a token's selected experts.  Start at
        # zero so the post-MoE all-reduce sums only locally-owned routes.
        output = torch.zeros_like(x) if moe_ep_size > 1 else torch.empty_like(x)
        num_tokens = int(x.shape[0])

        if (
            getattr(layer, "_dsv4_mxfp4_backend", "")
            == "flashinfer_native_mxfp4_sm110"
        ):
            from sglang.srt.layers.quantization.mxfp4_flashinfer_thor_native import (
                get_cached_native_mxfp4_workspace,
                native_mxfp4_moe,
            )

            workspace = next(
                (
                    candidate
                    for max_tokens, candidate in self._native_decode_workspaces
                    if num_tokens <= max_tokens
                ),
                None,
            )
            if workspace is None:
                workspace = get_cached_native_mxfp4_workspace(
                    num_experts=int(layer.num_local_experts),
                    top_k=int(topk_ids.shape[1]),
                    hidden=int(layer._thor_native_mxfp4_hidden),
                    intermediate=int(layer._thor_native_mxfp4_intermediate),
                    min_tokens=num_tokens,
                    device=x.device,
                )
            native_mxfp4_moe(
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                w13=layer.w13_weight,
                w13_scale=layer.w13_weight_scale_inv,
                w2=layer.w2_weight,
                w2_scale=layer.w2_weight_scale_inv,
                output=output,
                workspace=workspace,
                expert_map=(
                    layer._thor_ep_expert_map if moe_ep_size > 1 else None
                ),
            )
            return StandardCombineInput(hidden_states=output)

        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _get_cached_workspace,
            launch_sm120_moe,
        )
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (
            run_w4a16_moe,
        )

        workspace = (
            self._decode_workspace
            if self._decode_workspace is not None
            and num_tokens <= self._graph_max_tokens
            else None
        )

        if moe_ep_size > 1:
            if workspace is None:
                workspace = _get_cached_workspace(
                    backend="w4a16",
                    state_E=int(layer.num_local_experts),
                    weight_E=int(layer.num_experts),
                    routed_rows=num_tokens * int(topk_ids.shape[1]),
                    k=int(x.shape[1]),
                    n=int(layer._thor_w4a16_prepared.intermediate_size),
                    num_topk=int(topk_ids.shape[1]),
                    device=x.device,
                    quant_mode="w4a16",
                    activation="silu",
                )
            run_w4a16_moe(
                x,
                layer._thor_w4a16_prepared,
                topk_weights,
                topk_ids,
                activation="silu",
                intermediate_cache13=workspace.intermediate_cache13,
                intermediate_cache2=workspace.intermediate_cache2,
                output=output,
                fc1_c_tmp=workspace.fc1_c_tmp,
                fc2_c_tmp=workspace.fc2_c_tmp,
                packed_route_indices=workspace.packed_route_indices,
                block_expert_ids=workspace.block_expert_ids,
                packed_route_count=workspace.packed_route_count,
                expert_offsets=workspace.expert_offsets,
                expert_map=layer._thor_ep_expert_map,
            )
        else:
            launch_sm120_moe(
                a=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                w1_weight=layer.w13_weight,
                w1_weight_sf=layer.w13_weight_scale_inv,
                w1_alpha=layer._thor_w4a16_prepared.w13_global_scale,
                w2_weight=layer.w2_weight,
                w2_weight_sf=layer.w2_weight_scale_inv,
                w2_alpha=layer._thor_w4a16_prepared.w2_global_scale,
                num_experts=int(layer.num_local_experts),
                top_k=int(topk_ids.shape[1]),
                num_local_experts=int(layer.num_local_experts),
                scatter_output=output,
                activation="silu",
                quant_mode="w4a16",
                source_format="modelopt",
                _workspace=workspace,
                _prepared_weights=layer._thor_w4a16_prepared,
            )
        return StandardCombineInput(hidden_states=output)

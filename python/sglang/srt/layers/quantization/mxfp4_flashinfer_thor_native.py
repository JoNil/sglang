"""SM110-native MXFP8 x MXFP4 MoE built on FlashInfer's CUTLASS group GEMM."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


_ROUTE_BLOCK = int(os.environ.get("SGLANG_THOR_NATIVE_MXFP4_ROUTE_BLOCK", "1"))
if _ROUTE_BLOCK not in (1, 2, 4):
    raise ValueError("SGLANG_THOR_NATIVE_MXFP4_ROUTE_BLOCK must be 1, 2, or 4")
_COUNT_BLOCK_T = 256
_SORT_BLOCK_T = 256
_QUANT_BLOCK = 32
_QUANT_GROUPS_PER_PROGRAM = int(
    os.environ.get("SGLANG_THOR_MXFP8_GROUPS_PER_PROGRAM", "64")
)
if _QUANT_GROUPS_PER_PROGRAM not in (1, 2, 4, 8, 16, 32, 64):
    raise ValueError(
        "SGLANG_THOR_MXFP8_GROUPS_PER_PROGRAM must be 1, 2, 4, 8, 16, 32, or 64"
    )
_QUANT_VECTOR_MIN_TOKENS = int(
    os.environ.get("SGLANG_THOR_MXFP8_VECTOR_MIN_TOKENS", "8")
)
if _QUANT_VECTOR_MIN_TOKENS < 1:
    raise ValueError("SGLANG_THOR_MXFP8_VECTOR_MIN_TOKENS must be positive")
_FUSED_DECODE_PACK = int(
    os.environ.get("SGLANG_THOR_MXFP4_FUSED_DECODE_PACK", "0")
)
if _FUSED_DECODE_PACK not in (0, 1):
    raise ValueError("SGLANG_THOR_MXFP4_FUSED_DECODE_PACK must be 0 or 1")
_EAGER_WORKSPACES: dict[tuple, "NativeMxfp4Workspace"] = {}


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def max_packed_rows(num_routes: int, num_experts: int) -> int:
    maximum = int(num_routes) + int(num_experts) * (_ROUTE_BLOCK - 1)
    if int(num_routes) < int(num_experts):
        maximum = min(int(num_routes) * _ROUTE_BLOCK, maximum)
    return max(_ROUTE_BLOCK, maximum)


def max_active_groups(num_routes: int, num_experts: int) -> int:
    return max(1, min(int(num_routes), int(num_experts)))


def quant_groups_per_program(num_tokens: int) -> int:
    """Keep latency-optimal scalar groups for batch-one decode."""
    if int(num_tokens) < _QUANT_VECTOR_MIN_TOKENS:
        return 1
    return _QUANT_GROUPS_PER_PROGRAM


@triton.jit
def _zero_expert_counts_kernel(
    expert_counts,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    tl.store(expert_counts + experts, 0, mask=experts < NUM_EXPERTS)


@triton.jit
def _route_count_kernel(
    topk_ids,
    expert_map,
    expert_counts,
    route_to_packed,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EXPERT_MAP_SIZE: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    raw_ids = tl.load(
        topk_ids + offsets, mask=offsets < NUM_ROUTES, other=-1
    ).to(tl.int32)
    valid = (offsets < NUM_ROUTES) & (raw_ids >= 0)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        valid = valid & (raw_ids < EXPERT_MAP_SIZE)
        safe_ids = tl.minimum(
            tl.maximum(raw_ids, 0), EXPERT_MAP_SIZE - 1
        )
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(
            tl.int32
        )
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)
    else:
        valid = valid & (raw_ids < NUM_EXPERTS)
    safe_local_ids = tl.minimum(tl.maximum(ids, 0), NUM_EXPERTS - 1)
    tl.atomic_add(
        expert_counts + safe_local_ids, 1, sem="relaxed", mask=valid
    )
    tl.store(route_to_packed + offsets, -1, mask=offsets < NUM_ROUTES)


@triton.jit
def _route_count_small_kernel(
    topk_ids,
    expert_map,
    expert_counts,
    route_to_packed,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EXPERT_MAP_SIZE: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # Decode has at most 84 routes and fits in one program. Clear the expert
    # histogram here so graph replay avoids a separate launch.
    offsets = tl.arange(0, BLOCK_T)
    tl.store(expert_counts + offsets, 0, mask=offsets < NUM_EXPERTS)
    tl.debug_barrier()
    raw_ids = tl.load(
        topk_ids + offsets, mask=offsets < NUM_ROUTES, other=-1
    ).to(tl.int32)
    valid = (offsets < NUM_ROUTES) & (raw_ids >= 0)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        valid = valid & (raw_ids < EXPERT_MAP_SIZE)
        safe_ids = tl.minimum(
            tl.maximum(raw_ids, 0), EXPERT_MAP_SIZE - 1
        )
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(
            tl.int32
        )
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)
    else:
        valid = valid & (raw_ids < NUM_EXPERTS)
    safe_local_ids = tl.minimum(tl.maximum(ids, 0), NUM_EXPERTS - 1)
    tl.atomic_add(
        expert_counts + safe_local_ids, 1, sem="relaxed", mask=valid
    )
    tl.store(route_to_packed + offsets, -1, mask=offsets < NUM_ROUTES)


@triton.jit
def _route_pack_decode_kernel(
    topk_ids,
    expert_map,
    expert_counts,
    packed_route_indices,
    packed_route_count,
    m_indptr,
    active_expert_ids,
    active_group_count,
    expert_to_compact,
    write_offsets,
    route_to_packed,
    block_expert_ids,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EXPERT_MAP_SIZE: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """Pack a decode route table in one CTA instead of four launches."""
    routes = tl.arange(0, BLOCK_T)
    experts = tl.arange(0, BLOCK_E)
    route_mask = routes < NUM_ROUTES
    expert_mask = experts < NUM_EXPERTS

    tl.store(expert_counts + experts, 0, mask=expert_mask)
    tl.store(route_to_packed + routes, -1, mask=route_mask)
    tl.debug_barrier()

    raw_ids = tl.load(topk_ids + routes, mask=route_mask, other=-1).to(tl.int32)
    valid = route_mask & (raw_ids >= 0)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        valid = valid & (raw_ids < EXPERT_MAP_SIZE)
        safe_ids = tl.minimum(tl.maximum(raw_ids, 0), EXPERT_MAP_SIZE - 1)
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)
    else:
        valid = valid & (raw_ids < NUM_EXPERTS)
    safe_local_ids = tl.minimum(tl.maximum(ids, 0), NUM_EXPERTS - 1)
    tl.atomic_add(expert_counts + safe_local_ids, 1, sem="relaxed", mask=valid)
    tl.debug_barrier()

    counts = tl.load(expert_counts + experts, mask=expert_mask, other=0)
    padded = ((counts + ROUTE_BLOCK - 1) // ROUTE_BLOCK) * ROUTE_BLOCK
    padded = tl.where(expert_mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)
    active = expert_mask & (counts > 0)
    compact_group = tl.cumsum(active.to(tl.int32), axis=0) - 1
    num_active_groups = tl.sum(active.to(tl.int32), axis=0)

    tl.store(write_offsets + experts, prefix, mask=expert_mask)
    tl.store(
        expert_to_compact + experts,
        tl.where(active, compact_group, -1),
        mask=expert_mask,
    )
    tl.store(m_indptr + compact_group, prefix, mask=active)
    tl.store(active_expert_ids + compact_group, experts, mask=active)
    tl.store(m_indptr + num_active_groups, total)
    tl.store(active_group_count, num_active_groups)
    tl.store(packed_route_count, total)

    groups = tl.arange(0, BLOCK_G)
    tl.store(
        active_expert_ids + groups,
        0,
        mask=(groups >= num_active_groups) & (groups < MAX_GROUPS),
    )
    tl.store(
        m_indptr + groups,
        total,
        mask=(groups > num_active_groups) & (groups <= MAX_GROUPS),
    )
    for slot in tl.static_range(0, ROUTE_BLOCK - 1):
        padding_index = prefix + counts + slot
        tl.store(
            packed_route_indices + padding_index,
            NUM_ROUTES,
            mask=expert_mask & (slot < (padded - counts)),
        )
    tl.debug_barrier()

    packed = tl.atomic_add(
        write_offsets + safe_local_ids, 1, sem="relaxed", mask=valid
    )
    tl.store(packed_route_indices + packed, routes, mask=valid)
    tl.store(route_to_packed + routes, packed, mask=valid)
    route_compact_group = tl.load(
        expert_to_compact + safe_local_ids, mask=valid, other=0
    )
    # Expert row ranges are padded to ROUTE_BLOCK, so each packed block has
    # at least one real route. Races only store the same compact group value.
    tl.store(
        block_expert_ids + packed // ROUTE_BLOCK,
        route_compact_group,
        mask=valid,
    )


@triton.jit
def _route_prefix_kernel(
    expert_counts,
    packed_route_indices,
    packed_route_count,
    m_indptr,
    active_expert_ids,
    active_group_count,
    expert_to_compact,
    write_offsets,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < NUM_EXPERTS
    counts = tl.load(expert_counts + experts, mask=expert_mask, other=0)

    padded = ((counts + ROUTE_BLOCK - 1) // ROUTE_BLOCK) * ROUTE_BLOCK
    padded = tl.where(expert_mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)
    active = expert_mask & (counts > 0)
    compact_group = tl.cumsum(active.to(tl.int32), axis=0) - 1
    num_active_groups = tl.sum(active.to(tl.int32), axis=0)

    tl.store(write_offsets + experts, prefix, mask=expert_mask)
    tl.store(
        expert_to_compact + experts,
        tl.where(active, compact_group, -1),
        mask=expert_mask,
    )

    tl.store(m_indptr + compact_group, prefix, mask=active)
    tl.store(active_expert_ids + compact_group, experts, mask=active)
    tl.store(m_indptr + num_active_groups, total)
    tl.store(active_group_count, num_active_groups)
    tl.store(packed_route_count, total)

    # CUTLASS receives a graph-static group count. Represent the unused suffix
    # as zero-row groups without paying for another launch.
    groups = tl.arange(0, BLOCK_G)
    tl.store(
        active_expert_ids + groups,
        0,
        mask=(groups >= num_active_groups) & (groups < MAX_GROUPS),
    )
    tl.store(
        m_indptr + groups,
        total,
        mask=(groups > num_active_groups) & (groups <= MAX_GROUPS),
    )

    # Only the padding rows per expert need a sentinel. Real
    # routes are populated by the subsequent sort kernel.
    for slot in tl.static_range(0, ROUTE_BLOCK - 1):
        padding_index = prefix + counts + slot
        tl.store(
            packed_route_indices + padding_index,
            NUM_ROUTES,
            mask=expert_mask & (slot < (padded - counts)),
        )


@triton.jit
def _block_expert_kernel(
    m_indptr,
    active_group_count,
    block_expert_ids,
    ROUTE_BLOCK: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    compact_group = tl.program_id(0)
    active = compact_group < tl.load(active_group_count)
    begin = (
        tl.load(m_indptr + compact_group, mask=active, other=0)
        // ROUTE_BLOCK
    )
    end = (
        tl.load(m_indptr + compact_group + 1, mask=active, other=0)
        // ROUTE_BLOCK
    )
    offsets = tl.arange(0, BLOCK_B)
    for start in tl.range(begin, end, BLOCK_B):
        blocks = start + offsets
        tl.store(
            block_expert_ids + blocks,
            compact_group,
            mask=blocks < end,
        )


@triton.jit
def _route_sort_kernel(
    topk_ids,
    expert_map,
    packed_route_indices,
    write_offsets,
    route_to_packed,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EXPERT_MAP_SIZE: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    raw_ids = tl.load(
        topk_ids + offsets, mask=offsets < NUM_ROUTES, other=-1
    ).to(tl.int32)
    valid = (offsets < NUM_ROUTES) & (raw_ids >= 0)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        valid = valid & (raw_ids < EXPERT_MAP_SIZE)
        safe_ids = tl.minimum(
            tl.maximum(raw_ids, 0), EXPERT_MAP_SIZE - 1
        )
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)
    else:
        valid = valid & (raw_ids < NUM_EXPERTS)

    safe_local_ids = tl.minimum(tl.maximum(ids, 0), NUM_EXPERTS - 1)
    packed = tl.atomic_add(
        write_offsets + safe_local_ids, 1, sem="relaxed", mask=valid
    )
    tl.store(packed_route_indices + packed, offsets, mask=valid)
    tl.store(route_to_packed + offsets, packed, mask=valid)


@triton.jit
def _store_mxfp8_scale(
    scale_output,
    exponent,
    packed_row,
    expert,
    expert_row,
    scale_col,
    expert_start,
    active,
    SCALE_COLS: tl.constexpr,
):
    # CUTLASS's 128x4 block-scale layout:
    # (row_tile, col_tile, row_in_32, row_quadrant, col_in_4).
    sf_row_base = ((expert_start + expert * 127) // 128) * 128
    row_tile = expert_row // 128
    row_in_tile = expert_row % 128
    row_in_32 = row_in_tile % 32
    row_quadrant = row_in_tile // 32
    col_tile = scale_col // 4
    col_in_4 = scale_col % 4
    cols_tiles = SCALE_COLS // 4
    swizzled = (
        ((((row_tile * cols_tiles + col_tile) * 32 + row_in_32) * 4)
         + row_quadrant)
        * 4
        + col_in_4
    )
    output_index = sf_row_base * SCALE_COLS + swizzled
    raw_scale = tl.maximum(0.0, tl.minimum(255.0, exponent + 127.0)).to(
        tl.uint8
    )
    tl.store(scale_output + output_index, raw_scale, mask=active)


@triton.jit
def _route_quantize_kernel(
    source,
    packed_route_indices,
    block_expert_ids,
    packed_route_count,
    m_indptr,
    output,
    scale_output,
    NUM_ROUTES: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_PACKED_ROWS: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
):
    packed_row = tl.program_id(0)
    scale_cols = (
        tl.program_id(1) * GROUPS_PER_PROGRAM
        + tl.arange(0, GROUPS_PER_PROGRAM)
    )
    columns = scale_cols[:, None] * 32 + tl.arange(0, 32)[None, :]
    active_cols = scale_cols < HIDDEN // 32
    active_row = packed_row < tl.load(packed_route_count)
    route = tl.load(
        packed_route_indices + packed_row,
        mask=packed_row < MAX_PACKED_ROWS,
        other=NUM_ROUTES,
    )
    active = active_row & (route < NUM_ROUTES)
    token = route // TOP_K
    values = tl.load(
        source + token * HIDDEN + columns,
        mask=active & active_cols[:, None],
        other=0.0,
    ).to(tl.float32)
    maximum = tl.max(tl.abs(values), axis=1)
    safe_maximum = tl.maximum(maximum, 2.0**-126)
    exponent = tl.ceil(tl.log2(safe_maximum / 448.0))
    exponent = tl.maximum(-127.0, tl.minimum(127.0, exponent))
    scale = tl.exp2(exponent)
    quantized = values / scale[:, None]
    tl.store(
        output + packed_row * HIDDEN + columns,
        quantized,
        mask=active_row & active_cols[:, None],
    )

    expert = tl.load(
        block_expert_ids + packed_row // ROUTE_BLOCK,
        mask=active_row,
        other=0,
    )
    expert_start = tl.load(m_indptr + expert, mask=active_row, other=0)
    expert_row = packed_row - expert_start
    _store_mxfp8_scale(
        scale_output,
        exponent,
        packed_row,
        expert,
        expert_row,
        scale_cols,
        expert_start,
        active_row & active_cols,
        SCALE_COLS,
    )


@triton.jit
def _swiglu_quantize_kernel(
    fc1,
    block_expert_ids,
    packed_route_count,
    m_indptr,
    output,
    scale_output,
    INTERMEDIATE: tl.constexpr,
    FC1_COLS: tl.constexpr,
    MAX_PACKED_ROWS: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
):
    packed_row = tl.program_id(0)
    scale_cols = (
        tl.program_id(1) * GROUPS_PER_PROGRAM
        + tl.arange(0, GROUPS_PER_PROGRAM)
    )
    columns = scale_cols[:, None] * 32 + tl.arange(0, 32)[None, :]
    active_cols = scale_cols < INTERMEDIATE // 32
    active_row = packed_row < tl.load(packed_route_count)

    # The checkpoint loader stores the fused projection as [up; gate].
    up = tl.load(
        fc1 + packed_row * FC1_COLS + columns,
        mask=active_row & active_cols[:, None],
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        fc1 + packed_row * FC1_COLS + INTERMEDIATE + columns,
        mask=active_row & active_cols[:, None],
        other=0.0,
    ).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)) * up
    maximum = tl.max(tl.abs(activated), axis=1)
    safe_maximum = tl.maximum(maximum, 2.0**-126)
    exponent = tl.ceil(tl.log2(safe_maximum / 448.0))
    exponent = tl.maximum(-127.0, tl.minimum(127.0, exponent))
    scale = tl.exp2(exponent)
    quantized = activated / scale[:, None]
    tl.store(
        output + packed_row * INTERMEDIATE + columns,
        quantized,
        mask=active_row & active_cols[:, None],
    )

    expert = tl.load(
        block_expert_ids + packed_row // ROUTE_BLOCK,
        mask=active_row,
        other=0,
    )
    expert_start = tl.load(m_indptr + expert, mask=active_row, other=0)
    expert_row = packed_row - expert_start
    _store_mxfp8_scale(
        scale_output,
        exponent,
        packed_row,
        expert,
        expert_row,
        scale_cols,
        expert_start,
        active_row & active_cols,
        SCALE_COLS,
    )


@triton.jit
def _finalize_kernel(
    fc2,
    topk_weights,
    route_to_packed,
    output,
    NUM_TOKENS: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    column_mask = columns < HIDDEN
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for slot in tl.static_range(0, TOP_K):
        route = token * TOP_K + slot
        packed = tl.load(route_to_packed + route)
        valid = packed >= 0
        weight = tl.load(topk_weights + route).to(tl.float32)
        value = tl.load(
            fc2 + packed * HIDDEN + columns,
            mask=valid & column_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += value * weight
    tl.store(
        output + token * HIDDEN + columns,
        accumulator,
        mask=column_mask,
    )


@dataclass
class NativeMxfp4Workspace:
    num_experts: int
    top_k: int
    hidden: int
    intermediate: int
    max_tokens: int
    max_rows: int
    max_groups: int
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    expert_counts: torch.Tensor
    packed_route_count: torch.Tensor
    m_indptr: torch.Tensor
    active_expert_ids: torch.Tensor
    active_group_count: torch.Tensor
    expert_to_compact: torch.Tensor
    write_offsets: torch.Tensor
    route_to_packed: torch.Tensor
    routed_mxfp8: torch.Tensor
    routed_scales: torch.Tensor
    fc1: torch.Tensor
    activated_mxfp8: torch.Tensor
    activated_scales: torch.Tensor
    fc2: torch.Tensor


def allocate_native_mxfp4_workspace(
    *,
    num_experts: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    max_tokens: int,
    device: torch.device,
) -> NativeMxfp4Workspace:
    max_routes = max(1, int(max_tokens) * int(top_k))
    max_rows = max_packed_rows(max_routes, num_experts)
    max_groups = max_active_groups(max_routes, num_experts)
    max_blocks = _align_up(max_rows, _ROUTE_BLOCK) // _ROUTE_BLOCK

    input_sf_cols = _align_up(hidden // _QUANT_BLOCK, 4)
    act_sf_cols = _align_up(intermediate // _QUANT_BLOCK, 4)
    sf_rows = _align_up(max_rows + max_groups * 127, 128)
    return NativeMxfp4Workspace(
        num_experts=int(num_experts),
        top_k=int(top_k),
        hidden=int(hidden),
        intermediate=int(intermediate),
        max_tokens=int(max_tokens),
        max_rows=int(max_rows),
        max_groups=int(max_groups),
        packed_route_indices=torch.empty(
            max_rows, dtype=torch.int32, device=device
        ),
        block_expert_ids=torch.empty(
            max_blocks, dtype=torch.int32, device=device
        ),
        expert_counts=torch.empty(
            num_experts, dtype=torch.int32, device=device
        ),
        packed_route_count=torch.empty(1, dtype=torch.int32, device=device),
        m_indptr=torch.empty(max_groups + 1, dtype=torch.int32, device=device),
        active_expert_ids=torch.empty(
            max_groups, dtype=torch.int32, device=device
        ),
        active_group_count=torch.empty(
            1, dtype=torch.int32, device=device
        ),
        expert_to_compact=torch.empty(
            num_experts, dtype=torch.int32, device=device
        ),
        write_offsets=torch.empty(
            num_experts, dtype=torch.int32, device=device
        ),
        route_to_packed=torch.empty(
            max_routes, dtype=torch.int32, device=device
        ),
        routed_mxfp8=torch.empty(
            (max_rows, hidden), dtype=torch.float8_e4m3fn, device=device
        ),
        routed_scales=torch.empty(
            sf_rows * input_sf_cols, dtype=torch.uint8, device=device
        ),
        fc1=torch.empty(
            (max_rows, 2 * intermediate),
            dtype=torch.bfloat16,
            device=device,
        ),
        activated_mxfp8=torch.empty(
            (max_rows, intermediate),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        activated_scales=torch.empty(
            sf_rows * act_sf_cols, dtype=torch.uint8, device=device
        ),
        fc2=torch.empty(
            (max_rows, hidden), dtype=torch.bfloat16, device=device
        ),
    )


def get_cached_native_mxfp4_workspace(
    *,
    num_experts: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    min_tokens: int,
    device: torch.device,
) -> NativeMxfp4Workspace:
    """Return one grow-only eager workspace shared by all compatible layers."""
    device = torch.device(device)
    device_index = (
        torch.cuda.current_device()
        if device.index is None
        else int(device.index)
    )
    key = (
        device.type,
        device_index,
        int(num_experts),
        int(top_k),
        int(hidden),
        int(intermediate),
    )
    workspace = _EAGER_WORKSPACES.get(key)
    if workspace is None or workspace.max_tokens < int(min_tokens):
        # Avoid retaining a separate large prefill allocation for every layer.
        # Rounding reduces reallocations as chunk sizes vary.
        allocation_tokens = _align_up(max(1, int(min_tokens)), 256)
        workspace = allocate_native_mxfp4_workspace(
            num_experts=num_experts,
            top_k=top_k,
            hidden=hidden,
            intermediate=intermediate,
            max_tokens=allocation_tokens,
            device=device,
        )
        _EAGER_WORKSPACES[key] = workspace
    return workspace


def _pack_routes(
    topk_ids: torch.Tensor,
    workspace: NativeMxfp4Workspace,
    expert_map: torch.Tensor | None,
) -> None:
    num_routes = int(topk_ids.numel())
    block_e = _next_power_of_2(workspace.num_experts)
    block_g = _next_power_of_2(workspace.max_groups + 1)
    expert_map_tensor = expert_map if expert_map is not None else topk_ids
    expert_map_size = int(expert_map.numel()) if expert_map is not None else 0
    if _FUSED_DECODE_PACK and num_routes <= _COUNT_BLOCK_T:
        _route_pack_decode_kernel[(1,)](
            topk_ids,
            expert_map_tensor,
            workspace.expert_counts,
            workspace.packed_route_indices,
            workspace.packed_route_count,
            workspace.m_indptr,
            workspace.active_expert_ids,
            workspace.active_group_count,
            workspace.expert_to_compact,
            workspace.write_offsets,
            workspace.route_to_packed,
            workspace.block_expert_ids,
            NUM_ROUTES=num_routes,
            NUM_EXPERTS=workspace.num_experts,
            EXPERT_MAP_SIZE=expert_map_size,
            HAS_EXPERT_MAP=expert_map is not None,
            MAX_GROUPS=workspace.max_groups,
            ROUTE_BLOCK=_ROUTE_BLOCK,
            BLOCK_T=_COUNT_BLOCK_T,
            BLOCK_E=block_e,
            BLOCK_G=block_g,
            num_warps=8,
        )
        return
    if num_routes <= _COUNT_BLOCK_T:
        _route_count_small_kernel[(1,)](
            topk_ids,
            expert_map_tensor,
            workspace.expert_counts,
            workspace.route_to_packed,
            NUM_ROUTES=num_routes,
            NUM_EXPERTS=workspace.num_experts,
            EXPERT_MAP_SIZE=expert_map_size,
            HAS_EXPERT_MAP=expert_map is not None,
            BLOCK_T=_COUNT_BLOCK_T,
            num_warps=8,
        )
    else:
        _zero_expert_counts_kernel[(1,)](
            workspace.expert_counts,
            NUM_EXPERTS=workspace.num_experts,
            BLOCK_E=block_e,
            num_warps=4,
        )
        _route_count_kernel[(triton.cdiv(num_routes, _COUNT_BLOCK_T),)](
            topk_ids,
            expert_map_tensor,
            workspace.expert_counts,
            workspace.route_to_packed,
            NUM_ROUTES=num_routes,
            NUM_EXPERTS=workspace.num_experts,
            EXPERT_MAP_SIZE=expert_map_size,
            HAS_EXPERT_MAP=expert_map is not None,
            BLOCK_T=_COUNT_BLOCK_T,
            num_warps=4,
        )
    _route_prefix_kernel[(1,)](
        workspace.expert_counts,
        workspace.packed_route_indices,
        workspace.packed_route_count,
        workspace.m_indptr,
        workspace.active_expert_ids,
        workspace.active_group_count,
        workspace.expert_to_compact,
        workspace.write_offsets,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=workspace.num_experts,
        MAX_GROUPS=workspace.max_groups,
        ROUTE_BLOCK=_ROUTE_BLOCK,
        BLOCK_E=block_e,
        BLOCK_G=block_g,
        num_warps=8,
    )
    _block_expert_kernel[(workspace.max_groups,)](
        workspace.m_indptr,
        workspace.active_group_count,
        workspace.block_expert_ids,
        ROUTE_BLOCK=_ROUTE_BLOCK,
        BLOCK_B=64,
        num_warps=1,
    )
    _route_sort_kernel[(triton.cdiv(num_routes, _SORT_BLOCK_T),)](
        topk_ids,
        expert_map_tensor,
        workspace.packed_route_indices,
        workspace.write_offsets,
        workspace.route_to_packed,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=workspace.num_experts,
        EXPERT_MAP_SIZE=expert_map_size,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_T=_SORT_BLOCK_T,
        num_warps=4,
    )


def _indexed_group_gemm(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    m_indptr: torch.Tensor,
    expert_ids: torch.Tensor,
    mma_sm: int,
    tile_n: int,
    tile_k: int,
) -> None:
    from flashinfer.gemm import gemm_base

    int_workspace = gemm_base._get_cache_buf(
        "group_gemm_mxfp4_nt_groupwise_int_workspace",
        gemm_base.DEFAULT_WORKSPACE_SIZE,
        a.device,
    )
    float_workspace = gemm_base._get_cache_buf(
        "group_gemm_mxfp4_nt_groupwise_float_workspace",
        gemm_base.DEFAULT_WORKSPACE_SIZE,
        a.device,
    )
    _get_indexed_gemm_sm100_module().group_gemm_mxfp4_nt_groupwise_indexed(
        int_workspace,
        float_workspace,
        a,
        b,
        a_scale,
        b_scale,
        out,
        m_indptr,
        expert_ids,
        int(b.shape[1]),
        int(b.shape[2]) * 2,
        int(mma_sm),
        int(tile_n),
        int(tile_k),
    )


@functools.cache
def _get_indexed_gemm_sm100_module():
    """Build only the two Thor MXFP4/BF16 template families we can call."""
    from flashinfer.gemm import gemm_base
    from flashinfer.jit import env as jit_env

    spec = gemm_base.gen_gemm_sm100_module()
    wanted = {
        "group_gemm_mxfp4_groupwise_e4m3_bf16_mma1_swaptrue_sm100.cu",
        "group_gemm_mxfp4_groupwise_e4m3_bf16_mma2_swaptrue_sm100.cu",
    }
    spec.sources = [source for source in spec.sources if source.name in wanted]
    spec.sources.extend(
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "group_gemm_mxfp4_indexed_thor.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "group_gemm_mxfp4_indexed_thor_binding.cu",
        ]
    )
    spec.name = "gemm_sm100_thor_indexed_v2"
    return spec.build_and_load()


def native_mxfp4_moe(
    *,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    output: torch.Tensor,
    workspace: NativeMxfp4Workspace,
    expert_map: torch.Tensor | None = None,
    mma_sm: int = 2,
    tile_n: int | None = None,
    tile_k: int | None = None,
) -> torch.Tensor:
    from flashinfer.gemm import gemm_base

    num_tokens, hidden = x.shape
    if num_tokens > workspace.max_tokens:
        raise ValueError(
            f"native MXFP4 workspace supports {workspace.max_tokens} tokens, "
            f"got {num_tokens}"
        )
    if hidden != workspace.hidden:
        raise ValueError(f"hidden mismatch: {hidden} != {workspace.hidden}")
    if topk_ids.shape != (num_tokens, workspace.top_k):
        raise ValueError("topk_ids shape does not match workspace")
    if tile_n is None or tile_k is None:
        # The speculative verifier presents 7 or 14 target tokens. Prefill
        # benefits from K=256 once there are enough route rows, and from a
        # wider N tile at 512+ tokens. These thresholds are measured on SM110
        # with the production E256/H4096/I2048/top-k-6 shape.
        if 1 < num_tokens <= 14:
            tile_n, tile_k = 128, 256
        elif 14 < num_tokens < 512:
            tile_n, tile_k = 64, 256
        elif num_tokens >= 512:
            tile_n, tile_k = 192, 256
        else:
            tile_n, tile_k = 64, 128

    # The public checker advertises SM110, while its inner family predicate
    # currently omits it. The generated CUTLASS module itself supports SM110.
    gemm_base.is_sm100a_supported = lambda _device: True

    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    _pack_routes(topk_ids, workspace, expert_map)
    active_max_rows = max_packed_rows(
        int(topk_ids.numel()), workspace.num_experts
    )

    input_sf_cols = _align_up(hidden // _QUANT_BLOCK, 4)
    quant_groups = quant_groups_per_program(num_tokens)
    quant_warps = 4 if quant_groups >= 4 else 1
    _route_quantize_kernel[
        (
            active_max_rows,
            triton.cdiv(
                hidden // _QUANT_BLOCK, quant_groups
            ),
        )
    ](
        x,
        workspace.packed_route_indices,
        workspace.block_expert_ids,
        workspace.packed_route_count,
        workspace.m_indptr,
        workspace.routed_mxfp8,
        workspace.routed_scales,
        NUM_ROUTES=int(topk_ids.numel()),
        TOP_K=workspace.top_k,
        HIDDEN=hidden,
        MAX_PACKED_ROWS=active_max_rows,
        ROUTE_BLOCK=_ROUTE_BLOCK,
        SCALE_COLS=input_sf_cols,
        GROUPS_PER_PROGRAM=quant_groups,
        num_warps=quant_warps,
    )

    _indexed_group_gemm(
        a=workspace.routed_mxfp8,
        b=w13,
        a_scale=workspace.routed_scales,
        b_scale=w13_scale,
        out=workspace.fc1,
        m_indptr=workspace.m_indptr,
        expert_ids=workspace.active_expert_ids,
        mma_sm=mma_sm,
        tile_n=tile_n,
        tile_k=tile_k,
    )

    act_sf_cols = _align_up(workspace.intermediate // _QUANT_BLOCK, 4)
    _swiglu_quantize_kernel[
        (
            active_max_rows,
            triton.cdiv(
                workspace.intermediate // _QUANT_BLOCK,
                quant_groups,
            ),
        )
    ](
        workspace.fc1,
        workspace.block_expert_ids,
        workspace.packed_route_count,
        workspace.m_indptr,
        workspace.activated_mxfp8,
        workspace.activated_scales,
        INTERMEDIATE=workspace.intermediate,
        FC1_COLS=2 * workspace.intermediate,
        MAX_PACKED_ROWS=active_max_rows,
        ROUTE_BLOCK=_ROUTE_BLOCK,
        SCALE_COLS=act_sf_cols,
        GROUPS_PER_PROGRAM=quant_groups,
        num_warps=quant_warps,
    )

    _indexed_group_gemm(
        a=workspace.activated_mxfp8,
        b=w2,
        a_scale=workspace.activated_scales,
        b_scale=w2_scale,
        out=workspace.fc2,
        m_indptr=workspace.m_indptr,
        expert_ids=workspace.active_expert_ids,
        mma_sm=mma_sm,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    _finalize_kernel[(num_tokens, triton.cdiv(hidden, 256))](
        workspace.fc2,
        topk_weights,
        workspace.route_to_packed,
        output,
        NUM_TOKENS=num_tokens,
        TOP_K=workspace.top_k,
        HIDDEN=hidden,
        BLOCK_H=256,
        num_warps=8,
    )
    return output


__all__ = [
    "NativeMxfp4Workspace",
    "allocate_native_mxfp4_workspace",
    "get_cached_native_mxfp4_workspace",
    "max_active_groups",
    "max_packed_rows",
    "native_mxfp4_moe",
]

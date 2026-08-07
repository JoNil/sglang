"""Jetson Thor SM110 tests for the FlashInfer B12x W4A16 adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-small")


def test_map_global_topk_to_local_for_expert_parallel_rank():
    from sglang.srt.layers.quantization.mxfp4_flashinfer_thor_moe import (
        map_global_topk_to_local,
    )

    global_ids = torch.tensor([[0, 127, 128, 255], [64, 192, 256, -1]])
    local_ids = map_global_topk_to_local(
        global_ids,
        moe_ep_rank=1,
        num_global_experts=256,
        num_local_experts=128,
        num_fused_shared_experts=0,
    )

    assert torch.equal(
        local_ids,
        torch.tensor([[-1, -1, 0, 127], [-1, 64, -1, -1]]),
    )


def test_map_global_topk_to_local_preserves_replicated_shared_expert():
    from sglang.srt.layers.quantization.mxfp4_flashinfer_thor_moe import (
        map_global_topk_to_local,
    )

    # 256 routed + one shared globally; each EP2 rank stores 128 routed + the
    # same shared expert in its final local slot.
    global_ids = torch.tensor([[0, 127, 128, 255, 256, -1]])
    rank0 = map_global_topk_to_local(
        global_ids,
        moe_ep_rank=0,
        num_global_experts=257,
        num_local_experts=129,
        num_fused_shared_experts=1,
    )
    rank1 = map_global_topk_to_local(
        global_ids,
        moe_ep_rank=1,
        num_global_experts=257,
        num_local_experts=129,
        num_fused_shared_experts=1,
    )

    assert torch.equal(rank0, torch.tensor([[0, 127, -1, -1, 128, -1]]))
    assert torch.equal(rank1, torch.tensor([[-1, -1, 0, 127, 128, -1]]))


def test_expand_mxfp4_e8m0_scales_is_exact():
    from sglang.srt.layers.quantization.mxfp4_flashinfer_thor_moe import (
        expand_mxfp4_e8m0_scales,
    )

    source = torch.tensor([125, 126, 127, 128, 129], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    source = source.reshape(1, 1, -1)
    expanded = expand_mxfp4_e8m0_scales(source)
    expected = source.float().repeat_interleave(2, dim=-1)

    assert expanded.dtype == torch.float8_e4m3fn
    assert expanded.shape[-1] == 2 * source.shape[-1]
    assert torch.equal(expanded.float(), expected)


def _decode_e2m1(weight: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=weight.device,
    )
    raw = weight.view(torch.uint8)
    unpacked = torch.empty(
        (*raw.shape[:-1], raw.shape[-1] * 2),
        dtype=torch.uint8,
        device=raw.device,
    )
    unpacked[..., 0::2] = raw & 0xF
    unpacked[..., 1::2] = raw >> 4
    return lut[unpacked.long()]


def _dequant_mxfp4(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = _decode_e2m1(weight)
    scale = scales.float().repeat_interleave(32, dim=-1)
    return values * scale


@pytest.mark.parametrize("tokens", [1, 2])
def test_sm110_w4a16_matches_mxfp4_reference(monkeypatch, tokens: int):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Jetson Thor SM110 required")
    pytest.importorskip(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
    )

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.mxfp4_flashinfer_thor_moe import (
        Mxfp4FlashinferThorMoEMethod,
    )

    monkeypatch.setenv("SGLANG_THOR_CUDA_GRAPH_MAX_BS", "2")
    generator = torch.Generator(device="cuda").manual_seed(7)
    experts, hidden, intermediate, top_k = 4, 128, 128, 2
    w13 = torch.randint(
        -128,
        128,
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        -128,
        128,
        (experts, hidden, intermediate // 2),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    scale13 = torch.full(
        (experts, 2 * intermediate, hidden // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)
    scale2 = torch.full(
        (experts, hidden, intermediate // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
            self.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)
            self.w13_weight_scale_inv = torch.nn.Parameter(
                scale13.clone(), requires_grad=False
            )
            self.w2_weight_scale_inv = torch.nn.Parameter(
                scale2.clone(), requires_grad=False
            )
            self.num_local_experts = experts

    layer = Layer().cuda()
    method = Mxfp4FlashinferThorMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    method.create_moe_runner(
        layer,
        MoeRunnerConfig(
            num_experts=experts,
            num_local_experts=experts,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            top_k=top_k,
            activation="silu",
            is_gated=True,
        ),
    )
    method.process_weights_after_loading(layer)

    x = (
        torch.randn(
            tokens,
            hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.01
    )
    topk_ids = torch.arange(tokens * top_k, device="cuda", dtype=torch.int32)
    topk_ids = topk_ids.remainder(experts).reshape(tokens, top_k)
    topk_weights = torch.full(
        (tokens, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda"
    )
    logits = torch.zeros(tokens, experts, dtype=torch.float32, device="cuda")
    dispatch = StandardDispatchOutput(
        x,
        None,
        StandardTopKOutput(topk_weights, topk_ids, logits),
    )
    actual = method.apply(layer, dispatch).hidden_states.float()

    # The loader contract stores [up; gate]; FlashInfer's preparation helper
    # repacks it to the kernel's [gate; up] layout.
    w13_f = _dequant_mxfp4(w13, scale13)
    w2_f = _dequant_mxfp4(w2, scale2)
    expected = torch.zeros_like(actual)
    for token in range(tokens):
        for route in range(top_k):
            expert = int(topk_ids[token, route])
            up = torch.mv(w13_f[expert, :intermediate], x[token].float())
            gate = torch.mv(w13_f[expert, intermediate:], x[token].float())
            activated = torch.nn.functional.silu(gate).to(torch.bfloat16)
            activated = (activated * up.to(torch.bfloat16)).to(torch.bfloat16)
            routed = torch.mv(w2_f[expert], activated.float())
            expected[token] += topk_weights[token, route] * routed

    torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.5)

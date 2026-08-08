from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import CapCorrectLen
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepCandidates,
)
from sglang.srt.runtime_context import get_parallel


def _all_gather_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    return get_parallel().attn_tp_group.all_gather(tensor.contiguous(), dim=-1)


def sample_distributed_step_tokens(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    """Sample a sharded vocabulary with one two-float gather per row."""
    local_score, local_token = SampleStepCandidates.execute(
        step_logits=step_logits,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )
    local_token = local_token + int(vocab_start)
    # All DSV4 token ids are exactly representable in float32.  Combining the
    # key and id lets NCCL carry one tiny tensor and therefore one latency edge.
    candidate = torch.stack((local_score, local_token.to(torch.float32)), dim=-1)
    group = get_parallel().attn_tp_group
    if group.world_size == 1:
        return local_token
    gathered = group.all_gather(candidate.contiguous(), dim=-1).view(
        candidate.shape[0], group.world_size, 2
    )
    scores = gathered[..., 0]
    tokens = gathered[..., 1].to(torch.int64)
    best_score = scores.max(dim=1, keepdim=True).values
    sentinel = torch.full_like(tokens, torch.iinfo(torch.int64).max)
    return torch.where(scores == best_score, tokens, sentinel).min(dim=1).values


def gather_full_vocab(
    local_logits: torch.Tensor, *, vocab_size: int
) -> torch.Tensor:
    full = _all_gather_last_dim(local_logits)
    return full[..., : int(vocab_size)]


def _distributed_target_argmax(
    *, target_local: torch.Tensor, vocab_start: int, bs: int, rows: int
) -> torch.Tensor:
    scores, tokens = target_local.float().max(dim=-1)
    tokens = tokens.to(torch.int64) + int(vocab_start)
    packet = torch.stack((scores, tokens.to(torch.float32)), dim=-1).view(bs, -1)
    group = get_parallel().attn_tp_group
    if group.world_size == 1:
        return tokens.view(bs, rows)
    gathered = group.all_gather(packet.contiguous(), dim=-1).view(
        bs, group.world_size, rows, 2
    )
    all_scores = gathered[..., 0]
    all_tokens = gathered[..., 1].to(torch.int64)
    best_score = all_scores.max(dim=1, keepdim=True).values
    sentinel = torch.full_like(all_tokens, torch.iinfo(torch.int64).max)
    return torch.where(all_scores == best_score, all_tokens, sentinel).min(dim=1).values


def accept_greedy_distributed(
    *,
    candidates: torch.Tensor,
    target_local: torch.Tensor,
    vocab_start: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    rows = int(verify_num_draft_tokens)
    target_predict = _distributed_target_argmax(
        target_local=target_local, vocab_start=vocab_start, bs=bs, rows=rows
    )
    matches = target_predict[:, :-1] == candidates[:, 1:]
    correct_len = (
        torch.cumprod(matches.to(torch.int32), dim=1).sum(dim=1).to(torch.int32)
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
    row_ids = torch.arange(bs, device=candidates.device)
    bonus = target_predict[row_ids, correct_len.to(torch.long)].to(torch.int64)
    return correct_len, bonus, cap_trim_lens


def _global_log_norm(
    scaled_target: torch.Tensor, scaled_draft: torch.Tensor, *, bs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    target_rows = scaled_target.shape[1]
    draft_rows = scaled_draft.shape[1]
    local = torch.cat(
        (
            torch.logsumexp(scaled_target, dim=-1),
            torch.logsumexp(scaled_draft, dim=-1),
        ),
        dim=1,
    )
    group = get_parallel().attn_tp_group
    if group.world_size == 1:
        global_norm = local
    else:
        gathered = group.all_gather(local.contiguous(), dim=-1).view(
            bs, group.world_size, target_rows + draft_rows
        )
        global_norm = torch.logsumexp(gathered, dim=1)
    return global_norm[:, :target_rows], global_norm[:, target_rows:]


def _candidate_local_probs(
    *,
    probs: torch.Tensor,
    candidate_tokens: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    vocab_end = int(vocab_start) + probs.shape[-1]
    owned = (candidate_tokens >= int(vocab_start)) & (candidate_tokens < vocab_end)
    local_idx = (candidate_tokens - int(vocab_start)).clamp(
        min=0, max=probs.shape[-1] - 1
    )
    value = probs.gather(dim=-1, index=local_idx.unsqueeze(-1)).squeeze(-1)
    return torch.where(owned, value, 0.0)


def accept_sampling_distributed(
    *,
    candidates: torch.Tensor,
    target_local: torch.Tensor,
    draft_local: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    vocab_start: int,
    vocab_size: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact untruncated speculative sampling over TP vocabulary shards.

    The full p/q arrays never cross the fabric.  Two batched metadata gathers
    normalize and verify all rows, and one final gather publishes the residual
    sample.  Greedy rows share the same packets and are selected at the end.
    """
    bs = candidates.shape[0]
    rows = int(verify_num_draft_tokens)
    gamma = rows - 1
    local_vocab = target_local.shape[-1]
    target = target_local.view(bs, rows, local_vocab).float()
    draft = draft_local.view(bs, gamma, local_vocab).float()
    temp = temperatures.float().view(bs, 1, 1).clamp_min(1e-5)
    scaled_target = target / temp
    scaled_draft = draft / temp
    target_norm, draft_norm = _global_log_norm(
        scaled_target, scaled_draft, bs=bs
    )
    target_probs = torch.exp(scaled_target - target_norm.unsqueeze(-1))
    draft_probs = torch.exp(scaled_draft - draft_norm.unsqueeze(-1))

    proposal_tokens = candidates[:, 1 : gamma + 1]
    candidate_p = _candidate_local_probs(
        probs=target_probs[:, :gamma],
        candidate_tokens=proposal_tokens,
        vocab_start=vocab_start,
    )
    candidate_q = _candidate_local_probs(
        probs=draft_probs,
        candidate_tokens=proposal_tokens,
        vocab_start=vocab_start,
    )
    residual = torch.cat(
        (
            torch.clamp_min(target_probs[:, :gamma] - draft_probs, 0.0),
            target_probs[:, gamma : gamma + 1],
        ),
        dim=1,
    )
    local_mass = residual.sum(dim=-1)

    target_scores, target_tokens = scaled_target.max(dim=-1)
    target_tokens = target_tokens.to(torch.int64) + int(vocab_start)
    group = get_parallel().attn_tp_group
    rank = int(group.rank_in_group)
    uniform = torch.rand((bs, gamma + 1), dtype=torch.float32, device=target.device)
    if rank != 0:
        uniform.zero_()
    packet = torch.cat(
        (
            target_scores,
            target_tokens.to(torch.float32),
            candidate_p,
            candidate_q,
            local_mass,
            uniform,
        ),
        dim=1,
    )
    packet_width = packet.shape[1]
    if group.world_size == 1:
        gathered = packet.view(bs, 1, packet_width)
    else:
        gathered = group.all_gather(packet.contiguous(), dim=-1).view(
            bs, group.world_size, packet_width
        )

    cursor = 0
    all_target_scores = gathered[:, :, cursor : cursor + rows]
    cursor += rows
    all_target_tokens = gathered[:, :, cursor : cursor + rows].to(torch.int64)
    cursor += rows
    all_candidate_p = gathered[:, :, cursor : cursor + gamma]
    cursor += gamma
    all_candidate_q = gathered[:, :, cursor : cursor + gamma]
    cursor += gamma
    masses = gathered[:, :, cursor : cursor + rows]
    cursor += rows
    coins = gathered[:, 0, cursor : cursor + gamma + 1]

    candidate_p_global = all_candidate_p.sum(dim=1)
    candidate_q_global = all_candidate_q.sum(dim=1)
    accepts = coins[:, :gamma] * candidate_q_global < candidate_p_global
    sampling_len = (
        torch.cumprod(accepts.to(torch.int32), dim=1).sum(dim=1).to(torch.int32)
    )

    best_score = all_target_scores.max(dim=1, keepdim=True).values
    sentinel = torch.full_like(all_target_tokens, torch.iinfo(torch.int64).max)
    target_predict = torch.where(
        all_target_scores == best_score, all_target_tokens, sentinel
    ).min(dim=1).values
    greedy_matches = target_predict[:, :-1] == proposal_tokens
    greedy_len = (
        torch.cumprod(greedy_matches.to(torch.int32), dim=1)
        .sum(dim=1)
        .to(torch.int32)
    )
    raw_len = torch.where(greedy_mask.view(-1), greedy_len, sampling_len)

    row_ids = torch.arange(bs, device=target.device)
    selected_residual = residual[row_ids, sampling_len.to(torch.long)]
    selected_masses = masses[
        row_ids[:, None],
        torch.arange(group.world_size, device=target.device)[None, :],
        sampling_len.to(torch.long)[:, None],
    ]
    total_mass = selected_masses.sum(dim=1)
    target_u = coins[:, gamma] * total_mass
    prefix = selected_masses.cumsum(dim=1) - selected_masses
    owned = (target_u >= prefix[:, rank]) & (
        target_u < prefix[:, rank] + selected_masses[:, rank]
    )
    local_u = target_u - prefix[:, rank]
    local_cdf = selected_residual.cumsum(dim=-1)
    local_idx = torch.searchsorted(
        local_cdf.contiguous(), local_u.unsqueeze(-1), right=True
    ).squeeze(-1)
    local_idx = local_idx.clamp(max=local_vocab - 1)
    local_bonus = torch.where(
        owned,
        local_idx.to(torch.int64) + int(vocab_start) + 1,
        torch.zeros_like(local_idx, dtype=torch.int64),
    )
    degenerate = total_mass <= 0
    local_bonus = torch.where(
        degenerate & (rank == group.world_size - 1),
        torch.full_like(local_bonus, int(vocab_size)),
        local_bonus,
    )
    if group.world_size == 1:
        sampling_bonus = local_bonus - 1
    else:
        sampling_bonus = (
            group.all_gather(local_bonus.view(bs, 1).contiguous(), dim=-1).sum(dim=-1)
            - 1
        )

    cap_trim_lens = torch.zeros_like(raw_len)
    correct_len = raw_len
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=raw_len, verify_lens=cutoff_verify_lens
        )
    greedy_bonus = target_predict[row_ids, correct_len.to(torch.long)]
    capped_proposal = proposal_tokens[
        row_ids, correct_len.clamp(max=gamma - 1).to(torch.long)
    ]
    sampling_bonus = torch.where(
        cap_trim_lens > 0, capped_proposal, sampling_bonus
    )
    bonus = torch.where(greedy_mask.view(-1), greedy_bonus, sampling_bonus)
    return correct_len, bonus.to(torch.int64), cap_trim_lens

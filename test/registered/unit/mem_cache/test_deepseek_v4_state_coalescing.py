from __future__ import annotations

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


class _MemorySaverAdapter:
    def region(self, _tag):
        return nullcontext()


class TestDeepSeekV4StateCoalescing(unittest.TestCase):
    def test_layer_views_share_one_backing_and_keep_sentinels(self):
        owner = object.__new__(DeepSeekV4TokenToKVPool)
        owner.memory_saver_adapter = _MemorySaverAdapter()
        owner.custom_mem_pool = None
        owner._compress_state_backings = {}
        allocator = owner._make_layer_state_buffer_allocator("c4_attention", 3)

        with (
            patch(
                "sglang.srt.mem_cache.deepseek_v4_compress_state."
                "TorchMemorySaverAdapter.create",
                return_value=_MemorySaverAdapter(),
            ),
            patch(
                "sglang.srt.mem_cache.deepseek_v4_compress_state."
                "maybe_init_custom_mem_pool",
                return_value=(False, None, None),
            ),
        ):
            pools = [
                CompressStatePool(
                    size=8,
                    ring_size=4,
                    overlap=True,
                    head_dim=3,
                    dtype=torch.float32,
                    device="cpu",
                    enable_memory_saver=False,
                    ratio=4,
                    buffer_allocator=allocator,
                )
                for _ in range(3)
            ]

        backing = owner._compress_state_backings["c4_attention"]
        self.assertEqual(tuple(backing.shape), (3, 16, 12))
        storage_ptr = backing.untyped_storage().data_ptr()
        for layer, pool in enumerate(pools):
            view = pool.kv_score_buffer.kv_score
            self.assertTrue(view.is_contiguous())
            self.assertEqual(view.untyped_storage().data_ptr(), storage_ptr)
            self.assertEqual(view.data_ptr(), backing[layer].data_ptr())
            self.assertTrue(torch.equal(pool.kv_score_buffer[-1].kv, torch.zeros(6)))
            self.assertTrue(torch.isneginf(pool.kv_score_buffer[-1].score).all())

        pools[0].kv_score_buffer.kv_score[0].fill_(17)
        self.assertFalse(torch.equal(backing[0, 0], backing[1, 0]))

    def test_full_pool_initialization_uses_three_backings(self):
        owner = object.__new__(DeepSeekV4TokenToKVPool)
        owner.memory_saver_adapter = _MemorySaverAdapter()
        owner.custom_mem_pool = None
        owner.compression_ratios = [4, 128, 0, 4]
        owner._stage_start = 0
        owner._stage_end = 4
        owner.c4_state_pool_size = 8
        owner.c128_state_pool_size = 8
        owner.qk_nope_head_dim = 2
        owner.qk_rope_head_dim = 2
        owner.indexer_head_dim = 3
        owner.c4_state_dtype = torch.float32
        owner.c128_state_dtype = torch.float64
        owner.device = "cpu"
        owner.swa_page_size = 1
        owner.online_mtp_max_draft_tokens = 0
        owner.get_ring_size = lambda ratio: 4 if ratio == 4 else 8

        with (
            patch(
                "sglang.srt.mem_cache.deepseek_v4_compress_state."
                "TorchMemorySaverAdapter.create",
                return_value=_MemorySaverAdapter(),
            ),
            patch(
                "sglang.srt.mem_cache.deepseek_v4_compress_state."
                "maybe_init_custom_mem_pool",
                return_value=(False, None, None),
            ),
            patch(
                "sglang.srt.mem_cache.deepseek_v4_memory_pool."
                "_COALESCE_COMPRESS_STATES",
                True,
            ),
        ):
            owner._init_paged_compress_states(enable_memory_saver=False)

        self.assertEqual(
            set(owner._compress_state_backings),
            {"c4_attention", "c4_indexer", "c128_attention"},
        )
        self.assertEqual(owner._compress_state_backings["c4_attention"].shape[0], 2)
        self.assertEqual(owner._compress_state_backings["c4_indexer"].shape[0], 2)
        self.assertEqual(owner._compress_state_backings["c128_attention"].shape[0], 1)

        c4_attn = [
            owner.compress_state_pools[index].kv_score_buffer.kv_score
            for index in (0, 3)
        ]
        c4_indexer = [
            owner.indexer_compress_state_pools[index].kv_score_buffer.kv_score
            for index in (0, 3)
        ]
        self.assertEqual(len({v.untyped_storage().data_ptr() for v in c4_attn}), 1)
        self.assertEqual(
            len({v.untyped_storage().data_ptr() for v in c4_indexer}), 1
        )
        self.assertIsNone(owner.compress_state_pools[2])
        self.assertIsNone(owner.indexer_compress_state_pools[1])

if __name__ == "__main__":
    unittest.main()

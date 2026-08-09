import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin


class TestPersistentTileLangIndexerWorkspace(unittest.TestCase):
    def test_reuses_contiguous_flat_backing(self):
        backend = C4IndexerBackendMixin()

        first = backend._get_persistent_tilelang_logits_output(
            query_rows=4,
            current_seq_len=1024,
            capacity_rows=512,
            capacity_seq_len=2048,
            device=torch.device("cpu"),
        )
        backing = backend._persistent_tilelang_logits_workspace
        self.assertEqual(first.shape, (4, 1024))
        self.assertTrue(first.is_contiguous())
        self.assertEqual(backing.numel(), 512 * 2048)
        self.assertEqual(first.data_ptr(), backing.data_ptr())

        second = backend._get_persistent_tilelang_logits_output(
            query_rows=8,
            current_seq_len=1536,
            capacity_rows=512,
            capacity_seq_len=2048,
            device=torch.device("cpu"),
        )
        self.assertIs(backing, backend._persistent_tilelang_logits_workspace)
        self.assertTrue(second.is_contiguous())
        self.assertEqual(second.data_ptr(), backing.data_ptr())

    def test_grows_when_capacity_increases(self):
        backend = C4IndexerBackendMixin()
        backend._get_persistent_tilelang_logits_output(
            query_rows=2,
            current_seq_len=128,
            capacity_rows=4,
            capacity_seq_len=256,
            device=torch.device("cpu"),
        )
        grown = backend._get_persistent_tilelang_logits_output(
            query_rows=8,
            current_seq_len=512,
            capacity_rows=8,
            capacity_seq_len=512,
            device=torch.device("cpu"),
        )
        self.assertEqual(grown.shape, (8, 512))
        self.assertEqual(
            backend._persistent_tilelang_logits_workspace.numel(), 8 * 512
        )

    def test_optional_pretouch_zeros_existing_backing(self):
        backend = C4IndexerBackendMixin()
        with envs.SGLANG_OPT_DSV4_PRETOUCH_TILELANG_LOGITS.override(False):
            backend._get_persistent_tilelang_logits_output(
                query_rows=2,
                current_seq_len=128,
                capacity_rows=4,
                capacity_seq_len=256,
                device=torch.device("cpu"),
            )
        backing = backend._persistent_tilelang_logits_workspace
        backing.fill_(7)
        self.assertFalse(backend._persistent_tilelang_logits_workspace_pretouched)

        with envs.SGLANG_OPT_DSV4_PRETOUCH_TILELANG_LOGITS.override(True):
            backend._get_persistent_tilelang_logits_output(
                query_rows=2,
                current_seq_len=128,
                capacity_rows=4,
                capacity_seq_len=256,
                device=torch.device("cpu"),
            )

        self.assertTrue(backend._persistent_tilelang_logits_workspace_pretouched)
        self.assertEqual(torch.count_nonzero(backing).item(), 0)


if __name__ == "__main__":
    unittest.main()

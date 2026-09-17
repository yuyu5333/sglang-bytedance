import unittest

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.mem_cache.dsv41_staging_workspace import (
    MainKVStagingWorkspace,
    staging_geometry,
    staging_workspace_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="stage-a")


class TestMainKVStagingWorkspace(unittest.TestCase):
    def test_geometry_and_physical_budget(self):
        for topk, rows in ((1, 1), (64, 2), (65, 3), (1024, 64)):
            with self.subTest(topk=topk, rows=rows):
                width, pages = staging_geometry(topk, rows)
                self.assertEqual(width % 64, 0)
                self.assertGreaterEqual(width, topk)
                self.assertGreaterEqual(pages * 256, rows * width)
                self.assertLess((pages - 1) * 256, rows * width)
                ws = MainKVStagingWorkspace(torch.device("cpu"), topk, rows)
                self.assertEqual(ws.nbytes, staging_workspace_bytes(topk, rows))
                self.assertEqual(ws.cache.stride(0), KVLayout.V4.page_bytes(256))
                self.assertEqual(ws.cache.data_ptr(), ws.pages.data_ptr())
                ws.check_shape(rows, width)
                with self.assertRaisesRegex(ValueError, "exceeds reserved"):
                    ws.check_shape(rows + 1, width)
                with self.assertRaisesRegex(ValueError, "exceeds reserved"):
                    ws.check_shape(rows, width + 1)
        self.assertEqual(staging_workspace_bytes(1024), 38600704)

    def test_invalid_geometry(self):
        for topk, rows in ((0, 1), (-1, 1), (1, 0), (1, -1)):
            with self.subTest(topk=topk, rows=rows):
                with self.assertRaisesRegex(ValueError, "positive"):
                    staging_geometry(topk, rows)


if __name__ == "__main__":
    unittest.main()

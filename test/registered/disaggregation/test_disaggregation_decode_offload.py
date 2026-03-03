import os
import shutil
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

# Registering the test for CUDA CI with appropriate parameters
# Increasing estimated time since we run evaluation twice
register_cuda_ci(est_time=600, suite="stage-b-test-large-2-gpu")


class TestDisaggregationDecodeOffload(PDDisaggregationServerBase):
    """
    Test class for verifying KV cache offloading on the decode side in a 
    prefill-decode disaggregation setup.
    """

    @classmethod
    def setUpClass(cls):
        # Set environment variable to make offloading more frequent for testing purposes
        cls.old_stride = os.environ.get("SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE")
        cls.hicache_dir = "/tmp/hicache_test"
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.hicache_dir
        os.environ["SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE"] = "16"

        # Ensure a clean cache directory
        if os.path.exists(cls.hicache_dir):
            shutil.rmtree(cls.hicache_dir)
        os.makedirs(cls.hicache_dir, exist_ok=True)

        super().setUpClass()
        # Using a small model for faster test execution and reduced memory footprint
        # cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        # for test local model
        # cls.model = "/data00/Llama-3.2-1B-Instruct"
        cls.model = "/data00/Llama-3.1-8B-Instruct"
        

        # Non-blocking start of prefill and decode servers
        cls.start_prefill()
        cls.start_decode()

        # Wait for both servers to be ready before proceeding
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        # Restore the original environment variable state
        super().tearDownClass()
        if cls.old_stride is not None:
            os.environ["SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE"] = cls.old_stride
        else:
            os.environ.pop("SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE", None)
        
        os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", None)
        
        # Clean up the cache directory
        if os.path.exists(cls.hicache_dir):
            shutil.rmtree(cls.hicache_dir)

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "1",
            "--page-size",
            "16",
            "--enable-hierarchical-cache",
            "--hicache-storage-backend",
            "file",
            "--hicache-ratio",
            "2",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--disaggregation-decode-enable-offload-kvcache",
            "--num-reserved-decode-tokens",
            "128",
            "--hicache-ratio",
            "2",
            "--page-size",
            "16",
            "--hicache-storage-backend",
            "file",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_mmlu_double_eval(self):
        """
        Run two rounds of MMLU evaluation:
        1. First round: Decode node offloads KV cache back to disk (HiCache).
        2. Restart All Nodes to clear memory cache.
        3. Second round: Prefill node loads KV cache from disk (HiCache).
        Verify that both rounds produce consistent scores and compare performance.
        """
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            return_latency=True,
        )

        print("--- Starting First Round (Expected to Offload KV Cache) ---")
        metrics1, latency1 = run_eval(args)
        print(f"First round metrics: {metrics1}, Latency: {latency1:.3f} s")

        # Ensure all offloads are committed to disk
        import time
        print("Waiting for KV cache to be committed to disk...")
        time.sleep(10) 

        print("--- Restarting All Nodes (Prefill, Decode, and Router) ---")
        kill_process_tree(self.process_prefill.pid)
        kill_process_tree(self.process_decode.pid)
        kill_process_tree(self.process_lb.pid)
        self.process_prefill.wait()
        self.process_decode.wait()
        self.process_lb.wait()

        self.start_prefill()
        self.start_decode()
        self.launch_lb()
        self.wait_server_ready(self.prefill_url + "/health")
        self.wait_server_ready(self.decode_url + "/health")
        print("All nodes restarted and ready.")

        print("--- Starting Second Round (Expected to Load KV Cache from Storage) ---")
        metrics2, latency2 = run_eval(args)
        print(f"Second round metrics: {metrics2}, Latency: {latency2:.3f} s")

        # Assert score is above a minimum threshold for both rounds
        self.assertGreater(metrics1["score"], 0.60)
        self.assertGreater(metrics2["score"], 0.60)

        # Score should be consistent
        self.assertAlmostEqual(metrics1["score"], metrics2["score"], delta=0.01)
        
        # Calculate improvements
        latency_reduction = (latency1 - latency2) / latency1 * 100
        
        print(f"--- Comparison Results ---")
        print(f"Score: Round 1 = {metrics1['score']:.3f}, Round 2 = {metrics2['score']:.3f}")
        print(f"Latency: Round 1 = {latency1:.3f} s, Round 2 = {latency2:.3f} s")
        print(f"Latency Reduction: {latency_reduction:.2f}%")


if __name__ == "__main__":
    unittest.main()

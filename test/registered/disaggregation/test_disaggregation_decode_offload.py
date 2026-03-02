import os
import shutil
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
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

    def test_gsm8k_double_eval(self):
        """
        Run idx rounds of GSM8K evaluation:
        1. First round: Decode node offloads KV cache back to disk (HiCache).
        2. Restart Prefill node to clear its memory cache.
        3. Second round: Prefill node loads KV cache from disk (HiCache).
        Verify that both rounds produce consistent accuracy.
        """
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=20,
            max_new_tokens=512,
            parallel=16,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )

        print("--- Starting First Round (Expected to Offload KV Cache) ---")
        metrics1 = run_eval_few_shot_gsm8k(args)
        print(f"First round metrics: {metrics1}")

        # Ensure all offloads are committed to disk
        import time
        print("Waiting for KV cache to be committed to disk...")
        time.sleep(10) 

        print("--- Restarting Prefill Node (To clear memory cache) ---")
        kill_process_tree(self.process_prefill.pid)
        self.process_prefill.wait()
        self.start_prefill()
        self.wait_server_ready(self.prefill_url + "/health")
        print("Prefill node restarted and ready.")

        print("--- Starting Second Round (Expected to Load KV Cache from Storage) ---")
        metrics2 = run_eval_few_shot_gsm8k(args)
        print(f"Second round metrics: {metrics2}")

        # Assert accuracy is above a minimum threshold for both rounds
        self.assertGreater(metrics1["accuracy"], 0.30)
        self.assertGreater(metrics2["accuracy"], 0.30)

        # Accuracy should be consistent (ideally identical at temperature 0)
        self.assertAlmostEqual(metrics1["accuracy"], metrics2["accuracy"], delta=0.01)
        
        # Calculate improvements
        latency_reduction = (metrics1["latency"] - metrics2["latency"]) / metrics1["latency"] * 100
        throughput_improvement = (metrics2["output_throughput"] - metrics1["output_throughput"]) / metrics1["output_throughput"] * 100
        
        print(f"--- Comparison Results ---")
        print(f"Accuracy: Round 1 = {metrics1['accuracy']:.3f}, Round 2 = {metrics2['accuracy']:.3f}")
        print(f"Latency: Round 1 = {metrics1['latency']:.3f} s, Round 2 = {metrics2['latency']:.3f} s")
        print(f"Latency Reduction: {latency_reduction:.2f}%")
        print(f"Output Throughput: Round 1 = {metrics1['output_throughput']:.3f} token/s, Round 2 = {metrics2['output_throughput']:.3f} token/s")
        print(f"Throughput Improvement: {throughput_improvement:.2f}%")


if __name__ == "__main__":
    unittest.main()

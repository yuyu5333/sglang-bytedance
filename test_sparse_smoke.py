import sys
sys.path.insert(0, "/workspace/FlashMLA/tests")
import torch
import lib
from lib import RawTestParamForDecode as RawTestParam
from lib import generate_testcase_for_decode
import flash_mla.cuda as fmc

tp = RawTestParam(b=1, h_q=64, s_q=1, h_kv=1, s_kv=128, is_varlen=True, topk=64,
    have_topk_length=False, enable_attn_sink=True,
    block_size=64, d_qk=512, d_v=512, check_correctness=True, num_runs=0, seed=42)
tparam = tp.to_test_param()
tcase = generate_testcase_for_decode(tparam)
scope = tcase.kv_scope

out, lse, _, _ = fmc.sparse_decode_fwd(
    tcase.q, scope.get_kvcache_for_flash_mla(), scope.indices_in_kvcache,
    scope.topk_length, tcase.attn_sink,
    None, None, None, None, None,
    tparam.d_v, tcase.sm_scale)

print("out shape:", out.shape, "lse shape:", lse.shape)
print("out max:", float(out.abs().max().item()))
print("lse max:", float(lse.abs().max().item()))
print("Stage-2 all-None path PASS (no crash)")

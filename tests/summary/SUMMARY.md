# Test Summary

- Start: Sat Jan 24 13:18:43 PST 2026
- End: Sat Jan 24 13:33:03 PST 2026
- Total duration: 860s
- Git revision: `66a978d18a53184acc87a2fea9e63c0b4e65b983`
- Latest run log: `run_all_tests_summary_20260124_133303.txt`
- Environment log: `run_all_tests_env_20260124_133303.txt`

## Results

| Test | Status | Duration |
| --- | --- | --- |
| CPU Tests (all non-gpu tests) | ✅ Passed | 37s |
| GPU tests (tests/megatron/test_engine_megatron_prepp.py pre-PP readiness) | ✅ Passed | 193s |
| GPU tests (tests/megatron/test_single_trainer.py matrix) | ✅ Passed | 141s |
| GPU tests (tests/deepspeed/test_engine_deepspeed_prepp.py) | ❌ Failed | 24s |
| GPU tests (tests/deepspeed/test_text_autotp.py) | ✅ Passed | 27s |
| GPU tests (tests/deepspeed/test_text_autotp_dp.py) | ✅ Passed | 30s |
| GPU tests (tests/deepspeed/test_vision_sp.py) | ✅ Passed | 20s |
| GPU tests (tests/deepspeed/test_vision_sp_dp.py) | ✅ Passed | 30s |
| GPU tests (tests/dtensor/test_text_dtensor.py) | ✅ Passed | 20s |
| GPU tests (tests/integration/test_phase4_functional.py) | ✅ Passed | 14s |
| GPU tests (tests/megatron/test_engine_megatron_prepp.py) | ❌ Failed | 196s |
| GPU tests (tests/parallel/test_rdt_non_collocated.py) | ✅ Passed | 28s |
| GPU tests (tests/parallel/test_split_gather.py) | ✅ Passed | 20s |
| GPU tests (tests/parallel/test_vision_compare.py) | ✅ Passed | 30s |
| GPU tests (tests/parallel/test_vision_detailed.py) | ✅ Passed | 38s |


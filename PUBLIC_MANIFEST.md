# Public release manifest

Source-only candidate with reproduced qualification evidence.
Model weights, tokenizer binaries, training data, agent directives and private
benchmark harnesses are intentionally excluded.
Hashes are SHA-256 of the files as committed.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| .gitignore | 139 | 8b2ec16ad919ead95ee8b19161233b9493af6167e18e5f5b6c607a742e7d4e5c |
| ATTRIBUTION.md | 1860 | 50eecf758f29467406be50bed90d183c8b724b608869dd2d212a6281a9dd645c |
| LICENSE | 11343 | 832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e |
| QUALIFICATION.md | 6481 | ac47bc00c72af2eeef0afd290eb25942e069b6a844450e719d25a79c90915be1 |
| README.md | 7084 | 04db2051ead7de7d8cec614298c4bd681793412a5b69a1e068ff671f1ba223eb |
| REPRODUCTION.md | 3221 | 1f11f1cdff4b5fadd90893a6b7818ae88b8a90582b68a18c2cecc32d6bcc2603 |
| evidence/README.md | 1709 | 553518a47b1ef9145a1b4c5ca91a7e634027a23f1fd34b441a315ca55da775e9 |
| evidence/banking77_result.json | 441 | 34e96a19025bc9000c971f26da6d478e886ccd53edc6a256888483328d75a2cc |
| evidence/benchmark_v3_matched_typed.json | 429 | c8c505fe3803a0a38167de1b46984fcfaf2b875264e492d10a8fd35d236d0db9 |
| evidence/benchmark_v3_release.json | 619 | aa3ec4fb4c34182202f79d5272deaee91a67417235dd90ad56f5bb1b65ce0fff |
| evidence/export_metadata_audit.json | 1258 | bf40ff304d4aa1b677c6aa4fb81ff4a6f511a053374db94f475ced563a3ccb0f |
| evidence/fp16_comparison.json | 1298 | 1424a783b4244de742d0e780ce66191ea154640dbb42b4111723735a9f925a42 |
| evidence/fp16_variant_verification.json | 2488 | af693c0eb394dcf7ebde1f2425698fbff47a9e7ab3d3c221a6fc036e9c1dbd03 |
| evidence/gate_results_fp16.json | 1699 | 9b72540df53b69c1591b664fbdab55046fc6f1695fc53e6f5766c119190b1b55 |
| evidence/gate_results_trained_full.json | 1695 | 382f62be62ec49d513765adf071a9a2dd76be9df28184361df6bedd8ddd67f76 |
| evidence/gpu_verify_results.json | 1714 | 271f7928162eec5f60260045598f0689c61bb7c147a19a0b2aab04c91329dbd3 |
| evidence/probe_architecture_results.json | 1484 | 8da612dda73efaea72ea97e9de771ad848403f9f9518c5650bf52c2ac2498c63 |
| pyproject.toml | 1547 | 135d2c775286d9dde2cde66f294df6f8be23d61e23bc4ebdead7a98bf5f2c082 |
| scripts/compare_fp16_eval.py | 5629 | 327ba5bd6d611bc17a6525a298b219d9df31a5d1c9c73bf6ffe542dd21f63150 |
| scripts/evaluate_trained.py | 5370 | 5c58696419be2106790d7bbea07355ad1c49364f67c455705ba8e7381b682203 |
| scripts/export_fp16_variant.py | 4518 | a17e68399ae112dee71a7cd4a4d7a34724eec185ce4d7e404294052e74d397d3 |
| scripts/export_v3_checkpoint.py | 4778 | 4c2ca026e43428a4282068b5263e3ec647a2a52559a6d105f7ab30e18de38631 |
| scripts/gpu_verify.py | 9738 | fa3e0f46a2db106971bf429f3839b21646845ebbd11f5e776e7f88f73b960fa6 |
| scripts/probe_architecture.py | 3746 | f0f229c668e80539d5cda1b56ed9e0add2661d73833fc98484fda04fefcd64e0 |
| scripts/probe_eval_determinism.py | 2347 | 931736b15edfaaf2d07378abe45a7c3bb90814a23a631ff14f691759576df558 |
| scripts/reload_parity_child.py | 1575 | 4e2033f48ca67c837055c27f15be8382756bcd95354ec5133fdd74c45877c9e4 |
| scripts/run_systemone_gates.py | 14004 | e89c1d5bc743ed18db27aafb83b66bbf012c77887b77bb8a67e610f8fe8dbd29 |
| scripts/train_jev_distill.py | 30380 | c9c63cc6e885b0163a037244eebcefe01bd5e7608721c83bc74fb616e5fa5e72 |
| scripts/typed_decisions_data.py | 4944 | df99a553f394e34a4d54acac997daaa39d4c713ca72ce6cd1094b94675a1f401 |
| scripts/verify_card_claims.py | 20294 | 07fd1260921457ea526c704f5502a0683401c01f8402a35ad8920cd46cfbb2d7 |
| scripts/verify_export_metadata.py | 8608 | 36ca7a9edf454a3d349c45884c493034c6ed82f57675953248c7c1411b4fd8a3 |
| scripts/verify_fp16_variant.py | 8820 | 297c0e2a61a053eedbc45506627f9d6df5eef16a84702dfb482f53454fb5283f |
| serving/configuration_rockandegg_v3.py | 3218 | 2653c55677c42630468af4981ff0be4ec1a3bba198da430cfe779f4b1d46e23b |
| serving/modeling_rockandegg_v3.py | 14695 | 4c35986ed31bcf0c0acb72006376da3d396360a341571ea9366555751c8487b2 |
| serving/requirements-server.txt | 769 | 02d77cb11d75bf5101892e91573d8b152700aae43a7aa626aaebaae062e74e1f |
| serving/systemone_inference.py | 7178 | 84cf13fe6c730875bafbc38c8fde875ed7f540089c49cf939285e91407535bae |
| serving/systemone_server.py | 25329 | 9a27ccdf59f7d986dcb7e65db80d019787bf766fd6ceebb9ac558cc15f15a051 |
| serving/test_systemone_server_cpu.py | 11201 | 15aa862c0765cc1afe85a32be904aa53567074be7cb313de268be4fb0bde5f12 |
| src/rockandegg_v3/__init__.py | 1605 | f7894e10e2e32080117ba91d5808c66cff435b0a9e6e99344cb96541a2ac6e12 |
| src/rockandegg_v3/config.py | 3340 | 3501f580f5b76f43c4388667d3f4069b55b5902dbad39b50225d0b634dfd9d6a |
| src/rockandegg_v3/gates.py | 6775 | e0f1b0c3f9405960b1a173f1cc5f05a85cb664aaaddfb56dd6020f5fee8e4f9a |
| src/rockandegg_v3/inference.py | 6283 | 00d94c0f882bc844fb9937b33ff722689caff44912dc14830decae28cc1bc16a |
| src/rockandegg_v3/loss.py | 2729 | bd6d3f86d24de3f156aa0bc97f08ed471247119bb8fdfeb467a2a89f8cce7fcb |
| src/rockandegg_v3/model.py | 13451 | 8a13c75527e07843e006acb1b2bdaf41284f93bcfd34d024e2da35aaa561923c |
| src/rockandegg_v3/parallel.py | 6882 | 52495257288c02ec7b4e31eb45c39334bd124d81160591162d28a032d1c5ccd2 |
| src/rockandegg_v3/py.typed | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| tests/__init__.py | 55 | 6af9c001cc3bdece4b80437c234c98477d67882bfdd6f5f746af76c423315186 |
| tests/test_core.py | 6690 | fea4a83c34b06f6bbe960bb9df03f4a17b03561bc1d4d804983fa8e1a158967c |
| tests/test_parallel.py | 898 | 0d62d117b90252d822f581dea00060cc6cedae9176e771fac4e5617e3574b6d7 |
| tests/test_training_epochs.py | 5931 | 4cd3c61f890c875444dfe951888fe5014639a15752d01ddfee0c6f1f5e856530 |
| tests/test_typed_decisions.py | 4562 | 6f00b3056b73d54642379c6d85cfd305f7072c1ed5975031d024d200aca42e87 |

Total files: 51

## Explicitly excluded

- Private agent directives, informal notes, falsified-claim registers and internal workpapers (`AGENTS.md`, `notes.md`, `prompt*.md`, `VALIDATION.md`, `EVALUATION_CLAIMS*.md`, `SERVER_DESIGN.md`, the private mission directive, and others).
- Private-dependency code/tests (`banking77_probe.py`, `benchmark_vs_laya.py`, `tests/test_ece_definition.py`).
- Published weight binaries (~1.9 GB), tokenizer binaries, caches, logs, local checkpoints and original Git history.
- Evidence paths were normalized to public identifiers before bundling; measured values were not changed.

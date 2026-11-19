# DATA_LINEAGE_AUDIT.md — Phase I data lineage audit

**Date:** 2026-11-18 (UTC)  
**Scope:** RockAndEgg-313M-v3 ensemble-distillation task, Phase I only.  
**Machine-readable companion:** `artifacts/lineage_audit.json` (includes all 1020 TRAIN / 90 DEV / 90 RESERVED / 400 TEST dossier IDs and all measured hashes).  
**Scratch evidence:** `audit_work/lineage_split_audit.py`, `audit_work/lineage_split_result.json`, `audit_work/jev_leak_probe.py`, `audit_work/jev_leak_result.json`, `audit_work/build_companion.py`.

Evidence labels used below: **VERIFIED** = produced by an executed command in this audit; **OBSERVED** = read directly from a named file; **INFERRED** = deduction from those two, explicitly marked.

---

## 1. Scope & authority

- This audit is read-only with respect to every pre-existing file, checkpoint, dataset and manifest. No push/publish. New files created: this report, `artifacts/lineage_audit.json`, and the `audit_work/` scratch directory only. (Command record: `ls -la`, `git status --short` at audit start; `mkdir -p audit_work`.)
- Environment actually used (VERIFIED — executed `python -c "import sys,torch,transformers,datasets; ..."` plus `nvidia-smi --query-gpu=...`):
  - interpreter `/opt/pytorch/bin/python`, Python 3.13.18, torch 2.13.0+cu130, transformers 5.23.0, datasets 5.2.0;
  - one **NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB**.
- `AGENTS.md` is **stale about hardware**: it claims 2×T4 / `.venv/bin/python` / no `datasets`. VERIFIED false for this runtime (`nvidia-smi`, absence of `.venv`, successful `datasets` import). It is not used as evidence for any hardware claim in this report.
- Authority order for claims: executed commands and file contents (quoted with paths/line numbers) > audit reports > model cards. No number in this report was copied from `AGENTS.md` or from `prompt.md`.

---

## 2. Dataset identity & hashes

**VERIFIED — cache identity** (commands: `find /home/ubuntu/.cache/huggingface/hub -maxdepth 3 -iname '*typed-decisions*'`, `ls -la .../snapshots/`, `cat .../refs/main`):

| Property | Value |
|---|---|
| Dataset / config | `LocalLLaMA/typed-decisions`, config `all` (`train_full_backbone.py:38-40`) |
| Pinned revision | `e6aa632c4db73b59c7713bb21066ad1a8e1bfce6` |
| Hub snapshot | `/home/ubuntu/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/snapshots/e6aa632c4db73b59c7713bb21066ad1a8e1bfce6` — **the only snapshot dir present**; `refs/main` contains the same commit SHA |
| Arrow cache | `/home/ubuntu/.cache/huggingface/datasets/LocalLLaMA___typed-decisions/all/0.0.0/e6aa632c4db73b59c7713bb21066ad1a8e1bfce6/` |
| `all/train` parquet blob SHA-256 (HF blob name) | `46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5` |
| `all/test` parquet blob SHA-256 (HF blob name) | `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c` |

**VERIFIED — counts and structure** (`audit_work/lineage_split_result.json`, produced by `audit_work/lineage_split_audit.py`):
- train: **1200 dossiers / 6000 decisions**; test: **400 dossiers / 2000 decisions**; every dossier has exactly **5 questions** (`questions_per_dossier_train = questions_per_dossier_test = [5]`).
- `id` prefix: train dossiers are `tr_<workflow>_<nnnnnn>`; test dossiers are `<workflow>_<nnnnnn>` (sample: `tr_agent_trace_observability_000000` vs `agent_trace_observability_000000`). Workflows: agent_trace_observability / customer_service / invoice_processing / security_incidents.
- Test dossier IDs have **zero overlap with all 1200 train dossier IDs** (measured intersection `0`; `audit_work/lineage_split_result.json:test_split.overlap_with_all_train_dossiers`).

**Hash convention (VERIFIED, must be quoted with any hash):** `sha256("\n".join(sorted(ids)).encode("utf-8"))`, no trailing newline. A second variant `sha256(json.dumps(sorted(ids)).encode("utf-8"))` is also recorded. The TEST hash under the newline convention is
`441ba1b97b99f2e409e82e7186a1458751696f2e3fb7de5a573fe65185b00852` — **byte-identical** to the pre-existing `artifacts/clean_panel_manifest.json:sorted_dossier_ids_sha256`, an independent corroboration.

**Do not conflate a different published hash:** `sia_hu_measurements/measurement_record.json:15-17` records `sha256_canonical_train_dossiers = 4423fa8d…6e849` computed over *content* (`case_id|kind|options|target|sha256(state)` across 6000 decisions). It is not an ID-list hash; this audit did not attempt to reproduce that content serialization. No conflict between the two definitions.

---

## 3. Pipeline A — full-backbone campaign (RELEVANT to the 3 checkpoints)

### 3.1 Exact transformation, executed

Source: `train_full_backbone.py:74-164` and `:541-554`, executed by importing the real functions (not a reimplementation):

1. `load_cases(split)` expands each dossier into one `Case` per question with `case_id = f"{row['id']}:{qid}"` (`:97`). Dossier id = `case_id.rsplit(":",1)[0]` (`:134-136`).
2. `split_by_dossier(cases, 0.15, 0)`: `sorted({dossier_id(c)})`, `random.Random(0).shuffle`, `n_val = round(1200*0.15) = 180` → **1020 TRAIN / 180 VAL** dossiers (`:139-148`).
3. `split_dev_reserve(val)`: `sorted(180 ids)`, `random.Random(20261119).shuffle`, `half = 180//2 = 90` → **90 DEV / 90 RESERVED** dossiers (`:151-164`, `DEV_RESERVE_SEED = 20261119` at `:45`).

### 3.2 Exact counts and hashes (VERIFIED)

| Set | Dossiers | Decisions | sha256 sorted IDs (newline) | sha256 sorted IDs (json) |
|---|---:|---:|---|---|
| ALL train source | 1200 | 6000 | `1a388092396a8a8b119ace94b19e081212dbd664a5e4db9f0561777e3d9c239d` | `24a25baa60d7d3b858da52d9281cd8b727958c9d6ecf25e271dce410a4690ca8` |
| TRAIN | **1020** | 5100 | `790c02bad6f04834479a10f87647325a6c0a94cfd258ccbed78ad36172256552` | `f2bd0389cc3c071f41902dd85944b9509e456be1a0163a14a26b623ea2a7e2f1` |
| VAL | 180 | 900 | `c4f4489aafb551f73e71e7ce526d45d47fd3717c5ddc3d3609bf15c530c899b6` | `5f135981644ebb185739405495176f69a34a86c4870b237a9296556f115bae54` |
| **DEV** | **90** | 450 | `ca1260b66d2cdfa5dd523344829652a3ef65bcf3792b65dfc2058e1e8b477afa` | `316de85d52f2f507ea7082dbfabd3d363d2bb9be399411501f98c611d5c7df64` |
| **RESERVED** | **90** | 450 | `9520ba0790f64b7dc83a1b40e4ec523cb435af28463fc9c01aeda731c7ff58d9` | `c2978d2349353553039ffc1cd82f7173d702d40b77c01447f9c0f8b07179f22e` |
| TEST | 400 | 2000 | `441ba1b97b99f2e409e82e7186a1458751696f2e3fb7de5a573fe65185b00852` | `474af167211d9dd3a65fa2a8c6c124d647555b2071fec582fe89f3e78b87352a` |

Exact ID sets: `artifacts/lineage_audit.json:pipeline_A_full_backbone.train_ids / dev_ids / reserved_ids / test_ids`. First/last IDs (VERIFIED): TRAIN `tr_agent_trace_observability_000000 … tr_security_incidents_000299`; DEV `tr_agent_trace_observability_000024 … tr_security_incidents_000298`; RESERVED `tr_agent_trace_observability_000006 … tr_security_incidents_000289`; TEST `agent_trace_observability_000000 … security_incidents_000099`.

Per-workflow dossier counts (VERIFIED): TRAIN 258/251/250/261, DEV 21/22/27/20, RESERVED 21/27/23/19, TEST 100/100/100/100 (agent_trace / customer_service / invoice_processing / security_incidents).

### 3.3 Overlap table (VERIFIED — all measured, not assumed)

| Intersection | Count |
|---|---:|
| TRAIN ∩ DEV | **0** |
| TRAIN ∩ RESERVED | **0** |
| DEV ∩ RESERVED | **0** |
| DEV ∪ RESERVED == VAL (set equality) | **True** |
| TEST ∩ (all 1200 train dossiers) | **0** |

### 3.4 Checkpoint selection used DEV only

- `train_full_backbone.py:551-554` sets `selection_cases = val_cases`, then replaces it with the DEV half when `dev_selection` is true; `reserved_cases` is never evaluated in the training loop. Comment at `:155` states RESERVED is never used for selection.
- `train_full_backbone.py:612-623`: `_evaluate_and_track()` calls `evaluate_val(model, selection_cases, ...)`, updates `best_acc`, and writes `run_dir/"best"` **only on DEV improvement**.
- Per-run `artifacts/runs/<run>/summary.json` (OBSERVED) explicitly records `"selection_set": "DEV"`, `"reserved_used_for_selection": false`, with `best_val_accuracy` / `best_step`: seed0 0.7800 / 320, seed1 0.7533333333333333 / 400, seed2 0.7711111111111111 / 400. Backbone trainable = 312,892,928 params, head = 268,805 (same files) — corroborating an unfrozen backbone.
- `eval_full_backbone.py:42-61` rebuilds DEV/RESERVED with the same functions and `split_by_dossier(train_cases, 0.15, 0)`; `eval_full_backbone.py:185` evaluates only `("DEV", dev), ("RESERVED", reserve)`. **No `split="test"` call exists in `eval_full_backbone.py`** (VERIFIED by full-file grep of `test|TEST`). Its output `artifacts/final_eval.json` declares `protocol: "RESERVED evaluated once; Laya is out-of-fold; paired dossier bootstrap"`.

---

## 4. Pipeline B — public/older Jev-distill pipeline (SEPARATE; NOT the 3 checkpoints)

### 4.1 Loading is question-level

`typed_decisions_data.py:88-130` (`load_typed_decisions_cases`) expands each dossier into one `Case` per question with `case_id = f"{row['id']}:{qid}"` (`:122`). Note `:103` calls `load_dataset(DATASET_NAME, "all", split=split)` **without the pinned revision** (uses the cache's `refs/main`, which currently equals the target SHA — VERIFIED in §2; this is a reproducibility weakness, not a leak).

### 4.2 The exact splitting code (quoted from `train_jev_distill.py`, `args.dataset == "typed"` branch)

```
515:    if args.dataset == "typed":
516:        from typed_decisions_data import corpus_stats, load_typed_decisions_cases
...
523:        pool = load_typed_decisions_cases("train")
524:        rng = random.Random(args.seed + 11)  # noqa: S311 - deterministic split
525:        shuffled = list(pool)
526:        rng.shuffle(shuffled)
527:        calib = shuffled[:400]
528:        val = shuffled[400:800]
529:        train = shuffled[800:]
```

The unit of `shuffled` is a **question-level Case** (`id:qid`), so the slice boundaries cut through dossiers. (`stratified_sample` at `:240-250` also shuffles individual cases.)

### 4.3 Verdict: CONFIRMED for pipeline B

Executed reproduction of lines 523-529 at the script's default `seed = 20261116` (`train_jev_distill.py:492`), `audit_work/jev_leak_probe.py` / `audit_work/jev_leak_result.json`:
- pool 6000 decisions / 1200 dossiers → calib 400, val 400, train 5200;
- **616 of 1200 dossiers appear in more than one panel**: 263 in calib+train, 262 in train+val, 91 in all three;
- examples: `tr_agent_trace_observability_000003` and `_000004` appear in calib, train **and** val.

**This is question-level → dossier-level leakage. CONFIRMED for `train_jev_distill.py --dataset typed` only.**

### 4.4 Non-transfer statement (explicit)

This pipeline is **separate** from the full-backbone campaign. It trains only the SystemOne head with a frozen backbone (`train_jev_distill.py:505-512`, `HeadConfig(dtype_name="fp32")`, `head_parameters(m)`), and its splits are the question-level slices above. The 3 checkpoints under audit come from `train_full_backbone.py` (backbone unfrozen at `:386-393`; dossier-grouped splits at `:139-164`; DEV-only selection). **The Jev leak finding must NOT be transferred to the full-backbone campaign or to any of the 3 seeds.**

Related provenance nuance (OBSERVED): the *shipped* release head (RAE-006) was trained with `--dataset` ≠ typed, i.e. the `else:` branch at `train_jev_distill.py:544-561` on `SargeDev/jev-distill-corpus-v3` jsonl splits (`/scratch/RockAndEgg-313M-v3/PROVENANCE.md:73-93`), so the typed-branch leak does not touch it either. The typed branch produced only the non-shipped RAE-007 matched-data control (`PROVENANCE.md:105-123`). Current `train_jev_distill.py` SHA-256 = `d4159a6f2632e20e33a0547f7323074183dbe28b188fac9b5c4b34eafae53987` (VERIFIED `sha256sum`), matching PROVENANCE's "current" value; the "as run" release value `7996075b…` differs.

---

## 5. Teacher / student checkpoint provenance & exposure

### 5.1 Published RockAndEgg checkpoint (`/scratch/RockAndEgg-313M-v3`)

- **NOT trained on typed-decisions train dossiers** (OBSERVED, high confidence). `README.md:193-195`: head trained on `SargeDev/jev-distill-corpus-v3`; `README.md:455-457` and `:500-502`: typed-decisions `test` (400/2000) was evaluation-only for the release head, and the typed-trained "matched head" is a benchmark control **not part of this repository**. `PROVENANCE.md:73-93` (RAE-006) records corpus, revision `75111a13a8455032f0435050bfc7e0c9026ecfe8`, 80,000 train cases, seed 20261116, frozen backbone; `PROVENANCE.md:105-123` labels RAE-007 "EVALUATION ONLY, NOT SHIPPED".
- Local `model.safetensors` SHA-256 `3b208b86d6b5b87306091cbaec1b71007454eb4864b47efc636b4f633c875ac8` (OBSERVED: `/scratch/RockAndEgg-313M-v3/MANIFEST.sha256`; also cited in `sia_hu_measurements/measurement_record.json`).

### 5.2 Laya checkpoint (`convaiinnovations/laya-typed-decisions`)

- **YES, fine-tuned on exactly the 1200 typed-decisions train cases / 6000 decisions** (OBSERVED). Local cached model card `.../models--convaiinnovations--laya-typed-decisions/snapshots/989e28febe5886389496fe13a8036c62b2a20ae0/README.md:112-119`. Corroborated by `clef_distillation_report.md:60-61`. Checkpoint SHA-256 `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e` (`artifacts/ssf1-frozen/SSF1_REPORT.md` integrity table).
- Consequence (INFERRED): Laya is in-sample on the 1200 train dossiers; the 90 DEV and 90 RESERVED dossiers are a subset of those same 1200, so a direct Laya-vs-student comparison on DEV/RESERVED needs the out-of-fold Laya control used in `artifacts/final_eval.json`. That control exists (`clef_distillation_report.md:79-95`, 5 folds of 960/240 dossiers) and was used.

### 5.3 The 3 full-backbone checkpoints (VERIFIED by `sha256sum`, `ls -la`, `cat`)

| Path (`artifacts/runs/…/best/model`) | model.safetensors SHA-256 | seed | split_seed | epochs | best step / DEV acc | tokenizer.json + tokenizer_config.json |
|---|---|---|---:|---:|---|---|
| `fullbb_bf16_e2style` | `1b7fee92a9f8825ac001100232ad87e740bd7e0808ee49abc46386cc89c53f92` | 0 | 0 (key absent in run-root config; default applied by `train_full_backbone.py:545`; present in `artifacts/fullbb_bf16_e2style.json`) | 8 | 320 / 0.7800 | present |
| `fullbb_bf16_e2style_s1` | `209ae6012ecd0d3c2b974954a7b0e66d99a1b84605a5a11eb24d0edffea52e3d` | 1 | 0 | 8 | 400 / 0.7533 | present |
| `fullbb_bf16_e2style_s2` | `39af46c6663a2dd3b6d47eb56d3fa45d9200cd440b686f0928e94ae39d57bcd1` | 2 | 0 | 8 | 400 / 0.7711 | **MISSING** |

All three `best/model/config.json` files are otherwise identical: `architectures=["RockAndEggV3SystemOneModel"]`, `model_type="rockandegg_v3_systemone"`, 10 layers, hidden 1024, head_dim 128, `head lm_head:false / generation:false`, `dtype:"float32"`, `transformers_version:"5.23.0"`. Training configs `artifacts/fullbb_bf16_e2style.json`, `..._seed1.json`, `..._seed2.json`: `split:"train"`, `val_fraction:0.15`, bf16 autocast with fp32 master weights, `lr_backbone:2.5e-5`, `lr_head:1e-4`, `eval_every_steps:40`, `early_stop_patience:6`, "no teacher". No config or summary records any use of TEST or RESERVED for selection (§3.4).

**Confirmed defects / caveats in these checkpoints:**
1. **seed2 cannot be loaded standalone for inference** — `tokenizer.json` and `tokenizer_config.json` absent from `best/model`; tokenizer must be taken from seed0 (`fullbb_bf16_e2style/best/model`) or `/scratch/RockAndEgg-313M-v3` (VERIFIED `ls -la`).
2. **Stale inherited metadata** — all three saved `config.json` files still declare `systemone_head.backbone_frozen: true` and `training_dataset: "SargeDev/jev-distill-corpus-v3"`. Both are false for this campaign: `train_full_backbone.py:386-393` sets `backbone_frozen=false` and `requires_grad_(True)` for all backbone params, summary.json records 312,892,928 trainable backbone params, and the training config says `split:"train"` on typed-decisions. The frozen flag is re-applied only at save (`:515-516`) for deployment. Treat the config's `training_dataset` field as inherited from the base checkpoint, not as lineage evidence.
3. seed0 run-root `config.json` lacks the literal `split_seed` key (default 0 applies); seeds 1/2 include it. Reproducibility hazard only — actual split is identical (same `dossier_id` hash set shared by all 3, as executed in §3).

---

## 6. Panel usage history & clean-candidate assessment

| Panel | Size | Usage history (with citation) | Status |
|---|---|---|---|
| DEV (typed-decisions train dossiers) | 90 / 450 | Checkpoint selection for all 3 seeds (`train_full_backbone.py:612-623`; 3× `summary.json`); E1–E8 and Clef DEV decisions (`clef_distillation_report.md:116-131`, `artifacts/dev_clef_vs_e2.json`); `compare_dev.py`; `eval_full_backbone.py` DEV arm; teacher-target transfer set in current `experiments/ensemble_validation_20261118/distillation_preregistration.md:8` | **EXPOSED** |
| RESERVED (typed-decisions train dossiers) | 90 / 450 | Evaluated once in `artifacts/final_eval.json` / `final_comparison.md:17` after protocol freeze; `clef_distillation_report.md:143-148` notes it was **not** used for model selection and per operator instruction not used further; `eval_full_backbone.py` RESERVED arm | **EXPOSED-UNTIL-PROVEN** (one-shot, but probed) |
| SSF-1 (separate corpus, `moeshawky/ssf1-frozen`, 80 dev + 400 sealed dossiers × 8 questions) | not typed-decisions | C2 calibration selection by 5-fold dossier-grouped CV on its 80 DEV dossiers and one sealed 400-dossier evaluation (`ssf1_calibration.py:41,42,56,640,813-846`; `calibration_report.md:1-10,63,106-149`); only seed-0 checkpoint touched, byte hash guarded and unchanged. Post-mortem records a degenerate DEV noul temperature fit and a sealed NLL regression that was **reported, not tuned against** | **EXPOSED for SSF-1**; does not touch typed-decisions TEST IDs |
| TEST (typed-decisions `all/test`) | 400 / 2000 | Historical diagnostics on *other* checkpoints: `benchmark_vs_laya.py:4,410`, `make_laya_test_teacher.py:45`, `generate_clef_teacher.py:120-130,210-211` (guarded `--split test`, `diagnostic_only`, `never_use_as_training_targets`), `clef_distillation_report.md:134-153`, `artifacts/e{1,2,3}_test_results.json`, `artifacts/e8_clef_test_results.json`. **Never used by `train_full_backbone.py` or `eval_full_backbone.py`** (verified by file grep). Currently used as the "clean panel" by the parent experiment (`artifacts/ensemble_validation_report.json`, panel hash `441ba1b9…`) and its frozen 200/1000 half `artifacts/final_eval_panel_v1_manifest.json` | **CLEAN for the 3 seeds' weights** (see assessment) |

### Assessment — is TEST a clean panel for validating THESE 3 full-backbone seeds?

**HIGH confidence at the weight/selection level.** Justification, each independently verified:
1. TEST dossiers are disjoint from all 1200 source-train dossiers (`0` overlap, §2), hence disjoint from the 1020 dossiers actually trained on and from DEV/RESERVED.
2. No TEST load or TEST metric exists in `train_full_backbone.py` or `eval_full_backbone.py`; all three `summary.json` files record `selection_set="DEV"`, `reserved_used_for_selection=false`. The 3 seeds' `best` checkpoints were written solely on DEV improvement.
3. Historical TEST use is on *different* checkpoints (release head, E1–E3, Clef-matched/Laya), so it changed no weight in these 3 seeds; SSF-1 exposure was on a different corpus and on seed0 only, with the model hash unchanged.

**MEDIUM confidence at the analyst/protocol level.** TEST is not pristine in the absolute sense: it has been repeatedly inspected historically and, as of this audit, is being used to validate this exact ensemble (Phase III on all 400, Phase IV/V on a pre-registered frozen 200-dossier half — `experiments/ensemble_validation_20261118/distillation_preregistration.md`, `artifacts/final_eval_panel_v1_manifest.json`). That use is *post-freeze validation*, and the preregistration explicitly refuses to switch seed0 to seed1 despite seed1 scoring 1.45 pts higher on the panel — a genuine contamination control. But the 200 dossiers used for Phase III are no longer "untouched" for any *new* post-hoc choice about these 3 seeds; any further re-selection or re-tuning against the full 400 would downgrade this assessment.

Bottom line: **TEST qualifies as the cleanest available panel for these 3 seeds, with the stated high/medium split in confidence. RESERVED does not (one-shot probed); DEV does not (used for selection); SSF-1 is not on-corpus.**

---

## 7. Confirmed defects

1. **Pipeline B question-level dossier leakage** — `train_jev_distill.py:523-529`; 616/1200 dossiers span panels (§4.3). CONFIRMED, pipeline B only.
2. **seed2 checkpoint missing tokenizer files** — `artifacts/runs/fullbb_bf16_e2style_s2/best/model/` (§5.3). CONFIRMED.
3. **Stale provenance metadata in all 3 saved configs** — `backbone_frozen:true` and `training_dataset:"SargeDev/jev-distill-corpus-v3"` contradict the actual unfrozen typed-decisions training (§5.3). CONFIRMED by file reads and `train_full_backbone.py:386-393`.
4. **seed0 run config omits `split_seed`** — default 0 silently applied (§5.3). CONFIRMED, minor.
5. **SSF-1 C2 calibration DEV fit is degenerate / sealed NLL regresses** — reported in `calibration_report.md:106-149`, not tuned against sealed; the checkpoint bytes were not modified. OBSERVED from report, not an independent re-measurement by this audit.
6. **`typed_decisions_data.py:103` does not pin the dataset revision** while `train_full_backbone.py:76` does — pipeline B can silently follow `refs/main`. OBSERVED.

---

## 8. Unresolved questions

- **UNKNOWN — leak present in the *as-run* release script?** `PROVENANCE.md:139` records a different SHA-256 for `train_jev_distill.py` at release time (`7996075b…`) vs current (`d4159a6f…`); the documented delta is an epoch-loop fix (`:151-169`). Whether the typed-branch split code was identical then is not established. **Load-bearing? NO** for these 3 seeds (release head uses the other branch; 3 seeds use a different script).
- **UNKNOWN — exact set of panels inspected after Phase III.** The parent experiment's artifacts were being written concurrently (`experiments/ensemble_validation_20261118/`, timestamps 23:02-23:05). This audit read them but did not freeze their final state. Any new post-hoc test run against the full 400 after the 200-dossier final panel was frozen must be recorded by the parent.
- **UNKNOWN — whether C2 calibration parameters were applied at inference during any TEST measurement of the 3 seeds.** `calibration.json` exists and `sia_hu_calibration_intervention.json` records `applied_exactly_once: true` with unchanged checkpoint hash; `eval_full_backbone.py` uses `V3Arm` from `typed_eval.py` and does not load `calibration.json`. Probability-level (ECE/NLL) comparisons should state which convention was used.
- **INFERRED, not independently reproduced** — the `4423fa8d…` content hash in `sia_hu_measurements/measurement_record.json`; its method string is quoted in §2 but not re-executed here.
- **NOT EVALUATED** — no typed-decisions held-out ECE gate for the 3 seeds is claimed by this audit; it reports lineage only.

---

## 9. Reproducibility commands

```bash
cd /scratch/rockandegg-v3-code

# Runtime identity
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/opt/pytorch/bin/python -c "import sys,torch,transformers,datasets; print(sys.version, torch.__version__, transformers.__version__, datasets.__version__)"

# Dataset identity from cache
ls -la /home/ubuntu/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/snapshots/
cat /home/ubuntu/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/refs/main

# Pipeline A — executes the real train_full_backbone functions (repo root must be on PYTHONPATH)
USE_TF=0 PYTHONPATH=.:src:/scratch/RockAndEgg-313M-v3 \
  /opt/pytorch/bin/python audit_work/lineage_split_audit.py
# -> audit_work/lineage_split_result.json (counts, hashes, overlaps, full ID lists)

# Pipeline B — exact question-level split reproduction
USE_TF=0 PYTHONPATH=.:src:/scratch/RockAndEgg-313M-v3 \
  /opt/pytorch/bin/python audit_work/jev_leak_probe.py
# -> audit_work/jev_leak_result.json

# Checkpoint hashes
sha256sum artifacts/runs/fullbb_bf16_e2style/best/model/model.safetensors \
          artifacts/runs/fullbb_bf16_e2style_s1/best/model/model.safetensors \
          artifacts/runs/fullbb_bf16_e2style_s2/best/model/model.safetensors
sha256sum train_jev_distill.py

# Companion build (full ID sets)
/opt/pytorch/bin/python audit_work/build_companion.py     # -> artifacts/lineage_audit.json
```

Note: the task-supplied import prefix `PYTHONPATH=src:/scratch/RockAndEgg-313M-v3` omits the repo root, so a script located in `audit_work/` needs `PYTHONPATH=.:src:...` to import `train_full_backbone`. This is a harness detail, not a data finding.

---

**Final verdicts:** dossier counts **1020 TRAIN / 90 DEV / 90 RESERVED** (from 1200; all overlaps 0; test 400 disjoint); Jev leak **CONFIRMED for pipeline B only, 616/1200 dossiers contaminated**; TEST **qualifies as clean for validating the 3 full-backbone seeds at the weight/selection level (high confidence), with medium confidence at the analyst level due to ongoing preregistered use**.

# Q1-E7 preregistration — final-round class decomposition on CIC-IoT2023

**Status:** **ACCEPTED 2026-09-23 — GPU remains gated on Section 7 integrity checks**  
**Draft date:** 2026-09-22  
**Proposer:** **Sheryar Kiani**  
**Approver:** **Dr Hussain Kazmi**  
**Acceptance date:** **2026-09-23** (23-9-2026)  

This document freezes the Q1-E7 scope before any E7 rerun. Named acceptance and the corresponding dated amendment to `Q1_CONTRIBUTION_CONTRACT.md` were recorded on 2026-09-23. Acceptance permits implementation of the isolated runner, but it does not itself satisfy the remaining pre-GPU integrity gates, create class-level evidence, or convert the Q1-E6 conceptual bridge into a measured mechanism.

---

## 1. Purpose and evidential status

Q1-E7 will decompose the already reported **Final (round-30) macro-F1** behavior into fixed, exhaustively reported class-level metrics. It asks which classes contribute to the four paired factorial contrasts under severe CIC label skew.

E7 is a **preregistered descriptive decomposition of rerun final models**, not a new independent confirmation:

- the aggregate CIC confirmatory results are already known;
- the same pooled test set was evaluated at initialization and after every training round;
- the existing histories contain aggregate metrics only and do not retain final checkpoints or predictions;
- therefore all 24 scoped cells must be rerun to create auditable final-round predictions and class metrics;
- no model, threshold, class subset, wording, or continuation decision may be tuned from E7 results.

No significance, equivalence, rare-class improvement, Byzantine robustness, unseen-device, deployment, or causal mechanism claim is authorized by this preregistration. Q1-E6 remains a conceptual explanation, not a theorem. E7 may support bounded statements about the observed named classes only after the complete packet is frozen.

---

## 2. Binding experiment scope

| Item | Frozen value |
|---|---|
| Dataset | CIC-IoT2023 only |
| Preprocessing lineage | `scale_on=train` only; never mix with `scale_on=full` |
| Existing cache | `cic_27bfed227ef77fab.npz` |
| Cache key | `27bfed227ef77fab` |
| NPZ SHA-256 | `8f032df1760a895b2358b8ff2da71491b09820c7af8c097dd00465fd7b981b38` |
| Metadata SHA-256 | `e5e0ef795fff100af24f6813ca3a2a61442e386aa01bf92878a9cfc0b03cb3b4` |
| Heterogeneity | Dirichlet label partition, alpha = 0.1 only |
| Models | MLP and CNN |
| Arms | FedAvg (FA), UnweightedFedAvg (UW), FedTrimmedAvg (FT), WeightedFedTrimmedAvg (WT) |
| Seeds | 42, 123, 7 |
| Unique cells | 2 models x 4 arms x 3 seeds = **24** |
| Mode | `match_notebook` |
| Rounds | 30; class analysis uses round 30 only |
| Clients | 9, full participation |
| Client optimizer | Adam, learning rate 0.001 |
| Local work | 3 epochs, batch size 512 |
| Coordinate trim | configured fraction 0.1; one low and one high value removed per coordinate (`beta_eff=1/9`) |
| Test size | 1,206,424 records in immutable cache order |
| Device | CUDA on the Quadro P3200; one E7 process at a time |

The measured duration of the matching 24 confirmatory cells was about **54.7 GPU-hours**. The planning estimate is about **57.9 hours** on the same P3200 and is not a guaranteed floor.

### Explicit exclusions

E7 does not include N-BaIoT, alpha 0.5 or 1.0, FedProx, new hyperparameter tuning, CIC Wave 3, Q1-E4 seed expansion, Q1-E5 baselines, adversarial clients, or a new train/test split. It does not reopen C5 or C8. Q1-E8 remains deferred.

---

## 3. Frozen class order and holdout supports

Labels are the cache's integer labels 0 through 7 in the following exact order. These supports were independently recomputed from `y_test` in the frozen cache and must be reproduced before any class metric is accepted.

| ID | Class | Test support | Test share |
|---:|---|---:|---:|
| 0 | benign | 16,000 | 1.3262% |
| 1 | bruteforce | 2,613 | 0.2166% |
| 2 | ddos | 700,977 | 58.1037% |
| 3 | dos | 152,000 | 12.5992% |
| 4 | mirai | 301,416 | 24.9843% |
| 5 | recon | 16,452 | 1.3637% |
| 6 | spoofing | 12,000 | 0.9947% |
| 7 | web | 4,966 | 0.4116% |
|  | **Total** | **1,206,424** | **100%** |

No post-hoc threshold will define “rare” or “minority.” The manuscript will name classes and report their exact support or frequency.

---

## 4. Required per-cell artifacts and metrics

For each of the 24 final models, the evaluation must use the fixed label list `[0,1,2,3,4,5,6,7]` and `zero_division=0` and must write:

1. Final accuracy and Final macro-F1 reconstructed from stored predictions.
2. For every class: true positives, false positives, false negatives, support, precision, recall, and F1.
3. An 8 x 8 raw confusion matrix with **true class in rows** and **predicted class in columns**.
4. A row-normalized confusion matrix for display; rows with nonzero frozen support must sum to 1 within numerical tolerance.
5. A complete 30-round aggregate history.
6. A final-round model checkpoint stored as named NumPy arrays plus a non-pickled JSON model specification, or an equivalently safe non-pickled format.
7. One ordered `uint8` prediction vector of length 1,206,424.
8. Metadata binding every artifact to the model, arm, seed, alpha, configuration, cache key, cache hashes, environment, timestamps, and artifact SHA-256 values.

One shared `uint8` `y_true` vector may be written once in immutable cache order. No features, source filenames, row identifiers, logits, probabilities, or pickled full-model objects are required or authorized.

The written checkpoint must be reloaded into a fresh model. Predictions used for all reported metrics must be generated from that reload and must match the stored prediction vector exactly.

---

## 5. Frozen descriptive analysis

### 5.1 Seed summaries

Report every seed value. For each model x arm x class, report the arithmetic mean and sample standard deviation (`ddof=1`) across seeds 42, 123, and 7. These are descriptive summaries, not confidence intervals.

### 5.2 Paired factorial contrasts

For each model, class, metric, and matched seed, calculate the same one-factor contrasts as the aggregate experiment:

- adding trim without size weights: **FT - UW**;
- adding trim with size weights: **WT - FA**;
- adding size weights without trim: **FA - UW**;
- adding size weights with trim: **WT - FT**.

Apply the complete contrast matrix to **per-class F1 and recall**. Precision is reported as a diagnostic but is not a directional decision endpoint. Never interpret WT - UW as a one-factor effect because it changes both trim and weighting.

For every class-level contrast, report all three paired seed differences, their mean, their sample standard deviation, and counts of positive, negative, and exact-zero differences. A descriptive `consistent_direction` flag may be true only when all three differences are strictly positive or all three are strictly negative. Exact zero is a tie. The full class-by-model-by-contrast matrix must be reported; selecting only favorable classes is forbidden.

### 5.3 Macro-F1 contribution identity

For each model, seed, and factorial contrast, verify and report:

`delta macro-F1 = (1/8) * sum over classes of delta class-F1`.

The directly reconstructed aggregate difference and the right-hand side must agree within absolute tolerance `1e-12`. A display table may express class contributions as `delta class-F1 / 8`, but it must not imply causal attribution.

### 5.4 No inferential family

There are no E7 p-values, null-hypothesis tests, multiplicity-adjusted claims, equivalence tests, or test-row bootstraps under this freeze. The replication unit is the training seed, not the 1,206,424 test records. The test records must not be treated as independent experimental replicates.

---

## 6. Staged execution and continuation

No E7 runner existed when this preregistration was accepted. Named acceptance and the contract amendment were therefore recorded before implementation. The post-acceptance implementation adds an isolated runner, final-model capture seam, safe NumPy/JSON artifact helpers, freeze builder, and no-training integrity suite without modifying `fl_local/engine.py`, `fl_local/cic_cache.py`, or `run_q1_cic_confirmatory.py`. This implementation does not itself authorize GPU work: the generated freeze and the live pinned-environment preflight must pass and be reviewed before W0.

| Wave | Cells | Purpose |
|---|---:|---|
| E7-W0 | 1 | MLP, FA, alpha 0.1, seed 42; end-to-end checkpoint/reload/prediction integrity gate |
| E7-W1 | 7 | Remaining seed-42 cells |
| E7-W2 | 16 | All seed-123 and seed-7 cells |
| **Total** | **24** | Complete fixed grid |

W1 follows W0 only if W0 passes every technical gate. W2 follows W1 regardless of scientific signs. Class patterns, favorable or unfavorable, are never a stopping rule. Stop only for a predefined technical-integrity failure.

The existing train-scaler cache is mandatory. Cache rebuilding is forbidden. `scale_on=full`, a different cache key, CPU training, incomplete waves, or a provenance mismatch must fail closed.

---

## 7. Integrity gates

### Before any GPU work

- [x] This preregistration is marked ACCEPTED with named proposer, named approver, and acceptance date (**Sheryar Kiani / Dr Hussain Kazmi / 2026-09-23**).
- [x] `Q1_CONTRIBUTION_CONTRACT.md` contains a dated who/what/why amendment authorizing the frozen E7 scope.
- [x] The corrected runner and evaluation-module sources passed the generated file-hash freeze check; `fl_local/engine.py` and `run_q1_cic_confirmatory.py` remain byte-identical to their prior frozen hashes. A final post-record regeneration is still required before replacement preflight.
- [x] The generated freeze contains the ordered 24-cell list, output schema implementation, fixed class order/supports, and three per-seed partition identities (combined hash, client sizes, and client-index hashes). The expanded regression suite passed **17/17**; replacement preflight remains required.
- [x] The required environment is pinned to Python 3.12.10, PyTorch 2.5.1+cu121, CUDA 12.1, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.8.0, and Quadro P3200. The live match remains governed by the separate preflight gate below.
- [x] The existing cache and metadata identities match Section 2; the E7 runner exposes no cache-build option or cache-builder import.
- [x] All 24 existing alpha-0.1 confirmatory rows and their 30-round histories reconcile in the generated baseline-anchor snapshot.
- [x] A replacement live `--preflight` run in the pinned research environment produced `Q1_E7_PREFLIGHT.json` with `status=PASS` and `training_started=false`; its environment, 24 jobs, cache/meta identities, three partition identities, and existing shared-truth raw-value identity were reviewed on 2026-09-23. A final documentation-state refreeze and matching preflight are required because this checklist record is itself a frozen input.

Implementation review record, 2026-09-23: the baseline builder produced exactly 24 rows with independently expected SHA-256 `e43f8e7eebf4f536c691dcc84c93ee9fae36e7b8c7bfe630a79683624718d3e0`. Earlier pre-W0 corrections covered row-key normalization, duration precision, the missing `validate_prediction` import, and fail-fast validation of the shared truth vector; the corresponding suite reached **16/16**. The first failed W0 artifacts were preserved under `q1_e7_failed_attempts/2026-09-23_w0_missing-validate-prediction-import/`. The second W0 attempt again completed all 30 rounds and passed exact aggregate reproduction, checkpoint reload, prediction generation, and class-metric validation, but failed before raw-row commit while moving artifacts: checkpoint JSON and cell-metadata JSON shared the same basename and had collided in the flat staging directory. The staging map was corrected to prefix every temporary filename with its artifact field, and a regression test now requires all eight staging paths to be unique even when final basenames match. Neither failed attempt is a completed E7 cell. The seven partial files and empty staging directory from the second attempt were preserved under `q1_e7_failed_attempts/2026-09-23_w0_staging-basename-collision_x0upcdl/`; the live staging, history, checkpoint, metric, and cell locations were verified empty, the shared `predictions/y_true_uint8.npy` remained in place, and no raw result CSV existed. After this quarantine record was added, the corrected preliminary freeze retained the exact 24-row anchor hash and the expanded pinned-environment suite passed **17/17**. The subsequent replacement preflight bound config SHA-256 `fb811e6cd5149c1bc1ee41905d215008d5a5f20733726d93f656804e4392c6d6` to the expected cache, partitions, environment, and shared-truth raw SHA-256 `fe4ceccf96058485d6896e2a27b4aba0d5950f686a54fc2f53451535e3c408d6`, reported `status=PASS` and `training_started=false`, and left all wave counts at zero. A final documentation-state freeze regeneration and matching preflight remain required before another W0 run.

### For W0 and every completed cell

- exactly 30 ordered history rows, rounds 1 through 30;
- finite Final accuracy and Final macro-F1;
- aggregate round-30 accuracy and macro-F1 reproduce the corresponding existing confirmatory cell; W0 must pass this reproduction gate before W1;
- reloaded checkpoint produces exactly 1,206,424 predictions, each an integer in 0 through 7;
- shared truth vector exactly matches the frozen cache `y_test`;
- raw confusion-matrix total equals 1,206,424 and its row sums equal the eight frozen supports;
- accuracy and macro-F1 reconstructed from predictions agree with the stored aggregate values within absolute tolerance `1e-12`;
- all per-class metrics are finite and use `zero_division=0`;
- every required artifact exists, parses, and matches its recorded SHA-256 before the row is complete;
- result-index writes are atomic; an incomplete artifact set is never resumable as a completed cell.

Scientific signs are not integrity gates.

---

## 8. Planned output tree

E7 must use a new `q1_e7_results/` tree and must not overwrite any CIC confirmatory artifact.

- `csv/q1_e7_raw.csv` — 24-row artifact index and aggregate reconciliation fields
- `csv/history_*.csv` — 30-round histories
- `predictions/y_true_uint8.npy` — one shared truth vector
- `predictions/y_pred_*.npy` — one `uint8` final prediction vector per cell
- `checkpoints/*.npz` and `checkpoints/*.json` — non-pickled named state arrays and model specifications
- `metrics/per_class_*.csv` — complete class metrics per cell
- `metrics/confusion_raw_*.csv` and `metrics/confusion_rownorm_*.csv`
- `reporting/q1_e7_per_class_seed.csv`
- `reporting/q1_e7_per_class_seed_mean.csv`
- `reporting/q1_e7_paired_class_contrasts.csv`
- `reporting/Q1_E7_PACKET.md`
- `reporting/Q1_E7_PROVENANCE.json`
- `reporting/SHA256SUMS.txt`

Paths written into public reporting artifacts must be relative. The manifest must cover the processed cache/meta, frozen configuration, code, histories, checkpoints, prediction vectors, metrics, and packet. It does not retroactively hash the original approximately 309 CIC source CSVs unless a separate source-tree manifest is deliberately added before acceptance.

---

## 9. Storage, release, and privacy boundary

The required `uint8` truth and prediction vectors are small enough for local storage; float logits are unnecessary and must not be saved. Checkpoints and row-level predictions are internal audit artifacts, not privacy guarantees. They may carry membership or inversion risk and must not be publicly released until dataset licensing and model-artifact risk are reviewed. Public release may instead contain the complete aggregate class tables, confusion matrices, configuration, code hashes, and an approved subset of reproducibility artifacts.

---

## 10. Test-set limitation and permitted manuscript language

Final-round reporting avoids choosing the best of 30 test evaluations, but the pooled set is not an untouched validation set. E7 also revisits this already-used set after aggregate results are known. The manuscript must call E7 a **preregistered descriptive final-round class decomposition** or equivalent, not independent external validation.

Permitted after a complete frozen packet: exact observations such as “web-class recall increased in all three seeds for FT - UW in the CIC MLP alpha-0.1 cell,” accompanied by support, all seeds, and the full contrast table.

Forbidden: “trimming protects rare classes” as a general mechanism; “statistically significant”; “robust to Byzantine clients”; selective reporting of only classes with favorable signs; or any inference that the conceptual E6 mechanism was proven.

---

## 11. Acceptance block

Acceptance recorded before runner implementation and before any E7 GPU work:

> Proposer: Sheryar Kiani  
> Approver: Dr Hussain Kazmi  
> Acceptance date: 2026-09-23 (23-9-2026)  
> Accept Q1-E7 as the 24-cell, alpha-0.1 CIC final-round class decomposition defined in this document.

The dated who/what/why amendment was applied to `Q1_CONTRIBUTION_CONTRACT.md` on 2026-09-23. Acceptance does not authorize scope expansion, cache rebuilding, or GPU execution before the remaining Section 7 gates pass.

---

*ACCEPTED 2026-09-23 before implementation. No cache operation or committed E7 result row has occurred. Two full W0 attempts completed training but failed closed during post-training artifact handling; W0 remains 0/1. Both failed artifact sets are preserved outside the live result tree, the expanded suite passed 17/17 after the staging correction, and the replacement preflight passed without training. W0 remains blocked only until this final checklist-state record is refrozen and the matching preflight is reviewed.*

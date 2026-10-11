# Prereg draft — CIC confirmatory 2×2 (`scale_on=train`)

**Status:** **ACCEPTED 2026-09-12** — proposer **Sheryar Kiani**; approver **Dr Hussain Kazmi** (accepted as **Hussain Kazmi**; same person as the E3 freeze)  
**Rev:** r5 — integrity-only: raw rows store NPZ/meta hashes; provenance immutable after results; frozen JSON schema required  

**Date (UTC draft):** 2026-09-12  
**Acceptance date:** **2026-09-12** (12-9-2026)  
**r3:** scoring rule and runner hardening added **before any confirmatory GPU**. Same named parties as r2 ACCEPTED.  
**r4:** integrity-only (same named parties, **before GPU**): parse provenance on every wave; exact integer 30 (not truncating 30.9; inf rejected); decisions use **unrounded stored** Final F1, exact equality = tie.  
**r5:** integrity-only (same named parties, **before GPU**): each raw row stores NPZ/meta hashes; resume verifies them; provenance rewrite refused after any complete result; frozen config JSON must contain the full mandatory schema.  

**Depends on:** E3 contract **Done (prereg cells)** 2026-09-12; `fl_local/PREPROCESS_PROTOCOL.md`

This is a **new experiment**, not a re-gate. After `scale_on=train`, published / E1 / E1b / G6 bit-identity **will not hold**. Do **not** compare these cells statistically to `scale_on=full` runs.

---

## Who (accepted)

- Proposer: **Sheryar Kiani**  
- Approver: **Dr Hussain Kazmi** (credentials text: Hussain Kazmi)  
- Acceptance date: **2026-09-12**  

`--list` remains safe. Cache build and Wave 0+ are now allowed by this prereg.

---

## 2×2 (binding directions)

|  | Unweighted | Size-weighted |
|--|------------|---------------|
| **Untrimmed** | UW (`UnweightedFedAvg`) | FA (`FedAvg`) |
| **Trimmed** | FT (`FedTrimmedAvg`) | WT (`WeightedFedTrimmedAvg`) |

**Adding trim** (weighting held fixed): **FT−UW** and **WT−FA**. If trim helps, **both positive**.  
**Adding size-weight** (trim held fixed): **FA−UW** and **WT−FT**. If weighting hurts, **both negative**.

**Never** treat **WT−UW** as one factor (it changes trim **and** weight).

All four arms **run in every model×α×seed block**; each contrast uses a **matched pair** from that block (not “all four arms in every contrast”).

---

## Protocol (binding)

| Item | Value |
|------|--------|
| Dataset | CIC-IoT2023 only |
| Cache | `scale_on="train"` — split, then scaler on **train only** |
| Mix with `full` | **Forbidden** |
| Mode | `match_notebook` only |
| Device | **CUDA required** (CPU runs are **not** confirmatory; runner has no `--allow-cpu`) |
| Optimizer | **Adam**, lr=**1e-3** |
| Metrics | Final Acc, Final macro-F1, **mean round-wise macro-F1** (not ROC-AUC) |
| F3 | Descriptive only at n=3 |

Frozen CFG / hashes: `q1_cic_conf_results/reporting/Q1_CIC_CONFIRMATORY_FROZEN_CONFIG.json`

| Pin | Value |
|-----|--------|
| max_samples_per_file | 20000 |
| num_rounds | 30 |
| local_epochs | 3 |
| batch_size | 512 |
| lr | 1e-3 |
| test_fraction | 0.2 |
| split `random_state` | 42 (`GLOBAL_SEED`) |
| num_clients | 9 |
| trimmed_fraction | 0.1 (β_eff=1/9) |
| optimizer | adam (FedProx µ unused) |
| partition | `partition_dirichlet(y_train, 9, α, seed=seed)` after `set_global_seed(seed)` |
| MLP | Linear  in→128→64→64→32→C, ReLU, LayerNorm, Dropout 0.2/0.2/0.1 |
| CNN | Conv1d 1→32 k=5, 32→64 k=3, AdaptiveAvgPool1d(1), Linear 64→32→C |

---

## Waves

**Wave 0 is a real 30-round CNN run**, not tooling. **No Wave 0 until acceptance.**  
`--build-cache-only` is also **post-acceptance** (cache build is not a 30-round cell, but still confirmatory-path work).

Planning estimates from P3200 medians on **legacy** CIC `full` runs (MLP **122.66** min, CNN **166.84** min). **Not guaranteed floors.** `train` scaler should not change FL wall much.

| Stage | Unique cells | Planning estimate |
|-------|----------------|-------------------|
| Wave 0 smoke (CNN FedAvg α=1.0 s42) | 1 | ~2.8 h **plus cache build** |
| Wave 1 (C1 cells, seed 42) | 16 | ~38.6 h |
| Wave 2 (C1 cells, seeds 123 and 7) | 32 | ~77.2 h |
| **Core Waves 1–2** | **48** | **~115.8 h** |
| Wave 3 α=1.0 control | 24 | ~57.9 h |
| Wave 3 after Wave 0 reuse | **23 additional** | ~55.1 h |
| **Full unique grid** | **72** | **~173.7 h plus cache overhead** |

Staged job entries = **73** (Wave 0 duplicates one Wave 3 cell). Unique experiments = **72**.

### Continuation (not result-dependent)

| After | Next proceeds **regardless of 2×2 signs** unless a **predefined technical-integrity failure** |
|-------|----------------------------------------------------------------------------------------------|
| Wave 0 | Wave 1 |
| Wave 1 | **Wave 2** (do **not** wait to inspect s42 signs) |
| Wave 2 | optional Wave 3 |

**Technical-integrity failures** (stop; do not start the next wave): CUDA missing; cache `meta.scale_on ≠ "train"`; cache key = `b725017696446067`; history ≠ 30 rounds; Final_F1 not a finite float; history file missing/incomplete; cache key on the row ≠ loaded train key; prior required wave incomplete. **Signs of FT−UW, WT−FA, FA−UW, WT−FT are not integrity failures.**

Do **not** expand to 5–10 seeds in this prereg.

---

## Hypotheses (descriptive)

| ID | Statement |
|----|-----------|
| **HC1** | At α∈{0.5, 0.1}, **FT** ranks first on Final macro-F1 among the four arms (per model). α=1.0: **no ranking claim**. |
| **Htrim** | **FT−UW > 0** and **WT−FA > 0** (trim helps at both weightings). |
| **Hweight** | **FA−UW < 0** and **WT−FT < 0** (size-weight hurts at both trim states), at α∈{0.5, 0.1}. |

### Scoring rule (binding; frozen before GPU)

Final evaluation is **cell-wise** for each **model×α** combination using **mean Final macro-F1** across paired seeds **{42, 123, 7}**.

- **HC1** is supported in a cell only if **FT** has the **strictly highest** arm mean.  
- **Htrim** is supported only if **mean(FT−UW) > 0** and **mean(WT−FA) > 0**.  
- **Hweight** is supported only if **mean(FA−UW) < 0** and **mean(WT−FT) < 0**.  
- **Ties** are reported as **not supported**. Decisions use **unrounded stored** Final macro-F1 values (no display rounding). **Exact equality** of stored means (or of the two arm means in a contrast) is a **tie**.  
- Report **all paired seed differences and sign counts**.  
- **Wave-1 results are preliminary**; **final descriptive assessment occurs after Wave 2**.  
- α=1.0 (Wave 0 / Wave 3): **no HC1 ranking claim**.

Failures do **not** reopen E1/E1b `full` freezes. No F3 “significant” language.

---

## Out of scope

- N-BaIoT, Q1-E2, FedProx, seed expansion, IID UW  
- Mixing `full` vs `train` in one test family  
- Claiming the leak “did not matter” without this grid  

---

## Runner

`run_q1_cic_confirmatory.py`

- `--list` — no cache, no CUDA, no train (safe now)  
- `--build-cache-only` / `--wave N` — require **ACCEPTED** prereg (Who fields filled)  
- CUDA only; every load requires `meta.scale_on == "train"` **and** cache key **27bfed227ef77fab**  
- Frozen `file_sha256` enforced at startup  
- `--build-cache-only` writes NPZ/meta hashes + library versions before Wave 0  
- Resume: **finite** Final_F1, N_Rounds=30 (malformed = incomplete), 30-row history, matching cache key  
- Atomic raw-CSV replace  

```powershell
python3.12 -u run_q1_cic_confirmatory.py --list
python3.12 -u run_q1_cic_confirmatory.py --build-cache-only
python3.12 -u run_q1_cic_confirmatory.py --wave 0 --require-from-cache
python3.12 -u run_q1_cic_confirmatory.py --wave 1 --require-from-cache
python3.12 -u run_q1_cic_confirmatory.py --wave 2 --require-from-cache
```

---

## Acceptance checklist

- [x] Proposer / approver / date (**Sheryar Kiani** / **Dr Hussain Kazmi** / **2026-09-12**)  
- [x] Factor directions: trim **FT−UW** & **WT−FA**; weight **FA−UW** & **WT−FT**  
- [x] Wave 2 not gated on signs  
- [x] No GPU / no Wave 0 until ACCEPTED (this file is ACCEPTED; r3 scoring frozen **before** Wave 0)  
- [x] Frozen CFG JSON present; runner **enforces** hashes + expected cache key  
- [x] Three-seed scoring rule frozen (cell-wise means; ties = not supported; **unrounded stored values**)  
- [x] r4: provenance NPZ/meta verified on **every wave**; integer-exact N_Rounds  
- [x] r5: raw rows store NPZ/meta hashes; provenance locked after results; frozen JSON full schema  
- [x] No F3 “significant” language  

---

*Prereg ACCEPTED 2026-09-12. Not Q1-E2. Canonical claim contract is still `Q1_CONTRIBUTION_CONTRACT.md` (C5/C8 unchanged).*

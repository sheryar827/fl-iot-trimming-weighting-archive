# Prereg — Q1-E2 device-based N-BaIoT partition

**Status:** **ACCEPTED 2026-09-21** — proposer **Sheryar Kiani**; approver **Dr Hussain Kazmi**  
**Date (UTC, first written):** 2026-09-21  
**Acceptance date:** **2026-09-21** (21-9-2026)  
**Depends on:** CIC confirmatory Waves 1–2 **Done** on contract (2026-09-21); C6 / Hussain #8; `fl_local/PREPROCESS_PROTOCOL.md`

This is a **device-based partition experiment**. It is **not**:
- deployment on nine physical devices,
- a test of **previously unseen** devices (test set remains a pooled stratified holdout that includes all devices),
- Dirichlet / IID client synthesis,
- a Byzantine or Q1-E8 attack study.

Do **not** mix results with `scale_on=full` E1/E1b tables.

Wave 3 CIC remains **optional** and is **not** this campaign.

---

## Who (accepted)

- Proposer: **Sheryar Kiani**  
- Approver: **Dr Hussain Kazmi** (credentials text: Hussain Kazmi)  
- Acceptance date: **2026-09-21** (21-9-2026)  

`--list` remains safe. Cache build and Wave 0+ are now allowed by this prereg. Wave 3 CIC is **not** this campaign.

---

## Mapping (binding)

| Item | Rule |
|------|------|
| Dataset | N-BaIoT only |
| Clients | **k = physical device k** (`KNOWN_DEVICES` / LabelEncoder order) |
| Train partition | `fl_local/e2_partition.py` `partition_by_device_ids(cid_train)` |
| Test | **Pooled** global test loader (same row split as other `fl_local` caches: `test_fraction=0.2`, `stratify=y`, `random_state=42`) |
| Scaler | **`scale_on=train`**: split first, fit `StandardScaler` on the **pooled train matrix** (all devices), transform train+test. **Not** per-device / fully local / privacy-preserving preprocessing. New cache key includes `cid_test=1` (not `a9dd0c758635aebe`) |
| Test-device IDs | Cache **must** store `cid_test`. Audit: all 9 device ids appear in the pooled test split |
| Cap | `max_samples_per_file=20000` (unchanged; flattens some natural size imbalance — limitation, not un-capped here) |
| Mode | `match_notebook` |
| Optimizer | Adam, lr=1e-3 |
| K, β | 9 clients, trimmed_fraction=0.1 (β_eff=1/9) |
| Rounds / epochs / batch | 30 / 3 / 512 |

**Class coverage (must stay in the write-up):** Ennio doorbell and Samsung webcam **lack all Mirai classes** in this corpus. See `Q1_E2_DEVICE_CLASS_COVERAGE.md`. Re-count on the E2 train cache after it is built.

---

## Arms, seeds, metrics

**Arms (all four, every model×seed block):** FedAvg, UnweightedFedAvg, FedTrimmedAvg, WeightedFedTrimmedAvg.  
FedProx is **out of scope** for E2 (C5 already scoped on CIC).

**Models:** MLP and CNN.  
**Seeds:** {42, 123, 7}.  
**Metrics:** Final Acc, Final macro-F1, mean round-wise macro-F1 (**not** ROC-AUC).  
**F3:** descriptive only at n=3.

**2×2 (same directions as CIC confirmatory):**  
Trim added: **FT−UW**, **WT−FA** (help ⇒ both > 0).  
Weight added: **FA−UW**, **WT−FT** (hurt ⇒ both < 0).  
Never WT−UW as one factor.

**Scoring (unrounded stored Final F1; exact equality = tie = not supported):**  
For each **model**, mean over the three seeds.  
- **HE2-trim** supported iff mean(FT−UW)>0 and mean(WT−FA)>0.  
- **HE2-weight** supported iff mean(FA−UW)<0 and mean(WT−FT)<0.  
- **No HC1 / “FT first” claim** is pre-registered (C3: N-BaIoT gaps may be small). Report ranks without a pass/fail ranking hypothesis.  
- Report all paired seed Δs and per-contrast sign counts (3 seeds).  
- Wave-1 (MLP) results are **preliminary**; **final** descriptive assessment after Wave 2 (CNN).

---

## Waves (after acceptance)

Planning from prior N-BaIoT P3200 times (~40 min MLP, ~60–80 min CNN). **Estimates, not floors.**

| Wave | Cells | Role |
|------|------:|------|
| 0 | 1 | Cache `scale_on=train` + smoke **MLP FedAvg s42** |
| 1 | 12 | MLP × 4 arms × 3 seeds (includes Wave 0 cell) |
| 2 | 12 | CNN × 4 arms × 3 seeds |
| **Unique** | **24** | + cache build |

**Continuation:** Wave 2 proceeds after Wave 1 **regardless of signs**, except predefined integrity failure.

**Integrity / resume (runner `run_q1_e2_nbaiot.py`):**
- CUDA only; before this acceptance, `--list` was the only allowed command  
- `meta.scale_on == "train"`; cache key ≠ `a9dd0c758635aebe`  
- `cid_test` present **on the train-scaler NPZ only**; a `scale_on=full` rebuild must **not** add `cid_test` to the legacy v1 NPZ (same key `a9dd0c758635aebe`)  
- **9 non-empty**, disjoint train partitions covering all train rows  
- Pooled test contains **all 9** device ids  
- Provenance JSON records SHA-256 of the **processed** train-scaler **NPZ and meta.json**, plus library versions — **not** hashes of the original N-BaIoT source CSVs. This is **not** the CIC r5 source-tree file-hash set.  
- Those NPZ/meta hashes are stored on each raw row and checked on resume  
- If complete result rows exist and provenance JSON is **missing**, fail **closed** **before** cache load/build (do not rewrite, do not rebuild)  
- If provenance exists, NPZ/meta identity cannot change after a complete result (timestamp/version fields may refresh when hashes match)  
- `--require-from-cache` **must not build** a missing cache: missing NPZ/meta → exit 2; `load_or_build_nbaiot_cache(..., allow_build=False)` so the loader cannot write  
- **One process** is an **operator rule**, not an OS lock; the runner does not take a file lock  
- Integrity suite (`tools/test_q1_e2_integrity.py`) does not invoke live `--wave` / `--build-cache-only`; AST guard covers **direct** `subprocess.run([...])` list literals only (not Popen / `os.system` / assembled argv). Live cache **and** provenance are compared before/after, not required to be absent.

Do **not** start Wave 1 until Wave 0 integrity passes. Do **not** un-cap 20k or add FedProx/Median in this prereg.

### Platform (binding)

**Primary:** local **P3200**, python3.12, torch **2.5.1+cu121**, `match_notebook`.

**Colab / other GPU:** **not** the default. If used, the prereg must already be ACCEPTED **and** a **cross-platform pilot** must complete **before** mixing any Colab cell into E2 tables: N-BaIoT MLP FedAvg, device partition, seed 42, `scale_on=train`, 30 rounds. **Comparison rule:** pool with P3200 numbers **only** if Final F1 **and** all 30 history F1 **and** Acc are **bit-identical**. Otherwise treat Colab as a **separate lineage** (do not average, do not put in the same 3-seed mean). Accepted with **P3200** as primary. Do not later treat Colab as primary without that pilot rule.

---

## What this can and cannot show

**Can:** whether trim vs weight **signs** on N-BaIoT **device-based** clients look like the CIC Dirichlet confirmatory 2×2.  
**Cannot:** prove deployment, unseen-device generalization, Byzantine robustness, or that CIC HC1 “transfers.”

---

## Acceptance checklist

- [x] Proposer / approver / date  
- [x] Device≠Dirichlet; pooled test; not unseen-device  
- [x] `scale_on=train`; four arms; seeds {42,123,7}  
- [x] Class-coverage caveat (Ennio, Samsung)  
- [x] No HC1 ranking hypothesis  
- [x] No mix with `full` packets  
- [x] Nine non-empty partitions + `cid_test` audit in runner  
- [x] Pooled-train scaler disclosed (not private per-device)  
- [x] Colab not primary; bit-identical rule if used  
- [x] No GPU until ACCEPTED  
- [x] Integrity scoped: processed NPZ/meta hashes, not source CSVs; not “same as CIC r5”  
- [x] Missing provenance + complete rows fail closed before cache  
- [x] `--require-from-cache` cannot build (`allow_build=False`)  
- [x] One process = operator rule, not a lock  
- [x] `cid_test` only on `scale_on=train` NPZ (legacy full-scaler schema unchanged)  
- [x] Integrity suite isolated from live `--wave` / `--build-cache-only`; provenance compared before/after (reusable after ACCEPTED)  

---

*Contract amendment 2026-09-21 (Sheryar Kiani / Dr Hussain Kazmi): prereg ACCEPTED; no E2 cells yet. Wave 3 CIC is separate and optional.*

# Step 0 — Re-gate current engine (you run in PowerShell)

**Why:** All 36 E1b cells used `fl_local/engine.py` after the 2026-08-12 rewrite. Publication gates are older. Continuous lineage needs a fresh N-BaIoT MLP FedAvg α=1.0 s42 `match_notebook` PASS on **today’s** engine.

**You run all GPU commands.** Tools below only verify / build.

---

## A. Backup (if not already present)

```powershell
cd "C:\Elicit_FedTrim_Novelty_Audit\aoa-dear-i-have-commented-the\Research Paper"

Copy-Item "probe_results\probe_fast_match_notebook.json" `
  "probe_results\probe_fast_match_notebook_PRE_ENGINE_2026-08-02.json"
```

Do **not** skip this. The re-gate **overwrites** `probe_fast_match_notebook.json`.

---

## B. Run the gate (~38–110 min)

```powershell
python3.12 -u run_probe_fast.py --mode match_notebook
```

**Startup checks (first lines):**

- `device: cuda` (not cpu)
- cache `from_cache=True`
- key includes `a9dd0c758635aebe` if printed

Do **not** kill the job at 60 minutes (throttle can push ~110 min).

---

## C. Score with strong bar (not only exit code)

```powershell
python3.12 -u tools/verify_nbaiot_mlp_fedavg_gate.py
```

| Exit | Meaning |
|------|---------|
| **0** | **STRONG PASS** — endpoints exact + history bit-identical to published |
| **1** | FAIL endpoints / structure |
| **2** | Probe JSON missing |
| **3** | Endpoints OK but history not bit-identical (lineage weak) |

Runner exit 0 alone only means `|Δ F1| < 0.005` — **too loose** for lineage.

**Targets:** Final/Best F1 `0.8762425170224654`, Final/Best Acc `0.8956925949640226`, AUC `0.8579311208165772`.

**PASS scope:** shared path + `aggregate_fedavg_np` only (not trim/UW aggregators).

---

## D. After gate

| Result | Next |
|--------|------|
| Exit **0** | Optional packet (if not already built): `python3.12 -u _build_q1_e1b_three_primary_packet.py` then E1b amendment draft |
| Exit **1/3** | Diff new JSON vs `*_PRE_ENGINE_2026-08-02.json`; do not claim continuous lineage |

Campaign completeness anytime:

```powershell
python3.12 -u tools/status_report.py --arms e1 e1b
```

---

## Already on disk (no need to re-train E1b grid)

- E1 UW **36/36**, E1b WT **36/36** (`tools/status_report.py`)
- E1b packet builder: `_build_q1_e1b_three_primary_packet.py`
- Gate verifier: `tools/verify_nbaiot_mlp_fedavg_gate.py`

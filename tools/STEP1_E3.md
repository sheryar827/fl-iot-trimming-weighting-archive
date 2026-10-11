# Q1-E3 Step 1 — after G1–G3 PASS

**Prereq:** G1, G2, G3 all PASS (`tools/verify_q1_e3_gates.py`).  
**Prereg:** design r3 ACCEPTED 2026-08-15 (Sheryar Kiani / Dr Hussain Kazmi).

**Cost floor (updated after G1–G3 timings):** not 10×40 min.  
Using measured **~40 min** (µ=0-class: G1/G2) and **~64 min** (µ>0-class: G3), and treating SGD µ>0 as carrying similar prox overhead:

| Class | Cells | Nominal |
|-------|-------|---------|
| µ=0 | 1, 3, 5 | 3 × ~40 min |
| µ>0 | 2, 4, 6, 7, 8, 9, 10 | 7 × ~64 min |
| **Total** | **10** | **≈ 9.5 h floor** |

Thermal throttle can push higher. Old **6.7 h** = 10 × 40 min median (pre-G3).

---

## Commands (from `Research Paper/`)

```powershell
cd "C:\Elicit_FedTrim_Novelty_Audit\aoa-dear-i-have-commented-the\Research Paper"

python3.12 -u run_q1_e3_step1.py --list

# Recommended: run G4 early (cell 5) before other α=0.1 Adam cells
python3.12 -u run_q1_e3_step1.py --cell 5 --require-from-cache

# then remaining cells one-by-one, or:
python3.12 -u run_q1_e3_step1.py --all --require-from-cache
```

Startup must show: `device: cuda`, `from_cache=True`.

---

## Cell order

| # | α | opt | µ | Note |
|---|---|-----|---|------|
| 1 | 1.0 | sgd | 0 | H1 Mode A |
| 2 | 1.0 | sgd | 0.01 | H1 Mode A |
| 3 | 0.1 | sgd | 0 | H1 Mode B (Δ_SGD ok without G4; H1 floor needs G4) |
| 4 | 0.1 | sgd | 0.01 | same |
| 5 | 0.1 | adam | 0 | **G4 HARD** — Final F1/Acc + 30/30 vs published FA |
| 6 | 0.1 | adam | 0.01 | **G5 soft** published Prox cross-check |
| 7 | 0.1 | adam | 0.001 | dose (needs G4) |
| 8 | 0.1 | adam | 0.1 | dose (needs G4) |
| 9 | 0.1 | adam | 1.0 | dose (needs G4) |
| 10 | 1.0 | adam | 0.1 | H4a |

All runs: **FedProx** + `match_notebook` + seed **42**. SGD **momentum=0.9**.

### G4 failure blast radius (two effects — not “six cells uninterpretable”)

| Effect | Cells |
|--------|--------|
| **Invalidates** α=0.1 Adam dose–response (µ=0 anchor not lineage-matched) | **5–9** |
| **Removes H1 floor denominator** at α=0.1 (`Adam µ=0 = G4`); Δ_SGD itself still valid | **3–4** |

Practical recommendation: **run cell 5 early** — gates five cells outright and the H1 floor for two more.

**G5** fail: soft only; local dose–response still paired; `--all` continues.

---

## Outputs

- `q1_e3_results/csv/q1_e3_step1_raw.csv`
- `q1_e3_results/csv/history_e3s1_*.csv`
- `q1_e3_results/step1/cell_*.json`

Do **not** start Step 2 (CIC) until Step 1 is complete and scored against H1–H4.

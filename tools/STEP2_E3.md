# Q1-E3 Step 2 — after Step 1 complete

**Prereq:** G1–G3 PASS; Step 1 cells 1–10 on disk.  
**Amendment (H4c bar):** `q1_e3_results/reporting/Q1_E3_PREREG_AMENDMENT_STEP1_OUTCOMES_AND_H4C_BAR.md`  
Accept that amendment **before** treating Step 2 SGD H4c as pass/fail (bar is prospective).

---

## Cost

- Parent **~19 h** = 7 × ~165 min from **non-prox** CIC CNN runs → treat as **floor**.  
- Re-cost from **G6 (cell 1)** `Train_min` before locking the six SGD cells.  
- MLP 1.62× sketch (~26 h) is an **upper sketch only**, not a CNN prediction.

---

## Commands

```powershell
cd "C:\Elicit_FedTrim_Novelty_Audit\aoa-dear-i-have-commented-the\Research Paper"

python3.12 -u run_q1_e3_step2.py --list

# G6 first (HARD)
python3.12 -u run_q1_e3_step2.py --cell 1 --require-from-cache
# only if G6 PASS:

python3.12 -u run_q1_e3_step2.py --cell 2 --require-from-cache
# ... cells 3–7
```

Startup: `device: cuda`, `from_cache=True` (CIC cache).

---

## Cells

| # | seed | opt | µ | Role |
|---|------|-----|---|------|
| **1** | 42 | adam | 0.01 | **G6 HARD** + H2 local gap vs FA 0.6366122… |
| 2 | 42 | sgd | 0 | H1/H4c |
| 3 | 42 | sgd | 0.01 | H1/H4c |
| 4 | 123 | sgd | 0 | H1/H4c |
| 5 | 123 | sgd | 0.01 | H1/H4c |
| 6 | 7 | sgd | 0 | H1/H4c |
| 7 | 7 | sgd | 0.01 | H1/H4c |

### G6 targets (hard)

| Metric | Target |
|--------|--------|
| Final_F1 | 0.5671834167357673 |
| Final_Acc | 0.8903677314111789 |
| Best_F1 | 0.5713627996437929 |
| History | 30/30 F1+Acc vs published CIC CNN FedProx α=1.0 s42 |

### H4c Step 2 bar (after amendment acceptance)

**H1 floor (fixed per seed)** — published CIC CNN FedAvg α=1.0 Final_F1 as Adam µ=0 denominator (§6.1):

| Seed | FA Final_F1 | Floor (0.80×) |
|------|-------------|---------------|
| 42 | 0.636612200542919 | 0.5092897604343353 |
| 123 | 0.6041661178810206 | 0.4833328943048165 |
| 7 | 0.6547234479083687 | 0.523778758326695 |

Below floor → H1 **inconclusive** on that seed; seed **out of H4c aggregate**; raw table still reported.

Per floor-met seed, “tracks” if **all** of:
- max_r \|Δ F1\| ≤ **1.0 pp**
- \|Δ late-gain\| ≤ **1.0 pp**
- \|Δ r5\| ≤ **1.0 pp**

H4c **pass** = every floor-met seed tracks; **fail** = any floor-met seed fails a shape criterion; **inconclusive** = no seed meets floor. Floor-failed seeds are out of the aggregate (not track fails). Step 1 H4c stays **descriptive only**. Bar is data-informed; any threshold ~0.8–4.3 pp would have split Step 1 Mode A/B the same way.

---

## Outputs

- `q1_e3_results/csv/q1_e3_step2_raw.csv`
- `q1_e3_results/csv/history_e3s2_*.csv`
- `q1_e3_results/step2/cell_*.json`

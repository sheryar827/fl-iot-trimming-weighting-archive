# Q1-E3 scoring packet — Step 1 + Step 2 (descriptive; n=1 / n=3 as labelled)

**Status:** Reporting packet from on-disk runs. **Not** a C5 contract amendment.  
**Date (UTC):** 2026-08-19  
**F3 spirit:** no multiplicity-corrected “significant” claims.  
**H4c pass/fail** uses amendment draft §C (bar text dated 2026-08-16, **before** cell 2 `FinishedUTC` 2026-08-16T12:32:47Z). **Formal proposer/approver/date on that amendment is still empty** — H4c **fail** is therefore under a **pre-run draft bar**, disclosed as such, not a frozen contract clause.

**Number sources**
- Step 1: `q1_e3_results/step1/cell_*.json` + `q1_e3_results/csv/q1_e3_step1_raw.csv`
- Step 2: `q1_e3_results/step2/cell_*.json` + `q1_e3_results/csv/q1_e3_step2_raw.csv`
- Gates: `q1_e3_results/gates/`
- Local FA for H2: `q1_e1_results/csv/history_cic_CNN_FedAvg_a1.0_s42_match_notebook.csv`
- Amendment: `Q1_E3_PREREG_AMENDMENT_STEP1_OUTCOMES_AND_H4C_BAR.md`

Late-gain = max(F1 over rounds 6–30) − F1(round 5), F1 in **percentage points** unless a 0–1 fraction is shown.

---

## 1. Step 1 hypothesis board (N-BaIoT MLP s42)

| ID | α (prereg) | Outcome | Detail |
|----|------------|---------|--------|
| **H1** | 1.0 **and** 0.1 | **Rejected** | α=1.0 Δ = **+3.4661** pp (6.93× the 0.5 pp bar, **positive**) |
| **H2** | C5 CIC CNN (Step 2) | Tested in §2 | — |
| **H3** | **0.1 Adam** (cells 5–9) | **Strict-monotone fail** | Final rises µ=0.01→0.1 by **+0.4108** pp (73.5230→73.9338); then falls at µ=1.0 (**71.9076**). µ=0.1 is still **−9.4788 pp** vs µ=0 (83.4126→73.9338) — the uptick is **not** “µ=0.1 is good.” |
| **H4a** | **1.0 Adam** | **Pass** | r5: 86.14 → 84.06 → 77.16 (µ = 0, 0.01, 0.1) |
| **H4b** | **0.1 Adam** (Mode B) | **Partial by design of the claim** | Early↑/late↓ on µ ∈ {0, 0.001, 0.01}; **breaks** at µ ∈ {0.1, 1.0} |
| **H4c** | SGD both α | **Descriptive only** (no pass/fail) | Parent criterion has no numeric bar; cells already run. Step 2 H4c is §3. |

**Do not** label H3 “partial.” **Do not** label Step 1 H4c pass/fail/partial.

### Step 1 H1 (SGD)

| α | Floor met? | Δ Final (µ=0.01−µ=0) | vs 0.5 pp |
|---|------------|----------------------|-----------|
| 1.0 (cells 1–2) | yes (vs G1) | **+3.4661** pp | fail |
| 0.1 (cells 3–4) | yes (vs G4) | **+0.4330** pp | pass this α only |

Overall H1 **rejected**. For **this Step 1 (N-BaIoT MLP) result**, direction at α=1.0 is **help** — do **not** use the parent “µ too strong under SGD” branch **here**. That branch **does** apply to Step 2 / C5 SGD (§3).

### Step 1 H4c (descriptive; no verdict)

| α | max_r \|Δ F1\| (pp) | Δ late-gain (pp) | Δ r5 (pp) | Δ Final (pp) |
|---|---------------------|------------------|-----------|--------------|
| 1.0 | **4.81** | **+4.34** | −1.11 | **+3.47** |
| 0.1 | **0.78** | **−0.29** | −0.21 | **+0.43** |

---

## 2. Step 2 — G6 / H2 (CIC CNN α=1.0 s42 Adam)

| Metric | Value |
|--------|--------|
| Local Prox Final_F1 | **0.5671834167357673** (G6) |
| vs published CIC CNN FedProx α=1.0 s42 | **30/30 F1 and 30/30 Acc bit-identical** (max \|diff\| = 0). Parent G6 listed scalars; this run also cleared the **stricter** full-history bar. |
| Local FA Final_F1 | **0.636612200542919** (2026-08-07 history; 30/30 vs published) |
| **H2 gap s42** | **−6.942878 pp** (entirely local) |

The **0.636612200542919** used as the s42 H1-floor FA in §3 (published row) and as the H2 local FA here are **the same bit pattern** — local FA history is 30/30 F1+Acc identical to published FedAvg. Two provenances, one constant.

**Scope: s42 only** — not the 3-seed C5 mean (~6.7 pp). **H2 pass** at that scope.

G6 Train_min **249.51**. Config: `match_notebook`, cuda, cache `b725017696446067`, adam, µ=0.01.

---

## 3. Step 2 — C5 SGD pairs (full four-column table)

H1 floor = 0.80 × **published** CIC CNN FedAvg α=1.0 Final_F1 for that seed (amendment §C / parent §6.1). **Not** an Adam-vs-SGD fairness comparison of absolute F1.

| Seed | Floor (0.80×FA) | SGD µ=0 Final | vs floor | Δ Final (pp) | max_r \|ΔF1\| (pp) | Δ late-gain (pp) | Δ r5 (pp) | H1 | H4c (draft §C) |
|------|-----------------|---------------|----------|--------------|-------------------|------------------|-----------|----|----------------|
| **42** | 0.5092897604343353 | 0.5658105465928371 | **+5.6521 pp met** | **−0.7187** | **10.6266** (round 2) | **−2.3283** | **+1.6255** | **fail** (floor met) | **fail 3/3** |
| **123** | 0.4833328943048165 | 0.5462197624389744 | **+6.2887 pp met** | **−4.3231** | **6.3920** (round 1) | **−3.3242** | **−0.1159** (≤1.0) | **fail** (floor met) | **fail 2/3** |
| **7** | 0.5237787583266950 | 0.5215916313838613 | **−0.2187 pp miss** | **−1.7791** | **16.4204** (round 12) | **+3.0482** | **−3.9912** | **inconclusive** | **out of aggregate** |

s7 SGD µ=0 falls **0.219 pp** below its 80% floor, so **H1 is unscoreable on this seed** (floor test, not an Adam/SGD level claim).

### Aggregates (draft §C)

| Rule | Result |
|------|--------|
| **H1** (C5 SGD) | **Rejected** — two floor-met seeds fail \|Δ\|&lt;0.5 pp; s7 excluded |
| **H4c Step 2** | **Fail** (provisional on draft bar) — both floor-met seeds fail the 1.0 pp track criteria |
| Parent §7 **¬H1** branch (µ hurts under SGD) | **Applies to Step 2 / C5**, not to Step 1 |

---

## 4. Mechanism (keep scopes separate)

**Step 1 (N-BaIoT MLP, SGD)** — µ **helps** Final at both α (+3.47 / +0.43 pp). **Do not generalize** “under SGD, µ helps” beyond that dataset/model.

**Step 2 (CIC CNN, SGD)** — µ **hurts** Final on all three seeds (−0.72 / −4.32 / −1.78 pp). This **is** the parent §7 branch: floor met (s42, s123), µ hurts without Adam → **µ=0.01 too strong even without Adam** on this C5 SGD grid; C5’s existing scope (Adam µ=0.01 damage) is **unchanged** by SGD inertness (inertness is rejected).

**s42 only** (the only seed with a **local** Adam pair **and** local SGD pair):

> At C5 s42, µ=0.01 costs **−6.94 pp** under Adam (H2) and **−0.72 pp** under SGD — **same sign**, **9.66×** larger under Adam (6.942878 / 0.718696). Adam **amplifies** the proximal penalty; it does **not** create it. H1’s inertness prediction is **rejected**.

Both figures are paired local Δ(µ), not cross-optimizer absolute F1.

---

## 5. Timing

| Quantity | Value |
|----------|--------|
| Step 2 **raw train sum** (all 7 `Train_min`) | **1433.578 min = 23.893 h** |
| Elapsed cell 1 **finish** → cell 7 **finish** | **31.03 h** (`2026-08-16T09:29:04Z` → `2026-08-17T16:30:53Z`) |
| In that window | cells 2–7 train **19.73 h** + **11.30 h** idle (does **not** include cell 1’s 249.51 min) |
| Elapsed inferred cell 1 **train start** → cell 7 **finish** | **35.19 h** (`2026-08-16T05:19:33.58Z` → `2026-08-17T16:30:53Z`) = 23.89 h train + **11.30 h** idle |
| SGD µ=0.01 / µ=0 Train_min | 1.383× / 1.377× / 1.415× (s42 / s123 / s7) → **mean 1.392×** |

Inferred train start = cell 1 `FinishedUTC` **minus** `train_s` (**14970.683** s) → **2026-08-16T05:19:33.58Z**. That is **not** process start (excludes cache load and setup). Do not print `05:19:29Z`.

Use **1.39×** for future **SGD** prox costing on this CNN+P3200 path. Do **not** read 23.89 h as elapsed wall.

Step 1 µ>0 Adam cells 8–10 Train_min: **63.72 / 61.56 / 61.60** → **61.6–63.7 min**.

---

## 6. Config audit (Step 2, all 7)

From cell JSON: 30 rounds, `mode=match_notebook`, `device` cuda, optimizer/µ/seed/α as designed, SGD `sgd_momentum=0.9` when SGD. Cache key on raw rows: **b725017696446067**. `FinishedUTC` strictly increasing across cells 1→7.

**Not in packet:** client partition size lists. They were printed to stdout only for cells 1–7; **not** stored in those JSON/CSV files. After this campaign, `run_q1_e3_step2.py` was edited to put `partition_sizes` on the **report dict only** (same `len(part[c])` list already printed — **no RNG, no training-path change**). Cells 1–7 on disk have **no** `partition_sizes` key, which confirms the edit is **post-campaign / future runs only**.

---

## 7. What this packet does **not** do

- Amend C5 wording  
- Call Step 2 H4c fail **contract-frozen** until amendment acceptance  
- Treat Step 1 SGD sign reversal as true on CIC  
- Compare Adam vs SGD **absolute** F1 as a fair baseline  

---

*End of scoring packet.*

# Q1-E3 prereg amendment — Step 1 outcomes + prospective H4c bar (Step 2 only)

**Status:** **DRAFT — formal who/what/why acceptance still required**  
**Date (UTC draft):** 2026-08-16  
**Parent:** `Q1_E3_PREREG_DRAFT.md` (r3 ACCEPTED 2026-08-15)  
**Timing vs SGD cells:** this file’s H4c bar text existed **before** Step 2 cell 2 finished (`FinishedUTC` **2026-08-16T12:32:47Z**). Do **not** back-date acceptance unless that is truthful. Packet scoring that uses §C must **disclose** draft-bar vs accepted-bar.  
**Why:** Step 1 complete on disk; H4c needs a **numeric** bar frozen in writing **before** Step 2 SGD cells are scored as pass/fail.

This amendment does **not** re-open E1b or C5 claim text. It freezes **how Step 1 is reported** and **how Step 2 H4c is scored**.

---

## Who (fill at acceptance)

- Proposer: _[REQUIRED]_  
- Approver: _[REQUIRED]_  
- Acceptance date: _[REQUIRED]_  

---

## What

### A. Step 1 H1 scoring language (binding for packet / write-up)

**Replace** soft “H1 not confirmed” phrasing with:

| α | Floor | Δ_SGD Final (µ=0.01 − µ=0) | H1 bar \|Δ\| &lt; 0.5 pp | Verdict |
|---|-------|----------------------------|---------------------------|---------|
| 1.0 (cells 1–2) | met (vs G1 Adam µ=0) | **+3.4661** pp | fail (6.9× bar) | **¬H1 (rejected)** |
| 0.1 (cells 3–4) | met (vs G4 Adam µ=0) | **+0.4330** pp | pass | pass at this α only |

**Overall H1 (requires both α):** **rejected**.  
Direction at α=1.0 is **positive** (µ helps under SGD), not the parent §7 branch “µ hurts under SGD.” Record as:

> **¬H1 (Step 1 only, N-BaIoT MLP s42):** floor met; \|Δ\| not &lt; 0.5 pp at α=1.0; µ **improves** Final under SGD near-IID (and is small-positive at α=0.1). Do **not** use the parent “µ too strong under SGD” outcome branch for **this Step 1 result**. Step 2 / C5 SGD is a separate result (µ **hurts** there; that **is** the parent branch).

### B. Step 1 H4c — **descriptive only** (no pass/fail verdict)

Parent H4c has **no numeric threshold**. Step 1 SGD cells are already on disk, so **no retrospective bar**.

**Report only (Step 1, N-BaIoT MLP s42):**

| α | max_r \|Δ F1\| (pp) | Δ late-gain (pp) | Δ r5 (pp) | Δ Final (pp) |
|---|---------------------|------------------|-----------|--------------|
| 1.0 (cells 1–2) | **4.81** | **+4.34** | −1.11 | **+3.47** |
| 0.1 (cells 3–4) | **0.78** | **−0.29** | −0.21 | **+0.43** |

**Forbidden:** labeling Step 1 H4c “partial,” “pass,” or “fail.”

### C. Prospective H4c bar — **Step 2 SGD cells only** (CIC CNN α=1.0)

Applies only to Step 2 pairs (seed ∈ {42, 123, 7}): SGD µ=0 vs SGD µ=0.01, **same seed**, after **G6 PASS**.

**H1 floor (fixed, not optional):** applies **per seed** using the **published** CIC CNN FedAvg α=1.0 Final_F1 for that seed as the Adam µ=0 denominator (parent §6.1). Step 2 runs **no** local Adam FedAvg cells, so published FA is the only allowed reference:

| Seed | Published FA Final_F1 (denominator) | Floor = 0.80 × FA |
|------|--------------------------------------|-------------------|
| 42 | 0.636612200542919 | 0.5092897604343353 |
| 123 | 0.6041661178810206 | 0.4833328943048165 |
| 7 | 0.6547234479083687 | 0.523778758326695 |

A seed whose SGD µ=0 Final_F1 is **below** its floor is reported **inconclusive for H1** on that seed and is **excluded from the H4c aggregate**; the raw shape table is still reported for every seed.

**H4c “tracks” definition (all three must hold for a floor-met seed):**

1. **max_r** \|F1_r(µ=0.01) − F1_r(µ=0)\| ≤ **1.0 pp** over rounds 1…30  
2. **\|Δ late-gain\|** ≤ **1.0 pp** (late-gain per parent §2.1)  
3. **\|Δ r5 F1\|** ≤ **1.0 pp**

| Aggregate | Rule |
|-----------|------|
| H4c Step 2 **pass** | Every **floor-met** seed satisfies (1)–(3). (Floor-failed seeds are out of the aggregate, not counted as track fails.) |
| H4c Step 2 **fail** | At least one floor-met seed fails any of (1)–(3) |
| H4c Step 2 **inconclusive** | **No** seed meets the H1 floor (aggregate empty) |
| Per-seed table | Always report raw max_r, Δ late-gain, Δ r5, Δ Final, floor status |

**Rationale (data-informed, disclosed):** Step 1 Mode B SGD sat under 1.0 pp on all three shape gaps; Mode A SGD did not. The bar freezes that distinction **before** CIC SGD runs. The Step 1 separation does **not** hinge on choosing exactly 1.0: α=0.1’s largest gap is max_r **0.78** pp, while α=1.0 fails max_r by **4.8×** (4.81/1.0) and Δ late-gain by **4.3×** (4.34/1.0); \|Δ r5\| at α=1.0 is only 1.11 pp. **Any bar in roughly 0.8–4.3 pp** would yield the same Step 1 separation. Stating this pre-empts reverse-engineering objections; the value **1.0** is a convenient round threshold inside that robust range.

**Does not apply** to Step 1 N-BaIoT SGD (already run — descriptive only).

### D. Step 1 H3 / H4a / H4b summary (descriptive; for packet)

| ID | Step 1 outcome |
|----|----------------|
| **H3** | **Strict monotone fail (α=0.1 Adam, cells 5–9):** Final rises µ=0.01→0.1 by +0.41 pp (73.52→73.93); then falls at µ=1.0. The µ=0.1 point is still **−9.48 pp** vs µ=0 (83.41→73.93) — the uptick is **not** “µ=0.1 is good.” |
| **H4a** | **Pass** on {0, 0.01, 0.1} at α=1.0 Adam: r5 86.14 → 84.06 → 77.16 |
| **H4b** | **Partial by design of the claim:** classic early↑/late↓ holds on µ ∈ {0, 0.001, 0.01}; **breaks** at µ∈{0.1, 1.0} (r5 falls, late-gain rises). |

### E. Mechanism lines (packet-ready; **keep dataset scopes separate**)

**Step 1 only (N-BaIoT MLP s42):**

> Under Adam, µ **costs** Final F1 at both skews examined (α=1.0: −0.74 pp at µ=0.01; α=0.1: −9.89 pp at µ=0.01). Under **SGD on N-BaIoT MLP**, µ **helps** Final at both (+3.47 pp at α=1.0; +0.43 pp at α=0.1). That **sign reversal is N-BaIoT MLP only**. Separately, under Adam at severe skew the effect is **non-monotone**: µ≈0.01 buys early F1 and costs Final; larger µ suppresses early F1 too.

**Step 2 / C5 (CIC CNN):** SGD µ **hurts** Final on all three seeds (−0.72 / −4.32 / −1.78 pp). That **is** the parent §7 ¬H1 branch (floor met on s42 and s123). **Do not** copy the Step 1 “SGD helps / sign reversal” sentence onto CIC.

**s42 local pairs (only seed with local Adam and local SGD Δ):**

> At C5 s42, µ=0.01 costs **−6.94 pp** under Adam and **−0.72 pp** under SGD — same sign, **9.66×** larger under Adam. Adam **amplifies** the proximal penalty; it does **not** create it. H1’s inertness prediction is **rejected**.

**Do not use (unscoped):** “Optimizer-dependence is shown by sign reversal” as if it held on both datasets.

### F. Step 2 cost language

Parent ~19.25 h uses ~165–167 min/cell from **non-prox** CIC CNN E1b-class runs. Four of seven Step 2 cells have µ&gt;0. Treat **~19 h as a floor**; **re-cost after G6** wall time before committing the six SGD cells. An MLP-ratio upper sketch (~1.62×) is **not** a prediction for CNN.

### G. Timing note (Step 1 µ&gt;0 Adam cells 8–10)

Train_min: **63.72 / 61.56 / 61.60** → report **61.6–63.7 min**, not “~62–64” alone.

---

## Why

- Peer review of Step 1 write-up: H4c had no bar; “partial” was a post-hoc verdict.  
- H1 was refuted, not merely unsupported.  
- Mechanism one-liner understated sign-reversal evidence.  
- Step 2 H4c needs a freeze **before** CIC SGD GPU hours.

---

## Acceptance checklist

- [ ] Proposer / approver / date  
- [ ] H1 = **rejected** (not “not confirmed”)  
- [ ] Step 1 H4c = **descriptive only**  
- [ ] Step 2 H4c bar = 1.0 pp on max_r, \|Δ late-gain\|, \|Δ r5\|  
- [ ] Mechanism lines **scoped**: sign-reversal = Step 1 N-BaIoT MLP only; CIC CNN SGD µ hurts; s42 = Adam amplifies, does not create  

- [ ] Step 2 cost re-baselined after G6  

---

*End of amendment draft. Not a C5 contract freeze.*

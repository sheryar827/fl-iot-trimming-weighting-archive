# Q1-E3 prereg draft — FedProx µ × optimizer (Adam vs SGD)

**Status:** **ACCEPTED 2026-08-15** — proposer **Sheryar Kiani**; approver **Dr Hussain Kazmi**; form **Accept E3 prereg r3**  
**Date (UTC draft):** 2026-08-15  
**Rev:** r3 (+ tooling r3++ pre-G1 fixes; design unchanged)  

**Depends on:** E1b freeze complete; shared-path re-gate STRONG PASS 2026-08-15 (FedAvg path only)  
**Contract anchors:** **C5** (live claim); **§5 Q1-E3** **Done (prereg cells)** as of **2026-09-12** (see `Q1_E3_CONTRACT_AMENDMENT_DRAFT.md` APPLIED). **F3 spirit** (contract F3 text is FedTrim-specific: forbids claiming “statistically significant FedTrim wins after multiplicity correction at n=3”; this prereg applies the **same spirit** — no multiplicity-corrected “significant” language on E3 contrasts at n=1 or n=3)

This document freezes the **experimental design and hypotheses**. It is **not** a C5 claim freeze. Outcomes that change C5 wording require a later dated contract amendment.

### Acceptance record

- **Proposer:** Sheryar Kiani  
- **Approver:** Dr Hussain Kazmi  
- **Date:** 2026-08-15 (15-8-2026)  
- **Statement:** Accept E3 prereg r3  
- **Next:** G1 → G2 → G3 only until all three PASS (`tools/STEP0_E3_GATES.md`)

### Pre-G1 backup (mandatory after engine edit, before G1)

Engine edit invalidates `probe_results/probe_fast_match_notebook.json` (same class of event as the 2026-08-12 rewrite). **Before G1:**

```powershell
Copy-Item "probe_results\probe_fast_match_notebook.json" `
          "probe_results\probe_fast_match_notebook_PRE_E3ENGINE_2026-08-15.json"
```

The 2026-08-15 STRONG PASS JSON becomes the pre-E3-edit reference if G1 misses.

---

## 1. Why E3 first (procedural + scientific)

1. **Procedural:** C5 is a frozen claim. Q1-E3 is the only open E* the contract marks **Required for C5**. E2 / E4 / E7 support claims not yet made.  
2. **Scientific:** FedProx (µ=0.01, Adam) sits far from FedAvg on multiple cells; one untuned scalar spans a large **cross-cell** Prox−FedAvg range (about **−9.98 to +5.92** pp Final F1 on the 12 non-IID model×α means — a **15.9 pp range of µ’s effect across cells**, not a swing inside one cell).  
3. **Mechanism question:** Is the deficit an **Adam × µ** interaction, simply **µ too large**, or a **lineage / path** problem? E3 is designed to distinguish those branches.

**Not claimed as rationale:** “FedProx has the smallest seed SD” — **retracted** (grid-wide false).

---

## 2. Zero-cost pre-check (published histories only)

### 2.1 Definitions

| Term | Definition |
|------|------------|
| **ahead@r5** | FedProx F1(round 5) − FedAvg F1(round 5) (same model, α, seed) |
| **below@r30** | FedProx F1(round 30) &lt; FedAvg F1(round 30) |
| **late-gain** | max(F1 over rounds **6–30**) − F1(round **5**) |
| **PIN** | ahead@r5 &gt; 0 **and** FedProx late-gain &lt; FedAvg late-gain (and usually below@r30) |

### 2.2 Two failure modes (not one “pinning” story)

**Mode A — near-IID / mild skew (α = 1.0; both datasets):**  
FedProx is **behind by round 5 in 12/12** cells and **still below at round 30 in 10/12**.  
**Exceptions (late recovery, do not fold into uniform Mode A):**

| Cell | ahead@r5 | Final gap (Prox − FA) | Note |
|------|----------|------------------------|------|
| N-BaIoT CNN α=1.0 s123 | behind | **+0.18** pp | recovers to slightly above FA |
| N-BaIoT CNN α=1.0 s7 | behind | **+8.16** pp | large late recovery |

N-BaIoT CNN is exactly where **C8**’s seven late-run rescues live. s7’s Mode-A-then-recover pattern must be **named**, not averaged away.

**Mode B — severe skew (α = 0.1; primarily N-BaIoT):**  
In **4/6** N-BaIoT α=0.1 cells, FedProx is **ahead at r5**, has **lower late-gain**, and finishes **below** FedAvg — proximal term helping early, then capping late progress.

**Robust endpoint (N-BaIoT):** FedProx ends below FedAvg in **16/18** non-IID cells.

### 2.3 CIC (do not over-generalize Mode B)

| α | ahead@r5 | PIN | below@r30 |
|---|----------|-----|-----------|
| 1.0 | 0/6 | 0/6 | **6/6** |
| 0.5 | 2/6 | 2/6 | 4/6 |
| 0.1 | 2/6 | 1/6 | 2/6 |

- **C5 cell (CIC CNN α=1.0):** **3/3 Mode A** (behind@r5, below@r30); Final gaps ≈ **−6.94 / −4.14 / −9.02** (mean gap matches C5 ~**6.7** pp).  
- **CIC MLP α=0.1:** FedProx **wins** Final **3/3** (+7.35 / +7.02 / +3.41; mean **+5.93** ≈ grid **+5.92**). Mode B is **not** the CIC story at α=0.1.  
- Combined published non-IID (both datasets): roughly **9/36** ahead@r5, **8/36** PIN, **28/36** below@r30 (descriptive; not a test family).

### 2.4 What was retracted from earlier drafts

| Claim | Status |
|-------|--------|
| FedProx “converges faster early then saturates” as **general** pattern | **False** (only a minority of N-BaIoT cells) |
| Smaller max−min F1 span ⇒ pinning | **Weak** (Prox higher r1 in 14/18 N-BaIoT; span confounded) |
| Mechanism sweep only on α=1.0 | **Inadequate** (tests Mode A only) |
| FedProx “smallest seed SD of any arm” | **Retracted** |
| Gate 2 alone proves µ&gt;0 draws no RNG | **Overstated** — see §4 |
| H2 via published-only C5 reuse (no local CIC FedProx) | **Rejected for core design** — see §5 / H2 |
| α=0.1 Adam dose–response with published µ=0 / µ=0.01 anchors only | **Rejected** — see §5 Step 1 anchors |

### 2.5 Seed-SD context for n=1 mechanism cells

Across published non-IID Final_F1 seed SDs (algorithms × models × α × datasets):  
**min ≈ 0.02 · median ≈ 2.49 · max ≈ 11.82 pp**.

Unpaired single-seed dose–response is **indefensible** as a population claim. The design uses **paired within-seed, within-optimizer Δ(µ)** so the seed is held fixed; still **descriptive only** (**F3 spirit**). Do **not** cite “1.2–2.6 pp” as a grid-wide seed-SD band (that band was the **four-arm range at two CIC α=1.0 cells** in C1 wording only).

---

## 3. Engine / runner changes (before any science run)

**Current state (verified this build):**

- Proximal term present: `(µ/2)·Σ‖w−wᵗ‖²` with `clone().detach()` refs after loading global params; zip order matches `model.parameters()`.  
- `mu = cfg["fedprox_mu"] if algorithm == "FedProx" else 0.0` (match_notebook and fast).  
- At **µ=0**, `global_param_tensors = None` / prox branch skipped.  
- **`optim.Adam` hardcoded** at match_notebook client train and fast path; **no runner** exposes `--mu` or `--optimizer`.  
- **Zero** local FedProx rows in E1/E1b raw CSVs; **no FedProx cell has ever been gate-reproduced** locally (all gates to date were FedAvg).  
- Prior FedAvg local probes cover N-BaIoT/CIC × MLP/CNN in various forms; **FedProx gate matrix is 0/4** dataset×model combinations until G3 and G4 (below).

**Required code (minimal):**

1. Config / CLI: `fedprox_mu` (float), `optimizer ∈ {adam, sgd}` (default **adam**, default µ **0.01** for FedProx — preserve published semantics when flags omitted).  
2. Both train paths use the selected optimizer at **lr = 1e-3** (same as Adam / published).  
3. **SGD hyperparameters (frozen):**  
   `torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.0)`  
   - **momentum=0.9** is intentional: plain `momentum=0` at Adam-tuned lr is the slowest choice and makes the H1 undertraining floor more likely.  
   - This is **not** a claim that momentum-SGD is “the fair FedProx paper setting”; it is the pre-registered local SGD baseline for testing **Δ(µ) under a non-adaptive optimizer**.  
4. Existing FedAvg / E1 / E1b runners **unchanged in default behavior**.  
5. Dedicated runner(s) or flags for E3 cells under `q1_e3_results/`.

**No mid-program torch upgrade.** Pins remain torch **2.5.1+cu121**, python3.12, P3200, `match_notebook` only for table-adjacent work.

---

## 4. Gates (Step 0) — mandatory before science; stop rules

### 4.1 Core lineage gates (~2.0 h)

| Gate | Cell | Pass criterion |
|------|------|----------------|
| **G1 — FedAvg re-gate** | N-BaIoT MLP FedAvg α=1.0 s42, **Adam** | Final_F1 **0.8762425170224654**, Final_Acc **0.8956925949640226** exact; **30/30 F1 and Acc** bit-identical to published history (and PRE_E3ENGINE probe if present); recorded cfg: `optimizer=adam`, `fedprox_mu=0.0` (unused), `mode=match_notebook` |
| **G2 — µ=0 harness** | N-BaIoT MLP **FedProx** µ=**0** α=1.0 s42, **Adam** | **30/30 F1 and Acc** bit-identical to **G1**; Final_F1 and Final_Acc equal G1; recorded cfg: `algorithm=FedProx`, `optimizer=adam`, `fedprox_mu=0.0` |
| **G3 — first FedProx gate** | N-BaIoT MLP FedProx µ=**0.01** α=1.0 s42, **Adam** | Final_F1 **0.8688510782827605**; Final_Acc **0.8888074036256555**; Best_F1 **0.8747881297547481**; **and 30/30 F1 and Acc bit-identical to published FedProx history** (same bar as G1 — not scalars-only); recorded cfg: `optimizer=adam`, `fedprox_mu=0.01` |

### 4.2 Extended lineage (run with Step 1 / Step 2; G4 and G6 are hard stops, G5 is a published cross-check)

| Gate | Cell | Pass criterion | Role |
|------|------|----------------|------|
| **G4 — α=0.1 FedAvg (Adam µ=0)** | N-BaIoT MLP FedAvg / FedProx µ=0 α=**0.1** s42, Adam | Final_F1 **0.8341257368792174**; Final_Acc **0.8593683450437059** | Extends shared-path lineage to α=0.1; serves as **local paired µ=0 anchor** for Mode B dose–response |
| **G5 — α=0.1 FedProx µ=0.01** | N-BaIoT MLP FedProx µ=**0.01** α=**0.1** s42, Adam | Final_F1 **0.735229971381635** (published; Acc/Best from same raw row if used in packet) | Local paired µ=0.01 anchor at Mode B; soft check vs published |
| **G6 — C5 FedProx gate** | CIC CNN FedProx µ=**0.01** α=1.0 s42, Adam | Final_F1 **0.5671834167357673**; Final_Acc **0.8903677314111789**; Best_F1 **0.5713627996437929** | **H2** — local Prox; with local FA (2026-08-07) gives **s42** gap **−6.94 pp** (not 3-seed mean) |

G4–G5 are listed as gates because their targets are published bit-level Final metrics; they also count as **Step 1 cells 5–6** (do not double-count cost).

### 4.3 What G2 licenses (two-part — do not merge)

1. **Empirical (G2):** FedProx with µ=0 bit-identical to FedAvg proves the `algorithm == "FedProx"` branch introduces **no extra divergence when µ is off**.  
2. **Analytic (code inspection, µ&gt;0):** the only additions under µ&gt;0 are `[p.clone().detach() for p in model.parameters()]` and an **added loss term** — **neither draws from any RNG stream**.

**Together** they license **paired n=1** within-seed comparisons for the mechanism sweep. **G2 alone does not** prove the µ&gt;0 path is RNG-neutral.

### 4.4 Stopping rules

| Failure | Action |
|---------|--------|
| **G1 fails** | **Stop.** Shared path / engine regression. No E3 science. |
| **G2 fails** | **Stop.** Paired n=1 justification for FedProx branch collapses; Step 1 Δ readout is invalid for a different reason than G3. Debug µ=0 path before any sweep. |
| **G3 fails** | **Stop.** First FedProx lineage fail (µ&gt;0). Treat as path problem, not science. |
| **G4 fails** | **Stop** treating α=0.1 **Adam** dose–response (Step 1 cells **5–9**) as lineage-matched; do not mix published FA at α=0.1 with local µ&gt;0 as a “paired” curve. **Does not** invalidate SGD Δ(µ) on cells **3–4** (internally paired), but **does** remove the §6.1 H1 floor denominator at α=0.1 (Adam µ=0 = G4), so H1 at α=0.1 is **unscoreable** until G4 passes. |
| **G5 fails** (local µ=0.01 ≠ published within tight tolerance) | **Not a stop.** Cells 5–9 remain local and internally paired — report **local-only** dose–response; treat published α=0.1 Prox as a failed cross-check; investigate when convenient. Prefer exact match if numerically feasible. |
| **G6 fails** | **H2 fails** (lineage on C5 s42). Do **not** claim local reproduction of the s42 C5 gap; debug before interpreting SGD C5 cells as confirming Adam×µ on CIC. |

Order: **G1 → G2 → G3** first (~2 h). Then Step 1 including **G4/G5**. Then Step 2 including **G6** before or as the Adam C5 cell.

---

## 5. Experimental steps

### Step 1 — Mechanism sweep (both modes) — **10 new cells ≈ 9.5 h floor**

**Fixed:** dataset **N-BaIoT**, model **MLP**, seed **42**, `match_notebook`, lr=1e-3, 30 rounds.  
**Cost (updated after G1–G3):** not uniform 40 min. G1/G2 ≈ **39–40 min** (µ=0 path); G3 ≈ **64 min** (µ=0.01 Adam). Projecting µ=0-class cells (1,3,5) @ ~40 min and µ>0-class (2,4,6–10) @ ~64 min (assuming SGD µ>0 carries similar prox overhead) → **≈ 9.5 h floor**. Older **6.7 h** = 10 × 40 min median only.

| # | α | Optimizer | µ | Purpose |
|---|---|-----------|---|---------|
| 1 | 1.0 | SGD (mom=0.9) | 0 | H1 / H4c Mode A baseline |
| 2 | 1.0 | SGD (mom=0.9) | 0.01 | H1 / H4c Mode A |
| 3 | 0.1 | SGD (mom=0.9) | 0 | H1 / H4c Mode B baseline |
| 4 | 0.1 | SGD (mom=0.9) | 0.01 | H1 / H4c Mode B |
| 5 | 0.1 | Adam | **0** | **G4** + local µ=0 anchor for H3/H4b (**paired**) |
| 6 | 0.1 | Adam | **0.01** | **G5** + local µ=0.01 anchor for H3/H4b (**paired**) |
| 7–9 | 0.1 | Adam | 0.001, 0.1, 1.0 | H3 / H4b dose–response (all local, same seed) |
| 10 | 1.0 | Adam | 0.1 | H4a cheap monotonicity check in Mode A |

**Reused (not re-run if gates pass):** Adam µ=0 at α=1.0 ≡ G1; Adam µ=0.01 at α=1.0 ≡ G3.

**Why cells 5–6 are required (not optional):**  
Without them, the α=0.1 Adam curve would mix **local** µ∈{0.001,0.1,1.0} with **published / ungated** µ=0 and µ=0.01 — so Δ(µ) vs µ=0 would **not** be paired or RNG-matched, contradicting §6. G4 also extends FedAvg lineage to α=0.1 “for free.”

### Step 2 — C5 confirm + H2 — **7 cells ≈ 19.25 h**

| Cells | Purpose |
|-------|---------|
| CIC CNN α=1.0 s42 Adam FedProx µ=0.01 (**G6**, 1 run) | **H2** — local Prox on C5 s42 |
| CIC CNN α=1.0 × seeds {42, 123, 7} × SGD µ ∈ {0, 0.01} (6 runs) | H1/H4c at C5 Mode A under SGD |

Nominal **~165 min**/CIC CNN cell; G6 ≈ **+2.75 h**. Thermal floor: treat totals as **floors** (observed up to ~2.9×).

**Local FedAvg already on disk for H2:** CIC CNN FedAvg α=1.0 s42 passed **2026-08-07** (bit-exact over 30 rounds); history `q1_e1_results/csv/history_cic_CNN_FedAvg_a1.0_s42_match_notebook.csv`, Final_F1 **0.636612200542919**. No re-run of local Adam FedAvg required for H2 if that file remains valid. If G6 hits target **0.5671834167357673**, the **s42** gap is **−6.94 pp** entirely from local runs. C5’s ~**6.7** pp claim is a **3-seed mean** — not re-proven by s42 alone.

### Optional Step 3 — Mode C (CIC MLP α=0.1) — **not in core**

FedProx **wins** Final here (+5.93 mean). If run: e.g. SGD µ∈{0,0.01}.  
**Cost:** CIC MLP Train_s (18 local match_notebook): min **118.7**, median **122.7**, mean **144.1**, max **267.7**. Two cells ≈ **+4–5 h**.

**Default:** omit from first freeze.

---

## 6. Confounds and readout rules (write before results)

1. **lr=1e-3 is Adam-tuned.** SGD absolute F1 will lag Adam; that is expected.  
2. **SGD momentum=0.9** is frozen (§3); do not retune after seeing results.  
3. **Primary readout:** within-optimizer  
   **Δ(µ) = metric(µ) − metric(µ=0)**  
   for Final F1, r5 F1, and late-gain. **Never** Adam absolute level vs SGD absolute level as a fair comparison.  
4. **Paired n=1** on mechanism cells: same seed, same α, same optimizer, **local** µ=0 and µ&gt;0 anchors only. Seed-SD context §2.5. **Descriptive only (F3 spirit).**  
5. **Shape metrics** always use §2.1 definitions.  
6. **C8:** do not fold E3 runs into C8 denominators without a separate amendment. Name N-BaIoT CNN late-recovery (s7) when discussing Mode A exceptions.

### 6.1 H1 interpretability floor (**blocking — pre-registered**)

lr=1e-3 with a non-adaptive optimizer can **fail to train**. If SGD µ=0 and SGD µ=0.01 both sit near a majority-class floor, then Δ≈0 and H1 would “pass” while testing **nothing**. That is the most likely false-confirmation mode of this design.

**Rule (fixed before results):** for each cell where H1 is evaluated,

> H1 is **interpretable** only if  
> **Final_F1(SGD, µ=0) ≥ 0.80 × Final_F1(Adam, µ=0)**  
> on the **same** dataset × model × α × seed.  
> - Adam µ=0 at α=1.0 = G1 / FedAvg gate.  
> - Adam µ=0 at α=0.1 = G4 / Step 1 cell 5.  
> - At C5 Step 2: use published or local Adam FedAvg Final_F1 for CIC CNN α=1.0 s{seed} as the Adam µ=0 reference for that seed (published row is acceptable for the **floor denominator only**; SGD arms are still local).

| Floor result | How to report H1 |
|--------------|------------------|
| Floor **met** and \|Δ_SGD(0.01)\| &lt; 0.5 pp and trajectories track | **H1 confirmed** (descriptive) |
| Floor **met** and \|Δ\| ≥ 0.5 pp or trajectories diverge | **H1 rejected** |
| Floor **not met** | **H1 inconclusive** — **not** confirmed; do not use Δ_SGD to support Adam×µ |

The **0.80** threshold is a pre-registered judgment (not data-derived). Any fixed floor would work; **some** floor must be locked before results exist.

---

## 7. Pre-registered hypotheses

**Note on H3 vs H4b:** These are **not** contradictory. **H3** constrains **Final** F1 Δ(µ) (monotone non-increasing in µ). **H4b** constrains **round-5 F1** (up in µ) and **late-gain** (down in µ). Together they **are** the Mode B trade (early help, late cap). Different metrics by design.

| ID | Statement | Pass sketch (descriptive) |
|----|-----------|---------------------------|
| **H1** | Under **SGD (mom=0.9)**, \|Δ(0.01)\| &lt; **0.5 pp** Final F1 at N-BaIoT MLP α=1.0 s42 **and** α=0.1 s42 (and, if floor met, at C5 SGD seeds); trajectories of µ=0.01 track µ=0 | **Subject to §6.1 floor** on every cell scored |
| **H2** | Under **Adam** µ=0.01, local CIC CNN α=1.0 s42 FedProx matches **G6** targets; combined with the **local** FedAvg gate on the same cell (2026-08-07, Final_F1 **0.636612200542919**), the **s42** C5 gap of **−6.94 pp** is reproduced entirely from local runs. **Scope: s42 only** — C5’s ~6.7 pp is a 3-seed mean; no local 3-seed Adam Prox×FA pair exists in this prereg | **Tested by G6** + existing local FA history; not by G3 alone |
| **H3** | Under **Adam** at N-BaIoT MLP α=**0.1** s42, **Final** F1 Δ(µ) is **monotone non-increasing** over µ ∈ {0, 0.001, 0.01, 0.1, 1.0} using **local** anchors only (cells 5–9 + interpolating order) | Dose–response on **Final** |
| **H4a** | **Mode A (α=1.0, Adam):** raising µ **lowers** round-5 F1 | Shape / r5 |
| **H4b** | **Mode B (α=0.1, Adam):** raising µ **raises** round-5 F1 and **lowers** late-gain | Shape / r5 + late-gain (not Final) |
| **H4c** | **SGD, both α** (and C5 if floor met): µ=0.01 tracks µ=0 round-by-round; **no** meaningful shape change | Optimizer dependence of shapes |

### Outcome branches

| Result | Interpretation | Contract impact |
|--------|----------------|-----------------|
| **H1** (floor met) **∧ H2 ∧ (H4a∧H4b) ∧ H4c** | Deficit is **Adam×µ** (skew-dependent shape), not “FedProx is useless” | **C5 survives** with mechanism; Adam µ=0.01 FedProx is **not** a fairly configured absolute baseline vs FedTrim — **interpretation** of Prox comparators, **not** F9’s truth (CIC CNN Best FedTrim **63.81 / 64.68 / 56.34** at three α) |
| **H1 inconclusive** (floor fail) | Cannot use SGD Δ to argue Adam interaction | Report floor fail; do **not** claim H1; H2/H3/H4 may still stand |
| **¬H1** (floor met, µ hurts under SGD) | µ=0.01 too strong even without Adam | C5 scope unchanged; “µ never tuned” |
| **¬H2** (G6 fail) | **Lineage** on C5 **s42** | Stop local C5-gap claims; debug FedProx path on CIC CNN |
| **H4a xor H4b** fails | Single-shape story false | Report mode-specific truth |

---

## 8. Cost summary

| Block | Cells | Nominal wall |
|-------|-------|----------------|
| G1–G3 | 3 | ~2.0 h |
| Step 1 (incl. G4/G5) | 10 | **~9.5 h floor** (see §5; was 6.7 h pre-G3) |
| Step 2 (G6 + 6 SGD) | 7 | ~19.25 h |
| **Core total** | **20** | **≈ 30.5–31 h floor** (G1–G3 ~2.4 h measured + Step 1 + Step 2) |
| Optional Mode C | +2… | **+4–5 h** |

(Prior 17-cell / 24 h and pre-G3 “28 h” figures are **superseded**.)

---

## 9. Reporting / freeze rules

- Packet under `q1_e3_results/reporting/` with seed-level Final Acc/F1/AUC, r5 F1, late-gain, full histories, H1 floor ratios, SHA256SUMS.  
- **F3 spirit:** no “statistically significant …” claims after multiplicity correction on these contrasts; no Welch/BH shopping on n=1 or n=3 E3 cells.  
- Any C5 wording change → dated contract amendment (who / what / why).  
- After any engine edit: **G1** before claiming shared-path lineage still holds.  
- Hand-edits to packet files after generation must refresh SHA256SUMS (and note builder generation vs post-edit hash if needed).

---

## 10. Explicitly out of scope (this prereg)

- Full µ grid × all datasets × all models × all seeds  
- Learning-rate retune for SGD (momentum and lr frozen)  
- SCAFFOLD / FedNova / Median (Q1-E5)  
- Device split (Q1-E2)  
- Seed expansion families P1–P11 (Q1-E4)  
- Inferential tests on Prox−FA  
- Claiming Mode B on CIC without new evidence  
- Using published-only α=0.1 Adam anchors for the dose–response curve  

---

## 11. Acceptance checklist (before GPU)

- [x] Proposer / approver / date for **prereg acceptance** (**Sheryar Kiani** / **Dr Hussain Kazmi** / **2026-08-15**; Accept E3 prereg r3)  

- [ ] Mode A: **12/12** behind@r5, **10/12** below@r30; s123/s7 named  
- [ ] **H1 floor 80%** locked; inconclusive ≠ confirmed  
- [ ] SGD **momentum=0.9** frozen in engine plan  
- [ ] α=0.1 **local** Adam µ=0 and µ=0.01 (G4/G5) in Step 1  
- [ ] **H2:** G6 + local FA 2026-08-07; s42 gap −6.94 pp; not 3-seed mean  

- [ ] **Stop on G2** (and G1/G3/G4) as well as G3  
- [ ] H3 = Final; H4b = r5 + late-gain (metrics note)  
- [ ] F3 cited as **spirit** / quoted FedTrim-specific text  
- [ ] Core **20 cells ≈ 28 h**; Mode C optional +4–5 h  
- [ ] Engine change plan reviewed; gates ordered before science  

---

## 12. Recommended next actions (ordered)

1. **Supervisor acceptance** of prereg **r3** (Sheryar Kiani / Dr Hussain Kazmi / date) — not peer review alone.  
2. Implement optimizer + µ knobs + **SGD momentum=0.9**; defaults preserve current Adam / µ=0.01 behavior.  
3. **Backup** `probe_fast_match_notebook.json` → `…_PRE_E3ENGINE_….json` (see header).  
4. Run **G1 → G2 → G3** only; paste each gate footer for independent verify; **stop** on any fail.  
5. Step 1 (incl. G4/G5) → Step 2 (G6 then SGD C5 cells) **only after** G1–G3 all pass.  
6. Apply H1 floor when scoring; packet + draft C5 amendment only after results.

---

## 13. Helpers already on disk

| File | Role |
|------|------|
| `_e3_two_mode_precheck.py` | α-split Mode A/B tallies (published histories) |
| `_e3_trajectory_precheck.py` | earlier early/late helper |

Numbers in §2 were re-derived from published `history_*.csv` / `all_runs_raw.csv` in multi-pass review; treat scripts as convenience, not sole authority.

---

## 14. Changelog (draft only)

| Rev | Change |
|-----|--------|
| r1 | Initial prereg after two-mode redesign |
| r2 | H1 80% floor (blocking); SGD mom=0.9; +2 α=0.1 Adam anchors; G6 H2 on C5; G2 stop; H3/H4b metric clause; F3 spirit; 20 cells ≈ 28 h |
| r3 | §4.2 heading: G4/G6 hard, G5 soft; H2 fully local s42 −6.94 pp (G6 + 2026-08-07 FA); G5 fail = cross-check only |
| r3+ | Acceptance protocol (proposer/approver roles); pre-G1 probe JSON backup note — design unchanged |
| r3++ | Pre-G1 tooling fixes: G3 pass bar = scalars **+** 30/30 F1/Acc vs published Prox; G2 Acc channel; per-gate cfg asserts; runner `--require-from-cache` + CUDA refuse; G1 archive `fedprox_mu=0.0` |
| r3+++ | G4 blast radius corrected (5–9 invalidate; 3–4 lose H1 floor only); Step 1 cost **~9.5 h**; runner enforces full G4 bar + hard stop; G4–G5 = Step1 cells 5–6 |

---

*End of draft. Not a contract amendment.*

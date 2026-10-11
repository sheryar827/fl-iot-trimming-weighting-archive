# Draft amendment — Q1-E1b size-weighted trim complete (non-IID) + C1 precision

Date (UTC draft): 2026-08-15  
Status: **APPLIED 2026-08-15** — proposer Sheryar Kiani; approver Dr Hussain Kazmi; choice **A–G with corrected C1**

**Freeze options (both defensible):**

| Option | What to apply | When |
|--------|----------------|------|
| **1 (status today)** | **A–F only** | Marks E1b Done; open **same-day/next** C1 amendment (G) before ranking prose |
| **2 (recommended)** | **A–G together** | E1b Done **and** C1 corrected in one freeze |

**Applied option:** **2 (A–G together)** with corrected C1.  
**Do not** apply A–F and leave C1 as unscoped “first in all 6 cells.”

**C1 (item G)** uses the **corrected** form below (not “below SD of any single arm,” not “published arms only”).

---

## Who

- Proposer (named person): **Sheryar Kiani**
- Technical source: local Q1-E1b P3200 `match_notebook` + `q1_e1b_results/reporting/`; re-gate STRONG PASS 2026-08-15
- Approver (named person): **Dr Hussain Kazmi**
- Acceptance date: **2026-08-15**

---

## What — apply **A–G** as chosen

### A. §5 Q1-E1b **full table row**

**Current:**

```
| **Q1-E1b** | **Size-weighted trimmed mean** (recommended) | **In progress** (local P3200 `match_notebook` non-IID grid) | Completes 2×2 with Q1-E1 (**new** — no audit counterpart) |
```

**Replace with:**

```
| **Q1-E1b** | **Size-weighted trimmed mean** (`WeightedFedTrimmedAvg`; coordinate-wise trim then size-weighted mean of survivors; β_eff=1/9) on the **non-IID paper grid** (both datasets × both models × α ∈ {1.0, 0.5, 0.1} × seeds {42, 123, 7} = **36 runs**; Final Acc, Final F1, AUC under `q1_e1b_results/`; **IID excluded**) | **Done (non-IID)** | Completes trim×weight 2×2 with Q1-E1; reporting packet `q1_e1b_results/reporting/` |
```

### B. §1 working answer (full heading + paragraph)

**Current** (live contract — unscoped “ranks first in all six” must go):

```
### Working answer (descriptive; trim-vs-weight attribution pending Q1-E1b)

On the harder **CIC-IoT2023** benchmark, the **FedTrimmedAvg arm** ranks first on macro-F1 in all six model×α cells with **large** Final margins at α ∈ {0.5, 0.1}; on **N-BaIoT**, many aggregator gaps are small or **protocol-sensitive**. **Q1-E1** (unweighted FedAvg) **non-IID** is complete (**36** runs; three-primary packet under `q1_e1_results/reporting/`). At n=3, UW contrasts are **descriptive only** (F3). We still do **not** claim CIC FedTrim gains are “trimming alone”: full trim-vs-size-weighting attribution requires **Q1-E1b** (size-weighted trim) to complete the 2×2. **IID UW was not run** in this freeze.
```

**Replace with** (aligned with revised **C1**):

```
### Working answer (descriptive; trim-vs-weight 2×2 on disk; n=3)

On the harder **CIC-IoT2023** benchmark, the **FedTrimmedAvg arm** ranks first on Final **and** published Best macro-F1 in the **four α ∈ {0.5, 0.1} cells**, ahead of **every arm run in this study**, with **large** Final margins vs FedAvg; at **α=1.0** the arms are **not separable at n=3** (see **C1**). On **N-BaIoT**, many aggregator gaps are small or **protocol-sensitive**. **Q1-E1** (unweighted FedAvg) and **Q1-E1b** (size-weighted trim) **non-IID** grids are complete (**36 + 36** runs; packets under `q1_e1_results/reporting/` and `q1_e1b_results/reporting/`). At n=3, UW/WT contrasts are **descriptive only** (F3). Descriptive 2×2: holding weighting fixed, trim tends to help vs size-weighted FedAvg; holding trim fixed, re-introducing size weights often hurts under severe **CIC** skew (packet WT−FT; always label dataset). **WT−UW** changes both factors — not a single-factor attribution. **Not** “trimming alone” without those margins. **IID UW/WT not run** in this freeze.
```

### C. §8 next-action item 2

When applying, strike the **full** live item-2 paragraph, then:

```
2. ~~[full prior item-2 text from live contract]~~ **DONE (non-IID E1, 36 runs).** **DONE (non-IID E1b, 36 runs, `WeightedFedTrimmedAvg`).** Reporting: `q1_e1_results/reporting/` and `q1_e1b_results/reporting/`. Shared-path re-gate STRONG PASS 2026-08-15 (N-BaIoT MLP FedAvg α=1.0 s42). **Next experimental block (pick one by bottleneck after 2×2 read-out):** Q1-E2 (9-device N-BaIoT) **or** Q1-E3 (FedProx µ + Adam/SGD) **or** Q1-E4 (seeds on P1–P11) **or** Q1-E7 (per-class) — not all at once.
```

### D. Header **Amended:** (preserve prior lines; append)

Keep **2026-08-01** and **2026-08-11**. **Append:**

```
**Amended:** 2026-08-15 — *by:* Sheryar Kiani, approved Dr Hussain Kazmi; *what:* Q1-E1b non-IID **Done** (36 WeightedFedTrimmedAvg); §5 E1b; §1 working answer; §8 next after E1b; **C1** rewritten (first at α∈{0.5,0.1} vs every arm in study; α=1.0 non-separable; seed counts); *why:* 2×2 complete; re-gate STRONG PASS 2026-08-15; F2 precision; descriptive only at n=3  
```

(If **A–F only**, omit C1 from *what* and schedule G immediately after.)

### E. Number source (append E1b)

Add: `**Q1-E1b WT non-IID numbers:** Research Paper/q1_e1b_results/reporting/`

### F. Lineage sentence (with A–F)

```
Shared `match_notebook` path + `aggregate_fedavg_np` re-validated 2026-08-15 (probe STRONG PASS; history bit-identical to published). Does **not** re-validate unweighted trim or UnweightedFedAvg aggregators. Weighted trim unit-tested separately; E1b Final_F1 differs from published FedTrim on all CIC cells (weights applied).
```

### G. Claim **C1** (corrected form — use this text)

**Current:**

```
| **C1** | On **CIC-IoT2023**, the **FedTrimmedAvg arm** **ranks first** on Final (and published Best) macro-F1 in **all 6** model×α cells; Final margins vs FedAvg can be **large** at α ∈ {0.5, 0.1} (**+6.34 to +16.83** pp across the four cells; α=1.0 margins are small: **MLP +1.15 / CNN +0.41**). | Ranking + effect size; **not** “significant after FDR at n=3” |
```

**Replace with:**

```
| **C1** | On **CIC-IoT2023**, the **FedTrimmedAvg arm** ranks first on Final **and** published Best macro-F1 in the **four α ∈ {0.5, 0.1} cells**, ahead of **every arm run in this study** (FedAvg, FedProx, Q1-E1 unweighted FedAvg, Q1-E1b size-weighted trim), with **large** Final margins vs FedAvg (**+6.34 to +16.83** pp). At **α=1.0 the arms are not separable at n=3**: excluding FedProx, all arms lie within **1.28 pp (MLP) / 0.60 pp (CNN)** — of the same order as, or smaller than, the seed-to-seed SD of those four arms (**1.2–2.6 pp**) — and the WT−FedTrim paired difference has SD **0.76 pp (MLP) / 0.33 pp (CNN)** against means of **+0.13 / +0.19** pp. Nominal α=1.0 values, descriptive only (F3): FedTrim vs FedAvg **MLP +1.15 / CNN +0.41**; Q1-E1b size-weighted trim vs FedTrim Final **+0.13 pp (WT ahead in 1/3 seeds) / +0.19 pp (2/3 seeds)** (Best **+0.04 / +0.32** pp); Q1-E1 unweighted FedAvg stays **below** FedTrim Final in **all six** CIC cells. **No ranking claim is made at α=1.0.** | Effect located at **α ≤ 0.5**; α=1.0 explicitly non-separable (margin vs seed-SD / paired SD, **not** a hypothesis test); **not** “significant after FDR at n=3” |
```

**Rejected / withdrawn wordings:**

1. Unscoped “first in all 6 cells” (false once WT exists at α=1.0).  
2. “First among published arms only” (defensive comparator exclusion).  
3. “Below the seed SD of **any** single arm (1.2–2.6 pp)” — **false**: FedProx SDs 0.74 / 0.22; MLP four-arm spread 1.28 > FedAvg SD 1.23.

**Disk check (CIC Final F1 seed means, sample SD n=3):** α=0.5/0.1 → FT first in **8/8** Final+Best vs all five arms; α=1.0 WT−FT paired means +0.13/+0.19 with paired SD 0.76/0.33 and seed wins 1/3 and 2/3.

---

## Why

- E1b 36/36; packet verified; re-gate STRONG PASS 2026-08-15.  
- C1 must locate FedTrim’s ranking where it holds (α ≤ 0.5) and refuse ranking at α=1.0.  
- F2: no blanket “first” without a true qualifier.

## 2×2 write-up guard (not a contract edit)

- Single-factor: **WT−FA** / **UW−FT** (trim); **WT−FT** / **UW−FA** (weighting).  
- **WT−UW** is two-factor.  
- “Not separable” ≠ “not significant” (F3).

## Acceptance checklist

- [x] Proposer named (**Sheryar Kiani**)  
- [x] Approver named (**Dr Hussain Kazmi**)  
- [x] Date filled (**2026-08-15**)  
- [x] Choice: **A–G with corrected C1**  
- [x] Applied to `Q1_CONTRIBUTION_CONTRACT.md`  
- [x] Packet cited: `q1_e1b_results/reporting/Q1_E1b_Three_Primary_Packet.md`  
- [x] No F3 “significant” language  

## Not included

- Multiplicity-corrected tests  
- IID WT/UW  
- C8 changes  
- Top-Q1 readiness  
- Single-factor reading of WT−UW  

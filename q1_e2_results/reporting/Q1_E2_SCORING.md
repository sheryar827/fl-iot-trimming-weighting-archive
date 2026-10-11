# Q1-E2 scoring (device-based N-BaIoT, `scale_on=train`)

**Status:** **FROZEN 2026-09-22** on contract (Sheryar Kiani / Dr Hussain Kazmi).  
**Scored:** 2026-09-22 from unrounded stored Final F1.  
**Prereg:** `Q1_E2_PREREG_DRAFT.md` **ACCEPTED** 2026-09-21 (Sheryar Kiani / Dr Hussain Kazmi).  
**F3:** descriptive only at n=3. Exact equality = tie = not supported.  
**Do not** mix with `scale_on=full` E1/E1b. **Do not** claim deployment, unseen-device, or that CIC HC1 “transfers.”

**Cache:** `nbaiot_4ddb8b93b8821650` (`scale_on=train`; F=115 C=11; train 1,418,112 / test 354,529).  
**Provenance:** NPZ `c0a1e21cd6336d3a090e5feb9411758abbf8669dcca01bd476defdbcf8380854`, meta `0124ac4c793417d53bd062f4694e72ba903ed42beee8873d578adf57a3686b23`.  
**Integrity:** **24/24** unique cells; each history 30 rounds 1–30; last-round F1 bit-identical to stored Final F1; row NPZ/meta hashes match provenance; P3200; torch 2.5.1+cu121; `match_notebook`; partition=device.

Class coverage (train): Ennio doorbell and Samsung webcam **6/11** (no Mirai); other seven devices **11/11**. Pooled test contains all 9 device ids.

---

## 3-seed mean Final F1

| Model | FedAvg | UnweightedFedAvg | FedTrimmedAvg | WeightedFedTrimmedAvg |
|-------|--------|------------------|---------------|------------------------|
| MLP | 0.7934827068300899 | 0.7819234992608942 | 0.7729712641784282 | 0.7998153006931568 |
| CNN | 0.7537413433273343 | 0.7779569338543405 | 0.7385130920430134 | 0.7757958394388934 |

Ranks (no pass/fail ranking hypothesis):  
MLP: WT > FA > UW > FT.  
CNN: UW > WT > FA > FT.

---

## 2×2 (prereg directions)

Trim added: **FT−UW**, **WT−FA** (help ⇒ both mean Δ > 0).  
Weight added: **FA−UW**, **WT−FT** (hurt ⇒ both mean Δ < 0).  
Never WT−UW as one factor.

| Model | Contrast | Mean Δ | s42 / s123 / s7 signs |
|-------|----------|--------|------------------------|
| MLP | FT−UW | −0.0089522350824659 | 0+ / 3− |
| MLP | WT−FA | +0.0063325938630668 | 2+ / 1− |
| MLP | FA−UW | +0.0115592075691957 | 2+ / 1− |
| MLP | WT−FT | +0.0268440365147284 | 3+ / 0− |
| CNN | FT−UW | −0.0394438418113271 | 0+ / 3− |
| CNN | WT−FA | +0.0220544961115590 | 2+ / 1− |
| CNN | FA−UW | −0.0242155905270061 | 1+ / 2− |
| CNN | WT−FT | +0.0372827473958800 | 3+ / 0− |

**HE2-trim:** not supported on MLP or CNN (FT−UW < 0 on both; WT−FA > 0 on both). **0/2**.  
**HE2-weight:** not supported on MLP or CNN (WT−FT > 0 on both; FA−UW mixed). **0/2**.

Wave-1 MLP scoring is unchanged; this is the **final** descriptive assessment after Wave 2.

---

## What this can and cannot show

**Can:** on this device-based N-BaIoT 2×2, the CIC confirmatory sign pattern (trim help + weight hurt) did **not** hold at n=3.  
**Cannot:** prove Byzantine robustness, unseen-device generalization, or that size-weight “helps” as a new hypothesis (WT−FT was not a pre-registered help claim).

C5/C8 unchanged. Wave 3 CIC remains optional and deferred.

Paired-contrast CSV: `Q1_E2_PAIRED_CONTRASTS.csv`.

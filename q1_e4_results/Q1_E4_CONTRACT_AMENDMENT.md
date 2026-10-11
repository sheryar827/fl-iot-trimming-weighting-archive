# Amendment — Q1-E4 completion

**Status:** **APPLIED 2026-10-05** — proposer **Sheryar Kiani**; approver **Dr Hussain Kazmi** (supervisor)  
**Acceptance date:** **2026-10-05** (5-10-2026)  
**Parent:** `reporting/Q1_E4_TRAIN_SCALER_SEED_AMENDMENT_DRAFT.md` ACCEPTED 2026-09-26 at **0/112 new cells**; packet `reporting/Q1_E4_REPORTING_PACKET.md`

## Who (accepted)

- Proposer: **Sheryar Kiani**  
- Approver: **Dr Hussain Kazmi** (supervisor)  
- Acceptance date: **2026-10-05** (5-10-2026)

## What (applied to `Q1_CONTRIBUTION_CONTRACT.md` on 2026-10-05)

Q1-E4 train-only seed expansion on T1/T2/T5/T6 (`scale_on=train`; CIC alpha=0.1 MLP/CNN and N-BaIoT nine-device MLP/CNN; UW/FA/FT/WT) is **Done (112/112 new cells)**. Packet: `q1_e4_results/reporting/`. Waves 0–4 are 4/4, 28/28, 32/32, 32/32, and 16/16. New-seed quartets are 28/28. The packet has 160 seed rows, 16 prospective paired-t/Holm comparisons, and 16 combined ten-seed descriptive estimates.

Only T2, CIC-IoT2023 CNN, FedAvg − UnweightedFedAvg met the preregistered Holm-adjusted decision rule: mean **−10.7252** percentage points; unadjusted 95% paired-t confidence interval **[−15.9527, −5.4977]**; signs **0/0/7**; raw p-value **0.00240333**; Holm-adjusted p-value **0.0384533**. The other 15 comparisons are inconclusive. Htrim is not supported in T1, T2, T5, or T6. Hweight-hurt is not supported in T1, T2, T5, or T6, because T2 weighted-trim − unweighted-trim is inconclusive (Holm-adjusted p-value **1**). Inconclusive is not equivalence and does not show that an effect is absent. Ten-seed estimates and the interaction remain descriptive only.

No universal, equivalence, deployment, privacy, rare-class, or adversarial-robustness claim is authorized. Legacy P6–P11 triggers stay dormant. **C5/C6/C8 unchanged**. The 2026-09-26 preregistration line stays as history. Q1-E5 remains not started and has no accepted preregistration. CIC Wave 3 and E8 remain deferred. No further Q1-E4 GPU work is required.

## Why

The preregistered 112-cell grid and independently verified reporting packet are complete. This closes Q1-E4 without broadening the frozen scientific claims.

## Acceptance checklist

- [x] Proposer / approver / date (**Sheryar Kiani** / **Dr Hussain Kazmi** / **2026-10-05**)  
- [x] 2026-09-26 preregistration entry preserved as history, including the **0/112 new cells** wording the sealed runner still checks  
- [x] Contract header, Number source, Q1-E4 status, Q1-E4 section, Top-Q1 status, and next action updated  
- [x] C5/C6/C8 unchanged; C3/C4 legacy triggers not treated as satisfied  
- [x] E5 not started; CIC Wave 3 and E8 remain deferred  
- [x] No further Q1-E4 GPU authorized or required  
- [x] Sealed reporting manifest not regenerated

This post-packet governance record is intentionally outside `reporting/SHA256SUMS.txt`; the accepted 305-entry packet manifest was not regenerated.

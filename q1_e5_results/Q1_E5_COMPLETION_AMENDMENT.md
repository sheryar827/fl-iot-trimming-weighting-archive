# Amendment — Q1-E5 completion

**Status:** **APPLIED 2026-10-10** (10-10-2026) — proposer **Sheryar Kiani**; approver **Dr Hussain Kazmi** (supervisor)  
**Written:** 2026-10-10  
**Acceptance date:** **2026-10-10** (10-10-2026)  
**Parent design:** `q1_e5_results/Q1_E5_COORDMEDIAN_ONLY_AMENDMENT.md`, accepted 2026-10-07 (7-10-2026) at **0/28**  
**Parent preregistration:** `q1_e5_results/Q1_E5_PREREG_DRAFT.md`, accepted 2026-10-06 (10-6-2026) at **0/84**, SHA-256 `CD20C7A61B3A41AD214CC51BF4221F7837604FD1236F75282FD07BE346EC6517`  
**Packet:** `q1_e5_results/reporting/Q1_E5_REPORTING_PACKET.md`

## Who (accepted)

- Proposer: Sheryar Kiani
- Approver: Dr Hussain Kazmi
- Acceptance date: 2026-10-10 (10-10-2026)
- Decision: Accept this completion as written. Q1-E5, as reduced on 2026-10-07, is Done (28/28 CoordMedian–Adam cells). No block is supported. Inconclusive is not equivalence. FedNova remains deferred.

## What (applied to `Q1_CONTRIBUTION_CONTRACT.md` on 2026-10-10)

Q1-E5, as reduced on 2026-10-07, is **Done (28/28 CoordMedian–Adam cells)**. Packet: `q1_e5_results/reporting/`. The cells are T1, T2, T5, and T6, seeds {127, 131, 137, 139, 149, 151, 157}, CoordMedian–Adam at learning rate `1e-3`, sealed match-notebook local step, 30 rounds. The server aggregate is the fifth ordered value across nine clients and does not weight clients by sample size. The comparator is the sealed Q1-E4 FedTrimmedAvg–Adam row for the same block and seed. Those rows were not retrained. The first cell started at 2026-10-07T14:46:15Z. The last cell finished at 2026-10-09T19:25:19Z.

The decision endpoint is unrounded Final macro-F1 at round 30. Each test is a two-sided paired \(t\) test on the seven seed-wise differences, sealed FedTrimmedAvg–Adam minus CoordMedian–Adam, with \(n=7\) and \(\mathrm{df}=6\). Holm adjustment covers these four tests only, in the order T1, T2, T5, T6, at family-wise level 0.05. Means and unadjusted 95% paired-\(t\) intervals are in percentage points. Signs are negative, tie, and positive counts, in that order. Raw and Holm \(p\)-values use Python `.6g`.

| Block | Contrast | Mean | 95% CI | Signs | Raw p | Holm p | Decision |
|---|---|---:|---|---:|---:|---:|---|
| T1 | FT−median | 0.3471 | [−1.6900, 2.3842] | 3/0/4 | 0.691258 | 1 | inconclusive |
| T2 | FT−median | 0.6388 | [−1.7203, 2.9980] | 3/0/4 | 0.532201 | 1 | inconclusive |
| T5 | FT−median | −1.7913 | [−4.5363, 0.9537] | 4/0/3 | 0.161432 | 0.645727 | inconclusive |
| T6 | FT−median | −1.6825 | [−4.8882, 1.5233] | 3/0/4 | 0.246421 | 0.739263 | inconclusive |

No block is supported. A positive supported mean would say the trimmed mean is higher than the median in that block only. A negative supported mean would say the median is higher in that block only. Inconclusive is not equivalence, and no equivalence procedure was applied. These adjusted \(p\)-values are the four-test family. They are not the historical eight-test family, and they are not combined with the Q1-E4 Holm result.

FedNova–momentum-SGD and matched FedAvg–momentum-SGD remain deferred. Deferral is not a finding that FedNova is inferior. StepNormAdam is not approved. SCAFFOLD is not an arm. No further Q1-E5 GPU command is authorized. The commands `tune`, `tune-t1-attempt2`, `wave`, `resume`, `coordmedian`, and `coordmedian-continue` stay closed.

The Q1-E4 result stays. Its only supported comparison is T2 CIC-IoT2023 CNN FedAvg minus UnweightedFedAvg: mean **−10.7252** percentage points; unadjusted 95% paired-\(t\) interval **[−15.9527, −5.4977]**; signs **0/0/7** in the Q1-E4 order positive, tie, and negative; raw \(p\) **0.00240333**; Holm \(p\) **0.0384533**. The other 15 Q1-E4 comparisons stay inconclusive. Neither complete Q1-E4 hypothesis is supported.

No universal, deployment, privacy, rare-class, or adversarial-robustness claim is authorized. **C5/C6/C8 unchanged.** CIC Wave 3 and Q1-E8 remain deferred. The working manuscript table drafted on 2026-10-10 stays an unfrozen draft. This file does not freeze it.

## Why

The 28-cell grid accepted on 2026-10-07 and the reporting packet recomputed from those cells are complete. This closes that reduced study without broadening its claims.

## What stays historical

- The 2026-10-06 acceptance at **0/84**, including `Q1_E5_CONTRACT_AMENDMENT.md` and the presented pre-acceptance hash. The accepted preregistration file remains SHA-256 `CD20C7A61B3A41AD214CC51BF4221F7837604FD1236F75282FD07BE346EC6517`.
- The 2026-10-07 acceptance at **0/28** in `Q1_E5_COORDMEDIAN_ONLY_AMENDMENT.md`, including its signed acceptance list.
- The 2026-10-05 Q1-E4 contract line, including the historical strings **0/112 new cells** and **Q1-E5 not started**. The sealed E4 runner still checks those strings.
- The sealed comparator file `q1_e4_results/reporting/q1_e4_seed.csv`, SHA-256 `8a9218a130486b68c69f3dd40cf5dd82a217add985c66a91af6fe3dd97da90c9`.
- The Q1-E4 and Q1-E5 reporting manifests. This governance record stands outside both manifests.

## What the signature changed in the contract

Applied on 2026-10-10:

1. A new amendment-log line for this completion. The 2026-10-06 and 2026-10-07 log lines were not rewritten.
2. Q1-E5 is **Done (28/28)** in the program-status row. **ACCEPTED; 0/84** on 2026-10-06 and **ACCEPTED; 0/28** on 2026-10-07 remain in that row as history.
3. A Done paragraph was added to the Q1-E5 section. The 2026-10-06 and 2026-10-07 paragraphs were not deleted.
4. **Done (28/28)** was added to the Top Q1 parenthesis. The **0/84** and **0/28** phrases already there were not deleted.
5. The next-action sentence that still named a CoordMedian smoke cell was replaced. That sentence would have reopened a closed command. The **0/84** and **0/28** phrases in that sentence were kept.

## Acceptance checklist

- [x] Approver: Dr Hussain Kazmi
- [x] Acceptance date: 2026-10-10 (10-10-2026)
- [x] Decision line signed
- [x] Contract amendment-log line added without rewriting the 2026-10-06 or 2026-10-07 lines
- [x] Program-status row, Q1-E5 section, Top Q1 parenthesis, and the stale next-action sentence updated as specified above
- [x] 2026-10-05 **0/112 new cells** and **Q1-E5 not started** strings still present
- [x] `Q1_E5_PREREG_DRAFT.md` still hashes to `CD20C7A61B3A41AD214CC51BF4221F7837604FD1236F75282FD07BE346EC6517`
- [x] C5/C6/C8 unchanged; CIC Wave 3 and Q1-E8 still deferred
- [x] No further Q1-E5 GPU command authorized
- [x] Q1-E4 manifest and Q1-E5 manifest not regenerated
- [x] Working manuscript still marked as an unfrozen draft

## Acceptance

- Proposer: Sheryar Kiani
- Approver: Dr Hussain Kazmi
- Date: 2026-10-10 (10-10-2026)
- Decision: Accept this completion as written. Q1-E5, as reduced on 2026-10-07, is Done (28/28 CoordMedian–Adam cells). No block is supported. Inconclusive is not equivalence. FedNova remains deferred.

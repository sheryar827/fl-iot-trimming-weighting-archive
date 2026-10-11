# Amendment — Q1-E5 reduced to the 28-cell CoordMedian study

**Status:** **ACCEPTED 2026-10-07** (7-10-2026). This file does not edit `Q1_E5_PREREG_DRAFT.md`. Acceptance does not start a GPU command.  
**Written:** 2026-10-07  
**Accepted:** 2026-10-07 (7-10-2026)  
**Parent:** `Q1_E5_PREREG_DRAFT.md`, accepted SHA-256 `CD20C7A61B3A41AD214CC51BF4221F7837604FD1236F75282FD07BE346EC6517`  
**Stop record:** `q1_e5_results/Q1_E5_T1_NONFINITE_AMENDMENT.md`  
**Incident note:** `q1_e5_results/incidents/2026-10-06-t1-fednova-lr0.1/INCIDENT_AND_REPAIR.md`  
**Incident note SHA-256:** `17a7b49d1db8efef8263d2e46ae012a21ad39b84ad63cc8dda3460423272534a`

## Who

- Proposer: Sheryar Kiani
- Approver: Dr Hussain Kazmi
- Date: 2026-10-07 (7-10-2026)
- Decision: Defer FedNova after the Section 4.5 stop, and accept the 28-cell CoordMedian–Adam study specified below.

## Why this draft exists

Section 4.5 stopped T1. FedNova at learning rate 0.1 produced a non-finite client update during round 12 in both the preserved 2026-10-06 attempt and the 2026-10-07 attempt-2 rerun. The six validation history files are byte-for-byte identical across those attempts. Attempt 2 reloaded the scaler, indices, and partition before every cell. No T1 learning rate was locked. Decision cells remain 0/84.

The finished T1 scores are tuning diagnostics. They are not efficacy results, and they are not a basis for choosing 0.001 or 0.01. Gradient clipping, a replacement learning rate, and another recovery of the same grid are not proposed.

Section 10 of the accepted preregistration already names the alternative: the 28-cell Adam median comparison only, with FedNova explicitly deferred, written as its own acceptance. The reason for using that alternative now is the Section 4.5 stop. The compute budget is not the reason. This draft does not take effect silently.

## What would change

FedNova–momentum-SGD and matched FedAvg–momentum-SGD are deferred. Neither arm has a decision cell. Deferral is not a finding that FedNova is unstable or inferior.

The active study becomes 28 CoordMedian–Adam decision cells:

| Piece | Count |
|---|---:|
| Blocks | T1, T2, T5, T6 |
| Decision seeds | 127, 131, 137, 139, 149, 151, 157 |
| Arm | CoordMedian–Adam |
| Cells | 28 |
| SGD tuning runs | 0 |
| FedNova decision cells | 0 |
| FedAvg–momentum-SGD decision cells | 0 |

CoordMedian keeps the frozen Adam protocol in Section 3.1: `mode=match_notebook`, learning rate `1e-3`, PyTorch `2.5.1+cu121` Adam defaults (`betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0`, `amsgrad=False`), three local epochs, batch size 512, shuffle on, last batch kept, and proximal penalty 0. The partner is the sealed E4 FedTrimmedAvg–Adam row in `q1_e4_results/reporting/q1_e4_seed.csv`. No E4 arm is retrained. The coordinate-wise median remains the fifth ordered value. It uses no sample-size weight.

The only decision endpoint remains unrounded Final macro-F1 at round 30 on the sealed test set. Final accuracy, the mean of the 30 round-wise macro-F1 values, and Best-round scores stay descriptive. The round-wise mean is not ROC-AUC.

## Family

The original eight-test study ended before any decision cell. No decision-cell outcome from that study exists, so these four contrasts are not an outcome-selected subset of the eight, and they are not a selective report of four tests from eight. They form a separately accepted prospective study. The FedNova contrasts are not run, and they are not entered as nonsignificant results.

One family of **four** two-sided paired \(t\) tests, at family-wise level 0.05. Holm control applies only to these four tests. It does not apply across the Q1-E4 family, and it does not apply across the research program.

| Order | Block | Contrast |
|---|---|---|
| 1 | T1 | FT–Adam − CoordMedian–Adam |
| 2 | T2 | FT–Adam − CoordMedian–Adam |
| 3 | T5 | FT–Adam − CoordMedian–Adam |
| 4 | T6 | FT–Adam − CoordMedian–Adam |

Tied raw *p*-values keep this order. The Holm adjustment is cumulative and capped at 1. These four adjusted *p*-values are not the adjusted *p*-values the eight-test family would have produced. They are not combined with the Q1-E4 Holm *p*-values, and they are not a program-wide correction.

A positive Holm-supported mean says the trimmed mean is higher than the median in that block only. A negative Holm-supported mean says the median is higher. Every other result is inconclusive. Inconclusive is not equivalence. No TOST is authorized. One supported block is not a general claim. A median result is not evidence about FedNova.

For every comparison, report the mean paired difference in percentage points, the unadjusted 95% paired-\(t\) interval in percentage points, the raw *p*-value, the Holm *p*-value, and the negative, tie, and positive sign counts. Means and intervals are shown to four decimal places. *p*-values use Python's shortest round-trip format `.6g`. The zero-variance rules in Section 7.4 still apply, with \(n=7\) and \(\mathrm{df}=6\).

## Cell order

The 28 cells run in block order T1, T2, T5, T6. Within a block, the seed order is 127, 131, 137, 139, 149, 151, 157.

The first cell is the smoke cell: **T1, CoordMedian–Adam, seed 127**, learning rate `1e-3`. The accepted smoke cell, T1 FedAvg–momentum-SGD seed 127, is not part of this study and must not be started. If the CoordMedian smoke cell's saved row, 30-round history, or hash check fails, stop. Its score is not a reason to change the learning rate, the seeds, or the four-test family.

After a valid smoke cell, the other 27 cells run in the same block and seed order. They run regardless of signs or *p*-values. A technical failure pauses that cell for repair. It does not permit another seed, another learning rate, or a change to this family.

## Hours

The estimate is the CoordMedian row of the accepted Section 10 table, and no other row:

| Basis | Exact hours | Reported hours |
|---|---:|---:|
| Mean | 42.988554875486145 | 42.99 |
| Block maximum | 45.850877955749894 | 45.85 |

The 24 tuning runs, the two momentum-SGD arms, the full-campaign totals 165.81 and 176.85, and the 213-hour ceiling are not the budget for these 28 cells. The older 43–46 hour wording is not used.

## Preserved diagnostics

Both T1 attempts stay on disk. The 2026-10-06 files remain under `q1_e5_results/tuning/`. The 2026-10-07 files remain under `q1_e5_results/tuning/attempt2/`. Neither directory is an efficacy result. No learning rate is locked from either attempt.

## What acceptance authorizes

Acceptance does not edit the accepted preregistration and does not start a GPU cell. The authorized work is:

1. A CoordMedian-only runner for these 28 cells, in the order above, which refuses FedNova, matched FedAvg–momentum-SGD, SGD tuning, and the old FedAvg smoke cell.
2. CPU integrity tests for that refusal and for the four-test Holm rule.
3. A fresh no-training preflight after that code.
4. Then one smoke cell: T1, CoordMedian–Adam, seed 127.

The commands `tune`, `tune-t1-attempt2`, `wave`, and `resume` are not that runner.

Items 1 and 2 were added later on 2026-10-07. The CPU integrity suite then passed 43 tests, exit 0, in 2.192 seconds. That run did not rewrite `Q1_E5_PREFLIGHT.json`. Item 3 was run by the user and checked on 2026-10-07: `Q1_E5_PREFLIGHT.json` is status PASS, kind no-training-preflight, written_utc 2026-10-07T14:38:31Z, gpu_training false, scalers_written false, and partition_files_written false. Its SHA-256 is `b97099b27b2ce5ada4ea40e76e6528d9c63fa25b345dc9047f7d319478b8fb16`. A byte copy is `q1_e5_results/incidents/2026-10-06-t1-fednova-lr0.1/Q1_E5_PREFLIGHT_2026-10-07T14-38-31Z.json`. Item 4 ran and was checked. T1 CoordMedian–Adam seed 127, learning rate `1e-3`, is status complete, finite true, rounds 30, started 2026-10-07T14:46:15Z, finished 2026-10-07T16:28:46Z. Final macro-F1 is `0.506024873091221`. History SHA-256 is `ef4c2efd503647d188ca728ac8dba0dcc32ef4187c7f9e6269ba8dc4ce0ae3a4`. The saved CIC seed-127 partition matches `d6fbc871d4c6ca1fcd3d3ba9519c5227a78aa317ada71e24640dc38bcd0fdcec`. The process id is gone. No other CoordMedian cell file exists. The CPU integrity suite then passed 45 tests, exit 0, in 2.347 seconds. That run did not rewrite the preflight file, the smoke cell, or its history. The continuation then ran, skipping that cell, from 2026-10-08T03:01:39Z through the last cell at 2026-10-09T19:25:19Z. All 28 cells are complete, finite, and 30 rounds, at learning rate `1e-3`. The smoke-cell history was not rewritten. Its SHA-256 is still `ef4c2efd503647d188ca728ac8dba0dcc32ef4187c7f9e6269ba8dc4ce0ae3a4`.

The four-test Holm family was applied to FedTrimmedAvg–Adam minus CoordMedian–Adam. Means and unadjusted 95% paired-\(t\) intervals are in percentage points. Signs are negative, tie, and positive. Raw and Holm \(p\)-values use Python `.6g`.

| Block | Mean | 95% CI | Signs | Raw p | Holm p | Call |
|---|---:|---|---:|---:|---:|---|
| T1 | 0.3471 | [−1.6900, 2.3842] | 3/0/4 | 0.691258 | 1 | inconclusive |
| T2 | 0.6388 | [−1.7203, 2.9980] | 3/0/4 | 0.532201 | 1 | inconclusive |
| T5 | −1.7913 | [−4.5363, 0.9537] | 4/0/3 | 0.161432 | 0.645727 | inconclusive |
| T6 | −1.6825 | [−4.8882, 1.5233] | 3/0/4 | 0.246421 | 0.739263 | inconclusive |

No block is supported. Inconclusive is not equivalence. These adjusted \(p\)-values are the four-test family, not the historical eight-test family, and they are not combined with the Q1-E4 Holm result. FedNova remains deferred and is not a result.

## What remains closed

Q1-E4 is unchanged, including its single Holm result. C5, C6, and C8 are unchanged. No cache is rebuilt. No E4 arm is retrained. StepNormAdam and SCAFFOLD stay outside the study. The CoordMedian-only runner, its integrity tests, and the 2026-10-07T14:38:31Z no-training preflight have passed. All 28 CoordMedian–Adam cells are complete. No further Q1-E5 GPU command is authorized. The commands `tune`, `tune-t1-attempt2`, `wave`, `resume`, `coordmedian`, and `coordmedian-continue` stay closed.

## Acceptance

- Proposer: Sheryar Kiani
- Approver: Dr Hussain Kazmi
- Date: 2026-10-07 (7-10-2026)
- Decision: Defer FedNova after the Section 4.5 stop, and accept the 28-cell CoordMedian–Adam study specified in this amendment.

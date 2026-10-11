# Informational stop record — Q1-E5 T1 learning-rate search stopped

**Status:** **INFORMATIONAL STOP RECORD.** This file is not an acceptance and not a draft awaiting signature. It does not edit `Q1_E5_PREREG_DRAFT.md`. It authorizes no GPU command. The blank lines at the end stay blank: Dr Hussain Kazmi did not sign this file. The continuation decision is the accepted 28-cell amendment `q1_e5_results/Q1_E5_COORDMEDIAN_ONLY_AMENDMENT.md` (accepted 2026-10-07; proposer Sheryar Kiani; approver Dr Hussain Kazmi). This stop record does not hash that amendment.  
**Written:** 2026-10-07  
**Parent:** `Q1_E5_PREREG_DRAFT.md`, accepted SHA-256 `CD20C7A61B3A41AD214CC51BF4221F7837604FD1236F75282FD07BE346EC6517`. That file was hashed again on 2026-10-07 after the recovery and still matches.  
**Recovery approval:** `q1_e5_results/incidents/2026-10-06-t1-fednova-lr0.1/Q1_E5_T1_RERUN_APPROVAL.md`  
**Incident note:** `q1_e5_results/incidents/2026-10-06-t1-fednova-lr0.1/INCIDENT_AND_REPAIR.md`  
**Incident note SHA-256:** `17a7b49d1db8efef8263d2e46ae012a21ad39b84ad63cc8dda3460423272534a`

## Who

- Recorded by: Sheryar Kiani
- Approver of this file: none. The blank lines below are not a signature.
- Date of this record: 2026-10-07
- What this record does: preserve the facts of the T1 Section 4.5 stop. It does not accept a new study. The live continuation is the CoordMedian-only amendment named in the status line.

## What the recovery showed

Dr Hussain Kazmi approved one T1 recovery on 2026-10-07. The user ran `py -3.12 -u run_q1_e5.py tune-t1-attempt2`. It started at 2026-10-07T04:09:13Z, process id 120264, and stopped at 2026-10-07T11:12:14Z. The rerun used rates 0.001, 0.01, and 0.1, FedAvg before FedNova, seed 10007, and validation Final macro-F1. Each cell reloaded its scaler, indices, and partition. Output was written under `q1_e5_results/tuning/attempt2/`. The 2026-10-06 files were left in place.

The initial-parameter hash was `0491fc7dd205aa6e7c3127c252dee5671020c0748d730d99212fc6dcce4d9a5f`, the same value as the preserved attempt. Every attempt-2 cell stored the same artifact hashes: partition `b141f6d439a249ef30dec11453012b6591364ed2f583fa92af1f5b7a7947ee47`, scaler `8cb242fa385c9ee228898ada0a633d7ef05adee618fc9ceefaaea09d50c3e642`, combined index `6ea206ee8eb13a111e10a118339292fb5c8c1b12c85376a270cf6dce9b899649`, and matrix `1debafcc7ffacabdcbf44bd3ed874527cbfc0ded498055c11a11165d31c982e1`.

Five cells completed 30 finite rounds. Their stored histories are identical to the preserved histories. T1 FedNova at learning rate 0.1 is status `failed` and `finite` false. Eleven rounds were saved. Those eleven rounds are identical to the preserved history, including history SHA-256 `4d49879256bbeb998b856c5bd087c43c56837e1c6776e0eb873cf135a4a2f8d0`. Round 11 loss is `9.746182249449104e+16` and its macro-F1 is `0.049974839837118`. Round 12 raised `RuntimeError: delta_i is non-finite`. That cell started at 2026-10-07T10:43:52Z and finished at 2026-10-07T11:12:14Z.

The preserved cell still says `endpoint is not the start plus delta_i in float32`. The attempt-2 cell says `delta_i is non-finite`. The histories match, so both messages describe the same non-finite client update. The repaired check names it as non-finite. The FedNova server update was not changed, and this exception is inside the shared momentum-SGD client update, before that round's server step.

| Rate | FedAvg Final macro-F1 | FedNova Final macro-F1 |
|---|---|---|
| 0.001 | 0.4583327996126333 | 0.44398835718307417 |
| 0.01 | 0.4804099422091194 | 0.39112297919882344 |
| 0.1 | 0.49990466873723177 | failed in round 12 |

These five scores are diagnostics. They are not a learning-rate choice. Section 4.5 says that if any one of the six runs is missing, non-finite, or incomplete, stop, and do not select from the runs that remain. The 0.1 pair has no mean. No lock file exists. `tune.pid` is absent. No decision cell has started. The campaign remains 0/84.

## Why the accepted text is not enough to continue

Section 4.5 has been applied. T1 has no locked learning rate. Section 9 says the smoke cell is T1, FedAvg–momentum-SGD, seed 127, at T1's locked rate. That cell cannot start. The eight-test Holm family is conditional on the locked rates. It has not started.

Choosing 0.001 or 0.01 from the finished cells would break Section 4.5. Removing the T1 FedNova contrast, changing the candidate set, or starting T2, T5, or T6 would also change the accepted preregistration. None of those changes is made by this stop record.

Running `tune-t1-attempt2` again would skip the five completed cells and stop on the failed cell. It would not replace that cell.

## Text preserved here, not accepted through this file

The paragraph below records the stop this note described. Accepting it as an amendment to the preregistration was not done through this file, and filling the blank lines below would not create that acceptance. The accepted continuation is the 28-cell CoordMedian study. This paragraph does not replace Section 4.5.

> T1's six-run search was executed twice under rates \(\{10^{-3}, 10^{-2}, 10^{-1}\}\), seed 10007. Both times FedNova at \(10^{-1}\) produced a non-finite client update during round 12, with 11 finite rounds saved. The 2026-10-07 execution reloaded the scaler, indices, and partition before every cell and reproduced the 2026-10-06 validation histories exactly. Section 4.5 therefore stops T1. No T1 learning rate is locked. The T1 smoke cell and the T1 FedNova−FedAvg decision contrast do not start from these runs. The scores at \(10^{-3}\) and \(10^{-2}\) are not a basis for a later rate. Any new T1 candidate set, any change to the eight-test family, and any start of T2, T5, or T6 require a separate accepted amendment.

## Approval lines left blank

These lines stay blank on purpose. They are not an open signature request, and they are not a record that Dr Hussain Kazmi signed this file.

- Proposer:
- Approver:
- Date:
- Decision:

## Where the continuation is recorded

The 28-cell CoordMedian-only amendment is `q1_e5_results/Q1_E5_COORDMEDIAN_ONLY_AMENDMENT.md`. Dr Hussain Kazmi accepted that amendment on 2026-10-07 (7-10-2026); proposer Sheryar Kiani. This stop record does not hash that amendment. The campaign count in the narrative above, 0/84, is the count at the time of the stop. The later accepted study is 0/28 CoordMedian–Adam cells.

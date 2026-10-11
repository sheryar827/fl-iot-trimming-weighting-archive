# Q1-E5 technical approval — T1 tuning recovery

**Status:** **APPLIED.** One controlled technical recovery of T1 only. The full E5 campaign stays closed.  
**Recorded:** 2026-10-07  
**Incident note SHA-256:** `17a7b49d1db8efef8263d2e46ae012a21ad39b84ad63cc8dda3460423272534a`  
**Incident note:** `q1_e5_results/incidents/2026-10-06-t1-fednova-lr0.1/INCIDENT_AND_REPAIR.md`

## Acceptance

- Proposer: Sheryar Kiani
- Approver: Dr Hussain Kazmi
- Date: 2026-10-07
- Decision: Approve one controlled technical recovery of T1 only. Do not resume the full E5 campaign.

## What happened

On 2026-10-06 the first T1 tuning run stopped during FedNova at learning rate 0.1, round 12. Eleven rounds were saved. Round 11 validation loss was 9.746182249449104e+16. The client-update check raised `endpoint is not the start plus delta_i in float32` because a non-finite endpoint compares unequal under exact equality. That exception is inside the shared momentum-SGD client update, before that round's server step. It does not show that the FedNova server formula is wrong. The FedNova update was not changed.

Five other T1 cells finished 30 finite rounds. Those scores are diagnostics only. They are not a learning-rate choice. Section 4.5 still says that a missing, non-finite, or incomplete run stops the block, and the rate is not selected from the runs that remain. No T1 lock exists. No decision cell has started.

Those five cells also loaded the scaler, indices, and partition once for the block. The accepted preregistration requires each of the six runs to reload and rehash those artifacts. That lineage is not certified. The 2026-10-06 files stay in place as audit evidence.

## What this approval allows

1. Rerun all six T1 tuning cells under unchanged rates 0.001, 0.01, and 0.1, seed 10007, FedAvg before FedNova, and the same validation Final macro-F1 scoring.
2. Keep the 2026-10-06 attempt as audit evidence. Delete nothing. Leave `q1_e5_results/tuning/cells/` unchanged.
3. Write the rerun only under `q1_e5_results/tuning/attempt2/`. Reload the scaler, indices, and partition immediately before each cell.
4. After the updated integrity suite and a fresh no-training preflight both pass, the recovery command is `py -3.12 -u run_q1_e5.py tune-t1-attempt2`.
5. If all six attempt-2 cells are finite, lock T1's learning rate from those six cells, then stop for a check. The single smoke cell is T1, FedAvg–momentum-SGD, seed 127, and it is a later command. This rerun does not start it.
6. If FedNova at 0.1 is non-finite again, stop. Do not select a rate from the remaining cells. Prepare a transparent preregistration amendment for that outcome. This file is not that amendment.

## What stays closed until T1 passes this gate

T2, T5, and T6 tuning stay closed. Decision cells stay closed. `wave` and `resume` stay closed. `py -3.12 -u run_q1_e5.py tune` is not the recovery command. The accepted preregistration is unchanged.

The CPU integrity suite passed 36 tests on 2026-10-07, exit 0, in 2.511 seconds. The post-code no-training preflight passed at 2026-10-07T03:59:31Z: status PASS, `gpu_training` false, `scalers_written` false, `partition_files_written` false. CIC-IoT2023 matched (309 read, 0 excluded). N-BaIoT matched (89 read, 1 excluded). Its byte copy is `q1_e5_results/incidents/2026-10-06-t1-fednova-lr0.1/Q1_E5_PREFLIGHT_2026-10-07T03-59-31Z.json`. The earlier 2026-10-07T03:25:10Z report remains the incident copy from before this code.

## Outcome

The authorized command was run. It started at 2026-10-07T04:09:13Z and stopped at 2026-10-07T11:12:14Z. Five attempt-2 cells finished and matched the preserved histories. FedNova at 0.1 failed again during round 12 with `RuntimeError: delta_i is non-finite`. No T1 learning rate was locked. The draft amendment is `q1_e5_results/Q1_E5_T1_NONFINITE_AMENDMENT.md`. It is not accepted. This approval does not hash that draft. T2, T5, T6, wave, resume, and every decision cell remain closed.

# Q1-E5 technical incident and repair

Incident date: 2026-10-06. Repair note written: 2026-10-07. No learning rate is locked. No decision cell has been started. No GPU command was run for this repair.

## Incident

The user-started tuning command `py -3.12 -u run_q1_e5.py tune` began at 2026-10-06T06:16:13Z, process id 112976. It stopped at 2026-10-06T13:26:25Z.

Five T1 cells completed 30 finite validation rounds. T1, FedNova, learning rate 0.1, is status `failed`. Only 11 rounds were saved. Round 11 validation loss was 9.746182249449104e+16 and its macro-F1 was 0.049974839837118. Round 12 raised:

`RuntimeError: endpoint is not the start plus delta_i in float32`

The saved cell is `q1_e5_results/tuning/cells/T1_FedNova_lr0.1_s10007.json`. There is no T1 lock file. T2, T5, and T6 did not start. `tune.pid` is absent. The same `tune` command still stops on this failed cell and does not replace it.

The five completed Final macro-F1 values are tuning diagnostics only. They are not results and they are not a learning-rate choice.

| Rate | FedAvg | FedNova |
|---|---|---|
| 0.001 | 0.4583327996126333 | 0.44398835718307417 |
| 0.01 | 0.4804099422091194 | 0.39112297919882344 |
| 0.1 | 0.49990466873723177 | failed in round 12 |

Section 4.5 of the accepted preregistration says that if any one of the six runs is missing, non-finite, or incomplete, stop, and do not select a rate from the runs that remain. This cell is incomplete. The 0.1 pair has no mean.

## What the exception shows

The exception is raised inside the shared momentum-SGD client update, before that round's server step. It does not by itself show that the FedNova server formula is wrong. The FedNova update was not changed.

The check that fired subtracts and adds in float64, then casts back to float32. Every finite float32 value is exact in float64, so that identity holds for large finite magnitudes. The same identity done entirely in float32 does not: a deterministic sample of 4096 pairs near 1e20 fails the float32 round trip and passes the float64 one. `np.array_equal` is false for NaN, and the finiteness test used to run after that equality test. A non-finite endpoint was therefore reported as a reconstruction failure. Round 11's loss, about 9.75e16, shows the iterate was already extreme. The next client step produced a non-finite endpoint.

## Audit problems

1. `q1_e5_results/Q1_E5_PREFLIGHT.json` was first written at 2026-10-06T03:29:02Z, before the final E5 tuning implementation. That stale report is preserved at `q1_e5_results/incidents/2026-10-06-t1-fednova-lr0.1/Q1_E5_PREFLIGHT_2026-10-06T03-29-02Z.json`. The user ran `py -3.12 -u run_q1_e5.py preflight` after the repair. The canonical report was rewritten at 2026-10-07T03:25:10Z with status PASS, `gpu_training` false, and `scalers_written` false. CIC-IoT2023 matched (309 read, 0 excluded). N-BaIoT matched (89 read, 1 excluded).

2. The accepted preregistration requires each of the six tuning runs to reload the scaler, the index arrays, and the partition arrays and to recompute their hashes. The 2026-10-06 run loaded those artifacts once per block and reused them for the six cells. Those five completed cells are not a contract-compliant lineage.

## Repair

All 2026-10-06 tuning cell files, histories, the progress log, and the tuning artifacts are left in place. Nothing from that attempt was deleted.

- `assert_client_endpoint` in `fl_local/e5_protocol.py` rejects a non-finite start, delta, or endpoint before the equality test, with the message `delta_i is non-finite`. Finite endpoints still have to satisfy the exact float64 identity. The FedNova server update is unchanged.
- `reload_tuning_cell` re-reads the scaler, index files, and partition arrays and recomputes their hashes. `load_cell_for_training` calls that reload immediately before the cell is trained or skipped. The training loop calls it inside the learning-rate loop and the arm loop. It does not train from a plan loaded for all six cells ahead of time. A failed or non-finite cell still stops the block and is not replaced. If the scaler, index, partition, or matrix hash changes between cells in a block, the block stops.
- The 2026-10-06 cells remain the one-load-per-block attempt. They are preserved, and they are not certified lineage.
- Attempt 2 writes only under `q1_e5_results/tuning/attempt2/`. It reruns the six T1 cells at rates 0.001, 0.01, and 0.1, FedAvg then FedNova, seed 10007, with the same validation Final macro-F1 scoring. The preserved cell, init, lock, and progress files are hashed before that rerun and checked after every attempt-2 cell. A failed or non-finite cell stops the block and does not write a lock. T2, T5, T6, the smoke cell, wave, and resume are not started.
- Plain `py -3.12 -u run_q1_e5.py tune` stops before it writes, because the preserved attempt has a closed cell. It does not resume that cell, replace it, or append to the preserved progress log.
- The updated CPU integrity suite, `py -3.12 -m unittest tools.test_q1_e5_integrity`, passed 36 tests on 2026-10-07, exit 0, in 2.511 seconds. An earlier run of that suite, before this attempt-2 path, passed 30 tests in 2.620 seconds.

## Approval recorded

On 2026-10-07 Dr Hussain Kazmi approved one controlled technical recovery of T1 only. Proposer: Sheryar Kiani. The record is `q1_e5_results/incidents/2026-10-06-t1-fednova-lr0.1/Q1_E5_T1_RERUN_APPROVAL.md`. This note does not hash that approval file. The accepted preregistration is unchanged.

The approval allows the six-cell T1 rerun under `tune-t1-attempt2` after this suite and a fresh no-training preflight. It does not resume the full E5 campaign.

The no-training preflight at 2026-10-07T03:25:10Z passed before the attempt-2 code. A byte copy is stored at `q1_e5_results/incidents/2026-10-06-t1-fednova-lr0.1/Q1_E5_PREFLIGHT_2026-10-07T03-25-10Z.json`.

The user then ran `py -3.12 -u run_q1_e5.py preflight` after the attempt-2 code. The canonical report was rewritten at 2026-10-07T03:59:31Z with status PASS, kind `no-training-preflight`, `gpu_training` false, `scalers_written` false, and `partition_files_written` false. CIC-IoT2023 matched (309 read, 0 excluded). N-BaIoT matched (89 read, 1 excluded: `nbaiot_data/extracted/demonstrate_structure.csv`). A byte copy is stored at `q1_e5_results/incidents/2026-10-06-t1-fednova-lr0.1/Q1_E5_PREFLIGHT_2026-10-07T03-59-31Z.json`.

## Recovery outcome

The user ran `py -3.12 -u run_q1_e5.py tune-t1-attempt2`. It started at 2026-10-07T04:09:13Z, process id 120264, and stopped at 2026-10-07T11:12:14Z.

Five attempt-2 cells completed 30 finite rounds. Each stored validation history is identical to the preserved 2026-10-06 history for the same arm and rate. The initial-parameter hash is `0491fc7dd205aa6e7c3127c252dee5671020c0748d730d99212fc6dcce4d9a5f` in both attempts. T1 FedNova at learning rate 0.1 is status `failed`, `finite` false, with 11 rounds saved. Its history is identical to the preserved history, including history SHA-256 `4d49879256bbeb998b856c5bd087c43c56837e1c6776e0eb873cf135a4a2f8d0`. Round 11 loss is `9.746182249449104e+16` and its macro-F1 is `0.049974839837118`. Round 12 raised `RuntimeError: delta_i is non-finite`. The cell started at 2026-10-07T10:43:52Z and finished at 2026-10-07T11:12:14Z.

No lock file was written. `tune.pid` is absent. The 2026-10-06 progress log still ends at the 2026-10-06T13:26:25Z failure, and the preserved cell still records the earlier message `endpoint is not the start plus delta_i in float32`. The accepted preregistration file still hashes to `CD20C7A61B3A41AD214CC51BF4221F7837604FD1236F75282FD07BE346EC6517`.

Section 4.5 stops the block. No learning rate is selected. The five completed scores stay diagnostics. The draft amendment is `q1_e5_results/Q1_E5_T1_NONFINITE_AMENDMENT.md`. It is not accepted. This note does not hash that draft.

## What remains unauthorized

Do not run `py -3.12 -u run_q1_e5.py tune`. Do not run `py -3.12 -u run_q1_e5.py tune-t1-attempt2` again. A second attempt-2 command would skip the five completed cells and stop on the failed cell. Do not run `wave` or `resume`. Do not start T2, T5, T6, the smoke cell, or any other decision cell. Do not select 0.001 or 0.01 from the cells that finished. No learning rate is locked. No decision cell has been started.

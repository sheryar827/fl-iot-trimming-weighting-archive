# Q1-E1b three-primary packet (WeightedFedTrimmedAvg, non-IID)

Generated UTC: 2026-08-15T06:31:25.096367+00:00
Builder: `Research Paper/_build_q1_e1b_three_primary_packet.py`

## Scope

- **Arm:** WeightedFedTrimmedAvg (size-weighted coordinate-wise trim)
- **Comparators:** published FedAvg, FedProx, FedTrimmedAvg; local UnweightedFedAvg (E1)
- **Primary metrics:** Final Acc, Final F1, AUC
- **α:** 1.0, 0.5, 0.1 only — **non-IID only**
- **Seeds:** 42, 123, 7 → **36** WT runs
- **Claim type:** descriptive only at n=3 (F3)
- **Lineage note:** WT produced with engine after 2026-08-12. **Shared-path re-gate STRONG PASS** (2026-08-15): N-BaIoT MLP FedAvg α=1.0 s42 `match_notebook` bit-identical to published + pre-engine probe backup (`tools/verify_nbaiot_mlp_fedavg_gate.py` exit 0). Scope: shared train/eval path + `aggregate_fedavg_np` only — not a re-proof of unweighted trim or UW aggregators.

## Seed-mean margins (pp): WT − baseline

| Dataset | Model | α | WT−FA F1 | WT−FT F1 | WT−UW F1 | flip FA | flip FT | flip UW | BR F1 | WT−BR F1 |
|---------|-------|---|----------:|----------:|----------:|:-------:|:-------:|:-------:|-------|----------:|
| N-BaIoT | MLP | 1.0 | -0.04 | -0.04 | +0.11 | FLIP | FLIP | agree | FedTrimmedAvg | -0.04 |
| N-BaIoT | MLP | 0.5 | -0.04 | -0.07 | -0.46 | FLIP | agree | FLIP | FedTrimmedAvg | -0.07 |
| N-BaIoT | MLP | 0.1 | +0.18 | -1.81 | -2.47 | agree | agree | agree | FedTrimmedAvg | -1.81 |
| N-BaIoT | CNN | 1.0 | +5.67 | -0.41 | -0.39 | agree | FLIP | FLIP | FedTrimmedAvg | -0.41 |
| N-BaIoT | CNN | 0.5 | +2.95 | +3.26 | +0.35 | agree | agree | agree | FedAvg | +2.95 |
| N-BaIoT | CNN | 0.1 | +1.30 | +0.77 | -1.28 | agree | agree | FLIP | FedTrimmedAvg | +0.77 |
| CIC-IoT2023 | MLP | 1.0 | +1.28 | +0.13 | +0.56 | agree | agree | agree | FedTrimmedAvg | +0.13 |
| CIC-IoT2023 | MLP | 0.5 | +5.93 | -0.42 | +0.84 | agree | agree | agree | FedTrimmedAvg | -0.42 |
| CIC-IoT2023 | MLP | 0.1 | +6.39 | -10.45 | -5.16 | agree | agree | agree | FedTrimmedAvg | -10.45 |
| CIC-IoT2023 | CNN | 1.0 | +0.60 | +0.19 | +0.27 | agree | agree | agree | FedTrimmedAvg | +0.19 |
| CIC-IoT2023 | CNN | 0.5 | +4.94 | -1.60 | -0.65 | agree | agree | FLIP | FedTrimmedAvg | -1.60 |
| CIC-IoT2023 | CNN | 0.1 | +2.37 | -11.86 | -2.39 | FLIP | agree | agree | FedTrimmedAvg | -11.86 |

### CIC α=0.1 note

Low Final F1 at α=0.1 under E1b is a **CIC** pattern in seed-means; N-BaIoT α=0.1 is not in the same range. Always label dataset.

## Files

- `q1_e1b_wt_three_primary_seed.csv`
- `q1_e1b_wt_three_primary_seed_mean.csv`
- `q1_e1b_wt_three_primary_summary.json`
- `Q1_E1b_Three_Primary_Packet.md`
- `SHA256SUMS.txt`

## Next

1. Gate STRONG PASS recorded (2026-08-15) — shared-path lineage OK.
2. ~~Draft/apply E1b contract amendment~~ **APPLIED 2026-08-15** (A–G + corrected C1; Sheryar Kiani / Dr Hussain Kazmi).
3. Descriptive 2×2 write-up only (no F3 significance at n=3).

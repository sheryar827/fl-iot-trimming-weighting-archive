# Q1-E1 three-primary reporting packet (non-IID)

Generated UTC: 2026-08-09T09:11:14.452192+00:00
Builder: `Research Paper/_build_q1_e1_three_primary_packet.py`

## Scope

- **Arm:** UnweightedFedAvg (local `match_notebook`)
- **Comparators:** published FedAvg, FedProx, FedTrimmedAvg
- **Margins:** vs **FedAvg** and vs **best rival** (argmax among FA/Prox/FT per endpoint); Prox and FT margins also tabulated
- **Primary metrics (§4):** Final Acc, Final macro-F1, AUC (mean macro-F1 over 30 rounds)
- **Secondary (seed CSV only):** UW Best Acc / Best F1 from histories
- **Blocks:** N-BaIoT MLP/CNN; CIC-IoT2023 MLP/CNN
- **α:** 1.0, 0.5, 0.1 only — **non-IID only; IID excluded from this packet**
- **Seeds:** 42, 123, 7 (n=3) → **36** UW runs

## IID policy

- This packet is **non-IID only**. No UnweightedFedAvg IID runs exist.
- Published FA/Prox/FT **IID** rows in `Corrected_Results_Packet.md` remain valid for those arms only; they are **not** UW comparators and must **not** be pooled into UW margins.
- Claiming a full grid **including IID** requires **12** UW-IID cells plus a dated amendment.

## Seed-mean margins (pp): three primaries vs FA / Prox / FT / best rival

Margins = 100 × (UW − baseline) on the [0,1] probability scale. Level = seed-mean (n=3).

### Table A — vs FedAvg and FedTrimmedAvg (Final Acc / F1 / AUC)

| Dataset | Model | α | UW−FA Acc | UW−FA F1 | UW−FA AUC | flip FA | UW−FT Acc | UW−FT F1 | UW−FT AUC | flip FT |
|---------|-------|---|----------:|---------:|----------:|:-------:|----------:|---------:|----------:|:-------:|
| N-BaIoT | MLP | 1.0 | -0.17 | -0.14 | -0.29 | agree | -0.16 | -0.15 | -0.58 | agree |
| N-BaIoT | MLP | 0.5 | +0.33 | +0.42 | +0.01 | agree | +0.32 | +0.39 | -0.24 | FLIP |
| N-BaIoT | MLP | 0.1 | +1.76 | +2.65 | +2.40 | agree | +0.80 | +0.67 | +0.46 | agree |
| N-BaIoT | CNN | 1.0 | +4.30 | +6.06 | -0.66 | FLIP | +0.01 | -0.02 | -1.96 | agree |
| N-BaIoT | CNN | 0.5 | +1.92 | +2.61 | -0.44 | FLIP | +2.26 | +2.91 | -0.58 | FLIP |
| N-BaIoT | CNN | 0.1 | +1.39 | +2.58 | -1.01 | FLIP | +1.73 | +2.05 | +0.48 | agree |
| CIC-IoT2023 | MLP | 1.0 | +0.16 | +0.73 | +0.76 | agree | +0.25 | -0.42 | -0.40 | agree |
| CIC-IoT2023 | MLP | 0.5 | +0.59 | +5.09 | +4.79 | agree | +0.19 | -1.25 | -1.45 | agree |
| CIC-IoT2023 | MLP | 0.1 | -2.31 | +11.55 | +11.24 | agree | -0.26 | -5.29 | -5.26 | agree |
| CIC-IoT2023 | CNN | 1.0 | +0.15 | +0.33 | +0.34 | agree | +0.11 | -0.08 | -0.17 | agree |
| CIC-IoT2023 | CNN | 0.5 | +0.72 | +5.58 | +4.82 | agree | +0.29 | -0.95 | -1.15 | agree |
| CIC-IoT2023 | CNN | 0.1 | -1.12 | +4.76 | +2.86 | agree | +0.96 | -9.47 | -6.72 | agree |

### Table B — vs FedProx and vs best rival (Final Acc / F1 / AUC)

| Dataset | Model | α | UW−Prox Acc | UW−Prox F1 | UW−Prox AUC | BR Acc | UW−BR Acc | BR F1 | UW−BR F1 | BR AUC | UW−BR AUC |
|---------|-------|---|------------:|-----------:|------------:|--------|----------:|-------|---------:|--------|----------:|
| N-BaIoT | MLP | 1.0 | +0.75 | +0.86 | +0.45 | FedAvg | -0.17 | FedTrimmedAvg | -0.15 | FedTrimmedAvg | -0.58 |
| N-BaIoT | MLP | 0.5 | +2.57 | +3.01 | +0.87 | FedTrimmedAvg | +0.32 | FedTrimmedAvg | +0.39 | FedTrimmedAvg | -0.24 |
| N-BaIoT | MLP | 0.1 | +6.84 | +8.78 | +4.62 | FedTrimmedAvg | +0.80 | FedTrimmedAvg | +0.67 | FedTrimmedAvg | +0.46 |
| N-BaIoT | CNN | 1.0 | +3.24 | +4.54 | +1.17 | FedTrimmedAvg | +0.01 | FedTrimmedAvg | -0.02 | FedTrimmedAvg | -1.96 |
| N-BaIoT | CNN | 0.5 | +5.70 | +7.40 | +0.20 | FedAvg | +1.92 | FedAvg | +2.61 | FedTrimmedAvg | -0.58 |
| N-BaIoT | CNN | 0.1 | +8.55 | +12.56 | +2.80 | FedAvg | +1.39 | FedTrimmedAvg | +2.05 | FedAvg | -1.01 |
| CIC-IoT2023 | MLP | 1.0 | +0.48 | +3.56 | +4.24 | FedAvg | +0.16 | FedTrimmedAvg | -0.42 | FedTrimmedAvg | -0.40 |
| CIC-IoT2023 | MLP | 0.5 | +0.78 | +4.51 | +5.36 | FedTrimmedAvg | +0.19 | FedTrimmedAvg | -1.25 | FedTrimmedAvg | -1.45 |
| CIC-IoT2023 | MLP | 0.1 | -2.40 | +5.63 | +8.52 | FedProx | -2.40 | FedTrimmedAvg | -5.29 | FedTrimmedAvg | -5.26 |
| CIC-IoT2023 | CNN | 1.0 | +0.70 | +7.03 | +7.73 | FedTrimmedAvg | +0.11 | FedTrimmedAvg | -0.08 | FedTrimmedAvg | -0.17 |
| CIC-IoT2023 | CNN | 0.5 | +1.07 | +8.33 | +8.20 | FedTrimmedAvg | +0.29 | FedTrimmedAvg | -0.95 | FedTrimmedAvg | -1.15 |
| CIC-IoT2023 | CNN | 0.1 | -0.67 | +9.82 | +8.31 | FedAvg | -1.12 | FedTrimmedAvg | -9.47 | FedTrimmedAvg | -6.72 |

**Flip columns** = Final F1 vs AUC sign disagreement for that comparator. Best rival (BR) = argmax among {FedAvg, FedProx, FedTrimmedAvg} on that endpoint separately.

### Example (contract best-rival check)

CIC MLP α=0.1 Final Acc: best rival = **FedProx**, UW−best rival = **-2.3962 pp** (FedProx is best when its mean Acc exceeds FA/FT).

## Final F1 / AUC sign-flip counts (do not mix units)

| Level | Comparators | Denominator | Flips | by_block (explicit zeros) | Highlight |
|-------|-------------|-------------|------:|---------------------------|-----------|
| Seed-level | FA+FT only | 72 | **14** | {'N-BaIoT MLP': 6, 'N-BaIoT CNN': 7, 'CIC-IoT2023 MLP': 1, 'CIC-IoT2023 CNN': 0} | CIC CNN 0/18 |
| Seed-mean | FA+FT only | 24 | **5** | {'N-BaIoT MLP': 1, 'N-BaIoT CNN': 4, 'CIC-IoT2023 MLP': 0, 'CIC-IoT2023 CNN': 0} | N-BaIoT CNN 4/6 |
| Seed-level | FA+Prox+FT | 108 | **19** | {'N-BaIoT MLP': 7, 'N-BaIoT CNN': 10, 'CIC-IoT2023 MLP': 1, 'CIC-IoT2023 CNN': 1} | (incl. Prox) |
| Seed-mean | FA+Prox+FT | 36 | **5** | {'N-BaIoT MLP': 1, 'N-BaIoT CNN': 4, 'CIC-IoT2023 MLP': 0, 'CIC-IoT2023 CNN': 0} | (incl. Prox) |

## F3 / C8

- n=3: no multiplicity-corrected significance claims (F3).
- Descriptive means, margins, labelled sign counts **only**.
- **No** UW inferential tests under this freeze (no Welch, paired t, Holm/BH, TOST, or exploratory UW p-values). Any future UW tests require a **separate** dated amendment that freezes contrasts, endpoints, and family size first.
- C8 frozen at 0/36 vs 7/72.

## Files

- `q1_e1_uw_three_primary_seed.csv`
- `q1_e1_uw_three_primary_seed_mean.csv`
- `q1_e1_uw_f1_auc_flips_seed_level.csv`
- `q1_e1_uw_f1_auc_flips_seed_mean.csv`
- `q1_e1_uw_three_primary_summary.json`
- `Q1_E1_Three_Primary_Packet.md`
- `Q1_E1_AMENDMENT_DRAFT_UW_family_and_status.md`
- `Q1_E1_AMENDMENT_DRAFT_UW_family_and_status.TEMPLATE.md`
- `SHA256SUMS.txt`

## Next

1. Verify amendment draft is apply-ready (descriptive-only; exact A–G contract diffs in-canonical-file, including Number source).
2. Complete **who** (named proposer/approver/date) on the amendment draft and accept it.
3. Apply exact A–G to `Q1_CONTRIBUTION_CONTRACT.md` (E must be in-contract; Number source must include `q1_e1_results/reporting/`).
4. Only then plan Q1-E1b.

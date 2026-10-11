# CIC confirmatory 2×2 packet (`scale_on=train`)

Generated UTC: 2026-09-21T07:54:50.954490+00:00
Builder: `_build_q1_cic_conf_packet.py`

## Scope

- Dataset: **CIC-IoT2023** only
- Scaler: **split then train-only** (`scale_on=train`, cache `27bfed227ef77fab`)
- Waves 1–2: 2 models × α∈{0.5,0.1} × 4 arms × 3 seeds = **48** cells
- Wave 0 smoke (CNN FedAvg α=1.0 s42) on disk; **Wave 3 not run** — no α=1.0 ranking claim
- **Not** comparable to `scale_on=full` E1/E1b/published numbers in one test family
- Descriptive only (n=3, F3); frozen scoring rule in prereg r3/r4/r5
- Column **AUC** = **mean of 30 round-wise macro-F1** values, **not** ROC-AUC

## Hypothesis board (displayed to 5 decimals; **scoring used unrounded stored Final F1**)

| Model | α | FA | UW | FT | WT | HC1 | Htrim | Hweight |
|-------|---|----|----|----|----|-----|-------|---------|
| MLP | 0.5 | 0.61594 | 0.66156 | **0.67497** | 0.66704 | yes | yes | yes |
| MLP | 0.1 | 0.42598 | 0.55188 | **0.59686** | 0.50295 | yes | yes | yes |
| CNN | 0.5 | 0.57053 | 0.63050 | **0.63993** | 0.63511 | yes | yes | yes |
| CNN | 0.1 | 0.40552 | 0.46235 | **0.52651** | 0.43521 | yes | yes | yes |

HC1 = FT strictly highest **unrounded** mean. Htrim = mean(FT−UW)>0 and mean(WT−FA)>0. Hweight = mean(FA−UW)<0 and mean(WT−FT)<0. Exact equality of stored means = tie = not supported. None tied.

## Paired seed differences (12 rows: model × α × seed)

CSV stores contrasts formatted to **16 decimal digits** (not IEEE full precision). Display below is 4 decimals.

| Model | α | Seed | FT−UW | WT−FA | FA−UW | WT−FT |
|-------|---|------|------:|------:|------:|------:|
| MLP | 0.5 | 42 | +0.0256 | +0.0535 | -0.0336 | -0.0056 |
| MLP | 0.5 | 123 | +0.0067 | +0.0906 | -0.0943 | -0.0103 |
| MLP | 0.5 | 7 | +0.0080 | +0.0091 | -0.0090 | -0.0078 |
| MLP | 0.1 | 42 | +0.0245 | -0.0021 | -0.0826 | -0.1093 |
| MLP | 0.1 | 123 | +0.0623 | +0.1132 | -0.1635 | -0.1126 |
| MLP | 0.1 | 7 | +0.0481 | +0.1198 | -0.1316 | -0.0598 |
| CNN | 0.5 | 42 | +0.0288 | +0.0820 | -0.0549 | -0.0017 |
| CNN | 0.5 | 123 | +0.0020 | +0.1068 | -0.1181 | -0.0133 |
| CNN | 0.5 | 7 | -0.0025 | +0.0050 | -0.0069 | +0.0005 |
| CNN | 0.1 | 42 | +0.0516 | -0.0118 | -0.0048 | -0.0682 |
| CNN | 0.1 | 123 | +0.1187 | +0.0905 | -0.0506 | -0.0787 |
| CNN | 0.1 | 7 | +0.0222 | +0.0104 | -0.1151 | -0.1270 |

## Sign counts (of 3 seeds) for each contrast

| Model | α | FT−UW > 0 | WT−FA > 0 | FA−UW < 0 | WT−FT < 0 | both trim+ | both weight− |
|-------|---|-----------|-----------|-----------|-----------|------------|--------------|
| MLP | 0.5 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| MLP | 0.1 | 3/3 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| CNN | 0.5 | 2/3 | 3/3 | 3/3 | 2/3 | 2/3 | 2/3 |
| CNN | 0.1 | 3/3 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 |

## Files

- `q1_cic_conf_seed.csv` (48 rows)
- `q1_cic_conf_seed_mean.csv`
- `q1_cic_conf_hypothesis_score.csv`
- `q1_cic_conf_paired_contrasts_seed.csv` (12 rows)
- `q1_cic_conf_summary.json`
- raw: `../csv/q1_cic_conf_raw.csv`
- SHA256SUMS seals the **current** cache NPZ/meta and written histories, **not** an independent rebuild from the ~309 CIC source CSVs (those content hashes are not recorded)

## Next (order)

1. ~~Accept confirmatory amendment~~ **APPLIED 2026-09-21** (live contract).
2. **Next scientific GPU:** preregister **Q1-E2** (9-device N-BaIoT, `scale_on=train`). Wave 3 remains **optional** (α=1.0 control, no HC1), not the default next step.

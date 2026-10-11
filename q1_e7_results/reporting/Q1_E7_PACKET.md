# Q1-E7: final-round class decomposition on CIC-IoT2023

Generated UTC: 2026-09-26T15:25:00.844907+00:00

## Executive summary

- The preregistered GPU grid is complete: **24/24 cells** (2 models × 4 arms × 3 seeds), all at Dirichlet label-skew **alpha=0.1**, `scale_on=train`, and final round 30.
- Every committed Final accuracy and Final macro-F1 reproduces its frozen CIC confirmatory reference within the preregistered `1e-12` tolerance; the observed maximum absolute error is **0.000e+00**.
- The class packet is complete: **192 seed-level rows**, **64 seed-mean rows**, and **128 paired class-contrast rows** covering F1 and recall for every model, class, contrast, and seed.
- The macro-F1 contribution identity passes for all **24** model × seed × contrast groups; maximum absolute error is **1.388e-16** (limit `1e-12`).
- This is a **descriptive decomposition**, not independent validation or an inferential analysis. It does not establish a general rare-class mechanism.

## Frozen scope and provenance

| Item | Value |
|---|---|
| Dataset | CIC-IoT2023 |
| Cache/scaler | `27bfed227ef77fab`; `scale_on=train` |
| Models | MLP, CNN |
| Arms | FA = size-weighted/untrimmed; UW = unweighted/untrimmed; FT = unweighted/trimmed; WT = size-weighted/trimmed |
| Dirichlet label-skew alpha | 0.1 |
| Coordinate trim | Configured fraction 0.1; one lowest and one highest of nine client coordinates removed (`beta_eff=1/9`) |
| Seeds | 42, 123, 7 |
| Test records | 1,206,424 |
| Classes | 8, fixed label order |
| Measured training time | 52.03 GPU-hours across the 24 reruns |
| First/last committed cell UTC | 2026-09-23T15:45:10.254822+00:00 / 2026-09-26T14:07:11.757302+00:00 |
| Analysis | Final-round per-class precision, recall, F1; paired F1/recall contrasts |
| Inference | None: no p-values, confidence intervals, equivalence tests, or test-row bootstrap |

## Aggregate reproduction check: three-seed mean Final macro-F1

These means are a reconciliation of the reruns, not a new independent aggregate result.

| Model | FA mean | UW mean | FT mean | WT mean |
|---|---:|---:|---:|---:|
| MLP | 0.42598 | 0.55188 | 0.59686 | 0.50295 |
| CNN | 0.40552 | 0.46235 | 0.52651 | 0.43521 |

## Fixed class supports

| ID | Class | Support | Test share |
|---:|---|---:|---:|
| 0 | benign | 16,000 | 1.3262% |
| 1 | bruteforce | 2,613 | 0.2166% |
| 2 | ddos | 700,977 | 58.1037% |
| 3 | dos | 152,000 | 12.5992% |
| 4 | mirai | 301,416 | 24.9843% |
| 5 | recon | 16,452 | 1.3637% |
| 6 | spoofing | 12,000 | 0.9947% |
| 7 | web | 4,966 | 0.4116% |
|  | **Total** | **1,206,424** | **100.0000%** |

## Complete paired class-contrast matrix

The table below is exhaustive. Mean deltas use the three matched training seeds. Signs are ordered **42, 123, 7**. Seed deltas are stored at round-trip precision, with sample standard deviations and positive/negative/zero counts, in `q1_e7_paired_class_contrasts.csv`; this Markdown table is rounded for display only. Deltas are absolute 0–1 metric units; multiply by 100 for percentage points.

| Model | Class (support) | Contrast | Mean ΔF1 | F1 signs | Mean ΔRecall | Recall signs |
|---|---|---|---:|:---:|---:|:---:|
| MLP | benign (16,000) | FT−UW | +0.22672 | `+++` | +0.22792 | `+-+` |
| MLP | benign (16,000) | WT−FA | +0.11260 | `--+` | +0.12896 | `-++` |
| MLP | benign (16,000) | FA−UW | -0.18545 | `-+-` | -0.17652 | `---` |
| MLP | benign (16,000) | WT−FT | -0.29957 | `---` | -0.27548 | `-+-` |
| MLP | bruteforce (2,613) | FT−UW | -0.07442 | `-+-` | +0.16278 | `+++` |
| MLP | bruteforce (2,613) | WT−FA | +0.04122 | `--+` | +0.35999 | `+++` |
| MLP | bruteforce (2,613) | FA−UW | -0.14351 | `+--` | -0.16750 | `---` |
| MLP | bruteforce (2,613) | WT−FT | -0.02787 | `+--` | +0.02972 | `-++` |
| MLP | ddos (700,977) | FT−UW | -0.00437 | `+-+` | -0.03010 | `+-+` |
| MLP | ddos (700,977) | WT−FA | -0.05040 | `-+-` | -0.14865 | `+--` |
| MLP | ddos (700,977) | FA−UW | +0.04619 | `+-+` | +0.13837 | `+++` |
| MLP | ddos (700,977) | WT−FT | +0.00015 | `-+-` | +0.01982 | `++-` |
| MLP | dos (152,000) | FT−UW | +0.07919 | `-+-` | +0.11899 | `-++` |
| MLP | dos (152,000) | WT−FA | +0.31060 | `-++` | +0.43527 | `-++` |
| MLP | dos (152,000) | FA−UW | -0.28773 | `---` | -0.41079 | `---` |
| MLP | dos (152,000) | WT−FT | -0.05632 | `---` | -0.09451 | `--+` |
| MLP | mirai (301,416) | FT−UW | -0.00051 | `---` | -0.00055 | `-+-` |
| MLP | mirai (301,416) | WT−FA | -0.00023 | `-+-` | -0.00037 | `-+-` |
| MLP | mirai (301,416) | FA−UW | -0.00031 | `+--` | +0.00039 | `+-+` |
| MLP | mirai (301,416) | WT−FT | -0.00004 | `+--` | +0.00057 | `+-+` |
| MLP | recon (16,452) | FT−UW | +0.03820 | `-++` | -0.07201 | `-+-` |
| MLP | recon (16,452) | WT−FA | +0.07839 | `+++` | +0.04879 | `+++` |
| MLP | recon (16,452) | FA−UW | -0.16910 | `--+` | -0.25952 | `---` |
| MLP | recon (16,452) | WT−FT | -0.12891 | `---` | -0.13873 | `---` |
| MLP | spoofing (12,000) | FT−UW | +0.07068 | `+++` | +0.00811 | `-+-` |
| MLP | spoofing (12,000) | WT−FA | +0.15544 | `+++` | -0.02939 | `++-` |
| MLP | spoofing (12,000) | FA−UW | -0.25787 | `---` | -0.01083 | `+-+` |
| MLP | spoofing (12,000) | WT−FT | -0.17310 | `---` | -0.04833 | `+-+` |
| MLP | web (4,966) | FT−UW | +0.02433 | `+--` | +0.03423 | `++-` |
| MLP | web (4,966) | WT−FA | -0.03185 | `+--` | -0.16110 | `+--` |
| MLP | web (4,966) | FA−UW | -0.00942 | `--+` | +0.13881 | `-++` |
| MLP | web (4,966) | WT−FT | -0.06559 | `---` | -0.05652 | `---` |
| CNN | benign (16,000) | FT−UW | +0.26837 | `+++` | +0.16131 | `+-+` |
| CNN | benign (16,000) | WT−FA | +0.00230 | `--+` | +0.04981 | `-++` |
| CNN | benign (16,000) | FA−UW | -0.08505 | `-+-` | -0.09548 | `---` |
| CNN | benign (16,000) | WT−FT | -0.35112 | `---` | -0.20698 | `-+-` |
| CNN | bruteforce (2,613) | FT−UW | -0.02389 | `-+-` | +0.16801 | `-++` |
| CNN | bruteforce (2,613) | WT−FA | +0.12018 | `+++` | +0.45261 | `-++` |
| CNN | bruteforce (2,613) | FA−UW | -0.13892 | `+--` | -0.30859 | `---` |
| CNN | bruteforce (2,613) | WT−FT | +0.00516 | `+--` | -0.02398 | `--+` |
| CNN | ddos (700,977) | FT−UW | -0.04475 | `+--` | -0.07729 | `+--` |
| CNN | ddos (700,977) | WT−FA | -0.10969 | `+--` | -0.19827 | `+--` |
| CNN | ddos (700,977) | FA−UW | -0.00124 | `--+` | +0.02115 | `-++` |
| CNN | ddos (700,977) | WT−FT | -0.06618 | `---` | -0.09983 | `+--` |
| CNN | dos (152,000) | FT−UW | -0.03239 | `-++` | +0.08945 | `-++` |
| CNN | dos (152,000) | WT−FA | +0.24965 | `-++` | +0.36923 | `-++` |
| CNN | dos (152,000) | FA−UW | -0.22282 | `+--` | -0.15661 | `+--` |
| CNN | dos (152,000) | WT−FT | +0.05922 | `-+-` | +0.12317 | `-++` |
| CNN | mirai (301,416) | FT−UW | +0.00243 | `++-` | -0.00103 | `+--` |
| CNN | mirai (301,416) | WT−FA | +0.00669 | `-++` | -0.00023 | `-+-` |
| CNN | mirai (301,416) | FA−UW | -0.00559 | `+--` | +0.00041 | `+-+` |
| CNN | mirai (301,416) | WT−FT | -0.00133 | `---` | +0.00120 | `-++` |
| CNN | recon (16,452) | FT−UW | +0.15702 | `++-` | +0.05774 | `++-` |
| CNN | recon (16,452) | WT−FA | +0.05010 | `++-` | +0.03629 | `++-` |
| CNN | recon (16,452) | FA−UW | -0.06501 | `---` | -0.18152 | `---` |
| CNN | recon (16,452) | WT−FT | -0.17193 | `---` | -0.20297 | `---` |
| CNN | spoofing (12,000) | FT−UW | +0.12133 | `+++` | -0.01775 | `-+-` |
| CNN | spoofing (12,000) | WT−FA | +0.05918 | `+++` | -0.14272 | `-+-` |
| CNN | spoofing (12,000) | FA−UW | -0.06504 | `---` | +0.08647 | `+-+` |
| CNN | spoofing (12,000) | WT−FT | -0.12718 | `--+` | -0.03850 | `+--` |
| CNN | web (4,966) | FT−UW | +0.06521 | `++0` | +0.07639 | `++0` |
| CNN | web (4,966) | WT−FA | -0.14088 | `+--` | -0.30440 | `+--` |
| CNN | web (4,966) | FA−UW | +0.12902 | `-++` | +0.29373 | `-++` |
| CNN | web (4,966) | WT−FT | -0.07707 | `--0` | -0.08706 | `--0` |

## Direction counts across all eight named classes

These counts prevent selective emphasis on favorable classes. `Mixed/tied` includes any class whose three seed deltas are not all strictly positive or all strictly negative.

| Model | Metric | Contrast | All + | All − | Mixed/tied |
|---|---|---|---:|---:|---:|
| MLP | F1 | FT_minus_UW | 2 | 1 | 5 |
| MLP | F1 | WT_minus_FA | 2 | 0 | 6 |
| MLP | F1 | FA_minus_UW | 0 | 2 | 6 |
| MLP | F1 | WT_minus_FT | 0 | 5 | 3 |
| MLP | Recall | FT_minus_UW | 1 | 0 | 7 |
| MLP | Recall | WT_minus_FA | 2 | 0 | 6 |
| MLP | Recall | FA_minus_UW | 1 | 4 | 3 |
| MLP | Recall | WT_minus_FT | 0 | 2 | 6 |
| CNN | F1 | FT_minus_UW | 2 | 0 | 6 |
| CNN | F1 | WT_minus_FA | 2 | 0 | 6 |
| CNN | F1 | FA_minus_UW | 0 | 2 | 6 |
| CNN | F1 | WT_minus_FT | 0 | 4 | 4 |
| CNN | Recall | FT_minus_UW | 0 | 0 | 8 |
| CNN | Recall | WT_minus_FA | 0 | 0 | 8 |
| CNN | Recall | FA_minus_UW | 0 | 3 | 5 |
| CNN | Recall | WT_minus_FT | 0 | 1 | 7 |

## Macro-F1 contribution identity

For every row, `direct Δ macro-F1 = (1/8) × Σ class ΔF1`. Class contributions are arithmetic identities, not causal attributions.

| Model | Seed | Contrast | Direct Δmacro-F1 | Mean class ΔF1 | Absolute error |
|---|---:|---|---:|---:|---:|
| MLP | 42 | FT_minus_UW | +0.02453579 | +0.02453579 | 2.776e-17 |
| MLP | 42 | WT_minus_FA | -0.00213875 | -0.00213875 | 5.508e-17 |
| MLP | 42 | FA_minus_UW | -0.08258333 | -0.08258333 | 4.163e-17 |
| MLP | 42 | WT_minus_FT | -0.10925787 | -0.10925787 | 2.776e-17 |
| MLP | 123 | FT_minus_UW | +0.06231996 | +0.06231996 | 6.939e-17 |
| MLP | 123 | WT_minus_FA | +0.11322779 | +0.11322779 | 1.388e-16 |
| MLP | 123 | FA_minus_UW | -0.16351991 | -0.16351991 | 2.776e-17 |
| MLP | 123 | WT_minus_FT | -0.11261208 | -0.11261208 | 1.110e-16 |
| MLP | 7 | FT_minus_UW | +0.04807793 | +0.04807793 | 4.163e-17 |
| MLP | 7 | WT_minus_FA | +0.11982441 | +0.11982441 | 4.163e-17 |
| MLP | 7 | FA_minus_UW | -0.13159359 | -0.13159359 | 0.000e+00 |
| MLP | 7 | WT_minus_FT | -0.05984711 | -0.05984711 | 7.633e-17 |
| CNN | 42 | FT_minus_UW | +0.05158335 | +0.05158335 | 1.388e-17 |
| CNN | 42 | WT_minus_FA | -0.01181319 | -0.01181319 | 3.123e-17 |
| CNN | 42 | FA_minus_UW | -0.00481249 | -0.00481249 | 4.163e-17 |
| CNN | 42 | WT_minus_FT | -0.06820903 | -0.06820903 | 8.327e-17 |
| CNN | 123 | FT_minus_UW | +0.11867525 | +0.11867525 | 2.776e-17 |
| CNN | 123 | WT_minus_FA | +0.09051446 | +0.09051446 | 4.163e-17 |
| CNN | 123 | FA_minus_UW | -0.05056977 | -0.05056977 | 2.776e-17 |
| CNN | 123 | WT_minus_FT | -0.07873056 | -0.07873056 | 9.714e-17 |
| CNN | 7 | FT_minus_UW | +0.02224250 | +0.02224250 | 8.327e-17 |
| CNN | 7 | WT_minus_FA | +0.01037780 | +0.01037780 | 8.674e-17 |
| CNN | 7 | FA_minus_UW | -0.11510786 | -0.11510786 | 5.551e-17 |
| CNN | 7 | WT_minus_FT | -0.12697256 | -0.12697256 | 1.110e-16 |

## Interpretation boundary

Permitted after named completion acceptance and contract update: exact observations about a named class, model, contrast, support, and all three seed deltas, accompanied by the complete matrix.

As an added project-governance safeguard, bounded class observations remain on hold until that named completion acceptance and contract update.

Not permitted: ‘trimming protects rare classes’ as a general mechanism; statistical significance; proof of the Q1-E6 conceptual explanation; Byzantine robustness; unseen-device or deployment claims; or selective reporting of favorable classes.

This packet uses the already-revisited pooled CIC test set. It is a preregistered descriptive final-round decomposition, not independent external validation. It must not be mixed with `scale_on=full` E1/E1b results.

## Files and status

- `q1_e7_per_class_seed.csv`: 192 seed-level class rows.
- `q1_e7_per_class_seed_mean.csv`: 64 model × arm × class summaries (mean and sample SD, n=3).
- `q1_e7_paired_class_contrasts.csv`: 128 exhaustive F1/recall contrast rows.
- `Q1_E7_PROVENANCE.json`: scope, identities, validation maxima, and manifest boundary.
- `SHA256SUMS.txt`: relative-path SHA-256 manifest for the frozen cache/meta, code, committed artifacts, and this packet.
- The preregistration footer records the historical pre-run state and is intentionally not rewritten after results.

**Status:** Q1-E7 is computationally complete and this reporting packet is built. The live contract must remain unchanged until a named proposer, approver, and acceptance date approve the completion amendment. C5 and C8 remain unchanged; Q1-E8 and CIC Wave 3 remain deferred.

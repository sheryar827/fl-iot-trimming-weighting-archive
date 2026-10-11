# Q1-E4 train-scaler seed-expansion reporting packet

**Primary decision sample:** seven prospective seeds {127, 131, 137, 139, 149, 151, 157}.  
**Combined estimate:** ten seeds; data-informed descriptive estimate only.  
**Endpoint:** unrounded Final macro-F1 at round 30.  

## Prospective paired-t/Holm family

Exactly 16 comparisons were evaluated. Percentage-point values below are Final macro-F1 differences. Holm adjustment is across all 16 comparisons.

| Block | Contrast | Mean pp | 95% CI pp | Signs +/0/- | Raw p | Holm p | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| T1 | FT-UW | 0.3714 | [-4.2080, 4.9507] | 5/0/2 | 0.849257 | 1 | inconclusive |
| T1 | WT-FA | 7.4325 | [2.6661, 12.1990] | 7/0/0 | 0.00880574 | 0.12328 | inconclusive |
| T1 | FA-UW | -12.2101 | [-19.7262, -4.6941] | 1/0/6 | 0.00732394 | 0.109859 | inconclusive |
| T1 | WT-FT | -5.1490 | [-13.1925, 2.8946] | 2/0/5 | 0.168306 | 1 | inconclusive |
| T2 | FT-UW | -1.5393 | [-4.7906, 1.7121] | 3/0/4 | 0.290708 | 1 | inconclusive |
| T2 | WT-FA | 6.3105 | [1.1477, 11.4734] | 6/0/1 | 0.024293 | 0.315809 | inconclusive |
| T2 | FA-UW | -10.7252 | [-15.9527, -5.4977] | 0/0/7 | 0.00240333 | 0.0384533 | supported |
| T2 | WT-FT | -2.8754 | [-7.8214, 2.0706] | 2/0/5 | 0.20472 | 1 | inconclusive |
| T5 | FT-UW | -0.0201 | [-2.4796, 2.4394] | 4/0/3 | 0.984693 | 1 | inconclusive |
| T5 | WT-FA | -1.7038 | [-3.2609, -0.1466] | 0/0/7 | 0.0366684 | 0.440021 | inconclusive |
| T5 | FA-UW | 1.6414 | [-0.7365, 4.0193] | 5/0/2 | 0.142178 | 1 | inconclusive |
| T5 | WT-FT | -0.0423 | [-1.3127, 1.2281] | 4/0/3 | 0.937724 | 1 | inconclusive |
| T6 | FT-UW | 0.7091 | [-3.2827, 4.7008] | 6/0/1 | 0.679 | 1 | inconclusive |
| T6 | WT-FA | 1.2596 | [-4.7011, 7.2203] | 2/0/5 | 0.623612 | 1 | inconclusive |
| T6 | FA-UW | -0.4904 | [-5.7321, 4.7514] | 3/0/4 | 0.826545 | 1 | inconclusive |
| T6 | WT-FT | 0.0601 | [-4.2701, 4.3904] | 2/0/5 | 0.974004 | 1 | inconclusive |

## Block-level component rules

| Block | Htrim | Hweight-hurt |
|---|---|---|
| T1 | not_supported | not_supported |
| T2 | not_supported | not_supported |
| T5 | not_supported | not_supported |
| T6 | not_supported | not_supported |

## Combined ten-seed estimates

These rows are combined, data-informed estimates. They have no p-value or decision role and cannot override the seven-new-seed Holm decisions.

| Block | Contrast | Mean pp | 95% CI pp | Signs +/0/- |
|---|---|---:|---:|---:|
| T1 | FT-UW | 1.6093 | [-1.6788, 4.8974] | 8/0/2 |
| T1 | WT-FA | 7.5119 | [3.7146, 11.3093] | 9/0/1 |
| T1 | FA-UW | -12.3241 | [-17.2676, -7.3805] | 1/0/9 |
| T1 | WT-FT | -6.4215 | [-11.8016, -1.0413] | 2/0/8 |
| T2 | FT-UW | 0.8475 | [-2.9673, 4.6624] | 6/0/4 |
| T2 | WT-FA | 5.3082 | [1.4018, 9.2145] | 8/0/2 |
| T2 | FA-UW | -9.2125 | [-13.3871, -5.0380] | 0/0/10 |
| T2 | WT-FT | -4.7519 | [-8.6947, -0.8091] | 2/0/8 |
| T5 | FT-UW | -0.2826 | [-1.8862, 1.3209] | 4/0/6 |
| T5 | WT-FA | -1.0027 | [-2.3599, 0.3545] | 2/0/8 |
| T5 | FA-UW | 1.4957 | [-0.2159, 3.2074] | 7/0/3 |
| T5 | WT-FT | 0.7757 | [-0.5831, 2.1346] | 7/0/3 |
| T6 | FT-UW | -0.6870 | [-4.0253, 2.6514] | 6/0/4 |
| T6 | WT-FA | 1.5433 | [-2.5146, 5.6013] | 4/0/6 |
| T6 | FA-UW | -1.0697 | [-4.9103, 2.7709] | 4/0/6 |
| T6 | WT-FT | 1.1606 | [-1.9660, 4.2871] | 5/0/5 |

## Secondary descriptive interaction

Interaction = `(WT-FA) - (FT-UW)`. It is excluded from the Holm family and has no decision role.

| Sample | Block | Mean pp | 95% CI pp | Signs +/0/- |
|---|---|---:|---:|---:|
| 7 prospective seeds | T1 | 7.0612 | [0.3531, 13.7692] | 7/0/0 |
| 7 prospective seeds | T2 | 7.8498 | [1.8981, 13.8015] | 6/0/1 |
| 7 prospective seeds | T5 | -1.6837 | [-4.7734, 1.4060] | 3/0/4 |
| 7 prospective seeds | T6 | 0.5505 | [-8.9670, 10.0680] | 2/0/5 |
| 10 combined seeds | T1 | 5.9026 | [1.1290, 10.6763] | 9/0/1 |
| 10 combined seeds | T2 | 4.4606 | [-1.0309, 9.9522] | 6/0/4 |
| 10 combined seeds | T5 | -0.7200 | [-3.0769, 1.6369] | 5/0/5 |
| 10 combined seeds | T6 | 2.2303 | [-4.5132, 8.9738] | 4/0/6 |

## Interpretation boundaries

- Findings apply only to T1/T2/T5/T6 under the frozen train-scaler protocol.
- `WT-UW` is not reported as a one-factor effect.
- Inconclusive does not mean equivalent.
- N-BaIoT is participating-device pooled-holdout evidence, not unseen-device or deployment validation.
- No universal superiority, Byzantine robustness, or general rare-class-protection claim is authorized.
- The manifest seals processed caches and experiment artifacts; it does not hash the original CIC/N-BaIoT source CSV trees.

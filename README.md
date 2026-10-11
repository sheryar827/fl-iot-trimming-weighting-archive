# Reproducibility archive: trimming and client-size weighting

This archive accompanies the article *Trimming and Client-Size Weighting under Honest Label Heterogeneity: A Two-Factor Study in Federated IoT Botnet Detection* by Sheryar Kiani and Saad Farooq.

The article has not been submitted and has not been accepted. This archive is not the May 2026 three-arm snapshot.

## Push status

The result-packet tree is committed on this machine in `Research Paper/ieee_access_repro_bundle`. Publishing it needs one authenticated push from that directory:

```
git push -u origin main
```

The remote was created without an initial commit. Until the push succeeds, the remote can still be empty. The files listed below are in the local commit.

## What the local commit contains

- `manuscript/` holds the IEEE Access source, the PDF, the supplementary governance note, and the figures drawn from the stored results.
- `q1_cic_conf_results/`, `q1_e1_results/`, `q1_e1b_results/`, `q1_e2_results/`, `q1_e3_results/`, `q1_e4_results/`, `q1_e5_results/`, `q1_e6_results/`, and `q1_e7_results/` hold the written result packets, cell records, manifests, and the partition files that fit in git.
- `fl_local/` and `tools/` hold the local training and reporting code used to produce those packets.
- `processed_matrices/` holds the small metadata sidecars for the two processed feature matrices.

The Q1-E5 `tuning/` tree is omitted. It contains feasibility runs, not the decision cells, and it contains matrices above the git file limit.

## What is not here

The two processed feature matrices exceed GitHub's 100 MB file limit. They are identified by checksum and are not git blobs.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `cic_27bfed227ef77fab.npz` | 170,407,946 | `8f032df1760a895b2358b8ff2da71491b09820c7af8c097dd00465fd7b981b38` |
| `nbaiot_4ddb8b93b8821650.npz` | 243,513,210 | `c0a1e21cd6336d3a090e5feb9411758abbf8669dcca01bd476defdbcf8380854` |

Their metadata checksums are `e5e0ef795fff100af24f6813ca3a2a61442e386aa01bf92878a9cfc0b03cb3b4` and `0124ac4c793417d53bd062f4694e72ba903ed42beee8873d578adf57a3686b23`.

## How the matrices were built

CIC-IoT2023 is the public corpus from Neto et al., *Sensors*, 2023. N-BaIoT is the public corpus from Meidan et al., *IEEE Pervasive Computing*, 2018. Each source file is capped at 20,000 rows. A stratified split uses seed 42. A standard scaler is fit on the pooled training matrix only and then applied to train and test.

## Other public records

The earlier three-arm lineage is https://github.com/sheryar827/fl-iot-botnet-nbaiot-cic-iot2023. Its headline comparisons are not the results of this article.

The concept DOI https://doi.org/10.5281/zenodo.20456827 resolves to the immutable v1.0.0 record https://doi.org/10.5281/zenodo.20456828, published on 30 May 2026. That snapshot was not given a new version for this article.

## Environment

Python 3.12.10, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.8.0, SciPy 1.17.1, PyTorch 2.5.1+cu121, CUDA 12.1, Quadro P3200. The new figures use Matplotlib 3.10.8.

## Governance

Approval dates, the FedNova stop, and the distinction between the 42.066-hour elapsed sum and the 42.99-hour planning figure are in `manuscript/access_q1_supplement.pdf`. ORCID identifiers, funding, and conflict-of-interest statements were not supplied.

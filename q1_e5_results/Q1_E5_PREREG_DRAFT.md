# Q1-E5 preregistration draft — Adam median and momentum-SGD FedNova

**Status:** **ACCEPTED 2026-10-06** at **0/84** decision cells. No runner, integrity suite, preflight, cache change, or GPU cell has been started. StepNormAdam is not approved.  
**Draft date:** 2026-10-05  
**Revised:** 2026-10-05, after the second pre-approval audit, with a final reproducibility pass. Earlier revisions bind the sealed caches, keep CoordMedian on the Q1-E4 path, and freeze FedNova, generator lifetime, and tuning pairing. This pass binds `fl_local/e2_partition.py`, defines compact tuning indices, requires one shared SGD client update, saves the tuning scalers, and rechecks every invocation. It replaces the withdrawn StepNormAdam proposal.  
**Proposer:** Sheryar Kiani  
**Approver:** Dr Hussain Kazmi  
**Acceptance date:** 2026-10-06 (10-6-2026)  
**Presented text:** SHA-256 `C985EF651B23BC03987A17FD205BE314C8CCE9F13FE7C8C4E8B1CF8911E59D5F`, before this acceptance block was filled.  
**Parent evidence:** Q1-E4 is **Done (112/112 new cells)**, accepted 2026-10-05. This acceptance does not reopen it.

This document is the accepted Q1-E5 preregistration. The separate contract amendment is `Q1_E5_CONTRACT_AMENDMENT.md`. That amendment records this preregistration at **0/84** decision cells. No E5 cell has been run.

## 1. Question

This proposal has two separate comparisons.

1. Under the frozen Adam training, is the coordinate-wise trimmed mean higher or lower than the coordinate-wise median?
2. Under momentum SGD, is FedNova using the published momentum-SGD normalization higher or lower than FedAvg when both use that same solver and the same constant learning rate?

The second comparison is not a claim that these runs reproduce the original FedNova experiments. The models, datasets, batch size, epoch count, and weight decay differ from that paper. Only the local solver family and the normalizing formula are the published momentum-SGD rule.

Q1-E5 is an E4-informed extension. The blocks and seeds were chosen because the Q1-E4 results are already known. Its Holm family does not provide program-wide error control across Q1-E4 and Q1-E5. The two Holm adjustments are separate. A sentence that cites both families must say so.

The Q1-E4 boundary remains:

- one supported component only: T2, CIC-IoT2023 CNN, FedAvg − UnweightedFedAvg;
- the other 15 Q1-E4 comparisons inconclusive;
- trim-help not supported in T1, T2, T5, or T6;
- weight-hurt not supported in T1, T2, T5, or T6;
- inconclusive is not equivalence;
- no universal superiority, deployment, privacy, rare-class, or adversarial-robustness claim.

Q1-E5 must not be used to manufacture a more favorable trimming result. It does not rerun or add a *p*-value to the closed 16-comparison family.

## 2. Why StepNormAdam is not used

StepNormAdam was only a way to keep the local Adam trainer unchanged. The FedNova paper defines normalization for specified local-update structures, including vanilla SGD, momentum SGD, proximal SGD, and learning-rate schedules. The official implementation uses that SGD-family structure. It does not provide scalar normalization of Adam’s adaptive moments. Official code: [JYWa/FedNova](https://github.com/JYWa/FedNova), `distoptim/FedNova.py`. Paper: Wang, Liu, Liang, Joshi, and Poor, NeurIPS 2020.

A StepNormAdam result would not be that method. That campaign must not be run. SCAFFOLD is also outside this proposal.

## 3. Two matched protocols

### 3.1 Adam protocol, median only

CoordMedian uses the frozen E4 Adam protocol: `mode=match_notebook`, learning rate `1e-3`, PyTorch `2.5.1+cu121` Adam defaults (`betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0`, `amsgrad=False`), three local epochs, batch size 512, shuffle on, last batch kept, and proximal penalty 0. The partner is the sealed E4 FedTrimmedAvg row. No E4 arm is retrained.

Only the server aggregation changes. Local training and randomness stay on the E4 `match_notebook` path in Section 5.1, including `train_client_nb` and its `DataLoader` without an explicit generator. CoordMedian does not use the generators in Section 5.2.

With nine clients, the coordinate-wise median is the fifth ordered value at each parameter. It uses no sample-size weight.

### 3.2 Momentum-SGD protocol

FedNova and its matched FedAvg control use the same local solver:

| Setting | Frozen value |
|---|---|
| Local solver | momentum SGD |
| `momentum` | `0.9` |
| `dampening` | `0` |
| `nesterov` | `False` |
| proximal `mu` | `0` |
| server momentum `gmf` | `0` |
| `weight_decay` | `0` |
| local epochs | `3` |
| batch size | `512` |
| shuffle / last batch | shuffle on, last batch kept |
| learning-rate schedule | constant; Section 4 |

`weight_decay=0` matches this project. It is not the `1e-4` value in one official VGG command. Both SGD arms use `0`.

The learning rate is not copied from E4 and is not fixed at E3’s `1e-3`. The E3 preregistration states that `1e-3` is Adam-tuned and that SGD at that learning rate can fail to train. `1e-3` is only one candidate in Section 4.

At the start of every client, in every communication round, both SGD arms set the momentum buffer to zero, the cumulative update to zero, the normalizing counter to zero, and the step count to zero. A new optimizer is created for that client round. Momentum state does not carry across rounds or clients. This is the published momentum-SGD reset: the official `average` step zeroes the momentum buffer, cumulative update, counter, and normalizing vector before the next round.

Let \(w\) be the server parameters at the start of the round and \(w_i\) the parameters returned by client \(i\). With client step count \(\tau_i\) and \(\Delta_i=w_i-w\):

\[
c_0=0,\qquad c_k=0.9\,c_{k-1}+1.
\]

The coefficient vector is ordered from the last local step back to the first:

\[
a_i=(c_{\tau_i},c_{\tau_i-1},\ldots,c_1).
\]

\[
A_i=\lVert a_i\rVert_1=\sum_{k=1}^{\tau_i} c_k.
\]

\(A_i\) is that scalar. The server update divides by \(A_i\). It does not divide by the vector \(a_i\).

The weight is \(p_i=n_i/n\), in `float64`. The counts \(n_i\) and \(n\) are the rows actually trained in that run. In a tuning run, \(n_i\) is client \(i\)’s subtrain row count and \(n\) is the subtrain row count. In a decision run, \(n_i\) is client \(i\)’s full sealed-training row count and \(n\) is the sealed training row count. Both SGD arms in a run use the same \(p_i\). CoordMedian does not use \(p_i\).

FedNova then uses

\[
\tau_{\mathrm{eff}}=\sum_i p_i A_i,\qquad
w_{\mathrm{next}}=w+\tau_{\mathrm{eff}}\sum_i p_i\frac{\Delta_i}{A_i}.
\]

Matched FedAvg uses the same \(\Delta_i\) and the same \(p_i\), without \(A_i\):

\[
w_{\mathrm{next}}=w+\sum_i p_i\Delta_i.
\]

Both sums accumulate in `float64`, in client order `0` through `8`, and the result is cast back to the incoming parameter dtype. FedNova–SGD is not compared with the sealed Adam FedAvg rows.

### 3.3 One shared client update

FedNova and matched FedAvg call one E5-module function for every client round, in tuning and in decision runs. There are not two copies of the local loop. That function uses one operation order and one dtype. It returns the client endpoint, \(\Delta_i\), and \(A_i\). The endpoint equals the starting parameters plus \(\Delta_i\).

FedNova’s server step uses \(\Delta_i\) and \(A_i\). FedAvg’s server step uses the same \(\Delta_i\) and the same \(p_i\), and ignores \(A_i\). Only the server aggregation differs. The function is not `train_client_nb`, and `fl_local/engine.py` is not edited.

“Matched” means this same function, the same protocol, and the same randomness. It does not mean that the arms keep identical deltas after their server models diverge. A fixture that gives both calls the same starting parameters and the same minibatches must obtain byte-identical endpoints and byte-identical \(\Delta_i\) before aggregation. After round one, the actual deltas may differ because the global models differ.

The counted \(\tau_i\) is the number of `optimizer.step()` calls in that client round. The expected count is \(3\lceil n_i/512\rceil\). If the counted value differs, or if any \(A_i\le 0\), the cell fails.

## 4. Learning-rate selection and schedule

The locked value is a constant learning rate. Every local step, in all 30 rounds, uses that same number. There is no warmup and no decay. Tuning runs and decision runs use this same constant schedule.

The decision cells do not use the test set to choose the rate. The sealed test split stays untouched.

### 4.1 Splitter

The splitter is `sklearn.model_selection.train_test_split` from scikit-learn **1.8.0**:

- `test_size=0.2`
- `train_size=None`
- `shuffle=True`
- `random_state=10007`
- `stratify` as defined below

In this version, a float test size sets `n_validation = math.ceil(0.2 * n_rows)` and `n_subtrain = n_rows - n_validation`. Stratified counts then follow scikit-learn 1.8.0’s `_approximate_mode` allocation. No other rounding rule may be substituted.

Indices are positions into the sealed training-row order. They are saved as little-endian `int64` arrays before any tuning run. The recorded hash is SHA-256 of the subtrain bytes immediately followed by the validation bytes:

`sha256(subtrain_idx.tobytes() + val_idx.tobytes())`

with both arrays C-contiguous and dtype `<i8`. Each tuning run reloads the arrays and recomputes that hash. A mismatch stops the block.

### 4.2 Strata

For CIC, `stratify` is the sealed training label. For N-BaIoT, `stratify` is one code per training row, `device_id * (n_classes + 1) + label`, using the sealed device ids and labels. This keeps the device-by-label structure in both parts. If any stratum has fewer than two rows, stop. Do not drop the stratum or switch to an unstratified split.

After the split, the subtrain partition must leave all nine training clients nonempty. For N-BaIoT, every device id present in the sealed training ids must also remain in subtrain, and the device partition of subtrain is those devices. For CIC, the Dirichlet partition of subtrain uses seed 10007. If any client is empty, stop. Do not change the seed.

`subtrain_idx` and `val_idx` are positions in the full sealed-training array. The tuning client indices are not those positions. Form the compact subtrain matrix in `subtrain_idx` order, then apply the shared tuning scaler from Section 4.4. `X_subtrain` is that scaled compact matrix. Its rows are numbered \(0\) through \(n_{\mathrm{subtrain}}-1\).

The saved arrays `c0` through `c8` are positions in `X_subtrain`. They are disjoint, and their union is exactly \(\{0,1,\ldots,n_{\mathrm{subtrain}}-1\}\). Client \(i\) trains on `X_subtrain[c_i]`. The same rows are `X_unscaled[subtrain_idx[c_i]]` after this same scaler is applied. Both expressions must select those rows. An index into the full sealed-training array is not a valid value of `c_i`.

`partition_dirichlet` receives the compact subtrain labels in `subtrain_idx` order. `partition_by_device_ids` receives the compact subtrain device ids in that same order. Decision partitions remain indices into the full sealed-training array, as in Section 5.5.

### 4.3 Sealed-cache binding, before tuning

Before any tuning run, the no-training preflight reconstructs both sealed caches in memory and compares those arrays with the sealed files. It does not write an NPZ, a meta file, or a replacement cache. The acceptance check is equality of array bytes. A hash of a newly compressed NPZ is not that check, because container metadata can differ while the arrays do not.

These identities are bound. Hash comparison is case-insensitive.

| Object | Path or value | SHA-256 |
|---|---|---|
| CIC cache key | `27bfed227ef77fab` | — |
| CIC NPZ | `probe_results/cache/cic_27bfed227ef77fab.npz` | `8F032DF1760A895B2358B8FF2DA71491B09820C7AF8C097DD00465FD7B981B38` |
| CIC meta | `probe_results/cache/cic_27bfed227ef77fab.meta.json` | `E5E0EF795FFF100AF24F6813CA3A2A61442E386AA01BF92878A9CFC0B03CB3B4` |
| N-BaIoT cache key | `4ddb8b93b8821650` | — |
| N-BaIoT NPZ | `probe_results/cache/nbaiot_4ddb8b93b8821650.npz` | `C0A1E21CD6336D3A090E5FEB9411758ABBF8669DCCA01BD476DEFDBCF8380854` |
| N-BaIoT meta | `probe_results/cache/nbaiot_4ddb8b93b8821650.meta.json` | `0124AC4C793417D53BD062F4694E72BA903ED42BEEE8873D578ADF57A3686B23` |
| CIC parser | `fl_local/cic_cache.py` | `91A322560C7A4AD3D47ED70DCC70AB2E00F6DB134490DEBC872A35EB6B0760E3` |
| N-BaIoT parser | `fl_local/nbaiot_cache.py` | `55E0A136E54BB0759497AAE7BCCE1402ED9B56149614AAF38C4260B0F7C04693` |

The preflight imports both parser files. If either import fails, or if any other source file required by that import fails to load, the preflight stops. That failure is not caught and skipped.

The training interpreter must report this environment. A difference stops the preflight.

| Package | Required version |
|---|---|
| Python | `3.12.10` |
| NumPy | `2.3.5` |
| pandas | `3.0.1` |
| scikit-learn | `1.8.0` |
| PyTorch | `2.5.1+cu121` |
| CUDA | `12.1`, available |
| GPU | Quadro P3200 |

Reconstruction uses the sealed meta fields `extract_dir`, `max_samples_per_file=20000`, `test_fraction=0.2`, `global_seed=42`, and `scale_on=train`. The cache key computed from that resolved extract directory must equal the sealed key. The original split is scikit-learn 1.8.0 `train_test_split` with `test_size=0.2`, `random_state=42`, and `stratify` equal to the label array. `StandardScaler(with_mean=True, with_std=True)` is then fit on the full training split only. Train and test features are transformed and cast to `float32`. Labels and device ids are cast to `int64`. This is the original full-training scaling. It is separate from the tuning scaler in Section 4.4.

Every CSV the reconstruction selects is read to completion. An exception, an empty read, or a table below the parser’s column minimum stops the preflight and records the path. The preflight does not use a branch that prints `skip` and continues. Each path excluded by the sealed label rules is listed. The preflight then still has to pass the byte check below.

The sealed NPZ is loaded with `allow_pickle=False`. For each required array, the reconstructed array and the sealed array must have the same shape, the same dtype, and the same C-contiguous bytes, `np.ascontiguousarray(array).tobytes(order="C")`.

- CIC: `X_train`, `X_test`, `y_train`, `y_test`, and `cid_train`. The sealed CIC NPZ has no `cid_test`.
- N-BaIoT: those five arrays plus `cid_test`.

A mismatch stops the preflight. The unscaled training matrix held before this full-training scaler is the matrix used for the tuning split in Sections 4.1 and 4.2. Decision cells train on the sealed scaled arrays after the byte check has passed.

### 4.4 Tuning-only scaler

The sealed cache does not store the scaler mean or scale. Tuning does not train on the sealed scaled rows and does not rewrite the cache.

On the unscaled training matrix from Section 4.3, fit `sklearn.preprocessing.StandardScaler` from scikit-learn 1.8.0, with `with_mean=True` and `with_std=True`, on the subtrain rows only. Transform the subtrain and validation rows with that scaler and cast to `float32`. Decision cells do not use this scaler.

There is one scaler fit for CIC, shared by T1 and T2, and one for N-BaIoT, shared by T5 and T6. Both fits are written before the first of the 24 tuning runs. No run refits either scaler. All six runs in T1, T2, T5, and T6 reload the artifact for their dataset and verify `scaler_sha256`. A mismatch stops the block.

| Dataset | Blocks | Artifact |
|---|---|---|
| CIC-IoT2023 | T1 and T2 | `q1_e5_results/tuning/scaler_CIC-IoT2023_s10007.npz` and `.json` |
| N-BaIoT | T5 and T6 | `q1_e5_results/tuning/scaler_N-BaIoT_s10007.npz` and `.json` |

The NPZ stores `mean_`, `var_`, and `scale_`, each C-contiguous dtype `<f8`, with length equal to the feature count. The JSON stores `n_samples_seen_`, `feature_count`, `dtype`, `subtrain_index_sha256`, `validation_index_sha256`, and `scaler_sha256`.

`dtype` is `<f8`. `feature_count` equals the sealed meta `num_features`: 39 for CIC and 115 for N-BaIoT. On scikit-learn 1.8.0, `n_samples_seen_` is a NumPy scalar; it must be an exact integer equal to \(n_{\mathrm{subtrain}}\). The JSON stores that integer.

`subtrain_index_sha256` is SHA-256 of the C-contiguous `<i8` bytes of `subtrain_idx` alone. `validation_index_sha256` is the same hash of `val_idx` alone. The combined hash in Section 4.1 remains the check on the saved index file.

`scaler_sha256` is SHA-256 of this concatenation, in order:

1. `feature_count` as `uint64` little-endian;
2. the integer `n_samples_seen_` as `uint64` little-endian;
3. the UTF-8 bytes of `<f8`;
4. the `<f8` bytes of `mean_`, then `var_`, then `scale_`;
5. the 32 raw bytes of `subtrain_index_sha256`;
6. the 32 raw bytes of `validation_index_sha256`.

### 4.5 Selection

For each block, train both SGD arms at every rate in \(\{10^{-3}, 10^{-2}, 10^{-1}\}\), seed 10007, for 30 rounds, on the tuning split. That is six runs per block. If any one of the six is missing, non-finite, or incomplete, stop. Do not select from the runs that remain.

The block’s locked rate is the candidate whose two arms have the higher mean validation Final macro-F1. The mean is the `float64` mean of the two stored Final macro-F1 values. A tie is exact equality of those means. There is no tolerance. An exact tie selects the smaller learning rate.

Record the six validation scores, the means, and the locked rate before any decision cell in that block starts. Tuning rows are not decision rows and have no decision *p*-values. The Holm tests are conditional on the locked rates. They do not correct for the search over three candidates.

Decision cells are then trained on the full sealed training split and scored once on the sealed test set.

## 5. Paired randomness

The explicit `DataLoader` generator applies only to FedNova and matched FedAvg, including their tuning runs. CoordMedian uses the E4 path in Section 5.1.

### 5.1 CoordMedian keeps the E4 path

CoordMedian calls the unmodified `train_client_nb`. `fl_local/engine.py` is not edited, so the new median aggregation cannot be added inside `run_fl_experiment`. The E5 module therefore clones the surrounding `match_notebook` order and changes only the server aggregation.

The clone calls `set_global_seed(seed)` once, at the same point as `run_fl_experiment`, after the decision partition has been built and before the initial model is created. It then follows `_run_match_notebook`: create the initial model, read its parameters, evaluate round 0, and in each round call `train_client_nb` for clients `0` through `8`. `train_client_nb` creates a new module, copies the server parameters, and uses `DataLoader(..., shuffle=True)` with no `generator` argument. That shuffle consumes the global torch generator. The clone does not pass a generator and does not reseed inside the round.

`enable_repro_flags()` is called once before the first CoordMedian cell, as the E4 wave does. The clone does not change cuDNN or TF32 flags. Section 5.4 records the process values under which this path runs.

Before any CoordMedian cell, a CPU test must show that this clone reproduces the unmodified `match_notebook` path on a tiny FedTrimmedAvg fixture: the same seed, data, and partition produce the same parameter bytes and the same history when the clone’s aggregation is the sealed trimmed mean. A second fixture checks that the median aggregation itself is the fifth ordered value. The only substituted operation in the real CoordMedian cells is that median.

Each CoordMedian cell reuses the sealed E4 decision partition for its dataset and seed, as specified in Section 5.5.

### 5.2 Momentum-SGD generator lifetime

The two momentum-SGD decision arms do not call `train_client_nb`. Both call the one client-update function in Section 3.3. Their loop is owned by the E5 module.

Before each SGD decision arm, reset the global generators to that decision seed:

- `random.seed(seed)`
- `numpy.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`

The initial parameter vector is drawn from that reset. The two arms store the same initial-parameter hash. Client order is `0` through `8`. Both arms use the sealed E4 decision partition from Section 5.5.

Each client has one `torch.Generator`, created once after that reset and kept for all 30 rounds and all three local epochs. It is seeded once:

`generator.manual_seed(seed + 1_000_003 * (client_index + 1))`.

Every `DataLoader` for that client receives this same generator object, with `shuffle=True` and the last batch kept. The generator is not recreated at a round boundary or an epoch boundary. Recreating it from the client seed at every round would repeat the same shuffle in every round. That construction is a failed cell.

The same generator object advances across rounds, so later rounds do not repeat the first round’s shuffle. Both SGD arms use this lifetime and this seed formula. Their minibatch sequences therefore match.

### 5.3 The same pairing covers all six tuning runs

The reset rule is not limited to decision seeds. Inside one block, all six tuning runs use seed `10007` for the same four objects: initialization, client partition, client order, and minibatch-order construction.

Before each of the six runs, reset `random`, NumPy, torch, and `torch.cuda.manual_seed_all` to `10007`. The initial-parameter hash must be the same for all six. The client order is `0` through `8`. All six load the one tuning partition saved for that block under Section 5.5.

Each tuning run creates one persistent generator per client, after its own reset, and keeps that object for all 30 rounds. The seed formula is the Section 5.2 formula with `seed=10007`. The generator is not recreated during the run. All six runs therefore start from the same generator state and follow the same minibatch sequence. All six call the client-update function in Section 3.3.

### 5.4 CUDA flags

Before tuning, and again before the SGD decision waves, set and then leave unchanged:

- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.backends.cuda.matmul.allow_tf32 = False`
- `torch.backends.cudnn.allow_tf32 = True`

The first two are the E4 `enable_repro_flags()` values. The last two are the torch `2.5.1+cu121` process defaults measured on this machine on 2026-10-05. `torch.use_deterministic_algorithms` is not called. Section 9 requires every invocation to set these four flags and read them back. A process default, or an earlier preflight in another process, does not satisfy that check. CoordMedian reaches the same two cuDNN flags through `enable_repro_flags()` and does not add a different determinism call.

### 5.5 Saved partitions

The hash scheme is the sealed E4 scheme, implemented by `fl_local/e4_artifacts.py` (`f08b15541c48788a30c7d7cfbc485f13e9e03969eb1e7325aa39c2c268c70cd7`):

- partition: `sha256(cid uint64_le || length uint64_le || ordered indices int64_le)`, clients `0` through `8`, no prefix;
- one client: `sha256(ordered indices int64_le)`, no prefix;
- indices are C-contiguous dtype `<i8`.

The sealed partition document is `q1_e4_results/reporting/Q1_E4_FROZEN_CONFIG.json`, SHA-256 `3E78215912FB7E1315C7A133CC964038387A88EE8D73577D14390CCA6B28F9BE`. Every decision cell uses the entry for its dataset and seed. The cell’s saved document must equal that entry, including `sha256`, `client_sizes`, and all nine `client_index_sha256` values. For N-BaIoT, `pooled_test_device_ids` must match as well. The aggregate hashes are:

| Dataset | Seed | Partition SHA-256 |
|---|---|---|
| CIC-IoT2023 | 127 | `d6fbc871d4c6ca1fcd3d3ba9519c5227a78aa317ada71e24640dc38bcd0fdcec` |
| CIC-IoT2023 | 131 | `2a82f966a94246c04d00f4bf9dafcf05e35c9a8a4022ad93bed721e983855d61` |
| CIC-IoT2023 | 137 | `d0cf4a3171b3ecc4b7116604e5b5b1fe7e09d0fef9024ea66041554223814167` |
| CIC-IoT2023 | 139 | `e4d97f11c4b63e93050ba853aacf6d14b3839e428aa4e36eddd372e44a3662ed` |
| CIC-IoT2023 | 149 | `e98eb725a455d8aa41db6c13852d19b00f65d4e10fd1bd1681694b5b23623160` |
| CIC-IoT2023 | 151 | `dec1a709d74aba856626f2e3d67083bc2f86c4fe88c42dff57e0af63c9edc7bf` |
| CIC-IoT2023 | 157 | `9e7a9a32ffe0433cc53d36dd526f8c0872d91c2c4b82780c47ca669326239828` |
| N-BaIoT | device, all decision seeds | `066bf886d4a7c6828d4d8d9cd189c707bbb6cc0132273f480c95039107ba32c3` |

CIC decision partitions come from `partition_dirichlet` on the sealed training labels, alpha `0.1`, with the decision seed. N-BaIoT decision partitions, and N-BaIoT tuning partitions, come from `partition_by_device_ids` in `fl_local/e2_partition.py`. That file’s SHA-256 is `72778F968A03D2F934801B9270E4CDBFEB2759AD7960B20F514E2069C863E8A2`. A different device-partition implementation stops the run. The decision call receives the sealed training device ids. The tuning call receives the compact subtrain device ids from Section 4.2. T1 and T2 share the CIC partition for a seed. T5 and T6 share the one N-BaIoT device partition. CoordMedian, FedNova, and matched FedAvg all load that same partition. A mismatch with the frozen document stops the cell.

Each distinct decision partition is saved before the cell that first uses it:

- `q1_e5_results/partitions/decision_CIC-IoT2023_s<seed>.npz` and `.json`
- `q1_e5_results/partitions/decision_N-BaIoT_device.npz` and `.json`

The NPZ stores `c0` through `c8` as C-contiguous `<i8`. The JSON stores the same fields as the frozen entry: `sha256`, `client_sizes`, and `client_index_sha256`, and for N-BaIoT also `pooled_test_device_ids`. Each decision cell records that partition SHA-256.

Tuning partitions are a different artifact. They use seed `10007` and the compact subtrain rows from Section 4.2. `c0` through `c8` are positions in `X_subtrain`, disjoint and covering exactly \(0\) through \(n_{\mathrm{subtrain}}-1\). CIC uses `partition_dirichlet` on the compact subtrain labels, alpha `0.1`. N-BaIoT uses the bound `partition_by_device_ids` on the compact subtrain device ids. One partition is saved per block before any of that block’s six runs:

- `q1_e5_results/partitions/tuning_<block>_s10007.npz`
- `q1_e5_results/partitions/tuning_<block>_s10007.json`

The same scheme is used. Each of the six runs reloads the arrays and recomputes the hash. A mismatch stops the block. T1 and T2 must produce the same tuning-partition hash, and T5 and T6 must produce the same tuning-partition hash, because the split and the partition depend on the dataset and seed `10007`, not on the model. A tuning partition is not required to equal an E4 decision-partition hash.

## 6. Implementation boundary

`fl_local/engine.py` must not be modified. Its SHA-256 on 2026-10-05 is `7d201f9734b56c29457cd028933b0eca4862e241b17040e1cb6e88c6b0f130d2`. New code belongs in `fl_local/e5_aggregators.py` and a future E5 runner. The processed train-only caches and Q1-E4 rows must not be rebuilt.

A non-finite value, a failed step-count check, a failed index hash, a missing partner, an empty client, a sealed-array byte mismatch, a partition-hash mismatch, a scaler-hash mismatch, a parser import or CSV read failure, or a CUDA-flag mismatch quarantines the run. It is not replaced by another seed or another arm.

### 6.1 Integrity tests required before any GPU cell

`tools/test_q1_e5_integrity.py` must pass all of the following. The tests do not exist yet, and writing them is not authorized until this draft is accepted.

- The nine-client median is the fifth ordered value.
- A hand calculation with unequal step counts checks \(c_k\), the order \(a_i=(c_{\tau_i},\ldots,c_1)\), \(A_i=\lVert a_i\rVert_1\), \(\tau_{\mathrm{eff}}\), the sign of \(\Delta_i/A_i\), and the resulting FedNova update. Dividing by the vector \(a_i\) fails the test.
- A deterministic fixture checks that matched FedAvg equals \(w+\sum_i p_i\Delta_i\).
- A tuning fixture computes \(p_i\) from subtrain client sizes. A decision fixture computes \(p_i\) from full-training client sizes. Using the other size fails the test.
- The nine weights are nonnegative and sum to one. A zero or negative total fails.
- The counted `optimizer.step()` calls equal \(3\lceil n_i/512\rceil\) on a tiny loader.
- A second round starts with zero momentum state, zero cumulative update, and zero normalizing counter.
- FedNova and matched FedAvg call the same client-update function. Given the same starting parameters and the same minibatches, the returned endpoints and \(\Delta_i\) are byte-identical before aggregation. A follow-up that starts the two calls from different parameters may return different deltas. FedAvg’s server step ignores \(A_i\).
- The two SGD decision arms, and all six tuning runs in a block, produce the same initialization, partition, client order, and minibatch index sequence from the shared seed.
- Saved tuning indices `c0` through `c8` are disjoint and cover exactly \(0\) through \(n_{\mathrm{subtrain}}-1\). Using a full sealed-training position as `c_i` fails the test.
- The scaler hash covers `mean_`, `var_`, `scale_`, `n_samples_seen_`, the feature count, dtype `<f8`, and the two index hashes. T1 and T2 share one CIC artifact, and T5 and T6 share one N-BaIoT artifact.
- One persistent generator per client yields a different shuffle in round 2 than in round 1 on a fixture where recreation would repeat. Recreating the generator from the client-only seed at each round fails the test.
- The CoordMedian clone matches unmodified `match_notebook` parameter bytes on a tiny FedTrimmedAvg fixture, calls `train_client_nb`, and passes no `DataLoader` generator. The median fixture is the fifth ordered value.
- A CSV that raises, and a parser file that fails to import, stop the preflight. A branch that skips the failure fails the test.
- An array-byte fixture accepts equal C-contiguous bytes and rejects a check that uses only the compressed-NPZ hash.
- A non-finite value fails.
- `fl_local/engine.py` still has the hash above.
- A tuning, wave, or resume entry point that skips the Section 9.1 recheck fails, including an entry point that would train only CoordMedian.

## 7. Frozen design

### 7.1 Blocks

| Block | Dataset | Client construction | Model | Condition |
|---|---|---|---|---|
| **T1** | CIC-IoT2023 | 9 simulated Dirichlet clients | MLP | alpha = 0.1 |
| **T2** | CIC-IoT2023 | 9 simulated Dirichlet clients | CNN | alpha = 0.1 |
| **T5** | N-BaIoT | 9 source-device clients | MLP | fixed device partition |
| **T6** | N-BaIoT | 9 source-device clients | CNN | fixed device partition |

These are simulated clients made from dataset partitions. They are not physical devices and not a privacy audit. On CIC, a decision seed sets matched random conditions, including the partition. On N-BaIoT, the device partition is fixed across decision seeds.

Preprocessing for decision cells remains the sealed `scale_on=train` cache. Cache rebuild is forbidden. `scale_on=full` rows must not be pooled with these rows. Each decision cell has 30 rounds and nine clients.

### 7.2 Arms and cells

| Arm | Cells | Partner |
|---|---:|---|
| CoordMedian–Adam | 28 | sealed E4 FedTrimmedAvg–Adam |
| FedNova–momentum-SGD | 28 | matched FedAvg–momentum-SGD |
| FedAvg–momentum-SGD | 28 | the FedNova row with the same block and seed |
| **Decision total** | **84** |  |

Decision seeds: **{127, 131, 137, 139, 149, 151, 157}**. No additional or eighth decision seed may be added. Seeds {42, 123, 7} and tuning seed 10007 are not decision seeds.

The tuning stage is 4 blocks × 2 SGD arms × 3 learning rates × 1 seed = **24** runs. Those 24 runs are not part of the 84 and are not in the Holm family.

### 7.3 Bound E4 reference

Adam trimmed-mean partners come only from:

- file: `q1_e4_results/reporting/q1_e4_seed.csv`
- SHA-256: `8A9218A130486B68C69F3DD40CF5DD82A217ADD985C66A91AF6FE3DD97DA90C9`

The key is block, algorithm `FedTrimmedAvg`, and decision seed. A hash mismatch stops the analysis.

### 7.4 Endpoint and required reporting

The only decision endpoint is unrounded Final macro-F1 at round 30. For every one of the eight comparisons, the report must include:

- the mean paired difference in percentage points, \(100\) times the unrounded mean of the seven Final-macro-F1 differences;
- the unadjusted 95% paired-\(t\) interval, also in percentage points;
- the raw *p*-value and the Holm-adjusted *p*-value;
- the negative, tie, and positive sign counts, in that order.

A difference is positive when it is greater than zero, negative when it is less than zero, and a tie when it is exactly zero.

If all seven differences are exactly zero, the mean is zero, the interval is \([0,0]\), and the raw *p*-value is 1. If all seven differences are exactly equal to the same nonzero value, the mean is that value, the interval is the point interval at that mean, and the raw *p*-value is 0. In both zero-variance cases the \(t\) statistic is not given a finite denominator. Holm uses those raw *p*-values in the same way as the other comparisons. A positive sample variance uses the ordinary two-sided paired \(t\) test with \(n=7\) and \(\mathrm{df}=6\).

Final accuracy, the mean of the 30 round-wise macro-F1 values, and Best-round scores are descriptive. The round-wise mean is not ROC-AUC.

## 8. Contrasts and decisions

One family of **eight** two-sided paired \(t\) tests:

| Order | Block | Contrast |
|---|---|---|
| 1 | T1 | FT–Adam − CoordMedian–Adam |
| 2 | T2 | FT–Adam − CoordMedian–Adam |
| 3 | T5 | FT–Adam − CoordMedian–Adam |
| 4 | T6 | FT–Adam − CoordMedian–Adam |
| 5 | T1 | FedNova–momentum-SGD − FedAvg–momentum-SGD |
| 6 | T2 | FedNova–momentum-SGD − FedAvg–momentum-SGD |
| 7 | T5 | FedNova–momentum-SGD − FedAvg–momentum-SGD |
| 8 | T6 | FedNova–momentum-SGD − FedAvg–momentum-SGD |

Holm adjustment is across these eight tests only, at family-wise level 0.05. Tied raw *p*-values keep the table order. The adjustment is cumulative and capped at 1. It does not include the Q1-E4 tests and does not correct for learning-rate search.

For the median contrast, a positive Holm-supported mean says the trimmed mean is higher than the median in that block only. A negative Holm-supported mean says the median is higher. For the FedNova contrast, a positive Holm-supported mean says this momentum-SGD FedNova rule is higher than its matched FedAvg in that block only. A negative Holm-supported mean says it is lower. There is no preferred direction for FedNova. Every other result is inconclusive.

An exact zero difference is a tie. Inconclusive is not equivalence. No TOST is authorized. One supported block is not a general claim. A FedNova result is not evidence about trimming, and a median result is not evidence about FedNova.

## 9. Waves and stopping

No GPU work starts until this draft is accepted by name and date, the E5 module and CPU tests exist, the integrity suite passes, and a no-training preflight passes. That preflight includes the sealed-array byte binding in Section 4.3 and the CUDA-flag check in Section 5.4. Every later command repeats the checks in Section 9.1. An earlier preflight does not replace that repetition.

The tuning index files, the dataset’s tuning scaler, and the block’s tuning partition are hashed before the six runs. Tuning for a block finishes, and its learning rate is locked, before any decision cell in that block. The first decision GPU action is one smoke cell: **T1, FedAvg–momentum-SGD, seed 127**, using T1’s locked constant learning rate. If its saved row, 30-round history, or hash check fails, stop. Do not start another cell. Its score is not a reason to change the learning rate, seeds, or family.

| Wave | Cells | Contents |
|---|---:|---|
| Tuning | 24 | Validation runs in Section 4. Not decision cells. |
| 0 | 1 | Smoke cell: T1 FedAvg–momentum-SGD, seed 127 |
| 1 | 27 | Remaining FedAvg–momentum-SGD decision cells |
| 2 | 28 | FedNova–momentum-SGD, all four blocks and seven decision seeds |
| 3 | 28 | CoordMedian–Adam, all four blocks and seven decision seeds |

After a valid smoke cell, Waves 1–3 run in order regardless of signs or *p*-values. A technical failure pauses that cell for repair. It does not permit another seed, another learning rate, or a change to the eight-test family.

### 9.1 Check on every invocation

Every tuning command, wave command, and resume command runs the checks below before it trains. This includes a process that runs only CoordMedian. An earlier preflight does not carry into a new process. A torch process default does not satisfy a check that this command did not itself perform.

The command recomputes these hashes and requires the bound values:

- `fl_local/cic_cache.py`
- `fl_local/nbaiot_cache.py`
- `fl_local/engine.py`
- `fl_local/e4_artifacts.py`
- `fl_local/e2_partition.py`
- both sealed NPZ files and both sealed meta files
- `q1_e4_results/reporting/Q1_E4_FROZEN_CONFIG.json`
- `q1_e4_results/reporting/q1_e4_seed.csv`

It also requires the Python, NumPy, pandas, scikit-learn, PyTorch, CUDA, and GPU versions in Section 4.3. It sets the four CUDA flags in Section 5.4 and then reads those four values back.

It reloads every partition file used by the cells in that command and recomputes the partition hash. It reloads both tuning-scaler artifacts from Section 4.4 and recomputes `scaler_sha256`, including when the command trains no SGD cell. Both artifacts are written before the first tuning run; after that write, a missing file stops the command. A mismatch stops the command.

## 10. Compute estimate

The old **86–110** GPU-hour figure was for a 56-cell Adam-style campaign. It does not apply here.

The estimate uses the exact completed Q1-E4 block means and maxima and assumes a momentum-SGD cell takes about as long as an Adam cell in the same block. SGD may be somewhat cheaper. This estimate does not claim that saving. Tuning runs are budgeted as full-length cells.

| Piece | Cells | Exact mean-based hours | Exact maximum-based hours |
|---|---:|---:|---:|
| CoordMedian–Adam | 28 | 42.988554875486145 | 45.850877955749894 |
| Both momentum-SGD arms | 56 | 85.97710975097229 | 91.70175591149979 |
| Tuning, budgeted as full-length cells | 24 | 36.84733275041669 | 39.30075253349991 |
| **Exact total** | **108 runs** | **165.81299737687513** | **176.8533864007496** |
| **Reported total** |  | **165.81** | **176.85** |

The reported mean total is 165.812997 hours rounded directly to **165.81**. Rounding component rows first and then adding them produces 165.82 and must not be used. The reported maximum total is **176.85** hours. The operational ceiling is **213 GPU-hours**. Crossing it is a reason to pause for a hardware or software check. It is not a reason to drop cells, skip FedNova, or change the analysis.

If this budget is not acceptable, the safer reduced study is the 28-cell Adam median comparison only, with FedNova explicitly deferred. That reduced study must be written as its own acceptance. It is preferable to running StepNormAdam. This draft does not silently switch to that reduced study.

## 11. What acceptance would authorize

Acceptance requires a named proposer, a named approver, and a date, followed by a separate contract amendment recording this preregistration at **0/84** decision cells. That amendment is not part of this draft.

After those facts, and not before, the accepted plan would authorize the E5 module, the CPU tests, a no-training preflight, the tuning stage, the one smoke cell, and then Waves 1–3.

Acceptance would not authorize StepNormAdam, SCAFFOLD, another Q1-E4 cell, a FedNova–SGD versus Adam-FedAvg test, CIC alpha = 0.5, CIC Wave 3, Q1-E8, cache rebuild, an eighth decision seed, or a change to C5, C6, or C8.

## 12. Acceptance block

Filled from the supervisor acceptance on 2026-10-06 (10-6-2026).

- Proposer: Sheryar Kiani
- Approver: Dr Hussain Kazmi
- Acceptance date: 2026-10-06 (10-6-2026)
- Decision text: Accept this 84-cell Q1-E5 preregistration as written.
- This acceptance does not approve StepNormAdam.

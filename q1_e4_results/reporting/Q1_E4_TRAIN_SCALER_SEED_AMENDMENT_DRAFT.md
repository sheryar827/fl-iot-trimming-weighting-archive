# Q1-E4 preregistration amendment — train-only paired seed expansion to 10 total seeds

**Status:** **ACCEPTED 2026-09-26 — implementation and CPU-only gates authorized; no GPU until the integrity suite and no-training preflight pass**  
**Draft date:** 2026-09-26  
**Acceptance date:** 2026-09-26 (26-9-2026)  
**Purpose:** replace the active Q1-E4 execution plan for the leakage-corrected manuscript with a fixed train-only seed expansion, using seven untouched seeds for prospective replication and all ten seeds for combined estimation.  

## Named acceptance

- **Proposer:** **Sheryar Kiani**  
- **Approver:** **Dr Hussain Kazmi**  
- **Acceptance date:** **2026-09-26** (26-9-2026)  
- **Decision text:** **Accept Q1-E4 train-only 10-total-seed / 7-new-seed prospective replication preregistration as written.**  

The named acceptance and dated amendment to `Q1_CONTRIBUTION_CONTRACT.md` were
applied on 2026-09-26. Runner and integrity-suite implementation may now begin.
Cache building or modification remains forbidden. Do not start a GPU cell until
the accepted runner/configuration, CPU-only integrity suite, and no-training
preflight all pass.

## 1. Relationship to the live contract

Before this accepted amendment, the live contract defined Q1-E4 through the
older **P1–P11** `scale_on=full` three-arm plan. That lineage is not the main
evidence base of the leakage-corrected manuscript.

Under this accepted amendment, **T1/T2/T5/T6 below are the active Q1-E4
requirement for the current manuscript**. The P1–P11 definitions remain in the
contract as a clearly labelled **legacy `scale_on=full` plan that is inactive and
deferred**. P1–P11 authorize no GPU work and may be reopened only by another
dated amendment. Train-only T-family results must not be pooled with
`scale_on=full` results and do not satisfy or upgrade the P1–P11-specific C3/C4
triggers.

This amendment does not open CIC Wave 3, Q1-E5, or Q1-E8 and does not change
C5, C6, C8, or the bounded Q1-E7 reporting rule.

## 2. Research question and estimand

Q1-E4 asks whether the observed trimming and client-size-weighting effects are
stable across seeded experimental replicates in the two leakage-corrected
settings that carry the manuscript's main conclusion:

1. severe synthetic label skew on CIC-IoT2023 (Dirichlet alpha = 0.1); and
2. the natural nine-device N-BaIoT client partition.

The unit of replication is the **seeded experimental replicate**, not an arm,
round, client, class, or test example. Arms are paired within the same block and
seed. On CIC, one seed jointly determines the Dirichlet client partition and
training/model randomness, so the replicate captures both sources of
variability. On N-BaIoT, the source-device partition is fixed and the seed
changes training/model randomness only.

## 3. Frozen design

### 3.1 Blocks and arms

| Block | Dataset | Client construction | Model | Condition | Four arms in every seed |
|---|---|---|---|---|---|
| **T1** | CIC-IoT2023 | 9-client Dirichlet partition | MLP | alpha = 0.1 | UW, FA, FT, WT |
| **T2** | CIC-IoT2023 | 9-client Dirichlet partition | CNN | alpha = 0.1 | UW, FA, FT, WT |
| **T5** | N-BaIoT | 9 source-device clients | MLP | device partition | UW, FA, FT, WT |
| **T6** | N-BaIoT | 9 source-device clients | CNN | device partition | UW, FA, FT, WT |

Arm definitions:

- **UW:** `UnweightedFedAvg` — no coordinate trimming; equal client weight.
- **FA:** `FedAvg` — no coordinate trimming; client-sample-size weight.
- **FT:** `FedTrimmedAvg` — coordinate-wise trim; equal survivor weight.
- **WT:** `WeightedFedTrimmedAvg` — coordinate-wise trim; client-sample-size
  weight among survivors.

The trim setting remains beta = 0.1 with 9 clients. At each model-parameter
coordinate, the lowest and highest client values are removed and **7/9 values
remain**; the surviving client identities may differ by coordinate. T3/T4 (CIC
alpha = 0.5), alpha = 1.0, CIC Wave 3, FedProx, Median, SCAFFOLD/FedNova, and
IID cells are outside Q1-E4.

### 3.2 Seeds and fixed workload

Existing seeds, reused without retraining: **{42, 123, 7}**.  
New seeds: **{127, 131, 137, 139, 149, 151, 157}**.  
Final reporting order: **{7, 42, 123, 127, 131, 137, 139, 149, 151, 157}**.

The existing three seeds are already observed and helped motivate the selected
four-block scope. They are therefore **not** counted as prospective confirmation.
The seven new seeds constitute the prospective replication sample. The final
ten-seed table is a combined, data-informed effect-size estimate and sensitivity
summary, not an independent ten-seed confirmation.

The four factor contrasts and their directions were frozen in the antecedent
CIC-confirmatory and Q1-E2 preregistrations before those campaigns. However,
the choice to expand these four blocks to ten total seeds was made after their
three-seed outcomes were known. The new-versus-combined separation above is
therefore binding.

The new seeds are the first seven prime integers strictly greater than the
largest existing seed (123). This deterministic rule was fixed before examining
any new-seed result. No seed may be replaced, dropped, or added because of its
metric, rank, sign, variance, runtime, or partition shape. A failed cell is
rerun with the **same** seed after the technical cause is recorded.

The frozen workload is:

- **112 new cells** = 4 blocks x 4 arms x 7 new seeds;
- **160 combined cells** = 4 blocks x 4 arms x 10 total seeds; and
- a complete same-seed quartet is required before a seed contributes to a
  block's analysis.

A result-free feasibility check found all nine CIC clients non-empty for each
proposed new seed. The accepted runner must reproduce and record the partition
checks before training.

### 3.3 Preprocessing and fixed caches

All rows use `mode=match_notebook`, 30 communication rounds, and
`scale_on=train`. Cache building is forbidden in the Q1-E4 runner.
`--require-from-cache` is mandatory.

| Dataset lineage | Cache key | NPZ SHA-256 | Meta SHA-256 |
|---|---|---|---|
| CIC-IoT2023 train scaler | `27bfed227ef77fab` | `8f032df1760a895b2358b8ff2da71491b09820c7af8c097dd00465fd7b981b38` | `e5e0ef795fff100af24f6813ca3a2a61442e386aa01bf92878a9cfc0b03cb3b4` |
| N-BaIoT train scaler, device IDs retained | `4ddb8b93b8821650` | `c0a1e21cd6336d3a090e5feb9411758abbf8669dcca01bd476defdbcf8380854` | `0124ac4c793417d53bd062f4694e72ba903ed42beee8873d578adf57a3686b23` |

The existing three-seed anchors come from the frozen CIC confirmatory and Q1-E2
packets. They are read-only inputs to Q1-E4; the Q1-E4 runner must verify their
row keys, metrics, histories, cache bindings, and available manifests and must
not rewrite them.

For each arm in a matched quartet, the global seed is reset before partitioning
and model training. CIC uses the same seed for its Dirichlet partition and
training randomness. N-BaIoT keeps the fixed device partition and uses the seed
for training randomness.

### 3.4 Inherited scientific configuration (binding)

Q1-E4 inherits the accepted CIC-confirmatory and Q1-E2 scientific configuration;
it is a seed expansion, not a hyperparameter or architecture experiment.

| Pin | Binding value |
|---|---|
| Input cap | `max_samples_per_file=20000` |
| Split represented by each fixed cache | `test_fraction=0.2`, stratified by label, `random_state=42`; scaler fitted on pooled training rows only |
| Clients and participation | 9 clients; all 9 participate in every round; no client subsampling |
| CIC partition | `partition_dirichlet(y_train, 9, 0.1, seed=seed)` |
| N-BaIoT partition | accepted `partition_by_device_ids(cid_train)` mapping |
| Rounds / local epochs / batch | 30 / 3 / 512 |
| Local objective and optimizer | cross-entropy; `torch.optim.Adam`, lr=`1e-3`, otherwise PyTorch defaults; no FedProx term |
| Trim | coordinate-wise, `trimmed_fraction=0.1`; the lowest and highest client parameter values removed independently at each coordinate, leaving 7/9 values |
| MLP | input→128→64→64→32→classes; ReLU, LayerNorm after hidden linear layers, Dropout 0.2/0.2/0.1 |
| CNN | Conv1d 1→32 (k=5)→64 (k=3), ReLU, adaptive average pool, Linear 64→32→classes, LayerNorm(32), Dropout 0.2 |
| Reproducibility | reset Python/NumPy/Torch/CUDA seeds per cell; cuDNN deterministic=True and benchmark=False |
| Primary platform | local Quadro P3200; Python 3.12.10; Torch 2.5.1+cu121; CUDA 12.1; NumPy 2.3.5; pandas 3.0.1; scikit-learn 1.8.0 |
| Scientific engine | `fl_local/engine.py`, SHA-256 `7d201f9734b56c29457cd028933b0eca4862e241b17040e1cb6e88c6b0f130d2` |
| Device partition code | `fl_local/e2_partition.py`, SHA-256 `72778f968a03d2f934801b9270e4cdbfeb2759ad7960b20f514e2069c863e8a2` |

Every Q1-E4 cell must use the notebook-faithful `match_notebook` path. The new
runner may orchestrate and validate these frozen functions but may not change
their scientific behavior. A different engine, architecture, optimizer,
participation rule, partition rule, platform lineage, or preprocessing rule
requires a dated amendment **before** the affected run.

## 4. Operational waves (not scientific stopping rules)

| Stage | New cells | Fixed content | Gate |
|---|---:|---|---|
| **Preflight** | 0 | Acceptance, contract, exact 112-cell matrix, source/config hashes, cache hashes, existing 48 anchors, environment, partition coverage, empty/valid outputs | Must pass without training |
| **Wave 0** | 4 | Seed 127, FA only, one cell in each of T1/T2/T5/T6 | Smoke all dataset/model paths; cells are reused in Wave 1 |
| **Wave 1** | 28 | Complete every arm for seeds 127 and 131 after reusing Wave 0 | Produces n=5 only as an integrity checkpoint |
| **Wave 2** | 32 | All blocks and arms for seeds 137 and 139 | Continue regardless of observed results |
| **Wave 3** | 32 | All blocks and arms for seeds 149 and 151 | Continue regardless of observed results |
| **Wave 4** | 16 | All blocks and arms for seed 157 | Completes n=10 |

Wave 1 through Wave 4 proceed regardless of interim signs, ranks, confidence
intervals, or p-values. Intermediate combined n=5/n=7/n=9 states (two/four/six
new seeds) are for integrity and resume control only; they authorize no
scientific conclusion or cell selection.
An operational pause is allowed, but abandonment or redesign because results
look favorable or unfavorable requires a dated deviation record and cannot be
presented as the completed prospective replication.

Using the measured training time of the existing 48 train-only cells, linear
scaling estimates about **185 GPU-hours** for the 112 new cells on the Quadro
P3200. Plan approximately **200–215 elapsed GPU-hours** for overhead and
variation. These are planning estimates, not duration guarantees or stopping
floors. One process is permitted; the runner must enforce a process lock.

## 5. Frozen outcomes and paired analysis

### 5.1 Primary endpoint and contrasts

The primary endpoint is the **unrounded Final macro-F1 at round 30**.

For every block and seed, compute these four paired differences:

1. **FT - UW** (trim effect under equal client weighting);
2. **WT - FA** (trim effect under sample-size weighting);
3. **FA - UW** (weight effect without trimming); and
4. **WT - FT** (weight effect after trimming).

`WT - UW` is not a one-factor effect and must not be reported as one. Exact zero
is a tie.

### 5.2 Prospective new-seed inferential family

The inferential family contains exactly **16 comparisons**: 4 blocks x 4
contrasts. It uses **only the seven untouched new seeds** {127, 131, 137, 139,
149, 151, 157}. The three previously observed seeds are excluded from this
prospective hypothesis test. For each comparison, report:

- all 7 unrounded new-seed differences;
- mean difference in percentage points, SD, median, minimum, and maximum;
- positive / zero / negative sign counts;
- the unadjusted two-sided 95% paired-t confidence interval;
- the two-sided paired t-test (equivalently, a one-sample t-test of the paired
  differences against zero); and
- the raw p-value and **Holm-adjusted p-value across all 16 comparisons**, with
  family-wise alpha = 0.05.

A component direction is called **supported** only when its mean has the
preregistered direction and its Holm-adjusted p-value is below 0.05. A
Holm-adjusted result below 0.05 with the mean in the opposite direction is
reported as a **reversal**. Every other result is **inconclusive**; it does not
establish equivalence.

For each block:

- **Htrim** is supported only if both `FT-UW > 0` and `WT-FA > 0` component
  directions are supported; and
- **Hweight-hurt** is supported only if both `FA-UW < 0` and `WT-FT < 0`
  component directions are supported.

These support decisions use the seven new seeds only.

All null, mixed, reversed, tied, and inconclusive results remain in the report.
No outlier exclusion, seed deletion, comparator switching, or favorable-cell
subset is allowed.

No alternative test may replace this paired-t/Holm decision because it produces
a more favorable result. Distribution diagnostics may be displayed, but they
do not authorize seed deletion, test switching, or an unregistered conclusion.

### 5.3 Combined ten-seed estimation and secondary descriptive outcomes

For every block and contrast, also report all ten paired differences, the
combined mean in percentage points, SD, median, range, sign/tie counts, and an
unadjusted 95% paired-t confidence interval. Label these explicitly as
**combined data-informed estimates** because the original three seeds were
already observed when this expansion was designed. Do not call the ten-seed
table an independent confirmation, and do not use its p-values to override the
seven-new-seed Holm decision. No combined-ten-seed p-value belongs to the
accepted decision rule.

The following are secondary and descriptive only in this amendment:

- Final accuracy at round 30;
- mean of the 30 round-wise macro-F1 values (historically labelled `AUC`, but
  **not ROC-AUC**);
- Best accuracy and Best macro-F1, if retained for continuity;
- arm rankings, including the earlier HC1 ranking board; and
- the factorial interaction
  `(WT-FA) - (FT-UW)`, algebraically equal to `(WT-FT) - (FA-UW)`.

Report their paired differences and uncertainty where useful, but do not add
them to the 16-test family or make multiplicity-tested claims from them without
a later dated amendment. No TOST or equivalence claim is authorized.

## 6. Interpretation boundaries

- Q1-E4 estimates seed uncertainty for the four named train-only blocks only;
  it is not evidence for every alpha, dataset, architecture, or FL deployment.
- N-BaIoT remains a participating-device, pooled-holdout evaluation—not
  unseen-device testing or physical deployment.
- CIC and N-BaIoT may show different directions. A reversal is boundary
  evidence, not a failed experiment.
- Holm-adjusted paired evidence may support a named contrast in a named block;
  it does not authorize “universally superior,” “robust to Byzantine clients,”
  or general rare-class protection language.
- Q1-E7 remains the bounded descriptive class decomposition. Per-class rows are
  not extra independent replicates for Q1-E4.
- Results from `scale_on=train` and `scale_on=full` remain separate lineages.
- Final manuscript tables and prose must label inferential findings as based on
  the **seven prospective new seeds** and label every ten-seed result as a
  **combined, data-informed estimate**. CIC alpha=0.5 and every excluded block
  remain n=3 and must not inherit the Q1-E4 uncertainty or inferential claims.

## 7. Integrity and reporting requirements before GPU

Named acceptance does not authorize Wave 0. The runner must first be
implemented and integrity-tested, and the CPU-only integrity suite and
no-training preflight must pass before Wave 0. At minimum, the runner must:

1. pin and verify this accepted preregistration and its contract amendment;
2. freeze exact source/config hashes, library versions, job matrix, cache
   bindings, and output schema;
3. fail closed on missing or changed cache/provenance/anchor files;
4. validate nine non-empty CIC partitions covering every training row for all
   seven new seeds and the fixed nine-device N-BaIoT partition;
5. require CUDA and one process, use a lock, unique staging directories, atomic
   commits, and strict resume validation;
6. require exactly 30 history rows, finite metrics, exact-integer round count,
   matching round-30 metrics, and a complete same-seed four-arm quartet;
7. quarantine failed partial artifacts and rerun only the same failed cell;
8. write raw and history rows with cache/config/source hashes on every result;
9. build both the seven-new-seed inferential tables and the combined ten-seed
   estimation tables from the 48 verified existing anchors plus the 112 new
   rows; and
10. seal code, provenance, anchors, raw/history outputs, analysis tables, and
    the human-readable packet in a SHA-256 manifest. The manifest scope must
    disclose whether original source CSVs are excluded.

Changing hardware or software lineage, rebuilding a cache, changing a seed,
adding a block/endpoint/test, or changing an analysis decision requires a dated
amendment before the affected run.

## 8. Contract edits applied with named acceptance

The 2026-09-26 acceptance patch records who / what / why and applies all of the
following changes together:

1. Added a dated `**Amended:**` header line naming proposer and approver and
   recording this exact train-only n=10 plan.
2. Changed the Q1-E4 status row to **Prereg ACCEPTED; 0/112 new cells** and named
   T1/T2/T5/T6, the four arms, the seven prospective new seeds, the ten total
   seeds, and the separation between prospective inference and combined
   estimation.
3. Rewrote or prefixed the existing Section 4 **Significance (superiority)** and
   **Equivalence / null (C3)** rows so that their Welch/CI/TOST P1–P11 protocols
   are explicitly **legacy `scale_on=full`, inactive, and deferred**. They no
   longer read as a second active Q1-E4 analysis. Added a separate Section 4 row
   defining the active 16-comparison, seven-new-seed train-only paired-t/Holm
   family and the descriptive combined ten-seed estimates, explicitly separate
   from P1–P5, P6, P7–P11, the E1 descriptive family, and E7.
4. Added an active T1/T2/T5/T6 subsection and labelled P1–P11 as the inactive,
   deferred `scale_on=full` legacy plan. Kept their existing definitions for
   audit history and removed any implication that they authorize current GPU
   work.
5. Updated the next-action text to: implement and integrity-test the accepted
   Q1-E4 runner; run CPU-only preflight; then Wave 0. Q1-E5 remains next after
   Q1-E4; CIC Wave 3 and Q1-E8 remain deferred.

Acceptance freezes the scientific plan; it does not waive the integrity-suite
or preflight gate. All five contract edits have landed, but runner implementation
and integrity testing are not complete, and the CPU-only integrity suite and
no-training preflight have not yet passed. Therefore
**Q1-E4 is ACCEPTED at 0/112 new cells, and no Q1-E4 GPU command is authorized
yet.**

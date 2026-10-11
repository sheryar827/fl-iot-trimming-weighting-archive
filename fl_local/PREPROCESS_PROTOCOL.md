# Preprocessing protocol (two tracks — do not mix)

**Status:** protocol **specified in code** 2026-09-12. Binding for **new primary GPU** only after the CIC confirmatory prereg is **ACCEPTED**. `--list` is safe; **no Wave 0 / cache-build / training** before that.

Default runners stay on `"full"` so E1/E1b/E3 lineage is unchanged.

| Track | `scale_on` | Cache key | Use |
|-------|------------|-----------|-----|
| **Notebook-faithful (legacy)** | `"full"` **(default)** | unchanged `v1` (`a9dd0c758635aebe` / `b725017696446067` on this machine) | All E1 / E1b / E3 / gates already run |
| **Confirmatory** | `"train"` | **new** key (`scale_on=train` in hash) | Future primary experiments after this protocol is accepted |

## Legacy (`full`)

`StandardScaler.fit_transform` on **all rows**, **then** `train_test_split`. Test-set moments enter the scaler. This is **data leakage**. It matches the published notebooks. **Do not change this path** — doing so would invalidate G1–G6 and E1/E1b caches.

## Confirmatory (`train`)

1. Split train/test first (same seed, stratify, fraction).  
2. `fit` scaler on **train only**.  
3. `transform` train and test.  

Call: `load_or_build_*_cache(..., scale_on="train")`.

## Rules

- Never mix `full` and `train` caches in one statistical comparison.  
- Existing results = exploratory / notebook-faithful.  
- Confirmatory CIC 2×2 (recommended next GPU after E3 close) must use **`train` only**, all arms in the comparison, same seeds.  
- Do not `force_rebuild` the `v1` caches.

## Code

- `fl_local/nbaiot_cache.py`  
- `fl_local/cic_cache.py`  
Default remains `"full"`. Existing runners need **no** change to keep lineage.

"""Q1-E5 tuning artifacts and the 24 validation runs.

Decision cells are not started here. A locked learning rate is recorded only
after all six runs in that block are complete and finite.
"""
from __future__ import annotations

import gc
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import fl_local.engine as engine
from fl_local.e2_partition import partition_by_device_ids
from fl_local.e4_artifacts import atomic_write_json, hash_partition, partition_document
from fl_local.e5_aggregators import (
    BLOCKS,
    FEATURE_COUNTS,
    SCALER_FILES,
    SCALER_FOR_BLOCK,
    TRAIN_ROWS,
    TUNING_RATES,
    TUNING_SEED,
    arrays_byte_equal,
    combined_index_sha256,
    exact_nonnegative_int,
    fedavg_server_update,
    fednova_server_update,
    index_sha256,
    learning_rate_value,
    lock_learning_rate,
    pair_mean,
    require_finite,
    require_two_rows_per_stratum,
    scaler_sha256,
    sha256_bytes,
    sha256_file,
    tuning_holdout_sizes,
    tuning_index_files,
    tuning_matrix_files,
    tuning_partition_files,
    validate_compact_partition,
    weights_for_run,
)
from fl_local.e5_preflight import (
    PreflightStop,
    import_bound_parser,
    reconstruct_dataset,
    verify_bound_hashes,
    verify_environment,
    verify_partition_artifact,
    verify_scaler_artifact,
)
from fl_local.e5_protocol import (
    FEDAVG_CLIENT_UPDATE,
    FEDNOVA_CLIENT_UPDATE,
    apply_and_read_cuda_flags,
    client_generator,
    momentum_sgd_client_update,
)

PAPER = Path(__file__).resolve().parents[1]

DATASET_BLOCKS = {
    "CIC-IoT2023": ("T1", "T2"),
    "N-BaIoT": ("T5", "T6"),
}
CLASS_COUNTS = {"CIC-IoT2023": 8, "N-BaIoT": 11}
BLOCK_SPEC = {
    "T1": {"dataset": "CIC-IoT2023", "model": "MLP"},
    "T2": {"dataset": "CIC-IoT2023", "model": "CNN"},
    "T5": {"dataset": "N-BaIoT", "model": "MLP"},
    "T6": {"dataset": "N-BaIoT", "model": "CNN"},
}
ARMS = ("FedAvg", "FedNova")
ARM_NAMES = {
    "FedAvg": "matched FedAvg momentum-SGD",
    "FedNova": "FedNova using the published momentum-SGD normalization",
}
ROUNDS = 30
BATCH_SIZE = 512
LOCAL_EPOCHS = 3
EVAL_BATCH_SIZE = 4096


class TuningStop(RuntimeError):
    """A tuning cell is quarantined and is not replaced by another seed or arm."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


ATTEMPT2 = "attempt2"


def _tuning_home(attempt: str | None) -> str:
    if attempt is None:
        return "q1_e5_results/tuning"
    if attempt != ATTEMPT2:
        raise TuningStop(f"unknown tuning attempt {attempt!r}")
    return "q1_e5_results/tuning/attempt2"


def _log(root: Path, message: str, *, attempt: str | None = None) -> None:
    path = root / _tuning_home(attempt) / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = message.rstrip() + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
    print(line, end="", flush=True)


def _read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_savez(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _cell_rels(block: str, arm: str, rate_token: str, *, attempt: str | None = None) -> tuple[str, str]:
    stem = f"{_tuning_home(attempt)}/cells/{block}_{arm}_lr{rate_token}_s{TUNING_SEED}"
    return stem + ".json", stem + "_history.json"


def _lock_rel(block: str, *, attempt: str | None = None) -> str:
    return f"{_tuning_home(attempt)}/locks/{block}.json"


def _init_rel(block: str, *, attempt: str | None = None) -> str:
    return f"{_tuning_home(attempt)}/init/{block}.json"


def _dataset_files(dataset: str) -> tuple[str, ...]:
    files = []
    files.extend(SCALER_FILES[dataset])
    files.extend(tuning_index_files(dataset))
    files.extend(tuning_matrix_files(dataset))
    for block in DATASET_BLOCKS[dataset]:
        files.extend(tuning_partition_files(block))
    return tuple(files)


def _decision_names(root: Path) -> set[str]:
    folder = root / "q1_e5_results" / "partitions"
    if not folder.is_dir():
        return set()
    return {path.name for path in folder.iterdir() if path.name.startswith("decision_")}


def parameter_sha256(parameters) -> str:
    digest_bytes = bytearray()
    for layer in parameters:
        array = np.ascontiguousarray(np.asarray(layer, dtype="<f4"))
        digest_bytes.extend(array.tobytes(order="C"))
    return sha256_bytes(bytes(digest_bytes))


def assess_saved_cell(row) -> str:
    """Return skip, run, or stop. A failed or non-finite row is not rerun."""
    if row is None:
        return "run"
    status = str(row.get("status", ""))
    if status in {"failed", "nonfinite"}:
        return "stop"
    if status != "complete":
        return "run"
    if row.get("kind") != "tuning" or row.get("decision_cell") is not False:
        return "stop"
    if row.get("finite") is not True or row.get("steps_checked") is not True:
        return "stop"
    if row.get("seed") != TUNING_SEED or int(row.get("rounds", 0)) != ROUNDS:
        return "stop"
    try:
        final = float(row["Final_F1"])
    except (KeyError, TypeError, ValueError):
        return "stop"
    if not math.isfinite(final):
        return "stop"
    return "skip"


def _assert_block_map() -> None:
    for dataset, blocks in DATASET_BLOCKS.items():
        if tuple(block for block, name in SCALER_FOR_BLOCK.items() if name == dataset) != blocks:
            raise TuningStop(f"block map for {dataset} does not match SCALER_FOR_BLOCK")
        for block in blocks:
            if block not in BLOCK_SPEC or BLOCK_SPEC[block]["dataset"] != dataset:
                raise TuningStop(f"block spec missing {block}")
    if tuple(BLOCK_SPEC) != BLOCKS:
        raise TuningStop("block order is not T1, T2, T5, T6")


def _cover_indices(subtrain_idx: np.ndarray, val_idx: np.ndarray, n_train: int) -> None:
    if int(subtrain_idx.size + val_idx.size) != int(n_train):
        raise PreflightStop("tuning split", "subtrain and validation do not cover the training rows")
    seen = np.zeros(int(n_train), dtype=np.int8)
    for label, indices in (("subtrain", subtrain_idx), ("validation", val_idx)):
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= int(n_train)):
            raise PreflightStop("tuning split", f"{label} index is outside the training rows")
        if np.unique(indices).size != indices.size or bool(seen[indices].any()):
            raise PreflightStop("tuning split", f"{label} index overlaps another row")
        seen[indices] = 1
    if int(seen.sum()) != int(n_train):
        raise PreflightStop("tuning split", "subtrain and validation do not cover the training rows")


def _fit_dataset(root: Path, dataset: str) -> dict:
    """Byte-check the sealed cache, then fit the tuning scaler on unscaled subtrain rows."""
    rebuilt = reconstruct_dataset(root, dataset, keep_unscaled=True, purpose="prepare")
    x_unscaled = np.ascontiguousarray(rebuilt["X_unscaled_train"], dtype=np.float32)
    y_train = np.ascontiguousarray(rebuilt["y_train"], dtype=np.int64)
    cid_train = np.ascontiguousarray(rebuilt["cid_train"], dtype=np.int64)
    del rebuilt
    gc.collect()
    n_train = int(y_train.shape[0])
    if n_train != TRAIN_ROWS[dataset] or int(x_unscaled.shape[0]) != n_train:
        raise PreflightStop(dataset, f"training row count {n_train} != {TRAIN_ROWS[dataset]}")
    if int(x_unscaled.shape[1]) != FEATURE_COUNTS[dataset]:
        raise PreflightStop(dataset, "feature count mismatch on the unscaled training matrix")
    require_finite(x_unscaled, "unscaled training matrix")
    classes = np.unique(y_train)
    if int(classes.size) != CLASS_COUNTS[dataset] or int(classes[0]) != 0 or int(classes[-1]) != CLASS_COUNTS[dataset] - 1:
        raise PreflightStop(dataset, f"class ids are not 0..{CLASS_COUNTS[dataset] - 1}")
    n_classes = CLASS_COUNTS[dataset]
    if dataset == "CIC-IoT2023":
        strata = y_train
    elif dataset == "N-BaIoT":
        strata = cid_train * (n_classes + 1) + y_train
    else:
        raise PreflightStop(dataset, "unknown dataset")
    try:
        require_two_rows_per_stratum(strata)
    except ValueError as exc:
        raise PreflightStop(dataset, str(exc)) from exc
    indices = np.arange(n_train, dtype=np.int64)
    subtrain_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        train_size=None,
        shuffle=True,
        random_state=TUNING_SEED,
        stratify=strata,
    )
    subtrain_idx = np.ascontiguousarray(np.asarray(subtrain_idx, dtype="<i8"))
    val_idx = np.ascontiguousarray(np.asarray(val_idx, dtype="<i8"))
    n_sub, n_val = tuning_holdout_sizes(n_train)
    if int(subtrain_idx.size) != n_sub or int(val_idx.size) != n_val:
        raise PreflightStop(
            dataset,
            f"split sizes {subtrain_idx.size}/{val_idx.size} != {n_sub}/{n_val}",
        )
    _cover_indices(subtrain_idx, val_idx, n_train)
    if dataset == "N-BaIoT":
        full_ids = {int(value) for value in np.unique(cid_train).tolist()}
        sub_ids = {int(value) for value in np.unique(cid_train[subtrain_idx]).tolist()}
        if sub_ids != full_ids:
            raise PreflightStop(dataset, "a sealed training device id is missing from subtrain")
    print(f"E5 prepare: {dataset} split subtrain={n_sub} validation={n_val}", flush=True)
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_sub = np.ascontiguousarray(scaler.fit_transform(x_unscaled[subtrain_idx]).astype(np.float32))
    x_val = np.ascontiguousarray(scaler.transform(x_unscaled[val_idx]).astype(np.float32))
    seen = exact_nonnegative_int(scaler.n_samples_seen_, "n_samples_seen_")
    if seen != n_sub:
        raise PreflightStop(dataset, f"n_samples_seen_ {seen} != n_subtrain {n_sub}")
    mean = np.ascontiguousarray(np.asarray(scaler.mean_, dtype="<f8"))
    var = np.ascontiguousarray(np.asarray(scaler.var_, dtype="<f8"))
    scale = np.ascontiguousarray(np.asarray(scaler.scale_, dtype="<f8"))
    if mean.shape != (FEATURE_COUNTS[dataset],):
        raise PreflightStop(dataset, "scaler mean_ shape mismatch")
    y_sub = np.ascontiguousarray(np.asarray(y_train[subtrain_idx], dtype=np.int64))
    y_val = np.ascontiguousarray(np.asarray(y_train[val_idx], dtype=np.int64))
    if dataset == "CIC-IoT2023":
        partition = engine.partition_dirichlet(y_sub, 9, 0.1, seed=TUNING_SEED)
    else:
        try:
            partition = partition_by_device_ids(cid_train[subtrain_idx], num_clients=9)
        except ValueError as exc:
            raise PreflightStop(dataset, str(exc)) from exc
    partition = {
        client_id: np.ascontiguousarray(np.asarray(partition[client_id], dtype="<i8"))
        for client_id in range(9)
    }
    if any(int(partition[client_id].size) == 0 for client_id in range(9)):
        raise PreflightStop(dataset, "a tuning client is empty")
    try:
        validate_compact_partition(partition, n_sub)
    except ValueError as exc:
        raise PreflightStop(dataset, str(exc)) from exc
    for client_id in range(9):
        rows = partition[client_id]
        via_scaler = np.ascontiguousarray(
            scaler.transform(x_unscaled[subtrain_idx[rows]]).astype(np.float32)
        )
        direct = np.ascontiguousarray(x_sub[rows])
        if not arrays_byte_equal(direct, via_scaler):
            raise PreflightStop(dataset, f"client {client_id} slice does not match the shared scaler")
        del via_scaler, direct
    del scaler, x_unscaled, y_train, cid_train, strata, indices
    gc.collect()
    require_finite(x_sub, "X_subtrain")
    require_finite(x_val, "X_val")
    sub_hash = index_sha256(subtrain_idx)
    val_hash = index_sha256(val_idx)
    combined = combined_index_sha256(subtrain_idx, val_idx)
    digest = scaler_sha256(
        mean_=mean,
        var_=var,
        scale_=scale,
        n_samples_seen_=seen,
        feature_count=FEATURE_COUNTS[dataset],
        subtrain_index_sha256=sub_hash,
        validation_index_sha256=val_hash,
    )
    return {
        "dataset": dataset,
        "n_subtrain": n_sub,
        "n_validation": n_val,
        "n_train": n_train,
        "mean_": mean,
        "var_": var,
        "scale_": scale,
        "n_samples_seen_": seen,
        "subtrain_idx": subtrain_idx,
        "val_idx": val_idx,
        "subtrain_index_sha256": sub_hash,
        "validation_index_sha256": val_hash,
        "combined_sha256": combined,
        "scaler_sha256": digest,
        "partition": partition,
        "X_subtrain": x_sub,
        "y_subtrain": y_sub,
        "X_val": x_val,
        "y_val": y_val,
    }


def _write_dataset(root: Path, built: dict) -> dict:
    dataset = built["dataset"]
    scaler_npz, scaler_json = SCALER_FILES[dataset]
    atomic_savez(
        root / scaler_npz,
        mean_=built["mean_"],
        var_=built["var_"],
        scale_=built["scale_"],
    )
    atomic_write_json(
        root / scaler_json,
        {
            "dtype": "<f8",
            "feature_count": FEATURE_COUNTS[dataset],
            "n_samples_seen_": built["n_samples_seen_"],
            "scaler_sha256": built["scaler_sha256"],
            "subtrain_index_sha256": built["subtrain_index_sha256"],
            "validation_index_sha256": built["validation_index_sha256"],
        },
    )
    index_npz, index_json = tuning_index_files(dataset)
    atomic_savez(root / index_npz, subtrain_idx=built["subtrain_idx"], val_idx=built["val_idx"])
    atomic_write_json(
        root / index_json,
        {
            "combined_sha256": built["combined_sha256"],
            "dataset": dataset,
            "dtype": "<i8",
            "n_subtrain": built["n_subtrain"],
            "n_train": built["n_train"],
            "n_validation": built["n_validation"],
            "seed": TUNING_SEED,
            "subtrain_index_sha256": built["subtrain_index_sha256"],
            "validation_index_sha256": built["validation_index_sha256"],
        },
    )
    matrix_npz, matrix_json = tuning_matrix_files(dataset)
    atomic_savez(
        root / matrix_npz,
        X_subtrain=built["X_subtrain"],
        X_val=built["X_val"],
        y_subtrain=built["y_subtrain"],
        y_val=built["y_val"],
    )
    matrix_sha = sha256_file(root / matrix_npz)
    atomic_write_json(
        root / matrix_json,
        {
            "dataset": dataset,
            "dtype_x": "<f4",
            "dtype_y": "<i8",
            "kind": "section-4.4-transformed-matrix-cache",
            "note": "Derived cache of the Section 4.4 transform. Not a sealed cache.",
            "seed": TUNING_SEED,
            "sha256": matrix_sha,
            "X_subtrain_shape": [built["n_subtrain"], FEATURE_COUNTS[dataset]],
            "X_val_shape": [built["n_validation"], FEATURE_COUNTS[dataset]],
        },
    )
    document = partition_document(built["partition"], built["n_subtrain"], 9)
    stored = {
        f"c{client_id}": built["partition"][client_id] for client_id in range(9)
    }
    for block in DATASET_BLOCKS[dataset]:
        npz_rel, json_rel = tuning_partition_files(block)
        atomic_savez(root / npz_rel, **stored)
        atomic_write_json(root / json_rel, document)
    checked = _dataset_record(root, dataset)
    if checked["partition_sha256"] != document["sha256"]:
        raise PreflightStop(dataset, "saved tuning partition hash mismatch")
    if checked["scaler_sha256"] != built["scaler_sha256"]:
        raise PreflightStop(dataset, "saved scaler hash mismatch")
    if checked["combined_index_sha256"] != built["combined_sha256"]:
        raise PreflightStop(dataset, "saved index hash mismatch")
    if checked["matrix_sha256"] != matrix_sha:
        raise PreflightStop(dataset, "saved matrix hash mismatch")
    print(
        f"E5 prepare: {dataset} artifacts verified "
        f"partition={checked['partition_sha256']} scaler={checked['scaler_sha256']}",
        flush=True,
    )
    return checked


def _dataset_record(root: Path, dataset: str) -> dict:
    n_sub, n_val = tuning_holdout_sizes(TRAIN_ROWS[dataset])
    scaler_digest = verify_scaler_artifact(
        root / SCALER_FILES[dataset][0],
        root / SCALER_FILES[dataset][1],
        feature_count=FEATURE_COUNTS[dataset],
    )
    scaler_doc = json.loads((root / SCALER_FILES[dataset][1]).read_text(encoding="utf-8"))
    if exact_nonnegative_int(scaler_doc["n_samples_seen_"], "n_samples_seen_") != n_sub:
        raise PreflightStop(dataset, "n_samples_seen_ is not n_subtrain")
    index_npz, index_json = tuning_index_files(dataset)
    index_doc = _read_json(root / index_json)
    if index_doc is None:
        raise PreflightStop(root / index_json, "tuning index file missing")
    with np.load(root / index_npz, allow_pickle=False) as blob:
        subtrain_idx = np.array(blob["subtrain_idx"], copy=True)
        val_idx = np.array(blob["val_idx"], copy=True)
    sub_hash = index_sha256(subtrain_idx)
    val_hash = index_sha256(val_idx)
    combined = combined_index_sha256(subtrain_idx, val_idx)
    if sub_hash != str(index_doc.get("subtrain_index_sha256", "")).lower():
        raise PreflightStop(index_json, "subtrain index hash mismatch")
    if val_hash != str(index_doc.get("validation_index_sha256", "")).lower():
        raise PreflightStop(index_json, "validation index hash mismatch")
    if combined != str(index_doc.get("combined_sha256", "")).lower():
        raise PreflightStop(index_json, "combined index hash mismatch")
    if int(index_doc.get("n_subtrain", -1)) != n_sub or int(index_doc.get("n_validation", -1)) != n_val:
        raise PreflightStop(index_json, "index file row counts mismatch")
    if sub_hash != str(scaler_doc["subtrain_index_sha256"]).lower():
        raise PreflightStop(dataset, "scaler and index subtrain hashes differ")
    if val_hash != str(scaler_doc["validation_index_sha256"]).lower():
        raise PreflightStop(dataset, "scaler and index validation hashes differ")
    matrix_npz, matrix_json = tuning_matrix_files(dataset)
    matrix_doc = _read_json(root / matrix_json)
    if matrix_doc is None or not (root / matrix_npz).is_file():
        raise PreflightStop(matrix_json, "tuning matrix cache missing")
    matrix_sha = sha256_file(root / matrix_npz)
    if matrix_sha != str(matrix_doc.get("sha256", "")).lower():
        raise PreflightStop(matrix_json, "matrix cache hash mismatch")
    if matrix_doc.get("X_subtrain_shape") != [n_sub, FEATURE_COUNTS[dataset]]:
        raise PreflightStop(matrix_json, "matrix cache shape mismatch")
    if matrix_doc.get("X_val_shape") != [n_val, FEATURE_COUNTS[dataset]]:
        raise PreflightStop(matrix_json, "validation matrix shape mismatch")
    digests = []
    for block in DATASET_BLOCKS[dataset]:
        npz_rel, json_rel = tuning_partition_files(block)
        digests.append(
            verify_partition_artifact(root / npz_rel, root / json_rel, compact=True)
        )
    if digests[0] != digests[1]:
        raise PreflightStop(dataset, f"{DATASET_BLOCKS[dataset]} tuning partitions differ")
    return {
        "dataset": dataset,
        "blocks": list(DATASET_BLOCKS[dataset]),
        "n_subtrain": n_sub,
        "n_validation": n_val,
        "scaler_sha256": scaler_digest,
        "combined_index_sha256": combined,
        "subtrain_index_sha256": sub_hash,
        "validation_index_sha256": val_hash,
        "matrix_sha256": matrix_sha,
        "partition_sha256": digests[0],
    }


def _cells_exist(root: Path, dataset: str) -> bool:
    for block in DATASET_BLOCKS[dataset]:
        for rate_token in TUNING_RATES:
            for arm in ARMS:
                row_rel, _history_rel = _cell_rels(block, arm, rate_token)
                if (root / row_rel).is_file():
                    return True
    return False


def _clear_dataset(root: Path, dataset: str) -> None:
    for relative in _dataset_files(dataset):
        path = root / relative
        if path.is_file():
            path.unlink()


def prepare_tuning_artifacts(root=None) -> dict:
    """Write both scalers, both index files, the matrix caches, and four tuning partitions."""
    root = PAPER if root is None else Path(root)
    _assert_block_map()
    before = _decision_names(root)
    records = {}
    pending = []
    for dataset in ("CIC-IoT2023", "N-BaIoT"):
        try:
            records[dataset] = _dataset_record(root, dataset)
            print(f"E5 prepare: {dataset} artifacts already verify", flush=True)
        except (PreflightStop, OSError, ValueError, KeyError):
            if _cells_exist(root, dataset):
                raise PreflightStop(dataset, "tuning cells exist but the artifacts do not verify")
            pending.append(dataset)
    if pending:
        verify_bound_hashes(root)
        verify_environment()
        apply_and_read_cuda_flags()
        import_bound_parser("fl_local.cic_cache")
        import_bound_parser("fl_local.nbaiot_cache")
        for dataset in pending:
            _clear_dataset(root, dataset)
            built = _fit_dataset(root, dataset)
            records[dataset] = _write_dataset(root, built)
            del built
            gc.collect()
    if _decision_names(root) != before:
        raise PreflightStop("partitions", "prepare wrote a decision partition file")
    manifest = {
        "decision_partitions_written": False,
        "datasets": records,
        "gpu_training": False,
        "kind": "tuning-artifacts",
        "seed": TUNING_SEED,
        "written_utc": _utc(),
    }
    manifest_path = root / "q1_e5_results" / "tuning" / "ARTIFACTS.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _block_bindings(root: Path, block: str) -> dict:
    dataset = SCALER_FOR_BLOCK[block]
    record = _dataset_record(root, dataset)
    npz_rel, json_rel = tuning_partition_files(block)
    partition_sha = verify_partition_artifact(root / npz_rel, root / json_rel, compact=True)
    if partition_sha != record["partition_sha256"]:
        raise TuningStop(f"{block} partition hash is not the shared {dataset} hash")
    record["partition_sha256"] = partition_sha
    record["block"] = block
    return record


def _load_partition(root: Path, block: str, n_subtrain: int) -> dict:
    npz_rel, json_rel = tuning_partition_files(block)
    digest = verify_partition_artifact(root / npz_rel, root / json_rel, compact=True)
    with np.load(root / npz_rel, allow_pickle=False) as blob:
        partition = {
            client_id: np.ascontiguousarray(blob[f"c{client_id}"], dtype=np.int64)
            for client_id in range(9)
        }
    validate_compact_partition(partition, n_subtrain)
    if hash_partition(partition) != digest:
        raise TuningStop(f"{block} reloaded partition hash mismatch")
    return partition


def _load_matrices(root: Path, dataset: str, matrix_sha256: str) -> dict:
    matrix_npz, _matrix_json = tuning_matrix_files(dataset)
    if sha256_file(root / matrix_npz) != matrix_sha256:
        raise TuningStop(f"{dataset} matrix hash mismatch")
    n_sub, n_val = tuning_holdout_sizes(TRAIN_ROWS[dataset])
    features = FEATURE_COUNTS[dataset]
    with np.load(root / matrix_npz, allow_pickle=False) as blob:
        x_sub = np.ascontiguousarray(blob["X_subtrain"], dtype=np.float32)
        y_sub = np.ascontiguousarray(blob["y_subtrain"], dtype=np.int64)
        x_val = np.ascontiguousarray(blob["X_val"], dtype=np.float32)
        y_val = np.ascontiguousarray(blob["y_val"], dtype=np.int64)
    if x_sub.shape != (n_sub, features) or x_val.shape != (n_val, features):
        raise TuningStop(f"{dataset} matrix shape mismatch")
    if y_sub.shape != (n_sub,) or y_val.shape != (n_val,):
        raise TuningStop(f"{dataset} label shape mismatch")
    require_finite(x_sub, "X_subtrain")
    require_finite(x_val, "X_val")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    return {
        "dataset": dataset,
        "X_subtrain": x_sub,
        "y_subtrain": y_sub,
        "loader": loader,
    }


def _reference_init(root: Path, block: str, *, attempt: str | None = None) -> str | None:
    hashes = []
    init_doc = _read_json(root / _init_rel(block, attempt=attempt))
    if init_doc is not None:
        hashes.append(str(init_doc["initial_parameter_sha256"]))
    for rate_token in TUNING_RATES:
        for arm in ARMS:
            row = _read_json(root / _cell_rels(block, arm, rate_token, attempt=attempt)[0])
            if row is not None and row.get("status") == "complete":
                hashes.append(str(row["initial_parameter_sha256"]))
    if not hashes:
        return None
    if any(item != hashes[0] for item in hashes):
        raise TuningStop(f"{block} initial-parameter hashes disagree")
    return hashes[0]


def _quarantine(path: Path, document: dict, status: str, reason: str) -> None:
    document["status"] = status
    document["finite"] = False
    document["steps_checked"] = False
    document["decision_cell"] = False
    document["reason"] = reason
    document["finished_utc"] = _utc()
    atomic_write_json(path, document)


def _history_bytes(history) -> bytes:
    return (json.dumps(history, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def cell_disposition(
    root: Path,
    block: str,
    arm: str,
    rate_token: str,
    bindings: dict,
    *,
    attempt: str | None = None,
) -> str:
    row_rel, history_rel = _cell_rels(block, arm, rate_token, attempt=attempt)
    row = _read_json(root / row_rel)
    decision = assess_saved_cell(row)
    if decision != "skip":
        return decision
    recorded_attempt = row.get("attempt")
    if attempt == ATTEMPT2 and recorded_attempt != ATTEMPT2:
        raise TuningStop(f"{block} {arm} lr={rate_token} is not an attempt2 cell")
    if attempt is None and recorded_attempt is not None:
        raise TuningStop(f"{block} {arm} lr={rate_token} belongs to {recorded_attempt}")
    identity = (row.get("block"), row.get("arm"), row.get("learning_rate"), row.get("model"))
    expected = (block, arm, rate_token, BLOCK_SPEC[block]["model"])
    if identity != expected:
        raise TuningStop(f"{block} {arm} lr={rate_token} row identity mismatch")
    for key, label in (
        ("partition_sha256", "partition"),
        ("scaler_sha256", "scaler"),
        ("combined_index_sha256", "index"),
        ("matrix_sha256", "matrix"),
    ):
        if str(row.get(key, "")).lower() != bindings[key]:
            raise TuningStop(f"{block} {arm} lr={rate_token} {label} hash mismatch")
    history_path = root / history_rel
    if not history_path.is_file() or sha256_file(history_path) != str(row.get("history_sha256", "")).lower():
        raise TuningStop(f"{block} {arm} lr={rate_token} history hash mismatch")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if len(history) != ROUNDS or int(history[-1]["round"]) != ROUNDS:
        raise TuningStop(f"{block} {arm} lr={rate_token} history does not end at round 30")
    if history[-1]["macro_f1"] != row["Final_F1"]:
        raise TuningStop(f"{block} {arm} lr={rate_token} Final_F1 is not round 30")
    for item in history:
        for key in ("loss", "accuracy", "macro_f1"):
            if not math.isfinite(float(item[key])):
                raise TuningStop(f"{block} {arm} lr={rate_token} history is non-finite")
    return "skip"


def _train_one_cell(
    root: Path,
    block: str,
    arm: str,
    rate_token: str,
    bindings: dict,
    matrices: dict,
    partition: dict,
    *,
    attempt: str | None = None,
) -> None:
    if momentum_sgd_client_update is not FEDNOVA_CLIENT_UPDATE or momentum_sgd_client_update is not FEDAVG_CLIENT_UPDATE:
        raise TuningStop("FedNova and FedAvg must share one client-update function")
    dataset = bindings["dataset"]
    spec = BLOCK_SPEC[block]
    row_rel, history_rel = _cell_rels(block, arm, rate_token, attempt=attempt)
    row_path = root / row_rel
    history_path = root / history_rel
    device = torch.device("cuda")
    engine.set_global_seed(TUNING_SEED)
    model = engine.make_model(spec["model"], FEATURE_COUNTS[dataset], CLASS_COUNTS[dataset], device)
    params = [np.array(layer, dtype=np.float32, copy=True) for layer in engine.get_parameters_np(model)]
    init_hash = parameter_sha256(params)
    if attempt == ATTEMPT2:
        preserved = _read_json(root / _init_rel(block))
        if preserved is not None and init_hash != str(preserved["initial_parameter_sha256"]):
            raise TuningStop(f"{block} attempt2 initial-parameter hash does not match the preserved attempt")
    reference = _reference_init(root, block, attempt=attempt)
    if reference is not None and init_hash != reference:
        raise TuningStop(f"{block} initial-parameter hash changed")
    atomic_write_json(
        root / _init_rel(block, attempt=attempt),
        {"block": block, "initial_parameter_sha256": init_hash, "model": spec["model"], "seed": TUNING_SEED},
    )
    sizes = [int(partition[client_id].size) for client_id in range(9)]
    weights = weights_for_run(
        sizes,
        run_kind="tuning",
        subtrain_rows=bindings["n_subtrain"],
        full_training_rows=TRAIN_ROWS[dataset],
    )
    generators = [client_generator(TUNING_SEED, client_id) for client_id in range(9)]
    document = {
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "block": block,
        "combined_index_sha256": bindings["combined_index_sha256"],
        "dataset": dataset,
        "decision_cell": False,
        "initial_parameter_sha256": init_hash,
        "kind": "tuning",
        "learning_rate": rate_token,
        "matrix_sha256": bindings["matrix_sha256"],
        "model": spec["model"],
        "partition_sha256": bindings["partition_sha256"],
        "scaler_sha256": bindings["scaler_sha256"],
        "seed": TUNING_SEED,
        "started_utc": _utc(),
        "status": "running",
    }
    if attempt is not None:
        document["attempt"] = attempt
    atomic_write_json(row_path, document)
    _log(
        root,
        f"CELL_START block={block} arm={arm} lr={rate_token} seed={TUNING_SEED} init={init_hash}",
        attempt=attempt,
    )
    history = []
    lr = learning_rate_value(rate_token)
    try:
        for rnd in range(1, ROUNDS + 1):
            _log(root, f"ROUND_START block={block} arm={arm} lr={rate_token} round={rnd}", attempt=attempt)
            deltas = []
            areas = []
            for client_id in range(9):
                rows = partition[client_id]
                result = momentum_sgd_client_update(
                    params,
                    matrices["X_subtrain"][rows],
                    matrices["y_subtrain"][rows],
                    lr=lr,
                    generator=generators[client_id],
                    model_name=spec["model"],
                    num_features=FEATURE_COUNTS[dataset],
                    num_classes=CLASS_COUNTS[dataset],
                    device=device,
                    batch_size=BATCH_SIZE,
                    local_epochs=LOCAL_EPOCHS,
                )
                if result["momentum_state_at_start"] != 0:
                    raise TuningStop("momentum state was not reset")
                deltas.append(result["delta"])
                areas.append(result["A_i"])
            if arm == "FedAvg":
                params = fedavg_server_update(params, deltas, weights, areas)
            elif arm == "FedNova":
                params = fednova_server_update(params, deltas, weights, areas)
            else:
                raise TuningStop(f"unknown arm {arm}")
            for layer in params:
                require_finite(layer, "server parameters")
            engine.set_parameters_np(model, params)
            loss, acc, f1 = engine.evaluate_model_nb(model, matrices["loader"], device)
            if not (math.isfinite(float(loss)) and math.isfinite(float(acc)) and math.isfinite(float(f1))):
                raise TuningStop(f"round {rnd} validation score is non-finite")
            stored_f1 = repr(float(f1))
            history.append(
                {
                    "accuracy": repr(float(acc)),
                    "loss": repr(float(loss)),
                    "macro_f1": stored_f1,
                    "round": int(rnd),
                }
            )
            _log(
                root,
                f"ROUND block={block} arm={arm} lr={rate_token} round={rnd} macro_f1={stored_f1} finite=true",
                attempt=attempt,
            )
    except TuningStop as exc:
        payload = _history_bytes(history)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_bytes(payload)
        status = "nonfinite" if "non-finite" in str(exc) else "failed"
        document["history_sha256"] = sha256_bytes(payload)
        document["rounds_completed"] = len(history)
        _quarantine(row_path, document, status, str(exc))
        raise
    except Exception as exc:
        payload = _history_bytes(history)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_bytes(payload)
        document["history_sha256"] = sha256_bytes(payload)
        document["rounds_completed"] = len(history)
        _quarantine(row_path, document, "failed", f"{type(exc).__name__}: {exc}")
        raise TuningStop(f"{block} {arm} lr={rate_token} failed: {type(exc).__name__}: {exc}") from exc
    payload = _history_bytes(history)
    history_path.write_bytes(payload)
    document.update(
        {
            "Final_Acc": history[-1]["accuracy"],
            "Final_F1": history[-1]["macro_f1"],
            "finished_utc": _utc(),
            "finite": True,
            "history_sha256": sha256_bytes(payload),
            "rounds": ROUNDS,
            "status": "complete",
            "steps_checked": True,
        }
    )
    atomic_write_json(row_path, document)
    _log(
        root,
        f"CELL block={block} arm={arm} lr={rate_token} status=complete final_f1={document['Final_F1']}",
        attempt=attempt,
    )


def _scores_from_rows(root: Path, block: str, *, attempt: str | None = None) -> dict:
    scores = {}
    for rate_token in TUNING_RATES:
        scores[rate_token] = {}
        for arm in ARMS:
            row = _read_json(root / _cell_rels(block, arm, rate_token, attempt=attempt)[0])
            if assess_saved_cell(row) != "skip":
                raise TuningStop(f"{block} cannot lock because {arm} lr={rate_token} is not a complete finite run")
            scores[rate_token][arm] = row["Final_F1"]
    return scores


def reload_tuning_cell(root: Path, block: str) -> tuple[dict, dict]:
    """Reload the scaler, index files, and partition arrays for one cell."""
    bindings = _block_bindings(root, block)
    partition = _load_partition(root, block, bindings["n_subtrain"])
    return bindings, partition


def load_cell_for_training(
    root: Path,
    block: str,
    arm: str,
    rate_token: str,
    *,
    attempt: str | None = None,
):
    """Reload scaler, indices, and partition immediately before this cell."""
    bindings, partition = reload_tuning_cell(root, block)
    decision = cell_disposition(root, block, arm, rate_token, bindings, attempt=attempt)
    return decision, bindings, partition


_CELL_ARTIFACT_KEYS = (
    "scaler_sha256",
    "combined_index_sha256",
    "partition_sha256",
    "matrix_sha256",
)


def scan_block_cells(root: Path, block: str) -> list:
    """Reload artifacts once per cell. A failed or non-finite cell stops the block."""
    plan = []
    for rate_token in TUNING_RATES:
        for arm in ARMS:
            decision, bindings, partition = load_cell_for_training(root, block, arm, rate_token)
            if decision == "stop":
                raise TuningStop(
                    f"{block} {arm} lr={rate_token} is failed or non-finite and will not be replaced"
                )
            plan.append((rate_token, arm, decision, bindings, partition))
    identities = [tuple(item[3][key] for key in _CELL_ARTIFACT_KEYS) for item in plan]
    if any(identity != identities[0] for identity in identities):
        raise TuningStop(f"{block} scaler, index, or partition hash changed between cells")
    return plan


def _write_lock(root: Path, block: str, bindings: dict, *, attempt: str | None = None) -> str:
    scores = _scores_from_rows(root, block, attempt=attempt)
    locked = lock_learning_rate(scores)
    means = {
        rate_token: repr(pair_mean(scores[rate_token]["FedAvg"], scores[rate_token]["FedNova"]))
        for rate_token in TUNING_RATES
    }
    document = {
        "block": block,
        "dataset": bindings["dataset"],
        "decision_cells_started": False,
        "kind": "tuning-lock",
        "locked_learning_rate": locked,
        "means": means,
        "scores": scores,
        "seed": TUNING_SEED,
        "tie_rule": "exact-float64-equality-keeps-smaller-rate",
        "written_utc": _utc(),
    }
    if attempt is not None:
        document["attempt"] = attempt
    path = root / _lock_rel(block, attempt=attempt)
    existing = _read_json(path)
    if existing is not None:
        same = (
            existing.get("locked_learning_rate") == locked
            and existing.get("means") == means
            and existing.get("scores") == scores
            and existing.get("decision_cells_started") is False
        )
        if not same:
            raise TuningStop(f"{block} existing lock does not match the six stored scores")
        return locked
    atomic_write_json(path, document)
    _log(root, f"LOCK block={block} learning_rate={locked} means={means}", attempt=attempt)
    return locked


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    # os.kill(pid, 0) raises WinError 87 on this Windows Python even when the process is alive.
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, 0, int(pid))
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def _acquire_pid(root: Path, *, attempt: str | None = None) -> Path:
    path = root / _tuning_home(attempt) / "tune.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            old = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            old = 0
        if old and old != os.getpid() and _alive(old):
            raise TuningStop(f"tune process {old} is already running")
    path.write_text(str(os.getpid()), encoding="utf-8")
    return path


def _refuse_if_other_tune_is_alive(root: Path, attempt: str | None) -> None:
    other = None if attempt == ATTEMPT2 else ATTEMPT2
    path = root / _tuning_home(other) / "tune.pid"
    if not path.is_file():
        return
    try:
        old = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    if old and old != os.getpid() and _alive(old):
        raise TuningStop(f"tune process {old} is already running")


def _closed_preserved_cell(root: Path) -> str | None:
    """Return a preserved cell name that plain tune must not resume or replace."""
    folder = root / "q1_e5_results" / "tuning" / "cells"
    if not folder.is_dir():
        return None
    for path in sorted(folder.glob("*.json")):
        if path.name.endswith("_history.json"):
            continue
        row = _read_json(path)
        if isinstance(row, dict) and assess_saved_cell(row) == "stop":
            return path.name
    return None


def snapshot_original_attempt(root: Path) -> dict:
    """Hash the preserved attempt's cell, init, lock, and progress files."""
    home = root / "q1_e5_results" / "tuning"
    snaps = {}
    for relative in ("cells", "init", "locks"):
        folder = home / relative
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                snaps[str(path.relative_to(home)).replace("\\", "/")] = sha256_file(path)
    progress = home / "progress.log"
    if progress.is_file():
        snaps["progress.log"] = sha256_file(progress)
    return snaps


def assert_original_attempt_unchanged(root: Path, snapshot: dict) -> None:
    current = snapshot_original_attempt(root)
    if current != snapshot:
        raise TuningStop("the preserved 2026-10-06 tuning attempt was modified")


def _artifact_identity(bindings: dict) -> tuple:
    return tuple(bindings[key] for key in _CELL_ARTIFACT_KEYS)


def _run_tuning_blocks(root: Path, blocks: tuple[str, ...], *, attempt: str | None, order_label: str) -> dict:
    """Reload artifacts immediately before each cell, then train or skip that cell.

    A failed or non-finite cell stops the block. A lock is written only after
    every cell in the block is a complete finite run. No decision cell starts.
    """
    _assert_block_map()
    if attempt == ATTEMPT2 and blocks != ("T1",):
        raise TuningStop("attempt2 trains T1 only")
    if attempt is None:
        closed = _closed_preserved_cell(root)
        if closed is not None:
            raise TuningStop(
                f"the preserved tuning attempt has a closed cell ({closed}); "
                "plain tune will not resume it, replace it, or modify its files"
            )
    _refuse_if_other_tune_is_alive(root, attempt)
    pid_path = _acquire_pid(root, attempt=attempt)
    snapshot = snapshot_original_attempt(root) if attempt == ATTEMPT2 else None
    cache = {"dataset": None, "matrices": None}
    locks = {}
    try:
        verify_bound_hashes(root)
        verify_environment()
        apply_and_read_cuda_flags()
        if not torch.cuda.is_available():
            raise TuningStop("CUDA is not available")
        _log(
            root,
            f"TUNE_START utc={_utc()} pid={os.getpid()} order={order_label} "
            f"arms=FedAvg,FedNova rates=0.001,0.01,0.1 attempt={attempt or 'preserved'}",
            attempt=attempt,
        )
        for block in blocks:
            dataset = BLOCK_SPEC[block]["dataset"]
            identity = None
            bindings = None
            for rate_token in TUNING_RATES:
                for arm in ARMS:
                    decision, bindings, partition = load_cell_for_training(
                        root, block, arm, rate_token, attempt=attempt
                    )
                    cell_identity = _artifact_identity(bindings)
                    if identity is None:
                        identity = cell_identity
                    elif cell_identity != identity:
                        raise TuningStop(f"{block} scaler, index, or partition hash changed between cells")
                    if decision == "stop":
                        raise TuningStop(
                            f"{block} {arm} lr={rate_token} is failed or non-finite and will not be replaced"
                        )
                    if decision == "skip":
                        _log(root, f"CELL_SKIP block={block} arm={arm} lr={rate_token}", attempt=attempt)
                    else:
                        if cache["dataset"] != dataset:
                            cache["matrices"] = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            cache["matrices"] = _load_matrices(root, dataset, bindings["matrix_sha256"])
                            cache["dataset"] = dataset
                        _train_one_cell(
                            root,
                            block,
                            arm,
                            rate_token,
                            bindings,
                            cache["matrices"],
                            partition,
                            attempt=attempt,
                        )
                    if snapshot is not None:
                        assert_original_attempt_unchanged(root, snapshot)
            if bindings is None:
                raise TuningStop(f"{block} has no tuning cells")
            locks[block] = _write_lock(root, block, bindings, attempt=attempt)
            if snapshot is not None:
                assert_original_attempt_unchanged(root, snapshot)
        _log(
            root,
            "TUNING_DONE " + " ".join(f"{block}={locks[block]}" for block in blocks),
            attempt=attempt,
        )
        return {
            "attempt": attempt,
            "blocks": list(blocks),
            "decision_cells_started": False,
            "locked_learning_rates": locks,
            "smoke_cell_started": False,
            "status": "complete",
        }
    except Exception as exc:
        if snapshot is not None:
            try:
                assert_original_attempt_unchanged(root, snapshot)
            except TuningStop as preserved:
                raise TuningStop(f"{preserved}; while handling {type(exc).__name__}: {exc}") from exc
        try:
            _log(root, f"FAILED utc={_utc()} error={type(exc).__name__}: {exc}", attempt=attempt)
        except Exception:
            pass
        raise
    finally:
        cache["matrices"] = None
        gc.collect()
        if pid_path.is_file() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink(missing_ok=True)


def run_tuning(root=None) -> dict:
    """Run or resume the 24 validation cells. Does not start a decision cell."""
    root = PAPER if root is None else Path(root)
    return _run_tuning_blocks(root, BLOCKS, attempt=None, order_label="T1,T2,T5,T6")


def run_t1_attempt2(root=None) -> dict:
    """Rerun the six T1 cells into tuning/attempt2.

    Rates, seed, arm order, and scoring stay the same. The preserved attempt
    is not modified. T2, T5, T6, the smoke cell, wave, and resume are not started.
    A non-finite cell stops the rerun and does not write a lock.
    """
    root = PAPER if root is None else Path(root)
    return _run_tuning_blocks(root, ("T1",), attempt=ATTEMPT2, order_label="T1")

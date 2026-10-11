"""Pure artifact and metric helpers for the Q1-E7 CIC class decomposition.

This module deliberately has no torch import.  It is safe to import in no-GPU
integrity tests and contains no cache-build or training path.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


CLASS_NAMES = (
    "benign",
    "bruteforce",
    "ddos",
    "dos",
    "mirai",
    "recon",
    "spoofing",
    "web",
)
CLASS_SUPPORTS = (16000, 2613, 700977, 152000, 301416, 16452, 12000, 4966)
N_TEST = sum(CLASS_SUPPORTS)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest().lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().lower()


def _atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp_name = None
    try:
        with _atomic_path(path) as f:
            tmp_name = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, doc: Mapping) -> None:
    atomic_write_text(path, json.dumps(doc, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", prefix=f".{path.name}.",
            suffix=".tmp", dir=path.parent, delete=False
        ) as f:
            tmp_name = f.name
            w = csv.DictWriter(f, fieldnames=list(fieldnames))
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with _atomic_path(path) as f:
            tmp_name = f.name
            np.save(f, array, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with _atomic_path(path) as f:
            tmp_name = f.name
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def validate_partition(
    partition: Mapping[int, np.ndarray], n_total: int, num_clients: int = 9
) -> None:
    arrays = []
    for cid in range(num_clients):
        if cid not in partition:
            raise ValueError(f"partition missing client {cid}")
        idx = np.asarray(partition[cid], dtype=np.int64)
        if idx.ndim != 1 or len(idx) == 0:
            raise ValueError(f"client {cid} partition must be non-empty 1-D")
        if np.any((idx < 0) | (idx >= n_total)):
            raise ValueError(f"client {cid} has out-of-range indices")
        arrays.append(idx)
    joined = np.concatenate(arrays)
    if len(joined) != n_total:
        raise ValueError(f"partition covers {len(joined)} rows, expected {n_total}")
    if len(np.unique(joined)) != n_total:
        raise ValueError("partition indices are not disjoint and exhaustive")


def hash_partition(
    partition: Mapping[int, np.ndarray], num_clients: int = 9
) -> str:
    """Canonical SHA-256 over LE uint64 client/length and ordered LE int64 indices."""
    h = hashlib.sha256()
    for cid in range(num_clients):
        if cid not in partition:
            raise ValueError(f"partition missing client {cid}")
        idx = np.asarray(partition[cid], dtype="<i8")
        if idx.ndim != 1 or len(idx) == 0:
            raise ValueError(f"client {cid} partition must be non-empty 1-D")
        h.update(int(cid).to_bytes(8, "little", signed=False))
        h.update(int(len(idx)).to_bytes(8, "little", signed=False))
        h.update(np.ascontiguousarray(idx).tobytes(order="C"))
    return h.hexdigest().lower()


def validate_truth(y_true: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1 or len(y) != N_TEST:
        raise ValueError(f"y_true length {len(y)} != frozen {N_TEST}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"y_true dtype must be integer, got {y.dtype}")
    if np.any((y < 0) | (y >= len(CLASS_NAMES))):
        raise ValueError("y_true contains labels outside 0..7")
    counts = np.bincount(y.astype(np.int64, copy=False), minlength=len(CLASS_NAMES))
    if counts.tolist() != list(CLASS_SUPPORTS):
        raise ValueError(
            f"y_true supports {counts.tolist()} != frozen {list(CLASS_SUPPORTS)}"
        )
    return y.astype(np.uint8, copy=False)


def validate_prediction(y_pred: np.ndarray) -> np.ndarray:
    y = np.asarray(y_pred)
    if y.ndim != 1 or len(y) != N_TEST:
        raise ValueError(f"y_pred length {len(y)} != frozen {N_TEST}")
    if y.dtype != np.uint8:
        raise ValueError(f"stored y_pred dtype must be uint8, got {y.dtype}")
    if np.any((y < 0) | (y >= len(CLASS_NAMES))):
        raise ValueError("y_pred contains labels outside 0..7")
    return y.astype(np.uint8, copy=False)


def compute_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = validate_truth(y_true)
    yp = validate_prediction(y_pred)
    n_classes = len(CLASS_NAMES)
    encoded = yt.astype(np.int64) * n_classes + yp.astype(np.int64)
    cm = np.bincount(encoded, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    cm = cm.astype(np.int64, copy=False)
    if int(cm.sum()) != N_TEST:
        raise ValueError("confusion matrix total does not equal frozen test size")
    if cm.sum(axis=1).tolist() != list(CLASS_SUPPORTS):
        raise ValueError("confusion matrix row supports do not match frozen supports")
    tp_vec = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.int64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp_vec, predicted, out=np.zeros_like(tp_vec), where=predicted != 0)
    recall = np.divide(tp_vec, support, out=np.zeros_like(tp_vec), where=support != 0)
    fp_vec = predicted - tp_vec
    fn_vec = support.astype(np.float64) - tp_vec
    f1_denom = 2.0 * tp_vec + fp_vec + fn_vec
    f1 = np.divide(2.0 * tp_vec, f1_denom, out=np.zeros_like(tp_vec), where=f1_denom != 0)
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        row = {
            "ClassID": i,
            "Class": name,
            "Support": int(support[i]),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": float(precision[i]),
            "Recall": float(recall[i]),
            "F1": float(f1[i]),
        }
        if not all(math.isfinite(float(row[k])) for k in ("Precision", "Recall", "F1")):
            raise ValueError(f"non-finite class metric for {name}")
        rows.append(row)
    row_den = cm.sum(axis=1, keepdims=True).astype(np.float64)
    cm_norm = cm.astype(np.float64) / row_den
    acc = float(np.trace(cm) / N_TEST)
    macro_f1 = float(np.mean(f1))
    if not math.isfinite(acc) or not math.isfinite(macro_f1):
        raise ValueError("non-finite aggregate metric")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": rows,
        "confusion_raw": cm,
        "confusion_rownorm": cm_norm,
    }


def checkpoint_payload(state: Mapping[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    if not state:
        raise ValueError("empty checkpoint state")
    arrays: dict[str, np.ndarray] = {}
    entries = []
    for i, (name, value) in enumerate(state.items()):
        arr = np.asarray(value)
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"non-numeric dtype forbidden in checkpoint: {name}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"non-finite checkpoint values forbidden: {name}")
        key = f"p{i:03d}"
        arrays[key] = np.ascontiguousarray(arr)
        entries.append(
            {
                "state_key": str(name),
                "array_key": key,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            }
        )
    return {"format": "q1-e7-numpy-state-v1", "state": entries}, arrays


def read_checkpoint(npz_path: Path, spec_path: Path) -> dict[str, np.ndarray]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("format") != "q1-e7-numpy-state-v1":
        raise ValueError("unsupported checkpoint format")
    entries = spec.get("state")
    if not isinstance(entries, list) or not entries:
        raise ValueError("checkpoint state manifest missing")
    recorded_npz_sha = spec.get("npz_sha256")
    if recorded_npz_sha is not None and str(recorded_npz_sha).lower() != sha256_file(npz_path):
        raise ValueError("checkpoint NPZ hash differs from JSON specification")
    out: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as blob:
        want_keys = [str(e.get("array_key")) for e in entries]
        if sorted(blob.files) != sorted(want_keys):
            raise ValueError("checkpoint NPZ keys differ from manifest")
        for e in entries:
            name = str(e.get("state_key"))
            key = str(e.get("array_key"))
            if not name or name in out:
                raise ValueError("checkpoint state keys must be unique and non-empty")
            arr = np.asarray(blob[key])
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"non-numeric dtype forbidden in checkpoint: {name}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"non-finite checkpoint values forbidden: {name}")
            if str(arr.dtype) != str(e.get("dtype")):
                raise ValueError(f"checkpoint dtype mismatch for {name}")
            if list(arr.shape) != list(e.get("shape")):
                raise ValueError(f"checkpoint shape mismatch for {name}")
            out[name] = arr.copy()
    return out


def matrix_rows(matrix: np.ndarray) -> list[dict]:
    a = np.asarray(matrix)
    if a.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError(f"matrix shape {a.shape} != 8x8")
    fields = ["TrueClassID", "TrueClass", *CLASS_NAMES]
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        row = {"TrueClassID": i, "TrueClass": name}
        row.update({CLASS_NAMES[j]: a[i, j].item() for j in range(len(CLASS_NAMES))})
        rows.append({k: row[k] for k in fields})
    return rows


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def exact_int(value: object) -> int | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x != math.trunc(x):
        return None
    return int(x)

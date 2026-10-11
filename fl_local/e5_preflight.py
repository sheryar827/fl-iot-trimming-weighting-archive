"""No-training Q1-E5 preflight.

Reconstructs the sealed caches in memory and compares C-contiguous array
bytes. It does not write a cache, a scaler, or a partition file. CSV read
failures and parser import failures stop the run. Unknown-label exclusions
are listed and are not treated as read failures.
"""
from __future__ import annotations

import gc
import glob
import importlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fl_local.cic_cache import GLOBAL_SEED as CIC_GLOBAL_SEED
from fl_local.cic_cache import _cache_key as cic_cache_key
from fl_local.cic_cache import categorize_cic_attack
from fl_local.e2_partition import partition_by_device_ids
from fl_local.e4_artifacts import atomic_write_json, hash_partition, partition_document
from fl_local.e5_aggregators import (
    BOUND_SHA256,
    CIC_ARRAYS,
    DECISION_SEEDS,
    FEATURE_COUNTS,
    NBAIOT_ARRAYS,
    REQUIRED_ENVIRONMENT,
    ACCEPTED_PREREG_SHA256,
    arrays_byte_equal,
    exact_nonnegative_int,
    index_sha256,
    scaler_sha256,
    sha256_file,
    validate_compact_partition,
)
from fl_local.e5_protocol import apply_and_read_cuda_flags
from fl_local.nbaiot_cache import GLOBAL_SEED as NBAIOT_GLOBAL_SEED
from fl_local.nbaiot_cache import KNOWN_DEVICES
from fl_local.nbaiot_cache import _cache_key as nbaiot_cache_key

PAPER = Path(__file__).resolve().parents[1]

PREFLIGHT_CHECKS = (
    "bound_hashes",
    "environment",
    "cuda_flags",
    "parser_import",
    "sealed_array_bytes",
    "decision_partitions_in_memory",
)

SEALED_NPZ = {
    "CIC-IoT2023": "probe_results/cache/cic_27bfed227ef77fab.npz",
    "N-BaIoT": "probe_results/cache/nbaiot_4ddb8b93b8821650.npz",
}
SEALED_META = {
    "CIC-IoT2023": "probe_results/cache/cic_27bfed227ef77fab.meta.json",
    "N-BaIoT": "probe_results/cache/nbaiot_4ddb8b93b8821650.meta.json",
}
SEALED_KEYS = {"CIC-IoT2023": "27bfed227ef77fab", "N-BaIoT": "4ddb8b93b8821650"}


class PreflightStop(RuntimeError):
    def __init__(self, path, reason: str):
        self.path = str(path)
        self.reason = reason
        super().__init__(f"{reason}: {path}")


def read_required_csv(path, *, min_columns: int):
    """Read one selected CSV. An exception, an empty read, or a short table stops."""
    path = Path(path)
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise PreflightStop(path, f"CSV read failed ({type(exc).__name__}: {exc})") from exc
    if frame.shape[0] == 0:
        raise PreflightStop(path, "empty CSV")
    if frame.shape[1] < int(min_columns):
        raise PreflightStop(
            path,
            f"table has {frame.shape[1]} columns; minimum is {int(min_columns)}",
        )
    return frame


def import_bound_parser(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise PreflightStop(
            module_name,
            f"parser import failed ({type(exc).__name__}: {exc})",
        ) from exc


def import_parser_file(path):
    """Load a parser file. An import failure stops; it is not skipped."""
    path = Path(path)
    if not path.is_file():
        raise PreflightStop(path, "parser file is missing")
    module_name = "e5_parser_probe_" + sha256_file(path)[:12]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PreflightStop(path, "parser import failed")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PreflightStop(
            path,
            f"parser import failed ({type(exc).__name__}: {exc})",
        ) from exc
    return module


def require_operation_stops(operation: Callable[[], object]) -> None:
    """A branch that swallows the failure does not satisfy the preflight."""
    try:
        operation()
    except PreflightStop:
        return
    except Exception:
        return
    raise AssertionError("operation skipped the failure")


def verify_bound_hashes(root, expected: Mapping[str, str] | None = None) -> dict[str, str]:
    table = BOUND_SHA256 if expected is None else expected
    found = {}
    for relative, digest in table.items():
        path = Path(root) / relative
        if not path.is_file():
            raise PreflightStop(path, "bound file missing")
        actual = sha256_file(path)
        if actual != str(digest).lower():
            raise PreflightStop(path, f"bound hash mismatch actual={actual}")
        found[str(relative).replace("\\", "/")] = actual
    return found


def live_environment() -> dict:
    import platform

    import numpy
    import pandas
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "numpy": str(numpy.__version__),
        "pandas": str(pandas.__version__),
        "sklearn": str(sklearn.__version__),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def verify_environment(expected: Mapping | None = None) -> dict:
    required = dict(REQUIRED_ENVIRONMENT if expected is None else expected)
    actual = live_environment()
    if actual != required:
        raise PreflightStop("environment", f"environment mismatch want={required} live={actual}")
    return actual


def _load_meta(root: Path, dataset: str) -> tuple[dict, Path]:
    meta_path = root / SEALED_META[dataset]
    npz_path = root / SEALED_NPZ[dataset]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("max_samples_per_file") != 20000 or float(meta.get("test_fraction")) != 0.2:
        raise PreflightStop(meta_path, "sealed meta sampling fields mismatch")
    if int(meta.get("global_seed")) != 42 or meta.get("scale_on") != "train":
        raise PreflightStop(meta_path, "sealed meta seed or scale_on mismatch")
    if int(meta.get("num_features")) != FEATURE_COUNTS[dataset]:
        raise PreflightStop(meta_path, "sealed meta feature count mismatch")
    if str(meta.get("cache_key")) != SEALED_KEYS[dataset]:
        raise PreflightStop(meta_path, "sealed meta cache key mismatch")
    extract = Path(meta["extract_dir"])
    if dataset == "CIC-IoT2023":
        if CIC_GLOBAL_SEED != 42:
            raise PreflightStop("fl_local.cic_cache", "GLOBAL_SEED is not 42")
        computed = cic_cache_key(extract, 20000, 0.2, scale_on="train")
    else:
        if NBAIOT_GLOBAL_SEED != 42:
            raise PreflightStop("fl_local.nbaiot_cache", "GLOBAL_SEED is not 42")
        computed = nbaiot_cache_key(extract, 20000, 0.2, scale_on="train")
    if computed != SEALED_KEYS[dataset]:
        raise PreflightStop(extract, f"computed cache key {computed} != {SEALED_KEYS[dataset]}")
    return meta, npz_path


def _classify_nbaiot(path) -> tuple[str | None, str | None, str | None]:
    norm = str(path).lower().replace(" ", "_").replace("-", "_")
    parts = [part.lower().replace(" ", "_").replace("-", "_") for part in Path(path).parts]
    fname = Path(path).stem.lower().replace(" ", "_").replace("-", "_")
    if "demonstrate" in norm or "description" in norm:
        return None, None, "demonstrate or description"
    device = "unknown"
    for known in KNOWN_DEVICES:
        if known in norm:
            device = known
            break
    label = "unknown"
    if "benign" in norm:
        label = "benign"
    else:
        is_mirai = "mirai" in norm
        is_gafgyt = "gafgyt" in norm or "bashlite" in norm
        mirai_types = ["ack", "scan", "syn", "udpplain", "udp"]
        gafgyt_types = ["combo", "junk", "scan", "tcp", "udp"]
        if is_mirai:
            for attack in mirai_types:
                if attack in fname or any(attack == part for part in parts):
                    label = f"mirai_{attack}"
                    break
            if label == "unknown":
                label = "mirai_other"
        elif is_gafgyt:
            for attack in gafgyt_types:
                if attack in fname or any(attack == part for part in parts):
                    label = f"gafgyt_{attack}"
                    break
            if label == "unknown":
                label = "gafgyt_other"
    if label == "unknown" or device == "unknown":
        return None, None, "unknown device or label"
    return device, label, None


def _parse_cic(csv_paths: Sequence[str], max_samples: int) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    excluded = []
    for path in csv_paths:
        label = categorize_cic_attack(Path(path).stem)
        if label == "unknown":
            excluded.append({"path": str(path), "reason": "unknown label"})
            continue
        frame = read_required_csv(path, min_columns=5)
        if max_samples and len(frame) > max_samples:
            frame = frame.sample(n=max_samples, random_state=CIC_GLOBAL_SEED)
        for column in ("label", "Label", "LABEL"):
            if column in frame.columns:
                frame = frame.drop(columns=[column])
        frame = frame.copy()
        frame = frame.assign(device="cic_testbed", label=label)
        frames.append(frame)
    if not frames:
        raise PreflightStop("CIC-IoT2023", "no CIC rows loaded")
    combined = pd.concat(frames, ignore_index=True)
    text_columns = [
        column
        for column in combined.columns
        if column not in ("device", "label") and combined[column].dtype == "object"
    ]
    if text_columns:
        combined = combined.drop(columns=text_columns)
    return combined, excluded


def _parse_nbaiot(csv_paths: Sequence[str], max_samples: int) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    excluded = []
    for path in csv_paths:
        device, label, reason = _classify_nbaiot(path)
        if reason is not None:
            excluded.append({"path": str(path), "reason": reason})
            continue
        frame = read_required_csv(path, min_columns=10)
        if max_samples and len(frame) > max_samples:
            frame = frame.sample(n=max_samples, random_state=NBAIOT_GLOBAL_SEED)
        frame = frame.copy()
        frame = frame.assign(device=device, label=label)
        frames.append(frame)
    if not frames:
        raise PreflightStop("N-BaIoT", "no N-BaIoT rows loaded")
    return pd.concat(frames, ignore_index=True), excluded


def _scale_on_train(raw_df: pd.DataFrame, *, keep_unscaled: bool = False) -> dict[str, np.ndarray]:
    """Original full-training scaling: split, then StandardScaler fit on train only."""
    feature_df = raw_df.drop(columns=["device", "label"], errors="ignore")
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.fillna(0, inplace=True)
    features = feature_df.values.astype(np.float32)
    del feature_df
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_df["label"].values)
    device_encoder = LabelEncoder()
    client_ids = device_encoder.fit_transform(raw_df["device"].values)
    x_train, x_test, y_train, y_test, cid_train, cid_test = train_test_split(
        features,
        labels,
        client_ids,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    del features
    unscaled_train = None
    if keep_unscaled:
        unscaled_train = np.ascontiguousarray(np.array(x_train, dtype=np.float32, copy=True))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    scaled = {
        "X_train": np.ascontiguousarray(x_train),
        "X_test": np.ascontiguousarray(x_test),
        "y_train": np.ascontiguousarray(y_train.astype(np.int64, copy=False)),
        "y_test": np.ascontiguousarray(y_test.astype(np.int64, copy=False)),
        "cid_train": np.ascontiguousarray(cid_train.astype(np.int64, copy=False)),
        "cid_test": np.ascontiguousarray(cid_test.astype(np.int64, copy=False)),
        "classes": [str(label) for label in label_encoder.classes_],
        "devices": [str(device) for device in device_encoder.classes_],
    }
    if keep_unscaled:
        scaled["X_unscaled_train"] = unscaled_train
    return scaled


def _compare_sealed(reconstructed: dict, npz_path: Path, names: Sequence[str], *, forbidden: Sequence[str]) -> None:
    with np.load(npz_path, allow_pickle=False) as blob:
        present = set(blob.files)
        missing = [name for name in names if name not in present]
        if missing:
            raise PreflightStop(npz_path, f"sealed NPZ missing {missing}")
        for name in forbidden:
            if name in present:
                raise PreflightStop(npz_path, f"sealed NPZ unexpectedly contains {name}")
        for name in names:
            sealed = blob[name]
            built = reconstructed[name]
            if not arrays_byte_equal(built, sealed):
                raise PreflightStop(
                    npz_path,
                    (
                        f"{name} byte mismatch "
                        f"reconstructed shape={getattr(built, 'shape', None)} dtype={getattr(built, 'dtype', None)} "
                        f"sealed shape={sealed.shape} dtype={sealed.dtype}"
                    ),
                )


def reconstruct_dataset(root, dataset: str, *, keep_unscaled: bool = False, purpose: str = "preflight") -> dict:
    """Rebuild one sealed cache in memory and compare array bytes. Does not write a cache."""
    root = Path(root)
    meta, npz_path = _load_meta(root, dataset)
    extract = Path(meta["extract_dir"])
    csv_paths = sorted(glob.glob(str(extract / "**" / "*.csv"), recursive=True))
    if not csv_paths:
        raise PreflightStop(extract, "no CSV files")
    print(f"E5 {purpose}: reconstructing {dataset} from {len(csv_paths)} CSV files", flush=True)
    if dataset == "CIC-IoT2023":
        raw, excluded = _parse_cic(csv_paths, int(meta["max_samples_per_file"]))
        names = CIC_ARRAYS
        forbidden = ("cid_test",)
    elif dataset == "N-BaIoT":
        raw, excluded = _parse_nbaiot(csv_paths, int(meta["max_samples_per_file"]))
        names = NBAIOT_ARRAYS
        forbidden = ()
    else:
        raise PreflightStop(dataset, "unknown dataset")
    scaled = _scale_on_train(raw, keep_unscaled=keep_unscaled)
    del raw
    gc.collect()
    if scaled["classes"] != [str(label) for label in meta["classes"]]:
        raise PreflightStop(dataset, f"class labels differ: {scaled['classes']}")
    if scaled["devices"] != [str(device) for device in meta["devices"]]:
        raise PreflightStop(dataset, f"device labels differ: {scaled['devices']}")
    _compare_sealed(scaled, npz_path, names, forbidden=forbidden)
    summary = {
        "cache_key": SEALED_KEYS[dataset],
        "byte_equal": True,
        "arrays": {
            name: {"shape": list(scaled[name].shape), "dtype": str(scaled[name].dtype)}
            for name in names
        },
        "excluded": excluded,
        "files_selected": len(csv_paths),
        "files_read": len(csv_paths) - len(excluded),
    }
    print(
        f"E5 {purpose}: {dataset} array bytes match "
        f"({summary['files_read']} read, {len(excluded)} excluded)",
        flush=True,
    )
    unscaled = scaled.pop("X_unscaled_train", None) if keep_unscaled else None
    y_train = scaled.pop("y_train")
    cid_train = scaled.pop("cid_train")
    cid_test = scaled.pop("cid_test") if dataset == "N-BaIoT" else None
    scaled.clear()
    del scaled
    gc.collect()
    if keep_unscaled and unscaled is None:
        raise PreflightStop(dataset, "unscaled training matrix was not retained")
    returned = {
        "summary": summary,
        "y_train": y_train,
        "cid_train": cid_train,
        "cid_test": cid_test,
    }
    if keep_unscaled:
        returned["X_unscaled_train"] = unscaled
    return returned


def verify_decision_partitions(root, y_cic, cid_train, cid_test) -> dict[str, str]:
    """Recompute E5 decision partitions in memory. Does not save them."""
    import fl_local.engine as engine

    frozen_path = Path(root) / "q1_e4_results" / "reporting" / "Q1_E4_FROZEN_CONFIG.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    checked = {}
    y_cic = np.asarray(y_cic)
    for seed in DECISION_SEEDS:
        partition = engine.partition_dirichlet(y_cic, 9, 0.1, seed=int(seed))
        actual = partition_document(partition, len(y_cic), 9)
        expected = frozen["partitions"]["CIC-IoT2023"][str(seed)]
        if actual != expected:
            raise PreflightStop(f"CIC-IoT2023 seed {seed}", "decision partition mismatch")
        checked[f"CIC-IoT2023:{seed}"] = actual["sha256"]
        del partition
    cid_test = np.asarray(cid_test)
    observed = sorted({int(value) for value in cid_test.tolist()})
    if observed != list(range(9)):
        raise PreflightStop("N-BaIoT", "pooled test must contain all nine device ids")
    partition = partition_by_device_ids(np.asarray(cid_train), num_clients=9)
    actual = partition_document(partition, len(cid_train), 9)
    actual["pooled_test_device_ids"] = list(range(9))
    if actual != frozen["partitions"]["N-BaIoT"]:
        raise PreflightStop("N-BaIoT", "decision partition mismatch")
    checked["N-BaIoT:device"] = actual["sha256"]
    return checked


def verify_scaler_artifact(npz_path, json_path, *, feature_count: int) -> str:
    npz_path = Path(npz_path)
    json_path = Path(json_path)
    if not npz_path.is_file() or not json_path.is_file():
        raise PreflightStop(npz_path, "tuning scaler missing")
    document = json.loads(json_path.read_text(encoding="utf-8"))
    if document.get("dtype") != "<f8":
        raise PreflightStop(json_path, "scaler dtype must be <f8")
    features = exact_nonnegative_int(document.get("feature_count"), "feature_count")
    if features != int(feature_count):
        raise PreflightStop(json_path, "scaler feature_count mismatch")
    seen = exact_nonnegative_int(document.get("n_samples_seen_"), "n_samples_seen_")
    subtrain_hash = str(document.get("subtrain_index_sha256", "")).lower()
    validation_hash = str(document.get("validation_index_sha256", "")).lower()
    with np.load(npz_path, allow_pickle=False) as blob:
        mean = np.array(blob["mean_"], copy=True)
        var = np.array(blob["var_"], copy=True)
        scale = np.array(blob["scale_"], copy=True)
    digest = scaler_sha256(
        mean_=mean,
        var_=var,
        scale_=scale,
        n_samples_seen_=seen,
        feature_count=features,
        subtrain_index_sha256=subtrain_hash,
        validation_index_sha256=validation_hash,
    )
    if digest != str(document.get("scaler_sha256", "")).lower():
        raise PreflightStop(json_path, "scaler_sha256 mismatch")
    return digest


def verify_partition_artifact(
    npz_path,
    json_path,
    *,
    compact: bool,
    expected_sha256: str | None = None,
) -> str:
    npz_path = Path(npz_path)
    json_path = Path(json_path)
    if not npz_path.is_file() or not json_path.is_file():
        raise PreflightStop(npz_path, "partition file missing")
    document = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as blob:
        partition = {client_id: np.array(blob[f"c{client_id}"], copy=True) for client_id in range(9)}
    try:
        if compact:
            coverage = sum(int(len(partition[client_id])) for client_id in range(9))
            validate_compact_partition(partition, coverage)
        actual = hash_partition(partition)
    except ValueError as exc:
        raise PreflightStop(json_path, str(exc)) from exc
    if actual != str(document.get("sha256", "")).lower():
        raise PreflightStop(json_path, "partition hash mismatch")
    if expected_sha256 is not None and actual != str(expected_sha256).lower():
        raise PreflightStop(json_path, "partition hash does not match the frozen document")
    sizes = [int(len(partition[client_id])) for client_id in range(9)]
    if list(document.get("client_sizes", [])) != sizes:
        raise PreflightStop(json_path, "partition client_sizes mismatch")
    stored = [str(value).lower() for value in document.get("client_index_sha256", [])]
    rebuilt = [index_sha256(partition[client_id]) for client_id in range(9)]
    if stored != rebuilt:
        raise PreflightStop(json_path, "client index hash mismatch")
    return actual


def run_preflight(root=None, *, write_report: bool = True) -> dict:
    """Section 4.3 and Section 5.4. Scaler files are not required."""
    root = PAPER if root is None else Path(root)
    report_path = root / "q1_e5_results" / "Q1_E5_PREFLIGHT.json"
    if write_report and report_path.is_file():
        report_path.unlink()
    hashes = verify_bound_hashes(root)
    environment = verify_environment()
    flags = apply_and_read_cuda_flags()
    import_bound_parser("fl_local.cic_cache")
    import_bound_parser("fl_local.nbaiot_cache")
    cic = reconstruct_dataset(root, "CIC-IoT2023")
    nbaiot = reconstruct_dataset(root, "N-BaIoT")
    partitions = verify_decision_partitions(
        root, cic["y_train"], nbaiot["cid_train"], nbaiot["cid_test"]
    )
    del cic["y_train"], cic["cid_train"], cic["cid_test"]
    del nbaiot["y_train"], nbaiot["cid_train"], nbaiot["cid_test"]
    gc.collect()
    report = {
        "status": "PASS",
        "kind": "no-training-preflight",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "accepted_prereg_sha256": ACCEPTED_PREREG_SHA256,
        "checks": list(PREFLIGHT_CHECKS),
        "environment": environment,
        "cuda_flags": flags,
        "bound_hashes": hashes,
        "datasets": {"CIC-IoT2023": cic["summary"], "N-BaIoT": nbaiot["summary"]},
        "decision_partitions": partitions,
        "scalers_written": False,
        "partition_files_written": False,
        "gpu_training": False,
    }
    if write_report:
        atomic_write_json(report_path, report)
        report["report_path"] = str(report_path)
    return report

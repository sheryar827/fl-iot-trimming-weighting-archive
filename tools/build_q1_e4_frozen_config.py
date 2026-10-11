"""Build the deterministic, no-training Q1-E4 freeze.

The builder reads only the two exact existing train-scaler caches and the two
sealed three-seed result lineages.  It has no cache-building or training path
and refuses to rewrite the freeze after a preflight or E4 result exists.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_local.e4_artifacts import (  # noqa: E402
    CLIENT_HASH_SCHEME,
    PARTITION_HASH_SCHEME,
    atomic_write_csv,
    atomic_write_json,
    partition_document,
    sha256_file,
)
from run_q1_e4_train_scaler import (  # noqa: E402
    ALGORITHMS,
    ALL_SEEDS,
    ANALYSIS_SPEC,
    ANCHOR_FIELDS,
    BLOCKS,
    CACHE_SPECS,
    EXISTING_SEEDS,
    EXPECTED_CFG,
    EXPECTED_ENVIRONMENT,
    LIVE,
    MODE,
    NEW_SEEDS,
    PREREG_SHA256,
    SCALE_ON,
    all_jobs,
    contract_accepts_e4,
    prereg_accepted,
)


OUT = ROOT / "q1_e4_results"
REPORTING = OUT / "reporting"
ANCHORS = REPORTING / "Q1_E4_BASELINE_ANCHORS.csv"
FROZEN = REPORTING / "Q1_E4_FROZEN_CONFIG.json"
PREFLIGHT = REPORTING / "Q1_E4_PREFLIGHT.json"
RAW = OUT / "csv" / "q1_e4_raw.csv"

CIC_RAW = ROOT / "q1_cic_conf_results" / "csv" / "q1_cic_conf_raw.csv"
CIC_MANIFEST = ROOT / "q1_cic_conf_results" / "reporting" / "SHA256SUMS.txt"
E2_RAW = ROOT / "q1_e2_results" / "csv" / "q1_e2_raw.csv"
E2_MANIFEST = ROOT / "q1_e2_results" / "reporting" / "SHA256SUMS.txt"

EXPECTED_INPUT_HASHES = {
    "q1_e4_results/reporting/Q1_E4_TRAIN_SCALER_SEED_AMENDMENT_DRAFT.md": PREREG_SHA256,
    "fl_local/engine.py": "7d201f9734b56c29457cd028933b0eca4862e241b17040e1cb6e88c6b0f130d2",
    "fl_local/e2_partition.py": "72778f968a03d2f934801b9270e4cdbfeb2759ad7960b20f514e2069c863e8a2",
    "q1_cic_conf_results/csv/q1_cic_conf_raw.csv": "cadae7a0d3a35639c485bc2d8e9eae701ff5d82a0d320c12937c9f74615a74a5",
    "q1_e2_results/csv/q1_e2_raw.csv": "5abe64b5c398295908d2eae8dea9633239c29231a05eca356016d0e324082a69",
    "q1_cic_conf_results/reporting/SHA256SUMS.txt": "9607c022006e91415b0895ccc588fd0b8e72df2c4e0ae656536d95c7bf10b884",
    "q1_e2_results/reporting/SHA256SUMS.txt": "eaaab95ca93919ba3306b7c81a291dd02f72d8eb5893ab90923369510974322d",
}

FILES_TO_FREEZE = (
    "q1_e4_results/reporting/Q1_E4_TRAIN_SCALER_SEED_AMENDMENT_DRAFT.md",
    "run_q1_e4_train_scaler.py",
    "fl_local/e4_artifacts.py",
    "fl_local/e4_stats.py",
    "tools/build_q1_e4_frozen_config.py",
    "tools/build_q1_e4_packet.py",
    "tools/test_q1_e4_integrity.py",
    "fl_local/engine.py",
    "fl_local/e2_partition.py",
    "fl_local/cic_cache.py",
    "fl_local/nbaiot_cache.py",
    "fl_local/PREPROCESS_PROTOCOL.md",
    "q1_cic_conf_results/csv/q1_cic_conf_raw.csv",
    "q1_cic_conf_results/reporting/SHA256SUMS.txt",
    "q1_cic_conf_results/reporting/Q1_CIC_CONFIRMATORY_FROZEN_CONFIG.json",
    "q1_cic_conf_results/reporting/Q1_CIC_CONFIRMATORY_CACHE_PROVENANCE.json",
    "q1_e2_results/csv/q1_e2_raw.csv",
    "q1_e2_results/reporting/SHA256SUMS.txt",
    "q1_e2_results/reporting/Q1_E2_CACHE_PROVENANCE.json",
    "q1_e2_results/reporting/Q1_E2_PREREG_DRAFT.md",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        raise RuntimeError(f"missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _finite(value: object, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(numeric):
        raise RuntimeError(f"{label} is non-finite: {value!r}")
    return numeric


def _exact_int(value: object, label: str) -> int:
    numeric = _finite(value, label)
    if numeric != math.trunc(numeric):
        raise RuntimeError(f"{label} is not an exact integer: {value!r}")
    return int(numeric)


def _safe_root_path(relative: object) -> Path:
    rel = Path(str(relative).replace("\\", "/"))
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise RuntimeError(f"unsafe root-relative path: {relative!r}")
    resolved = (ROOT / rel).resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"path escapes project root: {relative!r}")
    return resolved


def _history_metrics(path: Path) -> dict:
    fields, rows = _read_csv(path)
    if fields != ["round", "accuracy", "f1", "loss"] or len(rows) != 30:
        raise RuntimeError(f"history schema/length mismatch: {path}")
    accuracy: list[float] = []
    f1: list[float] = []
    for index, row in enumerate(rows, 1):
        if _exact_int(row.get("round"), f"{path.name} round") != index:
            raise RuntimeError(f"history rounds are not exactly 1..30: {path}")
        acc = _finite(row.get("accuracy"), f"{path.name} accuracy")
        score = _finite(row.get("f1"), f"{path.name} f1")
        loss = _finite(row.get("loss"), f"{path.name} loss")
        if not 0.0 <= acc <= 1.0 or not 0.0 <= score <= 1.0 or loss < 0.0:
            raise RuntimeError(f"history value outside valid range: {path}")
        accuracy.append(acc)
        f1.append(score)
    return {
        "Final_Acc": accuracy[-1],
        "Final_F1": f1[-1],
        "Best_Acc": max(accuracy),
        "Best_F1": max(f1),
        "AUC": sum(f1) / len(f1),
    }


def _parse_and_verify_manifest(
    manifest: Path, *, base: Path, expected_entries: int
) -> dict[Path, str]:
    entries: dict[Path, str] = {}
    line_pattern = re.compile(r"^([0-9A-Fa-f]{64})\s{2}(.+)$")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = line_pattern.match(stripped)
        if not match:
            raise RuntimeError(f"malformed manifest line in {manifest}: {line!r}")
        expected_hash = match.group(1).lower()
        relative = Path(match.group(2).replace("\\", "/"))
        if relative.is_absolute():
            raise RuntimeError(f"manifest path must be relative: {relative}")
        path = (base / relative).resolve()
        root = ROOT.resolve()
        if path != root and root not in path.parents:
            raise RuntimeError(f"manifest path escapes project root: {relative}")
        if path in entries:
            raise RuntimeError(f"duplicate manifest path: {path}")
        if not path.is_file():
            raise RuntimeError(f"manifest file missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"manifest hash mismatch: {path}\nwant {expected_hash}\ngot  {actual_hash}"
            )
        entries[path] = expected_hash
    if len(entries) != expected_entries:
        raise RuntimeError(
            f"manifest entry count mismatch for {manifest}: {len(entries)} != {expected_entries}"
        )
    return entries


def _validate_input_hashes() -> None:
    for relative, expected in EXPECTED_INPUT_HASHES.items():
        path = _safe_root_path(relative)
        if not path.is_file():
            raise RuntimeError(f"required frozen input missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"required input hash mismatch {relative}\nwant {expected}\ngot  {actual}"
            )


def _load_cache_labels(dataset: str):
    expected = CACHE_SPECS[dataset]
    npz_path = LIVE.cache_npz(dataset)
    meta_path = LIVE.cache_meta(dataset)
    if not npz_path.is_file() or sha256_file(npz_path) != expected["npz_sha256"]:
        raise RuntimeError(f"{dataset} exact cache NPZ is missing or changed")
    if not meta_path.is_file() or sha256_file(meta_path) != expected["meta_sha256"]:
        raise RuntimeError(f"{dataset} exact cache metadata is missing or changed")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for field, value in (
        ("cache_key", expected["key"]),
        ("scale_on", SCALE_ON),
        ("n_train", expected["n_train"]),
        ("n_test", expected["n_test"]),
        ("num_features", expected["num_features"]),
        ("num_classes", expected["num_classes"]),
    ):
        if meta.get(field) != value:
            raise RuntimeError(f"{dataset} metadata mismatch: {field}")
    with np.load(npz_path, allow_pickle=False) as blob:
        if sorted(blob.files) != expected["npz_keys"]:
            raise RuntimeError(f"{dataset} NPZ schema mismatch: {sorted(blob.files)}")
        y_train = np.asarray(blob["y_train"])
        y_test = np.asarray(blob["y_test"])
        cid_train = np.asarray(blob["cid_train"])
        cid_test = np.asarray(blob["cid_test"]) if "cid_test" in blob.files else None
    for name, array, length in (
        ("y_train", y_train, expected["n_train"]),
        ("y_test", y_test, expected["n_test"]),
        ("cid_train", cid_train, expected["n_train"]),
    ):
        if tuple(array.shape) != (length,) or str(array.dtype) != "int64":
            raise RuntimeError(f"{dataset} {name} shape/dtype mismatch")
    if cid_test is not None and (
        tuple(cid_test.shape) != (expected["n_test"],) or str(cid_test.dtype) != "int64"
    ):
        raise RuntimeError(f"{dataset} cid_test shape/dtype mismatch")
    return meta, y_train, y_test, cid_train, cid_test


def _partition_dirichlet(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int
) -> dict[int, np.ndarray]:
    """Pure NumPy replica of the hash-pinned engine partition function."""
    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(labels))
    client_indices: dict[int, list[int]] = {client: [] for client in range(num_clients)}
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.maximum(proportions, 1e-6)
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client, chunk in enumerate(np.split(class_indices, splits)):
            client_indices[client].extend(chunk.tolist())
    for client in range(num_clients):
        if not client_indices[client]:
            donor = max(range(num_clients), key=lambda item: len(client_indices[item]))
            if client_indices[donor]:
                client_indices[client].append(client_indices[donor].pop())
    output: dict[int, np.ndarray] = {}
    for client in range(num_clients):
        rng.shuffle(client_indices[client])
        output[client] = np.asarray(client_indices[client], dtype=np.int64)
    return output


def _partition_documents(cic_y: np.ndarray, nb_cid_train: np.ndarray, nb_cid_test: np.ndarray) -> dict:
    cic: dict[str, dict] = {}
    for seed in ALL_SEEDS:
        partition = _partition_dirichlet(cic_y, 9, 0.1, seed)
        cic[str(seed)] = partition_document(partition, len(cic_y), 9)
        del partition
        gc.collect()
    device_partition = {
        client: np.where(nb_cid_train == client)[0].astype(np.int64)
        for client in range(9)
    }
    nb = partition_document(device_partition, len(nb_cid_train), 9)
    pooled_ids = sorted(set(int(value) for value in nb_cid_test.tolist()))
    if pooled_ids != list(range(9)):
        raise RuntimeError(f"N-BaIoT pooled test device IDs mismatch: {pooled_ids}")
    nb["pooled_test_device_ids"] = pooled_ids
    return {"CIC-IoT2023": cic, "N-BaIoT": nb}


def _select_source_rows(path: Path, dataset: str) -> dict[tuple[str, str, str], dict]:
    _, rows = _read_csv(path)
    selected: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if dataset == "CIC-IoT2023":
            try:
                alpha = float(row.get("Alpha", "nan"))
            except ValueError:
                continue
            if alpha != 0.1:
                continue
        elif row.get("Partition") != "device":
            continue
        model = row.get("Model", "")
        algorithm = row.get("Algorithm", "")
        seed = _exact_int(row.get("Seed"), f"{dataset} seed")
        key = (model, algorithm, str(seed))
        if key in selected:
            raise RuntimeError(f"duplicate selected {dataset} source row: {key}")
        selected[key] = row
    expected = {
        (model, algorithm, str(seed))
        for model in ("MLP", "CNN")
        for algorithm in ALGORITHMS
        for seed in EXISTING_SEEDS
    }
    if set(selected) != expected:
        raise RuntimeError(
            f"{dataset} selected anchor grid mismatch missing={expected-set(selected)} "
            f"extra={set(selected)-expected}"
        )
    return selected


def _anchor_rows(partitions: dict, manifest_entries: dict[str, dict[Path, str]]) -> list[dict]:
    source_rows = {
        "CIC-IoT2023": _select_source_rows(CIC_RAW, "CIC-IoT2023"),
        "N-BaIoT": _select_source_rows(E2_RAW, "N-BaIoT"),
    }
    source_raw = {"CIC-IoT2023": CIC_RAW, "N-BaIoT": E2_RAW}
    source_manifest = {"CIC-IoT2023": CIC_MANIFEST, "N-BaIoT": E2_MANIFEST}
    rows: list[dict] = []
    for block in BLOCKS:
        dataset = block["dataset"]
        cache = CACHE_SPECS[dataset]
        for seed in EXISTING_SEEDS:
            for algorithm in ALGORITHMS:
                key = (block["model"], algorithm, str(seed))
                source = source_rows[dataset][key]
                if (
                    source.get("Mode") != MODE
                    or source.get("ScaleOn") != SCALE_ON
                    or _exact_int(source.get("N_Rounds"), f"{dataset} {key} rounds") != 30
                    or source.get("CacheKey") != cache["key"]
                    or source.get("NpzSha256", "").lower() != cache["npz_sha256"]
                    or source.get("MetaSha256", "").lower() != cache["meta_sha256"]
                ):
                    raise RuntimeError(f"source binding mismatch for {dataset} {key}")
                history_rel = str(source.get("HistoryCSV", "")).replace("\\", "/")
                history = _safe_root_path(history_rel)
                if not history.is_file():
                    raise RuntimeError(f"source history missing for {dataset} {key}")
                history_hash = sha256_file(history)
                if manifest_entries[dataset].get(history.resolve()) != history_hash:
                    raise RuntimeError(f"source history is not sealed by its manifest: {history_rel}")
                metrics = _history_metrics(history)
                for field, value in metrics.items():
                    source_value = _finite(source.get(field), f"source {field} for {dataset} {key}")
                    tolerance = 1e-15 if field == "AUC" else 0.0
                    if abs(source_value - value) > tolerance:
                        raise RuntimeError(
                            f"source {field} mismatch for {dataset} {key}: "
                            f"stored={source_value!r} history={value!r} tolerance={tolerance}"
                        )
                if dataset == "CIC-IoT2023":
                    partition_sha = partitions[dataset][str(seed)]["sha256"]
                else:
                    partition_sha = partitions[dataset]["sha256"]
                rows.append(
                    {
                        "Block": block["block"],
                        "Dataset": dataset,
                        "Model": block["model"],
                        "Algorithm": algorithm,
                        "Condition": block["condition"],
                        "Seed": seed,
                        "Mode": MODE,
                        "ScaleOn": SCALE_ON,
                        "Final_Acc": source["Final_Acc"],
                        "Final_F1": source["Final_F1"],
                        "Best_Acc": source["Best_Acc"],
                        "Best_F1": source["Best_F1"],
                        "AUC": metrics["AUC"],
                        "Source_AUC": source["AUC"],
                        "N_Rounds": 30,
                        "CacheKey": cache["key"],
                        "NpzSha256": cache["npz_sha256"],
                        "MetaSha256": cache["meta_sha256"],
                        "PartitionSha256": partition_sha,
                        "HistoryCSV": history_rel,
                        "HistorySha256": history_hash,
                        "SourceRaw": str(source_raw[dataset].relative_to(ROOT)).replace("\\", "/"),
                        "SourceRawSha256": sha256_file(source_raw[dataset]),
                        "SourceManifest": str(source_manifest[dataset].relative_to(ROOT)).replace("\\", "/"),
                        "SourceManifestSha256": sha256_file(source_manifest[dataset]),
                    }
                )
    if len(rows) != 48:
        raise RuntimeError(f"expected exactly 48 E4 anchors, found {len(rows)}")
    return rows


def _freeze_files() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in FILES_TO_FREEZE:
        path = _safe_root_path(relative)
        if not path.is_file():
            raise RuntimeError(f"file to freeze is missing: {relative}")
        actual = sha256_file(path)
        expected = EXPECTED_INPUT_HASHES.get(relative)
        if expected is not None and actual != expected:
            raise RuntimeError(f"upstream frozen hash mismatch for {relative}: {actual}")
        output[relative] = actual
    return output


def _refuse_after_execution_state() -> None:
    if PREFLIGHT.exists():
        raise RuntimeError("refusing to regenerate E4 freeze after a preflight exists")
    if RAW.is_file() and _read_csv(RAW)[1]:
        raise RuntimeError("refusing to regenerate E4 freeze after result rows exist")
    for directory in (OUT / "cells", OUT / "csv", OUT / ".staging"):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.resolve() in {ANCHORS.resolve(), FROZEN.resolve()}:
                continue
            if path.name == "q1_e4_raw.csv":
                continue
            raise RuntimeError(f"refusing to regenerate E4 freeze with execution artifact: {path}")


def main() -> int:
    if not prereg_accepted(LIVE) or not contract_accepts_e4(LIVE):
        raise RuntimeError("E4 named acceptance or contract anchors are missing")
    _refuse_after_execution_state()
    _validate_input_hashes()
    cic_manifest_entries = _parse_and_verify_manifest(
        CIC_MANIFEST,
        base=CIC_MANIFEST.parent,
        expected_entries=62,
    )
    e2_manifest_entries = _parse_and_verify_manifest(
        E2_MANIFEST,
        base=ROOT / "q1_e2_results",
        expected_entries=33,
    )
    cic_meta, cic_y, _, _, _ = _load_cache_labels("CIC-IoT2023")
    nb_meta, _, _, nb_cid_train, nb_cid_test = _load_cache_labels("N-BaIoT")
    if nb_cid_test is None:
        raise RuntimeError("N-BaIoT train-scaler cache must contain cid_test")
    partitions = _partition_documents(cic_y, nb_cid_train, nb_cid_test)
    anchors = _anchor_rows(
        partitions,
        {
            "CIC-IoT2023": cic_manifest_entries,
            "N-BaIoT": e2_manifest_entries,
        },
    )
    atomic_write_csv(ANCHORS, ANCHOR_FIELDS, anchors)
    anchor_sha = sha256_file(ANCHORS)
    file_sha256 = _freeze_files()
    environment = dict(EXPECTED_ENVIRONMENT)
    document = {
        "rev": "q1-e4-r1-ACCEPTED",
        "freeze_date": "2026-09-26",
        "accepted": {
            "proposer": "Sheryar Kiani",
            "approver": "Dr Hussain Kazmi",
            "date": "2026-09-26",
            "prereg_sha256": PREREG_SHA256,
        },
        "scope": {
            "blocks": list(BLOCKS),
            "algorithms": list(ALGORITHMS),
            "existing_seeds": list(EXISTING_SEEDS),
            "new_seeds": list(NEW_SEEDS),
            "all_seed_reporting_order": list(ALL_SEEDS),
            "new_jobs": 112,
            "anchor_jobs": 48,
            "combined_jobs": 160,
            "waves": {"0": 4, "1": 28, "2": 32, "3": 32, "4": 16},
            "job_keys": [list(job.full_key) for job in all_jobs()],
            "mode": MODE,
            "scale_on": SCALE_ON,
            "partition_hash_scheme": PARTITION_HASH_SCHEME,
            "client_index_hash_scheme": CLIENT_HASH_SCHEME,
        },
        "analysis": ANALYSIS_SPEC,
        "cfg": EXPECTED_CFG,
        "environment": environment,
        "caches": {
            dataset: {
                **spec,
                "scale_on": SCALE_ON,
                "from_existing_only": True,
                "classes": (cic_meta if dataset == "CIC-IoT2023" else nb_meta).get("classes", []),
                "devices": (cic_meta if dataset == "CIC-IoT2023" else nb_meta).get("devices", []),
            }
            for dataset, spec in CACHE_SPECS.items()
        },
        "partitions": partitions,
        "baseline": {
            "anchor_csv": "q1_e4_results/reporting/Q1_E4_BASELINE_ANCHORS.csv",
            "anchor_sha256": anchor_sha,
            "rows": 48,
            "source_raw_sha256": {
                "CIC-IoT2023": EXPECTED_INPUT_HASHES["q1_cic_conf_results/csv/q1_cic_conf_raw.csv"],
                "N-BaIoT": EXPECTED_INPUT_HASHES["q1_e2_results/csv/q1_e2_raw.csv"],
            },
            "source_manifest_sha256": {
                "CIC-IoT2023": EXPECTED_INPUT_HASHES["q1_cic_conf_results/reporting/SHA256SUMS.txt"],
                "N-BaIoT": EXPECTED_INPUT_HASHES["q1_e2_results/reporting/SHA256SUMS.txt"],
            },
            "selection": (
                "CIC exact alpha=0.1 plus N-BaIoT exact device partition; "
                "both models, four arms, seeds 42/123/7"
            ),
        },
        "file_sha256": file_sha256,
    }
    atomic_write_json(FROZEN, document)
    print("Q1-E4 freeze built without training or cache writes")
    print(f"anchors: {ANCHORS} (48 rows, sha256={anchor_sha})")
    print(f"config:  {FROZEN} (sha256={sha256_file(FROZEN)})")
    print("jobs:    112 new cells; waves 4/28/32/32/16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

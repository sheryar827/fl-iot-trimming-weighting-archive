"""Build the no-GPU Q1-E7 reference snapshot and frozen configuration.

This builder is intentionally read-only with respect to the CIC cache and the
confirmatory campaign.  It refuses to run after an E7 result row exists and it
has no cache-building or training path.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_local.e7_artifacts import (  # noqa: E402
    CLASS_NAMES,
    CLASS_SUPPORTS,
    atomic_write_csv,
    atomic_write_json,
    hash_partition,
    sha256_bytes,
    sha256_file,
    validate_partition,
    validate_truth,
)


OUT = ROOT / "q1_e7_results"
REPORTING = OUT / "reporting"
RAW = OUT / "csv" / "q1_e7_raw.csv"
ANCHORS = REPORTING / "Q1_E7_BASELINE_ANCHORS.csv"
FROZEN = REPORTING / "Q1_E7_FROZEN_CONFIG.json"
CONFIRM_RAW = ROOT / "q1_cic_conf_results" / "csv" / "q1_cic_conf_raw.csv"
CACHE_NPZ = ROOT / "probe_results" / "cache" / "cic_27bfed227ef77fab.npz"
CACHE_META = ROOT / "probe_results" / "cache" / "cic_27bfed227ef77fab.meta.json"

MODELS = ("MLP", "CNN")
ALGORITHMS = (
    "FedAvg",
    "UnweightedFedAvg",
    "FedTrimmedAvg",
    "WeightedFedTrimmedAvg",
)
SEEDS = (42, 123, 7)
ALPHA = 0.1

EXPECTED_CACHE = {
    "key": "27bfed227ef77fab",
    "npz_sha256": "8f032df1760a895b2358b8ff2da71491b09820c7af8c097dd00465fd7b981b38",
    "meta_sha256": "e5e0ef795fff100af24f6813ca3a2a61442e386aa01bf92878a9cfc0b03cb3b4",
    "y_train_int64_sha256": "ad64b239410ceba5db74f43338873aecd4d9370a99f067ac900124b699351830",
    "y_test_int64_sha256": "d86367f69d6f718df19be4b806e11e7ab43d5d069875dad0f56b9cbc2aa9261d",
    "y_test_uint8_sha256": "fe4ceccf96058485d6896e2a27b4aba0d5950f686a54fc2f53451535e3c408d6",
}

EXPECTED_PARTITIONS = {
    42: {
        "sha256": "e43250a6149aabdf6fdb9566b84b62070d0df9e2d56cd62a019df65e84fcb7ca",
        "client_sizes": [390284, 803016, 14218, 278396, 8466, 121015, 1509335, 492468, 1208494],
    },
    123: {
        "sha256": "74e689e01bb04bb3506dcfabb66bf036b8be2cfaaf9f2a852326ee63274df03a",
        "client_sizes": [295771, 98232, 89579, 166924, 73376, 358193, 479245, 2963583, 300789],
    },
    7: {
        "sha256": "43b9931c694a938b74e611235948563d6ac96ed0f7cb56baf79e72d8911ce8f4",
        "client_sizes": [24657, 42122, 35685, 117148, 43321, 282165, 512004, 181807, 3586783],
    },
}

EXPECTED_ANCHOR_SHA256 = "e43f8e7eebf4f536c691dcc84c93ee9fae36e7b8c7bfe630a79683624718d3e0"

EXPECTED_SOURCE_HASHES = {
    "fl_local/engine.py": "7d201f9734b56c29457cd028933b0eca4862e241b17040e1cb6e88c6b0f130d2",
    "fl_local/cic_cache.py": "91a322560c7a4ad3d47ed70dcc70ab2e00f6db134490debc872a35eb6b0760e3",
    "fl_local/PREPROCESS_PROTOCOL.md": "4ce920836705e7a6ad001824d92996c75bdba9bcf5af495965edc4031759425f",
    "run_q1_cic_confirmatory.py": "7e7889c5a2260c95292711464070cf98493eadad57c7328bb28383acae8ad6d9",
    "q1_cic_conf_results/reporting/Q1_CIC_CONFIRMATORY_FROZEN_CONFIG.json": "fb8623f085e944493840e561c4765c606177e3223a3addca58b4700429db2d2c",
    "q1_cic_conf_results/reporting/Q1_CIC_CONFIRMATORY_CACHE_PROVENANCE.json": "444a25c446921703199153e467149593368e4d86185fcc5e7623ae31bc4b90ce",
}

FILES_TO_FREEZE = (
    *EXPECTED_SOURCE_HASHES,
    "fl_local/__init__.py",
    "q1_e7_results/reporting/Q1_E7_PREREG_DRAFT.md",
    "run_q1_e7_cic.py",
    "fl_local/e7_artifacts.py",
    "fl_local/e7_capture.py",
    "tools/build_q1_e7_frozen_config.py",
    "tools/test_q1_e7_integrity.py",
)

ANCHOR_FIELDS = (
    "Wave",
    "Model",
    "Algorithm",
    "Alpha",
    "Seed",
    "Mode",
    "ScaleOn",
    "Final_Acc",
    "Final_F1",
    "N_Rounds",
    "CacheKey",
    "NpzSha256",
    "MetaSha256",
    "PartitionSha256",
    "HistoryCSV",
    "HistorySha256",
)


def _job_keys() -> list[tuple[str, str, str, str]]:
    seed42 = [(m, a, "0.1", "42") for m in MODELS for a in ALGORITHMS]
    w0 = ("MLP", "FedAvg", "0.1", "42")
    w1 = [key for key in seed42 if key != w0]
    w2 = [
        (m, a, "0.1", str(seed))
        for seed in (123, 7) for m in MODELS for a in ALGORITHMS
    ]
    return [w0, *w1, *w2]


def _require_named_acceptance() -> None:
    prereg = ROOT / "q1_e7_results" / "reporting" / "Q1_E7_PREREG_DRAFT.md"
    contract = ROOT / "Q1_CONTRIBUTION_CONTRACT.md"
    ptext = prereg.read_text(encoding="utf-8")
    ctext = contract.read_text(encoding="utf-8")
    prereg_checks = (
        r"\*\*Status:\*\*\s+\*\*ACCEPTED 2026-09-23\b",
        r"\*\*Proposer:\*\*\s+\*\*Sheryar Kiani\*\*",
        r"\*\*Approver:\*\*\s+\*\*Dr Hussain Kazmi\*\*",
        r"\*\*Acceptance date:\*\*\s+\*\*2026-09-23\*\*",
    )
    if "[REQUIRED]" in ptext or not all(re.search(pattern, ptext) for pattern in prereg_checks):
        raise RuntimeError("E7 named preregistration acceptance is missing")
    if not (
        re.search(r"\*\*Amended:\*\* 2026-09-23 .*Q1-E7 prereg \*\*ACCEPTED\*\*.*\*\*24-cell\*\*", ctext)
        and re.search(r"\| \*\*Q1-E7\*\* .*\*\*Prereg ACCEPTED; 0/24\*\* 2026-09-23", ctext)
    ):
        raise RuntimeError("E7 contract acceptance anchors are missing")


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite(value: object, label: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(x):
        raise RuntimeError(f"{label} is non-finite: {value!r}")
    return x


def _exact_int(value: object, label: str) -> int:
    x = _finite(value, label)
    if x != math.trunc(x):
        raise RuntimeError(f"{label} is not an exact integer: {value!r}")
    return int(x)


def _partition_dirichlet(y: np.ndarray, num_clients: int, alpha: float, seed: int):
    """Pure NumPy replica of the hash-frozen engine partition function."""
    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(y))
    client_indices = {cid: [] for cid in range(num_clients)}
    for c in range(num_classes):
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.maximum(proportions, 1e-6)
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(class_idx, splits)):
            client_indices[cid].extend(chunk.tolist())
    for cid in range(num_clients):
        if not client_indices[cid]:
            donor = max(range(num_clients), key=lambda c: len(client_indices[c]))
            if client_indices[donor]:
                client_indices[cid].append(client_indices[donor].pop())
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
        client_indices[cid] = np.asarray(client_indices[cid], dtype=np.int64)
    return client_indices


def _validate_cache() -> tuple[dict, np.ndarray, np.ndarray]:
    if sha256_file(CACHE_NPZ) != EXPECTED_CACHE["npz_sha256"]:
        raise RuntimeError("CIC train-scaler NPZ hash mismatch")
    if sha256_file(CACHE_META) != EXPECTED_CACHE["meta_sha256"]:
        raise RuntimeError("CIC train-scaler metadata hash mismatch")
    meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
    if meta.get("cache_key") != EXPECTED_CACHE["key"] or meta.get("scale_on") != "train":
        raise RuntimeError("CIC cache key or scaler lineage mismatch")
    if meta.get("classes") != list(CLASS_NAMES):
        raise RuntimeError("CIC cache class order mismatch")
    with np.load(CACHE_NPZ, allow_pickle=False) as blob:
        expected_arrays = {"X_train", "y_train", "X_test", "y_test", "cid_train"}
        if set(blob.files) != expected_arrays:
            raise RuntimeError(f"unexpected CIC cache schema: {sorted(blob.files)}")
        y_train = np.asarray(blob["y_train"])
        y_test = np.asarray(blob["y_test"])
        if tuple(y_train.shape) != (4825692,) or str(y_train.dtype) != "int64":
            raise RuntimeError("CIC cache y_train shape/dtype mismatch")
        if tuple(y_test.shape) != (1206424,) or str(y_test.dtype) != "int64":
            raise RuntimeError("CIC cache y_test shape/dtype mismatch")
        n_train, n_test, n_features = len(y_train), len(y_test), int(meta["num_features"])
    if (len(y_train), len(y_test), n_train, n_test, n_features) != (4825692, 1206424, 4825692, 1206424, 39):
        raise RuntimeError("CIC cache dimensions differ from the accepted E7 scope")
    validate_truth(y_test)
    hashes = {
        "y_train_int64_sha256": sha256_bytes(np.ascontiguousarray(y_train.astype("<i8", copy=False)).tobytes()),
        "y_test_int64_sha256": sha256_bytes(np.ascontiguousarray(y_test.astype("<i8", copy=False)).tobytes()),
        "y_test_uint8_sha256": sha256_bytes(np.ascontiguousarray(y_test.astype(np.uint8, copy=False)).tobytes()),
    }
    for key, got in hashes.items():
        if got != EXPECTED_CACHE[key]:
            raise RuntimeError(f"CIC cache {key} mismatch: {got}")
    return meta, y_train, y_test


def _partition_docs(y_train: np.ndarray) -> dict:
    out = {}
    for seed in SEEDS:
        part = _partition_dirichlet(y_train, 9, ALPHA, seed)
        validate_partition(part, len(y_train), 9)
        combined = hash_partition(part, 9)
        sizes = [int(len(part[c])) for c in range(9)]
        expected = EXPECTED_PARTITIONS[seed]
        if combined != expected["sha256"] or sizes != expected["client_sizes"]:
            raise RuntimeError(f"partition identity mismatch for seed {seed}")
        out[str(seed)] = {
            "sha256": combined,
            "client_sizes": sizes,
            "client_index_sha256": [
                sha256_bytes(np.ascontiguousarray(part[c].astype("<i8", copy=False)).tobytes())
                for c in range(9)
            ],
        }
    return out


def _anchor_rows(partitions: dict) -> list[dict]:
    rows = _read_csv(CONFIRM_RAW)
    selected: dict[tuple[str, str, str, str], dict] = {}
    ordered_keys = _job_keys()
    wanted = set(ordered_keys)
    for row in rows:
        try:
            alpha = float(row.get("Alpha", "nan"))
        except ValueError:
            continue
        if alpha != ALPHA:
            continue
        key = (row.get("Model", ""), row.get("Algorithm", ""), str(alpha), row.get("Seed", ""))
        if key in selected:
            raise RuntimeError(f"duplicate alpha=0.1 confirmatory row: {key}")
        selected[key] = row
    if set(selected) != wanted:
        raise RuntimeError(f"confirmatory alpha=0.1 grid mismatch missing={wanted-set(selected)} extra={set(selected)-wanted}")

    anchors = []
    for key in ordered_keys:
                model, algorithm, _, seed_text = key
                seed = int(seed_text)
                row = selected[key]
                if row.get("Mode") != "match_notebook" or row.get("ScaleOn") != "train":
                    raise RuntimeError(f"reference mode/scaler mismatch for {key}")
                if row.get("CacheKey") != EXPECTED_CACHE["key"]:
                    raise RuntimeError(f"reference cache key mismatch for {key}")
                if row.get("NpzSha256", "").lower() != EXPECTED_CACHE["npz_sha256"]:
                    raise RuntimeError(f"reference NPZ hash mismatch for {key}")
                if row.get("MetaSha256", "").lower() != EXPECTED_CACHE["meta_sha256"]:
                    raise RuntimeError(f"reference metadata hash mismatch for {key}")
                if _exact_int(row.get("N_Rounds"), f"{key} N_Rounds") != 30:
                    raise RuntimeError(f"reference round count mismatch for {key}")
                final_acc = _finite(row.get("Final_Acc"), f"{key} Final_Acc")
                final_f1 = _finite(row.get("Final_F1"), f"{key} Final_F1")
                hist_rel = str(row.get("HistoryCSV", "")).replace("\\", "/")
                hist_path = (ROOT / hist_rel).resolve()
                confirm_root = (ROOT / "q1_cic_conf_results" / "csv").resolve()
                if confirm_root not in hist_path.parents or not hist_path.is_file():
                    raise RuntimeError(f"unsafe or missing reference history for {key}: {hist_rel}")
                hist = _read_csv(hist_path)
                if len(hist) != 30:
                    raise RuntimeError(f"history length mismatch for {key}")
                for i, h in enumerate(hist, 1):
                    if _exact_int(h.get("round"), f"{key} history round") != i:
                        raise RuntimeError(f"history order mismatch for {key}")
                    for field in ("accuracy", "f1", "loss"):
                        _finite(h.get(field), f"{key} history {field}")
                if float(hist[-1]["accuracy"]) != final_acc or float(hist[-1]["f1"]) != final_f1:
                    raise RuntimeError(f"round-30 history mismatch for {key}")
                anchors.append({
                    "Wave": 0 if key == ("MLP", "FedAvg", "0.1", "42") else (1 if seed == 42 else 2),
                    "Model": model,
                    "Algorithm": algorithm,
                    "Alpha": "0.1",
                    "Seed": seed,
                    "Mode": "match_notebook",
                    "ScaleOn": "train",
                    "Final_Acc": row["Final_Acc"],
                    "Final_F1": row["Final_F1"],
                    "N_Rounds": 30,
                    "CacheKey": EXPECTED_CACHE["key"],
                    "NpzSha256": EXPECTED_CACHE["npz_sha256"],
                    "MetaSha256": EXPECTED_CACHE["meta_sha256"],
                    "PartitionSha256": partitions[str(seed)]["sha256"],
                    "HistoryCSV": hist_rel,
                    "HistorySha256": sha256_file(hist_path),
                })
    if len(anchors) != 24:
        raise RuntimeError(f"expected 24 anchors, found {len(anchors)}")
    return anchors


def _freeze_files() -> dict:
    out = {}
    for rel in FILES_TO_FREEZE:
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"file to freeze is missing: {rel}")
        got = sha256_file(path)
        if rel in EXPECTED_SOURCE_HASHES and got != EXPECTED_SOURCE_HASHES[rel]:
            raise RuntimeError(f"upstream frozen hash mismatch for {rel}: {got}")
        out[rel] = got
    return out


def main() -> int:
    _require_named_acceptance()
    if RAW.is_file() and _read_csv(RAW):
        raise RuntimeError("refusing to regenerate the E7 freeze after result rows exist")
    meta, y_train, _ = _validate_cache()
    partitions = _partition_docs(y_train)
    anchors = _anchor_rows(partitions)
    atomic_write_csv(ANCHORS, ANCHOR_FIELDS, anchors)
    anchor_sha = sha256_file(ANCHORS)
    if anchor_sha != EXPECTED_ANCHOR_SHA256:
        raise RuntimeError(
            "baseline anchor snapshot differs from the independently checked 24-row bytes\n"
            f"want {EXPECTED_ANCHOR_SHA256}\n"
            f"got  {anchor_sha}"
        )
    file_sha = _freeze_files()
    doc = {
        "rev": "q1-e7-r1-ACCEPTED",
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": {
            "proposer": "Sheryar Kiani",
            "approver": "Dr Hussain Kazmi",
            "date": "2026-09-23",
        },
        "scope": {
            "dataset": "CIC-IoT2023",
            "models": list(MODELS),
            "algorithms": list(ALGORITHMS),
            "alpha": ALPHA,
            "seeds": list(SEEDS),
            "jobs": 24,
            "waves": {"0": 1, "1": 7, "2": 16},
            "job_keys": [list(key) for key in _job_keys()],
            "mode": "match_notebook",
            "scale_on": "train",
            "n_train": 4825692,
            "n_test": 1206424,
            "num_features": 39,
            "num_classes": 8,
            "partition_hash_scheme": "sha256(cid uint64_le || length uint64_le || ordered indices int64_le), cid 0..8, no prefix",
            "client_index_hash_scheme": "sha256(ordered indices int64_le), no prefix",
        },
        "cfg": {
            "max_samples_per_file": 20000,
            "num_rounds": 30,
            "local_epochs": 3,
            "batch_size": 512,
            "lr": 0.001,
            "test_fraction": 0.2,
            "fedprox_mu": 0.01,
            "optimizer": "adam",
            "sgd_momentum": 0.9,
            "trimmed_fraction": 0.1,
            "num_clients": 9,
            "seeds": [42, 123, 7],
            "models": ["MLP", "CNN"],
            "dirichlet_alphas": [1.0, 0.5, 0.1],
        },
        "environment": {
            "python": "3.12.10",
            "torch": "2.5.1+cu121",
            "cuda_available": True,
            "cuda": "12.1",
            "numpy": "2.3.5",
            "pandas": "3.0.1",
            "sklearn": "1.8.0",
            "gpu": "Quadro P3200",
        },
        "cache": {
            **EXPECTED_CACHE,
            "npz": "probe_results/cache/cic_27bfed227ef77fab.npz",
            "meta": "probe_results/cache/cic_27bfed227ef77fab.meta.json",
            "scale_on": "train",
            "from_existing_only": True,
            "meta_num_features": int(meta["num_features"]),
            "meta_num_classes": int(meta["num_classes"]),
        },
        "classes": {"names": list(CLASS_NAMES), "supports": list(CLASS_SUPPORTS)},
        "partitions": partitions,
        "baseline": {
            "anchor_csv": "q1_e7_results/reporting/Q1_E7_BASELINE_ANCHORS.csv",
            "anchor_sha256": anchor_sha,
            "rows": 24,
            "source_raw": "q1_cic_conf_results/csv/q1_cic_conf_raw.csv",
            "selection": "exact alpha=0.1, both models, four arms, seeds 42/123/7",
        },
        "file_sha256": file_sha,
    }
    atomic_write_json(FROZEN, doc)
    print("Q1-E7 freeze built without training or cache writes")
    print(f"anchors: {ANCHORS} ({len(anchors)} rows, sha256={anchor_sha})")
    print(f"config:  {FROZEN} (sha256={sha256_file(FROZEN)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

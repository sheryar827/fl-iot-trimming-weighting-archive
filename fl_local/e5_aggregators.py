"""Q1-E5 aggregators and reproducibility checks.

Pure NumPy and hashing. This module does not import torch and does not train.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

DECISION_SEEDS = (127, 131, 137, 139, 149, 151, 157)
TUNING_SEED = 10007
BLOCKS = ("T1", "T2", "T5", "T6")
FAMILY = (
    ("T1", "FT-CoordMedian"),
    ("T2", "FT-CoordMedian"),
    ("T5", "FT-CoordMedian"),
    ("T6", "FT-CoordMedian"),
    ("T1", "FedNova-FedAvg"),
    ("T2", "FedNova-FedAvg"),
    ("T5", "FedNova-FedAvg"),
    ("T6", "FedNova-FedAvg"),
)
ACCEPTED_PREREG_SHA256 = (
    "cd20c7a61b3a41ad214cc51bf4221f7837604fd1236f75282fd07be346ec6517"
)
PRESENTED_PREREG_SHA256 = (
    "c985ef651b23bc03987a17fd205be314c8cce9f13fe7c8c4e8b1cf8911e59d5f"
)
ENGINE_SHA256 = (
    "7d201f9734b56c29457cd028933b0eca4862e241b17040e1cb6e88c6b0f130d2"
)

BOUND_SHA256 = {
    "fl_local/cic_cache.py":
        "91a322560c7a4ad3d47ed70dcc70ab2e00f6db134490debc872a35eb6b0760e3",
    "fl_local/nbaiot_cache.py":
        "55e0a136e54bb0759497aae7bcce1402ed9b56149614aaf38c4260b0f7c04693",
    "fl_local/engine.py": ENGINE_SHA256,
    "fl_local/e4_artifacts.py":
        "f08b15541c48788a30c7d7cfbc485f13e9e03969eb1e7325aa39c2c268c70cd7",
    "fl_local/e2_partition.py":
        "72778f968a03d2f934801b9270e4cdbfeb2759ad7960b20f514e2069c863e8a2",
    "probe_results/cache/cic_27bfed227ef77fab.npz":
        "8f032df1760a895b2358b8ff2da71491b09820c7af8c097dd00465fd7b981b38",
    "probe_results/cache/cic_27bfed227ef77fab.meta.json":
        "e5e0ef795fff100af24f6813ca3a2a61442e386aa01bf92878a9cfc0b03cb3b4",
    "probe_results/cache/nbaiot_4ddb8b93b8821650.npz":
        "c0a1e21cd6336d3a090e5feb9411758abbf8669dcca01bd476defdbcf8380854",
    "probe_results/cache/nbaiot_4ddb8b93b8821650.meta.json":
        "0124ac4c793417d53bd062f4694e72ba903ed42beee8873d578adf57a3686b23",
    "q1_e4_results/reporting/Q1_E4_FROZEN_CONFIG.json":
        "3e78215912fb7e1315c7a133cc964038387a88ee8d73577d14390cca6b28f9be",
    "q1_e4_results/reporting/q1_e4_seed.csv":
        "8a9218a130486b68c69f3dd40cf5dd82a217add985c66a91af6fe3dd97da90c9",
}

REQUIRED_ENVIRONMENT = {
    "python": "3.12.10",
    "numpy": "2.3.5",
    "pandas": "3.0.1",
    "sklearn": "1.8.0",
    "torch": "2.5.1+cu121",
    "cuda": "12.1",
    "cuda_available": True,
    "gpu": "Quadro P3200",
}

CUDA_FLAGS = {
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
}

CIC_ARRAYS = ("X_train", "X_test", "y_train", "y_test", "cid_train")
NBAIOT_ARRAYS = CIC_ARRAYS + ("cid_test",)
GENERATOR_STRIDE = 1_000_003
FEATURE_COUNTS = {"CIC-IoT2023": 39, "N-BaIoT": 115}
SCALER_FOR_BLOCK = {"T1": "CIC-IoT2023", "T2": "CIC-IoT2023", "T5": "N-BaIoT", "T6": "N-BaIoT"}
SCALER_FILES = {
    "CIC-IoT2023": (
        "q1_e5_results/tuning/scaler_CIC-IoT2023_s10007.npz",
        "q1_e5_results/tuning/scaler_CIC-IoT2023_s10007.json",
    ),
    "N-BaIoT": (
        "q1_e5_results/tuning/scaler_N-BaIoT_s10007.npz",
        "q1_e5_results/tuning/scaler_N-BaIoT_s10007.json",
    ),
}


def tuning_partition_files(block: str) -> tuple[str, str]:
    if block not in BLOCKS:
        raise ValueError(f"unknown block {block!r}")
    return (
        f"q1_e5_results/partitions/tuning_{block}_s10007.npz",
        f"q1_e5_results/partitions/tuning_{block}_s10007.json",
    )


TRAIN_ROWS = {"CIC-IoT2023": 4_825_692, "N-BaIoT": 1_418_112}
TUNING_RATES = ("0.001", "0.01", "0.1")


def tuning_index_files(dataset: str) -> tuple[str, str]:
    if dataset not in FEATURE_COUNTS:
        raise ValueError(f"unknown dataset {dataset!r}")
    return (
        f"q1_e5_results/tuning/indices_{dataset}_s10007.npz",
        f"q1_e5_results/tuning/indices_{dataset}_s10007.json",
    )


def tuning_matrix_files(dataset: str) -> tuple[str, str]:
    """Derived cache of the Section 4.4 transform. Not a sealed cache."""
    if dataset not in FEATURE_COUNTS:
        raise ValueError(f"unknown dataset {dataset!r}")
    return (
        f"q1_e5_results/tuning/matrices_{dataset}_s10007.npz",
        f"q1_e5_results/tuning/matrices_{dataset}_s10007.json",
    )


def learning_rate_value(token: str) -> float:
    if token not in TUNING_RATES:
        raise ValueError(f"learning rate {token!r} is not a tuning candidate")
    return float(token)


def tuning_holdout_sizes(n_rows: int) -> tuple[int, int]:
    """Float test_size=0.2 in scikit-learn 1.8.0 uses math.ceil(0.2 * n)."""
    rows = exact_nonnegative_int(n_rows, "n_rows")
    if rows < 2:
        raise ValueError("tuning split requires at least two rows")
    n_validation = math.ceil(0.2 * rows)
    n_subtrain = rows - n_validation
    if n_subtrain <= 0 or n_validation <= 0:
        raise ValueError("tuning split produced an empty part")
    return int(n_subtrain), int(n_validation)


def require_two_rows_per_stratum(strata) -> None:
    values = np.asarray(strata)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("strata must be a non-empty 1-D array")
    _, counts = np.unique(values, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("a stratum has fewer than two rows")


def combined_index_sha256(subtrain_idx, val_idx) -> str:
    """sha256(subtrain bytes immediately followed by validation bytes)."""
    subtrain = np.ascontiguousarray(np.asarray(subtrain_idx, dtype="<i8"))
    validation = np.ascontiguousarray(np.asarray(val_idx, dtype="<i8"))
    if subtrain.ndim != 1 or validation.ndim != 1:
        raise ValueError("index hash requires 1-D arrays")
    return sha256_bytes(subtrain.tobytes(order="C") + validation.tobytes(order="C"))


def pair_mean(first, second) -> float:
    """float64 mean of the two stored validation Final macro-F1 values."""
    try:
        values = np.array([float(first), float(second)], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("a tuning score is missing or non-finite") from exc
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("a tuning score is missing or non-finite")
    return float(values.mean())


def lock_learning_rate(arm_scores: Mapping[str, Mapping[str, float]]) -> str:
    """Pick the rate with the higher float64 mean. An exact tie keeps the smaller rate.

    Rates are scanned in ascending order. The leader changes only when the new
    mean is strictly greater, so exact equality does not move to a larger rate.
    """
    if set(arm_scores) != set(TUNING_RATES):
        raise ValueError("all three learning rates are required before a lock")
    leader = None
    leader_mean = None
    for token in TUNING_RATES:
        scores = arm_scores[token]
        if set(scores) != {"FedAvg", "FedNova"}:
            raise ValueError(f"learning rate {token} is missing an arm")
        mean = pair_mean(scores["FedAvg"], scores["FedNova"])
        if leader is None or mean > leader_mean:
            leader = token
            leader_mean = mean
    return str(leader)


def decision_partition_files(dataset: str, seed: int | None = None) -> tuple[str, str]:
    if dataset == "CIC-IoT2023":
        if int(seed) not in DECISION_SEEDS:
            raise ValueError(f"CIC decision seed is not in the E5 set: {seed!r}")
        return (
            f"q1_e5_results/partitions/decision_CIC-IoT2023_s{int(seed)}.npz",
            f"q1_e5_results/partitions/decision_CIC-IoT2023_s{int(seed)}.json",
        )
    if dataset == "N-BaIoT":
        return (
            "q1_e5_results/partitions/decision_N-BaIoT_device.npz",
            "q1_e5_results/partitions/decision_N-BaIoT_device.json",
        )
    raise ValueError(f"unknown dataset {dataset!r}")


def exact_nonnegative_int(value, label: str) -> int:
    """Accept a Python int or a NumPy scalar that is an exact nonnegative integer."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an exact nonnegative integer")
    try:
        number = float(np.asarray(value, dtype=np.float64).reshape(()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an exact nonnegative integer") from exc
    if not math.isfinite(number) or number < 0.0 or number != math.trunc(number):
        raise ValueError(f"{label} must be an exact nonnegative integer")
    if number > float(2**64 - 1):
        raise ValueError(f"{label} does not fit in uint64")
    return int(number)


def sha256_file(path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().lower()


def require_finite(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains a non-finite value")
    return values


def coordinate_median(client_params: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    """Ninth-client coordinate-wise median: the fifth ordered value, index 4."""
    if len(client_params) != 9:
        raise ValueError(f"CoordMedian requires 9 clients, got {len(client_params)}")
    n_layers = len(client_params[0])
    if any(len(client) != n_layers for client in client_params):
        raise ValueError("clients do not have the same number of tensors")
    aggregated = []
    for layer in range(n_layers):
        stacked = np.stack([np.asarray(client[layer]) for client in client_params], axis=0)
        require_finite(stacked, "median input")
        ordered = np.sort(stacked, axis=0)
        aggregated.append(ordered[4])
    return aggregated


def coefficient_sequence(tau: int) -> np.ndarray:
    """Return (c_1, ..., c_tau) with c_0 = 0 and c_k = 0.9 c_{k-1} + 1."""
    if int(tau) != tau or tau <= 0:
        raise ValueError(f"tau must be a positive integer, got {tau!r}")
    coefficients = np.empty(int(tau), dtype=np.float64)
    previous = 0.0
    for step in range(int(tau)):
        previous = 0.9 * previous + 1.0
        coefficients[step] = previous
    return coefficients


def coefficient_vector(tau: int) -> np.ndarray:
    """a_i = (c_tau, c_{tau-1}, ..., c_1)."""
    forward = coefficient_sequence(tau)
    return forward[::-1].copy()


def normalizing_constant(tau: int) -> float:
    """Scalar A_i = ||a_i||_1."""
    vector = coefficient_vector(tau)
    total = float(np.sum(vector))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"A_i must be positive and finite, got {total}")
    return total


def client_weights(client_sizes: Sequence[int], *, row_count: int) -> np.ndarray:
    """p_i = n_i / n in float64. n_i must sum to the rows trained in this run."""
    sizes = np.asarray([int(size) for size in client_sizes], dtype=np.int64)
    if sizes.ndim != 1 or sizes.size != 9:
        raise ValueError("client weights require nine client sizes")
    if np.any(sizes < 0) or int(sizes.sum()) <= 0:
        raise ValueError("client sizes must be non-negative with a positive total")
    if int(sizes.sum()) != int(row_count):
        raise ValueError(
            f"client sizes sum to {int(sizes.sum())}, not the trained row count {int(row_count)}"
        )
    weights = sizes.astype(np.float64) / np.float64(row_count)
    if not np.all(weights >= 0.0) or not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("client weights must be nonnegative and sum to one")
    return weights


def weights_for_run(
    client_sizes: Sequence[int],
    *,
    run_kind: str,
    subtrain_rows: int,
    full_training_rows: int,
) -> np.ndarray:
    if run_kind == "tuning":
        return client_weights(client_sizes, row_count=subtrain_rows)
    if run_kind == "decision":
        return client_weights(client_sizes, row_count=full_training_rows)
    raise ValueError(f"run_kind must be 'tuning' or 'decision', got {run_kind!r}")


def _as_layer_list(deltas: Sequence[Sequence[np.ndarray]]) -> list[list[np.ndarray]]:
    if len(deltas) != 9:
        raise ValueError(f"server update requires 9 clients, got {len(deltas)}")
    layers = len(deltas[0])
    converted = []
    for client in deltas:
        if len(client) != layers:
            raise ValueError("delta tensors differ in count")
        converted.append([require_finite(np.asarray(layer, dtype=np.float64), "delta") for layer in client])
    return converted


def _nine_weights(weights: Sequence[float]) -> np.ndarray:
    p = np.asarray(list(weights), dtype=np.float64)
    if p.shape != (9,) or not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("weights must be nine finite nonnegative values")
    if abs(float(p.sum()) - 1.0) > 1e-12:
        raise ValueError("weights must sum to one")
    return p


def _cast_like(value: np.ndarray, template) -> np.ndarray:
    """Float64 accumulation, then the incoming parameter dtype."""
    return np.ascontiguousarray(np.asarray(value).astype(np.asarray(template).dtype, copy=False))


def fednova_server_update(
    start: Sequence[np.ndarray],
    deltas: Sequence[Sequence[np.ndarray]],
    weights: Sequence[float],
    normalizing_constants: Sequence[float],
) -> list[np.ndarray]:
    """w_next = w + tau_eff * sum_i p_i delta_i / A_i, with scalar A_i."""
    layers = _as_layer_list(deltas)
    p = _nine_weights(weights)
    a_values = np.asarray(list(normalizing_constants), dtype=np.float64)
    if a_values.shape != (9,):
        raise ValueError("FedNova requires nine scalar A_i values")
    if np.any(a_values <= 0.0) or not np.all(np.isfinite(a_values)):
        raise ValueError("A_i must be positive and finite")
    tau_eff = float(np.sum(p * a_values))
    updated = []
    for layer_index, base in enumerate(start):
        base64 = require_finite(np.asarray(base, dtype=np.float64), "server parameters")
        mixed = np.zeros_like(base64)
        for client_index in range(9):
            mixed = mixed + p[client_index] * layers[client_index][layer_index] / a_values[client_index]
        updated.append(_cast_like(base64 + tau_eff * mixed, base))
    return updated


def fedavg_server_update(
    start: Sequence[np.ndarray],
    deltas: Sequence[Sequence[np.ndarray]],
    weights: Sequence[float],
    normalizing_constants: Sequence[float] | None = None,
) -> list[np.ndarray]:
    """w_next = w + sum_i p_i delta_i. A_i is ignored."""
    del normalizing_constants
    layers = _as_layer_list(deltas)
    p = _nine_weights(weights)
    updated = []
    for layer_index, base in enumerate(start):
        base64 = require_finite(np.asarray(base, dtype=np.float64), "server parameters")
        mixed = np.zeros_like(base64)
        for client_index in range(9):
            mixed = mixed + p[client_index] * layers[client_index][layer_index]
        updated.append(_cast_like(base64 + mixed, base))
    return updated


def vector_division_differs_from_scalar(delta: np.ndarray, tau: int) -> bool:
    """The server must not divide a parameter vector by a_i."""
    vector = coefficient_vector(tau)
    scalar = float(np.sum(vector))
    delta64 = np.asarray(delta, dtype=np.float64)
    if delta64.size != vector.size:
        raise ValueError("this fixture requires delta and a_i to have the same length")
    scalar_result = delta64 / scalar
    vector_result = delta64 / vector
    return not np.allclose(scalar_result, vector_result)


def expected_local_steps(n_rows: int, *, epochs: int = 3, batch_size: int = 512) -> int:
    if int(n_rows) <= 0 or int(epochs) <= 0 or int(batch_size) <= 0:
        raise ValueError("step count inputs must be positive")
    return int(epochs) * ((int(n_rows) + int(batch_size) - 1) // int(batch_size))


def arrays_byte_equal(left, right) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    left_bytes = np.ascontiguousarray(left_array).tobytes(order="C")
    right_bytes = np.ascontiguousarray(right_array).tobytes(order="C")
    return left_bytes == right_bytes


def index_sha256(indices) -> str:
    array = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    if array.ndim != 1:
        raise ValueError("index hash requires a 1-D array")
    return sha256_bytes(array.tobytes(order="C"))


def scaler_sha256(
    *,
    mean_,
    var_,
    scale_,
    n_samples_seen_: int,
    feature_count: int,
    subtrain_index_sha256: str,
    validation_index_sha256: str,
) -> str:
    """Canonical hash from Section 4.4. dtype is required to be <f8."""
    mean = np.ascontiguousarray(np.asarray(mean_, dtype="<f8"))
    var = np.ascontiguousarray(np.asarray(var_, dtype="<f8"))
    scale = np.ascontiguousarray(np.asarray(scale_, dtype="<f8"))
    features = exact_nonnegative_int(feature_count, "feature_count")
    if features <= 0:
        raise ValueError("feature_count must be positive")
    if mean.shape != (features,) or var.shape != mean.shape or scale.shape != mean.shape:
        raise ValueError("scaler arrays must have shape (feature_count,)")
    seen = exact_nonnegative_int(n_samples_seen_, "n_samples_seen_")
    digest = hashlib.sha256()
    digest.update(features.to_bytes(8, "little", signed=False))
    digest.update(seen.to_bytes(8, "little", signed=False))
    digest.update(b"<f8")
    digest.update(mean.tobytes(order="C"))
    digest.update(var.tobytes(order="C"))
    digest.update(scale.tobytes(order="C"))
    digest.update(bytes.fromhex(subtrain_index_sha256))
    digest.update(bytes.fromhex(validation_index_sha256))
    return digest.hexdigest().lower()


def validate_compact_partition(partition: Mapping[int, np.ndarray], n_subtrain: int) -> None:
    """c0..c8 are disjoint and cover exactly 0 .. n_subtrain-1."""
    if int(n_subtrain) <= 0:
        raise ValueError("n_subtrain must be positive")
    seen = np.zeros(int(n_subtrain), dtype=np.int8)
    for client_id in range(9):
        if client_id not in partition:
            raise ValueError(f"tuning partition missing client {client_id}")
        indices = np.asarray(partition[client_id], dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"client {client_id} tuning indices must be non-empty")
        if np.any(indices < 0) or np.any(indices >= int(n_subtrain)):
            raise ValueError(
                f"client {client_id} index is not a compact subtrain position"
            )
        if np.unique(indices).size != indices.size or seen[indices].any():
            raise ValueError(f"client {client_id} overlaps another tuning client")
        seen[indices] = 1
    if int(seen.sum()) != int(n_subtrain):
        raise ValueError("tuning indices do not cover 0 .. n_subtrain-1")


# The first four names are the active CoordMedian family. The last four are historical.
MEDIAN_FAMILY = (
    ("T1", "FT-CoordMedian"),
    ("T2", "FT-CoordMedian"),
    ("T5", "FT-CoordMedian"),
    ("T6", "FT-CoordMedian"),
)


def holm_four(raw_p: Mapping[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Holm across the four CoordMedian tests only. Ties keep that order. Cap at 1."""
    if set(raw_p) != set(MEDIAN_FAMILY):
        raise ValueError("Holm input must be exactly the four CoordMedian comparisons")
    return _holm_adjust(raw_p, MEDIAN_FAMILY)


def holm_eight(raw_p: Mapping[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Historical eight-test Holm. Not the active CoordMedian study."""
    if set(raw_p) != set(FAMILY):
        raise ValueError("Holm input must be exactly the eight E5 comparisons")
    return _holm_adjust(raw_p, FAMILY)


def _holm_adjust(
    raw_p: Mapping[tuple[str, str], float], family: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], float]:
    order = {key: index for index, key in enumerate(family)}
    values = {}
    for key in family:
        p_value = float(raw_p[key])
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"invalid p-value for {key}: {raw_p[key]!r}")
        values[key] = p_value
    ranked = sorted(family, key=lambda key: (values[key], order[key]))
    adjusted = {}
    running = 0.0
    family_size = len(ranked)
    for rank, key in enumerate(ranked):
        candidate = min(1.0, (family_size - rank) * values[key])
        running = max(running, candidate)
        adjusted[key] = running
    return {key: adjusted[key] for key in family}


def same_parameter_bytes(left: Iterable[np.ndarray], right: Iterable[np.ndarray]) -> bool:
    left_list = list(left)
    right_list = list(right)
    if len(left_list) != len(right_list):
        return False
    return all(arrays_byte_equal(a, b) for a, b in zip(left_list, right_list))

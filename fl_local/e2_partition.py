"""Q1-E2 device-based train partition. Do not import into CIC confirmatory runner."""
from __future__ import annotations

import numpy as np


def partition_by_device_ids(cid_np: np.ndarray, num_clients: int | None = None):
    """Train-client k = physical device id k.

    Not Dirichlet. Not leave-one-device-out. Not a deployment on nine devices.
    """
    ids = np.asarray(cid_np, dtype=np.int64)
    n = int(ids.max()) + 1 if num_clients is None else int(num_clients)
    part = {cid: np.where(ids == cid)[0].astype(np.int64) for cid in range(n)}
    validate_nine_nonempty_partitions(part, n_train=int(ids.shape[0]), n_clients=n)
    return part


def validate_nine_nonempty_partitions(
    part: dict, n_train: int, n_clients: int = 9
) -> None:
    if len(part) != n_clients:
        raise ValueError(f"expected {n_clients} clients, got {len(part)}")
    sizes = []
    seen = np.zeros(n_train, dtype=np.int8)
    for cid in range(n_clients):
        idx = np.asarray(part[cid], dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"client {cid} is empty")
        if np.any(idx < 0) or np.any(idx >= n_train):
            raise ValueError(f"client {cid} has out-of-range indices")
        if seen[idx].any():
            raise ValueError(f"client {cid} overlaps another client")
        seen[idx] = 1
        sizes.append(int(idx.size))
    if int(seen.sum()) != n_train:
        raise ValueError("union of client indices does not cover the train set")
    return None

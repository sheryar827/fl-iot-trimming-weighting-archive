"""Pure integrity helpers for the accepted Q1-E4 seed expansion.

This module has no torch import, no cache builder, and no training path.  It is
safe for the CPU-only freeze builder and integrity suite.
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


PARTITION_HASH_SCHEME = (
    "sha256(cid uint64_le || length uint64_le || ordered indices int64_le), "
    "cid 0..8, no prefix"
)
CLIENT_HASH_SCHEME = "sha256(ordered indices int64_le), no prefix"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest().lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().lower()


def finite_float(value: object, label: str = "value") -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} is non-finite: {value!r}")
    return out


def exact_int(value: object, label: str = "value") -> int:
    out = finite_float(value, label)
    if out != math.trunc(out):
        raise ValueError(f"{label} is not an exact integer: {value!r}")
    return int(out)


def load_csv(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_binary_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp_name: str | None = None
    try:
        with _atomic_binary_file(path) as handle:
            tmp_name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def atomic_write_json(path: Path, document: Mapping) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_bytes(Path(path), payload.encode("utf-8"))


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping],
    *,
    temp_tag: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if temp_tag is not None:
        tag = str(temp_tag)
        if not tag or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in tag
        ):
            raise ValueError(
                "atomic CSV temp_tag must contain only letters, digits, '_' or '-'"
            )
        prefix = f".{path.name}.{tag}."
    else:
        prefix = f".{path.name}."
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=prefix,
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            tmp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def validate_partition(
    partition: Mapping[int, np.ndarray], n_total: int, num_clients: int = 9
) -> None:
    if set(partition) != set(range(num_clients)):
        raise ValueError("partition client IDs must be exactly 0..8")
    seen = np.zeros(int(n_total), dtype=np.uint8)
    covered = 0
    for client_id in range(num_clients):
        indices = np.asarray(partition[client_id])
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"client {client_id} partition must be non-empty 1-D")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"client {client_id} indices must be integers")
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= n_total):
            raise ValueError(f"client {client_id} has out-of-range indices")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"client {client_id} contains duplicate indices")
        if seen[indices].any():
            raise ValueError(f"client {client_id} overlaps another client")
        seen[indices] = 1
        covered += int(indices.size)
    if covered != int(n_total) or int(seen.sum()) != int(n_total):
        raise ValueError("partition is not a disjoint exhaustive cover")


def hash_partition(
    partition: Mapping[int, np.ndarray], num_clients: int = 9
) -> str:
    digest = hashlib.sha256()
    for client_id in range(num_clients):
        if client_id not in partition:
            raise ValueError(f"partition missing client {client_id}")
        indices = np.asarray(partition[client_id], dtype="<i8")
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"client {client_id} partition must be non-empty 1-D")
        digest.update(int(client_id).to_bytes(8, "little", signed=False))
        digest.update(int(indices.size).to_bytes(8, "little", signed=False))
        digest.update(np.ascontiguousarray(indices).tobytes(order="C"))
    return digest.hexdigest().lower()


def partition_document(
    partition: Mapping[int, np.ndarray], n_total: int, num_clients: int = 9
) -> dict:
    validate_partition(partition, n_total=n_total, num_clients=num_clients)
    return {
        "sha256": hash_partition(partition, num_clients),
        "client_sizes": [int(len(partition[c])) for c in range(num_clients)],
        "client_index_sha256": [
            sha256_bytes(
                np.ascontiguousarray(
                    np.asarray(partition[c], dtype="<i8")
                ).tobytes(order="C")
            )
            for c in range(num_clients)
        ],
    }


def safe_relative_file(root: Path, allowed_root: Path, relative: object) -> Path:
    rel = Path(str(relative).replace("\\", "/"))
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError(f"path must be safe and relative: {relative!r}")
    root_resolved = Path(root).resolve()
    allowed_resolved = Path(allowed_root).resolve()
    resolved = (root_resolved / rel).resolve()
    if resolved != allowed_resolved and allowed_resolved not in resolved.parents:
        raise ValueError(f"path escapes allowed output tree: {relative!r}")
    return resolved

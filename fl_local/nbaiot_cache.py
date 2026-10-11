"""Parse N-BaIoT once; cache preprocessed arrays for fast restarts."""
from __future__ import annotations

import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

GLOBAL_SEED = 42

KNOWN_DEVICES = [
    "danmini_doorbell",
    "ecobee_thermostat",
    "ennio_doorbell",
    "philips_b120n10_baby_monitor",
    "provision_pt_737e_security_camera",
    "provision_pt_838_security_camera",
    "samsung_snh_1011_n_webcam",
    "simplehome_xcs7_1002_wht_security_camera",
    "simplehome_xcs7_1003_wht_security_camera",
]


def find_extract_dir(workspace: Path, root: Path) -> Path:
    candidates = [
        workspace / "nbaiot_data" / "extracted",
        root / "nbaiot_data" / "extracted",
        root / "fl-iot-botnet-nbaiot-cic-iot2023-main" / "nbaiot_data" / "extracted",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "N-BaIoT extracted data not found. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def parse_nbaiot_csvs(csv_paths, max_samples=None, global_seed=GLOBAL_SEED):
    frames = []
    mirai_types = ["ack", "scan", "syn", "udpplain", "udp"]
    gafgyt_types = ["combo", "junk", "scan", "tcp", "udp"]

    for path in csv_paths:
        norm = str(path).lower().replace(" ", "_").replace("-", "_")
        parts = [
            p.lower().replace(" ", "_").replace("-", "_") for p in Path(path).parts
        ]
        fname = Path(path).stem.lower().replace(" ", "_").replace("-", "_")

        if "demonstrate" in norm or "description" in norm:
            continue

        device = "unknown"
        for d in KNOWN_DEVICES:
            if d in norm:
                device = d
                break

        label = "unknown"
        if "benign" in norm:
            label = "benign"
        else:
            is_mirai = "mirai" in norm
            is_gafgyt = "gafgyt" in norm or "bashlite" in norm
            if is_mirai:
                for atk in mirai_types:
                    if atk in fname or any(atk == p for p in parts):
                        label = f"mirai_{atk}"
                        break
                if label == "unknown":
                    label = "mirai_other"
            elif is_gafgyt:
                for atk in gafgyt_types:
                    if atk in fname or any(atk == p for p in parts):
                        label = f"gafgyt_{atk}"
                        break
                if label == "unknown":
                    label = "gafgyt_other"

        if label == "unknown" or device == "unknown":
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
            if df.shape[1] < 10:
                continue
            if max_samples and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=global_seed)
            # single-assign avoids fragmented-frame warnings on wide CSVs
            df = df.copy()
            df = df.assign(device=device, label=label)
            frames.append(df)
        except Exception as e:
            print(f"  skip {path}: {e}")

    if not frames:
        raise RuntimeError("No data loaded from CSVs")
    return pd.concat(frames, ignore_index=True)


def _cache_key(
    extract_dir: Path,
    max_samples: int,
    test_fraction: float,
    scale_on: str = "full",
) -> str:
    # v1 = notebook-faithful: StandardScaler on full X, then split (test leak).
    # train = split first, fit scaler on train only (confirmatory protocol).
    if scale_on == "full":
        raw = f"{extract_dir.resolve()}|{max_samples}|{test_fraction}|{GLOBAL_SEED}|v1"
    elif scale_on == "train":
        raw = (
            f"{extract_dir.resolve()}|{max_samples}|{test_fraction}|"
            f"{GLOBAL_SEED}|scale_on=train|cid_test=1"
        )
    else:
        raise ValueError(f"scale_on must be 'full' or 'train', got {scale_on!r}")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_or_build_nbaiot_cache(
    extract_dir: Path,
    cache_dir: Path,
    max_samples_per_file: int = 20000,
    test_fraction: float = 0.2,
    force_rebuild: bool = False,
    scale_on: str = "full",
    allow_build: bool = True,
) -> dict:
    """
    Returns dict with numpy arrays + metadata. Also writes:
      cache_dir / nbaiot_<key>.npz
      cache_dir / nbaiot_<key>.meta.json
    If allow_build is False and the cache files are missing, raises
    FileNotFoundError and does not parse CSVs or write NPZ/meta.
    """
    key = _cache_key(
        extract_dir, max_samples_per_file, test_fraction, scale_on=scale_on
    )
    npy_path = cache_dir / f"nbaiot_{key}.npz"
    meta_path = cache_dir / f"nbaiot_{key}.meta.json"

    if npy_path.exists() and meta_path.exists() and not force_rebuild:
        print(f"Loading N-BaIoT cache: {npy_path.name}")
        blob = np.load(npy_path, allow_pickle=False)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out = {
            "X_train": blob["X_train"],
            "y_train": blob["y_train"],
            "X_test": blob["X_test"],
            "y_test": blob["y_test"],
            "cid_train": blob["cid_train"],
            "meta": meta,
            "cache_path": str(npy_path),
            "from_cache": True,
        }
        if "cid_test" in blob.files:
            out["cid_test"] = blob["cid_test"]
        return out

    if not allow_build:
        raise FileNotFoundError(
            f"N-BaIoT cache missing and allow_build=False: {npy_path.name}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building N-BaIoT cache from {extract_dir} …")
    all_csvs = sorted(glob.glob(str(extract_dir / "**" / "*.csv"), recursive=True))
    print(f"  CSV files: {len(all_csvs)}")
    raw_df = parse_nbaiot_csvs(all_csvs, max_samples=max_samples_per_file)
    print(f"  samples={len(raw_df)} labels={raw_df['label'].nunique()}")

    feature_df = raw_df.drop(columns=["device", "label"], errors="ignore")
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.fillna(0, inplace=True)
    X = feature_df.values.astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(raw_df["label"].values)
    device_le = LabelEncoder()
    client_ids = device_le.fit_transform(raw_df["device"].values)

    if scale_on == "full":
        # Notebook-faithful (legacy): scaler sees test rows.
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(
            X,
            y,
            client_ids,
            test_size=test_fraction,
            random_state=GLOBAL_SEED,
            stratify=y,
        )
    elif scale_on == "train":
        X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(
            X,
            y,
            client_ids,
            test_size=test_fraction,
            random_state=GLOBAL_SEED,
            stratify=y,
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
    else:
        raise ValueError(f"scale_on must be 'full' or 'train', got {scale_on!r}")

    meta = {
        "num_features": int(X_train.shape[1]),
        "num_classes": int(len(le.classes_)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": list(map(str, le.classes_)),
        "devices": list(map(str, device_le.classes_)),
        "max_samples_per_file": max_samples_per_file,
        "test_fraction": test_fraction,
        "global_seed": GLOBAL_SEED,
        "extract_dir": str(extract_dir.resolve()),
        "cache_key": key,
        "scale_on": scale_on,
    }

    save_kw = dict(
        X_train=X_train,
        y_train=y_train.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        cid_train=cid_train.astype(np.int64),
    )
    # cid_test is E2/train-scaler only. Do not rewrite the legacy v1 NPZ schema
    # (same cache key) or E1/E1b/E3 hashes would change on a full-scaler rebuild.
    if scale_on == "train":
        save_kw["cid_test"] = cid_test.astype(np.int64)
    np.savez_compressed(npy_path, **save_kw)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {npy_path}  train={meta['n_train']} test={meta['n_test']}")

    return {
        "X_train": X_train,
        "y_train": y_train.astype(np.int64),
        "X_test": X_test,
        "y_test": y_test.astype(np.int64),
        "cid_train": cid_train.astype(np.int64),
        "cid_test": cid_test.astype(np.int64),
        "meta": meta,
        "cache_path": str(npy_path),
        "from_cache": False,
    }


def tensors_from_cache(data: dict, pin_memory: bool = True) -> dict:
    """Host tensors for indexing; pin if CUDA will be used."""
    def _t(a, dtype):
        t = torch.from_numpy(np.ascontiguousarray(a)).to(dtype)
        if pin_memory and torch.cuda.is_available():
            t = t.pin_memory()
        return t

    return {
        "X_train": _t(data["X_train"], torch.float32),
        "y_train": _t(data["y_train"], torch.long),
        "X_test": _t(data["X_test"], torch.float32),
        "y_test": _t(data["y_test"], torch.long),
        "meta": data["meta"],
    }

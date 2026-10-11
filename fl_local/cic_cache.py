"""
Parse CIC-IoT2023 once; cache preprocessed arrays for fast restarts.

Faithful to FL_IoT_Botnet_CIC_IoT2023_MLP_CNN.ipynb cells 3.1–4.1:
  - max_samples_per_file cap with random_state=GLOBAL_SEED (42)
  - 8-category collapse via categorize_cic_attack
  - drop original label columns; drop object dtype feature columns
  - synthetic device 'cic_testbed'
  - scale_on="full" (default, notebook-faithful): StandardScaler on full X then split
  - scale_on="train" (confirmatory): split first, fit scaler on train only

Data layout (manual download — not auto-fetched):
  <workspace>/cic_iot_2023_data/extracted/**/*.csv
  or Research Paper/cic_iot_2023_data/extracted/
"""
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


def find_cic_extract_dir(workspace: Path, root: Path) -> Path:
    """
    Accept either notebook layout (.../cic_iot_2023_data/extracted/) or a local
    dump (.../cic_iot_2023_dataset/ with per-attack subfolders of CSVs).
    """
    candidates = [
        root / "cic_iot_2023_dataset",  # observed local layout (per-class folders)
        workspace / "cic_iot_2023_dataset",
        workspace / "cic_iot_2023_data" / "extracted",
        root / "cic_iot_2023_data" / "extracted",
        root / "fl-iot-botnet-nbaiot-cic-iot2023-main" / "cic_iot_2023_data" / "extracted",
    ]
    for p in candidates:
        if p.exists() and any(p.rglob("*.csv")):
            return p
    preferred = root / "cic_iot_2023_dataset"
    raise FileNotFoundError(
        "CIC-IoT2023 CSVs not found.\n"
        "Download pre-processed CSVs only from:\n"
        "  https://www.unb.ca/cic/datasets/iotdataset-2023.html\n"
        "Place under one of:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + f"\nPreferred: {preferred.resolve()}"
    )


def categorize_cic_attack(fname: str) -> str:
    """Map a CIC-IoT2023 filename to one of 8 attack categories (notebook)."""
    f = fname.lower()
    if "benign" in f:
        return "benign"
    if "ddos" in f:
        return "ddos"
    if ("dos-" in f or "dos_" in f) and "ddos" not in f:
        return "dos"
    if "mirai" in f:
        return "mirai"
    if any(k in f for k in ["recon", "scan", "hostdiscovery", "vulnerability"]):
        return "recon"
    if any(k in f for k in ["browser", "backdoor", "command", "sql", "xss", "upload"]):
        return "web"
    if any(k in f for k in ["brute", "dictionary"]):
        return "bruteforce"
    if any(k in f for k in ["spoof", "mitm", "arp"]):
        return "spoofing"
    return "unknown"


def parse_cic_iot_2023_csvs(csv_paths, max_samples=None, global_seed=GLOBAL_SEED):
    """Notebook parse_cic_iot_2023_csvs — 8-class unified frame."""
    frames = []
    label_counts = defaultdict(int)
    fine_counts = defaultdict(int)

    for path in csv_paths:
        fname = Path(path).stem
        label = categorize_cic_attack(fname)
        if label == "unknown":
            print(f"  skip unrecognised: {Path(path).name}")
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            if df.shape[1] < 5:
                continue
            if max_samples and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=global_seed)
            for col in ("label", "Label", "LABEL"):
                if col in df.columns:
                    df = df.drop(columns=[col])
            df = df.copy()
            df = df.assign(device="cic_testbed", label=label)
            frames.append(df)
            label_counts[label] += len(df)
            fine_counts[fname] += len(df)
        except Exception as e:
            print(f"  skip {path}: {e}")

    if not frames:
        raise RuntimeError(
            "No CIC data loaded. Check cic_iot_2023_data/extracted/ filenames."
        )

    combined = pd.concat(frames, ignore_index=True)
    text_cols = [
        c
        for c in combined.columns
        if c not in ("device", "label") and combined[c].dtype == "object"
    ]
    if text_cols:
        print(f"  Dropping {len(text_cols)} non-numeric column(s): {text_cols}")
        combined = combined.drop(columns=text_cols)

    print(f"Loaded {len(combined):,} samples from {len(frames)} file(s).")
    print("8-category label distribution:")
    for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:12s} : {v:>10,}")
    print(f"Fine-grained types: {len(fine_counts)}")
    return combined


def _cache_key(
    extract_dir: Path,
    max_samples: int,
    test_fraction: float,
    scale_on: str = "full",
) -> str:
    if scale_on == "full":
        raw = f"cic|{extract_dir.resolve()}|{max_samples}|{test_fraction}|{GLOBAL_SEED}|v1"
    elif scale_on == "train":
        raw = (
            f"cic|{extract_dir.resolve()}|{max_samples}|{test_fraction}|"
            f"{GLOBAL_SEED}|scale_on=train"
        )
    else:
        raise ValueError(f"scale_on must be 'full' or 'train', got {scale_on!r}")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_or_build_cic_cache(
    extract_dir: Path,
    cache_dir: Path,
    max_samples_per_file: int = 20000,
    test_fraction: float = 0.2,
    force_rebuild: bool = False,
    scale_on: str = "full",
) -> dict:
    """
    Returns dict with numpy arrays + metadata.
    Writes: cache_dir / cic_<key>.npz and .meta.json
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        extract_dir, max_samples_per_file, test_fraction, scale_on=scale_on
    )
    npy_path = cache_dir / f"cic_{key}.npz"
    meta_path = cache_dir / f"cic_{key}.meta.json"

    if npy_path.exists() and meta_path.exists() and not force_rebuild:
        print(f"Loading CIC cache: {npy_path.name}")
        blob = np.load(npy_path, allow_pickle=False)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "X_train": blob["X_train"],
            "y_train": blob["y_train"],
            "X_test": blob["X_test"],
            "y_test": blob["y_test"],
            "cid_train": blob["cid_train"],
            "meta": meta,
            "cache_path": str(npy_path),
            "from_cache": True,
        }

    print(f"Building CIC cache from {extract_dir} …")
    all_csvs = sorted(glob.glob(str(extract_dir / "**" / "*.csv"), recursive=True))
    print(f"  CSV files: {len(all_csvs)}")
    if not all_csvs:
        raise FileNotFoundError(f"No CSVs under {extract_dir}")

    raw_df = parse_cic_iot_2023_csvs(
        all_csvs, max_samples=max_samples_per_file, global_seed=GLOBAL_SEED
    )

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
        "dataset": "CIC-IoT2023",
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
        "probe_target_mlp_fedavg_a1_s42_final_f1": 0.6646197239002002,
    }

    np.savez_compressed(
        npy_path,
        X_train=X_train,
        y_train=y_train.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        cid_train=cid_train.astype(np.int64),
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"  wrote {npy_path.name}  train={meta['n_train']} test={meta['n_test']} "
        f"F={meta['num_features']} C={meta['num_classes']}"
    )
    print(f"  classes: {meta['classes']}")

    return {
        "X_train": X_train,
        "y_train": y_train.astype(np.int64),
        "X_test": X_test,
        "y_test": y_test.astype(np.int64),
        "cid_train": cid_train.astype(np.int64),
        "meta": meta,
        "cache_path": str(npy_path),
        "from_cache": False,
    }


def tensors_from_cache(data: dict, pin_memory: bool = True) -> dict:
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

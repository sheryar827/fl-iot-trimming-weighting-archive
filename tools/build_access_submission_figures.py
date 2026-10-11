"""CPU-only figures and audits for the IEEE Access submission revision.

Reads sealed result tables and caches. Does not train, rewrite caches, or
edit sealed packets.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl_local.e2_partition import partition_by_device_ids  # noqa: E402
from fl_local.e4_artifacts import hash_partition  # noqa: E402
from fl_local.engine import IoTBotnetCNN, IoTBotnetMLP, partition_dirichlet  # noqa: E402

OUT = ROOT / "IEEE_Access_LaTeX_template__1_" / "generated"
CACHE = ROOT / "probe_results" / "cache"
SEED_CSV = ROOT / "q1_e4_results" / "reporting" / "q1_e4_seed.csv"
FROZEN = ROOT / "q1_e4_results" / "reporting" / "Q1_E4_FROZEN_CONFIG.json"
E7_MEAN = ROOT / "q1_e7_results" / "reporting" / "q1_e7_per_class_seed_mean.csv"
E4_CELLS = ROOT / "q1_e4_results" / "cells"
E4_HIST = ROOT / "q1_e4_results" / "csv"
E5_CELLS = ROOT / "q1_e5_results" / "coordmedian" / "cells"

PROSPECTIVE = [127, 131, 137, 139, 149, 151, 157]
BLOCK_ORDER = ["T1", "T2", "T5", "T6"]
CONTRASTS = [
    ("FT-UW", "FedTrimmedAvg", "UnweightedFedAvg"),
    ("WT-FA", "WeightedFedTrimmedAvg", "FedAvg"),
    ("FA-UW", "FedAvg", "UnweightedFedAvg"),
    ("WT-FT", "WeightedFedTrimmedAvg", "FedTrimmedAvg"),
]
PACKET_MEAN = {
    ("T1", "FT-UW"): 0.3714,
    ("T1", "WT-FA"): 7.4325,
    ("T1", "FA-UW"): -12.2101,
    ("T1", "WT-FT"): -5.1490,
    ("T2", "FT-UW"): -1.5393,
    ("T2", "WT-FA"): 6.3105,
    ("T2", "FA-UW"): -10.7252,
    ("T2", "WT-FT"): -2.8754,
    ("T5", "FT-UW"): -0.0201,
    ("T5", "WT-FA"): -1.7038,
    ("T5", "FA-UW"): 1.6414,
    ("T5", "WT-FT"): -0.0423,
    ("T6", "FT-UW"): 0.7091,
    ("T6", "WT-FA"): 1.2596,
    ("T6", "FA-UW"): -0.4904,
    ("T6", "WT-FT"): 0.0601,
}
PACKET_INTERACTION = {"T1": 7.0612, "T2": 7.8498, "T5": -1.6837, "T6": 0.5505}


def holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = (m - rank) * p_values[idx]
        running = max(running, raw)
        adj[idx] = min(1.0, running)
    return adj


def paired_row(diffs: np.ndarray) -> dict:
    diffs = np.asarray(diffs, dtype=np.float64)
    n = int(diffs.size)
    mean = float(diffs.mean())
    if n < 2 or float(diffs.var(ddof=1)) == 0.0:
        raise SystemExit("zero-variance paired contrast; stop rather than invent a p-value")
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t_res = stats.ttest_rel(diffs, np.zeros(n))
    signs = (
        int(np.sum(diffs > 0)),
        int(np.sum(diffs == 0)),
        int(np.sum(diffs < 0)),
    )
    return {
        "n": n,
        "mean_pp": mean * 100.0,
        "ci_low_pp": (mean - tcrit * se) * 100.0,
        "ci_high_pp": (mean + tcrit * se) * 100.0,
        "signs_pos": signs[0],
        "signs_tie": signs[1],
        "signs_neg": signs[2],
        "raw_p": float(t_res.pvalue),
        "diffs": diffs,
    }


def signflip_p(diffs: np.ndarray) -> float:
    """Two-sided exact sign-flip test on the sum of paired differences."""
    d = np.asarray(diffs, dtype=np.float64)
    n = int(d.size)
    observed = abs(float(d.sum()))
    count = 0
    total = 1 << n
    for mask in range(total):
        signed = 0.0
        for i in range(n):
            signed += d[i] if (mask >> i) & 1 else -d[i]
        if abs(signed) + 1e-15 >= observed:
            count += 1
    return count / total


def prospective_frame() -> pd.DataFrame:
    df = pd.read_csv(SEED_CSV)
    sub = df[df["Seed"].isin(PROSPECTIVE) & df["Block"].isin(BLOCK_ORDER)].copy()
    if sub["SeedRole"].nunique() != 1:
        raise SystemExit(f"unexpected seed roles: {sorted(sub['SeedRole'].unique())}")
    return sub


def contrast_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    points = []
    for block in BLOCK_ORDER:
        part = df[df["Block"] == block]
        for name, left, right in CONTRASTS:
            a = part[part["Algorithm"] == left].set_index("Seed")["Final_F1"]
            b = part[part["Algorithm"] == right].set_index("Seed")["Final_F1"]
            both = a.index.intersection(b.index)
            if list(both.sort_values()) != PROSPECTIVE and sorted(both) != PROSPECTIVE:
                if sorted(int(s) for s in both) != PROSPECTIVE:
                    raise SystemExit(f"{block} {name} seeds {sorted(both)}")
            diffs = np.array([float(a.loc[s] - b.loc[s]) for s in PROSPECTIVE])
            stat = paired_row(diffs)
            key = (block, name)
            shown = round(stat["mean_pp"], 4)
            if abs(shown - PACKET_MEAN[key]) > 5e-5:
                raise SystemExit(f"packet mismatch {key}: {shown} vs {PACKET_MEAN[key]}")
            stat.update({"block": block, "contrast": name})
            rows.append({k: v for k, v in stat.items() if k != "diffs"})
            for seed, diff in zip(PROSPECTIVE, diffs):
                points.append(
                    {
                        "block": block,
                        "contrast": name,
                        "seed": int(seed),
                        "diff_pp": float(diff) * 100.0,
                    }
                )
    out = pd.DataFrame(rows)
    p = holm(out["raw_p"].tolist())
    out["holm_p"] = p
    supported = out.loc[out["holm_p"] < 0.05, ["block", "contrast"]]
    if supported.shape[0] != 1 or tuple(supported.iloc[0]) != ("T2", "FA-UW"):
        raise SystemExit(f"official Holm support changed: {supported}")
    flip = []
    for _, row in out.iterrows():
        part = df[df["Block"] == row["block"]]
        spec = [c for c in CONTRASTS if c[0] == row["contrast"]][0]
        a = part[part["Algorithm"] == spec[1]].set_index("Seed")["Final_F1"]
        b = part[part["Algorithm"] == spec[2]].set_index("Seed")["Final_F1"]
        diffs = np.array([float(a.loc[s] - b.loc[s]) for s in PROSPECTIVE])
        flip.append(signflip_p(diffs))
    out["signflip_p"] = flip
    out["signflip_holm_p"] = holm(flip)
    if (out["signflip_holm_p"] < 0.05).any():
        raise SystemExit("a sign-flip comparison survived Holm; do not use the prepared sentence")
    return out, pd.DataFrame(points)


def interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block in BLOCK_ORDER:
        part = df[df["Block"] == block]

        def series(name: str) -> pd.Series:
            return part[part["Algorithm"] == name].set_index("Seed")["Final_F1"]

        wt_fa = np.array(
            [float(series("WeightedFedTrimmedAvg").loc[s] - series("FedAvg").loc[s]) for s in PROSPECTIVE]
        )
        ft_uw = np.array(
            [float(series("FedTrimmedAvg").loc[s] - series("UnweightedFedAvg").loc[s]) for s in PROSPECTIVE]
        )
        diffs = wt_fa - ft_uw
        stat = paired_row(diffs)
        shown = round(stat["mean_pp"], 4)
        if abs(shown - PACKET_INTERACTION[block]) > 5e-5:
            raise SystemExit(f"interaction mismatch {block}: {shown} vs {PACKET_INTERACTION[block]}")
        stat.update({"block": block})
        rows.append({k: v for k, v in stat.items() if k != "diffs"})
    return pd.DataFrame(rows)


def final_best_and_history(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df.copy()
    sub["gap_pp"] = (sub["Best_F1"] - sub["Final_F1"]) * 100.0
    summary = (
        sub.groupby("Block", sort=False)
        .agg(
            cells=("Final_F1", "size"),
            mean_final=("Final_F1", "mean"),
            mean_best=("Best_F1", "mean"),
            mean_gap_pp=("gap_pp", "mean"),
            max_gap_pp=("gap_pp", "max"),
            final_is_best=("gap_pp", lambda s: int(np.sum(np.abs(s) <= 1e-12))),
        )
        .reindex(BLOCK_ORDER)
        .reset_index()
    )
    best_rounds = []
    checked = 0
    for _, row in sub.iterrows():
        stem = None
        matches = list(E4_HIST.glob(f"history_e4_{row['Block']}_*_s{int(row['Seed'])}_train_match_notebook.csv"))
        algo = {
            "FedAvg": "FedAvg",
            "UnweightedFedAvg": "UnweightedFedAvg",
            "FedTrimmedAvg": "FedTrimmedAvg",
            "WeightedFedTrimmedAvg": "WeightedFedTrimmedAvg",
        }[row["Algorithm"]]
        hits = [p for p in matches if f"_{algo}_s{int(row['Seed'])}_" in p.name]
        if len(hits) != 1:
            raise SystemExit(f"history match {row['Block']} {algo} {row['Seed']}: {len(hits)}")
        hist = pd.read_csv(hits[0])
        if list(hist["round"]) != list(range(1, 31)):
            raise SystemExit(f"history rounds {hits[0].name}")
        final = float(hist.loc[hist["round"] == 30, "f1"].iloc[0])
        if abs(final - float(row["Final_F1"])) > 1e-12:
            raise SystemExit(f"history final mismatch {hits[0].name}")
        best_round = int(hist.loc[hist["f1"].idxmax(), "round"])
        best_rounds.append(best_round)
        checked += 1
    if checked != 112:
        raise SystemExit(f"expected 112 histories, checked {checked}")
    sub = sub.copy()
    sub["best_round"] = best_rounds
    rounds = (
        sub.groupby("Block", sort=False)
        .agg(
            median_best_round=("best_round", "median"),
            mean_best_round=("best_round", "mean"),
            best_at_30=("best_round", lambda s: int(np.sum(s == 30))),
        )
        .reindex(BLOCK_ORDER)
        .reset_index()
    )
    return summary, rounds


def cell_hours(directory: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for path in directory.glob(pattern):
        doc = json.loads(path.read_text(encoding="utf-8"))
        start = datetime.fromisoformat(doc["started_utc"])
        end = datetime.fromisoformat(doc["finished_utc"])
        hours = (end - start).total_seconds() / 3600.0
        rows.append({"file": path.name, "hours": hours, "block": doc.get("block") or doc.get("cell", {}).get("block")})
    if not rows:
        raise SystemExit(f"no cells in {directory} / {pattern}")
    return pd.DataFrame(rows)


def e4_hours() -> pd.DataFrame:
    rows = []
    for path in E4_CELLS.glob("e4_*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        job = doc["job"]
        seed = int(job[5])
        if seed not in PROSPECTIVE:
            continue
        rows.append(
            {
                "block": job[0],
                "algorithm": job[3],
                "seed": seed,
                "hours": float(doc["train_s"]) / 3600.0,
            }
        )
    df = pd.DataFrame(rows)
    if len(df) != 112:
        raise SystemExit(f"expected 112 E4 prospective cells, found {len(df)}")
    return (
        df.groupby("block", sort=False)
        .agg(cells=("hours", "size"), mean_h=("hours", "mean"), max_h=("hours", "max"), sum_h=("hours", "sum"))
        .reindex(BLOCK_ORDER)
        .reset_index()
    )


def e5_hours() -> dict:
    rows = []
    for path in E5_CELLS.glob("*.json"):
        if path.name.endswith(".history.json"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "started_utc" not in doc or "finished_utc" not in doc:
            continue
        start = datetime.fromisoformat(doc["started_utc"])
        end = datetime.fromisoformat(doc["finished_utc"])
        rows.append((end - start).total_seconds() / 3600.0)
    if len(rows) != 28:
        raise SystemExit(f"expected 28 E5 cells with clocks, found {len(rows)}")
    arr = np.array(rows)
    return {"cells": 28, "mean_h": float(arr.mean()), "max_h": float(arr.max()), "sum_h": float(arr.sum())}


def parameter_counts() -> list[dict]:
    rows = []
    for dataset, features, classes in (("CIC-IoT2023", 39, 8), ("N-BaIoT", 115, 11)):
        for name, cls in (("MLP", IoTBotnetMLP), ("CNN", IoTBotnetCNN)):
            model = cls(features, classes)
            count = int(sum(p.numel() for p in model.parameters()))
            rows.append(
                {
                    "dataset": dataset,
                    "model": name,
                    "features": features,
                    "classes": classes,
                    "parameters": count,
                }
            )
    return rows


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def label_rows(y: np.ndarray, partition: dict, n_classes: int, dataset: str, seed: str, kind: str) -> list[dict]:
    y = np.asarray(y)
    global_counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    q = global_counts / global_counts.sum()
    sizes = np.array([len(partition[c]) for c in range(9)], dtype=np.float64)
    missing = []
    js = []
    for cid in range(9):
        counts = np.bincount(y[partition[cid]], minlength=n_classes).astype(np.float64)
        missing.append(int(np.sum(counts == 0)))
        p = counts / counts.sum()
        js.append(js_divergence(p, q))
    return [
        {
            "dataset": dataset,
            "partition": kind,
            "seed": seed,
            "min_client": int(sizes.min()),
            "max_client": int(sizes.max()),
            "max_over_min": float(sizes.max() / sizes.min()),
            "cv": float(sizes.std(ddof=0) / sizes.mean()),
            "largest_share": float(sizes.max() / sizes.sum()),
            "mean_missing_classes": float(np.mean(missing)),
            "max_missing_classes": int(np.max(missing)),
            "mean_js": float(np.mean(js)),
            "max_js": float(np.max(js)),
        }
    ]


def heterogeneity() -> list[dict]:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    rows = []
    cic = np.load(CACHE / "cic_27bfed227ef77fab.npz")
    y = np.asarray(cic["y_train"])
    n_classes = int(y.max()) + 1
    cic_parts = frozen["partitions"]["CIC-IoT2023"]
    for seed in PROSPECTIVE:
        part = partition_dirichlet(y, 9, 0.1, int(seed))
        digest = hash_partition(part, 9)
        expected = cic_parts[str(seed)]["sha256"]
        if digest != expected:
            raise SystemExit(f"CIC seed {seed} alpha 0.1 hash {digest} != {expected}")
        sizes = [len(part[c]) for c in range(9)]
        if sizes != list(cic_parts[str(seed)]["client_sizes"]):
            raise SystemExit(f"CIC seed {seed} sizes mismatch")
        rows.extend(label_rows(y, part, n_classes, "CIC-IoT2023", str(seed), "Dirichlet alpha=0.1"))
    del cic
    nb = np.load(CACHE / "nbaiot_4ddb8b93b8821650.npz")
    y = np.asarray(nb["y_train"])
    cid = np.asarray(nb["cid_train"])
    part = partition_by_device_ids(cid, num_clients=9)
    digest = hash_partition(part, 9)
    expected = frozen["partitions"]["N-BaIoT"]["sha256"]
    if digest != expected:
        raise SystemExit(f"N-BaIoT device hash {digest} != {expected}")
    rows.extend(label_rows(y, part, int(y.max()) + 1, "N-BaIoT", "device", "source device"))
    del nb
    return rows


def void_rows(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x.view(np.dtype((np.void, x.shape[1] * 4))).reshape(-1)


def leakage() -> dict:
    report = {}
    for name, npz, has_cid_test in (
        ("CIC-IoT2023", "cic_27bfed227ef77fab.npz", False),
        ("N-BaIoT", "nbaiot_4ddb8b93b8821650.npz", True),
    ):
        blob = np.load(CACHE / npz)
        keys = set(blob.files)
        x_train = void_rows(blob["X_train"])
        x_test = void_rows(blob["X_test"])
        train_unique = np.unique(x_train)
        pos = np.searchsorted(train_unique, x_test)
        pos = np.clip(pos, 0, train_unique.size - 1)
        matched = train_unique[pos] == x_test
        y_train = np.asarray(blob["y_train"])
        y_test = np.asarray(blob["y_test"])
        train_classes = set(int(v) for v in np.unique(y_train))
        test_classes = set(int(v) for v in np.unique(y_test))
        item = {
            "npz_keys": sorted(keys),
            "n_train": int(x_train.size),
            "n_test": int(x_test.size),
            "unique_train_rows": int(train_unique.size),
            "test_rows_exactly_in_train": int(matched.sum()),
            "test_row_match_fraction": float(matched.mean()),
            "train_classes": sorted(train_classes),
            "test_classes": sorted(test_classes),
            "test_classes_missing_from_train": sorted(test_classes - train_classes),
            "train_classes_missing_from_test": sorted(train_classes - test_classes),
        }
        if has_cid_test and "cid_test" in keys:
            cid_train = np.asarray(blob["cid_train"])
            cid_test = np.asarray(blob["cid_test"])
            train_ids = set(int(v) for v in np.unique(cid_train))
            test_ids = set(int(v) for v in np.unique(cid_test))
            item.update(
                {
                    "train_source_ids": sorted(train_ids),
                    "test_source_ids": sorted(test_ids),
                    "source_ids_in_both": sorted(train_ids & test_ids),
                    "test_rows_from_train_source": int(np.isin(cid_test, list(train_ids)).sum()),
                }
            )
        else:
            cid_train = np.asarray(blob["cid_train"])
            item.update(
                {
                    "cid_test_stored": False,
                    "train_source_ids": sorted(int(v) for v in np.unique(cid_train)),
                }
            )
        report[name] = item
        del blob, x_train, x_test, train_unique
    return report


def draw(contrasts: pd.DataFrame, points: pd.DataFrame, interaction: pd.DataFrame, e7: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    labels = [f"{r.block} {r.contrast}" for r in contrasts.itertuples(index=False)]
    y = np.arange(len(contrasts))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 7.4), dpi=200)
    for i, row in enumerate(contrasts.itertuples(index=False)):
        yy = y[i]
        pts = points[(points.block == row.block) & (points.contrast == row.contrast)]["diff_pp"]
        ax.scatter(pts, np.full(len(pts), yy), s=18, color="#8AA0B4", zorder=2)
        color = "#0B6E4F" if row.holm_p < 0.05 else "#1F4E79"
        ax.plot([row.ci_low_pp, row.ci_high_pp], [yy, yy], color=color, lw=1.6, zorder=3)
        ax.scatter([row.mean_pp], [yy], marker="D", s=28, color=color, zorder=4)
    ax.axvline(0, color="#666666", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Final macro-F1 difference (percentage points)")
    ax.set_title("Seven seeds per contrast. Green marks the one Holm-supported paired-t comparison.")
    fig.tight_layout()
    fig.savefig(OUT / "q1_e4_forest.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    y = np.arange(len(interaction))[::-1]
    for i, row in enumerate(interaction.itertuples(index=False)):
        ax.plot([row.ci_low_pp, row.ci_high_pp], [y[i], y[i]], color="#7A5B00", lw=1.8)
        ax.scatter([row.mean_pp], [y[i]], marker="D", s=30, color="#7A5B00")
    ax.axvline(0, color="#666666", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(list(interaction["block"]))
    ax.set_xlabel("Interaction (WT−FA) − (FT−UW), percentage points")
    ax.set_title("Descriptive only. These four estimates are outside the 16-test Holm family.")
    fig.tight_layout()
    fig.savefig(OUT / "q1_e4_interaction.png")
    plt.close(fig)

    arms = ["FA", "UW", "FT", "WT"]
    models = ["MLP", "CNN"]
    classes = list(dict.fromkeys(e7["Class"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=200, sharey=True)
    for ax, model in zip(axes, models):
        grid = np.zeros((len(classes), len(arms)))
        for i, cls in enumerate(classes):
            for j, arm in enumerate(arms):
                hit = e7[(e7.Model == model) & (e7.Class == cls) & (e7.Arm == arm)]
                if len(hit) != 1:
                    raise SystemExit(f"heatmap cell {model} {cls} {arm}: {len(hit)}")
                grid[i, j] = float(hit["Mean_F1"].iloc[0])
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms)
        ax.set_title(f"CIC-IoT2023 {model}, alpha=0.1")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_yticklabels(classes, fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.88, wspace=0.18, top=0.88, bottom=0.12)
    cax = fig.add_axes([0.90, 0.16, 0.02, 0.66])
    fig.colorbar(im, cax=cax, label="Mean per-class F1")
    fig.savefig(OUT / "q1_e7_heatmap.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = prospective_frame()
    contrasts, points = contrast_table(df)
    interaction = interaction_table(df)
    gaps, rounds = final_best_and_history(df)
    hours = e4_hours()
    median_hours = e5_hours()
    params = parameter_counts()
    hetero = heterogeneity()
    audit = leakage()
    e7 = pd.read_csv(E7_MEAN)
    draw(contrasts, points, interaction, e7)
    contrasts.drop(columns=[], errors="ignore").to_csv(OUT / "e4_contrasts_with_signflip.csv", index=False)
    points.to_csv(OUT / "e4_contrast_points.csv", index=False)
    interaction.to_csv(OUT / "e4_interaction_descriptive.csv", index=False)
    gaps.to_csv(OUT / "e4_final_vs_best.csv", index=False)
    rounds.to_csv(OUT / "e4_best_round.csv", index=False)
    hours.to_csv(OUT / "e4_wall_clock.csv", index=False)
    pd.DataFrame(params).to_csv(OUT / "model_parameters.csv", index=False)
    pd.DataFrame(hetero).to_csv(OUT / "heterogeneity.csv", index=False)
    (OUT / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUT / "e5_wall_clock.json").write_text(json.dumps(median_hours, indent=2), encoding="utf-8")
    print("contrasts")
    print(contrasts.to_string(index=False))
    print("interaction")
    print(interaction.to_string(index=False))
    print("gaps")
    print(gaps.to_string(index=False))
    print("rounds")
    print(rounds.to_string(index=False))
    print("hours")
    print(hours.to_string(index=False))
    print("median", median_hours)
    print("params", params)
    print("hetero")
    print(pd.DataFrame(hetero).to_string(index=False))
    print("leakage keys", {k: audit[k].get("test_rows_exactly_in_train") for k in audit})


if __name__ == "__main__":
    main()

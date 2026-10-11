"""Build the Q1-E5 CoordMedian reporting packet from finished cells.

This command does not train, import torch, rebuild a cache, or edit the
sealed Q1-E4 packet, the accepted preregistration, or the working manuscript.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_local.e4_artifacts import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)
from fl_local.e4_stats import paired_summary  # noqa: E402
from fl_local.e5_aggregators import (  # noqa: E402
    ACCEPTED_PREREG_SHA256,
    BLOCKS,
    BOUND_SHA256,
    DECISION_SEEDS,
    MEDIAN_FAMILY,
    decision_partition_files,
    holm_four,
)


REPORTING = ROOT / "q1_e5_results" / "reporting"
CELLS = ROOT / "q1_e5_results" / "coordmedian" / "cells"
PROGRESS = ROOT / "q1_e5_results" / "coordmedian" / "progress.log"
SEED_CSV = REPORTING / "q1_e5_seed_pairs.csv"
PAIRED_CSV = REPORTING / "q1_e5_paired.csv"
SUMMARY_JSON = REPORTING / "q1_e5_summary.json"
PACKET_MD = REPORTING / "Q1_E5_REPORTING_PACKET.md"
MANIFEST = REPORTING / "SHA256SUMS.txt"
SEALED_SEED_CSV = ROOT / "q1_e4_results" / "reporting" / "q1_e4_seed.csv"
FROZEN_CONFIG = ROOT / "q1_e4_results" / "reporting" / "Q1_E4_FROZEN_CONFIG.json"
PREREG = ROOT / "q1_e5_results" / "Q1_E5_PREREG_DRAFT.md"
CONTRAST = "FT-CoordMedian"
FAMILY_ALPHA = 0.05

# Display contract of the drafted manuscript table. These strings are checked
# after the paired tests are computed. They are not inputs to the mean or p-value.
MANUSCRIPT_DISPLAY = (
    ("T1", "0.3471", "[-1.6900, 2.3842]", "3/0/4", "0.691258", "1", "inconclusive"),
    ("T2", "0.6388", "[-1.7203, 2.9980]", "3/0/4", "0.532201", "1", "inconclusive"),
    ("T5", "-1.7913", "[-4.5363, 0.9537]", "4/0/3", "0.161432", "0.645727", "inconclusive"),
    ("T6", "-1.6825", "[-4.8882, 1.5233]", "3/0/4", "0.246421", "0.739263", "inconclusive"),
)
MEAN_PP_REPR = {
    "T1": "0.34707417273825186",
    "T2": "0.638833572917353",
    "T5": "-1.7912852855501402",
    "T6": "-1.6824850719962376",
}
CI_PP_REPR = {
    "T1": ("-1.6900200597375576", "2.3841684052140613"),
    "T2": ("-1.720322072444092", "2.9979892182787977"),
    "T5": ("-4.536286406529948", "0.9537158354296669"),
    "T6": ("-4.888243648898091", "1.5232735049056152"),
}
T_REPR = {
    "T1": "0.4168977030912353",
    "T2": "0.6625969946299447",
    "T5": "-1.5967633530254206",
    "T6": "-1.2842179357189156",
}

PAIR_FIELDS = [
    "Block",
    "Dataset",
    "Model",
    "Condition",
    "Seed",
    "CoordMedian_Final_F1",
    "FedTrimmedAvg_Final_F1",
    "Delta",
    "CoordMedian_Final_Acc",
    "FedTrimmedAvg_Final_Acc",
    "PartitionSha256",
    "HistorySha256",
    "CellSha256",
    "StartedUTC",
    "FinishedUTC",
]
PAIRED_FIELDS = [
    "Block",
    "Dataset",
    "Model",
    "Contrast",
    "LeftArm",
    "RightArm",
    "SeedOrder",
    "DeltaBySeedJSON",
    "n",
    "df",
    "Mean_pp",
    "SD_pp",
    "Median_pp",
    "Min_pp",
    "Max_pp",
    "Negative",
    "Zero",
    "Positive",
    "CI95Low_pp",
    "CI95High_pp",
    "TStatistic",
    "RawP",
    "HolmP",
    "Decision",
    "SupportedDirection",
]


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pp(value: float) -> str:
    return repr(100.0 * float(value))


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is non-finite: {value!r}")
    return number


def _decision(mean: float, holm_p: float) -> tuple[str, str]:
    """Two-sided seven-seed rule. There is no preferred-direction reversal."""
    if not math.isfinite(mean) or not math.isfinite(holm_p):
        _fail("decision inputs must be finite")
    if holm_p < FAMILY_ALPHA and mean > 0.0:
        return "supported", "trimmed_mean_higher"
    if holm_p < FAMILY_ALPHA and mean < 0.0:
        return "supported", "median_higher"
    return "inconclusive", ""


def _bound_hash(relative: str) -> str:
    recorded = BOUND_SHA256[relative]
    digest = sha256_file(ROOT / relative)
    if digest != recorded:
        _fail(f"{relative} hash {digest} does not equal the bound digest")
    return digest


def _block_specs(frozen: dict) -> dict[str, dict]:
    rows = frozen["scope"]["blocks"]
    if [row["block"] for row in rows] != list(BLOCKS):
        _fail("frozen config blocks are not T1, T2, T5, T6")
    specs = {}
    for row in rows:
        specs[row["block"]] = {
            "dataset": row["dataset"],
            "model": row["model"],
            "condition": row["condition"],
        }
    return specs


def _cache_sha(dataset: str) -> str:
    if dataset == "CIC-IoT2023":
        return BOUND_SHA256["probe_results/cache/cic_27bfed227ef77fab.npz"]
    if dataset == "N-BaIoT":
        return BOUND_SHA256["probe_results/cache/nbaiot_4ddb8b93b8821650.npz"]
    _fail(f"unknown dataset {dataset}")
    return ""


def _cache_key(dataset: str) -> str:
    if dataset == "CIC-IoT2023":
        return "27bfed227ef77fab"
    if dataset == "N-BaIoT":
        return "4ddb8b93b8821650"
    _fail(f"unknown dataset {dataset}")
    return ""


def _expected_names() -> list[str]:
    names = []
    for block in BLOCKS:
        for seed in DECISION_SEEDS:
            stem = f"{block}_CoordMedian_s{int(seed)}"
            names.append(f"{stem}.json")
            names.append(f"{stem}.history.json")
    return names


def _check_history(cell: dict, history_path: Path) -> None:
    payload = history_path.read_bytes()
    digest = sha256_file(history_path)
    if digest != str(cell["history_sha256"]).lower():
        _fail(f"{history_path.name} hash does not match the cell record")
    rows = json.loads(payload.decode("utf-8"))
    if not isinstance(rows, list) or [int(row["round"]) for row in rows] != list(range(1, 31)):
        _fail(f"{history_path.name} is not scored rounds 1 through 30")
    for row in rows:
        for key in ("loss", "accuracy", "macro_f1"):
            _finite(row[key], f"{history_path.name} {key}")
    last = rows[-1]
    if last["macro_f1"] != cell["Final_F1"] or last["accuracy"] != cell["Final_Acc"]:
        _fail(f"{history_path.name} endpoint does not match Final_F1 and Final_Acc")


def _load_cells(specs: dict[str, dict]) -> list[dict]:
    found = sorted(path.name for path in CELLS.iterdir() if path.is_file())
    if found != sorted(_expected_names()):
        _fail("coordmedian/cells does not contain exactly the 28 cell and history files")
    loaded = []
    for block in BLOCKS:
        spec = specs[block]
        for seed in DECISION_SEEDS:
            path = CELLS / f"{block}_CoordMedian_s{int(seed)}.json"
            cell = _load_json(path)
            if cell.get("status") != "complete" or cell.get("finite") is not True:
                _fail(f"{path.name} is not a complete finite row")
            if cell.get("kind") != "decision" or cell.get("decision_cell") is not True:
                _fail(f"{path.name} is not a decision cell")
            if cell.get("arm") != "CoordMedian" or cell.get("arm_name") != "CoordMedian-Adam":
                _fail(f"{path.name} is not CoordMedian-Adam")
            if cell.get("block") != block or int(cell.get("seed", -1)) != int(seed):
                _fail(f"{path.name} identity does not match its filename")
            if int(cell.get("rounds", 0)) != 30:
                _fail(f"{path.name} did not record 30 rounds")
            if cell.get("learning_rate") != "1e-3" or float(cell.get("learning_rate_value", -1)) != 1e-3:
                _fail(f"{path.name} learning rate is not 1e-3")
            if cell.get("mode") != "match_notebook" or cell.get("optimizer") != "adam":
                _fail(f"{path.name} is not the sealed match-notebook Adam loop")
            if cell.get("comparator") != "sealed E4 FedTrimmedAvg-Adam":
                _fail(f"{path.name} comparator is not the sealed E4 row")
            if cell.get("dataset") != spec["dataset"] or cell.get("model") != spec["model"]:
                _fail(f"{path.name} dataset or model does not match the frozen block")
            if str(cell.get("cache_sha256", "")).lower() != _cache_sha(spec["dataset"]):
                _fail(f"{path.name} cache digest does not match the bound NPZ")
            _finite(cell["Final_F1"], f"{path.name} Final_F1")
            _finite(cell["Final_Acc"], f"{path.name} Final_Acc")
            _check_history(cell, CELLS / f"{block}_CoordMedian_s{int(seed)}.history.json")
            cell["_cell_sha256"] = sha256_file(path)
            loaded.append(cell)
    if len(loaded) != 28:
        _fail("the loaded plan is not 28 cells")
    return loaded


def _frozen_partition(frozen: dict, dataset: str, seed: int) -> dict:
    node = frozen["partitions"][dataset]
    if dataset == "CIC-IoT2023":
        return node[str(int(seed))]
    if dataset == "N-BaIoT":
        return node
    _fail(f"unknown dataset {dataset}")
    return {}


def _check_partition(cell: dict) -> str:
    _, json_rel = decision_partition_files(cell["dataset"], int(cell["seed"]))
    saved = _load_json(ROOT / json_rel)
    frozen_entry = _frozen_partition(
        cell["_frozen"], cell["dataset"], int(cell["seed"])
    )
    digest = str(saved["sha256"]).lower()
    if digest != str(cell["partition_sha256"]).lower():
        _fail(f"{cell['block']} seed {cell['seed']} partition does not match the cell")
    if digest != str(frozen_entry["sha256"]).lower():
        _fail(f"{cell['block']} seed {cell['seed']} partition does not match the frozen config")
    if saved["client_index_sha256"] != frozen_entry["client_index_sha256"]:
        _fail(f"{cell['block']} seed {cell['seed']} client index digests differ")
    if saved["client_sizes"] != frozen_entry["client_sizes"]:
        _fail(f"{cell['block']} seed {cell['seed']} client sizes differ")
    return digest


def _load_sealed_pairs(specs: dict[str, dict]) -> dict[tuple[str, int], dict]:
    if sha256_file(SEALED_SEED_CSV) != BOUND_SHA256["q1_e4_results/reporting/q1_e4_seed.csv"]:
        _fail("sealed q1_e4_seed.csv hash changed")
    selected: dict[tuple[str, int], dict] = {}
    with SEALED_SEED_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Algorithm"] != "FedTrimmedAvg":
                continue
            block = row["Block"]
            seed = int(row["Seed"])
            if block not in BLOCKS or seed not in DECISION_SEEDS:
                continue
            key = (block, seed)
            if key in selected:
                _fail(f"duplicate sealed FedTrimmedAvg row for {block} seed {seed}")
            if row["SeedRole"] != "prospective new seed" or row["ScaleOn"] != "train":
                _fail(f"{block} seed {seed} is not a train-scaler prospective row")
            spec = specs[block]
            if row["Dataset"] != spec["dataset"] or row["Model"] != spec["model"]:
                _fail(f"{block} seed {seed} dataset or model differs from the frozen block")
            if row["Condition"] != spec["condition"]:
                _fail(f"{block} seed {seed} condition differs from the frozen block")
            if row["CacheKey"] != _cache_key(spec["dataset"]):
                _fail(f"{block} seed {seed} cache key differs from the sealed cache")
            _finite(row["Final_F1"], f"sealed {block} seed {seed} Final_F1")
            _finite(row["Final_Acc"], f"sealed {block} seed {seed} Final_Acc")
            selected[key] = row
    expected = {(block, int(seed)) for block in BLOCKS for seed in DECISION_SEEDS}
    if set(selected) != expected:
        _fail("the sealed CSV does not contain exactly 28 prospective FedTrimmedAvg rows")
    return selected


def _pair_rows(cells: list[dict], sealed: dict[tuple[str, int], dict], specs: dict[str, dict]) -> list[dict]:
    rows = []
    for cell in cells:
        block = cell["block"]
        seed = int(cell["seed"])
        partner = sealed[(block, seed)]
        if str(partner["PartitionSha256"]).lower() != str(cell["partition_sha256"]).lower():
            _fail(f"{block} seed {seed} comparator partition differs from the CoordMedian cell")
        delta = _finite(partner["Final_F1"], "FT") - _finite(cell["Final_F1"], "median")
        rows.append(
            {
                "Block": block,
                "Dataset": specs[block]["dataset"],
                "Model": specs[block]["model"],
                "Condition": specs[block]["condition"],
                "Seed": str(seed),
                "CoordMedian_Final_F1": cell["Final_F1"],
                "FedTrimmedAvg_Final_F1": partner["Final_F1"],
                "Delta": repr(delta),
                "CoordMedian_Final_Acc": cell["Final_Acc"],
                "FedTrimmedAvg_Final_Acc": partner["Final_Acc"],
                "PartitionSha256": str(cell["partition_sha256"]).lower(),
                "HistorySha256": str(cell["history_sha256"]).lower(),
                "CellSha256": cell["_cell_sha256"],
                "StartedUTC": cell["started_utc"],
                "FinishedUTC": cell["finished_utc"],
            }
        )
    return rows


def _analyses(pair_rows: list[dict]) -> list[dict]:
    raw_p = {}
    prepared = []
    for block in BLOCKS:
        block_rows = [row for row in pair_rows if row["Block"] == block]
        if [int(row["Seed"]) for row in block_rows] != list(DECISION_SEEDS):
            _fail(f"{block} seed order is not the seven decision seeds")
        deltas = [_finite(row["Delta"], f"{block} delta") for row in block_rows]
        summary = paired_summary(deltas, include_test=True)
        if summary["n"] != 7 or summary["df"] != 6:
            _fail(f"{block} is not n=7, df=6")
        if summary["test_status"] != "finite" or summary["t_statistic"] is None:
            _fail(f"{block} did not take the ordinary paired-t path")
        raw_p[(block, CONTRAST)] = float(summary["raw_p"])
        prepared.append((block, block_rows, deltas, summary))
    if set(raw_p) != set(MEDIAN_FAMILY):
        _fail("Holm input is not the four CoordMedian comparisons")
    adjusted = holm_four(raw_p)
    analyses = []
    for block, block_rows, deltas, summary in prepared:
        decision, direction = _decision(float(summary["mean"]), float(adjusted[(block, CONTRAST)]))
        analyses.append(
            {
                "block": block,
                "dataset": block_rows[0]["Dataset"],
                "model": block_rows[0]["Model"],
                "deltas": deltas,
                "summary": summary,
                "holm_p": float(adjusted[(block, CONTRAST)]),
                "decision": decision,
                "supported_direction": direction,
            }
        )
    return analyses


def _display_row(analysis: dict) -> tuple[str, str, str, str, str, str, str]:
    summary = analysis["summary"]
    signs = f"{summary['negative']}/{summary['zero']}/{summary['positive']}"
    interval = f"[{float(_pp(summary['ci95_low'])):.4f}, {float(_pp(summary['ci95_high'])):.4f}]"
    return (
        analysis["block"],
        f"{float(_pp(summary['mean'])):.4f}",
        interval,
        signs,
        f"{float(summary['raw_p']):.6g}",
        f"{float(analysis['holm_p']):.6g}",
        analysis["decision"],
    )


def _check_display(analyses: list[dict]) -> list[tuple[str, str, str, str, str, str, str]]:
    displayed = tuple(_display_row(analysis) for analysis in analyses)
    if displayed != MANUSCRIPT_DISPLAY:
        _fail(f"recomputed table does not match the drafted manuscript: {displayed}")
    for analysis in analyses:
        block = analysis["block"]
        summary = analysis["summary"]
        if _pp(summary["mean"]) != MEAN_PP_REPR[block]:
            _fail(f"{block} mean percentage-point repr changed")
        if (_pp(summary["ci95_low"]), _pp(summary["ci95_high"])) != CI_PP_REPR[block]:
            _fail(f"{block} interval repr changed")
        if repr(summary["t_statistic"]) != T_REPR[block]:
            _fail(f"{block} t statistic repr changed")
        if analysis["decision"] != "inconclusive" or analysis["supported_direction"]:
            _fail(f"{block} is not an inconclusive row")
    return list(displayed)


def _paired_csv_rows(analyses: list[dict]) -> list[dict]:
    rows = []
    for analysis in analyses:
        summary = analysis["summary"]
        rows.append(
            {
                "Block": analysis["block"],
                "Dataset": analysis["dataset"],
                "Model": analysis["model"],
                "Contrast": CONTRAST,
                "LeftArm": "FedTrimmedAvg",
                "RightArm": "CoordMedian",
                "SeedOrder": ",".join(str(seed) for seed in DECISION_SEEDS),
                "DeltaBySeedJSON": json.dumps(
                    {str(seed): delta for seed, delta in zip(DECISION_SEEDS, analysis["deltas"])},
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "n": summary["n"],
                "df": summary["df"],
                "Mean_pp": _pp(summary["mean"]),
                "SD_pp": _pp(summary["sd"]),
                "Median_pp": _pp(summary["median"]),
                "Min_pp": _pp(summary["minimum"]),
                "Max_pp": _pp(summary["maximum"]),
                "Negative": summary["negative"],
                "Zero": summary["zero"],
                "Positive": summary["positive"],
                "CI95Low_pp": _pp(summary["ci95_low"]),
                "CI95High_pp": _pp(summary["ci95_high"]),
                "TStatistic": repr(summary["t_statistic"]),
                "RawP": repr(summary["raw_p"]),
                "HolmP": repr(analysis["holm_p"]),
                "Decision": analysis["decision"],
                "SupportedDirection": analysis["supported_direction"],
            }
        )
    return rows


def _summary(pair_rows: list[dict], analyses: list[dict], progress_lines: int) -> dict:
    return {
        "study": "Q1-E5 CoordMedian-Adam versus sealed FedTrimmedAvg-Adam",
        "endpoint": "unrounded Final macro-F1 at round 30",
        "difference": "sealed FedTrimmedAvg-Adam minus CoordMedian-Adam",
        "cells": 28,
        "n": 7,
        "df": 6,
        "family_alpha": FAMILY_ALPHA,
        "family": [list(key) for key in MEDIAN_FAMILY],
        "holm_scope": "these four tests only; not Q1-E4 and not the historical eight-test family",
        "supported_blocks": [
            analysis["block"] for analysis in analyses if analysis["decision"] == "supported"
        ],
        "equivalence_procedure": False,
        "fednova": "deferred",
        "manuscript_freeze": False,
        "completion_signature": False,
        "first_started_utc": min(row["StartedUTC"] for row in pair_rows),
        "last_finished_utc": max(row["FinishedUTC"] for row in pair_rows),
        "progress_log_lines": progress_lines,
        "sealed_seed_csv_sha256": BOUND_SHA256["q1_e4_results/reporting/q1_e4_seed.csv"],
        "frozen_config_sha256": BOUND_SHA256["q1_e4_results/reporting/Q1_E4_FROZEN_CONFIG.json"],
        "accepted_preregistration_sha256": ACCEPTED_PREREG_SHA256,
        "cache_npz_rehashed": False,
        "comparisons": [
            {
                "block": analysis["block"],
                "dataset": analysis["dataset"],
                "model": analysis["model"],
                "contrast": CONTRAST,
                "mean": analysis["summary"]["mean"],
                "mean_pp": _pp(analysis["summary"]["mean"]),
                "ci95_low_pp": _pp(analysis["summary"]["ci95_low"]),
                "ci95_high_pp": _pp(analysis["summary"]["ci95_high"]),
                "negative": analysis["summary"]["negative"],
                "zero": analysis["summary"]["zero"],
                "positive": analysis["summary"]["positive"],
                "t_statistic": repr(analysis["summary"]["t_statistic"]),
                "raw_p": repr(analysis["summary"]["raw_p"]),
                "holm_p": repr(analysis["holm_p"]),
                "decision": analysis["decision"],
                "supported_direction": analysis["supported_direction"] or None,
            }
            for analysis in analyses
        ],
    }


def _packet_text(displayed: list[tuple[str, str, str, str, str, str, str]], pair_rows: list[dict]) -> str:
    lines = [
        "# Q1-E5 coordinate-wise median reporting packet",
        "",
        "**Study:** 28 CoordMedian-Adam cells against the sealed Q1-E4 FedTrimmedAvg-Adam rows.  ",
        "**Endpoint:** unrounded Final macro-F1 at round 30.  ",
        "**Difference:** sealed FedTrimmedAvg-Adam minus CoordMedian-Adam.  ",
        "**Family:** four paired tests, in block order T1, T2, T5, T6, at Holm family-wise level 0.05.  ",
        "**Seeds:** {127, 131, 137, 139, 149, 151, 157}. Each test uses n=7 and df=6.",
        "",
        "This packet recomputes those four tests from `q1_e5_results/coordmedian/cells/` and the sealed `q1_e4_results/reporting/q1_e4_seed.csv`. The CSV algorithm field is `FedTrimmedAvg`. The accepted comparator name is FedTrimmedAvg-Adam. The builder does not train. It does not add a completion signature, and it does not freeze the manuscript.",
        "",
        "## Prospective paired-t/Holm family",
        "",
        "Exactly four comparisons were evaluated. Percentage-point values are 100 times the unrounded mean of the seven Final macro-F1 differences, displayed to four decimal places. Intervals are the unadjusted 95% paired-t intervals. p-values use Python `.6g`. Signs are negative, tie, and positive counts, in that order. Holm adjustment uses these four rows only. No row is supported.",
        "",
        "| Block | Contrast | Mean pp | 95% CI pp | Signs - / 0 / + | Raw p | Holm p | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in displayed:
        lines.append(
            f"| {row[0]} | {CONTRAST} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} |"
        )
    lines.extend(
        [
            "",
            "A positive supported mean would say the trimmed mean is higher than the median in that block only. A negative supported mean would say the median is higher in that block only. Inconclusive is not equivalence, and no equivalence procedure was applied.",
            "",
            "## Claim boundary",
            "",
            "- These Holm p-values are the four-test family. They are not the historical eight-test family, and they are not combined with the Q1-E4 Holm result.",
            "- The Q1-E4 packet still contains the only supported comparison in that family: T2 FedAvg minus UnweightedFedAvg. This packet does not rewrite that row.",
            "- FedNova is deferred. It is not a row in this family and it is not a finding that FedNova is inferior.",
            "- One supported block would not authorize a general aggregator claim. No block is supported.",
            "- The clients are simulated dataset partitions. This packet does not authorize a universal, deployment, privacy, rare-class, or adversarial-robustness claim.",
            "",
            "## Seed-wise Final macro-F1",
            "",
            "CoordMedian values are the cell-file strings. FedTrimmedAvg values are the sealed CSV strings. Delta is FedTrimmedAvg minus CoordMedian, in Final macro-F1 units.",
            "",
            "| Block | Seed | CoordMedian | FedTrimmedAvg | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        lines.append(
            f"| {row['Block']} | {row['Seed']} | {row['CoordMedian_Final_F1']} | "
            f"{row['FedTrimmedAvg_Final_F1']} | {row['Delta']} |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "- Each of the 28 rows is status complete, finite, 30 rounds, mode `match_notebook`, optimizer `adam`, and learning rate `1e-3`.",
            "- Each history file matches the cell's `history_sha256`. Scored rounds are 1 through 30. Round 30 `macro_f1` and `accuracy` match `Final_F1` and `Final_Acc`.",
            "- Each saved decision-partition document matches the cell and `Q1_E4_FROZEN_CONFIG.json` on the scheme digest, client index digests, and client sizes.",
            "- The sealed comparator file hash is `8a9218a130486b68c69f3dd40cf5dd82a217add985c66a91af6fe3dd97da90c9`.",
            "- The frozen-config hash is `3e78215912fb7e1315c7a133cc964038387a88ee8d73577d14390cca6b28f9be`.",
            "- The accepted preregistration hash remains `cd20c7a61b3a41ad214cc51bf4221f7837604fd1236f75282fd07be346ec6517`. This builder does not edit that file.",
            "- Each cell's recorded cache SHA-256 equals the bound NPZ digest. This builder does not re-read the NPZ bytes.",
            "- Paired tests use `fl_local/e4_stats.py` `paired_summary`. Holm adjustment uses `fl_local/e5_aggregators.py` `holm_four`.",
            "- `SHA256SUMS.txt` seals this packet and the files it read. It does not replace the Q1-E4 manifest.",
            "",
        ]
    )
    return "\n".join(lines)


def _manifest_paths() -> list[Path]:
    paths = [
        SEED_CSV,
        PAIRED_CSV,
        SUMMARY_JSON,
        PACKET_MD,
        SEALED_SEED_CSV,
        FROZEN_CONFIG,
        PREREG,
        ROOT / "fl_local" / "e4_stats.py",
        ROOT / "fl_local" / "e5_aggregators.py",
        ROOT / "tools" / "build_q1_e5_packet.py",
        PROGRESS,
    ]
    paths.extend(
        path
        for path in sorted(CELLS.glob("T*_CoordMedian_s*.json"))
        if not path.name.endswith(".history.json")
    )
    paths.extend(sorted(CELLS.glob("T*_CoordMedian_s*.history.json")))
    for dataset, seed in (("CIC-IoT2023", item) for item in DECISION_SEEDS):
        _, relative = decision_partition_files(dataset, int(seed))
        paths.append(ROOT / relative)
    _, nbaiot = decision_partition_files("N-BaIoT")
    paths.append(ROOT / nbaiot)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        _fail(f"manifest inputs are missing: {missing}")
    if len(paths) != len(set(path.resolve() for path in paths)):
        _fail("manifest path list contains a duplicate")
    return paths


def _write_manifest(paths: list[Path]) -> None:
    lines = []
    for path in sorted(paths, key=lambda item: os.path.relpath(item, REPORTING).replace("\\", "/")):
        relative = os.path.relpath(path, REPORTING).replace("\\", "/")
        lines.append(f"{sha256_file(path)}  {relative}")
    atomic_write_bytes(MANIFEST, ("\n".join(lines) + "\n").encode("utf-8"))


def _check_progress() -> int:
    text = PROGRESS.read_text(encoding="utf-8")
    if "FAILED" in text or "E5 stopped" in text:
        _fail("coordmedian progress.log contains a failure line")
    return len(text.splitlines())


def main() -> int:
    if "torch" in sys.modules:
        _fail("the packet builder imported torch")
    prereg_hash = _bound_hash("q1_e4_results/reporting/q1_e4_seed.csv")
    frozen_hash = _bound_hash("q1_e4_results/reporting/Q1_E4_FROZEN_CONFIG.json")
    if sha256_file(PREREG) != ACCEPTED_PREREG_SHA256:
        _fail("accepted preregistration hash changed")
    frozen = _load_json(FROZEN_CONFIG)
    specs = _block_specs(frozen)
    cells = _load_cells(specs)
    for cell in cells:
        cell["_frozen"] = frozen
        _check_partition(cell)
    sealed = _load_sealed_pairs(specs)
    pair_rows = _pair_rows(cells, sealed, specs)
    analyses = _analyses(pair_rows)
    displayed = _check_display(analyses)
    progress_lines = _check_progress()
    protected = {
        PREREG: PREREG.read_bytes(),
        SEALED_SEED_CSV: SEALED_SEED_CSV.read_bytes(),
        FROZEN_CONFIG: FROZEN_CONFIG.read_bytes(),
        ROOT / "q1_e4_results" / "reporting" / "SHA256SUMS.txt": (
            ROOT / "q1_e4_results" / "reporting" / "SHA256SUMS.txt"
        ).read_bytes(),
    }
    REPORTING.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(SEED_CSV, PAIR_FIELDS, pair_rows)
    paired_rows = _paired_csv_rows(analyses)
    atomic_write_csv(PAIRED_CSV, PAIRED_FIELDS, paired_rows)
    atomic_write_json(SUMMARY_JSON, _summary(pair_rows, analyses, progress_lines))
    atomic_write_bytes(PACKET_MD, _packet_text(displayed, pair_rows).encode("utf-8"))
    _write_manifest(_manifest_paths())
    for path, payload in protected.items():
        if path.read_bytes() != payload:
            _fail(f"builder rewrote a protected file: {path}")
    if prereg_hash != BOUND_SHA256["q1_e4_results/reporting/q1_e4_seed.csv"]:
        _fail("sealed seed hash was not retained")
    if frozen_hash != BOUND_SHA256["q1_e4_results/reporting/Q1_E4_FROZEN_CONFIG.json"]:
        _fail("frozen config hash was not retained")
    written = PACKET_MD.read_text(encoding="utf-8")
    for row in displayed:
        needle = f"| {row[0]} | {CONTRAST} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} |"
        if needle not in written:
            _fail("packet markdown is missing a recomputed row")
    if "torch" in sys.modules:
        _fail("the packet builder imported torch")
    print(
        f"wrote {PACKET_MD.relative_to(ROOT).as_posix()} "
        f"cells=28 comparisons=4 supported=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

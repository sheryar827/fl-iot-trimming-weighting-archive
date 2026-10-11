"""Build the final Q1-E4 packet after all 112 frozen new cells complete.

This reporting-only command has no cache-building or training path.  It uses
the frozen seven-new-seed analysis implementation and keeps the ten-seed table
strictly descriptive.
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

import run_q1_e4_train_scaler as runner  # noqa: E402
from fl_local.e4_artifacts import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)
from fl_local.e4_stats import (  # noqa: E402
    ALL_SEEDS,
    ARM_ORDER,
    BLOCK_ORDER,
    COMPARISON_ORDER,
    CONTRASTS,
    NEW_SEEDS,
    combined_estimates,
    paired_differences,
    paired_summary,
    prospective_analysis,
)


REPORTING = ROOT / "q1_e4_results" / "reporting"
SEED_CSV = REPORTING / "q1_e4_seed.csv"
PROSPECTIVE_CSV = REPORTING / "q1_e4_prospective_paired.csv"
COMBINED_CSV = REPORTING / "q1_e4_combined_estimates.csv"
SUMMARY_JSON = REPORTING / "q1_e4_summary.json"
PACKET_MD = REPORTING / "Q1_E4_REPORTING_PACKET.md"
MANIFEST = REPORTING / "SHA256SUMS.txt"

SEED_FIELDS = [
    "Block",
    "Dataset",
    "Model",
    "Condition",
    "Algorithm",
    "Seed",
    "SeedRole",
    "Final_Acc",
    "Final_F1",
    "Best_Acc",
    "Best_F1",
    "AUC",
    "ScaleOn",
    "CacheKey",
    "PartitionSha256",
    "Source",
]

PROSPECTIVE_FIELDS = [
    "Block",
    "Dataset",
    "Model",
    "Contrast",
    "LeftArm",
    "RightArm",
    "DesiredDirection",
    "SeedOrder",
    "DeltaBySeedJSON",
    "n",
    "df",
    "Mean_pp",
    "SD_pp",
    "Median_pp",
    "Min_pp",
    "Max_pp",
    "Positive",
    "Zero",
    "Negative",
    "CI95Low_pp",
    "CI95High_pp",
    "TStatistic",
    "RawP",
    "HolmP",
    "Decision",
]

COMBINED_FIELDS = [
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
    "Positive",
    "Zero",
    "Negative",
    "CI95Low_pp",
    "CI95High_pp",
    "Label",
]

CONTRAST_BY_NAME = {
    name: {"left": left, "right": right, "desired": desired}
    for name, left, right, desired in CONTRASTS
}
BLOCK_META = {row["block"]: row for row in runner.BLOCKS}


def _read_csv(path: Path) -> tuple[list[str] | None, list[dict]]:
    if not path.is_file():
        return None, []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is non-finite: {value!r}")
    return number


def _seed_rows(
    anchors: dict[tuple[str, str, str], dict], raw_rows: list[dict]
) -> tuple[list[dict], dict[tuple[str, str, int], float]]:
    new_by_key = {
        (row["Block"], row["Algorithm"], str(int(float(row["Seed"])))): row
        for row in raw_rows
    }
    expected_new = {job.key for job in runner.all_jobs()}
    if set(new_by_key) != expected_new:
        raise RuntimeError(
            f"new result set is not exactly 112 rows; "
            f"missing={expected_new-set(new_by_key)} extra={set(new_by_key)-expected_new}"
        )
    output: list[dict] = []
    final_f1: dict[tuple[str, str, int], float] = {}
    for block in BLOCK_ORDER:
        meta = BLOCK_META[block]
        for seed in ALL_SEEDS:
            for algorithm in ARM_ORDER:
                key = (block, algorithm, str(seed))
                source = anchors[key] if seed in runner.EXISTING_SEEDS else new_by_key[key]
                role = "existing observed anchor" if seed in runner.EXISTING_SEEDS else "prospective new seed"
                row = {
                    "Block": block,
                    "Dataset": meta["dataset"],
                    "Model": meta["model"],
                    "Condition": meta["condition"],
                    "Algorithm": algorithm,
                    "Seed": seed,
                    "SeedRole": role,
                    "Final_Acc": repr(_finite(source["Final_Acc"], "Final_Acc")),
                    "Final_F1": repr(_finite(source["Final_F1"], "Final_F1")),
                    "Best_Acc": repr(_finite(source["Best_Acc"], "Best_Acc")),
                    "Best_F1": repr(_finite(source["Best_F1"], "Best_F1")),
                    "AUC": repr(_finite(source["AUC"], "AUC")),
                    "ScaleOn": source["ScaleOn"],
                    "CacheKey": source["CacheKey"],
                    "PartitionSha256": source["PartitionSha256"],
                    "Source": (
                        "Q1_E4_BASELINE_ANCHORS.csv"
                        if seed in runner.EXISTING_SEEDS
                        else "q1_e4_raw.csv"
                    ),
                }
                if row["ScaleOn"] != runner.SCALE_ON:
                    raise RuntimeError(f"non-train scaler row in packet input: {key}")
                final_f1[(block, algorithm, seed)] = float(row["Final_F1"])
                output.append(row)
    if len(output) != 160 or len(final_f1) != 160:
        raise RuntimeError("Q1-E4 seed table must contain exactly 160 unique rows")
    return output, final_f1


def _delta_json(seeds: tuple[int, ...], deltas: list[float]) -> str:
    if len(seeds) != len(deltas):
        raise RuntimeError("seed/delta length mismatch")
    return json.dumps(
        {str(seed): float(delta) for seed, delta in zip(seeds, deltas)},
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity(block: str, contrast: str) -> dict:
    meta = BLOCK_META[block]
    definition = CONTRAST_BY_NAME[contrast]
    return {
        "Block": block,
        "Dataset": meta["dataset"],
        "Model": meta["model"],
        "Contrast": contrast,
        "LeftArm": definition["left"],
        "RightArm": definition["right"],
    }


def _pp(value: object) -> str:
    return repr(100.0 * float(value))


def _prospective_rows(final_f1: dict[tuple[str, str, int], float]) -> list[dict]:
    values = {
        key: value for key, value in final_f1.items() if key[2] in set(NEW_SEEDS)
    }
    analyses = prospective_analysis(values)
    rows = []
    for analysis in analyses:
        row = {
            **_identity(analysis["block"], analysis["contrast"]),
            "DesiredDirection": analysis["desired_direction"],
            "SeedOrder": ",".join(str(seed) for seed in NEW_SEEDS),
            "DeltaBySeedJSON": _delta_json(NEW_SEEDS, analysis["deltas"]),
            "n": analysis["n"],
            "df": analysis["df"],
            "Mean_pp": _pp(analysis["mean"]),
            "SD_pp": _pp(analysis["sd"]),
            "Median_pp": _pp(analysis["median"]),
            "Min_pp": _pp(analysis["minimum"]),
            "Max_pp": _pp(analysis["maximum"]),
            "Positive": analysis["positive"],
            "Zero": analysis["zero"],
            "Negative": analysis["negative"],
            "CI95Low_pp": _pp(analysis["ci95_low"]),
            "CI95High_pp": _pp(analysis["ci95_high"]),
            "TStatistic": (
                "" if analysis["t_statistic"] is None else repr(analysis["t_statistic"])
            ),
            "RawP": repr(analysis["raw_p"]),
            "HolmP": repr(analysis["holm_p"]),
            "Decision": analysis["decision"],
        }
        rows.append(row)
    if [(row["Block"], row["Contrast"]) for row in rows] != list(COMPARISON_ORDER):
        raise RuntimeError("prospective table order differs from the frozen family")
    return rows


def _combined_rows(final_f1: dict[tuple[str, str, int], float]) -> list[dict]:
    analyses = combined_estimates(final_f1)
    rows = []
    forbidden = {"TStatistic", "RawP", "HolmP", "Decision"}
    for analysis in analyses:
        row = {
            **_identity(analysis["block"], analysis["contrast"]),
            "SeedOrder": ",".join(str(seed) for seed in ALL_SEEDS),
            "DeltaBySeedJSON": _delta_json(ALL_SEEDS, analysis["deltas"]),
            "n": analysis["n"],
            "df": analysis["df"],
            "Mean_pp": _pp(analysis["mean"]),
            "SD_pp": _pp(analysis["sd"]),
            "Median_pp": _pp(analysis["median"]),
            "Min_pp": _pp(analysis["minimum"]),
            "Max_pp": _pp(analysis["maximum"]),
            "Positive": analysis["positive"],
            "Zero": analysis["zero"],
            "Negative": analysis["negative"],
            "CI95Low_pp": _pp(analysis["ci95_low"]),
            "CI95High_pp": _pp(analysis["ci95_high"]),
            "Label": analysis["label"],
        }
        if forbidden.intersection(row):
            raise RuntimeError("combined table contains a forbidden decision-test field")
        rows.append(row)
    if [(row["Block"], row["Contrast"]) for row in rows] != list(COMPARISON_ORDER):
        raise RuntimeError("combined table order differs from the frozen family")
    return rows


def _block_decisions(prospective_rows: list[dict]) -> dict[str, dict]:
    by_key = {(row["Block"], row["Contrast"]): row for row in prospective_rows}
    output = {}
    for block in BLOCK_ORDER:
        trim = {
            name: by_key[(block, name)]["Decision"] for name in ("FT-UW", "WT-FA")
        }
        weight = {
            name: by_key[(block, name)]["Decision"] for name in ("FA-UW", "WT-FT")
        }
        output[block] = {
            "Htrim": {
                "decision": (
                    "supported"
                    if all(value == "supported" for value in trim.values())
                    else "not_supported"
                ),
                "components": trim,
            },
            "Hweight-hurt": {
                "decision": (
                    "supported"
                    if all(value == "supported" for value in weight.values())
                    else "not_supported"
                ),
                "components": weight,
            },
        }
    return output


def _interaction_rows(
    final_f1: dict[tuple[str, str, int], float], seeds: tuple[int, ...], label: str
) -> list[dict]:
    output = []
    for block in BLOCK_ORDER:
        deltas = [
            (
                final_f1[(block, "WeightedFedTrimmedAvg", seed)]
                - final_f1[(block, "FedAvg", seed)]
            )
            - (
                final_f1[(block, "FedTrimmedAvg", seed)]
                - final_f1[(block, "UnweightedFedAvg", seed)]
            )
            for seed in seeds
        ]
        summary = paired_summary(deltas, include_test=False)
        output.append(
            {
                "block": block,
                "label": label,
                "seed_order": list(seeds),
                "deltas": deltas,
                "mean_pp": 100.0 * summary["mean"],
                "sd_pp": 100.0 * summary["sd"],
                "ci95_low_pp": 100.0 * summary["ci95_low"],
                "ci95_high_pp": 100.0 * summary["ci95_high"],
                "positive": summary["positive"],
                "zero": summary["zero"],
                "negative": summary["negative"],
            }
        )
    return output


def _packet_text(
    prospective: list[dict],
    combined: list[dict],
    block_decisions: dict[str, dict],
    prospective_interaction: list[dict],
    combined_interaction: list[dict],
) -> str:
    lines = [
        "# Q1-E4 train-scaler seed-expansion reporting packet",
        "",
        "**Primary decision sample:** seven prospective seeds {127, 131, 137, 139, 149, 151, 157}.  ",
        "**Combined estimate:** ten seeds; data-informed descriptive estimate only.  ",
        "**Endpoint:** unrounded Final macro-F1 at round 30.  ",
        "",
        "## Prospective paired-t/Holm family",
        "",
        "Exactly 16 comparisons were evaluated. Percentage-point values below are Final macro-F1 differences. Holm adjustment is across all 16 comparisons.",
        "",
        "| Block | Contrast | Mean pp | 95% CI pp | Signs +/0/- | Raw p | Holm p | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in prospective:
        lines.append(
            f"| {row['Block']} | {row['Contrast']} | {float(row['Mean_pp']):.4f} | "
            f"[{float(row['CI95Low_pp']):.4f}, {float(row['CI95High_pp']):.4f}] | "
            f"{row['Positive']}/{row['Zero']}/{row['Negative']} | "
            f"{float(row['RawP']):.6g} | {float(row['HolmP']):.6g} | {row['Decision']} |"
        )
    lines.extend(
        [
            "",
            "## Block-level component rules",
            "",
            "| Block | Htrim | Hweight-hurt |",
            "|---|---|---|",
        ]
    )
    for block in BLOCK_ORDER:
        lines.append(
            f"| {block} | {block_decisions[block]['Htrim']['decision']} | "
            f"{block_decisions[block]['Hweight-hurt']['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Combined ten-seed estimates",
            "",
            "These rows are combined, data-informed estimates. They have no p-value or decision role and cannot override the seven-new-seed Holm decisions.",
            "",
            "| Block | Contrast | Mean pp | 95% CI pp | Signs +/0/- |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in combined:
        lines.append(
            f"| {row['Block']} | {row['Contrast']} | {float(row['Mean_pp']):.4f} | "
            f"[{float(row['CI95Low_pp']):.4f}, {float(row['CI95High_pp']):.4f}] | "
            f"{row['Positive']}/{row['Zero']}/{row['Negative']} |"
        )
    lines.extend(
        [
            "",
            "## Secondary descriptive interaction",
            "",
            "Interaction = `(WT-FA) - (FT-UW)`. It is excluded from the Holm family and has no decision role.",
            "",
            "| Sample | Block | Mean pp | 95% CI pp | Signs +/0/- |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for sample, rows in (
        ("7 prospective seeds", prospective_interaction),
        ("10 combined seeds", combined_interaction),
    ):
        for row in rows:
            lines.append(
                f"| {sample} | {row['block']} | {row['mean_pp']:.4f} | "
                f"[{row['ci95_low_pp']:.4f}, {row['ci95_high_pp']:.4f}] | "
                f"{row['positive']}/{row['zero']}/{row['negative']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Findings apply only to T1/T2/T5/T6 under the frozen train-scaler protocol.",
            "- `WT-UW` is not reported as a one-factor effect.",
            "- Inconclusive does not mean equivalent.",
            "- N-BaIoT is participating-device pooled-holdout evidence, not unseen-device or deployment validation.",
            "- No universal superiority, Byzantine robustness, or general rare-class-protection claim is authorized.",
            "- The manifest seals processed caches and experiment artifacts; it does not hash the original CIC/N-BaIoT source CSV trees.",
            "",
        ]
    )
    return "\n".join(lines)


def _manifest_paths(
    frozen: dict,
    raw_rows: list[dict],
    anchors: dict[tuple[str, str, str], dict],
) -> list[Path]:
    paths = {ROOT / relative for relative in frozen["file_sha256"]}
    paths.update(
        {
            runner.LIVE.frozen,
            runner.LIVE.anchors,
            runner.LIVE.preflight,
            runner.LIVE.raw,
            SEED_CSV,
            PROSPECTIVE_CSV,
            COMBINED_CSV,
            SUMMARY_JSON,
            PACKET_MD,
        }
    )
    for dataset in runner.CACHE_SPECS:
        paths.add(runner.LIVE.cache_npz(dataset))
        paths.add(runner.LIVE.cache_meta(dataset))
    for row in raw_rows:
        paths.add(ROOT / row["HistoryCSV"])
        paths.add(ROOT / row["CellMetaJSON"])
    for row in anchors.values():
        paths.add(ROOT / row["HistoryCSV"])
        paths.add(ROOT / row["SourceRaw"])
        paths.add(ROOT / row["SourceManifest"])
    failed_attempts = runner.LIVE.out / "failed_attempts"
    if failed_attempts.exists():
        paths.update(path for path in failed_attempts.rglob("*") if path.is_file())
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"manifest inputs are missing: {missing}")
    return sorted(path.resolve() for path in paths)


def _write_manifest(paths: list[Path]) -> None:
    lines = []
    for path in paths:
        relative = os.path.relpath(path, REPORTING).replace("\\", "/")
        lines.append(f"{sha256_file(path)}  {relative}")
    atomic_write_bytes(MANIFEST, ("\n".join(lines) + "\n").encode("utf-8"))


def main() -> int:
    if not runner.prereg_accepted() or not runner.contract_accepts_e4():
        raise RuntimeError("accepted E4 preregistration/contract anchors are missing")
    frozen = runner.enforce_frozen_files()
    runner.require_preflight(frozen)
    runner.verify_caches_and_partitions(frozen)
    runner.verify_anchors(frozen)
    done = runner.done_keys()
    expected = {job.key for job in runner.all_jobs()}
    if done != expected:
        raise RuntimeError(
            f"Q1-E4 is incomplete; expected 112/112 valid rows, found {len(done)}"
        )
    runner.detect_orphans()
    fields, raw_rows = _read_csv(runner.LIVE.raw)
    if fields != runner.FIELDNAMES or len(raw_rows) != 112:
        raise RuntimeError("Q1-E4 raw CSV is not the exact frozen 112-row table")
    anchors = runner.load_anchors()
    seed_rows, final_f1 = _seed_rows(anchors, raw_rows)
    prospective = _prospective_rows(final_f1)
    combined = _combined_rows(final_f1)
    block_decisions = _block_decisions(prospective)
    prospective_interaction = _interaction_rows(
        final_f1, NEW_SEEDS, "prospective-seed descriptive interaction"
    )
    combined_interaction = _interaction_rows(
        final_f1, ALL_SEEDS, "combined data-informed descriptive interaction"
    )
    summary = {
        "format": "q1-e4-reporting-summary-v1",
        "primary_endpoint": runner.ANALYSIS_SPEC["primary_endpoint"],
        "scope": {
            "anchor_rows": 48,
            "new_rows": 112,
            "combined_rows": 160,
            "prospective_comparisons": 16,
            "combined_comparisons": 16,
        },
        "prospective_seed_order": list(NEW_SEEDS),
        "combined_seed_order": list(ALL_SEEDS),
        "prospective_rows": prospective,
        "block_decisions": block_decisions,
        "secondary_interaction": {
            "definition": "(WT-FA) - (FT-UW); descriptive only; excluded from Holm",
            "prospective_seeds": prospective_interaction,
            "combined_seeds": combined_interaction,
        },
        "combined_decision_p_values": False,
        "manifest_scope": {
            "includes": "frozen code/config, preflight, anchors, raw/history/cell metadata, processed cache NPZ/meta, failed-attempt audit evidence if present, analysis tables, summary, and packet",
            "excludes": "original CIC-IoT2023 and N-BaIoT source CSV trees",
            "manifest_self_hash": False,
        },
    }
    atomic_write_csv(SEED_CSV, SEED_FIELDS, seed_rows)
    atomic_write_csv(PROSPECTIVE_CSV, PROSPECTIVE_FIELDS, prospective)
    atomic_write_csv(COMBINED_CSV, COMBINED_FIELDS, combined)
    atomic_write_json(SUMMARY_JSON, summary)
    atomic_write_bytes(
        PACKET_MD,
        _packet_text(
            prospective,
            combined,
            block_decisions,
            prospective_interaction,
            combined_interaction,
        ).encode("utf-8"),
    )
    manifest_paths = _manifest_paths(frozen, raw_rows, anchors)
    _write_manifest(manifest_paths)
    print("Q1-E4 packet built without training or cache writes")
    print("seed rows: 160")
    print("prospective comparisons: 16")
    print("combined descriptive comparisons: 16 (no p-values)")
    print(f"manifest entries: {len(manifest_paths)}")
    print(f"packet: {PACKET_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

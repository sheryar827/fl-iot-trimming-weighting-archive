#!/usr/bin/env python3
"""
Campaign status report (read-only harness).

Counts complete non-IID match_notebook cells for Q1-E1 / Q1-E1b from raw CSVs.
Writes a dated derived note (not binding claims). Does not edit the contract.

Usage (from Research Paper/):
  python3.12 -u tools/status_report.py
  python3.12 -u tools/status_report.py --arms e1b
  python3.12 -u tools/status_report.py --no-write

Exit codes:
  0  all selected arms complete (and no corrupt rows among expected cells)
  1  incomplete (missing cells)
  2  corrupt / inconsistent rows (e.g. missing history, primary mismatch)
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("MLP", "CNN")
ALPHAS = ("1.0", "0.5", "0.1")
SEEDS = ("42", "123", "7")
MODE = "match_notebook"

CACHE_PIN = {
    "N-BaIoT": "a9dd0c758635aebe",
    "CIC-IoT2023": "b725017696446067",
}

ARMS = {
    "e1": {
        "label": "Q1-E1 UnweightedFedAvg (non-IID)",
        "algorithm": "UnweightedFedAvg",
        "expected": 36,
        "blocks": [
            {
                "dataset": "N-BaIoT",
                "raw": ROOT / "q1_e1_results" / "csv" / "q1_e1_nbaiot_raw.csv",
                "csv_dir": ROOT / "q1_e1_results" / "csv",
            },
            {
                "dataset": "CIC-IoT2023",
                "raw": ROOT / "q1_e1_results" / "csv" / "q1_e1_cic_raw.csv",
                "csv_dir": ROOT / "q1_e1_results" / "csv",
            },
        ],
    },
    "e1b": {
        "label": "Q1-E1b WeightedFedTrimmedAvg (non-IID)",
        "algorithm": "WeightedFedTrimmedAvg",
        "expected": 36,
        "blocks": [
            {
                "dataset": "N-BaIoT",
                "raw": ROOT / "q1_e1b_results" / "csv" / "q1_e1b_nbaiot_raw.csv",
                "csv_dir": ROOT / "q1_e1b_results" / "csv",
            },
            {
                "dataset": "CIC-IoT2023",
                "raw": ROOT / "q1_e1b_results" / "csv" / "q1_e1b_cic_raw.csv",
                "csv_dir": ROOT / "q1_e1b_results" / "csv",
            },
        ],
    },
}


def _grid() -> list[tuple[str, str, str]]:
    return [(m, a, s) for m in MODELS for a in ALPHAS for s in SEEDS]


def _complete_primary(row: dict) -> bool:
    for c in ("Final_Acc", "Final_F1", "AUC"):
        v = str(row.get(c, "")).strip()
        if not v:
            return False
        try:
            float(v)
        except ValueError:
            return False
    return True


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _history_ok(csv_dir: Path, hist_name: str) -> tuple[bool, str]:
    if not hist_name or not str(hist_name).strip():
        return False, "missing HistoryCSV field"
    p = csv_dir / hist_name
    if not p.exists():
        return False, f"history file missing: {hist_name}"
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 30:
        return False, f"history rounds={len(rows)} (want 30): {hist_name}"
    try:
        f1s = [float(r["f1"]) for r in rows]
        accs = [float(r["accuracy"]) for r in rows]
    except (KeyError, ValueError) as e:
        return False, f"history parse error {hist_name}: {e}"
    return True, ""


def scan_arm(arm_key: str) -> dict:
    cfg = ARMS[arm_key]
    algo = cfg["algorithm"]
    expected_cells: list[tuple[str, str, str, str]] = []
    for block in cfg["blocks"]:
        for m, a, s in _grid():
            expected_cells.append((block["dataset"], m, a, s))

    complete: dict[tuple[str, str, str, str], dict] = {}
    incomplete: list[str] = []
    corrupt: list[str] = []
    cache_issues: list[str] = []

    for block in cfg["blocks"]:
        ds = block["dataset"]
        pin = CACHE_PIN[ds]
        rows = _load_rows(block["raw"])
        seen_keys: set[tuple[str, str, str]] = set()
        for r in rows:
            if r.get("Algorithm") != algo:
                continue
            if str(r.get("Mode", "")) != MODE:
                continue
            alpha = str(r.get("Alpha", "")).strip()
            if alpha not in ALPHAS:
                continue
            model = str(r.get("Model", "")).strip()
            seed = str(r.get("Seed", "")).strip()
            key3 = (model, alpha, seed)
            cell = (ds, model, alpha, seed)
            label = f"{ds} {model} a{alpha} s{seed}"

            if key3 in seen_keys:
                corrupt.append(f"{label}: duplicate raw row")
            seen_keys.add(key3)

            if not _complete_primary(r):
                incomplete.append(f"{label}: incomplete Final_Acc/Final_F1/AUC")
                continue

            ck = str(r.get("CacheKey", "")).strip()
            if ck and ck != pin:
                cache_issues.append(f"{label}: CacheKey {ck} != pin {pin}")

            ok_h, msg_h = _history_ok(block["csv_dir"], str(r.get("HistoryCSV", "")))
            if not ok_h:
                corrupt.append(f"{label}: {msg_h}")
                continue

            # primary vs history endpoints
            try:
                with (block["csv_dir"] / r["HistoryCSV"]).open(newline="", encoding="utf-8") as f:
                    hist = list(csv.DictReader(f))
                accs = [float(h["accuracy"]) for h in hist]
                f1s = [float(h["f1"]) for h in hist]
                auc = sum(f1s) / len(f1s)
                if abs(float(r["Final_Acc"]) - accs[-1]) > 1e-9:
                    corrupt.append(f"{label}: Final_Acc raw!=history")
                    continue
                if abs(float(r["Final_F1"]) - f1s[-1]) > 1e-9:
                    corrupt.append(f"{label}: Final_F1 raw!=history")
                    continue
                if abs(float(r["AUC"]) - auc) > 1e-9:
                    corrupt.append(f"{label}: AUC raw!=mean history F1")
                    continue
            except Exception as e:  # noqa: BLE001 — report and continue
                corrupt.append(f"{label}: history check failed: {e}")
                continue

            if cell in complete:
                corrupt.append(f"{label}: duplicate complete cell")
            complete[cell] = r

    missing = []
    for cell in expected_cells:
        if cell not in complete:
            missing.append(f"{cell[0]} {cell[1]} a{cell[2]} s{cell[3]}")

    n_complete = len(complete)
    expected = cfg["expected"]
    return {
        "arm": arm_key,
        "label": cfg["label"],
        "algorithm": algo,
        "expected": expected,
        "complete": n_complete,
        "missing": missing,
        "incomplete_rows": incomplete,
        "corrupt": corrupt,
        "cache_issues": cache_issues,
        "ok_complete": n_complete == expected and not missing,
        "ok_clean": not corrupt and not incomplete and not cache_issues,
    }


def format_report(results: list[dict], generated: str) -> str:
    lines = [
        "# Campaign status (derived — not a claim freeze)",
        "",
        f"Generated UTC: {generated}",
        f"Root: `{ROOT.as_posix()}`",
        "",
        "Binding claims: `Q1_CONTRIBUTION_CONTRACT.md`. Map: `thesis-context.md`.",
        "This file is regenerated by `tools/status_report.py` from raw CSVs.",
        "",
    ]
    for r in results:
        lines.append(f"## {r['label']}")
        lines.append("")
        lines.append(f"- **Algorithm:** `{r['algorithm']}`")
        lines.append(f"- **Progress:** **{r['complete']} / {r['expected']}** complete non-IID `{MODE}` cells")
        status = "COMPLETE" if r["ok_complete"] and r["ok_clean"] else (
            "COMPLETE_WITH_ISSUES" if r["ok_complete"] else "INCOMPLETE"
        )
        lines.append(f"- **Status:** {status}")
        if r["missing"]:
            lines.append(f"- **Missing ({len(r['missing'])}):**")
            for m in r["missing"]:
                lines.append(f"  - {m}")
        else:
            lines.append("- **Missing:** none")
        if r["incomplete_rows"]:
            lines.append(f"- **Incomplete raw rows ({len(r['incomplete_rows'])}):**")
            for m in r["incomplete_rows"][:20]:
                lines.append(f"  - {m}")
        if r["corrupt"]:
            lines.append(f"- **Corrupt / mismatch ({len(r['corrupt'])}):**")
            for m in r["corrupt"][:20]:
                lines.append(f"  - {m}")
        if r["cache_issues"]:
            lines.append(f"- **Cache pin issues ({len(r['cache_issues'])}):**")
            for m in r["cache_issues"][:20]:
                lines.append(f"  - {m}")
        lines.append("")
    lines.append("## Exit meaning")
    lines.append("")
    lines.append("- `0` — selected arms complete and clean")
    lines.append("- `1` — incomplete (missing cells)")
    lines.append("- `2` — corrupt / primary–history / cache issues")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Q1-E1 / Q1-E1b campaign status from raw CSVs")
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["e1", "e1b"],
        choices=sorted(ARMS.keys()),
        help="Which campaigns to scan (default: e1 e1b)",
    )
    ap.add_argument(
        "--no-write",
        action="store_true",
        help="Print only; do not write status_latest.md",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path (default: q1_e1b_results/status_latest.md if e1b selected else q1_e1_results/status_latest.md)",
    )
    args = ap.parse_args()

    generated = datetime.now(timezone.utc).isoformat()
    results = [scan_arm(a) for a in args.arms]
    report = format_report(results, generated)
    print(report)

    if not args.no_write:
        if args.out is not None:
            out = args.out if args.out.is_absolute() else ROOT / args.out
        elif "e1b" in args.arms:
            out = ROOT / "q1_e1b_results" / "status_latest.md"
        else:
            out = ROOT / "q1_e1_results" / "status_latest.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Wrote {out.relative_to(ROOT)}")

    any_corrupt = any((r["corrupt"] or r["incomplete_rows"] or r["cache_issues"]) for r in results)
    any_missing = any(not r["ok_complete"] for r in results)
    if any_corrupt:
        return 2
    if any_missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

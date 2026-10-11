#!/usr/bin/env python3
"""
Strong-bar verification for N-BaIoT MLP FedAvg α=1.0 s42 match_notebook gate.

Compares probe_results/probe_fast_match_notebook.json to:
  - published constants (all_runs_raw.csv)
  - optional pre-engine backup JSON (30-round history)

Does NOT run training. You run the probe first:
  python3.12 -u run_probe_fast.py --mode match_notebook

Usage (from Research Paper/):
  python3.12 -u tools/verify_nbaiot_mlp_fedavg_gate.py
  python3.12 -u tools/verify_nbaiot_mlp_fedavg_gate.py --probe probe_results/probe_fast_match_notebook.json

Exit:
  0 = strong PASS (endpoints + AUC; history bit-identical to published if available)
  1 = FAIL endpoints / structure
  2 = probe file missing or unreadable
  3 = endpoints PASS but history not bit-identical (lineage weak)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Published N-BaIoT MLP FedAvg α=1.0 s42 (all_runs_raw + history)
PUB_FINAL_F1 = 0.8762425170224654
PUB_BEST_F1 = 0.8762425170224654
PUB_FINAL_ACC = 0.8956925949640226
PUB_BEST_ACC = 0.8956925949640226
PUB_AUC = 0.8579311208165772  # mean of 30 published f1

DEFAULT_PROBE = ROOT / "probe_results" / "probe_fast_match_notebook.json"
DEFAULT_BACKUP = ROOT / "probe_results" / "probe_fast_match_notebook_PRE_ENGINE_2026-08-02.json"
PUB_HISTORY = (
    ROOT
    / "fl-iot-botnet-nbaiot-cic-iot2023-main"
    / "fl_nbaiot_results"
    / "csv"
    / "history_MLP_FedAvg_a1.0_s42.csv"
)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _load_pub_history(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 30:
        raise ValueError(f"published history expected 30 rounds, got {len(rows)}")
    return rows


def _hist_from_probe(history: list) -> list[dict]:
    """Normalize probe history list to dicts with f1/accuracy."""
    out = []
    for h in history:
        if isinstance(h, dict):
            out.append(
                {
                    "f1": float(h.get("f1", h.get("Final_F1", "nan"))),
                    "accuracy": float(h.get("accuracy", h.get("acc", "nan"))),
                }
            )
        else:
            raise ValueError("history entries must be objects")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    ap.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    ap.add_argument("--published-history", type=Path, default=PUB_HISTORY)
    ap.add_argument(
        "--endpoint-tol",
        type=float,
        default=0.0,
        help="Absolute tolerance for Final/Best Acc/F1 (default 0 = exact)",
    )
    args = ap.parse_args()

    probe_path = args.probe if args.probe.is_absolute() else ROOT / args.probe
    if not probe_path.exists():
        print(f"FAIL: probe JSON missing: {probe_path}")
        print("Run first: python3.12 -u run_probe_fast.py --mode match_notebook")
        return 2

    try:
        report = json.loads(probe_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot read probe JSON: {e}")
        return 2

    res = report.get("results") or {}
    hist_raw = report.get("history") or []
    env = report.get("environment") or {}

    print("=" * 70)
    print("N-BaIoT MLP FedAvg a1.0 s42 — strong gate verify")
    print("=" * 70)
    print(f"probe: {probe_path}")
    print(f"device: {env.get('device')}  gpu: {env.get('gpu')}  torch: {env.get('torch')}")

    # --- endpoints ---
    checks = []
    try:
        final_f1 = float(res["final_f1"])
        best_f1 = float(res["best_f1"])
        final_acc = float(res["final_acc"])
        # best_acc may be absent in older JSON
        best_acc = float(res["best_acc"]) if "best_acc" in res else None
    except (KeyError, TypeError, ValueError) as e:
        print(f"FAIL: results block incomplete: {e}")
        return 1

    def near(a: float, b: float, tol: float) -> bool:
        return abs(a - b) <= tol

    tol = args.endpoint_tol
    checks.append(("Final_F1", final_f1, PUB_FINAL_F1, near(final_f1, PUB_FINAL_F1, tol)))
    checks.append(("Best_F1", best_f1, PUB_BEST_F1, near(best_f1, PUB_BEST_F1, tol)))
    checks.append(("Final_Acc", final_acc, PUB_FINAL_ACC, near(final_acc, PUB_FINAL_ACC, tol)))
    if best_acc is not None:
        checks.append(("Best_Acc", best_acc, PUB_BEST_ACC, near(best_acc, PUB_BEST_ACC, tol)))

    # Best equals Final on this cell
    checks.append(("Best_F1==Final_F1", best_f1, final_f1, near(best_f1, final_f1, tol)))
    if best_acc is not None:
        checks.append(("Best_Acc==Final_Acc", best_acc, final_acc, near(best_acc, final_acc, tol)))

    hist_ok = True
    hist = []
    try:
        hist = _hist_from_probe(hist_raw)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: history parse: {e}")
        return 1

    if len(hist) != 30:
        print(f"FAIL: history rounds={len(hist)} (want 30)")
        hist_ok = False
    else:
        f1s = [h["f1"] for h in hist]
        accs = [h["accuracy"] for h in hist]
        auc = _mean(f1s)
        # AUC vs published mean
        auc_ok = abs(auc - PUB_AUC) <= max(tol, abs(PUB_AUC) * 2e-16 * 4 + 1e-15)
        checks.append(("AUC_mean_F1", auc, PUB_AUC, auc_ok or near(auc, PUB_AUC, 1e-12)))
        # endpoints from history
        checks.append(("hist Final_F1", f1s[-1], final_f1, near(f1s[-1], final_f1, 1e-15)))
        checks.append(("hist Final_Acc", accs[-1], final_acc, near(accs[-1], final_acc, 1e-15)))
        checks.append(("hist Best_F1", max(f1s), best_f1, near(max(f1s), best_f1, 1e-15)))
        if best_acc is not None:
            checks.append(("hist Best_Acc", max(accs), best_acc, near(max(accs), best_acc, 1e-15)))

    print("\nEndpoint checks:")
    endpoint_pass = True
    for name, got, exp, ok in checks:
        d = abs(got - exp) if isinstance(got, float) else float("nan")
        flag = "OK" if ok else "FAIL"
        if not ok:
            endpoint_pass = False
        print(f"  {flag:4} {name:22} got={got:.15g}  exp={exp:.15g}  |Δ|={d:.6e}")

    if not hist_ok or not endpoint_pass:
        print("\nVERDICT: FAIL (endpoints / structure)")
        return 1

    # --- history vs published ---
    lineage_pass = True
    pub_path = args.published_history if args.published_history.is_absolute() else ROOT / args.published_history
    if pub_path.exists():
        pub = _load_pub_history(pub_path)
        max_df1 = 0.0
        max_dacc = 0.0
        for i, (h, p) in enumerate(zip(hist, pub), start=1):
            pf1 = float(p["f1"])
            pacc = float(p["accuracy"])
            max_df1 = max(max_df1, abs(h["f1"] - pf1))
            max_dacc = max(max_dacc, abs(h["accuracy"] - pacc))
        print(f"\nHistory vs published ({pub_path.name}):")
        print(f"  max|Δ f1|  = {max_df1:.6e}")
        print(f"  max|Δ acc| = {max_dacc:.6e}")
        if max_df1 != 0.0 or max_dacc != 0.0:
            lineage_pass = False
            print("  FAIL: not bit-identical to published history")
        else:
            print("  OK: bit-identical to published (30 rounds)")
    else:
        print(f"\nWARN: published history missing: {pub_path}")
        lineage_pass = False

    # --- vs pre-engine backup ---
    backup = args.backup if args.backup.is_absolute() else ROOT / args.backup
    if backup.exists():
        try:
            old = json.loads(backup.read_text(encoding="utf-8"))
            old_h = _hist_from_probe(old.get("history") or [])
            if len(old_h) == 30 and len(hist) == 30:
                max_df1 = max(abs(a["f1"] - b["f1"]) for a, b in zip(hist, old_h))
                max_dacc = max(abs(a["accuracy"] - b["accuracy"]) for a, b in zip(hist, old_h))
                print(f"\nHistory vs backup ({backup.name}):")
                print(f"  max|Δ f1|  = {max_df1:.6e}")
                print(f"  max|Δ acc| = {max_dacc:.6e}")
                if max_df1 != 0.0 or max_dacc != 0.0:
                    # first diverge round
                    for i, (a, b) in enumerate(zip(hist, old_h), start=1):
                        if a["f1"] != b["f1"] or a["accuracy"] != b["accuracy"]:
                            print(f"  first diverge round={i} f1 {b['f1']:.10f}->{a['f1']:.10f}")
                            break
                    print("  NOTE: backup is pre-2026-08-12 engine; diverge helps diagnose FAIL")
                else:
                    print("  OK: bit-identical to pre-engine backup")
            else:
                print(f"\nWARN: backup history length {len(old_h)}")
        except Exception as e:  # noqa: BLE001
            print(f"\nWARN: backup compare failed: {e}")
    else:
        print(f"\nWARN: backup missing: {backup}")
        print("  Create with: Copy-Item probe_results\\probe_fast_match_notebook.json "
              "probe_results\\probe_fast_match_notebook_PRE_ENGINE_2026-08-02.json")

    # runner loose flag
    loose = abs(final_f1 - PUB_FINAL_F1) < 0.005
    tight = abs(final_f1 - PUB_FINAL_F1) < 1e-4
    print(f"\nRunner-style: loose_0.5pp={loose}  tight_1e-4={tight}")
    print("Strong bar requires exact endpoints + bit-identical 30-round history.")

    if lineage_pass:
        print("\nVERDICT: STRONG PASS — continuous lineage OK for shared match_notebook path + FedAvg agg")
        print("Scope: does NOT re-validate FedTrimmedAvg / UnweightedFedAvg aggregators.")
        return 0

    print("\nVERDICT: ENDPOINTS PASS but HISTORY not bit-identical — lineage WEAK (exit 3)")
    return 3


if __name__ == "__main__":
    sys.exit(main())

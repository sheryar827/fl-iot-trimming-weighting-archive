#!/usr/bin/env python3
"""
Verify Q1-E3 gates G1–G3 against prereg targets (r3 + pre-G1 fixes).

Usage (from Research Paper/):
  python3.12 -u tools/verify_q1_e3_gates.py --gate G1
  python3.12 -u tools/verify_q1_e3_gates.py --gate G2
  python3.12 -u tools/verify_q1_e3_gates.py --gate G3

Exit:
  0 = PASS
  1 = FAIL
  2 = missing gate JSON / history / prerequisites
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "q1_e3_results" / "gates"
PUB_FA_HIST = (
    ROOT
    / "fl-iot-botnet-nbaiot-cic-iot2023-main"
    / "fl_nbaiot_results"
    / "csv"
    / "history_MLP_FedAvg_a1.0_s42.csv"
)
PUB_PROX_HIST = (
    ROOT
    / "fl-iot-botnet-nbaiot-cic-iot2023-main"
    / "fl_nbaiot_results"
    / "csv"
    / "history_MLP_FedProx_a1.0_s42.csv"
)

G1_FINAL_F1 = 0.8762425170224654
G1_FINAL_ACC = 0.8956925949640226
G3_FINAL_F1 = 0.8688510782827605
G3_FINAL_ACC = 0.8888074036256555
G3_BEST_F1 = 0.8747881297547481

STEM = {
    "G1": "gate_G1_FedAvg_Adam_a1.0_s42_match_notebook",
    "G2": "gate_G2_FedProx_mu0_Adam_a1.0_s42_match_notebook",
    "G3": "gate_G3_FedProx_mu001_Adam_a1.0_s42_match_notebook",
}


def _eq(a: float, b: float) -> bool:
    return a == b  # exact bit match for lineage


def _load_hist_csv(path: Path) -> tuple[list[float], list[float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    keys = {k.lower(): k for k in rows[0]}
    fk, ak, rk = keys["f1"], keys["accuracy"], keys["round"]
    by_f1 = {int(float(r[rk])): float(r[fk]) for r in rows}
    by_acc = {int(float(r[rk])): float(r[ak]) for r in rows}
    return [by_f1[i] for i in range(1, 31)], [by_acc[i] for i in range(1, 31)]


def _hist_from_report(report: dict) -> tuple[list[float], list[float]]:
    h = report["history"]
    f1s = [float(x["f1"]) for x in h]
    accs = [float(x["accuracy"]) for x in h]
    return f1s, accs


def _check_cfg(report: dict, *, algorithm: str, optimizer: str, fedprox_mu: float) -> bool:
    ok = True
    got_alg = str(report.get("algorithm", ""))
    got_opt = str(report.get("optimizer", "")).lower()
    got_mu = float(report.get("fedprox_mu"))
    if got_alg != algorithm:
        print(f"  FAIL algorithm: got {got_alg!r} want {algorithm!r}")
        ok = False
    else:
        print(f"  PASS algorithm={algorithm}")
    if got_opt != optimizer.lower():
        print(f"  FAIL optimizer: got {got_opt!r} want {optimizer!r}")
        ok = False
    else:
        print(f"  PASS optimizer={optimizer}")
    if not _eq(got_mu, float(fedprox_mu)):
        print(f"  FAIL fedprox_mu: got {got_mu} want {fedprox_mu}")
        ok = False
    else:
        print(f"  PASS fedprox_mu={fedprox_mu}")
    mode = str(report.get("mode", ""))
    if mode != "match_notebook":
        print(f"  FAIL mode: got {mode!r} want match_notebook")
        ok = False
    else:
        print("  PASS mode=match_notebook")
    return ok


def _count_match(a: list[float], b: list[float]) -> int:
    return sum(1 for x, y in zip(a, b) if _eq(x, y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["G1", "G2", "G3", "g1", "g2", "g3"])
    args = ap.parse_args()
    gate = args.gate.upper()
    stem = STEM[gate]
    jpath = GATES / f"{stem}.json"
    if not jpath.exists():
        print(f"FAIL: missing {jpath}")
        print(f"Run: python3.12 -u run_q1_e3_gates.py --gate {gate}")
        return 2

    report = json.loads(jpath.read_text(encoding="utf-8"))
    f1s, accs = _hist_from_report(report)
    if len(f1s) != 30 or len(accs) != 30:
        print(f"FAIL: history len f1={len(f1s)} acc={len(accs)} (want 30)")
        return 1

    print(f"VERIFY {gate}")
    print(f"  report: {jpath.relative_to(ROOT)}")
    print(f"  Final_F1={report['final_f1']:.16f}")
    print(f"  Final_Acc={report['final_acc']:.16f}")
    print(f"  Best_F1={report['best_f1']:.16f}")

    ok = True

    if gate == "G1":
        ok = (
            _check_cfg(
                report, algorithm="FedAvg", optimizer="adam", fedprox_mu=0.0
            )
            and ok
        )
        pub_f1, pub_acc = _load_hist_csv(PUB_FA_HIST)
        if not _eq(float(report["final_f1"]), G1_FINAL_F1):
            print(f"  FAIL Final_F1: got {report['final_f1']} want {G1_FINAL_F1}")
            ok = False
        else:
            print("  PASS Final_F1 exact")
        if not _eq(float(report["final_acc"]), G1_FINAL_ACC):
            print(f"  FAIL Final_Acc: got {report['final_acc']} want {G1_FINAL_ACC}")
            ok = False
        else:
            print("  PASS Final_Acc exact")
        n_f1 = _count_match(f1s, pub_f1)
        n_acc = _count_match(accs, pub_acc)
        if n_f1 != 30:
            print(f"  FAIL F1 history vs published: {n_f1}/30 bit-identical")
            ok = False
        else:
            print("  PASS 30/30 F1 bit-identical to published FedAvg")
        if n_acc != 30:
            print(f"  FAIL Acc history vs published: {n_acc}/30 bit-identical")
            ok = False
        else:
            print("  PASS 30/30 Acc bit-identical to published FedAvg")
        pre = (
            ROOT
            / "probe_results"
            / "probe_fast_match_notebook_PRE_E3ENGINE_2026-08-15.json"
        )
        if pre.exists():
            pre_r = json.loads(pre.read_text(encoding="utf-8"))
            ph = pre_r.get("history") or pre_r.get("result", {}).get("history")
            if ph and len(ph) == 30:
                pf = [float(x["f1"]) for x in ph]
                pa = [float(x["accuracy"]) for x in ph]
                n2 = _count_match(f1s, pf)
                n2a = _count_match(accs, pa)
                print(f"  vs PRE_E3ENGINE probe: F1 {n2}/30 Acc {n2a}/30")
                if n2 != 30 or n2a != 30:
                    print("  FAIL history vs PRE_E3ENGINE backup")
                    ok = False
            else:
                print("  NOTE: PRE_E3ENGINE probe history not comparable shape")

    elif gate == "G2":
        ok = (
            _check_cfg(
                report, algorithm="FedProx", optimizer="adam", fedprox_mu=0.0
            )
            and ok
        )
        g1_path = GATES / f"{STEM['G1']}.json"
        if not g1_path.exists():
            print(f"FAIL: G2 requires G1 report first: {g1_path}")
            return 2
        g1 = json.loads(g1_path.read_text(encoding="utf-8"))
        g1_f1s, g1_accs = _hist_from_report(g1)
        n_f1 = _count_match(f1s, g1_f1s)
        n_acc = _count_match(accs, g1_accs)
        if n_f1 != 30:
            print(f"  FAIL G2 vs G1 F1 history: {n_f1}/30 bit-identical")
            ok = False
        else:
            print("  PASS 30/30 F1 bit-identical to G1 (µ=0 harness)")
        if n_acc != 30:
            print(f"  FAIL G2 vs G1 Acc history: {n_acc}/30 bit-identical")
            ok = False
        else:
            print("  PASS 30/30 Acc bit-identical to G1 (µ=0 harness)")
        if not _eq(float(report["final_f1"]), float(g1["final_f1"])):
            print("  FAIL Final_F1 != G1")
            ok = False
        else:
            print("  PASS Final_F1 == G1")
        if not _eq(float(report["final_acc"]), float(g1["final_acc"])):
            print("  FAIL Final_Acc != G1")
            ok = False
        else:
            print("  PASS Final_Acc == G1")

    else:  # G3
        ok = (
            _check_cfg(
                report, algorithm="FedProx", optimizer="adam", fedprox_mu=0.01
            )
            and ok
        )
        if not _eq(float(report["final_f1"]), G3_FINAL_F1):
            print(f"  FAIL Final_F1: got {report['final_f1']} want {G3_FINAL_F1}")
            ok = False
        else:
            print("  PASS Final_F1 exact")
        if not _eq(float(report["final_acc"]), G3_FINAL_ACC):
            print(f"  FAIL Final_Acc: got {report['final_acc']} want {G3_FINAL_ACC}")
            ok = False
        else:
            print("  PASS Final_Acc exact")
        if not _eq(float(report["best_f1"]), G3_BEST_F1):
            print(f"  FAIL Best_F1: got {report['best_f1']} want {G3_BEST_F1}")
            ok = False
        else:
            print("  PASS Best_F1 exact")
        # Mandatory 30/30 vs published FedProx (prereg r3+ pre-G1 fix)
        if not PUB_PROX_HIST.exists():
            print(f"  FAIL published FedProx history missing: {PUB_PROX_HIST}")
            ok = False
        else:
            pub_f1, pub_acc = _load_hist_csv(PUB_PROX_HIST)
            n_f1 = _count_match(f1s, pub_f1)
            n_acc = _count_match(accs, pub_acc)
            if n_f1 != 30:
                print(f"  FAIL F1 history vs published FedProx: {n_f1}/30")
                ok = False
            else:
                print("  PASS 30/30 F1 bit-identical to published FedProx")
            if n_acc != 30:
                print(f"  FAIL Acc history vs published FedProx: {n_acc}/30")
                ok = False
            else:
                print("  PASS 30/30 Acc bit-identical to published FedProx")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify E2 raw/history/provenance and score HE2 2x2 (descriptive n=3)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "q1_e2_results" / "csv" / "q1_e2_raw.csv"
PROV = ROOT / "q1_e2_results" / "reporting" / "Q1_E2_CACHE_PROVENANCE.json"
ALGOS = (
    "FedAvg",
    "UnweightedFedAvg",
    "FedTrimmedAvg",
    "WeightedFedTrimmedAvg",
)
SEEDS = ("42", "123", "7")
MODELS = ("MLP", "CNN")


def _exact_int(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    n = math.trunc(x)
    return int(n) if x == n else None


def _hist_ok(path: Path, final_f1: float) -> tuple[bool, str, float | None]:
    if not path.is_file():
        return False, "missing history", None
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 30:
        return False, f"n={len(rows)}", None
    rnds, f1s = [], []
    for r in rows:
        n = _exact_int(r.get("round"))
        if n is None:
            return False, "bad round", None
        rnds.append(n)
        v = float(r.get("f1", ""))
        if not math.isfinite(v):
            return False, "bad f1", None
        f1s.append(v)
    if rnds != list(range(1, 31)):
        return False, "rounds", None
    if f1s[-1] != final_f1:
        return False, f"f1 mismatch {f1s[-1]} vs {final_f1}", None
    return True, "ok", sum(f1s) / len(f1s)


def main() -> int:
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    with RAW.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cells: dict[tuple[str, str, str], float] = {}
    issues: list[str] = []
    for r in rows:
        f1 = float(r["Final_F1"])
        key = (r["Model"], r["Algorithm"], str(int(float(r["Seed"]))))
        if key in cells:
            issues.append(f"dup {key}")
        cells[key] = f1
        if r["Mode"] != "match_notebook":
            issues.append(f"{key} mode")
        if r["ScaleOn"] != "train":
            issues.append(f"{key} scale")
        if r["Partition"] != "device":
            issues.append(f"{key} partition")
        if r["CacheKey"] != "4ddb8b93b8821650":
            issues.append(f"{key} key")
        if r["NpzSha256"].lower() != prov["npz_sha256"].lower():
            issues.append(f"{key} npz")
        if r["MetaSha256"].lower() != prov["meta_sha256"].lower():
            issues.append(f"{key} meta")
        if _exact_int(r["N_Rounds"]) != 30:
            issues.append(f"{key} nrounds")
        if r["GPU"] != "Quadro P3200":
            issues.append(f"{key} gpu")
        if r["Torch"] != "2.5.1+cu121":
            issues.append(f"{key} torch")
        hp = ROOT / str(r["HistoryCSV"]).replace("\\", "/")
        ok, msg, auc = _hist_ok(hp, f1)
        if not ok:
            issues.append(f"{key} hist {msg}")
        elif auc != float(r["AUC"]):
            issues.append(f"{key} auc {auc} vs {r['AUC']}")
        print(f"OK {key} F1={f1:.16f}" if ok and not any(str(key) in x for x in issues[-8:]) else f"{key} {msg}")

    missing = [
        (m, a, s)
        for m in MODELS
        for s in SEEDS
        for a in ALGOS
        if (m, a, s) not in cells
    ]
    print("n_raw", len(rows), "unique", len(cells), "missing", missing)
    print("issues", issues)

    out_rows = []
    print("\n=== 3-seed means (unrounded Final F1) ===")
    means: dict[tuple[str, str], float] = {}
    for m in MODELS:
        print(f"\n{m}")
        for a in ALGOS:
            mu = sum(cells[(m, a, s)] for s in SEEDS) / 3
            means[(m, a)] = mu
            print(f"  {a:24s} {mu:.16f}")
        order = sorted(((a, means[(m, a)]) for a in ALGOS), key=lambda x: -x[1])
        print("  rank", " > ".join(a for a, _ in order))

    contrasts = [
        ("FT-UW", "FedTrimmedAvg", "UnweightedFedAvg", "trim", True),
        ("WT-FA", "WeightedFedTrimmedAvg", "FedAvg", "trim", True),
        ("FA-UW", "FedAvg", "UnweightedFedAvg", "weight", False),
        ("WT-FT", "WeightedFedTrimmedAvg", "FedTrimmedAvg", "weight", False),
    ]
    print("\n=== paired contrasts ===")
    support: dict[str, dict[str, bool]] = {m: {} for m in MODELS}
    for m in MODELS:
        print(f"\n{m}")
        trim_ok = True
        weight_ok = True
        for name, a, b, factor, want_pos in contrasts:
            ds = [cells[(m, a, s)] - cells[(m, b, s)] for s in SEEDS]
            mu = sum(ds) / 3
            pos = sum(1 for d in ds if d > 0)
            neg = sum(1 for d in ds if d < 0)
            z = sum(1 for d in ds if d == 0)
            if factor == "trim":
                trim_ok = trim_ok and mu > 0
            else:
                weight_ok = weight_ok and mu < 0
            print(
                f"  {name:6s} mean={mu:.16f}  s42={ds[0]:.16f} s123={ds[1]:.16f} "
                f"s7={ds[2]:.16f}  signs +{pos}/-{neg}/tie{z}"
            )
            out_rows.append(
                {
                    "Model": m,
                    "Contrast": name,
                    "Factor": factor,
                    "MeanDelta": f"{mu:.16f}",
                    "d_s42": f"{ds[0]:.16f}",
                    "d_s123": f"{ds[1]:.16f}",
                    "d_s7": f"{ds[2]:.16f}",
                    "n_pos": str(pos),
                    "n_neg": str(neg),
                    "n_tie": str(z),
                }
            )
        support[m]["HE2-trim"] = trim_ok
        support[m]["HE2-weight"] = weight_ok
        print(f"  HE2-trim   {trim_ok}")
        print(f"  HE2-weight {weight_ok}")

    if issues or missing:
        print("FAIL")
        return 1
    print("\nALL 24 INTEGRITY OK")
    print("HE2-trim models", {m: support[m]["HE2-trim"] for m in MODELS})
    print("HE2-weight models", {m: support[m]["HE2-weight"] for m in MODELS})
    dest = ROOT / "q1_e2_results" / "reporting" / "Q1_E2_PAIRED_CONTRASTS.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

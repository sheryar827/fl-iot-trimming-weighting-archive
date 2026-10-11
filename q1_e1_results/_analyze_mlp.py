import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path

rows = list(
    csv.DictReader(open(Path(__file__).parent / "csv" / "q1_e1_nbaiot_raw.csv", encoding="utf-8"))
)
print("n_rows", len(rows))

by = defaultdict(list)
for r in rows:
    by[(r["Algorithm"], r["Alpha"])].append(float(r["Final_F1"]) * 100)

print("\n=== Final F1 % mean +/- sd (n=3) ===")
for alpha in ["1.0", "0.5", "0.1"]:
    for algo in ["FedAvg", "UnweightedFedAvg"]:
        vals = by[(algo, alpha)]
        m = sum(vals) / len(vals)
        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {algo:20s} a={alpha}: {m:.2f} +/- {sd:.2f}  {[round(v, 2) for v in vals]}")

print("\n=== Unweighted - FedAvg (pp) per seed ===")
for alpha in ["1.0", "0.5", "0.1"]:
    fa = {
        r["Seed"]: float(r["Final_F1"]) * 100
        for r in rows
        if r["Algorithm"] == "FedAvg" and r["Alpha"] == alpha
    }
    uw = {
        r["Seed"]: float(r["Final_F1"]) * 100
        for r in rows
        if r["Algorithm"] == "UnweightedFedAvg" and r["Alpha"] == alpha
    }
    diffs = []
    for s in sorted(fa, key=lambda x: int(x)):
        d = uw[s] - fa[s]
        diffs.append(d)
        print(f"  a={alpha} seed={s}: UW={uw[s]:.2f} FA={fa[s]:.2f} d={d:+.2f}")
    print(f"  a={alpha} mean UW-FA: {sum(diffs)/len(diffs):+.3f} pp  sd={stats.stdev(diffs):.3f}")

# Welch-ish note: n=3 only
print("\n=== Mean tables (Final F1 %) ===")
print(f"{'alpha':>6} {'FedAvg':>10} {'Unwt':>10} {'UW-FA':>10}")
for alpha in ["1.0", "0.5", "0.1"]:
    fa = sum(by[("FedAvg", alpha)]) / 3
    uw = sum(by[("UnweightedFedAvg", alpha)]) / 3
    print(f"{alpha:>6} {fa:10.2f} {uw:10.2f} {uw-fa:+10.2f}")

# vs packet
pack_path = Path(__file__).resolve().parents[1] / "corrected_nbaiot.csv"
if pack_path.exists():
    pack = list(csv.DictReader(open(pack_path, encoding="utf-8")))
    print("\n=== Local FedAvg mean vs corrected_nbaiot FedAvg (A100 Final) ===")
    for alpha in ["1.0", "0.5", "0.1"]:
        local = sum(by[("FedAvg", alpha)]) / 3
        for r in pack:
            if r["Model"] == "MLP" and r["Algorithm"] == "FedAvg" and str(r["alpha"]) == alpha:
                pub = float(r["Final_F1"])
                print(f"  a={alpha}: local={local:.2f}  packet={pub:.2f}  d={local-pub:+.2f}")

# vs published FedTrim if in packet
print("\n=== Context: packet FedTrim MLP Final F1 ===")
if pack_path.exists():
    for r in pack:
        if r["Model"] == "MLP" and r["Algorithm"] == "FedTrimmedAvg":
            print(f"  a={r['alpha']}: Trim={float(r['Final_F1']):.2f}")

ts = [float(r["Train_s"]) / 60 for r in rows]
print(
    f"\ntrain min/run: mean={sum(ts)/len(ts):.2f} "
    f"min={min(ts):.2f} max={max(ts):.2f} sum_h={sum(ts)/60:.2f}"
)
print("wall_min reported 529.8 h=", 529.8 / 60)

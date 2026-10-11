# Step 0 — Q1-E3 gates G1–G3 (after prereg acceptance)

**Hold:** Do **not** run until  
`Proposer: Sheryar Kiani` / `Approver: Dr Hussain Kazmi` / `Date: …` / `Accept E3 prereg r3`.

Prereg: `q1_e3_results/reporting/Q1_E3_PREREG_DRAFT.md` (r3).

---

## A. Backup (required before any gate that overwrites probe JSON)

Already recommended after engine edit:

```powershell
cd "C:\Elicit_FedTrim_Novelty_Audit\aoa-dear-i-have-commented-the\Research Paper"

Copy-Item "probe_results\probe_fast_match_notebook.json" `
  "probe_results\probe_fast_match_notebook_PRE_E3ENGINE_2026-08-15.json"
```

(If that file already exists from prep, do not overwrite a known-good backup without renaming.)

---

## B. Run gates in order (~2 h total)

```powershell
python3.12 -u run_q1_e3_gates.py --gate G1 --require-from-cache
python3.12 -u tools/verify_q1_e3_gates.py --gate G1
# paste G1 footer for independent verify; only if PASS:

python3.12 -u run_q1_e3_gates.py --gate G2 --require-from-cache
python3.12 -u tools/verify_q1_e3_gates.py --gate G2
# only if PASS:

python3.12 -u run_q1_e3_gates.py --gate G3 --require-from-cache
python3.12 -u tools/verify_q1_e3_gates.py --gate G3
```

**Startup (first lines must show):** `device: cuda` and `from_cache=True`. Runner **refuses CPU** (unless `--allow-cpu`) and aborts on cache rebuild when `--require-from-cache` is set.

| Gate | Cell | Pass |
|------|------|------|
| **G1** | N-BaIoT MLP FedAvg α=1.0 s42 Adam | Final_F1/Acc exact + **30/30 F1 and Acc** vs published; cfg adam / µ=0.0 |
| **G2** | FedProx **µ=0** same cell Adam | **30/30 F1 and Acc** vs G1; Final F1/Acc == G1; cfg FedProx adam µ=0 |
| **G3** | FedProx **µ=0.01** same cell Adam | Final F1/Acc/Best exact **and 30/30 F1+Acc** vs published FedProx; cfg adam µ=0.01 |

**Stop** on any fail (G2 fail invalidates paired n=1; G3 fail = FedProx lineage).

Outputs: `q1_e3_results/gates/*.json`, `q1_e3_results/csv/history_gate_*.csv`.

---

## C. After G1–G3 all PASS

Only then Step 1 / Step 2 cells from the prereg. No Step 1 before G3 PASS.

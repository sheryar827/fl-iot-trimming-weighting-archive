# Q1-E6 theory note — Byzantine trimming vs honest label skew

**Status:** **ACCEPTED 2026-09-22** as conceptual manuscript text (Sheryar Kiani / Dr Hussain Kazmi). **Not** a rare-class result and **not** a theorem.  
**Date (UTC first written):** 2026-09-21  
**Acceptance date:** **2026-09-22** (22-9-2026)  
**Does not** claim universal benefit of trimming or Byzantine robustness (E8 remains **DEFER**). Rare-class measurement remains **E7**.

---

## What coordinate-wise trimming was designed for

In Byzantine FL, a coordinate-wise trimmed mean discards extreme **client updates** (or coordinates) so a few arbitrary vectors cannot dominate the aggregate (Yin et al.–style robust mean). The threat model is **malicious or arbitrary** updates, not “harder classes.”

Our experiments are **honest**: every client runs the same training code on real traffic. Extremes come from **label mix and sample size**, not from an adversary.

---

## Why honest skew can still move the aggregate

Let client \(k\) return \(w_k\) after local SGD/Adam. Size-weighted FedAvg is \(\sum_k (n_k/n)\,w_k\). Two honest mechanisms then matter:

1. **Mass:** a device with more rows (or a 20k-capped class pile) pulls the average even if its labels are atypical.  
2. **Geometry of the update:** a client whose local data omit whole attack families (e.g. Ennio/Samsung: **no Mirai** in training) takes **a few local steps** (here **3 epochs**, not a solved empirical-risk minimizer) on a **different class mixture**. That \(w_k\) **can** sit in a tail of the coordinate-wise client cloud **without being Byzantine**. This is a **possible mechanism**, not a result proven before **Q1-E7**.

Coordinate-wise trim then discards extreme coordinates **as if** they were contamination. **Possible** effects (to be checked with per-class metrics in E7, not assumed here):

- **Help macro-F1** if those tails bury rare classes on the pooled test set;  
- **Hurt macro-F1** if those tails are the only clients that carry a class the test set still contains.

So trim is a **heuristic filter on honest heterogeneity**, not a robust mean under attacks. Whether it helps is **empirical** (CIC confirmatory 2×2 vs N-BaIoT C3) and **class-wise evidence is E7**, not this note.

Unweighted averaging (E1) is the other lever: it removes **mass**, not tails. The 2×2 (FT−UW, WT−FA vs FA−UW, WT−FT) exists so those levers are not conflated.

---

## Acc vs macro-F1

Accuracy can stay high when a majority class (benign / a frequent flood) is easy. **Macro-F1 averages per-class F1 scores**; each class F1 is computed from that class’s precision and recall. Macro-F1 therefore **moves** when minority attacks are dropped even if Acc does not. Trimming that improves Acc while cutting a rare attack is **not** a win under this paper’s primary metric (Final **macro-F1**). Whether rare-class F1 actually moves is an **E7** measurement, not a theorem. C4’s Best vs Final sensitivity is the same honesty: the test-set “best round” is not a licence to shop.

---

## What we do **not** claim

- Trim is **Byzantine-robust** in this codebase (no E8).  
- Trim **always** helps honest skew (N-BaIoT may stay small-gap).  
- Device-based FL (E2) is **deployment** or **unseen-device** generalization.  
- FedProx’s C5 hole is a theorem; it is an **Adam×µ** (and sometimes SGD) empirical interaction.

---

## One sentence for the paper

Coordinate-wise trimming, designed to limit arbitrary client vectors, can also clip **honest** updates that sit in the tails because of **label mix or size**; **if** those tails bury or carry rare classes, **macro-F1** (mean of per-class F1) may rise or fall—hence a 2×2 with unweighted averaging, not a robustness trophy, and hence **E7** before any rare-class claim.

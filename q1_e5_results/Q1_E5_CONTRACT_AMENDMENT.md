# Amendment — Q1-E5 preregistration acceptance

**Status:** **APPLIED 2026-10-06** — proposer **Sheryar Kiani**; approver **Dr Hussain Kazmi** (supervisor)  
**Acceptance date:** **2026-10-06** (10-6-2026)  
**Parent:** `Q1_E5_PREREG_DRAFT.md`, presented at SHA-256 `C985EF651B23BC03987A17FD205BE314C8CCE9F13FE7C8C4E8B1CF8911E59D5F` and accepted, after the acceptance block was filled, at SHA-256 `CD20C7A61B3A41AD214CC51BF4221F7837604FD1236F75282FD07BE346EC6517`

## Who (accepted)

- Proposer: **Sheryar Kiani**
- Approver: **Dr Hussain Kazmi** (supervisor)
- Acceptance date: **2026-10-06** (10-6-2026)
- Decision: Accept this 84-cell Q1-E5 preregistration as written.

## What (applied to `Q1_CONTRIBUTION_CONTRACT.md` on 2026-10-06)

Q1-E5 is **preregistration accepted** at **0/84** decision cells. The 24 tuning runs are not decision cells and are outside the Holm family. No runner, integrity suite, preflight, tuning artifact, or GPU cell exists.

The accepted design has two separate comparisons on T1, T2, T5, and T6, using seeds {127, 131, 137, 139, 149, 151, 157}:

- CoordMedian–Adam versus the sealed Q1-E4 FedTrimmedAvg–Adam rows.
- FedNova using the published momentum-SGD normalization versus matched FedAvg–momentum-SGD.

One Holm family contains the eight preregistered paired tests. StepNormAdam is not approved. SCAFFOLD is not an arm. FedNova–SGD is not compared with sealed Adam FedAvg. This acceptance does not reopen Q1-E4. **C5/C6/C8 unchanged.** CIC Wave 3 and Q1-E8 remain deferred.

No E5 GPU work is authorized until the E5 module, the CPU-only integrity suite, and a no-training preflight have passed. The 2026-10-05 contract line that said Q1-E5 was not started remains historical.

## Why

Dr Hussain Kazmi accepted the audited preregistration by name and date. The contract records that acceptance at 0/84 before any E5 implementation or training.

## Acceptance checklist

- [x] Proposer / approver / date (**Sheryar Kiani** / **Dr Hussain Kazmi** / **2026-10-06**)
- [x] Presented scientific text preserved by hash `C985EF651B23BC03987A17FD205BE314C8CCE9F13FE7C8C4E8B1CF8911E59D5F`
- [x] Accepted file hash `CD20C7A61B3A41AD214CC51BF4221F7837604FD1236F75282FD07BE346EC6517` recorded after the acceptance block was filled
- [x] Contract header, Q1-E5 status, Q1-E5 section, Top-Q1 status, and next action updated
- [x] 2026-09-26 **0/112** preregistration strings retained for the sealed E4 runner
- [x] C5/C6/C8 unchanged; Q1-E4 not reopened
- [x] StepNormAdam not approved
- [x] No E5 runner, preflight, tuning artifact, or GPU cell created

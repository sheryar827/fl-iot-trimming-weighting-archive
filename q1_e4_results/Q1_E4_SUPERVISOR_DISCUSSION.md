# Talk sheet for Q1-E4 completion

**What this is:** A plain-language note for the supervisor meeting.  
**Completion accepted and applied:** 2026-10-05 (5-10-2026). Proposer **Sheryar Kiani**. Approver **Dr Hussain Kazmi**. The contract change is `Q1_E4_CONTRACT_AMENDMENT.md` plus `Q1_CONTRIBUTION_CONTRACT.md`.  
**What this note is not:** It is not the contract, and it does not approve E5. The speech below is the note taken to the meeting.  
**Checked on:** 5 October 2026. The numbers were not changed by the later wording or acceptance edits.

You can read the first box out loud. The rest explains the same facts in short sentences.

## Please start with this

Dr Kazmi, the Q1-E4 runs are finished. All 112 new runs are done. The file checks match. The numbers were added again, and they match the report.

I am not asking for more training. I am asking you to accept that this experiment is complete, and to accept the result as written.

Before this acceptance, the contract said that zero of the 112 runs were done. That was because the yes on 26 September was only for the plan. The finish is now accepted, and a separate dated note has been applied. We will not start the next experiment, E5, until E5 has its own accepted preregistration.

## What we were testing

These were simulated federated clients. Each client was a partition of the dataset. This was not a deployment on physical devices. This was not a privacy audit.

We compared four ways to combine the client updates.

- **Plain average (UW).** Every client counts the same.
- **Size-weighted average (FedAvg, or FA).** A client with more training samples counts more. FedAvg uses the sample count, not the size of a computer.
- **Trimmed average (FT).** For each number in the model, the highest update and the lowest update are left out. The rest count the same.
- **Trim, then weight (WT).** The same extreme updates are left out. Then a client with more training samples counts more.

We used four settings only.

- **T1.** CIC-IoT2023 data, MLP model, uneven class split.
- **T2.** CIC-IoT2023 data, CNN model, uneven class split.
- **T5.** N-BaIoT data, MLP model, fixed device partition.
- **T6.** N-BaIoT data, CNN model, fixed device partition.

The score is the final macro-F1 after 30 rounds. Macro-F1 gives every class a voice, including small classes. We use the exact score, before rounding, to make the decision.

## What “finished” means

- All **112** new runs are done.
- All **28** groups of four methods are complete for the seven new seeds. A seed is a fixed starting number. On CIC, the seed sets matched random conditions, and the same seed gives every method those same conditions. On N-BaIoT, the device partition is fixed across seeds.
- The five run groups are complete: **4, 28, 32, 32, and 16**.
- The integrity tests passed **32 of 32**, and they were run twice.
- The report has **160** seed rows, **16** planned comparisons, and **16** combined ten-seed descriptive estimates.
- All **305** listed file fingerprints matched.
- The statistics were calculated again from the scores. No mismatch was found.

The full report is `Research Paper/q1_e4_results/reporting/Q1_E4_REPORTING_PACKET.md`.

## The one result we can count

We planned **16** comparisons before looking at these new runs. The seven new seeds are **127, 131, 137, 139, 149, 151, and 157**. The three older seeds were not used for this yes-or-no test.

Doing 16 tests can make a lucky result look special. Holm is the fairness rule we promised to use. It makes the pass line harder. Our pass line is **0.05**. A smaller Holm-adjusted p-value can meet the decision rule.

Only **one** comparison passed.

- **Setting:** T2, CIC-IoT2023, CNN.
- **Comparison:** size-weighted average minus plain average.
- **Average gap:** **−10.7252** percentage points.
- **Unadjusted 95% paired-t confidence interval:** **−15.9527 to −5.4977**.
- **Seven new seeds:** **0** higher, **0** tied, **7** lower.
- **Raw p-value:** **0.00240333**.
- **Holm-adjusted p-value:** **0.0384533**.

In plain words: on this one setting, giving a client with more training samples more weight gave a lower final score than counting every client the same. The drop was about **10.7** points. All seven new tries went the same way. The unadjusted confidence interval lies entirely below zero. This comparison met the preregistered Holm-adjusted decision rule.

The other **15** comparisons did **not** pass. We call them **not sure**. We do not call them “no difference.”

## Two bigger ideas were not supported

These are the two complete block-level hypotheses. Each one needs both of its parts to pass in the same setting.

**Hypothesis: trimming helps.**  
Trimmed average must beat plain average, and trim-then-weight must beat size-weighted average. Both must pass the Holm-adjusted decision rule.  
This did **not** happen in T1, T2, T5, or T6.

**Hypothesis: giving clients with more training samples more weight hurts.**  
Size-weighted average must be worse than plain average, and trim-then-weight must be worse than trimmed average. Both must pass the Holm-adjusted decision rule.  
The T2 result above is only the first half. The second half, trim-then-weight minus trimmed average, is **not sure**. Its Holm-adjusted p-value is **1**.  
So this hypothesis is also **not** supported in T1, T2, T5, or T6.

One comparison met the preregistered Holm-adjusted decision rule. This is not enough to support either complete block-level hypothesis.

## What we must not say

- A “not sure” result does **not** mean the methods are equal.
- It also does **not** prove that there is no effect.
- The 16 combined ten-seed descriptive estimates are not a second proof. The first three seeds had already been seen.
- The side check of how trimming and weighting work together is description only. It was not one of the 16 yes-or-no tests.
- We do not treat “trim and then weight” versus “plain average” as one single cause. That changes two things at once.
- These findings are only for the four settings above, under the frozen training setup.
- The N-BaIoT clients use a fixed partition of source devices in the dataset. This was not a physical-device deployment, and it is not proof about a new device outside the study.
- We do not claim that one method is always better, that the method stops attacks, or that it protects rare classes.
- We do not say this is the first study of its kind.

## What this acceptance does not do

The 26 September plan line remains in the contract on purpose. It still records **0 of 112** at preregistration. The completion line now says **Done (112/112 new cells)** on 2026-10-05. The E4 report is now a number source. The old “build the runner and start the runs” step is marked done.

E5 is not started and has no accepted preregistration.

## Please decide these points

1. Do you accept that the Q1-E4 runs and report are complete?
2. Do you accept the result as written: one comparison met the Holm-adjusted decision rule, fifteen comparisons are not sure, and neither complete block-level hypothesis is supported?
3. The three facts below were supplied and written into the completion note.
4. The **separate** completion note is applied. The 26 September plan note stays as history.
5. No more E4 training is needed.
6. Please do not ask us to start E5 training until E5 has its own accepted preregistration.

## Accepted at the meeting

- Proposer: **Sheryar Kiani**
- Approver: **Dr Hussain Kazmi**
- Date the finish is accepted: **2026-10-05** (5-10-2026)

These three facts are now in the completion amendment and the contract. They do not approve E5.

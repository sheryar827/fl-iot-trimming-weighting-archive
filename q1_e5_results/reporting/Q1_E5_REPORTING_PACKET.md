# Q1-E5 coordinate-wise median reporting packet

**Study:** 28 CoordMedian-Adam cells against the sealed Q1-E4 FedTrimmedAvg-Adam rows.  
**Endpoint:** unrounded Final macro-F1 at round 30.  
**Difference:** sealed FedTrimmedAvg-Adam minus CoordMedian-Adam.  
**Family:** four paired tests, in block order T1, T2, T5, T6, at Holm family-wise level 0.05.  
**Seeds:** {127, 131, 137, 139, 149, 151, 157}. Each test uses n=7 and df=6.

This packet recomputes those four tests from `q1_e5_results/coordmedian/cells/` and the sealed `q1_e4_results/reporting/q1_e4_seed.csv`. The CSV algorithm field is `FedTrimmedAvg`. The accepted comparator name is FedTrimmedAvg-Adam. The builder does not train. It does not add a completion signature, and it does not freeze the manuscript.

## Prospective paired-t/Holm family

Exactly four comparisons were evaluated. Percentage-point values are 100 times the unrounded mean of the seven Final macro-F1 differences, displayed to four decimal places. Intervals are the unadjusted 95% paired-t intervals. p-values use Python `.6g`. Signs are negative, tie, and positive counts, in that order. Holm adjustment uses these four rows only. No row is supported.

| Block | Contrast | Mean pp | 95% CI pp | Signs - / 0 / + | Raw p | Holm p | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| T1 | FT-CoordMedian | 0.3471 | [-1.6900, 2.3842] | 3/0/4 | 0.691258 | 1 | inconclusive |
| T2 | FT-CoordMedian | 0.6388 | [-1.7203, 2.9980] | 3/0/4 | 0.532201 | 1 | inconclusive |
| T5 | FT-CoordMedian | -1.7913 | [-4.5363, 0.9537] | 4/0/3 | 0.161432 | 0.645727 | inconclusive |
| T6 | FT-CoordMedian | -1.6825 | [-4.8882, 1.5233] | 3/0/4 | 0.246421 | 0.739263 | inconclusive |

A positive supported mean would say the trimmed mean is higher than the median in that block only. A negative supported mean would say the median is higher in that block only. Inconclusive is not equivalence, and no equivalence procedure was applied.

## Claim boundary

- These Holm p-values are the four-test family. They are not the historical eight-test family, and they are not combined with the Q1-E4 Holm result.
- The Q1-E4 packet still contains the only supported comparison in that family: T2 FedAvg minus UnweightedFedAvg. This packet does not rewrite that row.
- FedNova is deferred. It is not a row in this family and it is not a finding that FedNova is inferior.
- One supported block would not authorize a general aggregator claim. No block is supported.
- The clients are simulated dataset partitions. This packet does not authorize a universal, deployment, privacy, rare-class, or adversarial-robustness claim.

## Seed-wise Final macro-F1

CoordMedian values are the cell-file strings. FedTrimmedAvg values are the sealed CSV strings. Delta is FedTrimmedAvg minus CoordMedian, in Final macro-F1 units.

| Block | Seed | CoordMedian | FedTrimmedAvg | Delta |
|---|---:|---:|---:|---:|
| T1 | 127 | 0.506024873091221 | 0.5200658363741384 | 0.014040963282917374 |
| T1 | 131 | 0.6026520214262725 | 0.5863587189507125 | -0.016293302475560045 |
| T1 | 137 | 0.5881479334587789 | 0.6292905907627344 | 0.0411426573039555 |
| T1 | 139 | 0.5458037730226241 | 0.5261627917837095 | -0.019640981238914557 |
| T1 | 149 | 0.6281609027472308 | 0.641715151106488 | 0.013554248359257226 |
| T1 | 151 | 0.6114118818748889 | 0.5963351980390202 | -0.015076683835868687 |
| T1 | 157 | 0.6055013185966599 | 0.6120696092925507 | 0.006568290695890822 |
| T2 | 127 | 0.4245701071114253 | 0.4429604325292938 | 0.018390325417868503 |
| T2 | 131 | 0.5369303257896424 | 0.5336017372324453 | -0.0033285885571970875 |
| T2 | 137 | 0.4769148059144899 | 0.47014565199463126 | -0.0067691539198586614 |
| T2 | 139 | 0.453415109659283 | 0.46423694711696684 | 0.010821837457683836 |
| T2 | 149 | 0.5344712716199135 | 0.5752702548571845 | 0.04079898323727105 |
| T2 | 151 | 0.5506574183523423 | 0.51225728765751 | -0.03840013069483228 |
| T2 | 157 | 0.512686063021183 | 0.5358911401844624 | 0.023205077163279353 |
| T5 | 127 | 0.7856760705361004 | 0.7592001449397826 | -0.02647592559631784 |
| T5 | 131 | 0.7691675072548564 | 0.773432527837241 | 0.004265020582384649 |
| T5 | 137 | 0.7871568984536061 | 0.7872239585617554 | 6.706010814927499e-05 |
| T5 | 139 | 0.8249974002466519 | 0.7771526485046657 | -0.04784475174198621 |
| T5 | 149 | 0.8028217062876155 | 0.7709449484942392 | -0.03187675779337629 |
| T5 | 151 | 0.7705617801899922 | 0.7989903432991192 | 0.028428563109127003 |
| T5 | 157 | 0.8121624198193587 | 0.7602092411628683 | -0.051953178656490406 |
| T6 | 127 | 0.8355425264345079 | 0.7819396019823571 | -0.05360292445215076 |
| T6 | 131 | 0.7982623960253499 | 0.8066663651651588 | 0.008403969139808898 |
| T6 | 137 | 0.8308120135614749 | 0.8341751813059081 | 0.0033631677444331265 |
| T6 | 139 | 0.7390264853062013 | 0.7585221409182796 | 0.019495655612078333 |
| T6 | 149 | 0.8279728832870696 | 0.8362225044359995 | 0.008249621148929909 |
| T6 | 151 | 0.8032936387280741 | 0.7370985131345151 | -0.066195125593559 |
| T6 | 157 | 0.7870182448838974 | 0.7495299262446202 | -0.03748831863927715 |

## Provenance

- Each of the 28 rows is status complete, finite, 30 rounds, mode `match_notebook`, optimizer `adam`, and learning rate `1e-3`.
- Each history file matches the cell's `history_sha256`. Scored rounds are 1 through 30. Round 30 `macro_f1` and `accuracy` match `Final_F1` and `Final_Acc`.
- Each saved decision-partition document matches the cell and `Q1_E4_FROZEN_CONFIG.json` on the scheme digest, client index digests, and client sizes.
- The sealed comparator file hash is `8a9218a130486b68c69f3dd40cf5dd82a217add985c66a91af6fe3dd97da90c9`.
- The frozen-config hash is `3e78215912fb7e1315c7a133cc964038387a88ee8d73577d14390cca6b28f9be`.
- The accepted preregistration hash remains `cd20c7a61b3a41ad214cc51bf4221f7837604fd1236f75282fd07be346ec6517`. This builder does not edit that file.
- Each cell's recorded cache SHA-256 equals the bound NPZ digest. This builder does not re-read the NPZ bytes.
- Paired tests use `fl_local/e4_stats.py` `paired_summary`. Holm adjustment uses `fl_local/e5_aggregators.py` `holm_four`.
- `SHA256SUMS.txt` seals this packet and the files it read. It does not replace the Q1-E4 manifest.

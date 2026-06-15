# IPM Experiment Evaluation & Diagnostics Report

- **Run Mode**: full (Seeds: 3, Epochs: 50, Candidates: 100)
- **Split**: temporal | **Company features**: random
- **Device**: cpu
- **Average Candidate Padding Rate**: 0.83%
- **Cold-Start Statistics (Test Set)**:
  - Unseen Companies (frac_unseen): 10.16%
  - Rare Companies (frac_rare, <= 1 train transfer): 13.77%

## 1. Main Quantitative Results (Table 4)

| Model Architecture | Negative Sampling | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | NDCG@10 | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | - | 0.1206 ± 0.0000 | 0.2061 ± 0.0000 | 0.2497 ± 0.0000 | 0.3184 ± 0.0000 | 0.1951 ± 0.0000 | 0.2104 ± 0.0000 | 0.6129 ± 0.0000 |
| MostPop-IPC | - | 0.1373 ± 0.0000 | 0.2197 ± 0.0000 | 0.2639 ± 0.0000 | 0.3303 ± 0.0000 | 0.2070 ± 0.0000 | 0.2260 ± 0.0000 | 0.4747 ± 0.0000 |
| Recency | - | 0.0461 ± 0.0000 | 0.1541 ± 0.0000 | 0.2088 ± 0.0000 | 0.3003 ± 0.0000 | 0.1473 ± 0.0000 | 0.1675 ± 0.0000 | 0.6427 ± 0.0000 |
| CN | - | 0.0933 ± 0.0000 | 0.1132 ± 0.0000 | 0.1139 ± 0.0000 | 0.1141 ± 0.0000 | 0.1230 ± 0.0000 | 0.1079 ± 0.0000 | 0.5563 ± 0.0000 |
| AA | - | 0.0951 ± 0.0000 | 0.1131 ± 0.0000 | 0.1139 ± 0.0000 | 0.1141 ± 0.0000 | 0.1233 ± 0.0000 | 0.1081 ± 0.0000 | 0.5563 ± 0.0000 |
| SVD | Same-IPC Hard | 0.0342 ± 0.0000 | 0.0416 ± 0.0000 | 0.0457 ± 0.0000 | 0.0508 ± 0.0000 | 0.0579 ± 0.0000 | 0.0419 ± 0.0000 | 0.5238 ± 0.0000 |
| MLP | Same-IPC Hard | 0.0940 ± 0.0035 | 0.1587 ± 0.0037 | 0.1925 ± 0.0025 | 0.2481 ± 0.0014 | 0.1538 ± 0.0034 | 0.1633 ± 0.0027 | 0.5825 ± 0.0012 |
| LightGCN | Same-IPC Hard | 0.0208 ± 0.0004 | 0.0460 ± 0.0005 | 0.0680 ± 0.0006 | 0.1191 ± 0.0005 | 0.0650 ± 0.0004 | 0.0604 ± 0.0004 | 0.5110 ± 0.0007 |
| NGCF | Same-IPC Hard | 0.1210 ± 0.0003 | 0.2060 ± 0.0001 | 0.2495 ± 0.0007 | 0.3185 ± 0.0002 | 0.1951 ± 0.0001 | 0.2104 ± 0.0001 | 0.6142 ± 0.0005 |
| GraphSAGE | Same-IPC Hard | 0.0378 ± 0.0347 | 0.0628 ± 0.0391 | 0.0830 ± 0.0391 | 0.1309 ± 0.0376 | 0.0802 ± 0.0358 | 0.0756 ± 0.0369 | 0.5023 ± 0.0305 |
| GAT | Same-IPC Hard | 0.0214 ± 0.0067 | 0.0427 ± 0.0127 | 0.0585 ± 0.0191 | 0.0908 ± 0.0298 | 0.0589 ± 0.0108 | 0.0504 ± 0.0147 | 0.5147 ± 0.0236 |
| GraphSAGE+Debias | Pop-Debiased Hard | 0.1010 ± 0.0078 | 0.1168 ± 0.0051 | 0.1328 ± 0.0048 | 0.1733 ± 0.0057 | 0.1367 ± 0.0063 | 0.1294 ± 0.0056 | 0.5438 ± 0.0120 |
| GraphSAGE+logQ | Same-IPC Hard | 0.0544 ± 0.0356 | 0.0717 ± 0.0371 | 0.0829 ± 0.0393 | 0.1063 ± 0.0510 | 0.0868 ± 0.0385 | 0.0765 ± 0.0403 | 0.5240 ± 0.0261 |
| GraphSAGE+DropEdge | Same-IPC Hard | 0.0602 ± 0.0282 | 0.1078 ± 0.0396 | 0.1360 ± 0.0413 | 0.1886 ± 0.0385 | 0.1140 ± 0.0313 | 0.1162 ± 0.0348 | 0.5468 ± 0.0083 |
| GraphSAGE+Time | Same-IPC Hard | 0.0168 ± 0.0072 | 0.0389 ± 0.0104 | 0.0585 ± 0.0119 | 0.1044 ± 0.0155 | 0.0577 ± 0.0096 | 0.0520 ± 0.0107 | 0.4756 ± 0.0228 |
| GraphSAGE+IPS | Same-IPC Hard | 0.1060 ± 0.0005 | 0.1135 ± 0.0002 | 0.1208 ± 0.0002 | 0.1410 ± 0.0004 | 0.1292 ± 0.0003 | 0.1196 ± 0.0001 | 0.3872 ± 0.0000 |
| GAT+Debias | Pop-Debiased Hard | 0.0462 ± 0.0205 | 0.0574 ± 0.0183 | 0.0666 ± 0.0171 | 0.0898 ± 0.0147 | 0.0764 ± 0.0186 | 0.0638 ± 0.0180 | 0.5167 ± 0.0134 |
| GAT+logQ | Same-IPC Hard | 0.0692 ± 0.0006 | 0.0763 ± 0.0011 | 0.0835 ± 0.0011 | 0.1026 ± 0.0014 | 0.0965 ± 0.0007 | 0.0822 ± 0.0009 | 0.5384 ± 0.0056 |
| GAT+DropEdge | Same-IPC Hard | 0.0229 ± 0.0103 | 0.0357 ± 0.0113 | 0.0448 ± 0.0114 | 0.0667 ± 0.0108 | 0.0538 ± 0.0105 | 0.0410 ± 0.0107 | 0.4942 ± 0.0098 |
| GAT+Time | Same-IPC Hard | 0.0255 ± 0.0219 | 0.0362 ± 0.0215 | 0.0456 ± 0.0209 | 0.0687 ± 0.0199 | 0.0559 ± 0.0211 | 0.0428 ± 0.0210 | 0.4984 ± 0.0210 |
| GAT+IPS | Same-IPC Hard | 0.0927 ± 0.0160 | 0.1042 ± 0.0079 | 0.1156 ± 0.0047 | 0.1406 ± 0.0013 | 0.1196 ± 0.0110 | 0.1121 ± 0.0088 | 0.3901 ± 0.0042 |
| GraphSAGE+Debias+IPS | Pop-Debiased Hard | 0.1052 ± 0.0002 | 0.1129 ± 0.0005 | 0.1205 ± 0.0005 | 0.1410 ± 0.0001 | 0.1286 ± 0.0002 | 0.1192 ± 0.0002 | 0.3871 ± 0.0000 |
| GraphSAGE+Time+IPS | Same-IPC Hard | 0.1031 ± 0.0042 | 0.1116 ± 0.0022 | 0.1191 ± 0.0017 | 0.1398 ± 0.0004 | 0.1270 ± 0.0029 | 0.1176 ± 0.0022 | 0.3872 ± 0.0002 |
| GAT+Debias+IPS | Pop-Debiased Hard | 0.1039 ± 0.0002 | 0.1098 ± 0.0003 | 0.1190 ± 0.0002 | 0.1415 ± 0.0003 | 0.1273 ± 0.0002 | 0.1183 ± 0.0002 | 0.3872 ± 0.0000 |
| GAT+Time+IPS | Same-IPC Hard | 0.1037 ± 0.0004 | 0.1097 ± 0.0003 | 0.1189 ± 0.0003 | 0.1416 ± 0.0003 | 0.1271 ± 0.0003 | 0.1182 ± 0.0003 | 0.3872 ± 0.0001 |

*Note: AUC is the rank-AUC for a single positive (ties counted as 0.5, equal to sklearn roc_auc_score). MAP and AP are omitted because they equal MRR exactly under the single-positive protocol.*


## 2. Popularity Bias & Inversion Rate Diagnostics (Table 5)

| Model | Spearman Correlation (ρ) | Hard-Neg Inversion Rate | 
| :--- | :---: | :---: |
| MostPop | 1.0000 ± 0.0000 | 55.1607% ± 0.0000% |
| MostPop-IPC | 0.3470 ± 0.0000 | 53.2720% ± 0.0000% |
| Recency | 0.6992 ± 0.0000 | 47.5689% ± 0.0000% |
| CN | 0.0000 ± 0.0000 | 0.1817% ± 0.0000% |
| AA | 0.0000 ± 0.0000 | 0.1876% ± 0.0000% |
| SVD | 0.0281 ± 0.0000 | 2.3114% ± 0.0000% |
| MLP | 0.1800 ± 0.0149 | 18.7215% ± 1.3842% |
| LightGCN | -0.0003 ± 0.0007 | 48.9245% ± 0.0497% |
| NGCF | 0.9554 ± 0.0094 | 55.0609% ± 0.0431% |
| GraphSAGE | -0.0103 ± 0.1222 | 48.6300% ± 4.7726% |
| GAT | 0.0470 ± 0.1764 | 30.9901% ± 1.6673% |
| GraphSAGE+Debias | -0.0535 ± 0.0780 | 41.5058% ± 3.5021% |
| GraphSAGE+logQ | -0.0465 ± 0.0430 | 21.6723% ± 17.5263% |
| GraphSAGE+DropEdge | -0.0360 ± 0.0643 | 39.7241% ± 5.8171% |
| GraphSAGE+Time | 0.0264 ± 0.1256 | 50.7851% ± 2.4122% |
| GraphSAGE+IPS | -0.9991 ± 0.0002 | 44.6739% ± 0.0063% |
| GAT+Debias | -0.0512 ± 0.0140 | 27.7906% ± 1.0839% |
| GAT+logQ | -0.0134 ± 0.0083 | 27.5039% ± 0.2006% |
| GAT+DropEdge | -0.0958 ± 0.0391 | 29.0850% ± 0.6652% |
| GAT+Time | -0.0179 ± 0.0371 | 30.7304% ± 3.6602% |
| GAT+IPS | -0.9585 ± 0.0575 | 44.6467% ± 0.0318% |
| GraphSAGE+Debias+IPS | -0.9992 ± 0.0000 | 44.6766% ± 0.0115% |
| GraphSAGE+Time+IPS | -0.9974 ± 0.0019 | 44.6265% ± 0.0536% |
| GAT+Debias+IPS | -0.9990 ± 0.0002 | 44.6147% ± 0.0003% |
| GAT+Time+IPS | -0.9974 ± 0.0015 | 44.6197% ± 0.0130% |


### 2.1 GAT Attention Weight Analysis (D12)
- Mann-Whitney U test p-value (Hubs vs Non-Hubs attention weight): **1.0000e+00**
- Hub vs Non-Hub attention weights violin plot saved at `gat_attention_violin.png`

### 2.2 Stratified NDCG@10 (D14)
- GAT performance stratified by popularity across seeds (Option b):
  - **Head**: 0.0369 ± 0.0314
  - **Torso**: 0.0517 ± 0.0261
  - **Tail**: 0.1437 ± 0.1234
- Stratification bar plot saved at `popularity_stratified.png`

### 2.3 Mitigation Sweeps (B4, B5)
- IPS Penalty Beta NDCG@10 (per backbone): GraphSAGE {0.0: 0.0756, 0.5: 0.1193, 1.0: 0.1196, 2.0: 0.1196, 4.0: 0.1196} | GAT {0.0: 0.0504, 0.5: 0.1014, 1.0: 0.1121, 2.0: 0.1173, 4.0: 0.1181}
- Debiased Negative Sampling Alpha NDCG@10 (per backbone): GraphSAGE {0.0: 0.1542, 0.25: 0.0494, 0.5: 0.0920, 0.75: 0.0976, 1.0: 0.0536} | GAT {0.0: 0.0525, 0.25: 0.0425, 0.5: 0.0540, 0.75: 0.0610, 1.0: 0.0450}
- Plots saved to `ips_rerank_sweep.png` and `popularity_debiased_sweep.png` (one curve per backbone)

## 3. Pairwise Statistical Significance

Holm-Bonferroni corrected pairwise comparisons for NDCG@10:

> **WARNING**: only 3 seeds (<6). Wilcoxon signed-rank is undefined/underpowered, so ALL p-values below are forced to 1.0 and are NOT real non-results. Re-run with --seeds >= 6 (full mode uses 10).

Pre-registered comparison family: **24** pairs (Holm-Bonferroni corrected jointly across exactly these comparisons).

| Comparison Pair | Wilcoxon Raw p | Wilcoxon Adjusted p (Holm) | t-Test Raw p | t-Test Adjusted p (Holm) |
| :--- | :---: | :---: | :---: | :---: |
| GAT vs GraphSAGE | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| MostPop-IPC vs MostPop | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE vs MostPop | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE vs MostPop-IPC | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE vs SVD | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE vs NGCF | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT vs MostPop | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT vs MostPop-IPC | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT vs SVD | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT vs NGCF | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE+Debias vs GraphSAGE | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE+logQ vs GraphSAGE | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE+DropEdge vs GraphSAGE | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE+Time vs GraphSAGE | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE+IPS vs GraphSAGE | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE+Debias+IPS vs GraphSAGE | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GraphSAGE+Time+IPS vs GraphSAGE | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT+Debias vs GAT | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT+logQ vs GAT | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT+DropEdge vs GAT | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT+Time vs GAT | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT+IPS vs GAT | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT+Debias+IPS vs GAT | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |
| GAT+Time+IPS vs GAT | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 | 1.0000e+00 |


## 4. Demand Score Comparison (E19)
- **Demand (Original) NDCG@10**: 0.1443 ± 0.0000
- **Demand (Revised) NDCG@10**: 0.2151 ± 0.0000

## 5. Horizon Decay (NEW-1)
NDCG@10 by prediction horizon (months between train cutoff and the test transfer). Plot: `horizon_decay.png`.

| Model | 0-6mo | 6-12mo | 12-18mo | 18mo+ |
| :--- | :---: | :---: | :---: | :---: |
| MostPop | 0.0000 (n=0) | 0.0000 (n=0) | 0.2006 (n=36173) | 0.2136 (n=110684) |
| MostPop-IPC | 0.0000 (n=0) | 0.0000 (n=0) | 0.2212 (n=36173) | 0.2276 (n=110684) |
| SVD | 0.0000 (n=0) | 0.0000 (n=0) | 0.0338 (n=36173) | 0.0445 (n=110684) |
| NGCF | 0.0000 (n=0) | 0.0000 (n=0) | 0.2019 (n=36173) | 0.2132 (n=110684) |
| GraphSAGE | 0.0000 (n=0) | 0.0000 (n=0) | 0.0730 (n=36173) | 0.0765 (n=110684) |
| GAT | 0.0000 (n=0) | 0.0000 (n=0) | 0.0486 (n=36173) | 0.0510 (n=110684) |


## 6. IPC-Section Decomposition (NEW-2)
NDCG@10 split by IPC section (first letter of ipc4); (n) = #test queries in that section.

| Model | A | B | C | D | E | F | G | H |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | 0.216(27650) | 0.166(13287) | 0.206(24898) | 0.134(753) | 0.112(2650) | 0.118(4012) | 0.231(43172) | 0.222(30435) |
| MostPop-IPC | 0.231(27650) | 0.179(13287) | 0.222(24898) | 0.097(753) | 0.128(2650) | 0.151(4012) | 0.240(43172) | 0.247(30435) |
| SVD | 0.048(27650) | 0.054(13287) | 0.048(24898) | 0.056(753) | 0.063(2650) | 0.057(4012) | 0.039(43172) | 0.026(30435) |
| NGCF | 0.218(27650) | 0.165(13287) | 0.205(24898) | 0.132(753) | 0.113(2650) | 0.117(4012) | 0.231(43172) | 0.221(30435) |
| GraphSAGE | 0.085(27650) | 0.085(13287) | 0.082(24898) | 0.075(753) | 0.095(2650) | 0.079(4012) | 0.071(43172) | 0.063(30435) |
| GAT | 0.045(27650) | 0.047(13287) | 0.045(24898) | 0.039(753) | 0.038(2650) | 0.034(4012) | 0.056(43172) | 0.056(30435) |


## 7. Patent-Side Cold-Start (NEW-3)
- Fraction of test patents UNSEEN in training (patent-side cold start): **91.11%**
- NDCG@10 on the (new-patent, seen-company) subset vs all / seen-patent:

| Model | All | New-patent & Seen-company | Seen-patent |
| :--- | :---: | :---: | :---: |
| MostPop | 0.2104 | 0.2172 (n=122067) | 0.3353 |
| MostPop-IPC | 0.2260 | 0.2361 (n=122067) | 0.3352 |
| SVD | 0.0419 | 0.0000 (n=122067) | 0.4706 |
| NGCF | 0.2104 | 0.2181 (n=122067) | 0.3276 |
| GraphSAGE | 0.0756 | 0.0452 (n=122067) | 0.1138 |
| GAT | 0.0504 | 0.0346 (n=122067) | 0.0660 |


## 8. Error-Source Decomposition (NEW-4)
Share of FAILED queries (rank>1) attributable to each cause (priority: popularity mechanism > cold-start > residual).

| Model | Popular-hardneg | Rare/new positive | Semantic residual | #failures |
| :--- | :---: | :---: | :---: | :---: |
| GraphSAGE | 97.7% | 0.9% | 1.4% | 423924 |
| GAT | 60.1% | 5.8% | 34.1% | 431148 |


## 9. Qualitative Case Study (NEW-5)
Worst-ranked GAT examples (seed 0): model top-5 companies vs the true buyer.

- Patent `1020140041855` (IPC C01B): true buyer **주식회사 엔씨에프테크** ranked #101.0; GAT top-5 = [신택선, 주식회사 코틱스, 니쯔몰 아베딘 칸, 장국진, 날라탐비칼라이샐비]
- Patent `1020170062677` (IPC G06Q): true buyer **더블스틸 주식회사** ranked #101.0; GAT top-5 = [오민석, 김성회, 양형규, 최지연, 김명진]
- Patent `1020050106769` (IPC G06Q): true buyer **주식회사 앤트랩** ranked #101.0; GAT top-5 = [김성혜, 선영규, 안재진, 조재만, 김낙우]

## 10. Bootstrap 95% CIs over Test Queries (NEW-12)
Percentile CIs from resampling the per-query ranks (captures query-sampling variance that seed-std omits).

| Model | NDCG@10 [95% CI] | Hits@10 [95% CI] | MRR [95% CI] |
| :--- | :---: | :---: | :---: |
| MostPop | 0.2104 [0.2094, 0.2113] | 0.3183 [0.3170, 0.3197] | 0.1951 [0.1941, 0.1960] |
| MostPop-IPC | 0.2260 [0.2249, 0.2270] | 0.3303 [0.3289, 0.3317] | 0.2070 [0.2059, 0.2079] |
| SVD | 0.0419 [0.0413, 0.0423] | 0.0508 [0.0502, 0.0514] | 0.0579 [0.0574, 0.0584] |
| NGCF | 0.2104 [0.2094, 0.2114] | 0.3185 [0.3172, 0.3199] | 0.1951 [0.1941, 0.1960] |
| GraphSAGE | 0.0757 [0.0750, 0.0763] | 0.1309 [0.1300, 0.1319] | 0.0802 [0.0796, 0.0808] |
| GAT | 0.0504 [0.0499, 0.0510] | 0.0908 [0.0899, 0.0917] | 0.0589 [0.0585, 0.0594] |





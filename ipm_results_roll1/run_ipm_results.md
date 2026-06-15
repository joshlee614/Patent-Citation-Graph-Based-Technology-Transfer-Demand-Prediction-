# IPM Experiment Evaluation & Diagnostics Report

- **Run Mode**: full (Seeds: 3, Epochs: 50, Candidates: 100)
- **Split**: temporal | **Company features**: random
- **Device**: cpu
- **Average Candidate Padding Rate**: 0.97%
- **Cold-Start Statistics (Test Set)**:
  - Unseen Companies (frac_unseen): 11.44%
  - Rare Companies (frac_rare, <= 1 train transfer): 15.13%

## 1. Main Quantitative Results (Table 4)

| Model Architecture | Negative Sampling | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | NDCG@10 | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | - | 0.1193 ± 0.0000 | 0.2034 ± 0.0000 | 0.2481 ± 0.0000 | 0.3168 ± 0.0000 | 0.1933 ± 0.0000 | 0.2087 ± 0.0000 | 0.6073 ± 0.0000 |
| MostPop-IPC | - | 0.1414 ± 0.0000 | 0.2212 ± 0.0000 | 0.2646 ± 0.0000 | 0.3304 ± 0.0000 | 0.2096 ± 0.0000 | 0.2280 ± 0.0000 | 0.4735 ± 0.0000 |
| Recency | - | 0.0248 ± 0.0000 | 0.1445 ± 0.0000 | 0.2037 ± 0.0000 | 0.2992 ± 0.0000 | 0.1319 ± 0.0000 | 0.1557 ± 0.0000 | 0.6338 ± 0.0000 |
| CN | - | 0.0953 ± 0.0000 | 0.1170 ± 0.0000 | 0.1181 ± 0.0000 | 0.1185 ± 0.0000 | 0.1260 ± 0.0000 | 0.1113 ± 0.0000 | 0.5584 ± 0.0000 |
| AA | - | 0.0969 ± 0.0000 | 0.1168 ± 0.0000 | 0.1181 ± 0.0000 | 0.1185 ± 0.0000 | 0.1263 ± 0.0000 | 0.1115 ± 0.0000 | 0.5584 ± 0.0000 |
| SVD | Same-IPC Hard | 0.0504 ± 0.0000 | 0.0605 ± 0.0000 | 0.0647 ± 0.0000 | 0.0704 ± 0.0000 | 0.0750 ± 0.0000 | 0.0599 ± 0.0000 | 0.5335 ± 0.0000 |
| MLP | Same-IPC Hard | 0.0788 ± 0.0058 | 0.1431 ± 0.0055 | 0.1779 ± 0.0042 | 0.2330 ± 0.0032 | 0.1378 ± 0.0054 | 0.1480 ± 0.0048 | 0.5728 ± 0.0013 |
| LightGCN | Same-IPC Hard | 0.0243 ± 0.0012 | 0.0499 ± 0.0021 | 0.0722 ± 0.0027 | 0.1235 ± 0.0028 | 0.0687 ± 0.0015 | 0.0644 ± 0.0018 | 0.5139 ± 0.0010 |
| NGCF | Same-IPC Hard | 0.1182 ± 0.0001 | 0.2010 ± 0.0003 | 0.2453 ± 0.0001 | 0.3144 ± 0.0005 | 0.1916 ± 0.0001 | 0.2067 ± 0.0001 | 0.6067 ± 0.0002 |
| GraphSAGE | Same-IPC Hard | 0.0484 ± 0.0333 | 0.0825 ± 0.0515 | 0.1043 ± 0.0574 | 0.1498 ± 0.0594 | 0.0943 ± 0.0411 | 0.0915 ± 0.0469 | 0.5031 ± 0.0251 |
| GAT | Same-IPC Hard | 0.0404 ± 0.0068 | 0.0510 ± 0.0042 | 0.0593 ± 0.0033 | 0.0801 ± 0.0027 | 0.0695 ± 0.0050 | 0.0565 ± 0.0046 | 0.4941 ± 0.0054 |
| GraphSAGE+Debias | Pop-Debiased Hard | 0.0400 ± 0.0408 | 0.0607 ± 0.0458 | 0.0785 ± 0.0463 | 0.1232 ± 0.0458 | 0.0800 ± 0.0424 | 0.0734 ± 0.0438 | 0.5021 ± 0.0425 |
| GraphSAGE+logQ | Same-IPC Hard | 0.0809 ± 0.0129 | 0.1082 ± 0.0201 | 0.1238 ± 0.0284 | 0.1556 ± 0.0469 | 0.1200 ± 0.0191 | 0.1134 ± 0.0244 | 0.5456 ± 0.0102 |
| GraphSAGE+DropEdge | Same-IPC Hard | 0.0112 ± 0.0015 | 0.0302 ± 0.0032 | 0.0492 ± 0.0055 | 0.0986 ± 0.0117 | 0.0518 ± 0.0038 | 0.0455 ± 0.0054 | 0.4720 ± 0.0139 |
| GraphSAGE+Time | Same-IPC Hard | 0.0663 ± 0.0416 | 0.1116 ± 0.0625 | 0.1369 ± 0.0684 | 0.1849 ± 0.0696 | 0.1172 ± 0.0503 | 0.1183 ± 0.0567 | 0.5145 ± 0.0363 |
| GraphSAGE+IPS | Same-IPC Hard | 0.0937 ± 0.0345 | 0.1107 ± 0.0216 | 0.1223 ± 0.0160 | 0.1469 ± 0.0088 | 0.1228 ± 0.0259 | 0.1161 ± 0.0223 | 0.3957 ± 0.0042 |
| GAT+Debias | Pop-Debiased Hard | 0.0131 ± 0.0066 | 0.0260 ± 0.0092 | 0.0373 ± 0.0109 | 0.0638 ± 0.0132 | 0.0450 ± 0.0079 | 0.0335 ± 0.0088 | 0.4808 ± 0.0171 |
| GAT+logQ | Same-IPC Hard | 0.0819 ± 0.0003 | 0.0894 ± 0.0009 | 0.0970 ± 0.0010 | 0.1176 ± 0.0022 | 0.1095 ± 0.0009 | 0.0959 ± 0.0011 | 0.5433 ± 0.0016 |
| GAT+DropEdge | Same-IPC Hard | 0.0425 ± 0.0046 | 0.0537 ± 0.0033 | 0.0621 ± 0.0030 | 0.0846 ± 0.0046 | 0.0723 ± 0.0043 | 0.0595 ± 0.0041 | 0.5006 ± 0.0076 |
| GAT+Time | Same-IPC Hard | 0.0527 ± 0.0069 | 0.0605 ± 0.0066 | 0.0678 ± 0.0068 | 0.0881 ± 0.0074 | 0.0804 ± 0.0068 | 0.0666 ± 0.0070 | 0.5080 ± 0.0084 |
| GAT+IPS | Same-IPC Hard | 0.1166 ± 0.0002 | 0.1228 ± 0.0002 | 0.1314 ± 0.0003 | 0.1537 ± 0.0002 | 0.1396 ± 0.0002 | 0.1309 ± 0.0002 | 0.3926 ± 0.0000 |
| GraphSAGE+Debias+IPS | Pop-Debiased Hard | 0.1178 ± 0.0002 | 0.1252 ± 0.0004 | 0.1326 ± 0.0006 | 0.1529 ± 0.0003 | 0.1407 ± 0.0003 | 0.1315 ± 0.0003 | 0.3926 ± 0.0001 |
| GraphSAGE+Time+IPS | Same-IPC Hard | 0.0720 ± 0.0336 | 0.0989 ± 0.0205 | 0.1134 ± 0.0150 | 0.1416 ± 0.0080 | 0.1071 ± 0.0249 | 0.1026 ± 0.0213 | 0.4014 ± 0.0092 |
| GAT+Debias+IPS | Pop-Debiased Hard | 0.1165 ± 0.0002 | 0.1227 ± 0.0006 | 0.1313 ± 0.0005 | 0.1534 ± 0.0004 | 0.1395 ± 0.0003 | 0.1307 ± 0.0004 | 0.3927 ± 0.0001 |
| GAT+Time+IPS | Same-IPC Hard | 0.1164 ± 0.0001 | 0.1228 ± 0.0001 | 0.1317 ± 0.0002 | 0.1537 ± 0.0002 | 0.1395 ± 0.0001 | 0.1308 ± 0.0001 | 0.3927 ± 0.0001 |

*Note: AUC is the rank-AUC for a single positive (ties counted as 0.5, equal to sklearn roc_auc_score). MAP and AP are omitted because they equal MRR exactly under the single-positive protocol.*


## 2. Popularity Bias & Inversion Rate Diagnostics (Table 5)

| Model | Spearman Correlation (ρ) | Hard-Neg Inversion Rate | 
| :--- | :---: | :---: |
| MostPop | 1.0000 ± 0.0000 | 54.4638% ± 0.0000% |
| MostPop-IPC | 0.3435 ± 0.0000 | 53.0446% ± 0.0000% |
| Recency | 0.6955 ± 0.0000 | 47.3683% ± 0.0000% |
| CN | 0.0000 ± 0.0000 | 0.2014% ± 0.0000% |
| AA | 0.0000 ± 0.0000 | 0.2093% ± 0.0000% |
| SVD | 0.0329 ± 0.0000 | 3.0780% ± 0.0000% |
| MLP | 0.1608 ± 0.0055 | 15.8142% ± 0.5866% |
| LightGCN | -0.0002 ± 0.0008 | 48.6701% ± 0.0984% |
| NGCF | 0.9511 ± 0.0109 | 54.5892% ± 0.0185% |
| GraphSAGE | -0.0124 ± 0.0635 | 48.4306% ± 4.1164% |
| GAT | -0.0535 ± 0.0047 | 30.9263% ± 0.2403% |
| GraphSAGE+Debias | -0.0262 ± 0.0585 | 47.6305% ± 5.8256% |
| GraphSAGE+logQ | -0.0511 ± 0.0422 | 29.2636% ± 16.9777% |
| GraphSAGE+DropEdge | 0.0725 ± 0.0158 | 53.7323% ± 1.6674% |
| GraphSAGE+Time | 0.0448 ± 0.0184 | 44.7313% ± 7.5152% |
| GraphSAGE+IPS | -0.9888 ± 0.0148 | 44.6515% ± 1.0091% |
| GAT+Debias | -0.0790 ± 0.0104 | 31.5424% ± 1.2109% |
| GAT+logQ | -0.0097 ± 0.0233 | 27.7145% ± 0.5575% |
| GAT+DropEdge | -0.0390 ± 0.0291 | 31.3380% ± 0.7384% |
| GAT+Time | -0.0374 ± 0.0060 | 30.2437% ± 0.6706% |
| GAT+IPS | -0.9977 ± 0.0004 | 45.3123% ± 0.0033% |
| GraphSAGE+Debias+IPS | -0.9992 ± 0.0000 | 45.3650% ± 0.0068% |
| GraphSAGE+Time+IPS | -0.9769 ± 0.0158 | 43.4459% ± 1.8436% |
| GAT+Debias+IPS | -0.9991 ± 0.0001 | 45.2962% ± 0.0019% |
| GAT+Time+IPS | -0.9832 ± 0.0172 | 45.3076% ± 0.0003% |


### 2.1 GAT Attention Weight Analysis (D12)
- Mann-Whitney U test p-value (Hubs vs Non-Hubs attention weight): **1.0000e+00**
- Hub vs Non-Hub attention weights violin plot saved at `gat_attention_violin.png`

### 2.2 Stratified NDCG@10 (D14)
- GAT performance stratified by popularity across seeds (Option b):
  - **Head**: 0.0135 ± 0.0012
  - **Torso**: 0.0815 ± 0.0035
  - **Tail**: 0.3200 ± 0.0348
- Stratification bar plot saved at `popularity_stratified.png`

### 2.3 Mitigation Sweeps (B4, B5)
- IPS Penalty Beta NDCG@10 (per backbone): GraphSAGE {0.0: 0.0915, 0.5: 0.1100, 1.0: 0.1161, 2.0: 0.1259, 4.0: 0.1312} | GAT {0.0: 0.0565, 0.5: 0.1275, 1.0: 0.1309, 2.0: 0.1309, 4.0: 0.1309}
- Debiased Negative Sampling Alpha NDCG@10 (per backbone): GraphSAGE {0.0: 0.1249, 0.25: 0.1465, 0.5: 0.0441, 0.75: 0.1303, 1.0: 0.0715} | GAT {0.0: 0.0600, 0.25: 0.0532, 0.5: 0.0349, 0.75: 0.0519, 1.0: 0.0590}
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
- **Demand (Original) NDCG@10**: 0.2259 ± 0.0000
- **Demand (Revised) NDCG@10**: 0.2525 ± 0.0000

## 5. Horizon Decay (NEW-1)
NDCG@10 by prediction horizon (months between train cutoff and the test transfer). Plot: `horizon_decay.png`.

| Model | 0-6mo | 6-12mo | 12-18mo | 18mo+ |
| :--- | :---: | :---: | :---: | :---: |
| MostPop | 0.0000 (n=0) | 0.2174 (n=5126) | 0.2068 (n=35678) | 0.2088 (n=106052) |
| MostPop-IPC | 0.0000 (n=0) | 0.2459 (n=5126) | 0.2334 (n=35678) | 0.2254 (n=106052) |
| SVD | 0.0000 (n=0) | 0.0536 (n=5126) | 0.0572 (n=35678) | 0.0611 (n=106052) |
| NGCF | 0.0000 (n=0) | 0.2163 (n=5126) | 0.2041 (n=35678) | 0.2072 (n=106052) |
| GraphSAGE | 0.0000 (n=0) | 0.0935 (n=5126) | 0.0900 (n=35678) | 0.0919 (n=106052) |
| GAT | 0.0000 (n=0) | 0.0439 (n=5126) | 0.0532 (n=35678) | 0.0583 (n=106052) |


## 6. IPC-Section Decomposition (NEW-2)
NDCG@10 split by IPC section (first letter of ipc4); (n) = #test queries in that section.

| Model | A | B | C | D | E | F | G | H |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | 0.211(28089) | 0.159(14600) | 0.202(23689) | 0.116(870) | 0.140(3417) | 0.119(4105) | 0.228(41691) | 0.232(30395) |
| MostPop-IPC | 0.226(28089) | 0.169(14600) | 0.216(23689) | 0.096(870) | 0.184(3417) | 0.132(4105) | 0.239(41691) | 0.274(30395) |
| SVD | 0.059(28089) | 0.062(14600) | 0.069(23689) | 0.074(870) | 0.121(3417) | 0.077(4105) | 0.058(41691) | 0.045(30395) |
| NGCF | 0.212(28089) | 0.158(14600) | 0.202(23689) | 0.119(870) | 0.135(3417) | 0.118(4105) | 0.225(41691) | 0.227(30395) |
| GraphSAGE | 0.084(28089) | 0.084(14600) | 0.088(23689) | 0.068(870) | 0.088(3417) | 0.081(4105) | 0.098(41691) | 0.100(30395) |
| GAT | 0.046(28089) | 0.067(14600) | 0.054(23689) | 0.052(870) | 0.046(3417) | 0.045(4105) | 0.062(41691) | 0.058(30395) |


## 7. Patent-Side Cold-Start (NEW-3)
- Fraction of test patents UNSEEN in training (patent-side cold start): **87.71%**
- NDCG@10 on the (new-patent, seen-company) subset vs all / seen-patent:

| Model | All | New-patent & Seen-company | Seen-patent |
| :--- | :---: | :---: | :---: |
| MostPop | 0.2087 | 0.2121 (n=116909) | 0.3237 |
| MostPop-IPC | 0.2280 | 0.2366 (n=116909) | 0.3228 |
| SVD | 0.0599 | 0.0000 (n=116909) | 0.4868 |
| NGCF | 0.2067 | 0.2116 (n=116909) | 0.3115 |
| GraphSAGE | 0.0915 | 0.0809 (n=116909) | 0.1266 |
| GAT | 0.0565 | 0.0185 (n=116909) | 0.0202 |


## 8. Error-Source Decomposition (NEW-4)
Share of FAILED queries (rank>1) attributable to each cause (priority: popularity mechanism > cold-start > residual).

| Model | Popular-hardneg | Rare/new positive | Semantic residual | #failures |
| :--- | :---: | :---: | :---: | :---: |
| GraphSAGE | 98.0% | 0.6% | 1.4% | 419259 |
| GAT | 60.8% | 6.0% | 33.2% | 422765 |


## 9. Qualitative Case Study (NEW-5)
Worst-ranked GAT examples (seed 0): model top-5 companies vs the true buyer.

- Patent `1020160044203` (IPC C01B): true buyer **주식회사 비에스지머티리얼즈** ranked #101.0; GAT top-5 = [노종진, 한태희, 버세리엔 피엘씨, 주식회사유원, 광운대학교 산학협력단]
- Patent `1020170103572` (IPC E04H): true buyer **주식회사 투에스** ranked #101.0; GAT top-5 = [이용훈, 박현일, 차영욱, 백정훈, 박순규]
- Patent `1020170103572` (IPC E04H): true buyer **주식회사 진원** ranked #101.0; GAT top-5 = [강대현, 백정훈, 박재우, 정학근, 임재한]

## 10. Bootstrap 95% CIs over Test Queries (NEW-12)
Percentile CIs from resampling the per-query ranks (captures query-sampling variance that seed-std omits).

| Model | NDCG@10 [95% CI] | Hits@10 [95% CI] | MRR [95% CI] |
| :--- | :---: | :---: | :---: |
| MostPop | 0.2086 [0.2076, 0.2096] | 0.3168 [0.3154, 0.3181] | 0.1933 [0.1924, 0.1942] |
| MostPop-IPC | 0.2281 [0.2270, 0.2292] | 0.3304 [0.3290, 0.3318] | 0.2096 [0.2086, 0.2106] |
| SVD | 0.0598 [0.0592, 0.0605] | 0.0704 [0.0696, 0.0711] | 0.0750 [0.0744, 0.0756] |
| NGCF | 0.2067 [0.2057, 0.2077] | 0.3144 [0.3131, 0.3158] | 0.1916 [0.1907, 0.1925] |
| GraphSAGE | 0.0915 [0.0908, 0.0922] | 0.1498 [0.1488, 0.1508] | 0.0943 [0.0936, 0.0949] |
| GAT | 0.0565 [0.0559, 0.0571] | 0.0801 [0.0794, 0.0810] | 0.0695 [0.0690, 0.0701] |





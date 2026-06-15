# IPM Experiment Evaluation & Diagnostics Report

- **Run Mode**: full (Seeds: 3, Epochs: 50, Candidates: 100)
- **Split**: temporal | **Company features**: random
- **Device**: cpu
- **Average Candidate Padding Rate**: 1.19%
- **Cold-Start Statistics (Test Set)**:
  - Unseen Companies (frac_unseen): 12.82%
  - Rare Companies (frac_rare, <= 1 train transfer): 16.65%

## 1. Main Quantitative Results (Table 4)

| Model Architecture | Negative Sampling | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | NDCG@10 | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | - | 0.1149 ± 0.0000 | 0.1943 ± 0.0000 | 0.2355 ± 0.0000 | 0.3036 ± 0.0000 | 0.1861 ± 0.0000 | 0.1999 ± 0.0000 | 0.5934 ± 0.0000 |
| MostPop-IPC | - | 0.1359 ± 0.0000 | 0.2122 ± 0.0000 | 0.2529 ± 0.0000 | 0.3160 ± 0.0000 | 0.2018 ± 0.0000 | 0.2186 ± 0.0000 | 0.4568 ± 0.0000 |
| Recency | - | 0.0300 ± 0.0000 | 0.1350 ± 0.0000 | 0.1962 ± 0.0000 | 0.2883 ± 0.0000 | 0.1305 ± 0.0000 | 0.1520 ± 0.0000 | 0.6185 ± 0.0000 |
| CN | - | 0.0872 ± 0.0000 | 0.1087 ± 0.0000 | 0.1096 ± 0.0000 | 0.1100 ± 0.0000 | 0.1178 ± 0.0000 | 0.1029 ± 0.0000 | 0.5542 ± 0.0000 |
| AA | - | 0.0887 ± 0.0000 | 0.1086 ± 0.0000 | 0.1096 ± 0.0000 | 0.1100 ± 0.0000 | 0.1181 ± 0.0000 | 0.1031 ± 0.0000 | 0.5542 ± 0.0000 |
| SVD | Same-IPC Hard | 0.0400 ± 0.0000 | 0.0465 ± 0.0000 | 0.0500 ± 0.0000 | 0.0543 ± 0.0000 | 0.0629 ± 0.0000 | 0.0466 ± 0.0000 | 0.5250 ± 0.0000 |
| MLP | Same-IPC Hard | 0.0897 ± 0.0011 | 0.1529 ± 0.0023 | 0.1866 ± 0.0025 | 0.2407 ± 0.0020 | 0.1486 ± 0.0012 | 0.1575 ± 0.0011 | 0.5775 ± 0.0017 |
| LightGCN | Same-IPC Hard | 0.0237 ± 0.0007 | 0.0488 ± 0.0006 | 0.0701 ± 0.0005 | 0.1210 ± 0.0004 | 0.0676 ± 0.0006 | 0.0629 ± 0.0006 | 0.5122 ± 0.0005 |
| NGCF | Same-IPC Hard | 0.1150 ± 0.0001 | 0.1927 ± 0.0003 | 0.2337 ± 0.0004 | 0.3013 ± 0.0003 | 0.1855 ± 0.0000 | 0.1988 ± 0.0001 | 0.5936 ± 0.0003 |
| GraphSAGE | Same-IPC Hard | 0.0633 ± 0.0379 | 0.1055 ± 0.0555 | 0.1300 ± 0.0609 | 0.1766 ± 0.0624 | 0.1125 ± 0.0455 | 0.1127 ± 0.0510 | 0.4993 ± 0.0295 |
| GAT | Same-IPC Hard | 0.0503 ± 0.0049 | 0.0607 ± 0.0039 | 0.0687 ± 0.0038 | 0.0892 ± 0.0038 | 0.0793 ± 0.0042 | 0.0661 ± 0.0042 | 0.5098 ± 0.0024 |
| GraphSAGE+Debias | Pop-Debiased Hard | 0.0921 ± 0.0483 | 0.1130 ± 0.0464 | 0.1276 ± 0.0439 | 0.1622 ± 0.0380 | 0.1282 ± 0.0450 | 0.1212 ± 0.0442 | 0.5266 ± 0.0447 |
| GraphSAGE+logQ | Same-IPC Hard | 0.0905 ± 0.0263 | 0.1196 ± 0.0178 | 0.1380 ± 0.0148 | 0.1785 ± 0.0136 | 0.1322 ± 0.0200 | 0.1278 ± 0.0178 | 0.5425 ± 0.0135 |
| GraphSAGE+DropEdge | Same-IPC Hard | 0.0544 ± 0.0294 | 0.0930 ± 0.0395 | 0.1166 ± 0.0406 | 0.1658 ± 0.0379 | 0.1026 ± 0.0327 | 0.1021 ± 0.0350 | 0.4975 ± 0.0144 |
| GraphSAGE+Time | Same-IPC Hard | 0.0403 ± 0.0257 | 0.0727 ± 0.0371 | 0.0958 ± 0.0400 | 0.1462 ± 0.0430 | 0.0864 ± 0.0307 | 0.0845 ± 0.0345 | 0.4862 ± 0.0218 |
| GraphSAGE+IPS | Same-IPC Hard | 0.0719 ± 0.0428 | 0.0987 ± 0.0297 | 0.1164 ± 0.0226 | 0.1512 ± 0.0126 | 0.1087 ± 0.0329 | 0.1058 ± 0.0286 | 0.4175 ± 0.0122 |
| GAT+Debias | Pop-Debiased Hard | 0.0328 ± 0.0188 | 0.0445 ± 0.0197 | 0.0539 ± 0.0197 | 0.0775 ± 0.0182 | 0.0632 ± 0.0187 | 0.0509 ± 0.0188 | 0.4986 ± 0.0128 |
| GAT+logQ | Same-IPC Hard | 0.0886 ± 0.0004 | 0.0961 ± 0.0007 | 0.1036 ± 0.0009 | 0.1231 ± 0.0014 | 0.1158 ± 0.0006 | 0.1021 ± 0.0008 | 0.5488 ± 0.0021 |
| GAT+DropEdge | Same-IPC Hard | 0.0419 ± 0.0137 | 0.0537 ± 0.0129 | 0.0626 ± 0.0120 | 0.0846 ± 0.0112 | 0.0719 ± 0.0130 | 0.0593 ± 0.0126 | 0.5017 ± 0.0141 |
| GAT+Time | Same-IPC Hard | 0.0415 ± 0.0172 | 0.0526 ± 0.0115 | 0.0616 ± 0.0089 | 0.0835 ± 0.0055 | 0.0713 ± 0.0135 | 0.0585 ± 0.0117 | 0.5055 ± 0.0073 |
| GAT+IPS | Same-IPC Hard | 0.1298 ± 0.0001 | 0.1359 ± 0.0000 | 0.1452 ± 0.0001 | 0.1681 ± 0.0001 | 0.1528 ± 0.0001 | 0.1445 ± 0.0001 | 0.4067 ± 0.0001 |
| GraphSAGE+Debias+IPS | Pop-Debiased Hard | 0.1310 ± 0.0005 | 0.1378 ± 0.0019 | 0.1458 ± 0.0013 | 0.1670 ± 0.0007 | 0.1538 ± 0.0007 | 0.1450 ± 0.0004 | 0.4066 ± 0.0001 |
| GraphSAGE+Time+IPS | Same-IPC Hard | 0.0994 ± 0.0456 | 0.1171 ± 0.0316 | 0.1301 ± 0.0236 | 0.1567 ± 0.0140 | 0.1293 ± 0.0356 | 0.1234 ± 0.0311 | 0.4070 ± 0.0006 |
| GAT+Debias+IPS | Pop-Debiased Hard | 0.1258 ± 0.0057 | 0.1322 ± 0.0054 | 0.1414 ± 0.0051 | 0.1645 ± 0.0045 | 0.1491 ± 0.0052 | 0.1407 ± 0.0052 | 0.4063 ± 0.0006 |
| GAT+Time+IPS | Same-IPC Hard | 0.1298 ± 0.0005 | 0.1359 ± 0.0003 | 0.1451 ± 0.0001 | 0.1681 ± 0.0002 | 0.1528 ± 0.0003 | 0.1445 ± 0.0002 | 0.4067 ± 0.0000 |

*Note: AUC is the rank-AUC for a single positive (ties counted as 0.5, equal to sklearn roc_auc_score). MAP and AP are omitted because they equal MRR exactly under the single-positive protocol.*


## 2. Popularity Bias & Inversion Rate Diagnostics (Table 5)

| Model | Spearman Correlation (ρ) | Hard-Neg Inversion Rate | 
| :--- | :---: | :---: |
| MostPop | 1.0000 ± 0.0000 | 55.4346% ± 0.0000% |
| MostPop-IPC | 0.3498 ± 0.0000 | 54.8957% ± 0.0000% |
| Recency | 0.6918 ± 0.0000 | 48.4140% ± 0.0000% |
| CN | 0.0000 ± 0.0000 | 0.1861% ± 0.0000% |
| AA | 0.0000 ± 0.0000 | 0.1915% ± 0.0000% |
| SVD | 0.0289 ± 0.0000 | 2.9040% ± 0.0000% |
| MLP | 0.2038 ± 0.0037 | 18.8357% ± 0.4953% |
| LightGCN | 0.0000 ± 0.0005 | 48.8083% ± 0.0444% |
| NGCF | 0.9466 ± 0.0092 | 55.5085% ± 0.0326% |
| GraphSAGE | 0.0516 ± 0.0538 | 49.7406% ± 4.7866% |
| GAT | -0.0439 ± 0.0017 | 28.7951% ± 0.7376% |
| GraphSAGE+Debias | -0.0536 ± 0.0521 | 36.7073% ± 16.2116% |
| GraphSAGE+logQ | -0.0861 ± 0.0491 | 35.6848% ± 8.5262% |
| GraphSAGE+DropEdge | 0.0612 ± 0.0740 | 50.0871% ± 2.5840% |
| GraphSAGE+Time | 0.0791 ± 0.0660 | 52.0936% ± 3.8978% |
| GraphSAGE+IPS | -0.9725 ± 0.0205 | 42.0220% ± 2.4470% |
| GAT+Debias | -0.1100 ± 0.0676 | 28.3943% ± 0.1626% |
| GAT+logQ | -0.0107 ± 0.0170 | 25.7550% ± 0.0541% |
| GAT+DropEdge | -0.0594 ± 0.0236 | 29.2279% ± 1.1541% |
| GAT+Time | -0.0539 ± 0.0296 | 29.1455% ± 0.4829% |
| GAT+IPS | -0.9903 ± 0.0035 | 44.3024% ± 0.0083% |
| GraphSAGE+Debias+IPS | -0.9992 ± 0.0001 | 44.3290% ± 0.0544% |
| GraphSAGE+Time+IPS | -0.9847 ± 0.0204 | 43.9717% ± 0.5805% |
| GAT+Debias+IPS | -0.9935 ± 0.0076 | 44.2510% ± 0.0636% |
| GAT+Time+IPS | -0.9952 ± 0.0024 | 44.2973% ± 0.0038% |


### 2.1 GAT Attention Weight Analysis (D12)
- Mann-Whitney U test p-value (Hubs vs Non-Hubs attention weight): **1.0000e+00**
- Hub vs Non-Hub attention weights violin plot saved at `gat_attention_violin.png`

### 2.2 Stratified NDCG@10 (D14)
- GAT performance stratified by popularity across seeds (Option b):
  - **Head**: 0.0153 ± 0.0002
  - **Torso**: 0.0799 ± 0.0057
  - **Tail**: 0.3652 ± 0.0283
- Stratification bar plot saved at `popularity_stratified.png`

### 2.3 Mitigation Sweeps (B4, B5)
- IPS Penalty Beta NDCG@10 (per backbone): GraphSAGE {0.0: 0.1127, 0.5: 0.0996, 1.0: 0.1058, 2.0: 0.1309, 4.0: 0.1432} | GAT {0.0: 0.0661, 0.5: 0.1318, 1.0: 0.1445, 2.0: 0.1445, 4.0: 0.1445}
- Debiased Negative Sampling Alpha NDCG@10 (per backbone): GraphSAGE {0.0: 0.0428, 0.25: 0.0406, 0.5: 0.0766, 0.75: 0.0976, 1.0: 0.1075} | GAT {0.0: 0.0640, 0.25: 0.0690, 0.5: 0.0633, 0.75: 0.0417, 1.0: 0.0652}
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
- **Demand (Original) NDCG@10**: 0.2659 ± 0.0000
- **Demand (Revised) NDCG@10**: 0.2586 ± 0.0000

## 5. Horizon Decay (NEW-1)
NDCG@10 by prediction horizon (months between train cutoff and the test transfer). Plot: `horizon_decay.png`.

| Model | 0-6mo | 6-12mo | 12-18mo | 18mo+ |
| :--- | :---: | :---: | :---: | :---: |
| MostPop | 0.0000 (n=0) | 0.1906 (n=8458) | 0.2116 (n=47396) | 0.1947 (n=91003) |
| MostPop-IPC | 0.0000 (n=0) | 0.2172 (n=8458) | 0.2325 (n=47396) | 0.2115 (n=91003) |
| SVD | 0.0000 (n=0) | 0.0312 (n=8458) | 0.0634 (n=47396) | 0.0393 (n=91003) |
| NGCF | 0.0000 (n=0) | 0.1914 (n=8458) | 0.2096 (n=47396) | 0.1939 (n=91003) |
| GraphSAGE | 0.0000 (n=0) | 0.1057 (n=8458) | 0.1198 (n=47396) | 0.1097 (n=91003) |
| GAT | 0.0000 (n=0) | 0.0592 (n=8458) | 0.0629 (n=47396) | 0.0684 (n=91003) |


## 6. IPC-Section Decomposition (NEW-2)
NDCG@10 split by IPC section (first letter of ipc4); (n) = #test queries in that section.

| Model | A | B | C | D | E | F | G | H |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | 0.209(29411) | 0.155(15951) | 0.197(24552) | 0.129(831) | 0.114(3678) | 0.111(3837) | 0.213(37494) | 0.224(31103) |
| MostPop-IPC | 0.228(29411) | 0.166(15951) | 0.202(24552) | 0.131(831) | 0.151(3678) | 0.123(3837) | 0.221(37494) | 0.270(31103) |
| SVD | 0.045(29411) | 0.044(15951) | 0.051(24552) | 0.084(831) | 0.079(3678) | 0.051(3837) | 0.046(37494) | 0.041(31103) |
| NGCF | 0.211(29411) | 0.155(15951) | 0.197(24552) | 0.130(831) | 0.113(3678) | 0.110(3837) | 0.210(37494) | 0.220(31103) |
| GraphSAGE | 0.101(29411) | 0.099(15951) | 0.112(24552) | 0.085(831) | 0.075(3678) | 0.078(3837) | 0.122(37494) | 0.131(31103) |
| GAT | 0.057(29411) | 0.067(15951) | 0.062(24552) | 0.051(831) | 0.056(3678) | 0.052(3837) | 0.077(37494) | 0.067(31103) |


## 7. Patent-Side Cold-Start (NEW-3)
- Fraction of test patents UNSEEN in training (patent-side cold start): **89.79%**
- NDCG@10 on the (new-patent, seen-company) subset vs all / seen-patent:

| Model | All | New-patent & Seen-company | Seen-patent |
| :--- | :---: | :---: | :---: |
| MostPop | 0.1999 | 0.2104 (n=118101) | 0.3009 |
| MostPop-IPC | 0.2186 | 0.2303 (n=118101) | 0.3270 |
| SVD | 0.0466 | 0.0000 (n=118101) | 0.4561 |
| NGCF | 0.1988 | 0.2106 (n=118101) | 0.2887 |
| GraphSAGE | 0.1127 | 0.1210 (n=118101) | 0.1513 |
| GAT | 0.0661 | 0.0197 (n=118101) | 0.0260 |


## 8. Error-Source Decomposition (NEW-4)
Share of FAILED queries (rank>1) attributable to each cause (priority: popularity mechanism > cold-start > residual).

| Model | Popular-hardneg | Rare/new positive | Semantic residual | #failures |
| :--- | :---: | :---: | :---: | :---: |
| GraphSAGE | 98.9% | 0.1% | 1.0% | 412687 |
| GAT | 58.3% | 6.7% | 35.0% | 418431 |


## 9. Qualitative Case Study (NEW-5)
Worst-ranked GAT examples (seed 0): model top-5 companies vs the true buyer.

- Patent `1020180148145` (IPC H10B): true buyer **김민원** ranked #101.0; GAT top-5 = [전우진, 전우영, 박민지, 이민백, 대한민국(서울대학교 총장)]
- Patent `1020110040402` (IPC F24F): true buyer **주식회사 귀뚜라미** ranked #101.0; GAT top-5 = [박판종, 김무섭, 서정해, 고재상, 양홍선]
- Patent `1020180136760` (IPC A61K): true buyer **박해인** ranked #101.0; GAT top-5 = [도완락, 임매용, 배규운, 주식회사 씨피코스메틱, 심정선]

## 10. Bootstrap 95% CIs over Test Queries (NEW-12)
Percentile CIs from resampling the per-query ranks (captures query-sampling variance that seed-std omits).

| Model | NDCG@10 [95% CI] | Hits@10 [95% CI] | MRR [95% CI] |
| :--- | :---: | :---: | :---: |
| MostPop | 0.1999 [0.1990, 0.2010] | 0.3036 [0.3022, 0.3050] | 0.1861 [0.1852, 0.1871] |
| MostPop-IPC | 0.2186 [0.2176, 0.2196] | 0.3159 [0.3146, 0.3174] | 0.2018 [0.2008, 0.2028] |
| SVD | 0.0466 [0.0460, 0.0472] | 0.0543 [0.0536, 0.0549] | 0.0629 [0.0624, 0.0635] |
| NGCF | 0.1988 [0.1978, 0.1999] | 0.3013 [0.3000, 0.3027] | 0.1854 [0.1845, 0.1864] |
| GraphSAGE | 0.1127 [0.1119, 0.1135] | 0.1766 [0.1754, 0.1776] | 0.1125 [0.1118, 0.1132] |
| GAT | 0.0661 [0.0655, 0.0668] | 0.0892 [0.0883, 0.0900] | 0.0793 [0.0787, 0.0799] |




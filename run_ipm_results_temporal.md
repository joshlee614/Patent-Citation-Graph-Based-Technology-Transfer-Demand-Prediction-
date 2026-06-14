# IPM Experiment Evaluation & Diagnostics Report

- **Run Mode**: full (Seeds: 10, Epochs: 50, Candidates: 100)
- **Split**: temporal | **Company features**: random
- **Device**: cpu
- **Average Candidate Padding Rate**: 1.07%
- **Cold-Start Statistics (Test Set)**:
  - Unseen Companies (frac_unseen): 14.12%
  - Rare Companies (frac_rare, <= 1 train transfer): 17.78%

## 1. Main Quantitative Results (Table 4)

| Model Architecture | Negative Sampling | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | NDCG@10 | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | - | 0.1100 ± 0.0000 | 0.1917 ± 0.0000 | 0.2344 ± 0.0000 | 0.3018 ± 0.0000 | 0.1820 ± 0.0000 | 0.1968 ± 0.0000 | 0.5826 ± 0.0000 |
| MostPop-IPC | - | 0.1222 ± 0.0000 | 0.1933 ± 0.0000 | 0.2315 ± 0.0000 | 0.2889 ± 0.0000 | 0.1840 ± 0.0000 | 0.1990 ± 0.0000 | 0.4158 ± 0.0000 |
| Recency | - | 0.0293 ± 0.0000 | 0.1332 ± 0.0000 | 0.1921 ± 0.0000 | 0.2820 ± 0.0000 | 0.1277 ± 0.0000 | 0.1488 ± 0.0000 | 0.6022 ± 0.0000 |
| CN | - | 0.0561 ± 0.0000 | 0.0689 ± 0.0000 | 0.0694 ± 0.0000 | 0.0695 ± 0.0000 | 0.0822 ± 0.0000 | 0.0654 ± 0.0000 | 0.5341 ± 0.0000 |
| AA | - | 0.0570 ± 0.0000 | 0.0687 ± 0.0000 | 0.0694 ± 0.0000 | 0.0695 ± 0.0000 | 0.0824 ± 0.0000 | 0.0655 ± 0.0000 | 0.5341 ± 0.0000 |
| SVD | Same-IPC Hard | 0.0302 ± 0.0000 | 0.0363 ± 0.0000 | 0.0398 ± 0.0000 | 0.0442 ± 0.0000 | 0.0531 ± 0.0000 | 0.0366 ± 0.0000 | 0.5197 ± 0.0000 |
| MLP | Same-IPC Hard | 0.0823 ± 0.0044 | 0.1456 ± 0.0073 | 0.1790 ± 0.0090 | 0.2337 ± 0.0112 | 0.1415 ± 0.0062 | 0.1503 ± 0.0071 | 0.5738 ± 0.0066 |
| LightGCN | Same-IPC Hard | 0.0193 ± 0.0003 | 0.0434 ± 0.0006 | 0.0647 ± 0.0008 | 0.1150 ± 0.0009 | 0.0628 ± 0.0004 | 0.0578 ± 0.0005 | 0.5090 ± 0.0006 |
| NGCF | Same-IPC Hard | 0.1098 ± 0.0004 | 0.1916 ± 0.0003 | 0.2342 ± 0.0003 | 0.3011 ± 0.0001 | 0.1818 ± 0.0003 | 0.1964 ± 0.0002 | 0.5835 ± 0.0003 |
| GraphSAGE | Same-IPC Hard | 0.0421 ± 0.0314 | 0.0747 ± 0.0478 | 0.0961 ± 0.0522 | 0.1409 ± 0.0539 | 0.0865 ± 0.0385 | 0.0840 ± 0.0433 | 0.4729 ± 0.0348 |
| GAT | Same-IPC Hard | 0.0521 ± 0.0076 | 0.0696 ± 0.0056 | 0.0795 ± 0.0049 | 0.1020 ± 0.0045 | 0.0849 ± 0.0062 | 0.0736 ± 0.0058 | 0.5189 ± 0.0072 |
| GraphSAGE+Debias | Pop-Debiased Hard | 0.0666 ± 0.0582 | 0.0886 ± 0.0596 | 0.1068 ± 0.0594 | 0.1492 ± 0.0579 | 0.1059 ± 0.0576 | 0.1002 ± 0.0584 | 0.5123 ± 0.0511 |
| GraphSAGE+logQ | Same-IPC Hard | 0.0994 ± 0.0433 | 0.1175 ± 0.0426 | 0.1307 ± 0.0426 | 0.1615 ± 0.0455 | 0.1331 ± 0.0420 | 0.1251 ± 0.0428 | 0.5396 ± 0.0300 |
| GraphSAGE+DropEdge | Same-IPC Hard | 0.0506 ± 0.0378 | 0.0893 ± 0.0578 | 0.1120 ± 0.0629 | 0.1564 ± 0.0636 | 0.0977 ± 0.0462 | 0.0965 ± 0.0519 | 0.4851 ± 0.0308 |
| GraphSAGE+Time | Same-IPC Hard | 0.0690 ± 0.0364 | 0.1119 ± 0.0498 | 0.1363 ± 0.0536 | 0.1808 ± 0.0539 | 0.1175 ± 0.0415 | 0.1182 ± 0.0453 | 0.4958 ± 0.0433 |
| GraphSAGE+IPS | Same-IPC Hard | 0.1048 ± 0.0378 | 0.1274 ± 0.0260 | 0.1403 ± 0.0207 | 0.1681 ± 0.0128 | 0.1373 ± 0.0292 | 0.1321 ± 0.0258 | 0.4216 ± 0.0092 |
| GAT+Debias | Pop-Debiased Hard | 0.0527 ± 0.0217 | 0.0670 ± 0.0195 | 0.0770 ± 0.0180 | 0.1008 ± 0.0159 | 0.0841 ± 0.0196 | 0.0727 ± 0.0189 | 0.5149 ± 0.0134 |
| GAT+logQ | Same-IPC Hard | 0.0939 ± 0.0002 | 0.1010 ± 0.0008 | 0.1081 ± 0.0012 | 0.1272 ± 0.0020 | 0.1205 ± 0.0006 | 0.1069 ± 0.0009 | 0.5481 ± 0.0024 |
| GAT+DropEdge | Same-IPC Hard | 0.0590 ± 0.0112 | 0.0726 ± 0.0079 | 0.0814 ± 0.0070 | 0.1027 ± 0.0059 | 0.0895 ± 0.0091 | 0.0773 ± 0.0083 | 0.5185 ± 0.0082 |
| GAT+Time | Same-IPC Hard | 0.0693 ± 0.0057 | 0.0799 ± 0.0039 | 0.0881 ± 0.0038 | 0.1084 ± 0.0038 | 0.0980 ± 0.0046 | 0.0852 ± 0.0042 | 0.5214 ± 0.0068 |
| GAT+IPS | Same-IPC Hard | 0.1428 ± 0.0007 | 0.1488 ± 0.0004 | 0.1579 ± 0.0002 | 0.1805 ± 0.0001 | 0.1657 ± 0.0005 | 0.1573 ± 0.0003 | 0.4174 ± 0.0000 |
| GraphSAGE+Debias+IPS | Pop-Debiased Hard | 0.1447 ± 0.0010 | 0.1519 ± 0.0008 | 0.1590 ± 0.0006 | 0.1792 ± 0.0005 | 0.1672 ± 0.0007 | 0.1581 ± 0.0005 | 0.4173 ± 0.0001 |
| GraphSAGE+Time+IPS | Same-IPC Hard | 0.0950 ± 0.0350 | 0.1234 ± 0.0243 | 0.1381 ± 0.0190 | 0.1696 ± 0.0129 | 0.1312 ± 0.0270 | 0.1277 ± 0.0236 | 0.4274 ± 0.0225 |
| GAT+Debias+IPS | Pop-Debiased Hard | 0.1429 ± 0.0003 | 0.1486 ± 0.0003 | 0.1576 ± 0.0003 | 0.1801 ± 0.0002 | 0.1656 ± 0.0003 | 0.1571 ± 0.0002 | 0.4173 ± 0.0001 |
| GAT+Time+IPS | Same-IPC Hard | 0.1429 ± 0.0004 | 0.1488 ± 0.0003 | 0.1579 ± 0.0002 | 0.1805 ± 0.0002 | 0.1657 ± 0.0003 | 0.1573 ± 0.0002 | 0.4174 ± 0.0000 |

*Note: AUC is the rank-AUC for a single positive (ties counted as 0.5, equal to sklearn roc_auc_score). MAP and AP are omitted because they equal MRR exactly under the single-positive protocol.*


## 2. Popularity Bias & Inversion Rate Diagnostics (Table 5)

| Model | Spearman Correlation (ρ) | Hard-Neg Inversion Rate | 
| :--- | :---: | :---: |
| MostPop | 1.0000 ± 0.0000 | 56.3601% ± 0.0000% |
| MostPop-IPC | 0.3402 ± 0.0000 | 58.9251% ± 0.0000% |
| Recency | 0.6933 ± 0.0000 | 50.0481% ± 0.0000% |
| CN | 0.0000 ± 0.0000 | 0.1539% ± 0.0000% |
| AA | 0.0000 ± 0.0000 | 0.1574% ± 0.0000% |
| SVD | 0.0243 ± 0.0000 | 2.3862% ± 0.0000% |
| MLP | 0.1938 ± 0.0216 | 18.9655% ± 1.9807% |
| LightGCN | 0.0005 ± 0.0014 | 49.1378% ± 0.0593% |
| NGCF | 0.9607 ± 0.0091 | 56.3474% ± 0.0550% |
| GraphSAGE | 0.0730 ± 0.0672 | 51.4805% ± 6.0002% |
| GAT | -0.0471 ± 0.0121 | 28.3662% ± 0.8364% |
| GraphSAGE+Debias | 0.0071 ± 0.0484 | 46.6482% ± 6.7177% |
| GraphSAGE+logQ | -0.0611 ± 0.0482 | 30.7996% ± 15.4418% |
| GraphSAGE+DropEdge | 0.0655 ± 0.0638 | 50.6920% ± 5.4536% |
| GraphSAGE+Time | 0.0294 ± 0.1115 | 46.9796% ± 8.2572% |
| GraphSAGE+IPS | -0.9814 ± 0.0166 | 42.3243% ± 1.8984% |
| GAT+Debias | -0.0651 ± 0.0313 | 28.1794% ± 1.2549% |
| GAT+logQ | -0.0246 ± 0.0060 | 26.0039% ± 0.4437% |
| GAT+DropEdge | -0.0495 ± 0.0184 | 28.1192% ± 0.5330% |
| GAT+Time | -0.0433 ± 0.0106 | 27.7193% ± 0.4544% |
| GAT+IPS | -0.9951 ± 0.0023 | 43.4004% ± 0.0065% |
| GraphSAGE+Debias+IPS | -0.9991 ± 0.0003 | 43.4517% ± 0.0146% |
| GraphSAGE+Time+IPS | -0.9755 ± 0.0268 | 41.5049% ± 3.1640% |
| GAT+Debias+IPS | -0.9991 ± 0.0001 | 43.3983% ± 0.0023% |
| GAT+Time+IPS | -0.9947 ± 0.0038 | 43.3941% ± 0.0051% |


### 2.1 GAT Attention Weight Analysis (D12)
- Mann-Whitney U test p-value (Hubs vs Non-Hubs attention weight): **1.0000e+00**
- Hub vs Non-Hub attention weights violin plot saved at `gat_attention_violin.png`

### 2.2 Stratified NDCG@10 (D14)
- GAT performance stratified by popularity across seeds (Option b):
  - **Head**: 0.0164 ± 0.0024
  - **Torso**: 0.0821 ± 0.0063
  - **Tail**: 0.3705 ± 0.0391
- Stratification bar plot saved at `popularity_stratified.png`

### 2.3 Mitigation Sweeps (B4, B5)
- IPS Penalty Beta NDCG@10 (per backbone): GraphSAGE {0.0: 0.0840, 0.5: 0.1175, 1.0: 0.1321, 2.0: 0.1483, 4.0: 0.1567} | GAT {0.0: 0.0736, 0.5: 0.1496, 1.0: 0.1573, 2.0: 0.1573, 4.0: 0.1573}
- Debiased Negative Sampling Alpha NDCG@10 (per backbone): GraphSAGE {0.0: 0.0991, 0.25: 0.0883, 0.5: 0.0623, 0.75: 0.1054, 1.0: 0.1015} | GAT {0.0: 0.0800, 0.25: 0.0612, 0.5: 0.0601, 0.75: 0.0625, 1.0: 0.0817}
- Plots saved to `ips_rerank_sweep.png` and `popularity_debiased_sweep.png` (one curve per backbone)

## 3. Pairwise Statistical Significance

Holm-Bonferroni corrected pairwise comparisons for NDCG@10:

Pre-registered comparison family: **24** pairs (Holm-Bonferroni corrected jointly across exactly these comparisons).

| Comparison Pair | Wilcoxon Raw p | Wilcoxon Adjusted p (Holm) | t-Test Raw p | t-Test Adjusted p (Holm) |
| :--- | :---: | :---: | :---: | :---: |
| GAT vs GraphSAGE | 5.5664e-01 | 1.0000e+00 | 4.9286e-01 | 1.0000e+00 |
| MostPop-IPC vs MostPop | 1.9531e-03 | 4.6875e-02 | 0.0000e+00 | 0.0000e+00 |
| GraphSAGE vs MostPop | 1.9531e-03 | 4.6875e-02 | 2.6479e-05 | 3.7071e-04 |
| GraphSAGE vs MostPop-IPC | 1.9531e-03 | 4.6875e-02 | 2.2680e-05 | 3.4020e-04 |
| GraphSAGE vs SVD | 1.9531e-03 | 4.6875e-02 | 9.4353e-03 | 9.4353e-02 |
| GraphSAGE vs NGCF | 1.9531e-03 | 4.6875e-02 | 2.7366e-05 | 3.7071e-04 |
| GAT vs MostPop | 1.9531e-03 | 4.6875e-02 | 2.7363e-13 | 6.0199e-12 |
| GAT vs MostPop-IPC | 1.9531e-03 | 4.6875e-02 | 2.3318e-13 | 5.3631e-12 |
| GAT vs SVD | 1.9531e-03 | 4.6875e-02 | 1.2731e-08 | 2.1642e-07 |
| GAT vs NGCF | 1.9531e-03 | 4.6875e-02 | 3.1175e-13 | 6.5468e-12 |
| GraphSAGE+Debias vs GraphSAGE | 4.9219e-01 | 1.0000e+00 | 4.6593e-01 | 1.0000e+00 |
| GraphSAGE+logQ vs GraphSAGE | 1.6016e-01 | 1.0000e+00 | 1.0195e-01 | 7.1367e-01 |
| GraphSAGE+DropEdge vs GraphSAGE | 6.9531e-01 | 1.0000e+00 | 5.8370e-01 | 1.0000e+00 |
| GraphSAGE+Time vs GraphSAGE | 1.6016e-01 | 1.0000e+00 | 2.3889e-01 | 1.0000e+00 |
| GraphSAGE+IPS vs GraphSAGE | 8.3984e-02 | 6.7188e-01 | 5.8365e-02 | 4.6692e-01 |
| GraphSAGE+Debias+IPS vs GraphSAGE | 1.9531e-03 | 4.6875e-02 | 6.2434e-04 | 6.8678e-03 |
| GraphSAGE+Time+IPS vs GraphSAGE | 1.3672e-02 | 1.2305e-01 | 1.3878e-02 | 1.2490e-01 |
| GAT+Debias vs GAT | 1.0000e+00 | 1.0000e+00 | 8.7621e-01 | 1.0000e+00 |
| GAT+logQ vs GAT | 1.9531e-03 | 4.6875e-02 | 4.2455e-08 | 6.7928e-07 |
| GAT+DropEdge vs GAT | 3.2227e-01 | 1.0000e+00 | 2.4116e-01 | 1.0000e+00 |
| GAT+Time vs GAT | 1.9531e-03 | 4.6875e-02 | 2.8909e-05 | 3.7071e-04 |
| GAT+IPS vs GAT | 1.9531e-03 | 4.6875e-02 | 8.8840e-12 | 1.6880e-10 |
| GAT+Debias+IPS vs GAT | 1.9531e-03 | 4.6875e-02 | 1.0425e-11 | 1.8766e-10 |
| GAT+Time+IPS vs GAT | 1.9531e-03 | 4.6875e-02 | 8.2082e-12 | 1.6416e-10 |


## 4. Demand Score Comparison (E19)
- **Demand (Original) NDCG@10**: 0.2579 ± 0.0000
- **Demand (Revised) NDCG@10**: 0.2406 ± 0.0000

## 5. Horizon Decay (NEW-1)
NDCG@10 by prediction horizon (months between train cutoff and the test transfer). Plot: `horizon_decay.png`.

| Model | 0-6mo | 6-12mo | 12-18mo | 18mo+ |
| :--- | :---: | :---: | :---: | :---: |
| MostPop | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.1968 (n=220284) |
| MostPop-IPC | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.1990 (n=220284) |
| SVD | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0366 (n=220284) |
| NGCF | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.1964 (n=220284) |
| GraphSAGE | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0840 (n=220284) |
| GAT | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0736 (n=220284) |


## 6. IPC-Section Decomposition (NEW-2)
NDCG@10 split by IPC section (first letter of ipc4); (n) = #test queries in that section.

| Model | A | B | C | D | E | F | G | H |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | 0.200(41938) | 0.153(20589) | 0.194(37319) | 0.124(1191) | 0.121(4527) | 0.114(5922) | 0.214(63534) | 0.212(45264) |
| MostPop-IPC | 0.204(41938) | 0.149(20589) | 0.195(37319) | 0.076(1191) | 0.133(4527) | 0.129(5922) | 0.211(63534) | 0.222(45264) |
| SVD | 0.041(41938) | 0.047(20589) | 0.041(37319) | 0.045(1191) | 0.069(4527) | 0.053(5922) | 0.034(63534) | 0.023(45264) |
| NGCF | 0.203(41938) | 0.153(20589) | 0.194(37319) | 0.127(1191) | 0.118(4527) | 0.113(5922) | 0.213(63534) | 0.210(45264) |
| GraphSAGE | 0.078(41938) | 0.078(20589) | 0.081(37319) | 0.061(1191) | 0.064(4527) | 0.074(5922) | 0.091(63534) | 0.089(45264) |
| GAT | 0.060(41938) | 0.080(20589) | 0.069(37319) | 0.057(1191) | 0.061(4527) | 0.051(5922) | 0.080(63534) | 0.082(45264) |


## 7. Patent-Side Cold-Start (NEW-3)
- Fraction of test patents UNSEEN in training (patent-side cold start): **91.67%**
- NDCG@10 on the (new-patent, seen-company) subset vs all / seen-patent:

| Model | All | New-patent & Seen-company | Seen-patent |
| :--- | :---: | :---: | :---: |
| MostPop | 0.1968 | 0.2120 (n=176381) | 0.3238 |
| MostPop-IPC | 0.1990 | 0.2160 (n=176381) | 0.3127 |
| SVD | 0.0366 | 0.0000 (n=176381) | 0.4391 |
| NGCF | 0.1964 | 0.2124 (n=176381) | 0.3151 |
| GraphSAGE | 0.0840 | 0.0954 (n=176381) | 0.0915 |
| GAT | 0.0736 | 0.0215 (n=176381) | 0.0205 |


## 8. Error-Source Decomposition (NEW-4)
Share of FAILED queries (rank>1) attributable to each cause (priority: popularity mechanism > cold-start > residual).

| Model | Popular-hardneg | Rare/new positive | Semantic residual | #failures |
| :--- | :---: | :---: | :---: | :---: |
| GraphSAGE | 98.8% | 0.1% | 1.1% | 2110199 |
| GAT | 57.8% | 8.3% | 33.9% | 2088033 |


## 9. Qualitative Case Study (NEW-5)
Worst-ranked GAT examples (seed 0): model top-5 companies vs the true buyer.

- Patent `1020160009985` (IPC G06Q): true buyer **(주)메뉴잇** ranked #101.0; GAT top-5 = [조철회, 이광의, 임종철, (주)랭스터, 고명동]
- Patent `1020160010193` (IPC G06T): true buyer **주식회사 스타랩스** ranked #101.0; GAT top-5 = [이석규, 이황수, 백승권, 문정익, 정성관]
- Patent `1020100108597` (IPC G01J): true buyer **주식회사 브릴스** ranked #101.0; GAT top-5 = [채종억, 지동우, 나용진, 김내수, 주식회사엘디티]

## 10. Bootstrap 95% CIs over Test Queries (NEW-12)
Percentile CIs from resampling the per-query ranks (captures query-sampling variance that seed-std omits).

| Model | NDCG@10 [95% CI] | Hits@10 [95% CI] | MRR [95% CI] |
| :--- | :---: | :---: | :---: |
| MostPop | 0.1968 [0.1963, 0.1972] | 0.3018 [0.3012, 0.3024] | 0.1820 [0.1816, 0.1824] |
| MostPop-IPC | 0.1990 [0.1985, 0.1995] | 0.2889 [0.2883, 0.2895] | 0.1840 [0.1835, 0.1844] |
| SVD | 0.0366 [0.0363, 0.0368] | 0.0442 [0.0439, 0.0445] | 0.0531 [0.0529, 0.0534] |
| NGCF | 0.1964 [0.1959, 0.1968] | 0.3011 [0.3005, 0.3017] | 0.1818 [0.1813, 0.1821] |
| GraphSAGE | 0.0840 [0.0837, 0.0843] | 0.1409 [0.1404, 0.1413] | 0.0865 [0.0862, 0.0868] |
| GAT | 0.0735 [0.0732, 0.0738] | 0.1020 [0.1016, 0.1024] | 0.0849 [0.0846, 0.0852] |




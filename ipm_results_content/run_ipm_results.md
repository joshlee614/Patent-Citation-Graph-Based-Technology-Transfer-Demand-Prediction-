# IPM Experiment Evaluation & Diagnostics Report

- **Run Mode**: full (Seeds: 10, Epochs: 50, Candidates: 100)
- **Split**: temporal | **Company features**: content
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
| MLP | Same-IPC Hard | 0.0080 ± 0.0005 | 0.0212 ± 0.0012 | 0.0343 ± 0.0017 | 0.0685 ± 0.0034 | 0.0405 ± 0.0015 | 0.0317 ± 0.0016 | 0.3988 ± 0.0117 |
| LightGCN | Same-IPC Hard | 0.0190 ± 0.0006 | 0.0430 ± 0.0006 | 0.0644 ± 0.0006 | 0.1146 ± 0.0009 | 0.0625 ± 0.0005 | 0.0575 ± 0.0006 | 0.5091 ± 0.0007 |
| NGCF | Same-IPC Hard | 0.1098 ± 0.0004 | 0.1916 ± 0.0004 | 0.2344 ± 0.0003 | 0.3011 ± 0.0002 | 0.1818 ± 0.0003 | 0.1964 ± 0.0003 | 0.5837 ± 0.0004 |
| GraphSAGE | Same-IPC Hard | 0.0230 ± 0.0209 | 0.0496 ± 0.0325 | 0.0711 ± 0.0392 | 0.1191 ± 0.0498 | 0.0664 ± 0.0276 | 0.0623 ± 0.0323 | 0.4809 ± 0.0537 |
| GAT | Same-IPC Hard | 0.0297 ± 0.0275 | 0.0441 ± 0.0253 | 0.0556 ± 0.0241 | 0.0811 ± 0.0226 | 0.0620 ± 0.0251 | 0.0508 ± 0.0246 | 0.5043 ± 0.0241 |
| GraphSAGE+Debias | Pop-Debiased Hard | 0.0858 ± 0.0561 | 0.1042 ± 0.0538 | 0.1181 ± 0.0506 | 0.1505 ± 0.0435 | 0.1204 ± 0.0517 | 0.1124 ± 0.0504 | 0.5203 ± 0.0498 |
| GraphSAGE+logQ | Same-IPC Hard | 0.1083 ± 0.0508 | 0.1224 ± 0.0508 | 0.1333 ± 0.0505 | 0.1590 ± 0.0503 | 0.1390 ± 0.0498 | 0.1291 ± 0.0504 | 0.5286 ± 0.0456 |
| GraphSAGE+DropEdge | Same-IPC Hard | 0.0125 ± 0.0063 | 0.0352 ± 0.0200 | 0.0558 ± 0.0303 | 0.1042 ± 0.0467 | 0.0542 ± 0.0169 | 0.0493 ± 0.0236 | 0.4643 ± 0.0493 |
| GraphSAGE+Time | Same-IPC Hard | 0.0234 ± 0.0177 | 0.0495 ± 0.0330 | 0.0713 ± 0.0444 | 0.1199 ± 0.0645 | 0.0667 ± 0.0276 | 0.0628 ± 0.0357 | 0.4889 ± 0.0570 |
| GraphSAGE+IPS | Same-IPC Hard | 0.1380 ± 0.0114 | 0.1491 ± 0.0055 | 0.1570 ± 0.0035 | 0.1788 ± 0.0010 | 0.1628 ± 0.0077 | 0.1546 ± 0.0060 | 0.4177 ± 0.0008 |
| GAT+Debias | Pop-Debiased Hard | 0.0331 ± 0.0256 | 0.0480 ± 0.0256 | 0.0590 ± 0.0252 | 0.0837 ± 0.0236 | 0.0655 ± 0.0247 | 0.0541 ± 0.0246 | 0.5080 ± 0.0222 |
| GAT+logQ | Same-IPC Hard | 0.0945 ± 0.0006 | 0.1017 ± 0.0013 | 0.1090 ± 0.0015 | 0.1281 ± 0.0024 | 0.1213 ± 0.0011 | 0.1076 ± 0.0014 | 0.5489 ± 0.0033 |
| GAT+DropEdge | Same-IPC Hard | 0.0361 ± 0.0194 | 0.0510 ± 0.0209 | 0.0615 ± 0.0208 | 0.0856 ± 0.0207 | 0.0683 ± 0.0196 | 0.0567 ± 0.0200 | 0.5057 ± 0.0217 |
| GAT+Time | Same-IPC Hard | 0.0440 ± 0.0276 | 0.0573 ± 0.0266 | 0.0672 ± 0.0257 | 0.0909 ± 0.0239 | 0.0754 ± 0.0260 | 0.0633 ± 0.0259 | 0.5150 ± 0.0171 |
| GAT+IPS | Same-IPC Hard | 0.1407 ± 0.0043 | 0.1471 ± 0.0032 | 0.1564 ± 0.0026 | 0.1795 ± 0.0016 | 0.1639 ± 0.0034 | 0.1556 ± 0.0030 | 0.4172 ± 0.0004 |
| GraphSAGE+Debias+IPS | Pop-Debiased Hard | 0.1372 ± 0.0218 | 0.1465 ± 0.0142 | 0.1554 ± 0.0095 | 0.1779 ± 0.0036 | 0.1615 ± 0.0164 | 0.1534 ± 0.0135 | 0.4172 ± 0.0003 |
| GraphSAGE+Time+IPS | Same-IPC Hard | 0.1285 ± 0.0172 | 0.1456 ± 0.0072 | 0.1550 ± 0.0049 | 0.1788 ± 0.0027 | 0.1567 ± 0.0111 | 0.1501 ± 0.0088 | 0.4191 ± 0.0033 |
| GAT+Debias+IPS | Pop-Debiased Hard | 0.1388 ± 0.0090 | 0.1458 ± 0.0059 | 0.1554 ± 0.0042 | 0.1790 ± 0.0021 | 0.1624 ± 0.0068 | 0.1544 ± 0.0058 | 0.4173 ± 0.0002 |
| GAT+Time+IPS | Same-IPC Hard | 0.1429 ± 0.0005 | 0.1487 ± 0.0004 | 0.1577 ± 0.0004 | 0.1804 ± 0.0002 | 0.1657 ± 0.0004 | 0.1573 ± 0.0003 | 0.4174 ± 0.0001 |

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
| MLP | -0.1924 ± 0.0897 | 55.4079% ± 0.6747% |
| LightGCN | 0.0002 ± 0.0013 | 49.1223% ± 0.0698% |
| NGCF | 0.9579 ± 0.0054 | 56.3180% ± 0.0423% |
| GraphSAGE | 0.1655 ± 0.2069 | 53.6704% ± 6.0500% |
| GAT | -0.0929 ± 0.0560 | 28.6470% ± 1.5396% |
| GraphSAGE+Debias | -0.0134 ± 0.0733 | 39.3771% ± 15.7525% |
| GraphSAGE+logQ | -0.0667 ± 0.0626 | 36.5935% ± 12.1358% |
| GraphSAGE+DropEdge | 0.1898 ± 0.2080 | 56.5164% ± 2.0163% |
| GraphSAGE+Time | 0.2514 ± 0.2672 | 51.3512% ± 15.3103% |
| GraphSAGE+IPS | -0.9929 ± 0.0099 | 43.4052% ± 0.0665% |
| GAT+Debias | -0.0407 ± 0.0441 | 29.1375% ± 2.1093% |
| GAT+logQ | -0.0180 ± 0.0088 | 26.1371% ± 0.6428% |
| GAT+DropEdge | -0.0413 ± 0.0400 | 29.9050% ± 2.0570% |
| GAT+Time | -0.0362 ± 0.0221 | 29.0793% ± 2.1228% |
| GAT+IPS | -0.9980 ± 0.0018 | 43.3881% ± 0.0166% |
| GraphSAGE+Debias+IPS | -0.9963 ± 0.0083 | 43.3834% ± 0.1457% |
| GraphSAGE+Time+IPS | -0.9754 ± 0.0335 | 43.3451% ± 0.1401% |
| GAT+Debias+IPS | -0.9956 ± 0.0073 | 43.3840% ± 0.0238% |
| GAT+Time+IPS | -0.9975 ± 0.0018 | 43.3947% ± 0.0040% |


### 2.1 GAT Attention Weight Analysis (D12)
- Mann-Whitney U test p-value (Hubs vs Non-Hubs attention weight): **1.0000e+00**
- Hub vs Non-Hub attention weights violin plot saved at `gat_attention_violin.png`

### 2.2 Stratified NDCG@10 (D14)
- GAT performance stratified by popularity across seeds (Option b):
  - **Head**: 0.0146 ± 0.0018
  - **Torso**: 0.0792 ± 0.0046
  - **Tail**: 0.2294 ± 0.1622
- Stratification bar plot saved at `popularity_stratified.png`

### 2.3 Mitigation Sweeps (B4, B5)
- IPS Penalty Beta NDCG@10 (per backbone): GraphSAGE {0.0: 0.0623, 0.5: 0.1419, 1.0: 0.1546, 2.0: 0.1579, 4.0: 0.1582} | GAT {0.0: 0.0508, 0.5: 0.1531, 1.0: 0.1556, 2.0: 0.1568, 4.0: 0.1568}
- Debiased Negative Sampling Alpha NDCG@10 (per backbone): GraphSAGE {0.0: 0.0538, 0.25: 0.0482, 0.5: 0.0522, 0.75: 0.0781, 1.0: 0.0976} | GAT {0.0: 0.0570, 0.25: 0.0513, 0.5: 0.0469, 0.75: 0.0638, 1.0: 0.0802}
- Plots saved to `ips_rerank_sweep.png` and `popularity_debiased_sweep.png` (one curve per backbone)

## 3. Pairwise Statistical Significance

Holm-Bonferroni corrected pairwise comparisons for NDCG@10:

Pre-registered comparison family: **24** pairs (Holm-Bonferroni corrected jointly across exactly these comparisons).

| Comparison Pair | Wilcoxon Raw p | Wilcoxon Adjusted p (Holm) | t-Test Raw p | t-Test Adjusted p (Holm) |
| :--- | :---: | :---: | :---: | :---: |
| GAT vs GraphSAGE | 6.2500e-01 | 1.0000e+00 | 4.3041e-01 | 1.0000e+00 |
| MostPop-IPC vs MostPop | 1.9531e-03 | 4.6875e-02 | 0.0000e+00 | 0.0000e+00 |
| GraphSAGE vs MostPop | 1.9531e-03 | 4.6875e-02 | 5.4770e-07 | 8.7632e-06 |
| GraphSAGE vs MostPop-IPC | 1.9531e-03 | 4.6875e-02 | 4.7623e-07 | 8.0958e-06 |
| GraphSAGE vs SVD | 9.7656e-03 | 9.7656e-02 | 4.0527e-02 | 3.6474e-01 |
| GraphSAGE vs NGCF | 1.9531e-03 | 4.6875e-02 | 5.7942e-07 | 8.7632e-06 |
| GAT vs MostPop | 1.9531e-03 | 4.6875e-02 | 2.5251e-08 | 5.5553e-07 |
| GAT vs MostPop-IPC | 1.9531e-03 | 4.6875e-02 | 2.2126e-08 | 5.0890e-07 |
| GAT vs SVD | 2.7539e-01 | 1.0000e+00 | 1.1771e-01 | 8.2399e-01 |
| GAT vs NGCF | 1.9531e-03 | 4.6875e-02 | 2.5854e-08 | 5.5553e-07 |
| GraphSAGE+Debias vs GraphSAGE | 4.8828e-02 | 3.9062e-01 | 4.8985e-02 | 3.9188e-01 |
| GraphSAGE+logQ vs GraphSAGE | 1.9531e-02 | 1.7578e-01 | 8.3208e-03 | 8.3208e-02 |
| GraphSAGE+DropEdge vs GraphSAGE | 4.9219e-01 | 1.0000e+00 | 4.2536e-01 | 1.0000e+00 |
| GraphSAGE+Time vs GraphSAGE | 6.2500e-01 | 1.0000e+00 | 9.7610e-01 | 1.0000e+00 |
| GraphSAGE+IPS vs GraphSAGE | 1.9531e-03 | 4.6875e-02 | 2.0136e-05 | 2.6177e-04 |
| GraphSAGE+Debias+IPS vs GraphSAGE | 3.9062e-03 | 4.6875e-02 | 1.2086e-04 | 1.3295e-03 |
| GraphSAGE+Time+IPS vs GraphSAGE | 1.9531e-03 | 4.6875e-02 | 1.7984e-05 | 2.5178e-04 |
| GAT+Debias vs GAT | 6.9531e-01 | 1.0000e+00 | 8.2170e-01 | 1.0000e+00 |
| GAT+logQ vs GAT | 1.9531e-03 | 4.6875e-02 | 6.4036e-05 | 7.6843e-04 |
| GAT+DropEdge vs GAT | 6.2500e-01 | 1.0000e+00 | 6.0759e-01 | 1.0000e+00 |
| GAT+Time vs GAT | 3.2227e-01 | 1.0000e+00 | 3.2284e-01 | 1.0000e+00 |
| GAT+IPS vs GAT | 1.9531e-03 | 4.6875e-02 | 2.8015e-07 | 5.3229e-06 |
| GAT+Debias+IPS vs GAT | 1.9531e-03 | 4.6875e-02 | 2.4386e-07 | 4.8771e-06 |
| GAT+Time+IPS vs GAT | 1.9531e-03 | 4.6875e-02 | 3.9389e-07 | 7.0900e-06 |


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
| GraphSAGE | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0623 (n=220284) |
| GAT | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0508 (n=220284) |


## 6. IPC-Section Decomposition (NEW-2)
NDCG@10 split by IPC section (first letter of ipc4); (n) = #test queries in that section.

| Model | A | B | C | D | E | F | G | H |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | 0.200(41938) | 0.153(20589) | 0.194(37319) | 0.124(1191) | 0.121(4527) | 0.114(5922) | 0.214(63534) | 0.212(45264) |
| MostPop-IPC | 0.204(41938) | 0.149(20589) | 0.195(37319) | 0.076(1191) | 0.133(4527) | 0.129(5922) | 0.211(63534) | 0.222(45264) |
| SVD | 0.041(41938) | 0.047(20589) | 0.041(37319) | 0.045(1191) | 0.069(4527) | 0.053(5922) | 0.034(63534) | 0.023(45264) |
| NGCF | 0.202(41938) | 0.153(20589) | 0.194(37319) | 0.127(1191) | 0.119(4527) | 0.112(5922) | 0.213(63534) | 0.209(45264) |
| GraphSAGE | 0.072(41938) | 0.065(20589) | 0.072(37319) | 0.064(1191) | 0.073(4527) | 0.067(5922) | 0.056(63534) | 0.052(45264) |
| GAT | 0.042(41938) | 0.060(20589) | 0.044(37319) | 0.051(1191) | 0.058(4527) | 0.044(5922) | 0.052(63534) | 0.059(45264) |


## 7. Patent-Side Cold-Start (NEW-3)
- Fraction of test patents UNSEEN in training (patent-side cold start): **91.67%**
- NDCG@10 on the (new-patent, seen-company) subset vs all / seen-patent:

| Model | All | New-patent & Seen-company | Seen-patent |
| :--- | :---: | :---: | :---: |
| MostPop | 0.1968 | 0.2120 (n=176381) | 0.3238 |
| MostPop-IPC | 0.1990 | 0.2160 (n=176381) | 0.3127 |
| SVD | 0.0366 | 0.0000 (n=176381) | 0.4391 |
| NGCF | 0.1964 | 0.2124 (n=176381) | 0.3156 |
| GraphSAGE | 0.0623 | 0.0556 (n=176381) | 0.0687 |
| GAT | 0.0508 | 0.0196 (n=176381) | 0.0517 |


## 8. Error-Source Decomposition (NEW-4)
Share of FAILED queries (rank>1) attributable to each cause (priority: popularity mechanism > cold-start > residual).

| Model | Popular-hardneg | Rare/new positive | Semantic residual | #failures |
| :--- | :---: | :---: | :---: | :---: |
| GraphSAGE | 98.1% | 0.8% | 1.1% | 2152251 |
| GAT | 58.7% | 8.0% | 33.2% | 2137505 |


## 9. Qualitative Case Study (NEW-5)
Worst-ranked GAT examples (seed 0): model top-5 companies vs the true buyer.

- Patent `1020210012367` (IPC A61F): true buyer **구영준** ranked #101.0; GAT top-5 = [김동업, 최다미, 한국신발피혁연구원, 최 병 택, 윤혜원]
- Patent `1020160009985` (IPC G06Q): true buyer **(주)메뉴잇** ranked #101.0; GAT top-5 = [강충한, 권소은, 안기중, 홍은재, 김바라]
- Patent `1020160010193` (IPC G06T): true buyer **주식회사 스타랩스** ranked #101.0; GAT top-5 = [이석규, 이황수, 백승권, 정기현, 문정익]

## 10. Bootstrap 95% CIs over Test Queries (NEW-12)
Percentile CIs from resampling the per-query ranks (captures query-sampling variance that seed-std omits).

| Model | NDCG@10 [95% CI] | Hits@10 [95% CI] | MRR [95% CI] |
| :--- | :---: | :---: | :---: |
| MostPop | 0.1968 [0.1963, 0.1972] | 0.3018 [0.3012, 0.3024] | 0.1820 [0.1816, 0.1824] |
| MostPop-IPC | 0.1990 [0.1985, 0.1995] | 0.2889 [0.2883, 0.2895] | 0.1840 [0.1835, 0.1844] |
| SVD | 0.0366 [0.0363, 0.0368] | 0.0442 [0.0439, 0.0445] | 0.0531 [0.0529, 0.0534] |
| NGCF | 0.1964 [0.1959, 0.1968] | 0.3011 [0.3005, 0.3017] | 0.1818 [0.1813, 0.1822] |
| GraphSAGE | 0.0623 [0.0620, 0.0626] | 0.1191 [0.1187, 0.1196] | 0.0664 [0.0662, 0.0666] |
| GAT | 0.0508 [0.0505, 0.0510] | 0.0811 [0.0807, 0.0814] | 0.0620 [0.0618, 0.0622] |




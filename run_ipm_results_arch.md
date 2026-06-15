# IPM Experiment Evaluation & Diagnostics Report

- **Run Mode**: full (Seeds: 1, Epochs: 50, Candidates: 100)
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
| MLP | Same-IPC Hard | 0.0806 ± 0.0000 | 0.1404 ± 0.0000 | 0.1741 ± 0.0000 | 0.2312 ± 0.0000 | 0.1379 ± 0.0000 | 0.1473 ± 0.0000 | 0.5727 ± 0.0000 |
| LightGCN | Same-IPC Hard | 0.0193 ± 0.0000 | 0.0441 ± 0.0000 | 0.0658 ± 0.0000 | 0.1159 ± 0.0000 | 0.0631 ± 0.0000 | 0.0583 ± 0.0000 | 0.5088 ± 0.0000 |
| NGCF | Same-IPC Hard | 0.1098 ± 0.0000 | 0.1914 ± 0.0000 | 0.2342 ± 0.0000 | 0.3012 ± 0.0000 | 0.1817 ± 0.0000 | 0.1963 ± 0.0000 | 0.5841 ± 0.0000 |
| GraphSAGE | Same-IPC Hard | 0.0930 ± 0.0000 | 0.1521 ± 0.0000 | 0.1776 ± 0.0000 | 0.2157 ± 0.0000 | 0.1466 ± 0.0000 | 0.1502 ± 0.0000 | 0.4922 ± 0.0000 |
| GAT | Same-IPC Hard | 0.0403 ± 0.0000 | 0.0643 ± 0.0000 | 0.0763 ± 0.0000 | 0.1011 ± 0.0000 | 0.0772 ± 0.0000 | 0.0672 ± 0.0000 | 0.5265 ± 0.0000 |
| GraphSAGE+Debias | Pop-Debiased Hard | 0.0420 ± 0.0000 | 0.0848 ± 0.0000 | 0.1160 ± 0.0000 | 0.1772 ± 0.0000 | 0.0965 ± 0.0000 | 0.0989 ± 0.0000 | 0.5517 ± 0.0000 |
| GraphSAGE+logQ | Same-IPC Hard | 0.0995 ± 0.0000 | 0.1350 ± 0.0000 | 0.1566 ± 0.0000 | 0.2066 ± 0.0000 | 0.1459 ± 0.0000 | 0.1450 ± 0.0000 | 0.5594 ± 0.0000 |
| GraphSAGE+DropEdge | Same-IPC Hard | 0.0910 ± 0.0000 | 0.1541 ± 0.0000 | 0.1854 ± 0.0000 | 0.2359 ± 0.0000 | 0.1501 ± 0.0000 | 0.1568 ± 0.0000 | 0.5356 ± 0.0000 |
| GraphSAGE+Time | Same-IPC Hard | 0.0143 ± 0.0000 | 0.0334 ± 0.0000 | 0.0509 ± 0.0000 | 0.0946 ± 0.0000 | 0.0528 ± 0.0000 | 0.0463 ± 0.0000 | 0.4426 ± 0.0000 |
| GraphSAGE+IPS | Same-IPC Hard | 0.0493 ± 0.0000 | 0.0803 ± 0.0000 | 0.0991 ± 0.0000 | 0.1396 ± 0.0000 | 0.0904 ± 0.0000 | 0.0880 ± 0.0000 | 0.4386 ± 0.0000 |
| GAT+Debias | Pop-Debiased Hard | 0.0761 ± 0.0000 | 0.0866 ± 0.0000 | 0.0955 ± 0.0000 | 0.1186 ± 0.0000 | 0.1057 ± 0.0000 | 0.0931 ± 0.0000 | 0.5414 ± 0.0000 |
| GAT+logQ | Same-IPC Hard | 0.0939 ± 0.0000 | 0.1006 ± 0.0000 | 0.1076 ± 0.0000 | 0.1257 ± 0.0000 | 0.1202 ± 0.0000 | 0.1063 ± 0.0000 | 0.5497 ± 0.0000 |
| GAT+DropEdge | Same-IPC Hard | 0.0485 ± 0.0000 | 0.0667 ± 0.0000 | 0.0782 ± 0.0000 | 0.1020 ± 0.0000 | 0.0818 ± 0.0000 | 0.0714 ± 0.0000 | 0.5078 ± 0.0000 |
| GAT+Time | Same-IPC Hard | 0.0543 ± 0.0000 | 0.0667 ± 0.0000 | 0.0758 ± 0.0000 | 0.0968 ± 0.0000 | 0.0835 ± 0.0000 | 0.0719 ± 0.0000 | 0.4964 ± 0.0000 |
| GAT+IPS | Same-IPC Hard | 0.1422 ± 0.0000 | 0.1483 ± 0.0000 | 0.1575 ± 0.0000 | 0.1806 ± 0.0000 | 0.1651 ± 0.0000 | 0.1569 ± 0.0000 | 0.4174 ± 0.0000 |
| GraphSAGE+Debias+IPS | Pop-Debiased Hard | 0.1444 ± 0.0000 | 0.1521 ± 0.0000 | 0.1594 ± 0.0000 | 0.1791 ± 0.0000 | 0.1671 ± 0.0000 | 0.1580 ± 0.0000 | 0.4173 ± 0.0000 |
| GraphSAGE+Time+IPS | Same-IPC Hard | 0.1456 ± 0.0000 | 0.1527 ± 0.0000 | 0.1594 ± 0.0000 | 0.1787 ± 0.0000 | 0.1679 ± 0.0000 | 0.1585 ± 0.0000 | 0.4173 ± 0.0000 |
| GAT+Debias+IPS | Pop-Debiased Hard | 0.1428 ± 0.0000 | 0.1484 ± 0.0000 | 0.1574 ± 0.0000 | 0.1801 ± 0.0000 | 0.1655 ± 0.0000 | 0.1570 ± 0.0000 | 0.4173 ± 0.0000 |
| GAT+Time+IPS | Same-IPC Hard | 0.1422 ± 0.0000 | 0.1481 ± 0.0000 | 0.1568 ± 0.0000 | 0.1796 ± 0.0000 | 0.1650 ± 0.0000 | 0.1565 ± 0.0000 | 0.4171 ± 0.0000 |

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
| MLP | 0.1733 ± 0.0000 | 16.4757% ± 0.0000% |
| LightGCN | 0.0007 ± 0.0000 | 49.1561% ± 0.0000% |
| NGCF | 0.9483 ± 0.0000 | 56.2776% ± 0.0000% |
| GraphSAGE | -0.0661 ± 0.0000 | 47.2258% ± 0.0000% |
| GAT | -0.0276 ± 0.0000 | 29.3163% ± 0.0000% |
| GraphSAGE+Debias | -0.0546 ± 0.0000 | 43.5761% ± 0.0000% |
| GraphSAGE+logQ | -0.0687 ± 0.0000 | 32.3468% ± 0.0000% |
| GraphSAGE+DropEdge | 0.1506 ± 0.0000 | 43.7823% ± 0.0000% |
| GraphSAGE+Time | 0.0752 ± 0.0000 | 56.8498% ± 0.0000% |
| GraphSAGE+IPS | -0.9548 ± 0.0000 | 38.2931% ± 0.0000% |
| GAT+Debias | -0.0078 ± 0.0000 | 27.3111% ± 0.0000% |
| GAT+logQ | -0.0173 ± 0.0000 | 26.2976% ± 0.0000% |
| GAT+DropEdge | -0.0803 ± 0.0000 | 28.3771% ± 0.0000% |
| GAT+Time | -0.0783 ± 0.0000 | 28.3878% ± 0.0000% |
| GAT+IPS | -0.9965 ± 0.0000 | 43.3943% ± 0.0000% |
| GraphSAGE+Debias+IPS | -0.9992 ± 0.0000 | 43.4621% ± 0.0000% |
| GraphSAGE+Time+IPS | -0.9978 ± 0.0000 | 43.4582% ± 0.0000% |
| GAT+Debias+IPS | -0.9991 ± 0.0000 | 43.3960% ± 0.0000% |
| GAT+Time+IPS | -0.9967 ± 0.0000 | 43.4074% ± 0.0000% |


### 2.1 GAT Attention Weight Analysis (D12)
- Mann-Whitney U test p-value (Hubs vs Non-Hubs attention weight): **1.0000e+00**
- Hub vs Non-Hub attention weights violin plot saved at `gat_attention_violin.png`

### 2.2 Stratified NDCG@10 (D14)
- GAT performance stratified by popularity across seeds (Option b):
  - **Head**: 0.0165 ± 0.0000
  - **Torso**: 0.0665 ± 0.0000
  - **Tail**: 0.3334 ± 0.0000
- Stratification bar plot saved at `popularity_stratified.png`

### 2.3 Mitigation Sweeps (B4, B5)
- IPS Penalty Beta NDCG@10 (per backbone): GraphSAGE {0.0: 0.1502, 0.5: 0.0944, 1.0: 0.0880, 2.0: 0.1273, 4.0: 0.1537} | GAT {0.0: 0.0672, 0.5: 0.1475, 1.0: 0.1569, 2.0: 0.1570, 4.0: 0.1570}
- Debiased Negative Sampling Alpha NDCG@10 (per backbone): GraphSAGE {0.0: 0.1456, 0.25: 0.0448, 0.5: 0.0420, 0.75: 0.1446, 1.0: 0.0970} | GAT {0.0: 0.0809, 0.25: 0.0438, 0.5: 0.0774, 0.75: 0.0777, 1.0: 0.0622}
- Plots saved to `ips_rerank_sweep.png` and `popularity_debiased_sweep.png` (one curve per backbone)

## 3. Pairwise Statistical Significance

Holm-Bonferroni corrected pairwise comparisons for NDCG@10:

> **WARNING**: only 1 seeds (<6). Wilcoxon signed-rank is undefined/underpowered, so ALL p-values below are forced to 1.0 and are NOT real non-results. Re-run with --seeds >= 6 (full mode uses 10).

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
- **Demand (Original) NDCG@10**: 0.3281 ± 0.0000
- **Demand (Revised) NDCG@10**: 0.2565 ± 0.0000

## 5. Horizon Decay (NEW-1)
NDCG@10 by prediction horizon (months between train cutoff and the test transfer). Plot: `horizon_decay.png`.

| Model | 0-6mo | 6-12mo | 12-18mo | 18mo+ |
| :--- | :---: | :---: | :---: | :---: |
| MostPop | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.1968 (n=220284) |
| MostPop-IPC | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.1990 (n=220284) |
| SVD | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0366 (n=220284) |
| NGCF | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.1963 (n=220284) |
| GraphSAGE | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.1502 (n=220284) |
| GAT | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0672 (n=220284) |


## 6. IPC-Section Decomposition (NEW-2)
NDCG@10 split by IPC section (first letter of ipc4); (n) = #test queries in that section.

| Model | A | B | C | D | E | F | G | H |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | 0.200(41938) | 0.153(20589) | 0.194(37319) | 0.124(1191) | 0.121(4527) | 0.114(5922) | 0.214(63534) | 0.212(45264) |
| MostPop-IPC | 0.204(41938) | 0.149(20589) | 0.195(37319) | 0.076(1191) | 0.133(4527) | 0.129(5922) | 0.211(63534) | 0.222(45264) |
| SVD | 0.041(41938) | 0.047(20589) | 0.041(37319) | 0.045(1191) | 0.069(4527) | 0.053(5922) | 0.034(63534) | 0.023(45264) |
| NGCF | 0.202(41938) | 0.152(20589) | 0.195(37319) | 0.127(1191) | 0.119(4527) | 0.112(5922) | 0.213(63534) | 0.210(45264) |
| GraphSAGE | 0.128(41938) | 0.129(20589) | 0.150(37319) | 0.103(1191) | 0.103(4527) | 0.104(5922) | 0.168(63534) | 0.168(45264) |
| GAT | 0.079(41938) | 0.078(20589) | 0.073(37319) | 0.045(1191) | 0.064(4527) | 0.050(5922) | 0.066(63534) | 0.052(45264) |


## 7. Patent-Side Cold-Start (NEW-3)
- Fraction of test patents UNSEEN in training (patent-side cold start): **91.67%**
- NDCG@10 on the (new-patent, seen-company) subset vs all / seen-patent:

| Model | All | New-patent & Seen-company | Seen-patent |
| :--- | :---: | :---: | :---: |
| MostPop | 0.1968 | 0.2120 (n=176381) | 0.3238 |
| MostPop-IPC | 0.1990 | 0.2160 (n=176381) | 0.3127 |
| SVD | 0.0366 | 0.0000 (n=176381) | 0.4391 |
| NGCF | 0.1963 | 0.2125 (n=176381) | 0.3148 |
| GraphSAGE | 0.1502 | 0.1619 (n=176381) | 0.2472 |
| GAT | 0.0672 | 0.0201 (n=176381) | 0.0200 |


## 8. Error-Source Decomposition (NEW-4)
Share of FAILED queries (rank>1) attributable to each cause (priority: popularity mechanism > cold-start > residual).

| Model | Popular-hardneg | Rare/new positive | Semantic residual | #failures |
| :--- | :---: | :---: | :---: | :---: |
| GraphSAGE | 99.2% | 0.1% | 0.7% | 199793 |
| GAT | 57.9% | 8.7% | 33.4% | 211410 |


## 9. Qualitative Case Study (NEW-5)
Worst-ranked GAT examples (seed 0): model top-5 companies vs the true buyer.

- Patent `1020160009985` (IPC G06Q): true buyer **(주)메뉴잇** ranked #101.0; GAT top-5 = [조철회, 이광의, 임종철, 고명동, (주)랭스터]
- Patent `1020160010193` (IPC G06T): true buyer **주식회사 스타랩스** ranked #101.0; GAT top-5 = [이석규, 이황수, 백승권, 문정익, 정기현]
- Patent `1020100108597` (IPC G01J): true buyer **주식회사 브릴스** ranked #101.0; GAT top-5 = [채종억, 지동우, 나용진, 김내수, 주식회사엘디티]

## 10. Bootstrap 95% CIs over Test Queries (NEW-12)
Percentile CIs from resampling the per-query ranks (captures query-sampling variance that seed-std omits).

| Model | NDCG@10 [95% CI] | Hits@10 [95% CI] | MRR [95% CI] |
| :--- | :---: | :---: | :---: |
| MostPop | 0.1968 [0.1955, 0.1982] | 0.3018 [0.3000, 0.3038] | 0.1821 [0.1809, 0.1833] |
| MostPop-IPC | 0.1990 [0.1976, 0.2004] | 0.2889 [0.2870, 0.2909] | 0.1840 [0.1827, 0.1853] |
| SVD | 0.0366 [0.0359, 0.0373] | 0.0442 [0.0433, 0.0450] | 0.0531 [0.0524, 0.0538] |
| NGCF | 0.1964 [0.1951, 0.1978] | 0.3012 [0.2993, 0.3032] | 0.1818 [0.1805, 0.1831] |
| GraphSAGE | 0.1503 [0.1489, 0.1516] | 0.2158 [0.2140, 0.2175] | 0.1467 [0.1454, 0.1479] |
| GAT | 0.0671 [0.0663, 0.0680] | 0.1010 [0.0998, 0.1023] | 0.0772 [0.0764, 0.0780] |




## 13. GNN Architecture / Hyperparameter Sensitivity (E20)
NDCG@10 for GraphSAGE and GAT retrained over a grid of embedding dimension (hidden/out) and learning rate (seed 0). The best configuration reaches NDCG@10 = 0.1495, still far below the MostPop baseline (0.197): the failure is not an artifact of an under-sized or under-tuned model.

| Backbone | hidden/out | lr=0.005 | lr=0.01 | lr=0.02 |
| :--- | :---: | :---: | :---: | :---: |
| GraphSAGE | 16/8 | 0.0379 | 0.0496 | 0.1487 |
| GraphSAGE | 32/16 | 0.1089 | 0.0704 | 0.0366 |
| GraphSAGE | 64/32 | 0.0422 | 0.0472 | 0.1495 |
| GraphSAGE | 128/64 | 0.0477 | 0.0547 | 0.0467 |
| GAT | 16/8 | 0.0739 | 0.0835 | 0.0855 |
| GAT | 32/16 | 0.0769 | 0.0704 | 0.0862 |
| GAT | 64/32 | 0.0795 | 0.0943 | 0.1075 |
| GAT | 128/64 | 0.0794 | 0.1063 | 0.1099 |


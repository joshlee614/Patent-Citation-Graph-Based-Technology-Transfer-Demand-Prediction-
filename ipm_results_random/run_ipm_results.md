# IPM Experiment Evaluation & Diagnostics Report

- **Run Mode**: full (Seeds: 10, Epochs: 50, Candidates: 100)
- **Split**: random | **Company features**: random
- **Device**: cpu
- **Average Candidate Padding Rate**: 1.12%
- **Cold-Start Statistics (Test Set)**:
  - Unseen Companies (frac_unseen): 4.20%
  - Rare Companies (frac_rare, <= 1 train transfer): 7.59%

## 1. Main Quantitative Results (Table 4)

| Model Architecture | Negative Sampling | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | NDCG@10 | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | - | 0.1314 ± 0.0000 | 0.2123 ± 0.0000 | 0.2549 ± 0.0000 | 0.3273 ± 0.0000 | 0.2060 ± 0.0000 | 0.2192 ± 0.0000 | 0.6669 ± 0.0000 |
| MostPop-IPC | - | 0.1763 ± 0.0000 | 0.2804 ± 0.0000 | 0.3379 ± 0.0000 | 0.4253 ± 0.0000 | 0.2627 ± 0.0000 | 0.2903 ± 0.0000 | 0.6044 ± 0.0000 |
| Recency | - | 0.0310 ± 0.0000 | 0.1256 ± 0.0000 | 0.1875 ± 0.0000 | 0.2716 ± 0.0000 | 0.1232 ± 0.0000 | 0.1427 ± 0.0000 | 0.6151 ± 0.0000 |
| CN | - | 0.1044 ± 0.0000 | 0.1209 ± 0.0000 | 0.1215 ± 0.0000 | 0.1217 ± 0.0000 | 0.1318 ± 0.0000 | 0.1164 ± 0.0000 | 0.5602 ± 0.0000 |
| AA | - | 0.1073 ± 0.0000 | 0.1208 ± 0.0000 | 0.1214 ± 0.0000 | 0.1217 ± 0.0000 | 0.1323 ± 0.0000 | 0.1168 ± 0.0000 | 0.5602 ± 0.0000 |
| SVD | Same-IPC Hard | 0.2930 ± 0.0000 | 0.4326 ± 0.0000 | 0.5111 ± 0.0000 | 0.6222 ± 0.0000 | 0.3997 ± 0.0000 | 0.4420 ± 0.0000 | 0.8297 ± 0.0000 |
| MLP | Same-IPC Hard | 0.1062 ± 0.0045 | 0.1688 ± 0.0059 | 0.2011 ± 0.0064 | 0.2550 ± 0.0069 | 0.1641 ± 0.0053 | 0.1732 ± 0.0055 | 0.5864 ± 0.0040 |
| LightGCN | Same-IPC Hard | 0.0331 ± 0.0005 | 0.0679 ± 0.0008 | 0.0971 ± 0.0011 | 0.1611 ± 0.0012 | 0.0851 ± 0.0005 | 0.0854 ± 0.0007 | 0.5539 ± 0.0007 |
| NGCF | Same-IPC Hard | 0.1294 ± 0.0002 | 0.2089 ± 0.0003 | 0.2517 ± 0.0002 | 0.3233 ± 0.0003 | 0.2032 ± 0.0001 | 0.2162 ± 0.0002 | 0.6632 ± 0.0002 |
| GraphSAGE | Same-IPC Hard | 0.0647 ± 0.0453 | 0.1067 ± 0.0636 | 0.1324 ± 0.0691 | 0.1853 ± 0.0726 | 0.1166 ± 0.0535 | 0.1164 ± 0.0594 | 0.5595 ± 0.0390 |
| GAT | Same-IPC Hard | 0.0778 ± 0.0034 | 0.1504 ± 0.0041 | 0.1961 ± 0.0038 | 0.2783 ± 0.0029 | 0.1516 ± 0.0031 | 0.1648 ± 0.0029 | 0.6502 ± 0.0016 |
| GraphSAGE+Debias | Pop-Debiased Hard | 0.0284 ± 0.0156 | 0.0480 ± 0.0143 | 0.0668 ± 0.0141 | 0.1143 ± 0.0150 | 0.0692 ± 0.0145 | 0.0624 ± 0.0140 | 0.5175 ± 0.0162 |
| GraphSAGE+logQ | Same-IPC Hard | 0.0339 ± 0.0127 | 0.0486 ± 0.0142 | 0.0617 ± 0.0184 | 0.0947 ± 0.0321 | 0.0682 ± 0.0157 | 0.0582 ± 0.0183 | 0.5114 ± 0.0088 |
| GraphSAGE+DropEdge | Same-IPC Hard | 0.0231 ± 0.0322 | 0.0483 ± 0.0445 | 0.0700 ± 0.0473 | 0.1230 ± 0.0470 | 0.0682 ± 0.0369 | 0.0633 ± 0.0405 | 0.5268 ± 0.0256 |
| GraphSAGE+Time | Same-IPC Hard | 0.0569 ± 0.0434 | 0.0921 ± 0.0614 | 0.1159 ± 0.0658 | 0.1676 ± 0.0669 | 0.1056 ± 0.0506 | 0.1035 ± 0.0560 | 0.5502 ± 0.0348 |
| GraphSAGE+IPS | Same-IPC Hard | 0.0332 ± 0.0125 | 0.0450 ± 0.0071 | 0.0534 ± 0.0047 | 0.0731 ± 0.0018 | 0.0596 ± 0.0086 | 0.0497 ± 0.0071 | 0.3475 ± 0.0168 |
| GAT+Debias | Pop-Debiased Hard | 0.0223 ± 0.0123 | 0.0437 ± 0.0134 | 0.0637 ± 0.0166 | 0.1139 ± 0.0270 | 0.0658 ± 0.0127 | 0.0586 ± 0.0149 | 0.5448 ± 0.0239 |
| GAT+logQ | Same-IPC Hard | 0.0498 ± 0.0074 | 0.0741 ± 0.0354 | 0.0933 ± 0.0508 | 0.1379 ± 0.0761 | 0.0922 ± 0.0262 | 0.0858 ± 0.0380 | 0.5509 ± 0.0562 |
| GAT+DropEdge | Same-IPC Hard | 0.0706 ± 0.0214 | 0.1366 ± 0.0385 | 0.1797 ± 0.0475 | 0.2590 ± 0.0583 | 0.1407 ± 0.0315 | 0.1518 ± 0.0385 | 0.6373 ± 0.0397 |
| GAT+Time | Same-IPC Hard | 0.0509 ± 0.0325 | 0.0993 ± 0.0597 | 0.1330 ± 0.0746 | 0.2005 ± 0.0941 | 0.1103 ± 0.0493 | 0.1141 ± 0.0607 | 0.5962 ± 0.0672 |
| GAT+IPS | Same-IPC Hard | 0.0221 ± 0.0018 | 0.0495 ± 0.0014 | 0.0663 ± 0.0008 | 0.1010 ± 0.0010 | 0.0605 ± 0.0011 | 0.0559 ± 0.0009 | 0.3968 ± 0.0063 |
| GraphSAGE+Debias+IPS | Pop-Debiased Hard | 0.0446 ± 0.0017 | 0.0511 ± 0.0013 | 0.0572 ± 0.0010 | 0.0737 ± 0.0005 | 0.0673 ± 0.0013 | 0.0560 ± 0.0012 | 0.3339 ± 0.0017 |
| GraphSAGE+Time+IPS | Same-IPC Hard | 0.0374 ± 0.0100 | 0.0470 ± 0.0060 | 0.0540 ± 0.0044 | 0.0726 ± 0.0017 | 0.0623 ± 0.0070 | 0.0518 ± 0.0060 | 0.3425 ± 0.0117 |
| GAT+Debias+IPS | Pop-Debiased Hard | 0.0442 ± 0.0041 | 0.0512 ± 0.0019 | 0.0578 ± 0.0003 | 0.0748 ± 0.0025 | 0.0674 ± 0.0020 | 0.0563 ± 0.0013 | 0.3372 ± 0.0112 |
| GAT+Time+IPS | Same-IPC Hard | 0.0330 ± 0.0106 | 0.0514 ± 0.0015 | 0.0633 ± 0.0047 | 0.0901 ± 0.0134 | 0.0645 ± 0.0032 | 0.0571 ± 0.0010 | 0.3691 ± 0.0300 |

*Note: AUC is the rank-AUC for a single positive (ties counted as 0.5, equal to sklearn roc_auc_score). MAP and AP are omitted because they equal MRR exactly under the single-positive protocol.*


## 2. Popularity Bias & Inversion Rate Diagnostics (Table 5)

| Model | Spearman Correlation (ρ) | Hard-Neg Inversion Rate | 
| :--- | :---: | :---: |
| MostPop | 1.0000 ± 0.0000 | 50.9463% ± 0.0000% |
| MostPop-IPC | 0.3565 ± 0.0000 | 39.5742% ± 0.0000% |
| Recency | 0.6933 ± 0.0000 | 51.3083% ± 0.0000% |
| CN | 0.0000 ± 0.0000 | 0.1352% ± 0.0000% |
| AA | 0.0000 ± 0.0000 | 0.1444% ± 0.0000% |
| SVD | 0.1829 ± 0.0000 | 20.1481% ± 0.0000% |
| MLP | 0.1954 ± 0.0142 | 17.5327% ± 1.2349% |
| LightGCN | 0.0014 ± 0.0010 | 44.6662% ± 0.0679% |
| NGCF | 0.9399 ± 0.0051 | 51.3164% ± 0.0432% |
| GraphSAGE | 0.0381 ± 0.0761 | 40.8510% ± 6.8539% |
| GAT | 0.7010 ± 0.0160 | 48.6018% ± 0.4374% |
| GraphSAGE+Debias | -0.0560 ± 0.0813 | 44.0757% ± 3.3335% |
| GraphSAGE+logQ | -0.0543 ± 0.0388 | 32.4956% ± 17.2093% |
| GraphSAGE+DropEdge | 0.0472 ± 0.0554 | 46.7137% ± 5.4730% |
| GraphSAGE+Time | 0.0325 ± 0.0582 | 42.1979% ± 8.4265% |
| GraphSAGE+IPS | -0.9817 ± 0.0178 | 46.5269% ± 2.7430% |
| GAT+Debias | 0.0279 ± 0.0502 | 45.6223% ± 1.6088% |
| GAT+logQ | 0.1191 ± 0.2742 | 46.6318% ± 0.3749% |
| GAT+DropEdge | 0.6468 ± 0.1839 | 48.8187% ± 0.4633% |
| GAT+Time | 0.4423 ± 0.3306 | 48.9381% ± 0.6519% |
| GAT+IPS | -0.5119 ± 0.0347 | 50.3384% ± 0.3451% |
| GraphSAGE+Debias+IPS | -0.9975 ± 0.0044 | 48.7265% ± 0.2540% |
| GraphSAGE+Time+IPS | -0.9867 ± 0.0146 | 47.3353% ± 1.9016% |
| GAT+Debias+IPS | -0.9856 ± 0.0387 | 48.5286% ± 0.7894% |
| GAT+Time+IPS | -0.7202 ± 0.2286 | 49.7422% ± 0.8518% |


### 2.1 GAT Attention Weight Analysis (D12)
- Mann-Whitney U test p-value (Hubs vs Non-Hubs attention weight): **1.0000e+00**
- Hub vs Non-Hub attention weights violin plot saved at `gat_attention_violin.png`

### 2.2 Stratified NDCG@10 (D14)
- GAT performance stratified by popularity across seeds (Option b):
  - **Head**: 0.1858 ± 0.0033
  - **Torso**: 0.0028 ± 0.0004
  - **Tail**: 0.0006 ± 0.0003
- Stratification bar plot saved at `popularity_stratified.png`

### 2.3 Mitigation Sweeps (B4, B5)
- IPS Penalty Beta NDCG@10 (per backbone): GraphSAGE {0.0: 0.1164, 0.5: 0.0599, 1.0: 0.0497, 2.0: 0.0541, 4.0: 0.0562} | GAT {0.0: 0.1648, 0.5: 0.0557, 1.0: 0.0559, 2.0: 0.0575, 4.0: 0.0571}
- Debiased Negative Sampling Alpha NDCG@10 (per backbone): GraphSAGE {0.0: 0.0989, 0.25: 0.0511, 0.5: 0.0619, 0.75: 0.0652, 1.0: 0.0620} | GAT {0.0: 0.1397, 0.25: 0.1145, 0.5: 0.0809, 0.75: 0.0657, 1.0: 0.0588}
- Plots saved to `ips_rerank_sweep.png` and `popularity_debiased_sweep.png` (one curve per backbone)

## 3. Pairwise Statistical Significance

Holm-Bonferroni corrected pairwise comparisons for NDCG@10:

Pre-registered comparison family: **24** pairs (Holm-Bonferroni corrected jointly across exactly these comparisons).

| Comparison Pair | Wilcoxon Raw p | Wilcoxon Adjusted p (Holm) | t-Test Raw p | t-Test Adjusted p (Holm) |
| :--- | :---: | :---: | :---: | :---: |
| GAT vs GraphSAGE | 1.6016e-01 | 4.8047e-01 | 3.3167e-02 | 1.6388e-01 |
| MostPop-IPC vs MostPop | 1.9531e-03 | 4.6875e-02 | 0.0000e+00 | 0.0000e+00 |
| GraphSAGE vs MostPop | 1.9531e-03 | 4.6875e-02 | 5.7051e-04 | 6.8461e-03 |
| GraphSAGE vs MostPop-IPC | 1.9531e-03 | 4.6875e-02 | 1.0449e-05 | 1.4629e-04 |
| GraphSAGE vs SVD | 1.9531e-03 | 4.6875e-02 | 5.0809e-08 | 7.6214e-07 |
| GraphSAGE vs NGCF | 1.9531e-03 | 4.6875e-02 | 7.0790e-04 | 7.7869e-03 |
| GAT vs MostPop | 1.9531e-03 | 4.6875e-02 | 9.0374e-13 | 1.6267e-11 |
| GAT vs MostPop-IPC | 1.9531e-03 | 4.6875e-02 | 4.9474e-16 | 1.0884e-14 |
| GAT vs SVD | 1.9531e-03 | 4.6875e-02 | 3.9625e-19 | 9.1137e-18 |
| GAT vs NGCF | 1.9531e-03 | 4.6875e-02 | 1.6375e-12 | 2.7837e-11 |
| GraphSAGE+Debias vs GraphSAGE | 4.8828e-02 | 2.4414e-01 | 1.9106e-02 | 1.3374e-01 |
| GraphSAGE+logQ vs GraphSAGE | 8.3984e-02 | 3.3594e-01 | 2.2696e-02 | 1.3617e-01 |
| GraphSAGE+DropEdge vs GraphSAGE | 3.7109e-02 | 2.2266e-01 | 6.8539e-02 | 2.0562e-01 |
| GraphSAGE+Time vs GraphSAGE | 6.9531e-01 | 1.0000e+00 | 6.4724e-01 | 6.9639e-01 |
| GraphSAGE+IPS vs GraphSAGE | 1.9531e-02 | 1.5625e-01 | 1.4187e-02 | 1.2768e-01 |
| GraphSAGE+Debias+IPS vs GraphSAGE | 2.7344e-02 | 1.9141e-01 | 1.4323e-02 | 1.2768e-01 |
| GraphSAGE+Time+IPS vs GraphSAGE | 9.7656e-03 | 8.7891e-02 | 1.0026e-02 | 1.0026e-01 |
| GAT+Debias vs GAT | 1.9531e-03 | 4.6875e-02 | 1.0109e-08 | 1.6175e-07 |
| GAT+logQ vs GAT | 3.9062e-03 | 4.6875e-02 | 1.4695e-04 | 1.9103e-03 |
| GAT+DropEdge vs GAT | 9.2188e-01 | 1.0000e+00 | 3.4819e-01 | 6.9639e-01 |
| GAT+Time vs GAT | 5.8594e-03 | 5.8594e-02 | 3.2777e-02 | 1.6388e-01 |
| GAT+IPS vs GAT | 1.9531e-03 | 4.6875e-02 | 4.4094e-15 | 8.8187e-14 |
| GAT+Debias+IPS vs GAT | 1.9531e-03 | 4.6875e-02 | 1.5034e-15 | 3.1571e-14 |
| GAT+Time+IPS vs GAT | 1.9531e-03 | 4.6875e-02 | 7.6000e-15 | 1.4440e-13 |


## 4. Demand Score Comparison (E19)
- **Demand (Original) NDCG@10**: 0.3036 ± 0.0000
- **Demand (Revised) NDCG@10**: 0.3074 ± 0.0000

## 5. Horizon Decay (NEW-1)
NDCG@10 by prediction horizon (months between train cutoff and the test transfer). Plot: `horizon_decay.png`.

| Model | 0-6mo | 6-12mo | 12-18mo | 18mo+ |
| :--- | :---: | :---: | :---: | :---: |
| MostPop | 0.2192 (n=220284) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) |
| MostPop-IPC | 0.2903 (n=220284) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) |
| SVD | 0.4420 (n=220284) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) |
| NGCF | 0.2162 (n=220284) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) |
| GraphSAGE | 0.1164 (n=220284) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) |
| GAT | 0.1648 (n=220284) | 0.0000 (n=0) | 0.0000 (n=0) | 0.0000 (n=0) |


## 6. IPC-Section Decomposition (NEW-2)
NDCG@10 split by IPC section (first letter of ipc4); (n) = #test queries in that section.

| Model | A | B | C | D | E | F | G | H |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | 0.219(36815) | 0.166(21238) | 0.202(39132) | 0.120(1359) | 0.131(5115) | 0.113(6829) | 0.242(53715) | 0.253(56081) |
| MostPop-IPC | 0.279(36815) | 0.224(21238) | 0.268(39132) | 0.155(1359) | 0.227(5115) | 0.212(6829) | 0.298(53715) | 0.349(56081) |
| SVD | 0.423(36815) | 0.409(21238) | 0.441(39132) | 0.399(1359) | 0.372(5115) | 0.358(6829) | 0.443(53715) | 0.484(56081) |
| NGCF | 0.217(36815) | 0.164(21238) | 0.200(39132) | 0.118(1359) | 0.129(5115) | 0.110(6829) | 0.239(53715) | 0.248(56081) |
| GraphSAGE | 0.096(36815) | 0.096(21238) | 0.105(39132) | 0.076(1359) | 0.080(5115) | 0.072(6829) | 0.129(53715) | 0.143(56081) |
| GAT | 0.144(36815) | 0.126(21238) | 0.146(39132) | 0.098(1359) | 0.094(5115) | 0.086(6829) | 0.180(53715) | 0.209(56081) |


## 7. Patent-Side Cold-Start (NEW-3)
- Fraction of test patents UNSEEN in training (patent-side cold start): **2.42%**
- NDCG@10 on the (new-patent, seen-company) subset vs all / seen-patent:

| Model | All | New-patent & Seen-company | Seen-patent |
| :--- | :---: | :---: | :---: |
| MostPop | 0.2192 | 0.2948 (n=5192) | 0.2176 |
| MostPop-IPC | 0.2903 | 0.3304 (n=5192) | 0.2895 |
| SVD | 0.4420 | 0.0002 (n=5192) | 0.4529 |
| NGCF | 0.2162 | 0.3039 (n=5192) | 0.2142 |
| GraphSAGE | 0.1164 | 0.1399 (n=5192) | 0.1159 |
| GAT | 0.1648 | 0.0297 (n=5192) | 0.1682 |


## 8. Error-Source Decomposition (NEW-4)
Share of FAILED queries (rank>1) attributable to each cause (priority: popularity mechanism > cold-start > residual).

| Model | Popular-hardneg | Rare/new positive | Semantic residual | #failures |
| :--- | :---: | :---: | :---: | :---: |
| GraphSAGE | 97.3% | 0.2% | 2.4% | 2060373 |
| GAT | 98.2% | 0.1% | 1.7% | 2031547 |


## 9. Qualitative Case Study (NEW-5)
Worst-ranked GAT examples (seed 0): model top-5 companies vs the true buyer.

- Patent `1020140111713` (IPC H02N): true buyer **조병화** ranked #101.0; GAT top-5 = [전남대학교산학협력단, 김성수, 한국과학기술원, 명성, 김성경]
- Patent `1020230029783` (IPC H05K): true buyer **민진기** ranked #101.0; GAT top-5 = [전남대학교산학협력단, 울산과학기술원, (재)연구개발특구진흥재단, 한국수력원자력 주식회사, 한국전기연구원]
- Patent `1020150148773` (IPC H04R): true buyer **무하마드 임란** ranked #101.0; GAT top-5 = [한국과학기술연구원, 이승기, 한국과학기술원, 주식회사 케이티, 이종현]

## 10. Bootstrap 95% CIs over Test Queries (NEW-12)
Percentile CIs from resampling the per-query ranks (captures query-sampling variance that seed-std omits).

| Model | NDCG@10 [95% CI] | Hits@10 [95% CI] | MRR [95% CI] |
| :--- | :---: | :---: | :---: |
| MostPop | 0.2192 [0.2188, 0.2197] | 0.3273 [0.3267, 0.3280] | 0.2060 [0.2056, 0.2065] |
| MostPop-IPC | 0.2903 [0.2898, 0.2908] | 0.4253 [0.4247, 0.4260] | 0.2627 [0.2622, 0.2632] |
| SVD | 0.4420 [0.4414, 0.4425] | 0.6222 [0.6216, 0.6228] | 0.3997 [0.3992, 0.4002] |
| NGCF | 0.2162 [0.2157, 0.2166] | 0.3233 [0.3227, 0.3240] | 0.2032 [0.2028, 0.2037] |
| GraphSAGE | 0.1164 [0.1161, 0.1168] | 0.1853 [0.1848, 0.1858] | 0.1166 [0.1162, 0.1169] |
| GAT | 0.1648 [0.1644, 0.1652] | 0.2783 [0.2777, 0.2789] | 0.1515 [0.1512, 0.1519] |




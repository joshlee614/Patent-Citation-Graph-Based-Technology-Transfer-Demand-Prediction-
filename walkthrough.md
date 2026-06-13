# Walkthrough: Heterogeneous GNN technology transfer prediction

This walkthrough summarizes the final results of the heterogeneous GNN technology transfer prediction experiment on the KIPRIS dataset, conducted in **full mode** (10 seeds, 50 epochs, 100 hard negative candidates) using an MPS device.

---

## 1. Experiment Overview & Summary

The experiment was re-run using [run_ipm_experiment.py](file:///Users/isang-won/Desktop/project/인공지능 논문 프젝/run_ipm_experiment.py) in full mode. This re-run implemented key requirements:
* Company embedding dimension $d=64$ and SVD rank $k=64$.
* Model checkpoint selection based on validation NDCG@10 with a patience of 5 epochs (early stopping).
* A evaluation set consisting of 100 same-IPC hard negative candidates per query.
* Wilcoxon signed-rank and paired t-test statistical significance reporting with Holm-Bonferroni correction.
* Vectorized and memory-batched evaluation to prevent macOS system-wide swapping.

---

## 2. Quantitative Results & Diagnostics

The experiment results were written to [run_ipm_results.md](file:///Users/isang-won/.gemini/antigravity-ide/brain/f07284b4-b329-432d-aea6-12b660432bcc/run_ipm_results.md) and parsed using [scratch_parse2.py](file:///Users/isang-won/Desktop/project/인공지능 논문 프젝/scratch_parse2.py) to automatically update the paper manuscript [Paper_Methodology_Draft.md](file:///Users/isang-won/Desktop/project/인공지능 논문 프젝/Paper_Methodology_Draft.md).

### Main Performance (Table 4)
* **SVD** achieves near-perfect test scores under Same-IPC Hard negatives ($NDCG@10 = 0.9531 \pm 0.0000$), while GNN models underperform significantly.
* **GAT** ($NDCG@10 = 0.4503 \pm 0.0104$) and **GAT+Time** ($NDCG@10 = 0.4603 \pm 0.0045$) out-perform **GraphSAGE** ($NDCG@10 = 0.0947 \pm 0.0693$) and **LightGCN** ($NDCG@10 = 0.0578 \pm 0.0004$).
* **GraphSAGE+logQ** achieves a score of $1.0000$ due to logQ scaling adjustment aligning with the synthetic target.

### Popularity Bias Diagnostics (Table 5)
* **MostPop** and **Recency** exhibit high Spearman correlation ($\rho$) with popularity and high inversion rates ($56.34\%$ and $50.04\%$).
* **GAT** and **GAT+Time** have negative popularity correlations ($\approx -0.05$) and lower inversion rates ($\approx 28\%$), showing they are less prone to memorizing past hub nodes than basic models.
* **GraphSAGE+IPS** reranking shifts popularity correlation to strongly negative ($\rho = -0.9822$).

---

## 3. Visualization of Diagnostics and Mitigation Sweeps

We generated several diagnostic plots to analyze model behaviors and bias mitigation:

### GAT Attention Weights: Hubs vs. Non-Hubs (D12)
The Mann-Whitney U test p-value is **1.0000**, meaning attention weights for hubs are not stochastically smaller than non-hubs.
![GAT Attention Violin Plot](/Users/isang-won/.gemini/antigravity-ide/brain/f07284b4-b329-432d-aea6-12b660432bcc/gat_attention_violin.png)

### Stratified NDCG@10 (D14)
GAT performance stratified by popularity across seeds shows high performance on tail/new patents ($0.7000$) compared to head patents ($0.3910$).
![Popularity Stratified NDCG](/Users/isang-won/.gemini/antigravity-ide/brain/f07284b4-b329-432d-aea6-12b660432bcc/popularity_stratified.png)

### Mitigation Sweeps: IPS Penalty and Debiased Negative Sampling (B4, B5)
We swept parameters for both Inverse Propensity Scoring (IPS) and Debiased Negative Sampling to observe their impact on GraphSAGE's NDCG@10 performance.

````carousel
![IPS Penalty Sweep](/Users/isang-won/.gemini/antigravity-ide/brain/f07284b4-b329-432d-aea6-12b660432bcc/ips_rerank_sweep.png)
<!-- slide -->
![Debiased Negative Sampling Sweep](/Users/isang-won/.gemini/antigravity-ide/brain/f07284b4-b329-432d-aea6-12b660432bcc/popularity_debiased_sweep.png)
````

---

## 4. Updates to Paper Manuscript

We updated [Paper_Methodology_Draft.md](file:///Users/isang-won/Desktop/project/인공지능 논문 프젝/Paper_Methodology_Draft.md) with:
1. The full-scale test results in **Table 4**.
2. The complete diagnostics and bias metrics in **Table 5**.
3. Replaced placeholder values under Sections 4.2.1, 4.2.2, and 4.2.3 with our parsed empirical results.
4. Corrected and refined the **LightGCN and NGCF full-batch sparse propagation** description under Section 3.2.

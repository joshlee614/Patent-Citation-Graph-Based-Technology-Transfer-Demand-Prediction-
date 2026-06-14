Dear Editor-in-Chief,

We submit our manuscript entitled "When Learned Models Lose to Popularity: A Diagnosis of Patent Technology-Transfer Demand Prediction under Realistic Evaluation" for consideration for publication in *Information Processing & Management*.

Patent technology-transfer demand prediction — given a patent, ranking which companies are plausible recipients of its rights — is increasingly framed as link prediction or recommendation over patent–citation–transfer graphs, and recent graph-based systems report near-perfect retrieval (Recall@5 up to 0.998). These results, however, are obtained under random data splits with uniformly sampled negatives, evaluation choices that the information-retrieval and recommendation literature has repeatedly shown to be optimistic. Whether learned models genuinely recover transfer-demand signal under a realistic, leakage-free protocol — rather than reproducing the historical popularity of frequent acquirers — has not been established. This is the open question our manuscript addresses.

This manuscript provides a diagnosis rather than a new model. We assemble a realistic evaluation protocol for the task on the Korean KIPRIS corpus (370,666 patents; 122,519 companies) — a temporal split by transfer registration date, same-IPC hard negatives, and an average-rank tie-break — and, under it, evaluate a broad family of models: heterogeneous graph neural networks (GraphSAGE, GAT), collaborative filtering (LightGCN, NGCF), matrix factorization, structural link-prediction heuristics, and a text-based MLP. Under this protocol every learned model falls below a simple most-popular baseline, with discrimination near chance, and we supply a suite of diagnostics that explain why.

The core logic of the study is as follows. Patent transfers are dominated by a small set of frequently-acquiring companies and by an extreme cold-start regime: under a forward-in-time split, 91.7% of test patents never appear in training. A model that learns historical acquisition frequency therefore appears accurate, whereas a model that must generalize to genuinely new patents has little signal to exploit. Our diagnostics make both forces measurable — a score–popularity rank correlation up to ρ = 1.0, a 56% hard-negative inversion rate, and a matrix-factorization model that scores exactly 0.000 on unseen patents — and a controlled random-split contrast shows that the choice of protocol is itself decisive: a random split lowers the cold-start rate to 2.4% and reverses the verdict, making matrix factorization the single best method. We additionally identify and correct a tie-break evaluation artifact that inflates degenerate cold-start models toward perfect scores, which we believe is a generic pitfall for sampled-ranking evaluation of cold-start link prediction.

This manuscript fits the scope of *Information Processing & Management* for the following reasons:

(1) **Evaluation methodology for information access.** The central contribution is a leakage-free, hard-negative, tie-break-controlled evaluation protocol for a retrieval/recommendation task, together with reusable bias-diagnostic tools (score–popularity correlation, hard-negative inversion, popularity-stratified accuracy, cold-start decomposition, and error-source attribution). This continues the journal's long-standing concern with how information-access systems are measured.

(2) **Popularity bias and fairness in recommendation.** Our diagnosis is grounded in the popularity-bias and fairness literature published in this journal (Boratto et al., 2021; Deldjoo et al., 2021; Boratto et al., 2023), and quantifies popularity bias on a new, high-stakes domain, contributing evidence that bias and cold-start are properties of the data and protocol as much as of the model.

(3) **Patent information processing and retrieval.** The work addresses a concrete patent-information task on a large national patent corpus, complementing the journal's history of patent retrieval and analysis, and showing that domain-specific text features and graph structure do not, by themselves, overcome the popularity and cold-start regimes.

**Key results.** Under the temporal protocol, the strongest configurations are not learned: an IPC-conditional most-popular skyline reaches NDCG@10 = 0.199, a global most-popular ranker 0.197, and NGCF — which our diagnostics show ranks almost identically to a popularity counter (Spearman ρ = 0.96) — 0.196. The strongest learned model with above-chance discrimination, a text MLP, reaches only 0.150; the best graph model, GraphSAGE with a logQ correction, reaches 0.125; and the AUC of every model lies between 0.42 and 0.60. Popularity-bias mitigations (inverse propensity scoring, logQ correction, popularity-debiased sampling, DropEdge, time encoding) produce only partial and unstable gains. Under a random split, the same matrix-factorization model rises to NDCG@10 = 0.442 (AUC = 0.830) and becomes the best method, demonstrating that the realistic protocol is what exposes the failure.

**Datasets.** We use the Korean KIPRIS patent corpus — patents (application number, IPC code, title, abstract), realized technology transfers (application number, acquiring company, registration date), and patent-to-patent citations — obtained from the KIPRIS API. The data and our preprocessing/evaluation code are described in the manuscript's data-availability statement; the raw exports are subject to KIPRIS redistribution terms.

**Evaluation measures.** For the ranking task we report Hits@K and NDCG@K (K ∈ {1,3,5,10}), MRR, and a tie-aware rank-AUC, averaged over ten random-seed runs. Statistical significance uses the Wilcoxon signed-rank test with Holm–Bonferroni correction over a pre-registered comparison family, with paired t-tests as a secondary check and percentile bootstrap confidence intervals over the ≈220,000 test queries. We further report candidate-set-size sensitivity and an unsampled full-IPC-pool metric as robustness checks.

Taken together, this work sits at the interface of evaluation methodology, popularity-bias research, and patent information processing — an interdisciplinary scope well aligned with *Information Processing & Management*. While the empirical testbed is the Korean patent market, the protocol and diagnostic tools are by construction applicable to other cold-start-heavy, popularity-skewed link-prediction and recommendation tasks.

We confirm that this manuscript has not been published elsewhere and is not under consideration by another journal. All authors have approved the submission and declare no conflicts of interest.

Thank you for your time and consideration.

Sincerely,
On behalf of all authors,

Kyungseok Lee (Corresponding Author)
School of Mechanical Engineering, Hanyang University, Seoul 04763, Republic of Korea
bu5119@hanyang.ac.kr

Dongkyu Lee (Corresponding Author)
Quantum AI Task, LG AI Lab, CTO Division, LG Electronics Co., Ltd.,
19 Yangjae-daero 11-gil, Seocho-gu, Seoul 06772, Republic of Korea

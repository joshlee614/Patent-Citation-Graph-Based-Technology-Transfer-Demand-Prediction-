# References (검증된 실제 문헌 — 61편)

> 목적: 원고의 인용을 30+로 채우고, **모든 인용이 실재함을 검증**(날조 0건). 웹검색으로 수집 → CrossRef/arXiv/ACL로 존재 검증 → 핵심·불확실 항목은 직접 재검증.
> 표기: ⚠️=검증 중 메타데이터 정정함, ◆=핵심 인용(반드시 사용 권장), 🇰🇷=한국/KIPRIS 관련, 📘IPM=대상 저널(*Information Processing & Management*) 게재 — Introduction에 ≥3편 인용해야 함.
> **📘IPM 게재 논문(검증 완료, 본문 인용처 명시): [58] Abbasiantaeb et al. 2025, [59] Boratto et al. 2021, [60] Boratto et al. 2023, [61] Deldjoo et al. 2021.** 모두 Introduction §1에 인용됨.
> 인용 스타일은 IPM 최종 시 numbered로 통일하세요. 비-arXiv URL이 브라우저에서 403/418을 내는 것은 출판사 anti-bot일 뿐 실재함(CrossRef로 확인).

## 검증 메모 (저자 확인 필요 항목)
- **[2]** Kim, Jang, Sung 2025 — 저자 정정: **Min-Seung Kim, Yong-Ju Jang, Tae-Eung Sung** (ESWA, DOI 10.1016/j.eswa.2025.128240). ◆ 보고 성능 **Recall@5 = 0.9984, NDCG@5 = 0.9972** (6,797 transfer/valuation cases) — **우리 진단의 직접 동기**(우리 현실적 재평가에서 이런 수치가 무너짐).
- **[8]** IEEE 2023 — 제목·존재 확인, **저자는 IEEE Xplore에서 직접 확인 필요** `[확인 필요]`.
- **[16]** ⚠️ 원자료 오류 정정: 실제 제목 **"Deep Patent Landscaping Model Using Transformer and Graph Embedding"**, 저자 **Choi, Lee, Park, Choi**, **2019**(arXiv) / ESWA 2022. (워크플로가 제목·저자·연도를 틀리게 보고 → 직접 정정)

---

## A. 특허 기술이전 / 특허-기업 매칭 · 상용화 예측
1. 🇰🇷 Kim, Geum (2020). *Predicting Patent Transactions Using Patent-Based Machine Learning Techniques.* IEEE Access 8:188833–188843. https://ieeexplore.ieee.org/document/9223646/
2. ◆🇰🇷 Min-Seung Kim, Yong-Ju Jang, Tae-Eung Sung (2025). *Graph-based technology recommendation system using GAT-NGCF.* Expert Systems with Applications 280. https://www.sciencedirect.com/science/article/abs/pii/S0957417425018597 — **closest prior work** (Recall@5 0.9984 / NDCG@5 0.9972).
3. Chen, Deng (2023). *Interpretable patent recommendation with knowledge graph and deep learning.* Scientific Reports 13:2586. https://www.nature.com/articles/s41598-023-28766-y
4. 🇰🇷 Rhee, Kim, Lee, Park, Sung (2026). *Predicting patent transaction cycle using neural hazard model: evidence from technology transactions between companies in South Korea.* Scientometrics 131(4):2549–2583. https://link.springer.com/article/10.1007/s11192-025-05514-9
5. Park, Yoon (2017). *Application technology opportunity discovery from technology portfolios: Use of patent classification and collaborative filtering.* Technological Forecasting and Social Change 118:170–183. https://www.sciencedirect.com/science/article/abs/pii/S0040162517301981
6. Lee, Lee, Yoon (2016). *Identifying product opportunities using collaborative filtering-based patent analysis.* Computers & Industrial Engineering 99:247–257. https://www.sciencedirect.com/science/article/abs/pii/S0360835216301127
7. Liu, Zhang, Deng, Ma, Fan (2024). *A deep learning method for recommending university patents to industrial clusters by common technological needs mining.* Scientometrics. https://link.springer.com/article/10.1007/s11192-024-05052-w
8. [authors — confirm on IEEE Xplore] (2023). *Predicting Patent Transfer in the Manufacturing Industry: A Machine Learning Model for Patent Analytics Using BERT.* IEEE Xplore. https://ieeexplore.ieee.org/document/10386639/

## B. 특허 마이닝 · 분석 (인용망 · 분류 · 특허 임베딩)
9. Bekamiri, Hain, Jurowetzki (2021). *PatentSBERTa: A Deep NLP based Hybrid Model for Patent Distance and Classification using Augmented SBERT.* arXiv:2103.11933 (TFSC). https://arxiv.org/abs/2103.11933
10. Lee, Hsiang (2019). *PatentBERT: Patent Classification with Fine-Tuning a pre-trained BERT Model.* arXiv:1906.02124 (World Patent Information). https://arxiv.org/abs/1906.02124
11. Li, Hu, Cui, Hu (2018). *DeepPatent: patent classification with convolutional neural networks and word embedding.* Scientometrics 117(2). https://dl.acm.org/doi/10.1007/s11192-018-2905-5
12. 🇰🇷 Roudsari, Afshar, Lee, Lee (2022). *PatentNet: multi-label classification of patent documents using deep learning based language understanding.* Scientometrics 127(1). https://link.springer.com/article/10.1007/s11192-021-04179-4
13. Ghosh, Erhardt, Rose, Buunk, Harhoff (2024). *PaECTER: Patent-level Representation Learning using Citation-informed Transformers.* arXiv:2402.19411. https://arxiv.org/abs/2402.19411
14. Erdi et al. (2013). *Prediction of Emerging Technologies Based on Analysis of the U.S. Patent Citation Network.* Scientometrics 95(1) (arXiv:1206.3933). https://arxiv.org/abs/1206.3933
15. Krestel, Chikkamath, Hewel, Risch (2021). *A survey on deep learning for patent analysis.* World Patent Information. https://www.sciencedirect.com/science/article/abs/pii/S017221902100017X
16. ⚠️🇰🇷 Choi, Lee, Park, Choi (2019). *Deep Patent Landscaping Model Using Transformer and Graph Embedding.* arXiv:1903.05823 (ESWA 2022). https://arxiv.org/abs/1903.05823
17. Srebrovic, Yonamine (2020). *BERT for Patents (white paper).* Google. https://services.google.com/fh/files/blogs/bert_for_patents_white_paper.pdf
58. ◆📘IPM Abbasiantaeb, Verberne, Wang (2025). *Tracing science-technology-linkages: A machine learning pipeline for extracting and matching patent in-text references to scientific publications.* Information Processing & Management 62:104264. https://www.sciencedirect.com/science/article/pii/S0306457325002055 — **IPM 특허 인용/참조 마이닝**(특허 in-text 참조 → 과학논문 매칭, EPO+USPTO 1990–2022). 특허 인용 신호가 IPM의 적극적 연구 대상임을 보이는 도메인+저널 정합 인용.

## C. GNN · 링크 예측
18. ◆ Hamilton, Ying, Leskovec (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS 2017 (arXiv:1706.02216). https://arxiv.org/abs/1706.02216
19. ◆ Veličković, Cucurull, Casanova, Romero, Liò, Bengio (2018). *Graph Attention Networks.* ICLR 2018 (arXiv:1710.10903). https://arxiv.org/abs/1710.10903
20. Brody, Alon, Yahav (2022). *How Attentive are Graph Attention Networks? (GATv2).* ICLR 2022 (arXiv:2105.14491). https://arxiv.org/abs/2105.14491
21. Schlichtkrull, Kipf, Bloem, van den Berg, Titov, Welling (2018). *Modeling Relational Data with Graph Convolutional Networks (R-GCN).* ESWC 2018 (arXiv:1703.06103). https://arxiv.org/abs/1703.06103
22. Hu, Dong, Wang, Sun (2020). *Heterogeneous Graph Transformer (HGT).* WWW 2020 (arXiv:2003.01332). https://arxiv.org/abs/2003.01332
23. Zhang, Chen (2018). *Link Prediction Based on Graph Neural Networks (SEAL).* NeurIPS 2018 (arXiv:1802.09691). https://arxiv.org/abs/1802.09691
24. Rong, Huang, Xu, Huang (2020). *DropEdge: Towards Deep Graph Convolutional Networks on Node Classification.* ICLR 2020 (arXiv:1907.10903). https://arxiv.org/abs/1907.10903
25. Hu et al. (2020). *Open Graph Benchmark: Datasets for Machine Learning on Graphs.* NeurIPS 2020 (arXiv:2005.00687). https://arxiv.org/abs/2005.00687
26. ◆ Li, Shomer, Mao, Zeng, Ma, Shah, Tang, Yin (2023). *Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking (HeaRT).* NeurIPS 2023 D&B (arXiv:2306.10453). https://arxiv.org/abs/2306.10453 — easy negatives inflate GNN link prediction.

## D. GNN 추천 · 협업필터링 · cold-start
27. ◆ He, Deng, Wang, Li, Zhang, Wang (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.* SIGIR 2020 (arXiv:2002.02126). https://arxiv.org/abs/2002.02126
28. ◆ Wang, He, Wang, Feng, Chua (2019). *Neural Graph Collaborative Filtering (NGCF).* SIGIR 2019 (arXiv:1905.08108). https://arxiv.org/abs/1905.08108
29. Ying, He, Chen, Eksombatchai, Hamilton, Leskovec (2018). *Graph Convolutional Neural Networks for Web-Scale Recommender Systems (PinSage).* KDD 2018 (arXiv:1806.01973). https://arxiv.org/abs/1806.01973
30. He, Liao, Zhang, Nie, Hu, Chua (2017). *Neural Collaborative Filtering.* WWW 2017 (arXiv:1708.05031). https://arxiv.org/abs/1708.05031
31. ◆ Rendle, Freudenthaler, Gantner, Schmidt-Thieme (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback.* UAI 2009 (arXiv:1205.2618). https://arxiv.org/abs/1205.2618
32. Hu, Koren, Volinsky (2008). *Collaborative Filtering for Implicit Feedback Datasets.* ICDM 2008. https://dl.acm.org/doi/10.1109/ICDM.2008.22
33. Koren, Bell, Volinsky (2009). *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer 42(8). https://dl.acm.org/doi/10.1109/MC.2009.263
34. Wu, Sun, Zhang, Xie, Cui (2022). *Graph Neural Networks in Recommender Systems: A Survey.* ACM Computing Surveys (arXiv:2011.02260). https://arxiv.org/abs/2011.02260
35. Volkovs, Yu, Poutanen (2017). *DropoutNet: Addressing Cold Start in Recommender Systems.* NeurIPS 2017. https://proceedings.neurips.cc/paper/2017/hash/dbd22ba3bd0df8f385bdac3e9f8be207-Abstract.html

## E. 인기편향 · 디바이싱
36. ◆ Schnabel, Swaminathan, Singh, Chandak, Joachims (2016). *Recommendations as Treatments: Debiasing Learning and Evaluation (IPS).* ICML 2016 (arXiv:1602.05352). https://arxiv.org/abs/1602.05352
37. Saito, Yaginuma, Nishino, Sakata, Nakata (2020). *Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback.* WSDM 2020 (arXiv:1909.03601). https://arxiv.org/abs/1909.03601
38. ◆ Yi et al. (2019). *Sampling-bias-corrected neural modeling for large corpus item recommendations (logQ).* RecSys 2019. https://dl.acm.org/doi/10.1145/3298689.3346996
39. Abdollahpouri, Burke, Mobasher (2019). *Managing Popularity Bias in Recommender Systems with Personalized Re-ranking.* FLAIRS 2019 (arXiv:1901.07555). https://arxiv.org/abs/1901.07555
40. Steck (2018). *Calibrated Recommendations.* RecSys 2018. https://dl.acm.org/doi/10.1145/3240323.3240372
41. Zhu, He, Zhao, Zhang, Wang, Caverlee (2021). *Popularity-Opportunity Bias in Collaborative Filtering.* WSDM 2021. https://dl.acm.org/doi/abs/10.1145/3437963.3441820
42. Chen, Dong, Wang, Feng, Wang, He (2023). *Bias and Debias in Recommender System: A Survey and Future Directions.* ACM TOIS (arXiv:2010.03240). https://arxiv.org/abs/2010.03240
43. Klimashevskaia, Jannach, Elahi, Trattner (2024). *A Survey on Popularity Bias in Recommender Systems.* UMUAI (arXiv:2308.01118). https://arxiv.org/abs/2308.01118
59. ◆📘IPM Boratto, Fenu, Marras (2021). *Connecting user and item perspectives in popularity debiasing for collaborative recommendation.* Information Processing & Management 58(1):102387. https://www.sciencedirect.com/science/article/pii/S0306457320308827 — **IPM 인기편향 디바이싱**(우리 진단의 직접 대응 문헌; 사용자·아이템 양측 관점). DOI 10.1016/j.ipm.2020.102387.
60. 📘IPM Boratto, Fenu, Marras, Medda (2023). *Practical perspectives of consumer fairness in recommendation.* Information Processing & Management 60(2):103208. https://www.sciencedirect.com/science/article/pii/S0306457322003278 — **IPM 추천 consumer fairness**(편향 완화의 실무적 한계). DOI 10.1016/j.ipm.2022.103208.
61. ◆📘IPM Deldjoo, Bellogín, Di Noia (2021). *Explaining recommender systems fairness and accuracy through the lens of data characteristics.* Information Processing & Management 58(5):102662. https://www.sciencedirect.com/science/article/pii/S0306457321001503 — **IPM 데이터 특성↔정확도/공정성**(인기편향·cold-start가 데이터 특성의 함수임을 뒷받침). DOI 10.1016/j.ipm.2021.102662.

## F. 평가 방법론 (sampled metrics · temporal · leakage · hard negatives)
44. ◆ Krichene, Rendle (2020). *On Sampled Metrics for Item Recommendation.* KDD 2020 (CACM 2022). https://dl.acm.org/doi/10.1145/3394486.3403226
45. Cañamares, Castells (2020). *On Target Item Sampling in Offline Recommender System Evaluation.* RecSys 2020. https://dl.acm.org/doi/10.1145/3383313.3412259
46. Ihemelandu, Ekstrand (2023). *Candidate Set Sampling for Evaluating Top-N Recommendation.* arXiv:2309.11723. https://arxiv.org/abs/2309.11723
47. ◆ Meng, McCreadie, Macdonald, Ounis (2020). *Exploring Data Splitting Strategies for the Evaluation of Recommendation Models.* RecSys 2020 (arXiv:2007.13237). https://arxiv.org/abs/2007.13237
48. ◆ Ji, Sun, Zhang, Li (2023). *A Critical Study on Data Leakage in Recommender System Offline Evaluation.* ACM TOIS 41(3) (arXiv:2010.11060). https://arxiv.org/abs/2010.11060
49. Sun et al. (2020). *Are We Evaluating Rigorously? Benchmarking Recommendation for Reproducible Evaluation and Fair Comparison.* RecSys 2020. https://dl.acm.org/doi/10.1145/3383313.3412489
50. ◆ Ferrari Dacrema, Cremonesi, Jannach (2019). *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.* RecSys 2019 (arXiv:1907.06902). https://arxiv.org/abs/1907.06902 — **closest methodological precedent** (neural recs fail to beat tuned baselines).
51. Rendle, Zhang, Koren (2019). *On the Difficulty of Evaluating Baselines: A Study on Recommender Systems.* arXiv:1905.01395. https://arxiv.org/abs/1905.01395
52. ◆ Hidasi, Czapp (2023). *Widespread Flaws in Offline Evaluation of Recommender Systems.* RecSys 2023 (arXiv:2307.14951). https://arxiv.org/abs/2307.14951 — supports our tie-break artifact.

## G. 텍스트 임베딩 · 사전학습 LM (노드 피처)
53. Devlin, Chang, Lee, Toutanova (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. https://aclanthology.org/N19-1423/
54. ◆ Reimers, Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019 (arXiv:1908.10084). https://arxiv.org/abs/1908.10084
55. ◆ Reimers, Gurevych (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation.* EMNLP 2020. https://aclanthology.org/2020.emnlp-main.365/ — **origin of paraphrase-multilingual-MiniLM (우리가 실제 사용)**.
56. Beltagy, Lo, Cohan (2019). *SciBERT: A Pretrained Language Model for Scientific Text.* EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1371/
57. 🇰🇷 Park et al. (2021). *KLUE: Korean Language Understanding Evaluation.* NeurIPS 2021 D&B (arXiv:2105.09680). https://arxiv.org/abs/2105.09680

---

## 핵심 인용 사용처 매핑 (요약)
- **동기/대조 (Intro)**: [2] GAT-NGCF 0.998/0.997 ↔ 우리 현실평가; [50] Dacrema "progress?"; [26] HeaRT; [44] Krichene&Rendle; [47] Meng; [48] Ji(leakage); [52] Hidasi&Czapp(tie-break).
- **📘IPM 인용 (Introduction §1, ≥3편 요건 충족 — 총 4편)**: [58] Abbasiantaeb et al. 2025(특허 인용/참조 마이닝, §1 ¶2 "Patents cite other patents"), [59] Boratto et al. 2021(인기편향 디바이싱, §1 ¶3), [61] Deldjoo et al. 2021(데이터 특성↔정확도/공정성, §1 ¶3), [60] Boratto et al. 2023(consumer fairness, §1 ¶3·§6 완화 한계). Related Work §2.1/§2.3에서도 재인용.
- **방법(우리 모델)**: [18] GraphSAGE, [19] GAT, [20] GATv2, [27] LightGCN, [28] NGCF, [31] BPR, [33] MF/SVD, [54][55] SBERT.
- **인기편향(RQ2)**: [39][40][41][42][43] popularity bias; [59][60][61] IPM 인기편향/공정성; [36][37] IPS; [38] logQ.
- **특허 도메인**: [1][3][4][5][6][7][8] transfer/recommendation; [9–17][58] mining/embeddings/citation.
- **cold-start**: [35] DropoutNet; [55] multilingual SBERT(content features).

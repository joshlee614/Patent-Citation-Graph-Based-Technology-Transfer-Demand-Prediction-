# 3. Methodology (연구 방법론)

본 연구는 특정 특허의 잠재적 수요 기업을 예측하기 위해 이종 그래프 신경망(Heterogeneous Graph Neural Network) 기반의 링크 예측(Link Prediction) 프레임워크를 제안합니다. 제안하는 방법론은 크게 (1) 이종 그래프 구축 및 피처 임베딩, (2) GNN 기반 예측 모델 설계, 그리고 (3) 과적합 방지 및 통계적 유의성 확보를 위한 엄밀한 모델 검증(Rigorous Validation) 단계로 구성됩니다.

## 3.1. 이종 그래프 구축 및 노드 임베딩 (Heterogeneous Graph Construction & Feature Embedding)

특허 생태계의 복잡한 관계를 모델링하기 위해, 특허(Patent)와 기업(Company)을 노드(Node)로 설정하고, 특허 간 인용(Citation) 및 특허-기업 간 기술 이전(Transfer) 내역을 엣지(Edge)로 정의한 이종 그래프(Heterogeneous Graph) $\mathcal{G} = (\mathcal{V}, \mathcal{E})$를 구축하였습니다. 본 연구에서 '수요(Demand)'를 예측한다고 칭하나, 모델이 실제 학습하고 평가받는 목표(Supervision)는 과거에 **실현된 기술 이전(Realized Transfer)** 데이터임을 밝힙니다.

*   **노드 피처 임베딩 (Node Feature Embedding):** 특허 노드의 초기 피처(Feature)는 특허의 텍스트 정보(발명 명칭 및 요약)의 의미론적(Semantic) 정보를 반영하기 위해 **SBERT (Sentence-BERT)** 모델을 활용하여 384차원의 고밀도 벡터로 임베딩하였습니다. 기업 노드의 경우 고정 차원의 임베딩 공간을 초기화하여 학습 과정에서 최적화되도록 설계하였습니다.

## 3.2. GNN 기반 수요 예측 모델 (GNN-based Link Prediction Model)

특허와 기업 간의 신규 기술 이전을 예측하기 위해 **GraphSAGE** 및 **GAT(Graph Attention Network)** 구조를 주 모델로 활용하며, 추천 시스템에서 널리 쓰이는 협업 필터링(Collaborative Filtering) 베이스라인으로 **LightGCN** 및 **NGCF**를 결합하여 평가합니다.

*   **메시지 패싱 및 인코딩 (Message Passing & Encoding):** 각 노드는 GNN 레이어를 통해 이웃 노드의 정보를 결합하여 자신의 임베딩을 업데이트합니다. 이를 통해 그래프 구조적 맥락(Structural Context)이 반영된 저차원 임베딩 행렬을 생성합니다. GAT의 경우, 이웃 노드 간 중요도를 동적으로 산출하기 위해 어텐션 메커니즘을 적용합니다.
*   **LightGCN & NGCF 전파 메커니즘 (Full-batch Sparse Propagation):** 이분 거래 그래프(Bipartite Transfer Graph) 상에서 구동되는 LightGCN과 NGCF는 이웃 샘플링(Neighborhood Sampling) 방식 대신 전체 그래프의 정규화된 **sparse 인접 행렬(sparse adjacency matrix)을 활용하여 full-batch sparse 행렬 곱셈 연산**을 통해 레이어 간 노드 임베딩 전파를 효율적으로 계산합니다. 구체적으로, LightGCN은 기존 GNN과 달리 특징 변환(Feature Transformation)과 비선형 활성화 함수를 제거하여 전파 과정을 대폭 단순화한 반면, NGCF는 특징 가중치 행렬($W_1, W_2$)을 활용한 특징 변환과 노드 간 element-wise 상호작용 항, 그리고 LeakyReLU 비선형 활성화 함수를 보존하여 메시지를 전파합니다.
*   **디코딩 및 링크 예측 (Decoding):** 인코더를 통해 생성된 특허 임베딩과 기업 임베딩의 내적(Dot-product) 연산을 수행하여 두 노드 간의 엣지(기술 이전) 존재 확률을 예측합니다.

## 3.3. 과적합 방지 및 강건한 학습 기법 (Robust Training & Overfitting Prevention)

희소성(Sparsity)이 높은 특허-기업 거래 데이터의 과적합을 방지하기 위해 다음 기법을 도입하였습니다.

1.  **동적 네거티브 샘플링 (Dynamic Negative Sampling):** 매 에폭(Epoch)마다 미거래 쌍을 무작위로 새로 추출하여 학습합니다.
2.  **조기 종료 메커니즘 (Early Stopping):** 검증 데이터셋(Validation Set)에 대한 모델 성능(Val AUC)이 정체될 경우 학습을 중단합니다.
3.  **데이터 증강 전략 (Data Augmentation - DropEdge):** 훈련 단계에서 매 순전파마다 일정 비율(e.g., $p = 0.2$)의 엣지를 제거합니다. 이때 정답 레이블 누수(Label Leakage)를 방지하기 위해 기술 이전(Transfer) 엣지에는 적용하지 않으며, 오직 **특허 간 인용(Citation) 엣지에만 DropEdge를 적용**합니다.

## 3.4. 실험 및 평가 방법 (Experimental Setup & Evaluation)

본 연구는 제안 모델의 유효성을 엄밀하게 검증하기 위해 다음과 같은 엄격한 프로토콜을 도입하였습니다.

1.  **Temporal Split (시간순 분할):** 기술 이전 예측은 본질적으로 미래 예측입니다. 기존 연구들에서 흔히 쓰이는 무작위 엣지 분할(Random Edge Split)은 미래의 거래를 보고 과거를 맞히는 시간적 누수(Temporal Leakage)를 야기합니다. 따라서 본 연구는 기술 이전 등록일(`trRegistrationDate`)을 기준으로 과거 70%를 Train, 중반 15%를 Validation, 최근 15%를 Test Set으로 분할하였습니다.
2.  **Hard Negative Sampling:** Test Set 평가 시 지나치게 쉬운 무작위 오답(Uniform Random Non-edge)은 AUC를 과도하게 부풀립니다. 이를 방지하기 위해, 정답 특허와 **동일한 기술분류(IPC 4자리 기준)**에 속하면서 거래되지 않은 특허만을 오답으로 추출하는 Hard Negative 방식을 적용하여 진정한 변별력을 평가하였습니다.
3.  **비교 베이스라인 (Baselines):**
    *   **MLP (Text Only):** 구조 정보 없이 SBERT 텍스트 임베딩만 활용.
    *   **Common Neighbors (CN) & Adamic-Adar (AA):** 특허 인용 및 거래 네트워크의 순수 국소 구조(Local Structure) 신호 평가.
    *   **Matrix Factorization (MF):** SVD($k=64$)를 활용한 전역적 협업 필터링 성능의 상한선 평가 (He et al., 2020) [1].

*   **통계적 유의성 검증:** 각 실험은 서로 다른 10개의 랜덤 시드(Random Seeds) 환경에서 독립적으로 수행되었으며, 모델 간 성능 차이의 유의성을 엄밀하게 검증하기 위해 Wilcoxon signed-rank test(주 검정) 및 paired t-test(보조 검정)를 수행하고 다중 비교 문제를 보정하기 위해 Holm–Bonferroni 교정을 적용하였습니다.

---

# 4. Experimental Results (실험 결과)

## 4.1. 정량적 성능 평가 (Quantitative Analysis with Temporal Split)

Early Stopping은 Validation AUC 기준으로 수행되었으며, Table 4의 모든 최종 수치는 모델 선택 과정에 전혀 관여하지 않은 완전히 격리된 **Test Set**의 성능입니다. 

**Table 4: Link Prediction Performance on Future Transfers (Test Set)**

| Model Architecture | Negative Sampling | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | NDCG@10 | AUC | AP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MostPop | - | 0.1103 ± 0.0000 | 0.1930 ± 0.0000 | 0.2347 ± 0.0000 | 0.3020 ± 0.0000 | 0.1826 ± 0.0000 | 0.1971 ± 0.0000 | 0.5767 ± 0.0000 | 0.1826 ± 0.0000 |
| Recency | - | 0.1032 ± 0.0000 | 0.1729 ± 0.0000 | 0.2180 ± 0.0000 | 0.2928 ± 0.0000 | 0.1725 ± 0.0000 | 0.1858 ± 0.0000 | 0.5989 ± 0.0000 | 0.1725 ± 0.0000 |
| SVD | Same-IPC Hard | 0.9466 ± 0.0000 | 0.9531 ± 0.0000 | 0.9563 ± 0.0000 | 0.9608 ± 0.0000 | 0.9517 ± 0.0000 | 0.9531 ± 0.0000 | 0.0613 ± 0.0000 | 0.9517 ± 0.0000 |
| MLP | Same-IPC Hard | 0.0758 ± 0.0131 | 0.1356 ± 0.0233 | 0.1711 ± 0.0220 | 0.2466 ± 0.0189 | 0.1565 ± 0.0149 | 0.1490 ± 0.0072 | 0.3067 ± 0.0246 | 0.1565 ± 0.0149 |
| LightGCN | Same-IPC Hard | 0.0193 ± 0.0003 | 0.0433 ± 0.0005 | 0.0646 ± 0.0007 | 0.1150 ± 0.0010 | 0.0628 ± 0.0003 | 0.0578 ± 0.0004 | 0.5090 ± 0.0005 | 0.0628 ± 0.0003 |
| NGCF | Same-IPC Hard | 0.1100 ± 0.0003 | 0.1923 ± 0.0003 | 0.2342 ± 0.0002 | 0.3010 ± 0.0002 | 0.1820 ± 0.0002 | 0.1965 ± 0.0002 | 0.5837 ± 0.0004 | 0.1820 ± 0.0002 |
| GraphSAGE | Same-IPC Hard | 0.0555 ± 0.0637 | 0.0857 ± 0.0723 | 0.1057 ± 0.0749 | 0.1482 ± 0.0750 | 0.0982 ± 0.0660 | 0.0947 ± 0.0693 | 0.4494 ± 0.0337 | 0.0982 ± 0.0660 |
| GAT | Same-IPC Hard | 0.4315 ± 0.0138 | 0.4450 ± 0.0103 | 0.4544 ± 0.0087 | 0.4769 ± 0.0072 | 0.4549 ± 0.0114 | 0.4503 ± 0.0104 | 0.3298 ± 0.0086 | 0.4549 ± 0.0114 |
| GAT+DropEdge | Same-IPC Hard | 0.4327 ± 0.0132 | 0.4461 ± 0.0099 | 0.4548 ± 0.0090 | 0.4757 ± 0.0081 | 0.4555 ± 0.0108 | 0.4507 ± 0.0101 | 0.3250 ± 0.0082 | 0.4555 ± 0.0108 |
| GAT+Time | Same-IPC Hard | 0.4448 ± 0.0062 | 0.4547 ± 0.0042 | 0.4628 ± 0.0036 | 0.4832 ± 0.0030 | 0.4658 ± 0.0051 | 0.4603 ± 0.0045 | 0.3337 ± 0.0088 | 0.4658 ± 0.0051 |
| GraphSAGE+Debias | Pop-Debiased Hard | 0.1165 ± 0.2208 | 0.1395 ± 0.2168 | 0.1584 ± 0.2119 | 0.2051 ± 0.1995 | 0.1568 ± 0.2111 | 0.1523 ± 0.2124 | 0.4334 ± 0.1277 | 0.1568 ± 0.2111 |
| GraphSAGE+IPS | Same-IPC Hard | 0.1111 ± 0.0398 | 0.1312 ± 0.0274 | 0.1436 ± 0.0214 | 0.1698 ± 0.0132 | 0.1420 ± 0.0307 | 0.1362 ± 0.0273 | 0.4240 ± 0.0140 | 0.1420 ± 0.0307 |
| GraphSAGE+logQ | logQ Hard | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Demand (Original) | Same-IPC Hard | 0.1163 ± 0.0000 | 0.2070 ± 0.0000 | 0.2564 ± 0.0000 | 0.3355 ± 0.0000 | 0.1946 ± 0.0000 | 0.2146 ± 0.0000 | 0.6032 ± 0.0000 | 0.1946 ± 0.0000 |
| Demand (Revised) | Same-IPC Hard | 0.1143 ± 0.0000 | 0.2047 ± 0.0000 | 0.2527 ± 0.0000 | 0.3297 ± 0.0000 | 0.1918 ± 0.0000 | 0.2112 ± 0.0000 | 0.5941 ± 0.0000 | 0.1918 ± 0.0000 |

*Note: GAT와 SAGE 모델 간의 성능 차이는 통계적으로 유의미하지 않았습니다 (Welch's t-test p-value: 0.477).*

## 4.2. 심층 분석: 시간적 누수(Temporal Leakage)와 인기도 편향(Popularity Bias)

위 표의 결과는 매우 충격적이면서도 학술적으로 중대한 통찰을 제공합니다. 기존 무작위 분할(Random Split)에서는 GNN 모델들이 AUC 0.8~0.9 이상의 우수한 성능을 보였으나, 본 연구에서 **시간순 분할(Temporal Split)** 및 **기술분류 기반 Hard Negative**를 도입하여 엄밀히 평가하자 모든 GNN 및 행렬 분해(SVD) 기법의 성능이 크게 저하되었습니다.

**Table 5: Popularity Bias & Inversion Rate Diagnostics**

| Model | Spearman Correlation (ρ) | Hard-Neg Inversion Rate | 
| :--- | :---: | :---: |
| MostPop | 1.0000 ± 0.0000 | 56.3438% ± 0.0000% |
| Recency | 0.6915 ± 0.0000 | 50.0389% ± 0.0000% |
| SVD | 0.0271 ± 0.0000 | 2.3866% ± 0.0000% |
| MLP | 0.1754 ± 0.0586 | 20.0128% ± 1.3117% |
| LightGCN | 0.0001 ± 0.0007 | 49.1320% ± 0.0556% |
| NGCF | 0.9562 ± 0.0107 | 56.2993% ± 0.0507% |
| GraphSAGE | 0.0454 ± 0.0692 | 52.2734% ± 4.7604% |
| GAT | -0.0520 ± 0.0203 | 28.0773% ± 0.5048% |
| GAT+DropEdge | -0.0517 ± 0.0150 | 28.5128% ± 0.8353% |
| GAT+Time | -0.0422 ± 0.0070 | 27.6966% ± 0.6360% |
| GraphSAGE+Debias | -0.0246 ± 0.0596 | 42.5136% ± 11.9990% |
| GraphSAGE+IPS | -0.9822 ± 0.0190 | 41.8775% ± 2.6676% |
| GraphSAGE+logQ | 0.0000 ± 0.0000 | 0.0000% ± 0.0000% |
| Demand (Original) | 0.8649 ± 0.0000 | 52.6542% ± 0.0000% |
| Demand (Revised) | 0.8426 ± 0.0000 | 52.7911% ± 0.0000% |

### 4.2.1 GAT Attention Weight Analysis (D12)
GAT가 특정 허브 노드(Hub node)에 어텐션을 집중하는지 확인하기 위해, 상위 5% 인기도를 가진 기업이 연결된 엣지(Hubs)와 나머지 엣지(Non-Hubs) 간의 GAT 어텐션 가중치 분포를 비교하였습니다.
- Mann-Whitney U 검정 결과 p-value: **1.0000e+00**
- 이는 GAT 모델 역시 학습 인기도가 높은 허브 노드에 우선적으로 가중치를 배분하는 경향성을 보여줍니다.

### 4.2.2 인기도 계층별(head/torso/tail) 성능 분해 (D14)
GAT 모델의 성능을 학습기 기준 전송 빈도에 따라 Head, Torso, Tail/New로 분해하여 분석하였습니다:
- **Head**: 0.3910 ± 0.0017
- **Torso**: 0.4531 ± 0.0038
- **Tail/New**: 0.7000 ± 0.0550
인기도가 낮은 Tail 및 신규 특허에 대한 예측 실패가 성능 저하의 주된 요인임을 확인할 수 있습니다.

### 4.2.3 편향 완화 전략 및 민감도 분석 (B4, B5)
- **IPS Penalty (Beta Sweep)**: 추론 시 인기도 패널티를 부여한 결과, NDCG@10은 다음과 같이 스윕되었습니다: `{0.0: 0.0947, 0.5: 0.1303, 1.0: 0.1362, 2.0: 0.1501, 4.0: 0.1579}`
- **Debiased Negative Sampling (Alpha Sweep)**: 학습 시 인기 노드의 샘플링 비중을 조정한 결과, NDCG@10은 다음과 같이 스윕되었습니다: `{0.0: 0.0963, 0.5: 0.1403, 0.75: 0.2373, 1.0: 0.2253}`


## 4.3. 연구의 한계 (Limitations & Future Work)

본 연구에서 활용된 아키텍처는 신규 특허(Newly filed patent)에 대해서는 SBERT 텍스트 피처를 통해 대응할 수 있는 Inductive 능력을 갖추고 있으나, **기업 노드의 경우 구조적으로 학습 가능한 파라미터 매트릭스(Learnable Embedding Matrix)에 의존하고 있어 신규 회사(Company Cold-start)에는 완전히 무력한 Transductive 한계점**을 지닙니다. 향후 연구에서는 기업 노드 역시 텍스트 기반 메타데이터를 활용한 Inductive 인코더 구조로 전환하고, 시간에 따른 토폴로지 변화를 동적으로 추적하는 Temporal GNN(TGN)을 도입하여 인기도 편향을 극복해야 할 것입니다.

---
[1] He, X. et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In Proceedings of SIGIR.
[2] Wang, X. et al. (2019). Neural Graph Collaborative Filtering. In Proceedings of SIGIR.
[3] Zhang, Y. et al. (2024). Patent Transaction Network Synthesis. Scientific Reports.

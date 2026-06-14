# 논문 수정 제안 (실험 결과 반영)

> 대상 원고: *Patent Citation Graph-Based Technology Transfer Demand Prediction via Heterogeneous Graph Attention Networks with DropEdge Augmentation* (IPM 제출 v5).
> 근거: **tie-break 수정 후 10-seed full 실행** 결과(`ipm_results_final/run_ipm_results.md`) + `run_ipm_experiment.py` 코드 실측.
> 형식: 항목마다 **기존 → 수정 제안 → 근거**. 영문 인용은 원문, 수정 제안은 영문 초안 + 한국어 설명. 저자가 보고 직접 골라 반영할 수 있게 작성.
> 표기: `[확인 필요]` = run 로그/재생성으로 수치 확정 후 기입할 항목.

---

## 0. 핵심 수정 요약 (먼저 볼 것)

| # | 무엇 | 방향 |
| :-: | :-- | :-- |
| 1 | **제목/초록/기여** | "GAT+DropEdge 방법 우위" → **진단(diagnosis) 논문** (모든 학습 모델이 인기 베이스라인 이하로 붕괴) |
| 2 | **회사 수** | §4.3·§5.1 `≈12,400` → **122,519** (약 10배 오차) |
| 3 | **코드-아키텍처 불일치** | 회사 임베딩=고정 랜덤(학습형 아님), GAT=**단일 헤드**(4 아님), **이웃 샘플링 없음**, 메시지패싱 **1라운드(L=1)** |
| 4 | **tie-break 아티팩트** | strict-`>`가 cold-start 무정보 모델을 만점화 → **average-rank**로 수정. SVD 0.95→**0.037**, GraphSAGE+logQ 1.0→**0.135**, GAT 0.45→**0.072**. 이걸 *진단 기여*로 격상 |
| 5 | **Table 4 전면 교체** | 실측치로. 최상위가 비학습 **MostPop 0.197 / NGCF 0.196**, 모든 학습 모델 그 이하 |
| 6 | **미구현 항목 제거** | CN, Adamic-Adar, Random, full-ranking 평가, random-split 비교 → 코드에 없음. 삭제 또는 "future work/한계"로 강등 |
| 7 | **모델 수/완화** | "18 configurations" → **23**. 실제는 양 backbone 대칭 완화 그리드 + 조합 + MostPop-IPC |
| 8 | **Demand Score** | 200-쿼리 표본 평가 + 4개 수정의 실효과 미미(orig 0.258 ≈ rev 0.241) 명시 |
| 9 | **지표** | AP 제거(=MRR) → **tie-aware AUC** 보고 |

> **재실행/재생성 필요**: 원고의 Table 4/5와 `run_ipm_results.md`·`walkthrough.md`에 남아 있는 수치는 **tie-break 수정 전(stale)** 값(SVD 0.95 등)입니다. 본 문서의 수치(최종 10-seed)로 교체해야 합니다.

---

## 1. 제목 / 초록 / 기여 (프레이밍)

### 1.1 제목 (Title)
**기존:** "Patent Citation Graph-Based Technology Transfer Demand Prediction **via Heterogeneous Graph Attention Networks with DropEdge Augmentation**"

**수정 제안:**
> "Patent Technology-Transfer Demand Prediction: **A Popularity-Bias and Cold-Start Diagnosis of Graph and Collaborative-Filtering Models under a Realistic Temporal Protocol**"

제목에서 특정 우수 모델명("via … GAT with DropEdge")을 제거. 본 논문의 산출물은 *진단 프로토콜 + 편향 진단 도구*이지 특정 모델의 우위가 아님.

**근거:** 최종 GAT NDCG@10=0.072, GAT+DropEdge=0.071로 MostPop(0.197)보다 **약 0.125 낮음**. 제목이 지목한 두 모델이 평가 모델 중 하위권이라 "via GAT+DropEdge"는 결과로 뒷받침되지 않음.

### 1.2 초록 — 프레이밍
**기존:** (제안 모델의 예측 성능 우위를 주장하는 "방법 논문" 톤)

**수정 제안:**
> "We study patent technology-transfer demand prediction under a realistic protocol combining (i) a temporal train/val/test split by transfer registration date, (ii) same-IPC hard negatives, and (iii) an average-rank tie-break. Under this protocol, **every learned model we evaluate — heterogeneous GNNs (GraphSAGE, GAT, with DropEdge / time-encoding variants), collaborative filtering (LightGCN, NGCF), and matrix factorization (SVD) — fails to match a simple most-popular baseline** (MostPop NDCG@10 = 0.197; the best learned model, GAT+logQ, reaches 0.135), and all models score near chance in AUC (≈0.42–0.60). We trace this to two compounding factors: a strong popularity bias in the historical transfer signal and an extreme patent cold-start regime (**91.7% of test patents are unseen in training**). Our contributions are (1) the evaluation protocol, (2) a suite of bias diagnostics that explain *why* the models fail, and (3) a symmetric grid of partial, unstable mitigations."

**근거:** 실측 NDCG@10 — MostPop 0.197 / NGCF 0.196 최상위, 학습 모델 최댓값 GAT+logQ 0.135. AUC 범위 0.416–0.602. 테스트 특허 91.67% 미관측.

### 1.3 초록/기여 — tie-break 아티팩트 명시 (권장)
**기존:** (언급 없음)

**수정 제안 (기여 문장에 추가):**
> "We additionally report a measurement artifact: a strict greater-than tie-break inflates cold-start models to near-perfect scores by ranking the positive first on ties. Switching to average-rank removes it — e.g. SVD NDCG@10 drops from 0.95 to 0.037 and GraphSAGE+logQ from 1.00 to 0.135 once ties are scored correctly."

**근거:** 91.7% cold-start에서 무정보 모델 점수가 동점 붕괴 → strict-`>`가 만점화하던 것이 가장 큰 평가 결함. 수정 후 SVD 0.037 / logQ 0.135 / GAT 0.072.

### 1.4 기여 진술 (Contributions)
**수정 제안:**
> "(i) *Protocol:* a temporally split, same-IPC hard-negative, average-rank evaluation that, unlike random-split AUC reporting, exposes popularity bias and cold-start. (ii) *Diagnosis:* under this protocol all learned models (GNN/CF/MF) collapse below MostPop; a diagnostic toolset attributes this to popularity bias (e.g. NGCF score–popularity Spearman ρ = 0.96) and cold-start, and uncovers the tie-break inflation artifact. (iii) *Mitigation baselines:* a symmetric mitigation grid (Debias, logQ, DropEdge, time-encoding, IPS, and combinations, over both GraphSAGE and GAT; 23 configurations) yielding only partial, unstable gains — GAT+logQ improves GAT's NDCG@10 from 0.072 to 0.107 (Wilcoxon+Holm significant) but stays below MostPop, while IPS variants raise NDCG@10 to as high as 0.157 yet drive AUC to 0.417 (below chance)."

**근거:** NGCF ρ=0.96, GAT+logQ 0.072→0.107(유의), IPS 변형 NDCG 0.13–0.157 / AUC 0.417. 실제 23개 모델.

---

## 2. §4.1 Demand Score 개선

### 2.1 평가 범위 — 200-쿼리 표본 명시
**기존:** Demand Score를 same-IPC 후보셋에서 평가; 4개 수정(Table 2); "empirical effect … reported in Section 6.1."

**수정 제안:**
> "Unlike the learned models, scored on all ~220k test queries, the Demand Score requires a per-query citation-BFS that is computationally expensive; we therefore evaluate it on a **fixed random sample of 200 test queries** from the same temporal test split, using identical same-IPC candidate sets. Demand-Score numbers are estimates on this sample, reported for diagnostic comparison rather than as a competitive baseline."

**근거:** 코드상 Demand는 per-query 인용 BFS가 매우 느려 200-쿼리 표본으로만 평가(`--demand_sample`). 다른 모델(~220,284 쿼리)과 평가 규모가 다름.

### 2.2 4개 수정의 실효과 — "효과 미미" 명시
**기존:** "We correct four structural deficiencies … their empirical effect is reported in Section 6.1." (개선 함의)

**수정 제안:**
> "We correct four structural deficiencies of the original Demand Score (Table 2) for completeness. **On the 200-query evaluation, however, the revised score is empirically indistinguishable from the original** (NDCG@10 0.241 vs 0.258; the original is in fact marginally higher), so the four fixes do not translate into a measurable ranking improvement under this protocol. We retain Table 2 as a description of the score's construction, not as evidence of a gain, and do not claim the revised Demand Score as a contribution."

Table 2 캡션: *"…These changes are definitional; their measured effect on ranking (200-query sample) is negligible — see §4.1."*

**근거:** 최종 E19 실측 — Demand Original 0.2579 vs Revised 0.2406 (거의 동일, original이 약간 높음). "4개 수정이 효과적"은 데이터로 뒷받침 안 됨.

### 2.3 절 참조 정정
**기존:** "… reported in Section 6.1."
**수정 제안:** "… reported in §6.3 (E19, evaluated on a 200-query sample due to the cost of per-query citation BFS)."
**근거:** Original/Revised 비교는 E19에서 산출. 본문 절 번호를 실제 결과 절과 일치시킬 것. (현 `run_ipm_results.md`의 6.3 표는 stale 값이므로 재생성 후 교체 `[확인 필요]`.)

---

## 3. §4.2 노드 임베딩

### 3.1 §4.2.2 회사 표현 — GNN은 학습형 임베딩이 아님
**기존:** "Company nodes are initialised as a **learnable embedding matrix** of dimension d=64, **jointly optimised** … transductive … a company absent from the training graph has no embedding and **cannot be scored**."

**수정 제안:**
> "Company representations differ by model family. For the GNN encoders (GraphSAGE, GAT and variants), each company is assigned a **fixed random feature vector** $x_c \in \mathbb{R}^{64}$ (drawn once as $\mathcal{N}(0,0.1^2)$, never updated by gradient descent); the only learned company-side parameters are those of a shared linear projection `company_lin`. The GNN path is therefore **not transductive at the embedding level**: an unseen company still receives a random feature row and *can* be scored — though, having received no transfer message, its representation carries no learned signal. Only the collaborative-filtering baselines (LightGCN, NGCF) use a genuinely learnable per-company ID embedding (`nn.Embedding`, d=64)."

**근거:** `run_ipm_experiment.py` `company_x = torch.randn(NUM_COMPANIES, 64) * 0.1` → `data['company'].x` (옵티마이저 미등록, 고정). 학습되는 회사측 파라미터는 `company_lin`뿐. `LightGCN`/`NGCF`만 `nn.Embedding(n_c, dim)`.

### 3.2 "cannot be scored" 정정
**기존:** "…cannot be scored."
**수정 제안:** "No model in this study refuses to score a held-out company: GNN models score it from its random feature, LightGCN/NGCF from an untrained ID embedding. Cold-start manifests as **uninformative** scores, not an inability to produce a score."
**근거:** 모든 후보가 `data['company'].x` 인덱싱으로 점수화됨. 미관측 회사도 평가에 포함(회사 미관측 14.12%, 희소 17.78%).

---

## 4. §4.3 그래프 구성/통계

### 4.1 회사 수 정정 (≈12,400 → 122,519)
**기존:** "Company nodes (**|C| ≈ 12,400**) carry trainable d=64 embeddings. … mean company degree ~3.9."

**수정 제안:**
> "Patent nodes ($|P| = 370{,}666$) carry frozen 384-d SBERT features. Company nodes ($|C| = 122{,}519$) are the prediction targets (GNN encoders use a fixed 64-d random feature per company; only LightGCN/NGCF attach a trainable ID embedding — see §4.2.2). The company-degree distribution is extremely long-tailed: 14.1% of test companies are unseen in training and 17.8% are sparse (≤1 transfer)."

**근거:** 실측 회사 수 122,519 (논문 ≈12,400은 ~10배 오차). transfer/citation 엣지 절대 수치는 run 로그에서 확정 후 기입 `[확인 필요]`.

### 4.2 mean-degree / hub 진술 재서술
**기존:** "…mean company degree ~3.9." (|C|≈12,400 가정 기반)
**수정 제안:** 단일 평균값 대신 "long-tail + hub 집중" 구조로. 정량 근거는 Table 5의 Spearman ρ(MostPop 1.00, NGCF 0.96)·inversion(MostPop 56.4%).
**근거:** 회사 수 정정으로 기존 평균차수 산식이 어긋남. mean degree를 쓰려면 (확정 transfer 엣지 수)/122,519로 재계산 `[확인 필요]`.

---

## 5. §4.4 GNN 아키텍처 (코드 불일치 정정)

### 5.1 §4.4.1 GraphSAGE
**기존:** "…sampling **10 neighbours per layer** … **Two layers (L=2)** … hidden dimension 32."
**수정 제안:**
> "Our GraphSAGE encoder performs a **single message-passing round per edge type** (`company→patent`, `patent→company`, `patent→patent`), each by one `SageLayer` aggregating **the full neighbourhood (no neighbour sampling)** via `scatter_add` normalised by in-degree. Hidden dim = 32, output dim = 16. The effective propagation depth is one hop (L=1)."
**근거:** `sage_c2p/p2c/p2p` edge-type별 1회 호출(1라운드); `neigh_sum.scatter_add_(...)` 전체 이웃 합산(샘플링 없음); `FullModel(hidden_dim=32, out_dim=16)`.

### 5.2 §4.4.2 GAT
**기존:** "**Four attention heads** … concatenate/average … **Two layers** … **Neighbourhood sampling (10 per layer)**."
**수정 제안:**
> "Our GAT encoder uses a **single attention head**: each `GatLayer` learns two attention vectors (`att_self`, `att_neigh`), computes additive coefficients, applies a per-target softmax over the **full neighbourhood**, and aggregates all neighbours via `scatter_add` (no sampling). One `GatLayer` per edge type → a single message-passing round (L=1). Hidden 32, output 16. With one head there is no concatenation/averaging step; the attention diagnostic (§6.3.1) reads per-edge weights directly."
**근거:** `GatLayer`는 `att_self`/`att_neigh` 각 `nn.Parameter(1,out_dim)` 1개 = 단일 헤드; edge-type별 1라운드; `out.scatter_add_(...)` 전체 이웃.

### 5.3 §4.4.5 디코더
**수정 제안:**
> "The decoder scores a (patent, company) pair as the **dot product of their 16-d encoder outputs** (`out_dim=16`), with no extra MLP/bias; the raw dot product is the ranking score (and the training logit, under BCE-with-logits on sampled negatives)."
**근거:** `out_dim=16`, 별도 디코더 MLP 없음. (학습 손실/네거티브 세부는 학습 절에서 코드 기준으로 기술.)

> (LightGCN/NGCF 설명은 대체로 코드와 일치 — 큰 수정 불요.)

---

## 6. §4.5 평가 프로토콜

### 6.1 §4.5.2 지표 — AP 제거, tie-aware AUC
**기존:** "We additionally report **AP (Average Precision)**, for all models including SVD."
**수정 제안:**
> "Because each test query has exactly one positive, AP reduces to the reciprocal rank and coincides with MRR; we therefore **omit AP** and instead report a **tie-aware AUC**: for one positive vs $n_{neg}$ negatives, $\text{AUC} = \Pr(s_{pos}>s_{neg}) + \tfrac12\Pr(s_{pos}=s_{neg})$ averaged over negatives, so ties contribute 0.5 (consistent with `roc_auc_score`). We report Hits@K, MRR, NDCG@K, and tie-aware AUC."
**근거:** 코드: `aps = 1.0/ranks` (= MRR, 미표시), AUC는 `(pos>neg)+0.5*(pos==neg)`로 보고.

### 6.2 §4.5.5 Cold-start — "배제" 정정 + 특허측 명시
**기존:** "When the candidate company is absent (cases ii, iii), the query is **excluded** from the ranking evaluation."
**수정 제안:**
> "We do **not** exclude any query on the company side; unseen companies retain a fixed random feature and are scored normally (14.1% of test positives are unseen, 17.8% sparse). The dominant cold-start regime is on the **patent** side: **91.7% of test patents are unseen in training.** Patents are nominally inductive (SBERT features available regardless of training membership), yet accuracy still collapses on cold patents for SVD (NDCG@10 = 0.000 on new patents) and GAT (0.021), while even GraphSAGE stays low (0.107) — see §6.3.5."
**근거:** 코드에 배제 로직 없음; 모든 후보 점수화. §7 실측: 특허 미관측 91.67%, SVD 신규특허 0.000, GAT 0.021, GraphSAGE 0.107.

### 6.3 §4.5.6 Tie-breaking — 핵심 방법 포인트로 격상 (CN/AA 제거)
**기존:** "**MostPop, Recency, CN, and AA** assign integer or discrete scores, producing ties. All models apply the average-rank tie-breaking rule."
**수정 제안:**
> "Tie-breaking is a load-bearing methodological choice, not a detail confined to discrete baselines. We use the **average-rank** convention, $\text{rank} = \#\{neg>pos\} + \tfrac12\#\{neg=pos\} + 1$, so a no-information model assigning identical scores lands at the middle (chance) rank rather than rank 1. This matters beyond integer baselines: SVD and GAT collapse to all-zero/all-tied scores on cold-start patents (an unseen patent has a zero SVD latent factor, so every candidate scores 0). Under a strict-`>` rule these ties place the positive above every tied negative and hand such models a *perfect* score; average-rank removes this. We adopted the correction after observing the strict rule inflated SVD NDCG@10 from 0.037 to 0.95 and GraphSAGE+logQ from 0.135 to 1.0."
**근거:** `tie_aware_ranks` = average-rank; 수정 전후 SVD 0.037↔0.95, logQ 0.135↔1.0, GAT 0.072↔0.45. **CN/AA는 코드에 미구현 → 문장에서 제거.**

### 6.4 §5.4 Sampled vs full-ranking — 미구현 주장 삭제
**기존:** "We report **full-ranking** results (over all companies) for Random, MostPop, Demand, and GAT."
**수정 제안:** (삭제 또는 한계로)
> "All evaluation uses sampled same-IPC hard negatives. We do **not** perform full-ranking over the full company catalogue (122,519); a full-ranking comparison, and local-structure baselines (Common Neighbors, Adamic-Adar), are left to future work."
**근거:** 코드에 full-ranking 경로 없음(same-IPC hard-neg만). Random/CN/AA도 미구현.

---

## 7. §4.6 완화 + §5 실험 셋업

### 7.1 §5.3 모델 목록 — "Eighteen" + 미구현 베이스라인
**기존:** "**Eighteen** model configurations … Common Neighbors, Adamic-Adar, Random, Demand (orig/rev) …"; 완화 3종.
**수정 제안:**
> "We evaluate **23 model configurations**: nine base models — three non-learning skylines (**MostPop**, **IPC-conditional MostPop**, **Recency**), **SVD** (k=64), a text-only **MLP**, two CF models (**LightGCN**, **NGCF**), two heterogeneous GNNs (**GraphSAGE**, **GAT**) — plus a **symmetric mitigation grid** on *both* GNN backbones {Debias, logQ, DropEdge, Time, IPS} and two stacked combinations (Debias+IPS, Time+IPS) per backbone."
> **삭제:** CN, Adamic-Adar, Random (코드 미구현). **강등:** Demand(orig/rev)는 200-쿼리 표본 진단이므로 메인 Table 4에서 빼고 보충으로.
**근거:** 코드: `BASE_MODELS` 9개 + `MITIGATIONS` 5종×2 backbone + `COMBOS` 2종×2 = 23개 `record_model`. CN/AA/Random 미구현.

### 7.2 §5.5 / Table 3 — 하이퍼파라미터 정정
**기존:** "GAT heads = 4", "neighbour sample size = 10", "GNN layers (L) = 2".
**수정 제안 (해당 행 교체):**
> - **Attention heads: 1** (single head)
> - **Neighbourhood sampling: none** (full-neighbour scatter-add)
> - **Message-passing rounds: 1** (one layer per edge type; effectively L=1)
> - hidden 32 / output 16 / company feature dim 64
> - Tie-breaking: **average-rank**
> - IPS β, Debias α: 단일값 아님 → "see sensitivity sweep"
**근거:** 코드 실측(단일 헤드, 무샘플링, 1라운드, dims). β/α는 스윕.

### 7.3 §4.6 완화 효과 — 정직한 서술
**기존:** 완화 3종을 제안 기법(개선)으로 서술.
**수정 제안:**
> "Mitigations are **partial and unstable; none recovers a learned model above the popularity skyline.** logQ gives the most consistent gain — GAT NDCG@10 0.072 → 0.107 (Wilcoxon+Holm significant) — yet still below MostPop (0.197). IPS re-ranking raises NDCG@10 (up to 0.157 depending on backbone/combo; GraphSAGE+IPS only 0.128) but **degrades AUC to 0.417, below chance**, indicating a popularity-penalty artifact rather than better discrimination. Debiased sampling is high-variance with no stable gain. DropEdge and time-encoding have essentially no effect (§6.5)."
**근거:** GAT+logQ 0.107/0.550(유의), GAT+IPS 0.157/0.417, GraphSAGE+IPS 0.128/0.430, GraphSAGE+Debias 0.075/0.491.

### 7.4 GAT+DropEdge "개선 기법" 주장
**기존:** GAT+DropEdge를 효과 있는 증강으로.
**수정 제안:** "DropEdge on citation edges has **no material effect**: GAT+DropEdge (0.071/0.516) is indistinguishable from GAT (0.072/0.516). We do not claim DropEdge as a contributing mechanism."
**근거:** 실측 거의 동일. 제목/초록의 "with DropEdge" 프레이밍과 충돌.

---

## 8. §6 Results — 실측치로 교체

> 원고의 Table 4/5와 §4.2.x 수치는 **tie-break 수정 전(stale)** 값(SVD 0.95, GAT 0.45, logQ 1.0). 아래로 교체.

### 8.1 Table 4 (메인) — 전면 교체 (10 seeds, average-rank)
**수정 제안 (서술 + 핵심 표):**
> "Under the temporal split with same-IPC hard negatives and average-rank tie-breaking, **every trained model — GNN, CF, MF — falls below the popularity baselines.** The two strongest systems are non-learned: MostPop (NDCG@10 0.197, AUC 0.583) and the popularity-driven NGCF (0.196 / 0.584). All AUC cluster near chance (0.5)."

| Model | NDCG@10 | AUC | | Model | NDCG@10 | AUC |
| :-- | :-: | :-: | :-: | :-- | :-: | :-: |
| MostPop | 0.197 | 0.583 | | GraphSAGE+logQ | 0.135 | 0.546 |
| MostPop-IPC | 0.199 | 0.416 | | GraphSAGE+DropEdge | 0.114 | 0.486 |
| Recency | 0.149 | 0.602 | | GraphSAGE+Time | 0.106 | 0.491 |
| SVD | 0.037 | 0.520 | | GraphSAGE+IPS | 0.128 | 0.430 |
| MLP | 0.150 | 0.576 | | GAT+Debias | 0.063 | 0.509 |
| LightGCN | 0.058 | 0.509 | | GAT+logQ | 0.107 | 0.550 |
| NGCF | 0.196 | 0.584 | | GAT+DropEdge | 0.071 | 0.516 |
| GraphSAGE | 0.096 | 0.482 | | GAT+Time | 0.074 | 0.516 |
| GAT | 0.072 | 0.516 | | GAT+IPS | 0.157 | 0.417 |
| GraphSAGE+Debias | 0.075 | 0.491 | | GraphSAGE+Debias+IPS | 0.155 | 0.417 |
| | | | | GraphSAGE+Time+IPS | 0.129 | 0.435 |
| | | | | GAT+Debias+IPS | 0.157 | 0.417 |
| | | | | GAT+Time+IPS | 0.155 | 0.417 |

> (전체 Hits@1/3/5/10·MRR 열은 `run_ipm_results.md`에서 옮겨 채울 것.)

**근거:** 최종 10-seed. 구표 SVD 0.95/GAT 0.45/logQ 1.0은 cold-start 동점을 strict-`>`로 만점화한 아티팩트.

### 8.2 §6.2 Temporal vs Random split — 미실행
**기존:** "무작위 분할에서는 GNN이 AUC 0.8~0.9…" (측정한 듯 서술)
**수정 제안:** "We do **not** run a controlled random-split baseline; the protocol fixes the temporal 70/15/15 split throughout. Prior random-split evaluations report AUC in the 0.8–0.9 range; a matched random-vs-temporal contrast is left to future work (limitation)."
**근거:** 코드에 random-split 분기 없음. 0.8~0.9는 본 실험 측정값 아님.

### 8.3 §6.3.1 GAT Attention (D12)
**기존:** "…허브 노드에 우선적으로 가중치를 배분하는 경향." (p=1.0과 모순)
**수정 제안:** "A Mann–Whitney U test on hub vs non-hub attention weights gives **p = 1.0**: attention does **not** down-weight hub neighbours — the hypothesis that GAT filters popularity hubs is **rejected**. (The encoder uses a single attention head, limiting hub/non-hub separation.)"
**근거:** D12 실측 p=1.0; GAT 단일 헤드.

### 8.4 §6.3.2 Score–Popularity (D13) + inversion (D15)
**수정 제안 (실측 표):**

| Model | Spearman ρ (score vs train-pop) | Hard-neg inversion |
| :-- | :-: | :-: |
| MostPop | 1.00 | 56.4% |
| NGCF | 0.96 | 56.3% |
| Recency | 0.69 | 50.0% |
| MLP | 0.19 | 18.7% |
| GraphSAGE | 0.07 | 49.8% |
| SVD | 0.02 | 2.4% |
| GAT | −0.05 | 28.3% |

> "NGCF's score is almost perfectly explained by training popularity (ρ = 0.96), effectively reproducing MostPop (ρ = 1.00) — why it nearly matches MostPop in Table 4 despite being 'trained'. GAT (ρ = −0.05) and SVD (ρ = 0.02) are popularity-orthogonal, but decorrelation does **not** yield higher accuracy. SVD's very low inversion (2.4%) reflects orthogonality, not skill (NDCG@10 = 0.037)."

**근거:** 최종 Table 5 실측. (구 `run_ipm_results.md`의 D15=3.4%는 fast 모드 stale.)

### 8.5 §6.3.3 Stratified (D14)
**기존:** "Head 0.391 / Torso 0.453 / Tail 0.700" (구 GAT)
**수정 제안:** "GAT NDCG@10 by positive's training frequency: **Head 0.016, Torso 0.082, Tail 0.355.** GAT is relatively better on cold/tail targets than popular ones, but all three are low in absolute terms."
**근거:** 최종 GAT 계층별 실측. 구값은 동점-만점 아티팩트.

### 8.6 (신규 절) §6.3.5 Patent Cold-Start
**수정 제안:**
> "**The cold-start regime.** 91.7% of test patents are unseen in training. Decomposing NDCG@10 into new-patent (with existing company) vs existing-patent:"

| Model | All | New patent + existing company | Existing patent |
| :-- | :-: | :-: | :-: |
| SVD | 0.037 | 0.000 | 0.439 |
| GAT | 0.072 | 0.021 | 0.021 |
| GraphSAGE | 0.096 | 0.107 | 0.129 |

> "SVD scores **0.000 on new patents** — all its (small) signal comes from the ~8% seen-patent subset, confirming it is purely transductive. No model can lean on seen patents because they are so rare."

**근거:** §7 실측.

### 8.7 (신규 절) §6.3.6 Error-Source Attribution
**수정 제안:**
> "**Why predictions fail.** Attributing each ranking error: for GraphSAGE, **98.5%** of failures are popularity hard-negative inversions, 0.2% rare/new targets, 1.3% residual. For GAT: **57.8% / 8.3% / 33.9%**. GraphSAGE fails almost entirely through popularity bias; GAT spreads errors across popularity and a larger residual, consistent with its ~zero popularity correlation."
**근거:** §8 실측.

### 8.8 §6.4 Mitigation Results
**수정 제안:** §7.3과 동일 (부분·불안정; logQ만 유의·작음; IPS는 AUC chance 이하). 구 alpha/beta sweep 수치(stale)는 최종 스윕값으로 교체 `[확인 필요: run_ipm_results.md §2.3]`.

### 8.9 §6.5 Ablation
**수정 제안:** "Structural regularizers show **no meaningful effect**: GAT+DropEdge (0.071) ≈ GAT (0.072); GAT+Time (0.074), GraphSAGE+Time (0.106) track their backbones. The architecture is minimal (single round, full-neighbour, single-head), so these toggles do not change representational capacity."
**근거:** 실측.

### 8.10 §6.6 Statistics
**기존:** "GAT와 SAGE 차이 통계적으로 유의하지 않음 (Welch t-test p=0.477)."
**수정 제안:** "Across 10 seeds (Wilcoxon + Holm): **GAT vs MostPop is significant — but GAT is lower** (0.072 < 0.197); **GAT+logQ vs GAT is significant** (0.072 → 0.107); **GAT vs GraphSAGE is not significant**. The significant comparisons confirm degradation and a small mitigation effect, not a method advantage."
**근거:** §3 통계. 구 Welch 단건은 Wilcoxon+Holm로 교체.

---

## 9. 재확인 / 재생성 체크리스트

- [ ] **transfer / citation 엣지 절대 수** — run 로그에서 확정 후 §4.3·§5.1 기입 `[확인 필요]`
- [ ] **mean company degree** — (transfer 엣지 수)/122,519 재계산 또는 long-tail 서술로 대체
- [ ] **`run_ipm_results.md` · `walkthrough.md` 재생성** — 현재 stale(SVD 0.95 등). 최종 `ipm_results_final`로 교체
- [ ] **Table 4 전체 Hits/MRR 열** — `ipm_results_final/run_ipm_results.md` §1에서 옮겨 채우기
- [ ] **완화 스윕(β/α) 최종값** — `run_ipm_results.md` §2.3에서 기입
- [ ] **§6.3.2 Recency inversion / GraphSAGE Spearman** — 최종 Table 5 값 사용(Recency 50.0%, GraphSAGE ρ 0.07)
- [ ] **그림 5종** (`gat_attention_violin`, `popularity_stratified`, `ips_rerank_sweep`, `popularity_debiased_sweep`, `horizon_decay`) — `ipm_results_final`의 최신본으로 교체

---

## 부록 A. 코드-논문 불일치 일람 (검증됨)

| 논문 서술 | 실제 코드 | 조치 |
| :-- | :-- | :-- |
| 회사 노드 = 학습형 임베딩(d=64), 미관측 시 점수 불가 | GNN은 **고정 랜덤 피처** + `company_lin`만 학습; 미관측도 점수화. (LightGCN/NGCF만 학습형 ID 임베딩) | §4.2.2/§4.3 정정 |
| GAT 4 heads, concat/avg | **단일 헤드** | §4.4.2 정정 |
| 이웃 10개/레이어 샘플링 | **샘플링 없음**(전체 scatter-add) | §4.4 정정 |
| 2 layers (L=2) | edge-type별 1 레이어 = **1라운드(L=1)** | §4.4/Table 3 정정 |
| CN, Adamic-Adar, Random 베이스라인 | **미구현** | §5.3/§5.4 삭제 |
| full-ranking 평가(전체 회사) | **미구현**(same-IPC만) | §5.4 삭제/한계 |
| AP 보고 | **미보고**(=MRR), AUC 보고 | §4.5.2/§5.2 정정 |
| 18 configurations | **23** | §5.3 정정 |
| 회사 12,400 | **122,519** | §4.3/§5.1 정정 |
| Demand 전체 평가 + 4수정 효과 | **200 표본** + 효과 미미 | §4.1 정정 |
| tie-break(이산 베이스라인만) | average-rank, **cold-start SVD/GAT 동점 붕괴가 핵심** | §4.5.6 격상 |

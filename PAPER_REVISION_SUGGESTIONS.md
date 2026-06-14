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

### 7.5 §5.6 Reproducibility — 하드웨어/런타임 명시 (신규)
**기존:** §5.6에 코드·데이터 가용성은 있으나 **하드웨어·런타임(Colab T4) 미기재**.
**수정 제안 (§5.6에 한 단락 추가):**
> "All experiments were run on a single NVIDIA Tesla T4 GPU (Google Colab Free), using PyTorch and PyTorch Geometric. Under the temporal protocol (n_neg = 100), the full ten-seed run over the 23 model configurations completes in approximately 70 minutes of wall-clock time, with a per-seed cost of roughly 330–360 s; the seed-invariant rule-based Demand Score (evaluated on a 200-query sample because of its per-query citation breadth-first search) and the candidate-set-size sensitivity sweep add a one-time overhead of a few minutes. The compute device is auto-selected in the order CUDA → MPS → CPU and can be overridden; we note that CPU runs are substantially slower and that Apple-MPS does not support the sparse operations used by the LightGCN/NGCF baselines, so a CUDA GPU is the intended environment. Random seeds (torch and numpy) are fixed per run for reproducibility."
**근거:** `torch.cuda.get_device_name(0)` = "Tesla T4"; 측정 per-seed ~330–365s, 10-seed ~70분(데이터 로드+후보생성+학습+진단 포함). device 자동선택 cuda→mps→cpu (`--device`로 오버라이드 가능). MPS sparse 미지원으로 GPU(CUDA) 필요. (Table 3 하단 또는 §5.6 본문에 삽입; 정확한 GPU 시간/메모리는 최종 run 로그로 확정 권장 `[확인 필요]`.)

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

---
---

# 신규 작성 섹션 (원고에 없던 부분 — Abstract / Introduction / Related Work / Limitations / Conclusion)

> 원고 v5는 §4 Methodology에서 시작해 Abstract·Introduction·Related Work·Limitations·Conclusion이 없습니다. 아래는 **신규 작성**(`기존: 없음 → 수정 제안: 영문 초안`). IPM 형식, 과장 없음, 진단(diagnosis) 톤. 인용은 **실재 표준 문헌만** author-year로, 특허 도메인 인용은 `[domain ref — 저자가 채움]`(날조 방지). 결과 의존 수치는 `[random-split 결과로 확정]`. 모든 수치는 **average-rank tie-break 기준 최종 10-seed** 값.

## Abstract

**기존:** (없음 — 신규 작성)

**수정 제안:**

> Predicting which company will acquire the rights to a given patent — patent technology-transfer demand prediction — can be framed as link prediction on a heterogeneous graph that couples a patent–cites–patent citation network with patent–transfer–company edges. We evaluate this task on the Korean KIPRIS dataset (370,666 patents, 122,519 companies) under a realistic protocol: transfer edges are split temporally by registration date, candidates are drawn as same-IPC hard negatives, and ties are resolved by average rank. Under this protocol every learned model we test — heterogeneous GNNs (GraphSAGE, GAT), collaborative filtering (LightGCN, NGCF), matrix factorization (SVD), and a text MLP — falls below a simple most-popular baseline (MostPop NDCG@10 = 0.197; NGCF, whose scores are effectively a popularity ranking at Spearman ρ = 0.96, reaches 0.196, whereas the best genuinely learned model, GAT+logQ, reaches only 0.135), with AUC near chance (0.42–0.60). Two causes account for this: popularity bias, where 98.5% of GraphSAGE and 57.8% of GAT errors are losses to a more popular hard negative and model scores correlate with company popularity up to Spearman ρ = 1.0; and extreme patent cold-start, with 91.7% of test patents unseen in training (SVD NDCG@10 = 0.000 on new patents). We further document an evaluation artifact whereby a strict greater-than tie-break inflates degenerate cold-start models toward perfect scores, which average-rank tie-breaking removes. Our contributions are the evaluation protocol, a suite of bias and cold-start diagnostics, and a set of mitigation baselines (popularity-debiased sampling, IPS / log-popularity re-ranking, logQ correction), which yield only partial and unstable gains.

**근거/메모:** ~250단어. NGCF 괄호 보정 반영(초록-본문 일관성). random-split 비교 문장은 의도적으로 제외(RQ1 미확정) — 원하면 결과 확정 후 "relative to a random split, …" 한 문장 추가.

---

## 1. Introduction

**기존:** (없음 — 신규 작성. 원고가 "three research questions stated in Section 1"을 참조하나 §1이 부재 → RQ를 여기서 정의)

**수정 제안:**

Patent technology transfer — the licensing, assignment, or sale of patent rights from one party to another — is a central mechanism by which research output is converted into commercial use. For technology-holding institutions (universities, public research institutes, and firms with idle portfolios) and for the intermediaries and policy agencies that support commercialization, a recurring operational question is: *given a particular patent, which companies are plausible recipients of its rights?* Answering this question well would shorten the search for licensees, inform targeted matchmaking, and help allocate scarce brokerage effort [Kim & Geum, 2020; Lee et al., 2016]. In the Korean setting in particular, where public technology-transfer programs and the KIPRIS patent registry make large-scale transaction records observable, the question is both economically meaningful and empirically tractable [Rhee et al., 2026; Park & Yoon, 2017]. We refer to this task as *technology-transfer demand prediction*, with the explicit caveat that the supervision available is realized transfers rather than latent demand; the model is trained and evaluated against transactions that actually occurred.

The task has a natural relational structure. Patents cite other patents, patents are transferred to companies, and companies accumulate portfolios of transferred patents. This induces a heterogeneous graph with *patent* and *company* nodes and at least two edge types — patent–cites–patent and patent–transfer–company — in which demand prediction becomes a link-prediction problem: score the likelihood of a (patent, company) transfer edge. Cast this way, the problem invites the now-standard toolkit of graph representation learning and collaborative filtering. Graph neural networks such as GraphSAGE [Hamilton et al., 2017] and graph attention networks [Veličković et al., 2018; Brody et al., 2022] can propagate textual and structural signal across the citation and transfer topology; collaborative-filtering models such as NGCF [Wang et al., 2019] and LightGCN [He et al., 2020] are designed precisely for bipartite interaction graphs of the patent–company kind; and matrix factorization remains a strong implicit-feedback baseline [Rendle et al., 2009]. Patent text, encoded with sentence embeddings [Reimers & Gurevych, 2019], provides node features that should, in principle, transfer to patents not seen during training. The expectation that these methods will outperform simple heuristics is the default working hypothesis in much of the recommendation and link-prediction literature, and it has motivated a growing line of patent-mining and patent-recommendation systems [Krestel et al., 2021; Choi et al., 2019; Chen & Deng, 2023; Liu et al., 2024]. Most directly, recent graph-based patent recommenders report near-perfect retrieval — for instance, a GAT–NGCF firm–patent recommender on Korean technology-transfer data reports Recall@5 = 0.998 and NDCG@5 = 0.997 [Kim et al., 2025] — results that, as we show, do not survive a leakage-free, same-IPC hard-negative re-evaluation.

That expectation, however, is sensitive to how performance is measured. Several evaluation choices that are common in the link-prediction and recommendation literature can systematically overstate how well a learned model would perform in deployment. First, *random edge splits* break the temporal order of transactions and allow a model to be trained on future transfers while being asked to recover past ones, a form of leakage that does not exist when predictions are made forward in time [Meng et al., 2020]. Second, *uniform random negatives* make the ranking task easy: distinguishing a true transfer from an unrelated patent in a different technology area is a far weaker test than distinguishing it from a plausible same-field alternative, and the resulting AUC can be inflated relative to a deployment setting in which all candidates are topically relevant [Krichene & Rendle, 2020]. Third, headline AUC reporting can mask the popularity bias known to dominate recommendation under skewed exposure [Abdollahpouri et al., 2019; Steck, 2018; Zhu et al., 2021], and can hide the behavior of models on cold-start items that carry no interaction history. A realistic protocol must therefore combine temporal ordering, hard negatives drawn from the same technology class, explicit cold-start accounting, and care over how ties in the score function are broken when ranking.

In this paper we evaluate technology-transfer demand prediction under such a protocol and report a primarily diagnostic finding: under a temporal 70/15/15 split of KIPRIS transfers, with same-IPC hard negatives (n_neg = 100) and an average-rank tie-break, *every learned model we test falls below simple popularity baselines.* The dataset comprises 370,666 patents, 122,519 companies, and roughly 910,000 citation edges, yielding about 220,284 temporally-held-out test queries. The best non-learned method, a most-popular ranker, attains NDCG@10 = 0.197; NGCF — which our diagnostics show to be effectively a popularity ranker (score–popularity Spearman ρ = 0.96) — reaches 0.196. The best learned configuration, GAT with a logQ sampled-softmax correction, reaches only NDCG@10 = 0.135, and the AUC of all models lies near or below chance (0.42–0.60). This pattern holds across heterogeneous GNNs (GraphSAGE, GAT), collaborative filtering (LightGCN, NGCF), matrix factorization (SVD), and a text-only MLP. It mirrors a broader concern that learned recommenders often fail to beat well-tuned simple baselines [Ferrari Dacrema et al., 2019] and that easy negatives can inflate GNN link-prediction relative to heuristic hard negatives [Li et al., 2023].

We trace the result to two causes and quantify each. The first is *popularity bias*: 98.5% of GraphSAGE failures and 57.8% of GAT failures are losses to a more-popular same-IPC hard negative; the hard-negative inversion rate (the rate at which a more-popular negative is ranked above the true positive) is 56% for the most-popular ranker and NGCF and 28% for GAT; and score–popularity Spearman correlation is 1.0 for the most-popular baseline and 0.96 for NGCF. A GAT attention analysis (Mann–Whitney U over hub vs. non-hub edges, p = 1.0) shows no systematic down-weighting of popular hubs, consistent with the models reproducing rather than correcting the popularity signal. The second cause is *extreme patent cold-start*: 91.7% of test patents are unseen in training (with 14.1% of test companies unseen and a further 17.8% sparse), and on new patents SVD collapses to NDCG@10 = 0.000. We then examine whether popularity-bias mitigations can recover learned models above the popularity baselines. Across more than 23 configurations — popularity-debiased negative sampling, IPS / log-popularity re-ranking, logQ correction, DropEdge, and a transfer-edge time encoding, applied symmetrically to both GNN backbones — the mitigations are partial and unstable. The logQ correction significantly improves GAT (NDCG@10 0.072 → 0.107, Wilcoxon signed-rank with Holm–Bonferroni correction) but remains below the most-popular baseline; IPS re-ranking raises NDCG@10 to about 0.157 but pushes AUC to 0.417, below chance.

A further contribution is methodological. We identify an evaluation artifact arising from the tie-break rule used when many candidates receive identical scores. Under a strict greater-than rule, cold-start models whose scores are zero or degenerate place the positive first among tied candidates and so appear near-perfect — SVD reaches NDCG@10 = 0.95 and GraphSAGE+logQ reaches 1.00 — purely because their scores fail to differentiate candidates. Replacing the strict rule with an average-rank tie-break removes the artifact and returns these models to their true levels (SVD → 0.037, GraphSAGE+logQ → 0.135). We report this both because it materially changes the numbers and because it is, to our reading, a generic pitfall for any sampled-ranking evaluation of cold-start link prediction, consistent with recent catalogues of widespread offline-evaluation flaws [Hidasi & Czapp, 2023].

These observations motivate three research questions, which we state and answer in the remainder of the paper:
- **RQ1.** Does a realistic protocol — temporal split combined with same-IPC hard negatives — change measured demand-prediction performance relative to a conventional random split? `[random-split 결과로 확정]`
- **RQ2.** Under realistic evaluation, what accounts for the observed performance — popularity bias, patent cold-start, or both — and how can each be quantified?
- **RQ3.** Can popularity-bias mitigations recover learned models to a level above simple popularity baselines?

The contributions of this paper are as follows. (i) We assemble a realistic evaluation protocol for patent technology-transfer demand prediction over the KIPRIS dataset, combining temporal splitting, same-IPC hard negatives, average-rank tie-breaking, ten-seed repetition, Wilcoxon signed-rank tests with Holm–Bonferroni correction, and bootstrap confidence intervals over queries. (ii) Under this protocol we provide a diagnosis: heterogeneous GNNs, collaborative filtering, matrix factorization, and a text MLP all fall below simple popularity baselines, with AUC near chance. (iii) We supply a set of bias-diagnostic tools — GAT hub-vs-non-hub attention testing, score–popularity correlation, hard-negative inversion, popularity-stratified NDCG, patent cold-start decomposition, and error-source attribution — that localize the failure to popularity bias and extreme cold-start. (iv) We evaluate a broad set of mitigations and show they are at best partial, often trading ranking gains for below-chance AUC. (v) We document the strict-tie-break evaluation artifact and the average-rank correction. The aim is not to advance a winning model but to provide an evaluation protocol and a diagnosis that we hope will discipline future work on this task.

The remainder of the paper is organized as follows. Section 2 reviews related work on link prediction, collaborative filtering, popularity bias, cold-start, and patent-domain prediction. Section 3 describes the KIPRIS dataset, the heterogeneous graph construction, and the models evaluated. Section 4 presents the evaluation protocol. Section 5 reports the main results and the bias and cold-start diagnostics. Section 6 evaluates the mitigation strategies. Section 7 discusses limitations and threats to validity, and Section 8 concludes.

**근거/메모:** 모든 수치 FACTS 일치. 저자 할 일: ① `[domain ref]` 4곳 채우기(특허 마이닝·기술이전 선행연구), ② RQ1 답에 random-split 대조 수치 추가, ③ 섹션 번호(2~8)를 최종 목차에 맞춰 조정, ④ 인용 스타일(IPM은 보통 numbered)로 §3 기존 [1] 식과 통일.

---

## 2. Related Work

**기존:** (없음 — 신규 작성)

### 2. (도입)
Our study sits at the intersection of four lines of work: (i) patent analytics and technology-transfer prediction, which motivates the task; (ii) graph neural networks for link prediction and recommendation, which supply the model families we evaluate; (iii) popularity bias and debiasing in recommendation, which frames our central diagnosis; and (iv) evaluation methodology for ranking under temporal and sampled-metric constraints, which our protocol builds on. We also draw on a fifth, classical line — structural link-prediction heuristics — as informative non-learned baselines. The recurring theme is that our contribution is diagnostic rather than method-superiority: we do not propose a model that outperforms prior art, but an evaluation protocol and a set of bias-diagnostic tools that explain *why*, on this task and dataset, learned models do not separate from simple popularity baselines.

### 2.1 Patent Analytics and Technology-Transfer Prediction
Patents have long been studied as a structured, text-rich, and densely interlinked record of innovation, and a substantial body of work uses patent metadata, classification codes, and citation networks to characterize technological trajectories, measure firm-level innovation, and forecast emerging technologies [Krestel et al., 2021; Erdi et al., 2013]. Within this literature, *technology transfer* — the assignment or licensing of patent rights from an inventing entity to an acquiring firm — has been examined both as an economic phenomenon and, more recently, as a prediction target [Kim & Geum, 2020; Rhee et al., 2026]. Prior approaches range from feature-engineered and text (BERT) classifiers over bibliographic and classification signals [Kim & Geum, 2020] to collaborative-filtering over IPC/portfolio structure [Park & Yoon, 2017; Lee et al., 2016] and network/graph formulations that treat assignment as a relation in a patent–firm graph [Kim et al., 2025; Chen & Deng, 2023; Liu et al., 2024]. Notably, several of these graph recommenders report near-perfect retrieval (e.g., Recall@5 = 0.998, NDCG@5 = 0.997 [Kim et al., 2025]) under random-split, sampled-negative evaluation — precisely the evaluation regime our protocol revisits. We frame the task as link prediction on a heterogeneous graph and evaluate it on the Korean KIPRIS corpus (370,666 patents, 122,519 companies, ~910,000 citation edges). Three properties distinguish this setting and motivate our diagnosis: the supervision signal is a *realized* transfer rather than latent demand; the prediction is intrinsically prospective, so a leakage-free formulation must respect temporal order; and the patent side is severely cold-start (under a temporal split, 91.7% of test patents never appear in training).

### 2.2 Graph Neural Networks for Link Prediction and Recommendation
GraphSAGE (Hamilton et al., 2017) introduced inductive neighbor sampling and aggregation; the Graph Attention Network (GAT; Veličković et al., 2018) replaced fixed aggregation weights with learned attention, and GATv2 (Brody et al., 2022) corrected a static-attention limitation in the original parameterization. In the recommendation setting, LightGCN (He et al., 2020) showed that removing feature transformation and nonlinearity from propagation yields a simpler, often stronger collaborative-filtering model, whereas NGCF (Wang et al., 2019) retains feature-transformation and element-wise interaction terms. DropEdge (Rong et al., 2020) randomly removes edges during training as a structural regularizer. We adopt these architectures essentially as published and ask whether, under a leakage-free temporal split with same-IPC hard negatives, they separate from non-learned baselines. They do not: the best learned configuration (GAT with logQ) reaches NDCG@10 = 0.135, below MostPop at 0.197, with AUC near chance (≈0.42–0.60). Subsection 2.3 examines why.

### 2.3 Popularity Bias and Debiasing
A well-documented failure mode of implicit-feedback recommenders is popularity bias: models score frequently interacted items highly regardless of relevance, and standard offline metrics reward this because popular items are also more likely to be the held-out positives (Steck, 2011; Abdollahpouri et al., 2019). Inverse-propensity-scoring (IPS) methods reweight observations by the inverse probability of exposure (Schnabel et al., 2016; Saito et al., 2020); sampling-bias-corrected training, exemplified by the logQ correction of Yi et al. (2019), adjusts sampled-softmax logits by the log-frequency of sampled items; and popularity-aware negative sampling draws harder negatives in proportion to a power of item frequency. The BPR framework (Rendle et al., 2009) provides the pairwise objective much of this work builds on. Popularity bias is the central finding of our diagnosis, made explicit rather than assumed: score–popularity Spearman correlation is 1.0 for MostPop and 0.96 for NGCF (which thus ranks almost identically to a popularity counter); the hard-negative inversion rate is 56% for MostPop and NGCF and 28% for GAT; and 98.5% of GraphSAGE failures and 57.8% of GAT failures are losses to a strictly more-popular hard negative. We evaluate the standard mitigations symmetrically over both GNN backbones (more than 23 configurations): the corrections are partial and unstable — logQ significantly improves GAT (NDCG@10 0.072 → 0.107, Wilcoxon + Holm) yet remains below MostPop, and IPS re-ranking raises NDCG@10 to roughly 0.157 while pushing AUC to 0.417, below chance.

### 2.4 Evaluation Methodology: Temporal Splits, Hard Negatives, and Sampled Metrics
Random edge splits allow a model to be trained on interactions that occur after the test interactions; Meng et al. (2020) document how leakage-prone protocols inflate offline results and argue for time-aware splitting. Easy uniform-random negatives make metrics optimistic, so harder negatives are used (the BPR line, Rendle et al., 2009); however, Krichene and Rendle (2020) show that metrics over a small sampled candidate set can be inconsistent with their full-corpus counterparts and even reorder methods. Our protocol responds to both concerns — temporal 70/15/15 split, 100 same-IPC hard negatives, ten seeds with Wilcoxon + Holm and bootstrap CIs, and an n_neg sensitivity analysis — and surfaces an evaluation artifact that is itself part of the contribution: under a strict greater-than tie-break, cold-start models with degenerate (all-tied) scores inflate to near-perfect (SVD NDCG@10 = 0.95, GraphSAGE+logQ = 1.00) because tied positives are placed first; an average-rank tie-break removes this (the same models fall to 0.037 and 0.135). We therefore treat average-rank tie-breaking as a necessary component of a realistic protocol for cold-start-heavy data.

### 2.5 Structural Link-Prediction Baselines
Before learned representations, link prediction used structural similarity heuristics. Common Neighbors counts shared neighbors, and Adamic–Adar (Adamic & Adar, 2003) down-weights high-degree common neighbors; Liben-Nowell and Kleinberg (2007) systematized these as link-prediction baselines. These heuristics are parameter-free and serve as a sanity check on whether a learned model extracts more than first-order structural signal. We include Common Neighbors and Adamic–Adar, computed by intersecting the citation neighbors of a patent with the patents already held by a candidate company. Where 91.7% of test patents are unseen, such structural signals are often undefined for the query side, and their performance helps separate two explanations for the learned models' weakness: insufficient structural signal versus dominance of a non-structural popularity prior.

**근거/메모:** 인용 전부 실재(검토 통과). 저자 할 일: 2.1의 `[domain ref]` 3곳 채우기; GATv2는 배경 인용으로만(직접 평가 안 함); §3의 numbered 인용과 스타일 통일.

---

## 7. Limitations and Future Work

**기존:** (원고 §4.3 "연구의 한계"가 짧게 존재 — company cold-start·TGN만 언급. 본 섹션으로 확장·영어화하여 대체)

**수정 제안:**

### 7.1 Limitations
**External validity (single dataset, single jurisdiction).** All experiments use the Korean KIPRIS corpus (370,666 patents, 122,519 companies). The popularity bias and cold-start regime we report are properties of this market's transfer record; we cannot claim the same collapse of learned models below a most-popular baseline would reproduce on a market with a different transfer density or company-size distribution. The diagnosis is established for one jurisdiction and is a hypothesis elsewhere.

**Sampled, same-IPC negatives rather than full ranking.** Ranking metrics are computed against n_neg = 100 same-IPC hard negatives per query, not against the full catalogue of 122,519 companies. Sampled metrics are known to be unreliable estimators of full-ranking quality and to reorder methods (Krichene & Rendle, 2020); the same-IPC restriction further conditions the candidate set on a popularity-correlated signal. Our absolute numbers and model ordering are specific to this candidate construction.

**Citation edges are not time-filtered.** The patent–cites–patent edges used for message passing are taken from the full citation graph and are not truncated at the split point. Because a patent can accumulate citations after the cutoff, this admits a possible feature-side leakage on the citation channel even though the supervision (transfer edges) is split strictly by registration date. We have not quantified this effect; it remains an untested confound.

**Single chronological cutoff.** We use a single 70/15/15 temporal split, and do not perform rolling-origin (expanding-window) evaluation, so the stability of the diagnosis across cutoffs is unverified. We also report a controlled random-split contrast `[random-split 결과로 확정]`; the substantially higher AUC values commonly reported under random splits in prior work motivate this contrast.

**Minimal encoder architecture.** The GNN encoders are deliberately small: company nodes carry fixed-random initial features (no content), and message-passing uses a single propagation round with, for GAT, a single attention head. This is a lower bound on what a graph model could extract, chosen to keep the comparison controlled; we cannot exclude that a deeper, multi-head, or content-initialised encoder would behave differently.

**Demand Score evaluated on a small sample.** The rule-based Demand Score requires a per-query citation breadth-first search that is expensive at scale, so it is evaluated on a fixed random sample of 200 test queries rather than the full ~220,000-query test set. Its numbers are point estimates on that sample; the four structural corrections we apply produce no measurable ranking improvement — the original score (NDCG@10 = 0.258) is in fact marginally higher than the revised one (0.241).

**Unanalysed degenerate behaviour on cold patents.** On cold-start patents, SVD and GAT collapse to all-zero or all-tied candidate scores (an unseen patent has a zero latent factor under SVD, so every candidate ties). We handle this at evaluation time with an average-rank tie-break, which removes the inflation artifact (SVD NDCG@10 0.95 → 0.037, GraphSAGE+logQ 1.00 → 0.135). The mechanism that drives GAT specifically toward tied scores on cold patents is not fully characterised; we observe and correct for it without a complete explanation.

### 7.2 Future Work
**A second dataset for external validity.** Replication on an independent market — for example, USPTO patent-assignment records — under the identical protocol would establish whether the diagnosis is a property of patent transfer markets broadly or of KIPRIS specifically.

**Full-ranking and larger candidate sets.** Re-evaluating against the full company catalogue (n_neg = all) and reporting candidate-set-size sensitivity beyond the sampled regime would remove the dependence on sampled negatives. Local-structure baselines (Common Neighbors, Adamic–Adar) fit naturally into this comparison.

**Time-aware structure and temporal models.** Time-filtering the citation edges at the split point would close the feature-side leakage gap, and temporal graph models that consume edge timestamps directly (TGN/TGAT-style encoders) are the natural way to model how the transfer topology evolves.

**Stronger encoders.** GATv2 (Brody et al., 2022), with multi-head and multi-layer configurations, would raise the architectural ceiling above the minimal single-head, single-round encoder used here.

**Content-based company representations.** Replacing the fixed-random company embedding with content-derived features (company text, IPC profile, prior-transfer profile) is the targeted remedy for company cold-start, under which no held-out company can currently be scored from informative evidence (14.1% of test companies unseen, 17.8% sparse).

**Stronger and temporally-aware mitigations.** The mitigations evaluated here are partial and unstable — logQ significantly improves GAT (0.072 → 0.107) yet stays below MostPop, and IPS raises NDCG@10 to ~0.157 while pushing AUC to 0.417, below chance. Mitigations calibrated jointly for ranking and AUC, and coupled to the temporal structure rather than applied as post-hoc re-ranking, are needed before a learned model can be expected to clear the popularity baseline.

**근거/메모:** Demand 방향(orig 0.258 > rev 0.241) 명시, "test companies" 정정, random-split AUC는 비수치로 완화. 저자 할 일: citation 누수 정량화 시 수치 삽입; §7.1의 진단 절 번호 교체.

---

## 8. Conclusion

**기존:** (없음 — 신규 작성)

**수정 제안:**

This study examined patent technology-transfer demand prediction as a link-prediction task over a heterogeneous patent–citation–transfer graph, and asked whether learned models recover useful demand signal once the task is evaluated under a realistic protocol. Across heterogeneous graph neural networks (GraphSAGE, GAT), collaborative-filtering recommenders (LightGCN, NGCF), matrix factorization (SVD), and a text-based MLP, we find that no learned model surpasses a simple popularity baseline when the evaluation uses a temporal split, same-IPC hard negatives, and an average-rank tie-break. The strongest configurations are not learned: MostPop reaches NDCG@10 = 0.197 and NGCF — which our diagnostics show is effectively a popularity ranker (score–popularity Spearman ρ = 0.96) — reaches 0.196, while the best learned model, GAT with logQ correction, reaches only 0.135, and all models report AUC near chance (0.42–0.60). Two compounding causes account for this outcome. First, popularity bias: 98.5% of GraphSAGE errors and 57.8% of GAT errors are losses to a more popular hard negative, the hard-negative inversion rate is 56% for MostPop and NGCF, and the score–popularity correlation reaches ρ = 1.0 for MostPop. Second, extreme patent cold-start: 91.7% of test patents are unseen during training, and SVD scores NDCG@10 = 0.000 on these new patents. The popularity-bias mitigations we evaluate — popularity-debiased negative sampling, IPS and log-popularity re-ranking, logQ sampled-softmax correction, DropEdge, and time encoding, applied symmetrically across both graph backbones over more than 23 configurations — produce only partial and unstable gains: logQ significantly improves GAT (NDCG@10 0.072 → 0.107, Wilcoxon signed-rank with Holm–Bonferroni correction) yet remains below MostPop, and IPS re-ranking raises NDCG@10 to roughly 0.157 only at the cost of driving AUC to 0.417, below chance. We additionally document an evaluation artifact that must be controlled: under a strict greater-than tie-break, cold-start models whose scores are degenerate or zero have all candidates tie, the positive is placed first by construction, and metrics inflate to near-perfect values (SVD NDCG@10 = 0.95, GraphSAGE+logQ = 1.00); replacing the tie-break with average rank removes the artifact and returns these to 0.037 and 0.135 respectively. `[random-split contrast: temporal-vs-random 결과로 확정]`

The contributions of this work are therefore diagnostic rather than a claim of model superiority. We provide (i) a realistic evaluation protocol for patent transfer-demand prediction — temporal splitting, same-IPC hard negatives, average-rank tie-breaking, multi-seed runs with Wilcoxon signed-rank tests under Holm–Bonferroni correction, and bootstrap confidence intervals over queries; (ii) a diagnostic toolset that attributes the observed performance to its sources, including attention hub-versus-non-hub analysis, score–popularity correlation, hard-negative inversion, popularity-stratified NDCG, and a patent cold-start decomposition; and (iii) a set of mitigation baselines that quantify how far, and how unstably, popularity de-biasing can move learned models. The practical implication for this domain is that reported performance is strongly contingent on the evaluation design: results obtained under random splits, sampled negatives, or strict tie-breaking can overstate model quality by a wide margin. We recommend that future work on patent transfer-demand prediction, and on cold-start-heavy link prediction more broadly, report realistic, tie-aware metrics against hard negatives and against a strong popularity skyline, and treat surpassing that skyline — not merely outperforming a weaker learned baseline — as the bar for demonstrating that a model has captured demand signal beyond popularity. Addressing the underlying obstacles will likely require inductive, content-based company representations [Volkovs et al., 2017; Reimers & Gurevych, 2020] and demand signals that are not collinear with historical transfer frequency.

**근거/메모:** 수치 전부 average-rank 확정값. ⚠️ 원고 Table 4가 average-rank 기준으로 재생성된 뒤 게재해야 본 Conclusion과 숫자가 정합(현 `Paper_Methodology_Draft.md` 표는 구버전). 저자 할 일: RQ1 random-split 문장 채우기, 마지막 `[domain ref]`.

---

## 저자 액션 체크리스트 (신규 섹션)
- [ ] `[domain ref — 저자가 채움]` 총 ~7곳: 특허 마이닝/기술이전 예측 실재 선행연구로 교체
- [ ] **RQ1** random-split 대조 결과 → Introduction RQ1·§8·Limitations 3곳에 수치 채우기 (`[random-split 결과로 확정]`)
- [ ] 인용 스타일 통일: 본문 §3의 `[1]` 식 numbered ↔ 신규 섹션 author-year → IPM 표준(보통 numbered)로 일원화
- [ ] 섹션 번호 정렬: 현 원고 3=Methodology/4=Results 체계에 맞춰 본 신규 섹션 번호(1·2·7·8) 재매핑
- [ ] **Table 4를 average-rank 기준 최종 10-seed로 재생성** 후 게재 — 신규 섹션 수치와 자동 정합 (CLAUDE.md: 숫자 바뀌면 results.md 먼저 재생성)

---
---

# IPM 강화용 추가 섹션 (문제정식화 · 구성타당도 · 위협 · 평가 체크리스트 · 신규 ablation)

> 결과와 무관하게 지금 넣을 수 있는, IPM 심사를 단단하게 하는 서술. `기존: 없음 → 수정 제안`.

## P1. Problem Formulation & Notation (신규 §3 도입부)

**기존:** (형식적 문제 정의·표기 부재 — 원고가 §4부터 시작)

**수정 제안:**
> We model patent technology-transfer demand prediction as link prediction on a heterogeneous graph $G=(V,E)$ with node sets $V = P \cup C$ ($P$ patents, $|P|=370{,}666$; $C$ companies, $|C|=122{,}519$) and two undirected-typed edge sets: citations $E_{\text{cite}} \subseteq P\times P$ ($|E_{\text{cite}}|\approx 910{,}000$) and realized transfers $E_{\text{tr}} \subseteq P\times C$. Each patent carries a frozen 384-dimensional sentence embedding $x_p$; company features $x_c$ are fixed-random (or, in one ablation, content-based, §[ablation]). An encoder $f_\theta$ maps nodes to representations $h_p,h_c\in\mathbb{R}^{16}$ using message passing over $E_{\text{cite}}$ and the *training-split* transfers only, and a parameter-free decoder scores a candidate pair by the dot product $s(p,c)=h_p^\top h_c$. For a held-out test transfer $(p,c^{+})$ we form a candidate set $\mathrm{Cand}(p)=\{c^{+}\}\cup C^{-}(p)$, where $C^{-}(p)$ are $n_{\text{neg}}=100$ *same-IPC hard negatives* (companies that transferred a patent of $p$'s four-digit IPC class in training but not $p$), and report the average-rank of $c^{+}$ in $\mathrm{Cand}(p)$.

**Notation (표):**
| Symbol | Meaning |
| :-- | :-- |
| $G=(V,E)$, $V=P\cup C$ | heterogeneous graph; patent / company nodes |
| $E_{\text{cite}}, E_{\text{tr}}$ | citation edges; (realized) transfer edges |
| $x_p\in\mathbb{R}^{384}$, $x_c$ | frozen SBERT patent feature; company feature (random / content) |
| $h_p,h_c\in\mathbb{R}^{16}$ | encoder node representations |
| $s(p,c)=h_p^\top h_c$ | dot-product link score |
| $C^{-}(p)$, $n_{\text{neg}}$ | same-IPC hard negatives; candidate-set size (100) |
| $\mathrm{rank}(c^{+}\mid p)$ | average-rank of the positive among candidates |

**근거:** 코드의 실제 구성(hetero graph, 384-d SBERT, out_dim 16, dot decoder, 100 same-IPC hard-neg, average-rank)과 일치. **권장: 그래프 스키마/파이프라인 그림 1개**(`paper_figures/fig0_pipeline.png`)를 §3에 배치.

## P2. Construct Validity — "realized transfer ≠ latent demand" (위협 명시)

**기존:** Intro에 한 문장만.

**수정 제안 (Threats/Discussion에 한 단락):**
> A construct-validity caveat underlies the whole study: our supervision is *realized* transfers, not latent demand. A company that would value a patent but never transacted it is, by construction, a negative. Consequently a model that perfectly predicted realized transfers would still only recover the historical matching process — including its popularity and exposure biases — rather than an unobserved demand. We therefore frame the target as *transfer-demand as revealed by historical assignment*, and we read the popularity-bias diagnosis (a model reproducing the assignment frequency distribution) partly as a property of the supervision signal itself, not solely of the models.

**근거:** demand=realized transfer라는 조작적 정의의 한계를 리뷰어가 반드시 물음 — 선제 명시.

## P3. Threats to Validity (정식 절)

**수정 제안 (신규 절):**
> **Internal validity.** Transfer supervision is split strictly by registration date and message-passed only on the train split, removing target leakage; popularity/recency/time features are computed from train only. A residual threat is that citation edges are not time-filtered (§Limitations). Tie-breaking is average-rank, which we show is necessary to avoid a strict-`>` inflation artifact.
> **External validity.** Results are on a single jurisdiction (KIPRIS); we cannot claim the collapse generalizes to other markets without replication (we discuss USPTO and public-benchmark replication as future work).
> **Construct validity.** See P2 — the target is realized transfer, an operationalization of demand.
> **Statistical-conclusion validity.** Ten seeds with Wilcoxon signed-rank + Holm–Bonferroni and bootstrap CIs over queries; we report effect sizes (metric differences) alongside adjusted p-values, and explicitly compute the chance floor for the sampled-ranking metrics.

**근거:** IPM은 validity 4분류를 선호. 현 코드/프로토콜이 이미 충족하는 부분(시간분할·통계)과 미충족(외적타당성·citation 누수)을 솔직히 매핑.

## P4. An Evaluation Checklist (패키징된 방법론 기여)

**수정 제안 (박스/표로 §평가 또는 §결론에):**
> We distill the protocol into a reusable checklist for cold-start-heavy link-prediction / recommendation evaluation:
> 1. **Temporal split** by event time (not random) to avoid future-to-past leakage.
> 2. **Hard negatives** from the same semantic class (here, same-IPC), not uniform random.
> 3. **Average-rank tie-breaking** so a no-information (all-tied) model scores at chance, not perfect.
> 4. **A strong popularity skyline** (global and class-conditional MostPop) as the bar to beat — not merely a weaker learned baseline.
> 5. **Tie-aware AUC** reported alongside top-K metrics (a high NDCG with chance AUC signals a tie/degeneracy artifact).
> 6. **Cold-start accounting**: report the unseen fraction (item and user side) and decompose metrics by it.
> 7. **Multi-seed** runs with paired non-parametric tests + multiple-comparison correction, and **candidate-set-size sensitivity**.

**근거:** "patent 음성결과"를 *재사용 가능한 평가 처방*으로 격상 → 방법론 기여를 눈에 보이게(IPM 일반화 요구 충족).

## P5. (신규 ablation) Content vs random company features

**수정 제안 (Results/Ablation 행 추가):**
> A natural objection is that the GNNs underperform only because company nodes carry no content. We therefore ablate the company representation: replacing the fixed-random feature with the **mean frozen-SBERT of each company's training-received patents** (leakage-free; cold companies retain a random fallback). [결과로 확정] We report whether this content representation lets any GNN exceed the popularity baseline; [if it does not] this rules out "uninformative company features" as the sole explanation and strengthens the popularity-bias + cold-start diagnosis.

**근거:** `--company_feat content` 구현 완료(mean train-patent SBERT). 이 ablation을 돌리면 "기업 피처가 랜덤이라 진 것" 반박을 차단. 실행: `--company_feat content` 추가해 한 번 더 → GNN 행만 비교(나머지 동일).

## P6. 정량 보강 메모
- **Chance floor 명시**: random ranking 기준 NDCG@10 ≈ 0.045, AUC = 0.5를 결과표 각주로. (무정보=동점→average-rank rank 51 → NDCG 0이 chance보다도 낮은 이유까지 한 문장.)
- **효과크기**: p값과 함께 평균 NDCG@10 차이(Δ) 보고. 이미 표에 mean±std 있으니 Δ만 명시.
- **"below popularity" 격차의 bootstrap CI**: GAT−MostPop 등 핵심 격차에 쿼리 bootstrap CI(코드의 ranks 저장 활용) — §10 부트스트랩을 모델쌍 차이로 확장 가능.

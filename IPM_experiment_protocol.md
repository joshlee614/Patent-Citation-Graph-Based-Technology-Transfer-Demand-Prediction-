# IPM 제출용 추가 실험 실행 명세서 (Implementation-Ready)
### Patent Technology Transfer Demand Prediction — Heterogeneous GNN
목표 저널: *Information Processing & Management* (Elsevier, JCR IF 13.3 카테고리, Q1) — IR/추천/link-prediction 방법론 심사 기준에 맞춤.
원칙: 진단(diagnosis) 단독 논문에서 **현실적 평가 프로토콜 + 편향 진단 + 부분 완화**의 완결된 방법론 기여로 전환.

> 주의: 어떤 실험 세트도 게재를 "보장"하지는 못합니다. 다만 아래는 IPM 리뷰어가 link-prediction/추천 논문에서 통상 요구하는 항목을 빠짐없이 메우도록 설계되어, 리젝 사유를 구조적으로 제거합니다.

---

## 0. 공통 설정 (모든 실험 공유)

### 0.1 자료구조 규약
```python
# nodes
#   patent ids: 0 .. |P|-1
#   company ids: 0 .. |C|-1
# edges (모두 시간 분할 기준 'train' 그래프에서만 메시지 패싱)
#   edge_index_dict = {
#       ('patent','cites','patent'): LongTensor[2, E_cite],
#       ('patent','transfer','company'): LongTensor[2, E_transfer_train],
#       ('company','rev_transfer','patent'): LongTensor[2, E_transfer_train],
#   }
# x_dict = { 'patent': FloatTensor[|P|, 384],   # SBERT (frozen)
#            'company': nn.Embedding(|C|, d) }   # 학습형

# 평가용
#   test_edges: List[(p, c_pos, ipc4)]              # 시간 분할 test (최근 15%)
#   ipc_company_index: Dict[ipc4 -> set(company)]   # 같은 IPC를 전송한 회사 풀
#   train_pop: np.ndarray[|C|]                      # 학습기 전송 빈도 (degree)
#   company_first_seen: np.ndarray[|C|]             # 학습기 첫 등장 여부/시점
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # ★ 5 → 10으로 증가 (검정력 확보)
KS = (1, 3, 5, 10)
N_HARD_NEG = 100   # per-positive 후보 negative 수 (민감도 실험 E2에서 변화시킴)
```

### 0.2 하드 네거티브 후보 생성 (평가 고정)
```python
def build_candidates(p, c_pos, ipc4, ipc_company_index, train_transfer_set,
                     n_neg=N_HARD_NEG, rng=None):
    """같은 IPC를 전송했지만 patent p는 전송하지 않은 회사 중에서 n_neg 추출."""
    pool = ipc_company_index[ipc4] - {c_pos}
    pool = [c for c in pool if (p, c) not in train_transfer_set]
    if len(pool) > n_neg:
        pool = list(rng.choice(pool, size=n_neg, replace=False))
    return [c_pos] + list(pool)   # index 0 = positive
```
- 후보 셋은 **seed 고정**(평가 재현성). 학습 negative와 분리.

---

## 1. 평가 지표 모듈 — 필수 (A1)

### 1.1 정의 (단일 positive per query 기준)
- 후보 셋 내 positive의 1-indexed 순위를 `rank`라 할 때:
  - **Hits@K** = 1 if `rank ≤ K` else 0
  - **MRR** = 1 / `rank`
  - **NDCG@K** = (1 / log₂(`rank`+1)) if `rank ≤ K` else 0  (IDCG=1이므로 그대로)
  - **Recall@K** = Hits@K (positive 1개이므로 동일 — 논문에 명시)
  - **MAP** = MRR과 동일(단일 positive) — 중복 보고 피하고 MRR만 사용
- 쿼리 전체 평균이 최종 지표.

### 1.2 코드
```python
import numpy as np

def per_query_metrics(rank, ks=KS):
    out = {}
    for k in ks:
        out[f"hits@{k}"] = float(rank <= k)
        out[f"ndcg@{k}"] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
    out["mrr"] = 1.0 / rank
    return out

def aggregate(rank_list, ks=KS):
    metrics = [per_query_metrics(r, ks) for r in rank_list]
    keys = metrics[0].keys()
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}
```

### 1.3 보고 형식
- **Table (main)**: 10개 모델 × {Hits@1, Hits@5, Hits@10, MRR, NDCG@10, AUC, AP}. mean±std (10 seeds).
- AUC/AP는 후방호환용으로 유지하되, **본문 해석은 랭킹 지표 중심**.

### 1.4 Sampled-metric 방어 (A2)
- IPM은 sampled negative 지표를 불신하는 흐름(Krichene & Rendle, 2020)을 인지.
- **둘 중 하나 수행**:
  1. 같은 IPC 후보 전체(샘플링 X)로 full ranking → 지표 산출, 또는
  2. n_neg ∈ {50, 100, 200, all}에서 지표 안정성 보고(→ E2와 통합) + "동일 IPC 풀 자체가 현실적 후보 집합"이라는 정당화 문단.

---

## 2. 평가 프로토콜 루프 — 필수 (A1과 한 몸)

```python
import torch

@torch.no_grad()
def evaluate_model(model, data, test_edges, ipc_index, train_set, rng, ks=KS):
    model.eval()
    h = model.encode(data.x_dict, data.edge_index_dict)   # {'patent':..., 'company':...}
    ranks = []
    for (p, c_pos, ipc4) in test_edges:
        cand = build_candidates(p, c_pos, ipc4, ipc_index, train_set, rng=rng)
        cand_t = torch.tensor(cand, device=h['company'].device)
        score = h['company'][cand_t] @ h['patent'][p]      # dot-product decoder
        order = torch.argsort(score, descending=True)
        rank = (order == 0).nonzero(as_tuple=False).item() + 1
        ranks.append(rank)
    return aggregate(ranks, ks), ranks   # ranks는 D(진단)에서 재사용
```
- **반환된 `ranks`와 후보별 score를 캐싱**해 두면 D절 진단 실험을 추가 학습 없이 수행 가능.

---

## 3. 새 베이스라인 — 필수~강력권장

### 3.1 MostPop (인기도 전용) — ★최우선 (C8)
popularity bias 가설의 직접 검증. 학습 없이 추론.
```python
def mostpop_rank(p, c_pos, cand, train_pop):
    scores = np.array([train_pop[c] for c in cand])   # 학습기 전송 빈도
    order = np.argsort(-scores, kind="stable")
    return int(np.where(order == 0)[0][0]) + 1
```
- **해석 포인트**: MostPop이 hard-negative test에서 chance 이하로 떨어지면 → "인기도만으로는 미래 전송을 못 맞춘다 = 데이터 구조적 한계" 확증. GNN과의 격차 자체가 진단 자료.

### 3.2 Recency 베이스라인 (C9)
```python
def recency_rank(p, c_pos, cand, company_last_active):  # 학습기 마지막 활동 시점
    scores = np.array([company_last_active[c] for c in cand])
    order = np.argsort(-scores, kind="stable")
    return int(np.where(order == 0)[0][0]) + 1
```

### 3.3 LightGCN (C10) — IPM 표준 추천 GNN 베이스라인
patent–company **transfer 이분 그래프**에서 구동(인용 그래프 제외 버전으로 명확화).
```python
import torch, torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, n_p, n_c, dim=64, n_layers=3):
        super().__init__()
        self.emb_p = nn.Embedding(n_p, dim)
        self.emb_c = nn.Embedding(n_c, dim)
        nn.init.normal_(self.emb_p.weight, std=0.1)
        nn.init.normal_(self.emb_c.weight, std=0.1)
        self.n_layers = n_layers
        # norm_adj: 정규화된 이분 인접행렬 (torch.sparse) D^-1/2 A D^-1/2

    def propagate(self, norm_adj):
        x = torch.cat([self.emb_p.weight, self.emb_c.weight], 0)
        outs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(norm_adj, x)
            outs.append(x)
        x = torch.stack(outs, 0).mean(0)         # layer combination
        return x[:self.emb_p.num_embeddings], x[self.emb_p.num_embeddings:]

    def score(self, hp, hc, p_idx, c_idx):
        return (hp[p_idx] * hc[c_idx]).sum(-1)
```
- BPR loss로 학습, 시간 분할/동일 IPC hard-neg 평가는 §2 루프 재사용.

### 3.4 NGCF (C11)
- LightGCN과 동일 데이터·분할에서 NGCF 전파(이웃 변환 + element-wise 상호작용 항) 적용. PyG의 `NGCFConv` 부재 시 직접 구현, 또는 공개 구현 포팅. LightGCN과 한 표에 나란히.

---

## 4. 완화(mitigation) 실험 — 필수 (B). **최소 2개 회복 입증**

### 4.1 Popularity-debiased negative sampling (B4)
학습 negative를 인기도 비례로 추출 → "인기=정답" 단축경로 차단.
```python
def sample_neg_debiased(n_samples, train_pop, alpha=0.75, rng=None):
    prob = (train_pop + 1.0) ** alpha
    prob = prob / prob.sum()
    return rng.choice(len(train_pop), size=n_samples, p=prob)
# alpha ∈ {0.0(uniform), 0.5, 0.75, 1.0} 그리드 → 회복 곡선 보고
```

### 4.2 IPS / popularity-penalty re-ranking (B5)
추론 시 점수에서 인기도 패널티 차감. 재학습 불필요(가장 저렴).
```python
def ips_rerank_rank(p, c_pos, cand, base_scores, train_pop, beta=1.0):
    pen = beta * np.log(np.array([train_pop[c] for c in cand]) + 1.0)
    adj = base_scores - pen
    order = np.argsort(-adj, kind="stable")
    return int(np.where(order == 0)[0][0]) + 1
# beta ∈ {0, 0.5, 1, 2, 4} 스윕 → NDCG@10 vs beta 곡선
```

### 4.3 logQ correction (B6)
- 학습 손실에서 sampled-softmax bias 보정: `logit -= log(q(c))`, `q(c) ∝ train_pop`. GraphSAGE/GAT 디코더에 적용 후 동일 평가.

### 4.4 시간 모델 1종 (B7) — "시간을 모델링하면 회복되는가"
- **옵션 A (간단)**: 노드/엣지에 시점 인코딩 추가. transfer 엣지의 `trRegistrationDate`를 sinusoidal time embedding으로 패턴, GAT 입력에 concat.
- **옵션 B (정식)**: PyG `TGNMemory` 기반 TGN. transfer 이벤트를 시간순 스트림으로 구성.
```python
# 옵션 A 스켈레톤
def time_encoding(ts, dim=16):
    # ts: normalized timestamp in [0,1]
    freqs = torch.exp(torch.arange(0, dim, 2) * (-np.log(1e4)/dim))
    pe = torch.zeros(ts.shape[0], dim)
    pe[:, 0::2] = torch.sin(ts[:, None] * freqs)
    pe[:, 1::2] = torch.cos(ts[:, None] * freqs)
    return pe   # 회사 임베딩 또는 엣지 특징에 concat
```
- **보고**: 시간 모델이 chance 위로 회복하면 본문 핵심 주장("정적 GNN의 시간 일반화 실패")이 강화됨.

### 4.5 완화 실험 공통 보고
- **Figure**: x축=완화 강도(alpha 또는 beta), y축=NDCG@10/AUC, chance(0.5) 수평선. 회복 곡선이 핵심 그림.
- **Table**: baseline(GraphSAGE/GAT) vs 각 완화 적용본 지표 + Wilcoxon p값.

---

## 5. 진단 분석 — 필수 (D). popularity bias를 실증

### 5.1 GAT 어텐션 가중치 분석 (D12) — 논문이 "미검증"이라 한 부분
```python
# GATConv(..., return_attention_weights=True) 사용
out, (edge_index_att, alpha) = gat_conv(x, edge_index, return_attention_weights=True)
# 엣지를 hub/non-hub로 라벨: 회사 degree 상위 5% = hub
is_hub = train_pop[edge_company_ids] >= np.quantile(train_pop, 0.95)
alpha_hub    = alpha[is_hub].cpu().numpy()
alpha_nonhub = alpha[~is_hub].cpu().numpy()
# 분포 비교 + Mann-Whitney U 검정
from scipy.stats import mannwhitneyu
stat, pval = mannwhitneyu(alpha_hub, alpha_nonhub, alternative="less")
```
- **보고**: hub vs non-hub α 분포(violin) + 검정 p값. GAT가 hub를 down-weight 하는지 결론. 안 하면 GAT 선택 근거 재서술.

### 5.2 점수–인기도 상관 (D13)
```python
from scipy.stats import spearmanr
# 모든 후보(p, c)에 대한 모델 점수와 train_pop[c]
rho, p = spearmanr(all_scores, all_company_pops)
```
- 모델별 ρ를 표로. ρ가 높을수록 popularity bias 직접 증거. MostPop과 비교.

### 5.3 인기도 계층별(head/torso/tail) 성능 분해 (D14)
```python
def stratify(test_edges, ranks, train_pop):
    q33, q66 = np.quantile(train_pop[train_pop > 0], [0.33, 0.66])
    buckets = {"head": [], "torso": [], "tail/new": []}
    for (p, c_pos, _), r in zip(test_edges, ranks):
        pop = train_pop[c_pos]
        b = "tail/new" if pop <= q33 else ("torso" if pop <= q66 else "head")
        buckets[b].append(r)
    return {b: aggregate(rs) for b, rs in buckets.items() if rs}
```
- **보고**: 계층별 NDCG@10 막대그래프. 실패가 tail/new에 집중됨을 시각화.

### 5.4 Hard-negative 역전율 (D15) — AUC<0.5 메커니즘 정량화
```python
def inversion_rate(test_edges, all_cand_scores, train_pop, pop_thr_q=0.9):
    thr = np.quantile(train_pop, pop_thr_q)
    inv, tot = 0, 0
    for (p, c_pos, _), (cand, score) in zip(test_edges, all_cand_scores):
        s_pos = score[0]
        for j, c in enumerate(cand[1:], start=1):
            if train_pop[c] >= thr:          # '역사적 인기' 네거티브만
                tot += 1
                inv += int(score[j] > s_pos) # 인기 네거티브가 정답을 이김
    return inv / tot
```
- **보고**: 역전율 한 수치 + 모델별 비교. "정답이 인기 네거티브 아래로 밀린다"를 직접 증명.

### 5.5 Cold-start 정량화 (D16)
```python
frac_unseen = np.mean([train_pop[c_pos] == 0 for (_, c_pos, _) in test_edges])
frac_rare   = np.mean([train_pop[c_pos] <= 1 for (_, c_pos, _) in test_edges])
```
- test positive 중 학습기 미등장/희소 회사 비중 → transductive 한계(§4.7)와 수치 연결.

---

## 6. 강건성·민감도 — 권장

### 6.1 다중 시간 분할(rolling-origin) (E17)
```python
cutoffs = ["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"]
# 각 cutoff: train = ≤cutoff, test = (cutoff, cutoff+12개월]
# 모든 모델 지표를 cutoff별로 산출 → 현상이 특정 분할의 우연이 아님 입증
```
- **보고**: cutoff별 핵심 지표 라인플롯. §4.6 주장 강화.

### 6.2 후보 셋 크기 민감도 (E18)
- `N_HARD_NEG ∈ {50, 100, 200, all}` 변화 시 지표 안정성. §1.4(sampled-metric 방어)와 통합 보고.

### 6.3 원본 vs 개선 Demand Score 실증 (E19) — §3.2 당위성 증명
```python
# 동일 test/후보 셋에서 두 버전의 per-patent 랭킹 점수 산출
ranks_orig    = [demand_score_rank(p, c_pos, cand, version="original")  ...]
ranks_revised = [demand_score_rank(p, c_pos, cand, version="revised")   ...]
# aggregate 비교 + Wilcoxon
```
- **보고**: Table 4에 "Demand Score (original)" 행 추가. 4개 수정의 효과를 수치로.

### 6.4 Seeds + 올바른 검정 (E20)
```python
from scipy.stats import wilcoxon
# per-seed paired 차이에 대해 (Welch's paired 라는 모순 표현 제거)
stat, p = wilcoxon(metric_modelA_per_seed, metric_modelB_per_seed)
```
- 10 seeds. 모든 모델쌍 비교에 Wilcoxon signed-rank(주) + paired t-test(보조) 병기.

---

## 7. 외적 타당성 — 선택(강화용)

- **2차 데이터셋 (F21)**: USPTO patent assignment/reassignment 등에서 동일 파이프라인 재현. popularity bias가 KIPRIS 특유가 아님을 입증.
- 불가 시: "단일 국가 데이터" 한계를 Limitations에 명시 + 외적 타당성 위협 논의.

---

## 8. 실험 매트릭스 (무엇이 어느 표/그림으로 가는가)

| ID | 실험 | 우선순위 | 산출물(논문) | 새 학습 필요 |
|----|------|----------|--------------|--------------|
| A1 | 랭킹 지표 전면화 | 필수 | Table 4 교체 | X (점수 재사용) |
| A2 | sampled-metric 방어 | 필수 | §평가 + 부록 | X |
| B4 | popularity-debiased NS | 필수 | Fig.(회복곡선)+Table | O |
| B5 | IPS re-ranking | 필수 | Fig.(beta 스윕) | X |
| B6 | logQ correction | 권장 | Table | O |
| B7 | 시간 모델(TGN/time-enc) | 필수 | Table | O |
| C8 | MostPop | 필수 | Table 4 행 추가 | X |
| C9 | Recency | 권장 | Table 4 행 추가 | X |
| C10 | LightGCN | 필수 | Table 4 행 추가 | O |
| C11 | NGCF | 권장 | Table 4 행 추가 | O |
| D12 | 어텐션 분석 | 필수 | Fig.(violin)+검정 | X |
| D13 | 점수–인기도 상관 | 필수 | Table(ρ) | X |
| D14 | 계층별 성능 | 필수 | Fig.(막대) | X |
| D15 | 역전율 | 필수 | 본문 수치 | X |
| D16 | cold-start 정량 | 필수 | 본문 수치 | X |
| E17 | rolling-origin | 권장 | Fig.(라인) | O |
| E18 | 후보 크기 민감도 | 권장 | 부록 | X |
| E19 | Demand orig vs revised | 권장 | Table 4 행 | X |
| E20 | seeds10 + Wilcoxon | 필수 | 전 표 각주 | O(재실행) |
| F21 | 2차 데이터셋 | 선택 | §외적타당성 | O |

**구현 순서 권장**: A1 → C8 → D13/D14/D15/D16 (학습 불필요, 기존 점수 재사용으로 빠른 성과) → B5(IPS, 저렴) → C10(LightGCN) → B4/B7(재학습) → D12(어텐션) → E20(seeds 확대) → E17/E19 → F21.

---

## 9. 재현성 체크리스트 (IPM 필수 제출 요건)

- [ ] 모든 모델 하이퍼파라미터 표 (레이어 수, GraphSAGE 이웃 샘플 크기, 회사 임베딩 차원 d, 디코더, 옵티마이저, lr, epoch, early stopping)
- [ ] 10 seeds 전 per-seed 원값 (supplementary)
- [ ] 하드웨어/런타임 (GPU, 학습 시간/모델)
- [ ] **코드·데이터 가용성 명시** (KIPRIS 접근 절차, 전처리 스크립트, 평가 코드 repo)
- [ ] 후보 셋·negative 생성 시드 고정 절차 명시
- [ ] placeholder 인용 전부 확정 (Zhang et al. 2024, PTNS 등)
- [ ] 통계 검정: Wilcoxon signed-rank, 다중비교 보정(Holm–Bonferroni) 명시
- [ ] §4.2 "Nine model configurations" → 실제 개수와 일치하도록 수정

---

## 10. 핵심 메시지(이 실험들이 합쳐졌을 때의 논문 서사)
1. 랜덤 분할·uniform negative 평가는 patent 전송 수요예측 성능을 **과대평가**한다 (A1, E17).
2. 현실적 평가에서 정적 GNN·SVD는 chance 이하로 붕괴하며, 원인은 **popularity bias**다 (C8, D12–D15).
3. 이는 모델 결함이 아니라 데이터의 구조적 속성이며 **부분 완화 가능**하다 (B4–B7).
4. 따라서 본 연구의 기여는 (i) 현실적 평가 프로토콜, (ii) 편향 진단 도구 세트, (iii) 완화 베이스라인이다.

→ 이 4단 서사가 갖춰지면 IPM 리뷰어의 "그래서 무엇을 기여했는가"가 해소됩니다.

# 실험 · 지표 정리 (Experiments & Metrics)

> `run_ipm_experiment.py`가 한 번 실행될 때 **무엇을(모델·진단·완화) 어떤 지표로** 산출하는지 한눈에 정리한 문서.
> 실행: `python run_ipm_experiment.py --mode full --device cpu --data_dir <KIPRIS> --emb_path <emb.pt> --artifact_dir <out>`

---

## 1. 평가 지표 (Metrics)

단일 positive(정답 기업 1개) per 쿼리 기준. 후보셋 = 정답 1 + 같은 IPC hard negative n개.

| 지표 | 의미 | 비고 |
| :-- | :-- | :-- |
| **Hits@K** (1/3/5/10) | 정답이 상위 K 안에 들어온 쿼리 비율 | Recall@K와 동일(positive 1개) |
| **MRR** | 평균 역순위 1/rank | 단일 positive에선 MAP와 동일 |
| **NDCG@10** | 1/log₂(rank+1), rank≤10일 때 | **본문 핵심 지표** |
| **AUC** | rank-AUC = P(정답>네거) + 0.5·P(동점) | T0-1에서 동점 0.5 처리로 교정(진짜 ROC-AUC) |
| ~~AP~~ | (제거) MRR과 수학적으로 동일 | 중복이라 표에서 뺌 |

> 통계: **Wilcoxon signed-rank(주) + paired t-test(보조)**, **Holm-Bonferroni**를 사전등록 비교 family(약 24쌍)에 공동 적용. seed<6이면 경고.
> 불확실성: seed별 std **+ 쿼리단위 bootstrap 95% CI**(NEW-12).

---

## 2. 모델 · 베이스라인 (총 23개)

### 2.1 비학습 베이스라인
| 모델 | 설명 |
| :-- | :-- |
| MostPop | 전역 인기도(학습기 전송 빈도)로 랭킹 |
| **MostPop-IPC** (NEW-9) | **같은 IPC 안에서의 인기도** — 더 강한 인기 skyline |
| Recency | 학습기 마지막 활동 시점 |
| SVD | 전송 이분행렬 절단 SVD(rank 64) |

### 2.2 학습형 베이스라인
| 모델 | 설명 |
| :-- | :-- |
| MLP | SBERT 특허피처 → MLP |
| LightGCN / NGCF | 전송 이분그래프 CF GNN |
| GraphSAGE / GAT | 이종그래프 메시지패싱(인용+전송) |

### 2.3 완화 × backbone 대칭 그리드 (NEW-6 / B6)
각 완화를 **GraphSAGE·GAT 양쪽**에 동일 적용:

| 완화 | 방식 | 재학습 |
| :-- | :-- | :-: |
| +Debias | 인기비례 네거티브 샘플링(α=0.75) | O |
| +logQ (B6) | 손실에서 log q(c) 차감(sampled-softmax 보정) | O |
| +DropEdge | 인용 엣지 20% 드롭 | O |
| +Time | sinusoidal 시점 인코딩 | O |
| +IPS | 추론 시 인기 패널티 재랭킹(β=1) | X |

→ `GraphSAGE+{Debias,logQ,DropEdge,Time,IPS}`, `GAT+{…}` (10개)

### 2.4 완화 조합 (NEW-7)
`{GraphSAGE,GAT}+Debias+IPS`, `{GraphSAGE,GAT}+Time+IPS` (4개)

---

## 3. 진단 (Diagnostics)

| ID | 진단 | 산출물 |
| :-- | :-- | :-- |
| D12 | GAT 어텐션 hub vs non-hub (Mann-Whitney U) | `gat_attention_violin.png` |
| D13 | 점수–인기도 Spearman ρ (전 모델) | Table 5 |
| D14 | head/torso/tail 계층별 NDCG (GAT) | `popularity_stratified.png` |
| D15 | hard-neg 역전율 (전 모델) | Table 5 |
| D16 | 회사측 cold-start 비율 | 헤더 |
| **NEW-1** | **horizon 감쇠**(컷오프 후 0–6/6–12/12–18/18+개월) | `horizon_decay.png` + §5 |
| **NEW-2** | **IPC 섹션(A–H)별** 성능 분해 | §6 |
| **NEW-3** | **특허측 cold-start**(미관측 특허 비율 + 신규특허&기존기업 subset) | §7 |
| **NEW-4** | **오류원인 분해**(인기 hard-neg / 희소·신규 정답 / 의미 잔차) | §8 |
| **NEW-5** | **정성 case study**(top-5 vs 정답) | §9 |
| **NEW-12** | **쿼리단위 bootstrap 95% CI** | §10 |

---

## 4. 완화 민감도 스윕 (B4 / B5 / NEW-8)

| 스윕 | 그리드 | 산출물 |
| :-- | :-- | :-- |
| IPS β | {0, 0.5, 1, 2, 4} × 양 backbone | `ips_rerank_sweep.png` |
| Debias α | {0, **0.25**, 0.5, 0.75, 1} × 양 backbone | `popularity_debiased_sweep.png` |

(NEW-8: α=0.25 추가 + 양 backbone로 확장)

---

## 5. 출력물 (artifact_dir)

- `run_ipm_results.md` — Table 4(23모델×7지표) + Table 5(진단) + §3 통계 + §5–10 신규 진단
- 그림: `gat_attention_violin.png`, `popularity_stratified.png`, `ips_rerank_sweep.png`, `popularity_debiased_sweep.png`, `horizon_decay.png`

---

## 6. 핵심 논지와의 연결 (4단 서사)

1. (현실적 평가) 시간분할 + 같은-IPC hard negative — §평가 프로토콜
2. (붕괴/인기편향) D13·D15·NEW-4(실패의 ~50%가 인기 hard-neg) · MostPop-IPC가 GNN과 비등 → 인기편향 실증
3. (부분완화) Debias/IPS/logQ 양 backbone 대칭 + β/α 회복곡선 + 조합
4. (기여) 평가 프로토콜 + 진단 도구셋 + 완화 베이스라인

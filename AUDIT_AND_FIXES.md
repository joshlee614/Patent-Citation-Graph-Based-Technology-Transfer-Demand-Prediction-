# 코드 전수 검증 · 수정 내역 (Audit & Fixes)

> 대상: `run_ipm_experiment.py` (IPM 제출용 메인 실험 스위트)
> 목적: PM이 **다른 컴퓨터에서 실행**하기 전에, 무엇이 잘못됐고(Audit) 무엇을 어떻게 고쳤는지(Fixes)를 한 문서로 정리.
> 이 문서는 작업이 진행되며 계속 갱신됩니다. 마지막 갱신: Phase 1–2 (검증 하니스 + Tier 0).

---

## 0. 검증 방법 (어떻게 "맞다"를 보장하나)

이 작업 폴더에는 실제 데이터(`kipris-csv/`, `patent_embeddings.pt`)가 없습니다. 그래서 **합성(mock) 데이터로 전체 파이프라인을 실제 실행**하여 런타임·통합 오류를 잡습니다. 숫자 자체는 의미 없지만, **코드가 끝까지 도는지 / 표·그림이 제대로 생성되는지**를 검증합니다.

- `make_mock_data.py` — KIPRIS 스키마와 동일한 작은 가짜 데이터 + 가짜 SBERT 임베딩 생성.
- 검증 실행 커맨드(이 문서 전반에서 사용):
  ```bash
  python make_mock_data.py --out_dir /tmp/mock_kipris --emb_path /tmp/mock_kipris/patent_embeddings.pt --n_companies 60 --n_transfers 1200
  python run_ipm_experiment.py --mode fast --seeds 2 --epochs 2 --n_neg 8 \
      --device cpu --data_dir /tmp/mock_kipris --emb_path /tmp/mock_kipris/patent_embeddings.pt \
      --artifact_dir /tmp/ipm_out
  ```
- 검증 수준 표기:
  - **[smoke]** mock 데이터로 끝까지 실행됨(EXIT=0) + 산출물 생성 확인.
  - **[unit]** 새로 추가한 순수 함수를 합성 입력으로 단위 검증.
  - **[PM-run]** 학습 동역학이 바뀌어 의미 있는 검증은 **실제 데이터 full 실행 필요** — PM 환경에서 확인.

> ⚠️ **중요**: `--device cpu`가 필요했던 이유 — 이 맥에서 device가 `mps`로 자동선택되는데, LightGCN/NGCF가 쓰는 **sparse 텐서가 MPS에서 미지원**이라 크래시합니다(`norm_adj.to(device)`). PM 머신이 CUDA거나 CPU면 문제 없지만, 안전하게 `--device` 옵션을 추가했습니다(아래 T0-4).

---

## 1. 가장 심각한 문제 (출판 차단 수준)

코드 감사(11개 에이전트) 결과, "실험이 부족"한 게 아니라 **보고된 결과의 무결성**에 문제가 있습니다. 리뷰어 시뮬레이션 판정: 현 상태 **Reject(데스크리젝/무결성 회부 가능성)**.

### C1. 커밋된 `run_ipm_results.md`가 코드로 재현 불가 — **[실행으로 재확인됨]**
- 코드가 출력하는 Table 4는 **9열**(`Model Architecture | Negative Sampling | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | NDCG@10 | AUC`)인데, 커밋된 파일은 **6열**(`Model | Hits@1 | Hits@5 | Hits@10 | MRR | NDCG@10`).
- 커밋된 파일엔 `GraphSAGE+logQ = 1.0000 ± 0.0000` 행이 있으나, **코드 모델 리스트(`:929`)는 logQ를 명시적으로 제외**("excluding logQ")하고 `logq_alpha`는 모든 호출부에서 `0.0`이라 **그 행을 만들 코드 경로가 없음**.
- 커밋된 파일은 `Wilcoxon p=5.0e-01` → **2-seed fast 실행값**(10-seed 아님).
- 경로 `/Users/isang-won/.gemini/antigravity-ide/...` → **다른 도구/이전 버전 산출물**로 추정.
- **결론**: 이 파일은 논문 근거로 쓰면 재현성·무결성 리젝. **깨끗한 full 실행으로 통째 재생성 필요(PM).**

### C2. 보고 숫자가 논지와 모순
- 논지: "현실적 평가에서 정적 GNN·SVD가 chance로 붕괴". 그러나 커밋 표는 SVD NDCG@10≈0.98, GAT≈0.93 — chance 근처가 아님. 게다가 `walkthrough.md`(full)와 `run_ipm_results.md`(fast)의 숫자가 서로 다름.
- **대응 방향(연구 판단)**: full 실행 후 (a) 진짜 붕괴면 chance floor `1/(n_neg+1)` 대비로 보고, (b) 아니면 논지를 "인기편향을 *이용해* 높게 유지된다"로 재구성. → 본 코드 작업은 (a)/(b) 어느 쪽이든 근거를 산출하도록 정비.

### C3. Citation 그래프 시간 누수
- transfer(정답) 엣지는 train만 써서 누수 없음. 그러나 **인용 엣지는 날짜 필터가 없음**(citings.csv에서 날짜 컬럼 자체를 안 읽음). GNN 임베딩이 cutoff 이후 미래 인용 구조를 흡수.
- **대응**: NEW-10(citation ablation) + 날짜 가드 필터(컬럼 존재 시)로 처리 예정.

---

## 2. PM 체크표(✓/△/✗) vs 실제 코드 — 5곳 불일치

| ID | PM | 실제 | 핵심 |
| :-- | :-: | :-: | :-- |
| A1 랭킹지표 | ✓ | ✅ | 일치 |
| **A2 sampled 방어** | ✓ | ❌ 없음 | n_neg 스윕 전무 (과대평가) |
| B4 debias NS | ✓ | ✅ | 일치 |
| B5 IPS rerank | ✓ | ✅ | 일치 |
| **B6 logQ** | ✗ | ⚠️ 부분 | 함수 완성·항상 OFF, 결과표 logQ행은 재현불가 (과소평가) |
| B7 시간모델 | △ | ⚠️ 부분 | timestamp만(정식 TGN 없음) — 일치 |
| C8–C11 | ✓ | ✅ | 일치 (CF는 cold-patent에서 random init 한계) |
| D12–D16 | ✓ | ✅ | 일치 (D12·D14는 GAT만, D16은 회사측만) |
| E17 rolling | ✗ | ❌ | 일치 |
| **E18 후보크기** | △ | ❌ 없음 | padding rate만 (과대평가) |
| **E19 demand orig/rev** | ✓ | ⚠️ 사실상 no-op | 두 버전 랭킹이 데이터 따라 동일/거의동일 (과대평가) |
| **E20 통계** | ✓ | ⚠️ 부분 | Holm 7쌍만(66 아님), 커밋결과 2-seed (과대평가) |
| F21 2차데이터 | ✗ | ❌ | 일치 |
| **NEW-8 HP곡선** | ✗ | ⚠️ 부분 | 스윕 이미 있음, alpha=0.25만 결손 (과소평가) |
| NEW-1~7·9~13 | ✗ | ❌ | 일치 (미구현) |

---

## 3. 수정 내역 (완료된 것)

### Phase 1 — 검증 하니스 (신규 파일)
- `make_mock_data.py` 추가. KIPRIS 스키마 호환 가짜 데이터 생성기. **[smoke]로 전 작업 검증의 토대.**
- **수정 전 원본 스크립트가 mock에서 EXIT=0** 임을 먼저 확인(green 기준선) → 이후 변경의 회귀 여부를 귀속 가능.

### Phase 2 — Tier 0 (무결성·신뢰성 정비)

| 코드ID | 무엇이 틀렸나 | 어떻게 고쳤나 | 위치 | 검증 |
| :-- | :-- | :-- | :-- | :-: |
| **T0-1** | `AUC = mean(pos>neg)` — 동점을 0.5가 아닌 0으로 처리해 **진짜 ROC-AUC가 아님**. 정수점수(MostPop)에서 AUC를 인위적으로 깎음. | 동점=0.5 가산하는 **rank-AUC**로 변경 (`(pos>neg)+0.5*(pos==neg)`, sklearn roc_auc_score와 동일). 3곳(`evaluate_embeddings`, `evaluate_mostpop`, `evaluate_recency`). | `:483`, `:519`, `:532` | [smoke]+[unit] AUC≥MRR 전모델 성립 |
| **T0-2** | Table 4의 `AP` 열이 `1/rank`로 **MRR과 글자단위 동일**(중복 정보). AUC는 계산만 하고 표엔 미표시. | 표의 AP 열을 **AUC 열로 교체** + 정직한 각주("AUC는 단일positive rank-AUC; MAP/AP는 MRR과 동일하여 생략"). | `:1332`–`:1349` | [smoke] AUC≠MRR 확인 |
| **T0-3** | Holm-Bonferroni가 **손으로 고른 7쌍에만** 적용(전체 66쌍 아님) → 다중비교 보정이 과소(p-hacking 소지). fast모드(<6 seed)에서 **전 p값이 조용히 1.0**으로 떨어져 "유의하지 않음"과 구분 불가. | 전체 `C(len(models),2)` 쌍으로 Holm 공동 보정. primary 7쌍은 ★로 표시·선두 정렬. **seed<6이면 명시적 WARNING** 출력. | `:1209`–`:1253` | [smoke] family=66, ★ 선두, 경고 출력 |
| **T0-4** | `--artifact_dir` 기본값이 **타인 머신 경로**(`/Users/isang-won/...`). sparse+MPS 크래시로 일부 머신에서 실행 불가. 데이터 경로 하드코딩. | 기본값을 `./ipm_artifacts`로 교정 + 자동 생성. `--device {auto,cpu,cuda,mps}`, `--data_dir`, `--emb_path` CLI 추가(기본 동작 불변). | `:669`–`:709` | [smoke] CPU 실행 green |
| **T0-5** | fast/full 로그가 실제 seed/epoch/n_neg와 무관하게 "2 seeds, 5 epochs"로 **고정 출력**(오해 소지). | 실제 값으로 포맷. | `:676`, `:681` | [smoke] |

> **C1(재현불가 결과파일) 처리**: 코드의 출력 스키마/내용을 올바르게 정비했으므로, **PM이 실제 데이터로 full 실행하면 신뢰 가능한 `run_ipm_results.md`가 새로 생성**됩니다. 기존 커밋 파일은 폐기 권장(코드 산출물 아님).

### Phase 3 — 대칭 완화 그리드 (B6 / NEW-6 / NEW-7 / NEW-8)

가장 큰 리팩터. 기존엔 완화 기법이 **backbone에 비대칭으로 고정**(Debias·IPS는 GraphSAGE만, DropEdge·Time은 GAT만)이라 "효과가 기법 때문인지 backbone 때문인지" 분리 불가였습니다(내적타당성 결함). 이를 **(backbone × 완화) 대칭 그리드**로 재구성.

| 코드ID | 무엇이 틀렸나 | 어떻게 고쳤나 | 검증 |
| :-- | :-- | :-- | :-: |
| **B6** | `logQ` 손실함수는 완성돼 있으나 `logq_alpha=0`으로 **한 번도 활성화 안 됨**, 모델 리스트에도 없음. 결과표의 logQ 행은 코드 산출물 아님. | `GraphSAGE+logQ`·`GAT+logQ`를 **실제 학습**(logq_alpha=1.0)해 Table 4·진단·통계에 포함. `compute_loss_logq`도 견고화(numpy 명시). | [smoke] logQ 행 생성·실값 |
| **NEW-6** | 완화×backbone 비대칭(교란). | `BACKBONES × MITIGATIONS` 루프로 **모든 완화를 GraphSAGE·GAT 양쪽**에 동일 seed·후보셋으로 적용. `time_enc`를 `train_gnn`에 통합해 GAT 전용 수기 학습 루프 제거(SAGE+Time도 가능). 모델 12→**22개**. | [smoke] 22행, 대칭 쌍 존재 |
| **NEW-7** | 완화 조합 없음(IPS는 plain 모델에만 적용). | `+Debias+IPS`, `+Time+IPS`를 양 backbone에 추가(학습된 모델 재사용, 재학습 없음). | [smoke] 조합 행 존재 |
| **NEW-8** | alpha 그리드에 0.25 결손, 스윕이 SAGE만. | `alpha=0.25` 추가 + beta/alpha 스윕을 **양 backbone**으로 확장, 그림도 backbone별 곡선. | [smoke] 양 backbone 스윕 출력 |

> 통계 family: 모델이 22개가 되며 all-pairs(C(22,2)=231)는 과다 → **사전등록(pre-registered) 비교 family 21쌍**(아키텍처 1 + backbone×강베이스라인 6 + 각 완화/조합 vs 자기 backbone 14)으로 명시하고 Holm을 그 family에 공동 적용. (T0-3의 all-pairs를 이 사전등록 방식으로 대체 — 감사 권고 "정당화된 사전등록 부분집합" 충족.)
>
> ⚠️ **비용 주의(PM)**: 모델이 22개 + 양-backbone 스윕(재학습)으로 늘어, **full 실행 시간이 이전보다 크게 증가**합니다(대략 2~3배). 필요시 `--epochs`/`--seeds`로 조절.

---

## 4. 전체 진행 상태 (18종 + Tier 0)

상태: ✅완료·검증 / 🔧진행예정 / ⏳대기

| 항목 | 내용 | 상태 | 재학습 |
| :-- | :-- | :-: | :-: |
| T0 | 무결성·신뢰성 정비(AUC/AP·Holm·device·경로·로그) | ✅ | - |
| B6 | logQ 모델 활성화(GraphSAGE+logQ / GAT+logQ) | ✅ | O |
| NEW-6 | 완화×backbone 대칭 매트릭스 | ✅ | O |
| NEW-7 | 완화 조합(Debias+IPS, Time+IPS) | ✅ | X |
| NEW-8 | HP 곡선 완성(alpha=0.25, 양 backbone) | ✅ | O |
| NEW-1 | horizon 감쇠 | ✅ | X |
| NEW-2 | IPC 분야별 분해 | ✅ | X |
| NEW-3 | 특허 cold-start subset | ✅ | X |
| NEW-4 | 오류원인 분해 | ✅ | X |
| NEW-5 | 정성 case study | ✅ | X |
| NEW-9 | IPC-conditional MostPop | ✅ | X |
| NEW-12 | bootstrap CI(쿼리단위) | ✅ | X |
| NEW-10 | citation/reverse-edge ablation | ⏳ | O |
| NEW-13 | train negative 비율 민감도 | ⏳ | O |
| E18 | candidate 크기 스윕 | ⏳ | X(평가만) |
| B7 | 정식 TGAT 시간모델 | ⏳ | O |
| E17 | rolling-origin 다중 컷오프 | ⏳ | O |
| NEW-11 | SBERT 변형 robustness | ⏳ | O |
| F21 | 2차 데이터셋(USPTO) 추상화 | ⏳ | O(외부데이터) |

---

## 5. 새 CLI 옵션 (PM 실행 참고)

```bash
python run_ipm_experiment.py \
  --mode full \                 # full=10seed/50ep/100neg, fast=2/5/20
  --device auto \               # auto(cuda>mps>cpu) | cpu | cuda | mps   ← MPS sparse 미지원 시 cpu
  --data_dir kipris-csv \       # KIPRIS CSV 폴더
  --emb_path patent_embeddings.pt \   # SBERT 임베딩(.pt)
  --artifact_dir ./ipm_artifacts      # 결과·그림 출력 폴더(자동 생성)
```

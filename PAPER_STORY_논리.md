# 이 논문의 전체 이야기 (남에게 설명하는 버전)

> 핵심 명제: **"이 논문이 어떤 주장을 하느냐가 어떤 실험을 강제한다."** 실험의 당위성은 거기서 나온다.

---

## 1. 원래 논문은 무엇이었나
처음 형태는 전형적인 **method 논문**이었다 — *"우리가 이종 GNN(GAT 등)으로 특허 기술이전 수요를 잘 예측한다."* 특허–인용–기업 그래프에서 링크예측을 하고, 선행연구(예: GAT–NGCF가 Recall@5 = 0.998)처럼 높은 점수를 보고하는 게 목표였다. 즉 기여는 **"방법의 우월성"** 에 걸려 있었다.

## 2. 무슨 위기에 부딪혔나
정직하게 **현실적 평가**(시간 분할 + 동일-IPC 하드네거티브 + average-rank tie-break)로 돌리자, 모든 학습모델이 **MostPop 아래 + AUC ≈ chance**로 무너졌다. 그러면 **"우리 방법이 작동한다"** 는 주장은 성립 불가다. 논문이 죽는다 — *method 논문으로서는.*

## 3. 어떻게 살렸나 — 입증책임(burden of proof)의 이동
"방법이 좋다"가 불가능하면, 기여를 **옮길 수 있는 자리는 셋뿐**이다. 이게 논문의 새 뼈대다:

- **(a) 측정에 관한 주장** — 기존 평가가 성능을 *과대평가*했다.
- **(b) 메커니즘에 관한 주장** — 실패의 원인은 모델 버그가 아니라 *popularity bias + 시간 이동(cold-start)* 이다.
- **(c) 가용성에 관한 주장** — 이 편향은 *부분적으로* 완화 가능하다.

각 주장은 고유한 입증책임을 진다. **각 실험은 그 책임을 갚으려고 존재한다.** 그래서 "왜 이 실험을 했나?"의 답은 언제나 **"어떤 주장을 방어하려고"** 로 환원된다. 이게 당위성의 근원이다.

## 4. 세 주장이 강제한 실험들

### 주장 (a) "기존 평가가 과대평가했다"
- **temporal vs random split 대조 (RQ1)** — *필수.* (a)는 "방법"이 아니라 "측정"에 관한 주장이라, **같은 모델·같은 데이터에서 두 평가를 대조**하지 않으면 말 자체가 안 된다. random split = 누수의 메커니즘이니, 그걸 끄고 켜서 숫자가 무너지는 걸(SVD 0.44→0.04) 보여야 "누수가 부풀렸다"가 증명된다. → *우리 결과: 랜덤이면 SVD가 최고(0.442), 시간분할이면 0.037. 정확히 뒤집힘.*
- **same-IPC hard negative** — 과대평가는 누수만이 아니라 **너무 쉬운 negative**에서도 온다. 분야가 전혀 다른 회사를 negative로 쓰면 "분야 구분"만 해도 AUC가 높다. hard negative라야 "분야 구분"과 "수요 구분"이 분리되고, "기존 숫자는 분야 구분 점수였다"는 진단이 가능해진다.
- **sampled vs full-pool 대조 (§5.6)** — hard negative로 가는 순간 스스로 구멍을 연다: *"네 sampled metric(100개)은 믿을 만하냐(Krichene & Rendle)?"* 자기가 연 반론을 자기가 닫는다. 전체 풀(평균 900)에서도 순위 유지.
- **rolling-origin (§5.9)** — (a)는 "현실 평가가 어려움을 드러낸다"인데, **단일 70/15/15 컷 하나로는 "그 한 날짜가 운 나빴던 것"** 에 답을 못 한다. 3개 컷오프로 굴려야 단일 관측이 규칙성으로 격상된다. → *우리: 3개 시점 전부 학습모델 < 인기. (★ 이번에 추가해서 이 구멍을 닫음.)*

### 주장 (b) "원인은 popularity bias다" — 가장 무거운 책임
sub-chance 결과의 **기본 반박은 항상 "네 모델이 버그거나 튜닝이 나쁘다"**. 이걸 못 막으면 (b)는 전부 무너진다. 그래서 이 묶음은 *확증*이 아니라 **반증 가능한 severe test(죽을 수 있는 시험)** 로 설계돼야 힘이 생긴다 — "실패할 수도 있었는데 살아남았다"여야 증거다.

- **MostPop = 축이자 control** — 인기도만으로 랭킹하는 자명한 베이스라인이 *같이* 무너지거나 GNN과 비슷하면, 실패는 GNN 구현 탓일 수 없다 — 신호 자체의 성질이다. "나쁜 모델" 가설을 배제하는 control. 없으면 (b)는 반증 불가능 = 과학적 주장이 못 됨.
- **Spearman ρ (D13)** — "bias"에 *측정 가능한 지시 대상*을 준다. 예측이 과거 빈도를 따라가면 ρ가 높다 = "모델이 demand가 아니라 popularity를 한다"의 조작적 정의. → *NGCF ρ=0.96, MostPop 1.0.*
- **계층 분해 (D14)** — severe test. popularity bias 가설은 "실패가 저빈도/신규에 집중"이라는 *구체적 패턴*을 예측한다. 균일하면 가설 반증. 죽을 수 있는 예측을 던져 살아남아서 증거가 된다.
- **inversion rate (D15)** — 집계 이상(AUC<0.5)을 *개별 사례 메커니즘*으로 내린다: "인기 negative가 진짜 파트너를 실제로 추월한다". AUC<chance는 반직관적이라 "라벨 뒤집힌 버그?" 의심을 부르는데, 사례로 보여줘야 풀린다. → *56%.*
- **attention 분석 (D12)** — 논문이 *스스로 한 주장* 때문에 강제. "GAT가 hub를 down-weight 한다"를 함의했으면, 검증 안 된 자기 주장은 부채. 확인(p=1.0, 안 눌러줌)하든 철회하든 해야 함.
- **cold-start 정량화 (D16)** — 주장의 *범위*를 못 박는다. "신규라 구조적으로 점수화 불가"인 실패와 "모델이 잘못 랭킹한" 실패는 다른 종류인데, 안 나누면 비평가가 둘을 섞는다. → *91.7% 미관측, SVD 신규특허 0.000.*

### 주장 (c) "부분 완화 가능"
- 세 완화(**debiased NS / IPS / time**)가 *각각 다른 가설적 원인*(학습 신호 불균형 / 추론기 노출 / 시간정보 부재)을 때리도록 설계된 게 논리적으로 중요. 하나가 회복시키면 → 그게 (b)에서 지목한 원인이 맞았다는 **역방향 확증** → (b)와 (c)가 서로를 받친다.
- **양 backbone 대칭 적용 (control)** — 회복이 "기법" 때문인지 "backbone" 때문인지 분리하려면 backbone 고정·기법만 변경. → *우리는 IPS/logQ/Debias/DropEdge/Time을 GraphSAGE·GAT 양쪽에 대칭 적용. (★ 통제 빈틈을 메운 설계.)*
- **민감도 곡선 (β, α)** — "튜닝된 한 점만 좋은 cherry-picking"을 막음. **조합(Debias+IPS 등)** — "단독으로 chance 못 넘으면 천장이 근본적인가 아직 안 건드린 건가"에 답. → discussion에서 "구조적 한계"를 얼마나 강하게 말할지 결정. *결과: logQ만 소폭 유의, IPS는 AUC를 chance 이하로. 부분적·불안정.*

### 모든 주장을 떠받치는 위생(hygiene) 실험
- **Demand 원본 vs 개선 (E19)** — "네 가지를 개선했다"고 단정했으니, 측정 없는 개선 주장은 순환논증. 자기 책임 갚기.
- **10 seeds + Wilcoxon+Holm (E20)** — 모든 수치 비교는 암묵적으로 "이 차이가 진짜"를 주장. 분산·검정 없으면 seed 노이즈. GNN std가 커서 더더욱.
- **LightGCN/NGCF (구조 control)** — "약한 GNN이라 실패?"를 배제. 추천 정통 모델도 같이 실패하면 "과제가 어렵다"가 특정 인코더를 넘어 일반화. (MostPop=신호 control, 이건=구조 control.)
- **bootstrap CI** — seed std는 *모델 확률성*만, 쿼리 bootstrap은 *"하필 이 test set"* 불확실성을 잡음. 둘 다 보고해야 나머지가 통제됨.
- **content 회사피처 ablation (§5.8)** — "GNN이 회사피처가 랜덤이라 진 거 아냐?"를 배제. content로도 안 됨(오히려 악화). (★ 이번 추가.)
- **아키텍처/하이퍼파라미터 sweep (§5.10)** — "(b)의 기본 반박" 중 "튜닝/크기 부족"을 정면으로 배제. 24설정 최고 0.150 < 0.197. (★ 이번 추가.)

## 5. 어떤 선행연구들과 엮이나 (related work = 각 주장의 외부 근거)
- **(a) 측정** — Krichene & Rendle 2020(sampled metric 불일치), Meng 2020·Ji 2023(split 누수), Cañamares 2020·Ihemelandu 2023(target sampling), Ferrari Dacrema 2019(신경추천이 튜닝된 baseline 못 이김), Li 2023 HeaRT(쉬운 negative가 GNN 링크예측을 부풀림), Hidasi & Czapp 2023(오프라인 평가 결함 — 우리 tie-break 아티팩트의 근거).
- **(b) 메커니즘** — Abdollahpouri 2019·Steck 2018·Zhu 2021(인기편향), 그리고 **IPM 게재** Boratto 2021·Boratto 2023·Deldjoo 2021(인기/공정성·데이터 특성) → "같은 저널의 문제의식 위에 있다".
- **모델 자체** — Hamilton 2017(GraphSAGE), Veličković 2018·Brody 2022(GAT/v2), He 2020(LightGCN), Wang 2019(NGCF), Rendle 2009(BPR), Reimers 2019/2020(SBERT).
- **(c) 완화** — Schnabel 2016·Saito 2020(IPS), Yi 2019(logQ), Rong 2020(DropEdge), Volkovs 2017(cold-start).
- **동기/대조** — Kim 2025(GAT–NGCF, Recall@5 0.998) = "우리가 재평가하는 바로 그 near-perfect"; Kim&Geum 2020·Park&Yoon 2017·Rhee 2026 등(특허 전이/기회 발굴).

## 6. 지금 무엇이 채워졌고 / 무엇이 아직 부족한가 (★ 정확한 현재 상태)

**이미 채워진 것** (manuscript 또는 결과파일에 있음 — 일부는 이번 작업에서 추가):
RQ1 random 대조 · 동일-IPC 하드네거 · full-pool(샘플드 방어) · **rolling-origin(추가)** · MostPop & **MostPop-IPC**(동일-IPC 인기 control) · Spearman/inversion/attention/stratified/cold-start/error-source · **양 backbone 대칭 완화 + β/α 곡선 + 조합** · LightGCN/NGCF · 10seed+Wilcoxon+Holm+bootstrap · Demand score · per-IPC 분해(결과md) · horizon 감쇠(결과md, KIPRIS에선 거의 degenerate) · **content ablation(추가)** · **아키텍처 sweep(추가)**.

→ **즉 당신이 앞서 "부재"로 보신 것들(rolling-origin, IPC-조건부 MostPop, per-IPC, backbone 통제, bootstrap 등)은 현재 버전에선 대부분 이미 해결됐습니다.**

**아직 진짜로 빈 곳** (각 빈틈 = 무방비로 남는 주장):
1. **2차 데이터(USPTO/EPO) — 가장 큰 구멍.** "현실 평가가 어려움을 드러낸다 / 이건 일반적 교훈"이 **단일 관할(KIPRIS)** 에 인질로 잡힘. 일반성 주장은 독립 사례 ≥2개를 요구. → *Major revision 1순위. (보류 중.)*
2. **citation-graph ablation.** 우리가 citation 토폴로지를 쓰는데, 그게 *뭔가 기여하는지* 한 번도 분리 검증 안 함. (제목을 줄이며 "Citation Graph-Based"를 뺐기에 *내부 모순*은 완화됐지만, "인용이 신호를 주는가"는 여전히 미검증. LightGCN/NGCF가 transfer-only라 부분 비교축은 있음.)
3. **정식 시간 GNN(TGN 등).** time concat은 "시간 모델링이 돕는가"의 약한 시험이라, 결론을 underpower.
4. **구성적(positive) 처방 부재.** 전부 negative. "진단만 하고 처방은?"에 약함(프로토콜+implications로 일부 방어).
5. **patent inductive 부분집합 명시 측정** — cold-start 분해로 부분 커버되나, "patent 노드 inductive"라는 자기 주장을 그 부분집합에서 직접 측정하면 더 단단.

## 7. 한 입으로 (엘리베이터 버전)
> "원래는 'GNN으로 특허 전이 수요 잘 맞춘다'는 방법 논문이었는데, 정직하게 현실적으로 평가하니 인기 베이스라인보다도 못했다. 그래서 기여를 **(a) 기존 평가가 과대평가였다 (b) 원인은 popularity bias+cold-start다 (c) 완화는 부분적이다** 로 옮겼다. 그러면 각 주장이 입증책임을 지고, **실험 하나하나가 그 책임을 갚거나 회의론자의 반박 하나를 닫는다** — temporal/random 대조·hard negative·full-pool·rolling-origin은 (a)를, MostPop control·Spearman·계층·inversion·attention·cold-start는 (b)를, 대칭 완화·민감도 곡선은 (c)를 방어한다. 남은 가장 큰 빈틈은 **2차 데이터(외적 타당성)** 와 **구성적 처방**이다."

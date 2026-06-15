# 초안 → IPM(Elsevier) 투고본: 무엇을 / 어떻게 / 왜 바꿨나

> 남에게 설명할 수 있게 정리한 문서. 두 시점의 차이 = **(A) 처음 마크다운 초안** vs **(B) Elsevier 가이드라인 + elsarticle 템플릿을 받은 뒤의 투고 패키지**.

---

## 0. 한 문장 요약
"내용이 완성된 **마크다운 초안 한 개**"를, **IPM(이중익명·Elsevier)의 규정대로** 형식·구조·필수요소·그림·문장을 전부 맞춘 **투고 패키지(익명 본문 + 별도 title page + 커버레터 + 하이라이트 + 그림 9 + 참고문헌)** 로 변환했다.

## 1. 출발점 (A — 처음 초안)
- `PAPER_DRAFT.md` 하나에 Abstract→Conclusion이 다 들어간 **마크다운** 문서.
- 인용은 `(Hamilton et al., 2017)` 식 **author–year**, 표·그림은 마크다운, 저자 블록 없음, 제목은 구어체로 길었음(`When Learned Models Lose to Popularity…`).
- 즉 "**내용은 맞지만 저널 형식이 아닌**" 상태.

## 2. 핵심 변환표 (무엇을 → 왜)
| 항목 | (A) 처음 초안 | (B) 투고본 | 왜 (어떤 규정/이유) |
|---|---|---|---|
| **파일 형식** | Markdown 1개 | **elsarticle LaTeX `.tex`** | 가이드: "editable source(.tex), **PDF는 소스로 불가**" |
| **인용 스타일** | author–year | **numbered `[6]`** + `thebibliography`(42편) | elsarticle-num 스타일, IPM 관행 |
| **파일 구조** | 저자+본문 한 덩어리 | **익명 본문 `manuscript.tex` + 별도 `title_page.tex`** | 가이드: "**double-anonymized** → title page와 anonymized manuscript를 **별도 파일**로" |
| **저자 정보** | (없음) | title page에만: 4인·소속·**교신=이동규**·CRediT·선언 | 이중익명(본문엔 저자 금지) + CRediT 필수 |
| **표** | 마크다운 표 | **booktabs (세로줄·음영 없음)** | 가이드: "avoid vertical rules and shading" |
| **그림** | 옛 그림 일부(`files/`) | **`Figure_1~9` 별도파일·300dpi PNG+벡터 PDF·Times New Roman** | 가이드: "각 그림 별도파일, `Figure_N` 명명, ≥300dpi/벡터" |
| **Abstract** | ~260단어 | **247단어**, 인용 0 | 가이드: "**≤250 words**, 인용 지양, 단독으로 성립" |
| **Keywords** | (없음) | **6개** | 가이드: "1–7개, 'and/of' 회피" |
| **Highlights** | (없음) | **5개, 각 ≤85자** | Elsevier Highlights 규격 |
| **선언문** | (없음) | 경쟁이해 · **펀딩(없음)** · **GenAI(Claude, 문법/언어)** · 데이터이용성(이메일 요청) | 가이드 declarations(본문 references 직전 + title page) |
| **커버레터** | (없음) | 샘플 양식대로 IPM·우리 내용으로 | 제출 필수 |
| **제목** | 구어체·길었음 | **학술형·12단어** ("…: A Diagnostic Evaluation") | IPM 학술 레지스터 |
| **문장 톤** | "the exact reverse / is decisive" 등 **AI스러운 과장** | **측정된 학술 레지스터**(과장 제거, 본문 bold 14→4) | 리뷰어 인상·저널 수준 |
| **논리 공백** | loss 모호·완화식 없음·MostPop-IPC 역설 미설명·Recency 미정의 | **전부 보강**(BCE 수식, IPS/logQ/α식, AUC 역설 설명, 정의) | 리뷰어가 막을 지점 선제 차단 |
| **강건성 실험** | 메인 + 일부 | + **content ablation·rolling-origin·아키텍처 민감도·비샘플 full-pool·RQ1 random 대조** | "회사피처/단일컷오프/튜닝/샘플링 탓" 반박 차단 |

## 3. 어떻게 진행했나 (작업 방식)
1. **템플릿·가이드 먼저 해독** — `elsarticle-template-num.tex` 구조(frontmatter/abstract/highlights/keyword/numbered refs)와 가이드의 필수항목(이중익명·Abstract·표/그림·선언·CRediT)을 항목화.
2. **그림 결정** — 옛 그림(`files/`)은 부정확(예: company "learnable"·GAT "4 heads")해서, **데이터-최신 우리 그림**을 채택하고 제출 규격(300dpi/벡터/Times)으로 재생성. overview Figure 1은 정확·세련 버전을 위한 **프롬프트**를 별도 작성.
3. **본문 변환** — 마크다운 prose를 LaTeX로 옮기고, 인용을 numbered로 바꾸고, 표를 booktabs로, 그림을 `\includegraphics`+캡션으로.
4. **단계마다 검증** — (a) **컴파일**(미해결 참조 0, 28쪽), (b) **적대적 검증 워크플로**(수치 정합·인용 실재성·주장 근거 → 발견된 결함 수정), (c) **문장 폴리시 워크플로**(AI과장 제거·레지스터·논리공백 → ~45개 교정), (d) **최종 교열**(상호참조·중복·익명성).
5. **강건성 실험** — 맥에서 **안 끊기게(resume 기능 추가)** 돌리고, 결과를 표(Table 6~8)로 본문에 박제.
6. **패키징** — 익명 manuscript + title page + 커버레터 + 하이라이트 + 그림9 + 요약 + 체크리스트를 zip으로.

## 4. "남에게 한 문단으로 설명한다면"
> 처음엔 결과까지 다 들어간 마크다운 초안이 있었다. 거기에 **IPM이 Elsevier 이중익명 저널**이라는 점이 핵심이었다 — 그래서 ① 한 파일을 **익명 본문 + 별도 저자 title page**로 분리하고, ② 마크다운을 **elsarticle LaTeX + numbered 참고문헌**으로 옮기고, ③ 가이드가 요구하는 **Abstract(≤250)·키워드·하이라이트·CRediT·각종 선언·커버레터**를 채우고, ④ 그림을 **별도 파일·300dpi/벡터·Times New Roman**으로 규격화했다. 동시에 **AI스러운 과장 표현을 걷어내고 논리 공백을 메워** 저널 수준의 문장으로 올렸고, **리뷰어가 물을 법한 추가 실험**(회사피처 content ablation·rolling-origin·아키텍처 민감도)을 더해 방어선을 깔았다. 마지막에 모든 걸 **투고 zip 패키지**로 묶었다.

## 5. 아직 사람이 해야 하는 것 (자동화 불가)
ORCID(4인) · 코드 Zenodo DOI · 이동규 이메일 확정 · **리뷰 기간 GitHub repo private** · 경쟁이해 선언 `.docx` · 추천 리뷰어(선택).

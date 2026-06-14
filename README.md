# Patent Technology-Transfer Demand Prediction — A Diagnosis

Research code for a study of **patent technology-transfer demand prediction** (given a patent, rank which company is likely to receive its rights) framed as link prediction on a heterogeneous patent-citation + patent-transfer-company graph, evaluated on the Korean **KIPRIS** dataset.

**Main finding (diagnosis).** Under a realistic protocol — temporal split, same-IPC hard negatives, and an average-rank tie-break — every learned model (GraphSAGE, GAT, LightGCN, NGCF, SVD, MLP) falls **below simple popularity baselines**, and AUC is near chance. The causes are popularity bias and extreme patent cold-start (≈92% of test patents are unseen in training). Mitigations (IPS, logQ, popularity-debiased sampling, DropEdge) are partial and unstable. We also document a **tie-break evaluation artifact** that inflates cold-start models toward perfect scores under a strict `>` rule.

This repository is the experiment suite that produces the tables, figures, and diagnostics for the manuscript. There is no application — only research scripts.

## Quick start

### Google Colab (recommended — free GPU)
Open the notebook and run the cells top to bottom (set Runtime → T4 GPU first):

[`run_on_colab.ipynb`](run_on_colab.ipynb) → Open in Colab:
`https://colab.research.google.com/github/<owner>/<repo>/blob/main/run_on_colab.ipynb`

### Local
```bash
pip install -r requirements.txt              # for CUDA, install matching torch from pytorch.org first
python run_ipm_experiment.py --mode full --device cuda \
  --data_dir kipris-csv --emb_path patent_embeddings.pt \
  --demand_sample 200 --artifact_dir ./ipm_artifacts
```
Device auto-selects CUDA → MPS → CPU; override with `--device`. **Apple MPS does not support the sparse ops used by LightGCN/NGCF**, so use `--device cpu` on Apple Silicon (slow) or a CUDA GPU.

## Key flags (`run_ipm_experiment.py`)
| Flag | Meaning |
| :-- | :-- |
| `--mode {fast,full}` | fast = 2 seeds/5 epochs (smoke); full = 10 seeds/50 epochs/100 hard negs |
| `--split {temporal,random}` | temporal = realistic; random = leaky baseline for the temporal-vs-random contrast (RQ1) |
| `--company_feat {random,content}` | GNN company node features: fixed random, or mean train-patent SBERT (leakage-free content) |
| `--nneg_sweep` | candidate-set-size sensitivity {50,100,200} for key models (sampled-metric defense) |
| `--full_pool` · `--full_pool_cap N` | **unsampled** full-IPC-pool ranking (seed 0, key models): rank the positive against *all* eligible same-IPC companies, not n_neg sampled negatives — the strongest sampled-metric defense (Krichene & Rendle, 2020). Pools above N (default 1000) are down-capped |
| `--demand_sample N` | evaluate the slow rule-based Demand Score on N test queries (default 200) |
| `--device {auto,cpu,cuda,mps}` · `--data_dir` · `--emb_path` · `--artifact_dir` | environment |

## Required inputs (not in this repo)
`.gitignore` excludes `kipris-csv/`, `*.csv`, `*.pt`. Supply locally before a real run:
- **`kipris-csv/`** with `patents.csv` (`patApplicationNumber`, `patIpcNumber`, `patTitle`, `patAbstract`), `transfers.csv` (`trApplicationNumber`, `trCorrelatorName`, `trRegistrationDate`), `citings.csv` (`citStandardApplicationNumber`, `citApplicationNumber`).
- **`patent_embeddings.pt`** — frozen SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) embeddings, row-aligned to `patents.csv`. Generated/cached by `train_gnn_full_scale.py`.

For a quick functional test without real data, `make_mock_data.py` writes tiny KIPRIS-shaped fixtures, and `tests_new_logic.py` unit-tests the diagnostic helpers.

## Data availability
The KIPRIS patent data is obtained from the **Korea Intellectual Property Rights Information Service (KIPRIS) API** (http://www.kipris.or.kr / plus.kipris.or.kr). Access requires a (free) KIPRIS Plus API key; the fields used are listed above. We release the preprocessing and evaluation code here; the raw KIPRIS exports are redistributed subject to KIPRIS terms. A `patent_embeddings.pt` regeneration script (`train_gnn_full_scale.py`) is included so embeddings can be rebuilt from the patent text.

## Outputs (`--artifact_dir`)
`run_ipm_results.md` (Table 4 main results, Table 5 diagnostics, statistics, §5–11 diagnostics, §12 unsampled full-pool ranking when `--full_pool`) plus figures (`gat_attention_violin.png`, `popularity_stratified.png`, `ips_rerank_sweep.png`, `popularity_debiased_sweep.png`, `horizon_decay.png`, `nneg_sweep.png`). Publication figures are generated separately by `make_figures.py` → `paper_figures/`.

## Repository guide
- **Code**: `run_ipm_experiment.py` (the full suite), `make_mock_data.py` (test fixtures), `tests_new_logic.py` (unit tests), `make_figures.py` (paper figures). Legacy standalone evaluators (`evaluate_*`, `train_gnn_full_scale.py`, `tune_hyperparams.py`, `methodology_demo.py`) predate the current suite.
- **Manuscript material**: `PAPER_REVISION_SUGGESTIONS.md` (full 기존→수정 revision incl. Abstract/Intro/Related Work/Limitations/Conclusion), `REFERENCES.md` (verified bibliography), `EXPERIMENTS_AND_METRICS.md`, `AUDIT_AND_FIXES.md` (audit + fix log), `Paper_Methodology_Draft.md`, `IPM_experiment_protocol.md`.

## Reproducibility
Random seeds (torch, numpy) are fixed per run. The full ten-seed run takes ≈70 minutes on one NVIDIA Tesla T4 (Colab). Statistical tests: Wilcoxon signed-rank with Holm–Bonferroni correction over a pre-registered comparison family, plus bootstrap CIs over test queries. Tie-breaking is **average-rank** (a no-information model lands at chance, not rank 1).

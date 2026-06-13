# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for an academic paper (target journal: *Information Processing & Management*) on **patent technology-transfer demand prediction** with heterogeneous GNNs over the Korean KIPRIS patent dataset. The scientific contribution is a *diagnosis*: under a realistic temporal-split + same-IPC hard-negative evaluation, static GNNs and SVD collapse toward chance because of **popularity bias**, and the paper provides the evaluation protocol, bias-diagnostic tools, and partial mitigations.

There is **no application** here — only experiment scripts that produce tables, plots, and a results markdown for the manuscript. There is no package manifest, no test suite, and no linter config.

## Running experiments

All scripts are run directly with `python <script>.py` from the repo root. Device is auto-selected `cuda → mps → cpu` (developed on Apple Silicon / MPS).

The current, comprehensive experiment suite is **[run_ipm_experiment.py](run_ipm_experiment.py)** — it implements the full protocol in [IPM_experiment_protocol.md](IPM_experiment_protocol.md):

```bash
# Fast smoke run: 2 seeds, 5 epochs, 20 hard negs; slices data to top-500 companies / ≤2000 patents
python run_ipm_experiment.py --mode fast --artifact_dir ./out

# Full run: 10 seeds, 50 epochs, 100 hard negs, full dataset
python run_ipm_experiment.py --mode full --artifact_dir ./out

# Overridable knobs
python run_ipm_experiment.py --mode full --seeds 5 --epochs 30 --n_neg 200 --artifact_dir ./out
```

`--artifact_dir` is where `run_ipm_results.md` and the four diagnostic PNGs are written. **Its default is a stale absolute path on another machine (`/Users/isang-won/...`) — always pass `--artifact_dir` explicitly.**

To run one model/diagnostic in isolation there is no flag — the suite runs everything per seed in `main()`. Iterate by using `--mode fast` and/or editing the model loop.

## Required inputs (NOT in the repo)

`.gitignore` excludes `kipris-csv/`, `*.csv`, and `*.pt`, so the data and precomputed embeddings must be supplied locally before any real run:

- **`kipris-csv/`** directory containing `patents.csv` (`patApplicationNumber`, `patIpcNumber`, `patTitle`, `patAbstract`), `transfers.csv` (`trApplicationNumber`, `trCorrelatorName`, `trRegistrationDate`), and `citings.csv` (`citStandardApplicationNumber`, `citApplicationNumber`). Application numbers are cleaned to digits-only before joining.
- **`patent_embeddings.pt`** in the repo root — a torch tensor of frozen SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) embeddings, **row-aligned to the original `patents.csv` row order**. Every evaluation script `torch.load`s this; only [train_gnn_full_scale.py](train_gnn_full_scale.py) generates it (and caches it) from `patTitle + patAbstract`.

Without these files the scripts fail at load time. `methodology_demo.py` is the exception — it generates mock data and is self-contained.

## Architecture (run_ipm_experiment.py)

- **Heterogeneous graph**: `patent` nodes carry frozen SBERT features (384-d); `company` nodes are a learned `nn.Embedding`. Edge types: `patent–cites–patent` (citation graph) and `patent↔transfer↔company` (the prediction target, message-passed only on the train split).
- **Custom hand-written layers** `SageLayer` / `GatLayer` (scatter-based) inside `CustomHeteroGNN`, *not* PyG's `to_hetero`. GAT can return attention weights (`return_attn=True`) for the D12 diagnostic, and supports DropEdge and a sinusoidal time encoding on transfer edges. `FullModel` wraps the encoder with a dot-product link decoder.
- **Models evaluated together** (each gets a `record_model(...)` call): MostPop, Recency, SVD, MLP, LightGCN, NGCF, GraphSAGE, GAT, GAT+DropEdge, GAT+Time, plus the mitigation variants GraphSAGE+Debias, GraphSAGE+IPS, GraphSAGE+logQ. LightGCN/NGCF run on the bipartite transfer graph via a normalized sparse adjacency (`get_norm_adj`).
- **Evaluation protocol**: transfers sorted by `trRegistrationDate` and split 70/15/15 (train/val/test). For each test `(patent, positive_company, ipc4)`, `build_candidates` draws `n_neg` **same-IPC companies that didn't receive this patent** as hard negatives (seed-fixed; index 0 is the positive). Ranking metrics are Hits@K / MRR / NDCG@K (`KS = (1,3,5,10)`). Training uses early stopping on validation NDCG@10 (patience 5).
- **Diagnostics** reuse cached scores (no retraining): GAT hub-vs-non-hub attention (Mann-Whitney U, D12), score–popularity Spearman ρ (D13), popularity-stratified NDCG head/torso/tail (D14), hard-negative inversion rate (D15), cold-start fraction (D16).
- **Mitigations**: popularity-debiased negative sampling (`alpha` sweep, B4), IPS / log-popularity penalty re-ranking at inference (`beta` sweep, B5), and logQ sampled-softmax correction (B6).
- **Statistics**: Wilcoxon signed-rank (primary) + paired t-test (secondary) across seeds, with Holm–Bonferroni correction via `statsmodels.multipletests`.

## Two generations of scripts — know which to touch

- **Current / authoritative**: `run_ipm_experiment.py` (the suite above).
- **Legacy standalone evaluators** (older PyG `to_hetero` / `RandomLinkSplit` style, different seed sets, narrower scope): `evaluate_5_seeds.py`, `evaluate_gat_5seeds.py`, `evaluate_dropedge_5seeds.py`, `evaluate_full_scale_5seeds.py`, `evaluate_non_gnn_baselines.py`, `evaluate_temporal_hard_neg.py`. `train_gnn_full_scale.py` (produces `patent_embeddings.pt`, `tune_hyperparams.py` (grid search → `best_gnn_model.pt`), and `methodology_demo.py` (mock-data 6-step demo incl. XAI) also belong here. Prefer extending `run_ipm_experiment.py` over reviving these unless asked.
- **Do NOT run** `modify_script.py`, `update_evaluate.py`, or `scratch_parse2.py` blindly. The first two are one-off codegen utilities that **rewrite other `.py` files in place via regex**; `scratch_parse2.py` parses results into a manuscript at **hardcoded `/Users/isang-won/...` paths** that don't exist here. Treat them as historical scaffolding.

## Manuscript & protocol docs

- [IPM_experiment_protocol.md](IPM_experiment_protocol.md) — the implementation spec; the experiment matrix (A1–F21) is the source of truth for what each experiment must produce and which table/figure it feeds.
- [Paper_Methodology_Draft.md](Paper_Methodology_Draft.md) — the methodology + results manuscript. Its Tables 4/5 and §4.2 are populated from `run_ipm_results.md`.
- [walkthrough.md](walkthrough.md), [run_ipm_results.md](run_ipm_results.md) — last results summary and the generated results table. Note these contain stale `/Users/isang-won/...` links from a previous run.

When numbers change, regenerate `run_ipm_results.md` from a run, then reflect deltas into the manuscript tables — do not hand-edit results into the paper without a backing run.

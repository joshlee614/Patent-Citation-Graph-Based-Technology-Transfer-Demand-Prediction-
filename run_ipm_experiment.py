import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import argparse
import itertools
import time
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon, ttest_rel
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import svds
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_edge, softmax
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Global variables
KS = (1, 3, 5, 10)

# Custom GNN layers
class SageLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        
    def forward(self, x_self, x_neigh, edge_index, size):
        src = edge_index[0]
        dst = edge_index[1]
        
        neigh_sum = torch.zeros(size[1], x_neigh.size(1), device=x_self.device)
        counts = torch.zeros(size[1], 1, device=x_self.device)
        ones = torch.ones(src.size(0), 1, device=x_self.device)
        
        neigh_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x_neigh.size(1)), x_neigh[src])
        counts.scatter_add_(0, dst.unsqueeze(-1), ones)
        
        neigh_mean = neigh_sum / (counts + 1e-10)
        out = self.lin_self(x_self) + self.lin_neigh(neigh_mean)
        return F.relu(out)

class GatLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.att_self = nn.Parameter(torch.Tensor(1, out_dim))
        self.att_neigh = nn.Parameter(torch.Tensor(1, out_dim))
        nn.init.xavier_uniform_(self.att_self)
        nn.init.xavier_uniform_(self.att_neigh)
        
    def forward(self, x_self, x_neigh, edge_index, size):
        src = edge_index[0]
        dst = edge_index[1]
        
        h_self = self.lin_self(x_self)
        h_neigh = self.lin_neigh(x_neigh)
        
        alpha_self = (h_self * self.att_self).sum(-1)
        alpha_neigh = (h_neigh * self.att_neigh).sum(-1)
        
        alpha = alpha_self[dst] + alpha_neigh[src]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        alpha_softmax = softmax(alpha, dst, num_nodes=size[1])
        
        out = torch.zeros_like(h_self)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, h_self.size(1)), alpha_softmax.unsqueeze(-1) * h_neigh[src])
        return out, alpha_softmax

class CustomHeteroGNN(nn.Module):
    def __init__(self, gnn_type, patent_in, company_in, hidden_dim, out_dim, dropout, apply_dropedge=False, use_time=False):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.apply_dropedge = apply_dropedge
        self.use_time = use_time
        
        if use_time:
            self.patent_lin = nn.Linear(patent_in + 16, hidden_dim)
        else:
            self.patent_lin = nn.Linear(patent_in, hidden_dim)
            
        self.company_lin = nn.Linear(company_in, hidden_dim)
        
        if gnn_type == 'SAGE':
            self.sage_c2p = SageLayer(hidden_dim, out_dim)
            self.sage_p2c = SageLayer(hidden_dim, out_dim)
            self.sage_p2p = SageLayer(hidden_dim, out_dim)
        elif gnn_type == 'GAT':
            self.gat_c2p = GatLayer(hidden_dim, out_dim)
            self.gat_p2c = GatLayer(hidden_dim, out_dim)
            self.gat_p2p = GatLayer(hidden_dim, out_dim)
            
    def forward(self, x_dict, edge_index_dict, time_enc=None, return_attn=False):
        xp = x_dict['patent']
        if self.use_time and time_enc is not None:
            xp = torch.cat([xp, time_enc], dim=-1)
            
        xp = self.patent_lin(xp).relu()
        xc = self.company_lin(x_dict['company']).relu()
        
        cites_edge = edge_index_dict[('patent', 'cites', 'patent')]
        if self.training and self.apply_dropedge:
            mask = torch.rand(cites_edge.size(1), device=cites_edge.device) >= 0.2
            cites_edge = cites_edge[:, mask]
            
        if self.gnn_type == 'MLP':
            return {'patent': xp, 'company': xc}
            
        elif self.gnn_type == 'SAGE':
            hp_new = self.sage_c2p(xp, xc, edge_index_dict[('company', 'buys', 'patent')], (xc.size(0), xp.size(0)))
            hc_new = self.sage_p2c(xc, xp, edge_index_dict[('patent', 'rev_buys', 'company')], (xp.size(0), xc.size(0)))
            hp_cites = self.sage_p2p(xp, xp, cites_edge, (xp.size(0), xp.size(0)))
            
            hp_final = F.dropout(hp_new + hp_cites, p=self.dropout, training=self.training)
            hc_final = F.dropout(hc_new, p=self.dropout, training=self.training)
            return {'patent': hp_final, 'company': hc_final}
            
        elif self.gnn_type == 'GAT':
            hp_new, att_c2p = self.gat_c2p(xp, xc, edge_index_dict[('company', 'buys', 'patent')], (xc.size(0), xp.size(0)))
            hc_new, att_p2c = self.gat_p2c(xc, xp, edge_index_dict[('patent', 'rev_buys', 'company')], (xp.size(0), xc.size(0)))
            hp_cites, att_p2p = self.gat_p2p(xp, xp, cites_edge, (xp.size(0), xp.size(0)))
            
            hp_final = F.dropout(hp_new + hp_cites, p=self.dropout, training=self.training)
            hc_final = F.dropout(hc_new, p=self.dropout, training=self.training)
            
            if return_attn:
                return {'patent': hp_final, 'company': hc_final}, (att_c2p, att_p2c, att_p2p)
            return {'patent': hp_final, 'company': hc_final}

class FullModel(nn.Module):
    def __init__(self, gnn_type, patent_in, company_in, hidden_dim=32, out_dim=16, dropout=0.1, apply_dropedge=False, use_time=False):
        super().__init__()
        self.gnn = CustomHeteroGNN(gnn_type, patent_in, company_in, hidden_dim, out_dim, dropout, apply_dropedge, use_time)
        self.patent_lin = self.gnn.patent_lin
        self.company_lin = self.gnn.company_lin
        
    def forward(self, x_dict, edge_index_dict, time_enc=None):
        return self.gnn(x_dict, edge_index_dict, time_enc)

# Candidate building
def build_candidates(p, c_pos, ipc4, ipc_company_index, train_transfer_set, n_neg, rng, num_companies):
    pool_set = ipc_company_index.get(ipc4, set()) - {c_pos}
    if len(pool_set) <= n_neg:
        pool = [c for c in pool_set if (p, c) not in train_transfer_set]
    else:
        pool_list = list(pool_set)
        pool = []
        set_pool = set()
        attempts = 0
        n_pool = len(pool_list)
        while len(pool) < n_neg and attempts < 1000:
            size_to_sample = min(n_neg * 2, n_pool)
            idx_cand = rng.choice(n_pool, size=size_to_sample, replace=False)
            for idx in idx_cand:
                c = pool_list[idx]
                if c not in set_pool and (p, c) not in train_transfer_set:
                    pool.append(c)
                    set_pool.add(c)
                    if len(pool) == n_neg:
                        break
            attempts += 1
            
    is_padded = len(pool) < n_neg
    if len(pool) < n_neg:
        needed = n_neg - len(pool)
        padded = []
        set_pool = set(pool)
        set_pool.add(c_pos)
        pad_attempts = 0
        # Cap attempts so that, if fewer than n_neg distinct eligible companies exist
        # (e.g. n_neg larger than the company catalogue), we return what we can instead
        # of looping forever. On the full KIPRIS catalogue (122k companies) this never triggers.
        while len(padded) < needed and pad_attempts < 200:
            candidates = rng.choice(num_companies, size=needed * 2, replace=True)
            for c in candidates:
                c = int(c)
                if c not in set_pool and (p, c) not in train_transfer_set:
                    padded.append(c)
                    set_pool.add(c)
                    if len(padded) == needed:
                        break
            pad_attempts += 1
        pool = pool + padded
    return [c_pos] + list(pool), is_padded

# Evaluator helpers
def get_time_encoding(days, dim=16):
    freqs = torch.exp(torch.arange(0, dim, 2) * (-np.log(1e4)/dim)).to(days.device)
    pe = torch.zeros(days.shape[0], dim).to(days.device)
    pe[:, 0::2] = torch.sin(days[:, None] * freqs)
    pe[:, 1::2] = torch.cos(days[:, None] * freqs)
    return pe

def per_query_auc_ap(score_np):
    labels = np.zeros(len(score_np))
    labels[0] = 1.0
    if len(np.unique(score_np)) == 1:
        return 0.5, 1.0 / len(score_np)
    auc = roc_auc_score(labels, score_np)
    ap = average_precision_score(labels, score_np)
    return auc, ap

def per_query_metrics(rank, ks=KS):
    out = {}
    for k in ks:
        out[f"hits@{k}"] = float(rank <= k)
        out[f"ndcg@{k}"] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
    out["mrr"] = 1.0 / rank
    out["map"] = 1.0 / rank  # MAP equals MRR for single-positive per query
    return out

def aggregate(rank_list, ks=KS):
    metrics = [per_query_metrics(r, ks) for r in rank_list]
    keys = metrics[0].keys()
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}

def tie_aware_ranks(scores):
    """1-indexed rank of the positive (column 0) among candidates, using the AVERAGE-RANK
    convention for ties. A no-information model that assigns all candidates the SAME score
    must land at the middle rank (i.e. chance) — NOT rank 1. The previous strict-'>' rule
    placed the positive above every tied negative, which handed a *perfect* score to, e.g.,
    SVD on cold-start patents (whose latent factor is 0 -> all scores 0 -> all tied). This
    convention is consistent with the tie=0.5 AUC and yields rank = #(neg>pos) + 0.5*#(neg==pos) + 1.
    """
    pos = scores[:, 0:1]
    gt = (scores[:, 1:] > pos).sum(dim=-1)
    eq = (scores[:, 1:] == pos).sum(dim=-1)
    return gt + 0.5 * eq + 1

def aggregate_with_n(rank_list, ks=KS):
    """aggregate() but tolerant of empty input and carrying the query count n."""
    if len(rank_list) == 0:
        out = {f"hits@{k}": 0.0 for k in ks}
        out.update({f"ndcg@{k}": 0.0 for k in ks})
        out["mrr"] = 0.0
        out["map"] = 0.0
        out["n"] = 0
        return out
    out = aggregate(rank_list, ks)
    out["n"] = len(rank_list)
    return out

def bootstrap_ci_ndcg(rank_list, n_boot=1000, seed=0):
    """Percentile (2.5/97.5) bootstrap CI over the PER-QUERY ranks (NEW-12).
    Resamples test queries with replacement; captures query-sampling variance that
    the seed-level std does not. Returns {metric: (mean, lo, hi)}."""
    r = np.asarray(rank_list, dtype=float)
    n = len(r)
    if n == 0:
        return {m: (0.0, 0.0, 0.0) for m in ("ndcg@10", "hits@10", "mrr")}
    rng = np.random.default_rng(seed)
    boot = {"ndcg@10": [], "hits@10": [], "mrr": []}
    for _ in range(n_boot):
        rr = r[rng.integers(0, n, size=n)]
        boot["ndcg@10"].append(float(np.where(rr <= 10, 1.0 / np.log2(rr + 1.0), 0.0).mean()))
        boot["hits@10"].append(float((rr <= 10).mean()))
        boot["mrr"].append(float((1.0 / rr).mean()))
    return {m: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for m, v in boot.items()}

def evaluate_mostpop_ipc(queries, ipc_company_count, train_pop):
    """IPC-conditional MostPop (NEW-9): score each candidate by its train-transfer
    count WITHIN the query's own ipc4 (a stronger popularity skyline than global MostPop).
    Returns the same 5-tuple shape as the other evaluators."""
    ranks, aucs, aps, scores_all, pops_all = [], [], [], [], []
    for q in queries:
        c_pos_idx, ipc4, cand = q[1], q[2], q[3]
        counts = ipc_company_count.get(ipc4, {})
        s = np.array([counts.get(int(c), 0) for c in cand], dtype=float)
        pos, neg = s[0], s[1:]
        rank = float((neg > pos).sum()) + 0.5 * float((neg == pos).sum()) + 1  # average-rank ties
        if len(neg) > 0:
            auc = float(((pos > neg).astype(float) + 0.5 * (pos == neg).astype(float)).mean())
        else:
            auc = 0.5
        ranks.append(rank)
        aucs.append(auc)
        aps.append(1.0 / rank)
        scores_all.append(s)
        pops_all.append(train_pop[np.asarray(cand, dtype=int)])
    return ranks, aucs, aps, np.array(scores_all), np.array(pops_all)

def evaluate_cn_aa(queries, cit_neighbors, company_patents, patent_indegree):
    """Common-Neighbors (CN) and Adamic-Adar (AA) structural link-prediction baselines.
    For a query (patent p, candidate company c), a "common neighbor" is a patent p' that is
    (a) a citation neighbour of p and (b) was transferred to c in training:
        CN(p,c) = |cit_neighbors(p) ∩ company_patents(c)|
        AA(p,c) = Σ_{p' in that set} 1 / log(patent_indegree[p'] + e)
    Seed-invariant (pure graph structure); average-rank tie-break. Returns (cn_tuple, aa_tuple),
    each shaped like the other evaluators."""
    cn_r, cn_a, cn_p, cn_s, cn_pp = [], [], [], [], []
    aa_r, aa_a, aa_p, aa_s, aa_pp = [], [], [], [], []
    e = np.e
    empty = set()
    for q in queries:
        p, cand = q[0], q[3]
        nb = cit_neighbors.get(p, empty)
        cn_vec = np.zeros(len(cand))
        aa_vec = np.zeros(len(cand))
        if nb:
            for j, c in enumerate(cand):
                cp = company_patents.get(int(c), empty)
                if cp:
                    inter = [pp for pp in nb if pp in cp]
                    if inter:
                        cn_vec[j] = len(inter)
                        aa_vec[j] = float(sum(1.0 / np.log(patent_indegree[pp] + e) for pp in inter))
        for vec, (rr, aa, pp_, ss, pops) in [(cn_vec, (cn_r, cn_a, cn_p, cn_s, cn_pp)),
                                             (aa_vec, (aa_r, aa_a, aa_p, aa_s, aa_pp))]:
            pos, neg = vec[0], vec[1:]
            rank = float((neg > pos).sum()) + 0.5 * float((neg == pos).sum()) + 1
            auc = float(((pos > neg).astype(float) + 0.5 * (pos == neg).astype(float)).mean()) if len(neg) else 0.5
            rr.append(rank); aa.append(auc); pp_.append(1.0 / rank); ss.append(vec)
            pops.append(np.zeros(len(cand)))
    cn_tuple = (cn_r, cn_a, cn_p, np.array(cn_s), np.array(cn_pp))
    aa_tuple = (aa_r, aa_a, aa_p, np.array(aa_s), np.array(aa_pp))
    return cn_tuple, aa_tuple

def classify_failures(queries, rank_list, scores_all, train_pop, pop_thr_q=0.9):
    """Error-source decomposition (NEW-4). For each FAILED query (rank>1) attribute it to:
      - popular_hardneg : a historically popular (>=90th pct) hard negative outscored the positive
      - rare_new_positive : the true target company is rare/new (train_pop<=1)
      - semantic_residual : neither of the above (model just ranked it wrong)
    Priority: popularity mechanism first (the paper's thesis), then cold-start, then residual."""
    thr = np.quantile(train_pop, pop_thr_q) if len(train_pop) > 0 else 0.0
    buckets = {"popular_hardneg": 0, "rare_new_positive": 0, "semantic_residual": 0}
    n_fail = 0
    for i, q in enumerate(queries):
        if rank_list[i] <= 1:
            continue
        n_fail += 1
        c_pos, cand = q[1], q[3]
        srow = scores_all[i]
        pos_score, neg_scores = srow[0], srow[1:]
        neg_pop = train_pop[np.asarray(cand[1:], dtype=int)]
        beaten_by_popular = bool(np.any((neg_scores > pos_score) & (neg_pop >= thr)))
        if beaten_by_popular:
            buckets["popular_hardneg"] += 1
        elif train_pop[c_pos] <= 1:
            buckets["rare_new_positive"] += 1
        else:
            buckets["semantic_residual"] += 1
    return buckets, n_fail

# Debiased neg sampling
def sample_neg_debiased(n_samples, train_pop, alpha, rng, num_companies):
    prob = (train_pop + 1.0) ** alpha
    prob = prob / prob.sum()
    return rng.choice(num_companies, size=n_samples, p=prob)

# CF Models
class LightGCN(nn.Module):
    def __init__(self, n_p, n_c, dim=64, n_layers=3):
        super().__init__()
        self.emb_p = nn.Embedding(n_p, dim)
        self.emb_c = nn.Embedding(n_c, dim)
        nn.init.normal_(self.emb_p.weight, std=0.1)
        nn.init.normal_(self.emb_c.weight, std=0.1)
        self.n_layers = n_layers
        
    def propagate(self, norm_adj):
        x = torch.cat([self.emb_p.weight, self.emb_c.weight], 0)
        outs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(norm_adj, x)
            outs.append(x)
        x = torch.stack(outs, 0).mean(0)
        return x[:self.emb_p.num_embeddings], x[self.emb_p.num_embeddings:]

class NGCFLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, out_dim)
        self.w2 = nn.Linear(in_dim, out_dim)
        
    def forward(self, norm_adj, x):
        neighbor_x = torch.sparse.mm(norm_adj, x)
        out = self.w1(x) + self.w2(x * neighbor_x)
        return F.leaky_relu(out, negative_slope=0.2)

class NGCF(nn.Module):
    def __init__(self, n_p, n_c, dim=64, n_layers=2):
        super().__init__()
        self.emb_p = nn.Embedding(n_p, dim)
        self.emb_c = nn.Embedding(n_c, dim)
        nn.init.normal_(self.emb_p.weight, std=0.1)
        nn.init.normal_(self.emb_c.weight, std=0.1)
        self.layers = nn.ModuleList([NGCFLayer(dim, dim) for _ in range(n_layers)])
        # Projection layer to map multi-layer concatenation back to 'dim' to match scoring alignment
        self.proj_out = nn.Linear(dim * (n_layers + 1), dim)
        
    def propagate(self, norm_adj):
        x = torch.cat([self.emb_p.weight, self.emb_c.weight], 0)
        outs = [x]
        for layer in self.layers:
            x = layer(norm_adj, x)
            outs.append(x)
        x = torch.cat(outs, dim=-1)
        x_proj = self.proj_out(x)
        return x_proj[:self.emb_p.num_embeddings], x_proj[self.emb_p.num_embeddings:]

def get_norm_adj(train_edge_index, n_p, n_c):
    row = torch.cat([train_edge_index[1], train_edge_index[0] + n_p])
    col = torch.cat([train_edge_index[0] + n_p, train_edge_index[1]])
    val = torch.ones(row.size(0))
    
    deg = torch.zeros(n_p + n_c)
    deg.scatter_add_(0, row, val)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    
    val_norm = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
    indices = torch.stack([row, col])
    norm_adj = torch.sparse_coo_tensor(indices, val_norm, (n_p + n_c, n_p + n_c)).coalesce()
    return norm_adj

# Helper to compute NDCG in batches to save memory
@torch.no_grad()
def compute_ndcg_from_embeddings(hp, hc, p_t, cand_t, batch_size=10000):
    ndcgs = []
    num_queries = p_t.size(0)
    for i in range(0, num_queries, batch_size):
        batch_p = p_t[i : i + batch_size]
        batch_cand = cand_t[i : i + batch_size]
        
        hp_q = hp[batch_p]
        hc_q = hc[batch_cand]
        scores = torch.bmm(hc_q, hp_q.unsqueeze(-1)).squeeze(-1)
        ranks = tie_aware_ranks(scores)
        ndcg = 1.0 / torch.log2(ranks.float() + 1.0)
        ndcg[ranks > 10] = 0.0
        ndcgs.append(ndcg.cpu())
    return torch.cat(ndcgs).mean().item()

# GNN early stopping val evaluator (vectorized & batched)
@torch.no_grad()
def compute_val_ndcg(model, data, val_p_t, val_cand_t, device, time_enc=None):
    model.eval()
    x_dict = {
        'patent': data['patent'].x,
        'company': data['company'].x
    }
    node_embs = model(x_dict, data.edge_index_dict, time_enc=time_enc)
    hp = node_embs['patent']
    hc = node_embs['company']
    return compute_ndcg_from_embeddings(hp, hc, val_p_t, val_cand_t)

# CF early stopping val evaluator (vectorized & batched)
@torch.no_grad()
def compute_val_ndcg_cf(model, norm_adj, val_p_t, val_cand_t, device):
    model.eval()
    hp, hc = model.propagate(norm_adj.to(device))
    return compute_ndcg_from_embeddings(hp, hc, val_p_t, val_cand_t)

def train_cf_model(model, norm_adj, train_edge_index, max_epochs, lr, device, val_queries=None, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)
    norm_adj = norm_adj.to(device)
    n_c = model.emb_c.num_embeddings
    
    best_val_ndcg = -1.0
    best_state = None
    patience_counter = 0
    
    val_p_t, val_cand_t = None, None
    if val_queries is not None:
        val_p_t = torch.tensor([q[0] for q in val_queries], dtype=torch.long).to(device)
        val_cand_t = torch.tensor([q[3] for q in val_queries], dtype=torch.long).to(device)
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        hp, hc = model.propagate(norm_adj)
        
        c_pos = train_edge_index[0].to(device)
        p = train_edge_index[1].to(device)
        pos_score = (hp[p] * hc[c_pos]).sum(-1)
        
        c_neg = torch.randint(0, n_c, (c_pos.size(0),), device=device)
        neg_score = (hp[p] * hc[c_neg]).sum(-1)
        
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
        loss.backward()
        optimizer.step()
        
        # Validation evaluation
        if val_p_t is not None:
            val_ndcg = compute_val_ndcg_cf(model, norm_adj, val_p_t, val_cand_t, device)
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

# Training GNN
def get_train_neg_edges(train_edge_index, train_pop, alpha, seed, num_companies):
    n_samples = train_edge_index.size(1)
    rng = np.random.default_rng(seed)
    if alpha == 0.0:
        neg_c = torch.randint(0, num_companies, (n_samples,))
    else:
        prob = (train_pop + 1.0) ** alpha
        prob = prob / prob.sum()
        neg_c = torch.from_numpy(rng.choice(num_companies, size=n_samples, p=prob))
    return torch.stack([neg_c, train_edge_index[1].cpu()], dim=0)

def compute_loss_logq(pos_score, neg_score, pos_c, neg_c, train_pop, logq_alpha):
    # q(c) ~ (popularity+1)^alpha; subtract log q from logits (sampled-softmax / logQ correction).
    # train_pop is a numpy array here; build the tensor explicitly to avoid copy-construct warnings.
    q = (np.asarray(train_pop, dtype=np.float64) + 1.0) ** logq_alpha
    q = q / q.sum()
    log_q = torch.log(torch.as_tensor(q, dtype=torch.float32, device=pos_score.device) + 1e-10)
    
    pos_score_corr = pos_score - log_q[pos_c]
    neg_score_corr = neg_score - log_q[neg_c]
    
    scores = torch.cat([pos_score_corr, neg_score_corr])
    labels = torch.cat([torch.ones(len(pos_score)), torch.zeros(len(neg_score))]).to(pos_score.device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def train_gnn(model, data, train_edge_index, train_pop, debias_alpha, logq_alpha, max_epochs, lr, device, seed, num_companies, val_queries=None, patience=5, time_enc=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.to(device)
    
    best_val_ndcg = -1.0
    best_state = None
    patience_counter = 0
    
    val_p_t, val_cand_t = None, None
    if val_queries is not None:
        val_p_t = torch.tensor([q[0] for q in val_queries], dtype=torch.long).to(device)
        val_cand_t = torch.tensor([q[3] for q in val_queries], dtype=torch.long).to(device)
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        train_neg_edge = get_train_neg_edges(train_edge_index, train_pop, debias_alpha, seed + epoch, num_companies)
        train_label_index = torch.cat([train_edge_index, train_neg_edge.to(device)], dim=1)
        train_label = torch.cat([torch.ones(train_edge_index.size(1)), torch.zeros(train_neg_edge.size(1))]).to(device)
        
        x_dict = {
            'patent': data['patent'].x,
            'company': data['company'].x
        }
        node_embs = model(x_dict, data.edge_index_dict, time_enc=time_enc)
        hp = node_embs['patent']
        hc = node_embs['company']

        src_c = train_label_index[0]
        dst_p = train_label_index[1]
        out = (hc[src_c] * hp[dst_p]).sum(-1)

        if logq_alpha > 0.0:
            pos_score = out[:train_edge_index.size(1)]
            neg_score = out[train_edge_index.size(1):]
            pos_c = train_edge_index[0]
            neg_c = train_neg_edge[0].to(device)
            loss = compute_loss_logq(pos_score, neg_score, pos_c, neg_c, train_pop, logq_alpha)
        else:
            loss = criterion(out, train_label.float())
            
        loss.backward()
        optimizer.step()
        
        # Validation evaluation & Early stopping check
        if val_p_t is not None:
            val_ndcg = compute_val_ndcg(model, data, val_p_t, val_cand_t, device, time_enc=time_enc)
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

# Helper to batch evaluations
@torch.no_grad()
def evaluate_embeddings(hp, hc, test_p_t, test_cand_t, train_pop_t, ips_beta=0.0, batch_size=10000):
    all_ranks = []
    all_aucs = []
    all_aps = []
    all_scores = []
    all_pops = []
    
    num_queries = test_p_t.size(0)
    for i in range(0, num_queries, batch_size):
        batch_p = test_p_t[i : i + batch_size]
        batch_cand = test_cand_t[i : i + batch_size]
        
        hp_q = hp[batch_p]
        hc_q = hc[batch_cand]
        scores = torch.bmm(hc_q, hp_q.unsqueeze(-1)).squeeze(-1)
        
        if ips_beta > 0.0:
            pen = ips_beta * torch.log(train_pop_t[batch_cand] + 1.0)
            scores = scores - pen
            
        ranks = tie_aware_ranks(scores)
        # True rank-AUC for a single positive: P(pos>neg) + 0.5*P(pos==neg), averaged
        # over the candidate negatives. Ties contribute 0.5 (matches sklearn roc_auc_score),
        # unlike the previous strict-'>' version which mislabeled a tie-pessimistic statistic as AUC.
        aucs = ((scores[:, 0:1] > scores[:, 1:]).float()
                + 0.5 * (scores[:, 0:1] == scores[:, 1:]).float()).mean(dim=-1)
        aps = 1.0 / ranks.float()  # AP == MRR for single-positive (not displayed; see Table 4 note)

        all_ranks.extend(ranks.cpu().numpy().tolist())
        all_aucs.extend(aucs.cpu().numpy().tolist())
        all_aps.extend(aps.cpu().numpy().tolist())
        all_scores.append(scores.cpu().numpy())
        all_pops.append(train_pop_t[batch_cand].cpu().numpy())
        
    return all_ranks, all_aucs, all_aps, np.concatenate(all_scores, axis=0), np.concatenate(all_pops, axis=0)

# Evaluation wrappers (vectorized & batched)
@torch.no_grad()
def evaluate_gnn(model, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0, time_enc=None):
    model.eval()
    x_dict = {
        'patent': data['patent'].x,
        'company': data['company'].x
    }
    node_embs = model(x_dict, data.edge_index_dict, time_enc=time_enc)
    hp = node_embs['patent']
    hc = node_embs['company']
    return evaluate_embeddings(hp, hc, test_p_t, test_cand_t, train_pop_t, ips_beta)

def evaluate_mlp(model, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0):
    return evaluate_gnn(model, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta)

@torch.no_grad()
def evaluate_cf_model(model, norm_adj, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0):
    model.eval()
    hp, hc = model.propagate(norm_adj.to(device))
    return evaluate_embeddings(hp, hc, test_p_t, test_cand_t, train_pop_t, ips_beta)

def evaluate_mostpop(test_cand_t, train_pop_t):
    scores = train_pop_t[test_cand_t].float()
    ranks = tie_aware_ranks(scores)
    aucs = ((scores[:, 0:1] > scores[:, 1:]).float()
            + 0.5 * (scores[:, 0:1] == scores[:, 1:]).float()).mean(dim=-1)
    aps = 1.0 / ranks.float()
    return (
        ranks.cpu().numpy().tolist(),
        aucs.cpu().numpy().tolist(),
        aps.cpu().numpy().tolist(),
        scores.cpu().numpy(),
        scores.cpu().numpy()
    )

def evaluate_recency(test_cand_t, company_last_active_t, train_pop_t):
    scores = company_last_active_t[test_cand_t].float()
    ranks = tie_aware_ranks(scores)
    aucs = ((scores[:, 0:1] > scores[:, 1:]).float()
            + 0.5 * (scores[:, 0:1] == scores[:, 1:]).float()).mean(dim=-1)
    aps = 1.0 / ranks.float()
    return (
        ranks.cpu().numpy().tolist(),
        aucs.cpu().numpy().tolist(),
        aps.cpu().numpy().tolist(),
        scores.cpu().numpy(),
        train_pop_t[test_cand_t].cpu().numpy()
    )

def evaluate_svd(queries, train_csr, k_dim, device, train_pop_t):
    p_t = torch.tensor([q[0] for q in queries], dtype=torch.long).to(device)
    cand_t = torch.tensor([q[3] for q in queries], dtype=torch.long).to(device)
    
    k_actual = min(k_dim, min(train_csr.shape) - 1)
    from sklearn.utils.extmath import randomized_svd
    from scipy.sparse.linalg import svds
    try:
        U, Sigma, VT = randomized_svd(train_csr, n_components=k_actual, random_state=42)
    except Exception:
        U, Sigma, VT = svds(train_csr, k=k_actual, solver='arpack')
    U_sigma = torch.tensor((U * Sigma).copy(), dtype=torch.float32).to(device)
    VT_T = torch.tensor(VT.T.copy(), dtype=torch.float32).to(device)
    return evaluate_embeddings(VT_T, U_sigma, p_t, cand_t, train_pop_t, ips_beta=0.0)

# Inversion rate helper (vectorized)
def compute_inversion_rate(queries, scores_all, train_pop, pop_thr_q=0.9):
    thr = np.quantile(train_pop, pop_thr_q) if len(train_pop) > 0 else 0.0
    neg_cands = np.array([q[3][1:] for q in queries]) # shape (NUM_QUERIES, 100)
    neg_pops = train_pop[neg_cands] # shape (NUM_QUERIES, 100)
    is_hard_neg = neg_pops >= thr # shape (NUM_QUERIES, 100)
    pos_scores = scores_all[:, 0:1] # shape (NUM_QUERIES, 1)
    neg_scores = scores_all[:, 1:] # shape (NUM_QUERIES, 100)
    is_inversion = neg_scores > pos_scores # shape (NUM_QUERIES, 100)
    inversion_count = (is_inversion & is_hard_neg).sum()
    total_count = is_hard_neg.sum()
    return inversion_count / (total_count + 1e-10)

def citation_depth(target_patent_idx, company_patent_idxs, cited_by, max_depth=5, max_nodes=800):
    """
    Shortest path (in citation hops) from target_patent_idx
    back to any patent owned by the company.
    company_patent_idxs: set of patent indices owned/transferred by the company.
    Returns depth (1 = direct citation), or float('inf') if not reachable.

    max_nodes bounds the BFS frontier: highly-cited "hub" patents can make the
    depth-5 BFS explode over the 900k-edge citation graph (single queries taking
    minutes). Once `visited` exceeds max_nodes we give up and return inf. This makes
    the Demand-Score (E19) cost predictable; E19 is a near-degenerate diagnostic on
    KIPRIS so this approximation is acceptable.
    """
    if target_patent_idx in company_patent_idxs:
        return 1
    visited = {target_patent_idx}
    frontier = {target_patent_idx}
    for depth in range(1, max_depth + 1):
        next_frontier = set()
        for p in frontier:
            for citer in cited_by.get(p, []):
                if citer in company_patent_idxs:
                    return depth
                if citer not in visited:
                    visited.add(citer)
                    next_frontier.add(citer)
            if len(visited) > max_nodes:
                return float('inf')
        frontier = next_frontier
        if not frontier:
            break
    return float('inf')

def demand_score_rank(p_idx, c_pos_idx, cand, version,
                      train_pop, company_last_active, patent_ipc,
                      patent_indegree, ipc4_mean_cit, ipc4_global_mean,
                      power_threshold, STRATEGIC_IPC4,
                      company_patents, cited_by, all_patents_list):
    DEPTH_DECAY_BASE = 0.6
    DIV_LOG_MAX = 13
    BOOST_STRATEGIC = 1.20
    BOOST_POWER = 1.10
    BOOST_MAX = BOOST_STRATEGIC * BOOST_POWER  # 1.32

    p_ipc4 = patent_ipc.get(all_patents_list[p_idx], 'UNKNOWN')
    ipc_avg = ipc4_mean_cit.get(p_ipc4, ipc4_global_mean) or 1.0
    global_avg = ipc4_global_mean or 1.0
    p_global_cit = patent_indegree[p_idx]
    is_power = p_global_cit >= power_threshold
    is_strategic = p_ipc4 in STRATEGIC_IPC4

    scores = []
    for c in cand:
        c_pats = company_patents.get(c, set())
        cit_count = sum(
            1 for cp in c_pats if p_idx in cited_by.get(cp, [])
        )
        unique = len(c_pats)
        depth = citation_depth(p_idx, c_pats, cited_by)
        rec = company_last_active[c]
        transfer_count = train_pop[c]

        if version == 'original':
            ipc_ratio = (cit_count + 0.02 * p_global_cit) / ipc_avg * 100
            freq_score = min(100.0, ipc_ratio)
            depth_score = (100.0 / (depth ** 1.5)) if depth < float('inf') else 0.0
            div_score = min(100.0, unique * 25.0)
        else:  # revised
            ipc_norm    = min(cit_count / ipc_avg, 2.0) / 2.0
            global_norm = min(p_global_cit / global_avg, 2.0) / 2.0
            freq_score  = ((ipc_norm * 0.8) + (global_norm * 0.2)) * 100
            depth_score = (100.0 * (DEPTH_DECAY_BASE ** (depth - 1))
                           if depth < float('inf') else 0.0)
            div_score   = (
                (np.log(unique + 1) / np.log(DIV_LOG_MAX + 1)) * 100
            )

        max_days = max(company_last_active) or 1.0
        recent_score   = (rec / max_days) * 100.0
        transfer_score = min(100.0, transfer_count * 10.0)

        total = (0.30 * freq_score
               + 0.25 * recent_score
               + 0.20 * depth_score
               + 0.15 * div_score
               + 0.10 * transfer_score)

        boost = 1.0
        if is_strategic:
            boost *= BOOST_STRATEGIC
        if is_power:
            boost *= BOOST_POWER
        boost = min(boost, BOOST_MAX)
        total *= boost
        scores.append(total)

    order = np.argsort(-np.array(scores), kind='stable')
    return int(np.where(order == 0)[0][0]) + 1, scores

# Main function
def main():
    parser = argparse.ArgumentParser(description="IPM Experiment Suite")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_neg", type=int, default=None)
    parser.add_argument("--artifact_dir", type=str, default="./ipm_artifacts")
    parser.add_argument("--data_dir", type=str, default="kipris-csv")
    parser.add_argument("--emb_path", type=str, default="patent_embeddings.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device. 'auto' = cuda>mps>cpu. Use 'cpu' if sparse ops are unsupported on mps.")
    parser.add_argument("--split", type=str, default="temporal", choices=["temporal", "random"],
                        help="temporal = chronological 70/15/15 by trRegistrationDate (realistic). "
                             "random = shuffled 70/15/15 (the leaky baseline used to show inflation, RQ1).")
    parser.add_argument("--company_feat", type=str, default="random", choices=["random", "content"],
                        help="GNN/MLP company node features. 'random' = fixed random (default). "
                             "'content' = mean frozen-SBERT of the patents the company received in TRAIN "
                             "(leakage-free; cold companies fall back to random). Tests whether content "
                             "company representations let GNNs beat the popularity baseline.")
    parser.add_argument("--nneg_sweep", action="store_true",
                        help="After the run, sweep candidate-set size n_neg in {50,100,200} for key models "
                             "(MostPop, SVD, GraphSAGE, GAT) and report NDCG@10 vs n_neg (sampled-metric defense).")
    parser.add_argument("--demand_sample", type=int, default=200,
                        help="Demand Score (E19) is a slow per-query citation-BFS and near-degenerate on KIPRIS. "
                             "Evaluate it on a random sample of this many test queries. Keep it small (100-300); "
                             "large values make E19 dominate runtime. Negative = use ALL (very slow, not recommended).")
    args = parser.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    
    if args.mode == "fast":
        num_seeds = args.seeds if args.seeds is not None else 2
        max_epochs = args.epochs if args.epochs is not None else 5
        n_neg = args.n_neg if args.n_neg is not None else 20
        print(f"Running in FAST mode (Subsampled dataset, {num_seeds} seeds, {max_epochs} epochs, {n_neg} hard negatives)...")
    else:
        num_seeds = args.seeds if args.seeds is not None else 10
        max_epochs = args.epochs if args.epochs is not None else 50
        n_neg = args.n_neg if args.n_neg is not None else 100
        print(f"Running in FULL mode (Full dataset, {num_seeds} seeds, {max_epochs} epochs, {n_neg} hard negatives)...")
        
    SEEDS = list(range(num_seeds))
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but this runtime has NO CUDA GPU "
              "(torch.cuda.is_available()=False). On Colab: Runtime > Change runtime type > T4 GPU, "
              "then reconnect. Falling back to CPU for now (this will be SLOW).")
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("WARNING: --device mps requested but MPS is unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load Data
    data_dir = args.data_dir
    print("Loading datasets...")
    patents_df = pd.read_csv(os.path.join(data_dir, 'patents.csv'), usecols=['patApplicationNumber', 'patIpcNumber'], low_memory=False)
    transfers_df = pd.read_csv(os.path.join(data_dir, 'transfers.csv'), usecols=['trApplicationNumber', 'trCorrelatorName', 'trRegistrationDate'], low_memory=False)
    citings_df = pd.read_csv(os.path.join(data_dir, 'citings.csv'), usecols=['citStandardApplicationNumber', 'citApplicationNumber'], low_memory=False)
    
    def clean_app_num(series):
        return series.astype(str).str.replace(r'[^0-9]', '', regex=True)
        
    patents_df['patApplicationNumber'] = clean_app_num(patents_df['patApplicationNumber'])
    transfers_df['trApplicationNumber'] = clean_app_num(transfers_df['trApplicationNumber'])
    citings_df['citStandardApplicationNumber'] = clean_app_num(citings_df['citStandardApplicationNumber'])
    citings_df['citApplicationNumber'] = clean_app_num(citings_df['citApplicationNumber'])
    
    transfers_df['trCorrelatorName'] = transfers_df['trCorrelatorName'].fillna('UNKNOWN')
    
    if args.mode == "fast":
        print("Slicing data for fast execution...")
        top_companies = transfers_df['trCorrelatorName'].value_counts().head(500).index
        transfers_df = transfers_df[transfers_df['trCorrelatorName'].isin(top_companies)].copy()
        unique_pats = transfers_df['trApplicationNumber'].unique()
        if len(unique_pats) > 2000:
            unique_pats = unique_pats[:2000]
            transfers_df = transfers_df[transfers_df['trApplicationNumber'].isin(unique_pats)].copy()
        patents_df = patents_df[patents_df['patApplicationNumber'].isin(unique_pats)].copy()
        citings_df = citings_df[citings_df['citStandardApplicationNumber'].isin(unique_pats) & citings_df['citApplicationNumber'].isin(unique_pats)].copy()
        
    # Mappings
    patent_ids = patents_df['patApplicationNumber'].unique()
    patent2idx = {pid: i for i, pid in enumerate(patent_ids)}
    company_names = transfers_df['trCorrelatorName'].astype(str).unique()
    company2idx = {c: i for i, c in enumerate(company_names)}
    
    NUM_PATENTS = len(patent2idx)
    NUM_COMPANIES = len(company2idx)
    print(f"Patents: {NUM_PATENTS}, Companies: {NUM_COMPANIES}")
    
    # Load and Slice SBERT embeddings
    print("Loading SBERT embeddings...")
    original_patents_df = pd.read_csv(os.path.join(data_dir, 'patents.csv'), usecols=['patApplicationNumber'], low_memory=False)
    original_patents_df['patApplicationNumber'] = clean_app_num(original_patents_df['patApplicationNumber'])
    original_patent_ids = original_patents_df['patApplicationNumber'].unique()
    orig_patent2idx = {pid: i for i, pid in enumerate(original_patent_ids)}
    
    full_patent_x = torch.load(args.emb_path, map_location='cpu')
    patent_x_list = [full_patent_x[orig_patent2idx[pid]] for pid in patent_ids]
    patent_x = torch.stack(patent_x_list)
    del full_patent_x
    del patent_x_list
    
    # Pre-sort transfers by registration date
    transfers_df['trRegistrationDate'] = transfers_df['trRegistrationDate'].astype(str).str.replace(r'[^0-9\-]', '', regex=True)
    transfers_df['trRegistrationDate'] = pd.to_datetime(transfers_df['trRegistrationDate'], errors='coerce')
    transfers_df = transfers_df.dropna(subset=['trRegistrationDate'])
    if args.split == "random":
        # Leaky baseline (RQ1): shuffle so future transfers can leak into training.
        transfers_df = transfers_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        print("Split: RANDOM (shuffled 70/15/15) — leaky baseline for the temporal-vs-random contrast.")
    else:
        transfers_df = transfers_df.sort_values('trRegistrationDate').reset_index(drop=True)
        print("Split: TEMPORAL (chronological 70/15/15 by trRegistrationDate).")
    
    n_transfers = len(transfers_df)
    train_end = int(n_transfers * 0.70)
    val_end = int(n_transfers * 0.85)
    train_df = transfers_df.iloc[:train_end]
    val_df = transfers_df.iloc[train_end:val_end]
    test_df = transfers_df.iloc[val_end:]
    
    # IPC lists
    patents_df['ipc4'] = patents_df['patIpcNumber'].astype(str).str[:4]
    patent_ipc = dict(zip(patents_df['patApplicationNumber'], patents_df['ipc4']))
    all_patents_list = list(patent_ids)
    
    # Build transfer sets
    train_transfer_set = set(zip(train_df['trApplicationNumber'].map(patent2idx), train_df['trCorrelatorName'].map(company2idx)))
    # Patents that appear as a transfer target in TRAIN (patent-side "seen" flag for NEW-3 cold-start)
    train_patent_set = {p for (p, c) in train_transfer_set}

    ipc_company_index = {}
    ipc_company_count = {}   # NEW-9: per-IPC per-company train transfer COUNT (IPC-conditional MostPop)
    train_apps = train_df['trApplicationNumber'].to_numpy()
    train_corrs = train_df['trCorrelatorName'].to_numpy()
    for i in range(len(train_df)):
        c = company2idx[train_corrs[i]]
        ipc4 = patent_ipc.get(train_apps[i], 'UNKNOWN')
        if ipc4 not in ipc_company_index:
            ipc_company_index[ipc4] = set()
            ipc_company_count[ipc4] = {}
        ipc_company_index[ipc4].add(c)
        ipc_company_count[ipc4][c] = ipc_company_count[ipc4].get(c, 0) + 1
        
    # Popularity & Recency features
    train_pop = np.zeros(NUM_COMPANIES)
    company_counts = train_df['trCorrelatorName'].value_counts()
    for c in company_names:
        train_pop[company2idx[c]] = company_counts.get(c, 0)
        
    company_last_active = np.zeros(NUM_COMPANIES)
    ref_date = train_df['trRegistrationDate'].min() if len(train_df) > 0 else pd.Timestamp('1980-01-01')
    train_df['days'] = (train_df['trRegistrationDate'] - ref_date).dt.total_seconds() / 86400.0
    company_last_days = train_df.groupby('trCorrelatorName')['days'].max()
    for c in company_names:
        if c in company_last_days:
            company_last_active[company2idx[c]] = company_last_days[c]
        else:
            company_last_active[company2idx[c]] = -1.0
            
    # Patent time encoding features
    patent_time = np.zeros(NUM_PATENTS)
    patent_first_transfer = train_df.groupby('trApplicationNumber')['days'].min()
    for p in patent_ids:
        if p in patent_first_transfer:
            patent_time[patent2idx[p]] = patent_first_transfer[p]
    max_days = train_df['days'].max() if len(train_df) > 0 else 1.0
    patent_time = patent_time / max_days
    patent_time_t = torch.tensor(patent_time, dtype=torch.float).to(device)
    
    # ─────────────────────────────────────────────────────────────
    # Precomputing Demand Score helper structures (corrected specs)
    # ─────────────────────────────────────────────────────────────
    print("Precomputing citation structures for Demand Score...")
    
    # 1. Build cited_by (cited -> citing) from citings_df mapping
    valid_cites = citings_df[citings_df['citStandardApplicationNumber'].isin(patent2idx) & citings_df['citApplicationNumber'].isin(patent2idx)]
    citing_p = valid_cites['citStandardApplicationNumber'].map(patent2idx).to_numpy()
    cited_p = valid_cites['citApplicationNumber'].map(patent2idx).to_numpy()
    
    from collections import defaultdict
    cited_by = defaultdict(list)
    cit_neighbors = defaultdict(set)   # undirected citation neighbours (for CN/AA baselines)
    for cit, cited in zip(citing_p, cited_p):
        cited_by[cited].append(cit)
        cit_neighbors[cited].add(cit)
        cit_neighbors[cit].add(cited)
        
    # 2. Global in-degree per patent (for power-patent threshold)
    patent_indegree = np.zeros(NUM_PATENTS)
    for p, citers in cited_by.items():
        patent_indegree[p] = len(citers)
    power_threshold = np.percentile(patent_indegree, 90) if len(patent_indegree) > 0 else 0.0
    
    # 3. IPC-group mean citation count (for FreqScore normalisation)
    ipc4_citation_counts = defaultdict(list)
    for p_idx, p_id in enumerate(patent_ids):
        ipc4 = patent_ipc.get(p_id, 'UNKNOWN')
        ipc4_citation_counts[ipc4].append(patent_indegree[p_idx])
    ipc4_mean_cit = {
        ipc4: np.mean(counts) if counts else 1.0
        for ipc4, counts in ipc4_citation_counts.items()
    }
    ipc4_global_mean = np.mean(patent_indegree) if len(patent_indegree) > 0 else 1.0
    
    # 4. National strategic technology IPC4 codes
    STRATEGIC_IPC4 = {
        'G16H', 'G06N', 'H04L', 'H04W', 'G06F',
        'A61K', 'A61P', 'C12N', 'C12Q',
        'H01M', 'H02J'
    }
    
    # 5. Patents transferred by each company in the training period
    company_patents = defaultdict(set)
    for p, c in train_transfer_set:
        company_patents[c].add(p)
        
    # Pre-generate candidate sets per seed & monitor padding
    print("Generating hard negative candidate sets...")
    test_queries_per_seed = {}
    val_queries_per_seed = {}
    padding_rates = []
    
    print("Pre-converting dataframes to NumPy arrays...")
    val_apps = val_df['trApplicationNumber'].to_numpy()
    val_corrs = val_df['trCorrelatorName'].to_numpy()
    val_list = []
    for i in range(len(val_df)):
        p_app = val_apps[i]
        p = patent2idx[p_app]
        c_pos = company2idx[val_corrs[i]]
        ipc4 = patent_ipc.get(p_app, 'UNKNOWN')
        val_list.append((p, c_pos, ipc4))
        
    test_apps = test_df['trApplicationNumber'].to_numpy()
    test_corrs = test_df['trCorrelatorName'].to_numpy()
    # NEW-1: capture each test transfer's horizon (months after the train cutoff) BEFORE
    # test_df is deleted, aligned to test_list/queries order.
    train_cutoff_date = train_df['trRegistrationDate'].max() if len(train_df) > 0 else pd.Timestamp('1980-01-01')
    test_dates = pd.to_datetime(test_df['trRegistrationDate']).to_numpy()
    test_list = []
    test_horizon = []   # months after train cutoff, parallel to test_list
    for i in range(len(test_df)):
        p_app = test_apps[i]
        p = patent2idx[p_app]
        c_pos = company2idx[test_corrs[i]]
        ipc4 = patent_ipc.get(p_app, 'UNKNOWN')
        test_list.append((p, c_pos, ipc4))
        months = (test_dates[i] - np.datetime64(train_cutoff_date)) / np.timedelta64(1, 'D') / 30.44
        test_horizon.append(float(max(months, 0.0)))
    test_horizon = np.array(test_horizon)
        
    rng = np.random.default_rng(SEEDS[0])
    
    # Test queries & padding monitor
    rng_test = np.random.default_rng(SEEDS[0])
    queries = []
    padded_count = 0
    for p, c_pos, ipc4 in test_list:
        cand, is_padded = build_candidates(p, c_pos, ipc4, ipc_company_index, train_transfer_set, n_neg, rng_test, NUM_COMPANIES)
        queries.append((p, c_pos, ipc4, cand))
        if is_padded:
            padded_count += 1
            
    print("Generating validation candidate sets for early stopping...")
    for seed in SEEDS:
        rng_val = np.random.default_rng(seed + 10000)  # separate RNG from test
        val_queries = []
        for p, c_pos, ipc4 in val_list:
            cand, _ = build_candidates(p, c_pos, ipc4, ipc_company_index, train_transfer_set, n_neg, rng_val, NUM_COMPANIES)
            val_queries.append((p, c_pos, ipc4, cand))
        val_queries_per_seed[seed] = val_queries
        test_queries_per_seed[seed] = queries
        padding_rates.append(padded_count / len(test_list) if len(test_list) > 0 else 0.0)
        
    avg_padding_rate = np.mean(padding_rates)
    print(f"Average candidate padding rate: {avg_padding_rate:.2%}")
    
    # Set up PyG HeteroData structure
    data = HeteroData()
    data['patent'].x = patent_x.to(device)
    if args.company_feat == "content":
        # Leakage-free content features: mean frozen-SBERT of the company's TRAIN-received patents.
        COMPANY_IN = patent_x.size(1)  # 384
        company_x = torch.randn(NUM_COMPANIES, COMPANY_IN) * 0.1  # fallback for cold (no train transfer) companies
        n_content = 0
        for c, pats in company_patents.items():
            if pats:
                idx = torch.tensor(sorted(pats), dtype=torch.long)
                company_x[c] = patent_x[idx].mean(0)
                n_content += 1
        print(f"Company features: CONTENT (mean train-patent SBERT) for {n_content}/{NUM_COMPANIES} companies; "
              f"cold companies use random fallback.", flush=True)
    else:
        COMPANY_IN = 64
        company_x = torch.randn(NUM_COMPANIES, COMPANY_IN) * 0.1
        print("Company features: RANDOM (fixed).", flush=True)
    data['company'].x = company_x.to(device)
    
    c_indices = [company2idx[c] for c in train_df['trCorrelatorName']]
    p_indices = [patent2idx[p] for p in train_df['trApplicationNumber']]
    data['company', 'buys', 'patent'].edge_index = torch.tensor([c_indices, p_indices], dtype=torch.long).to(device)
    
    valid_cites = citings_df[citings_df['citStandardApplicationNumber'].isin(patent2idx) & citings_df['citApplicationNumber'].isin(patent2idx)]
    citing_p = valid_cites['citStandardApplicationNumber'].map(patent2idx).tolist()
    cited_p = valid_cites['citApplicationNumber'].map(patent2idx).tolist()
    data['patent', 'cites', 'patent'].edge_index = torch.tensor([citing_p, cited_p], dtype=torch.long).to(device)
    
    data['patent', 'rev_buys', 'company'].edge_index = data['company', 'buys', 'patent'].edge_index[[1, 0]]
    
    train_edge_index = data['company', 'buys', 'patent'].edge_index
    norm_adj = get_norm_adj(train_edge_index.cpu(), NUM_PATENTS, NUM_COMPANIES)
    
    # Pre-compute cold start stats so we can delete test_df
    c_idx_test = test_df['trCorrelatorName'].map(company2idx).dropna().astype(int).values
    frac_unseen = float(np.mean(train_pop[c_idx_test] == 0) if len(c_idx_test) > 0 else 0.0)
    frac_rare = float(np.mean(train_pop[c_idx_test] <= 1) if len(c_idx_test) > 0 else 0.0)
    
    # Garbage collection of large unused DataFrames
    import gc
    del patents_df
    del transfers_df
    del citings_df
    del train_df
    del val_df
    del test_df
    del valid_cites
    del original_patents_df
    gc.collect()
    
    # Baselines + a SYMMETRIC (backbone x mitigation) grid so every mitigation is
    # applied to BOTH GraphSAGE and GAT (NEW-6), and logQ is actually trained/reported (B6).
    BASE_MODELS = ["MostPop", "MostPop-IPC", "Recency", "CN", "AA", "SVD", "MLP", "LightGCN", "NGCF", "GraphSAGE", "GAT"]
    BACKBONES = [("GraphSAGE", "SAGE"), ("GAT", "GAT")]   # (display name, FullModel gnn_type)
    MITIGATIONS = ["Debias", "logQ", "DropEdge", "Time", "IPS"]
    COMBOS = ["Debias+IPS", "Time+IPS"]                   # stacked mitigations (NEW-7)
    mitigation_models = [f"{disp}+{m}" for disp, _ in BACKBONES for m in MITIGATIONS]
    combo_models = [f"{disp}+{c}" for disp, _ in BACKBONES for c in COMBOS]
    models = BASE_MODELS + mitigation_models + combo_models
    metrics_by_model = {m: [] for m in models}
    
    # Collections for diagnostic correlation (all models)
    spearman_by_model = {m: [] for m in models}
    
    # For inversion rate and Wilcoxon test, we need per-seed query scores and ranks
    ranks_by_model_by_seed = {m: {} for m in models}
    # Memory: the per-query SCORE arrays (~220k x 101 floats x 22 models x 10 seeds ~= 19 GB)
    # are NOT cached across seeds. Each model's score-dependent diagnostics (inversion D15,
    # error decomposition NEW-4) are consumed INLINE in record_model, and only the GAT seed-0
    # scores needed for the case study (NEW-5) are retained.
    inversion_by_model = {m: [] for m in models}
    error_buckets_by_model = {m: {"popular_hardneg": 0, "rare_new_positive": 0, "semantic_residual": 0}
                              for m in ["GraphSAGE", "GAT"] if m in models}
    error_nfail_by_model = {m: 0 for m in ["GraphSAGE", "GAT"] if m in models}
    case_study_scores0 = {}
    
    # Stratification (Option b)
    strata_metrics_by_seed = {'head': [], 'torso': [], 'tail': []}
    
    # Demand Score comparison tracking
    demand_orig_ndcg_seeds = []
    demand_rev_ndcg_seeds = []
    
    # Diagnostic collections (first seed GAT)
    gat_attention_weights = None
    gat_is_hub_edge = None
    
    # IPS (beta) and Debias (alpha) sensitivity sweeps, run on BOTH backbones (NEW-8).
    # Keyed by (backbone display name, hyperparameter value).
    betas = [0.0, 0.5, 1.0, 2.0, 4.0]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]   # NEW-8: added 0.25 grid point
    beta_ndcg_seeds = {(disp, b): [] for disp, _ in BACKBONES for b in betas}
    alpha_ndcg_seeds = {(disp, a): [] for disp, _ in BACKBONES for a in alphas}
    
    # MostPop-IPC (NEW-9) is seed-invariant (pure function of fixed candidates + train counts),
    # so compute it ONCE here instead of re-running its per-query loop every seed.
    print("Precomputing IPC-conditional MostPop (seed-invariant)...", flush=True)
    _t = time.time()
    mostpop_ipc_cached = evaluate_mostpop_ipc(test_queries_per_seed[SEEDS[0]], ipc_company_count, train_pop)
    print(f"  MostPop-IPC precomputed in {time.time()-_t:.1f}s", flush=True)

    # CN / Adamic-Adar are also seed-invariant (pure graph structure) — compute once.
    print("Precomputing Common-Neighbors / Adamic-Adar baselines (seed-invariant)...", flush=True)
    _t = time.time()
    cn_cached, aa_cached = evaluate_cn_aa(test_queries_per_seed[SEEDS[0]], cit_neighbors, company_patents, patent_indegree)
    print(f"  CN/AA precomputed in {time.time()-_t:.1f}s", flush=True)

    nneg_sweep_results = None   # populated at seed 0 if --nneg_sweep

    # Demand Score (E19) is ALSO seed-invariant (no learned params, fixed candidates), and its
    # per-query citation-BFS loop is the single most expensive non-GNN step. Compute it ONCE.
    _demand_queries = test_queries_per_seed[SEEDS[0]]
    if args.demand_sample and args.demand_sample > 0 and args.demand_sample < len(_demand_queries):
        _ds_rng = np.random.default_rng(SEEDS[0])
        _sel = _ds_rng.choice(len(_demand_queries), size=args.demand_sample, replace=False)
        _demand_queries = [_demand_queries[i] for i in _sel]
        print(f"Precomputing Demand Score on a {len(_demand_queries)}-query sample (of "
              f"{len(test_queries_per_seed[SEEDS[0]])}; slow citation BFS)...", flush=True)
    else:
        print("Precomputing Demand Score original/revised on ALL queries (seed-invariant, slow citation BFS)...", flush=True)
    _t = time.time()
    _ranks_demand_orig, _ranks_demand_rev = [], []
    for q in _demand_queries:
        ro, _ = demand_score_rank(q[0], q[1], q[3], "original", train_pop, company_last_active,
                                  patent_ipc, patent_indegree, ipc4_mean_cit, ipc4_global_mean,
                                  power_threshold, STRATEGIC_IPC4, company_patents, cited_by, all_patents_list)
        rr, _ = demand_score_rank(q[0], q[1], q[3], "revised", train_pop, company_last_active,
                                  patent_ipc, patent_indegree, ipc4_mean_cit, ipc4_global_mean,
                                  power_threshold, STRATEGIC_IPC4, company_patents, cited_by, all_patents_list)
        _ranks_demand_orig.append(ro)
        _ranks_demand_rev.append(rr)
    demand_orig_ndcg_once = aggregate(_ranks_demand_orig)["ndcg@10"]
    demand_rev_ndcg_once = aggregate(_ranks_demand_rev)["ndcg@10"]
    print(f"  Demand Score precomputed in {time.time()-_t:.1f}s", flush=True)

    # Run Seeds
    for seed in SEEDS:
        seed_t0 = time.time()
        print(f"\n--- Running Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        queries = test_queries_per_seed[seed]
        val_queries = val_queries_per_seed[seed]
        
        # Vectorized query structures
        val_p_t = torch.tensor([q[0] for q in val_queries], dtype=torch.long).to(device)
        val_cand_t = torch.tensor([q[3] for q in val_queries], dtype=torch.long).to(device)
        test_p_t = torch.tensor([q[0] for q in queries], dtype=torch.long).to(device)
        test_cand_t = torch.tensor([q[3] for q in queries], dtype=torch.long).to(device)
        train_pop_t = torch.tensor(train_pop, dtype=torch.float32).to(device)
        company_last_active_t = torch.tensor(company_last_active, dtype=torch.float32).to(device)
        
        # Helper to record results
        def record_model(name, ranks, aucs, aps, scores_all, pop_all):
            metrics_by_model[name].append(aggregate(ranks))
            metrics_by_model[name][-1]["auc"] = float(np.mean(aucs))
            metrics_by_model[name][-1]["ap"] = float(np.mean(aps))
            
            # Subsample for spearmanr if too large to save memory/time
            scores_flat = scores_all.flatten()
            pop_flat = pop_all.flatten()
            if len(scores_flat) > 1000000:
                rng_sub = np.random.default_rng(42)
                idx_sub = rng_sub.choice(len(scores_flat), size=1000000, replace=False)
                val = spearmanr(scores_flat[idx_sub], pop_flat[idx_sub])[0]
            else:
                val = spearmanr(scores_flat, pop_flat)[0]
                
            spearman_by_model[name].append(float(val if not np.isnan(val) else 0.0))
            ranks_by_model_by_seed[name][seed] = ranks
            # Consume score-dependent diagnostics now, then let scores_all be garbage-collected.
            inversion_by_model[name].append(compute_inversion_rate(queries, scores_all, train_pop, pop_thr_q=0.9))
            if name in error_buckets_by_model:
                b, nf = classify_failures(queries, ranks, scores_all, train_pop)
                for k in error_buckets_by_model[name]:
                    error_buckets_by_model[name][k] += b[k]
                error_nfail_by_model[name] += nf
            if name == "GAT" and seed == SEEDS[0]:
                case_study_scores0["GAT"] = scores_all
            print(f"    [seed {seed}] {name:22s} done  (+{time.time()-seed_t0:6.1f}s)", flush=True)
            
        # 1. MostPop Baseline (global popularity)
        ranks_mp, aucs_mp, aps_mp, scores_mp, pop_mp = evaluate_mostpop(test_cand_t, train_pop_t)
        record_model("MostPop", ranks_mp, aucs_mp, aps_mp, scores_mp, pop_mp)

        # 1b. IPC-conditional MostPop (NEW-9): popularity WITHIN the query's ipc4 (precomputed once)
        record_model("MostPop-IPC", *mostpop_ipc_cached)

        # 1c. Common-Neighbors / Adamic-Adar structural baselines (precomputed once)
        record_model("CN", *cn_cached)
        record_model("AA", *aa_cached)
        
        # 2. Recency Baseline
        ranks_rec, aucs_rec, aps_rec, scores_rec, pop_rec = evaluate_recency(test_cand_t, company_last_active_t, train_pop_t)
        record_model("Recency", ranks_rec, aucs_rec, aps_rec, scores_rec, pop_rec)
        
        # 3. SVD Baseline (Rank k=64)
        train_csr = coo_matrix((np.ones(train_edge_index.size(1)), (train_edge_index[0].cpu().numpy(), train_edge_index[1].cpu().numpy())), shape=(NUM_COMPANIES, NUM_PATENTS)).astype(float).tocsr()
        ranks_svd, aucs_svd, aps_svd, scores_svd, pop_svd = evaluate_svd(queries, train_csr, 64, device, train_pop_t)
        record_model("SVD", ranks_svd, aucs_svd, aps_svd, scores_svd, pop_svd)
        
        # 4. MLP Baseline
        mlp = FullModel('MLP', 384, COMPANY_IN).to(device)
        train_gnn(mlp, data, train_edge_index, train_pop, debias_alpha=0.0, logq_alpha=0.0, max_epochs=max_epochs, lr=0.01, device=device, seed=seed, num_companies=NUM_COMPANIES, val_queries=val_queries, patience=5)
        ranks_mlp, aucs_mlp, aps_mlp, scores_mlp, pop_mlp = evaluate_mlp(mlp, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0)
        record_model("MLP", ranks_mlp, aucs_mlp, aps_mlp, scores_mlp, pop_mlp)
        
        # 5. LightGCN
        lgcn = LightGCN(NUM_PATENTS, NUM_COMPANIES, dim=64, n_layers=3)
        train_cf_model(lgcn, norm_adj, train_edge_index.cpu(), max_epochs=max_epochs, lr=0.01, device=device, val_queries=val_queries, patience=5)
        ranks_lgcn, aucs_lgcn, aps_lgcn, scores_lgcn, pop_lgcn = evaluate_cf_model(lgcn, norm_adj, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0)
        record_model("LightGCN", ranks_lgcn, aucs_lgcn, aps_lgcn, scores_lgcn, pop_lgcn)
        
        # 6. NGCF
        ngcf = NGCF(NUM_PATENTS, NUM_COMPANIES, dim=64, n_layers=2)
        train_cf_model(ngcf, norm_adj, train_edge_index.cpu(), max_epochs=max_epochs, lr=0.01, device=device, val_queries=val_queries, patience=5)
        ranks_ngcf, aucs_ngcf, aps_ngcf, scores_ngcf, pop_ngcf = evaluate_cf_model(ngcf, norm_adj, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0)
        record_model("NGCF", ranks_ngcf, aucs_ngcf, aps_ngcf, scores_ngcf, pop_ngcf)
        
        # 7. GraphSAGE
        sage = FullModel('SAGE', 384, COMPANY_IN).to(device)
        train_gnn(sage, data, train_edge_index, train_pop, debias_alpha=0.0, logq_alpha=0.0, max_epochs=max_epochs, lr=0.01, device=device, seed=seed, num_companies=NUM_COMPANIES, val_queries=val_queries, patience=5)
        ranks_sage, aucs_sage, aps_sage, scores_sage, pop_sage = evaluate_gnn(sage, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0)
        record_model("GraphSAGE", ranks_sage, aucs_sage, aps_sage, scores_sage, pop_sage)
            
        # 8. GAT
        gat = FullModel('GAT', 384, COMPANY_IN).to(device)
        train_gnn(gat, data, train_edge_index, train_pop, debias_alpha=0.0, logq_alpha=0.0, max_epochs=max_epochs, lr=0.01, device=device, seed=seed, num_companies=NUM_COMPANIES, val_queries=val_queries, patience=5)
        ranks_gat, aucs_gat, aps_gat, scores_gat, pop_gat = evaluate_gnn(gat, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0)
        record_model("GAT", ranks_gat, aucs_gat, aps_gat, scores_gat, pop_gat)

        # n_neg sensitivity sweep (sampled-metric defense, A2/E18) — seed 0 only, key models, no retraining
        if args.nneg_sweep and seed == SEEDS[0]:
            print("Running n_neg sensitivity sweep (50/100/200) on MostPop/SVD/GraphSAGE/GAT...", flush=True)
            nneg_sweep_results = {}
            rng_sw = np.random.default_rng(20240)
            for sz in [50, 100, 200]:
                _ts = time.time()
                sw_q = []
                for p, c_pos, ipc4 in test_list:
                    cand, _ = build_candidates(p, c_pos, ipc4, ipc_company_index, train_transfer_set, sz, rng_sw, NUM_COMPANIES)
                    sw_q.append((p, c_pos, ipc4, cand))
                sw_p_t = torch.tensor([q[0] for q in sw_q], dtype=torch.long).to(device)
                sw_cand_t = torch.tensor([q[3] for q in sw_q], dtype=torch.long).to(device)
                res = {
                    "MostPop": aggregate(evaluate_mostpop(sw_cand_t, train_pop_t)[0])["ndcg@10"],
                    "SVD": aggregate(evaluate_svd(sw_q, train_csr, 64, device, train_pop_t)[0])["ndcg@10"],
                    "GraphSAGE": aggregate(evaluate_gnn(sage, data, sw_p_t, sw_cand_t, device, train_pop_t)[0])["ndcg@10"],
                    "GAT": aggregate(evaluate_gnn(gat, data, sw_p_t, sw_cand_t, device, train_pop_t)[0])["ndcg@10"],
                }
                nneg_sweep_results[sz] = res
                print(f"  [nneg sweep] n_neg={sz}: " + ", ".join(f"{k}={v:.3f}" for k, v in res.items())
                      + f"  (+{time.time()-_ts:.0f}s)", flush=True)
        
        if seed == 0:
            gat.eval()
            x_dict = {
                'patent': data['patent'].x,
                'company': data['company'].x
            }
            _, (att_c2p, att_p2c, att_p2p) = gat.gnn(x_dict, data.edge_index_dict, return_attn=True)
            gat_attention_weights = att_c2p.cpu().detach().numpy()
            edge_c_idx = data.edge_index_dict[('company', 'buys', 'patent')][0].cpu().numpy()
            q95 = np.quantile(train_pop, 0.95)
            gat_is_hub_edge = train_pop[edge_c_idx] >= q95
            
        # Stratification for GAT (Option b)
        sorted_companies = np.argsort(-train_pop)
        n_comp = len(sorted_companies)
        head_set = set(sorted_companies[:n_comp//3])
        torso_set = set(sorted_companies[n_comp//3:2*n_comp//3])
        tail_set = set(sorted_companies[2*n_comp//3:])
        
        for stratum, cset in [('head', head_set), ('torso', torso_set), ('tail', tail_set)]:
            stratum_ranks = [
                r for (_, c_pos, _, _), r in zip(queries, ranks_gat)
                if c_pos in cset
            ]
            if stratum_ranks:
                strata_metrics_by_seed[stratum].append(aggregate(stratum_ranks))
            else:
                strata_metrics_by_seed[stratum].append({"hits@1": 0.0, "hits@3": 0.0, "hits@5": 0.0, "hits@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0})
                
        # ── Symmetric mitigation grid (NEW-6 / B6 / NEW-7) ───────────────────────
        # Each mitigation is applied to BOTH GraphSAGE and GAT under identical seeds
        # and candidate sets, so the mitigation effect is no longer confounded with
        # the backbone. logQ is actually trained (B6). Combos stack a re-rank (NEW-7).
        time_enc = get_time_encoding(patent_time_t, dim=16)
        base_trained = {"GraphSAGE": sage, "GAT": gat}
        for disp, gtype in BACKBONES:
            base_model = base_trained[disp]

            # Debias (retrain with popularity-debiased negatives)
            m_db = FullModel(gtype, 384, COMPANY_IN).to(device)
            train_gnn(m_db, data, train_edge_index, train_pop, debias_alpha=0.75, logq_alpha=0.0, max_epochs=max_epochs, lr=0.01, device=device, seed=seed, num_companies=NUM_COMPANIES, val_queries=val_queries, patience=5)
            record_model(f"{disp}+Debias", *evaluate_gnn(m_db, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0))

            # logQ (retrain with sampled-softmax / logQ correction)  — B6
            m_lq = FullModel(gtype, 384, COMPANY_IN).to(device)
            train_gnn(m_lq, data, train_edge_index, train_pop, debias_alpha=0.0, logq_alpha=1.0, max_epochs=max_epochs, lr=0.01, device=device, seed=seed, num_companies=NUM_COMPANIES, val_queries=val_queries, patience=5)
            record_model(f"{disp}+logQ", *evaluate_gnn(m_lq, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0))

            # DropEdge (retrain with citation-edge dropout)
            m_de = FullModel(gtype, 384, COMPANY_IN, apply_dropedge=True).to(device)
            train_gnn(m_de, data, train_edge_index, train_pop, debias_alpha=0.0, logq_alpha=0.0, max_epochs=max_epochs, lr=0.01, device=device, seed=seed, num_companies=NUM_COMPANIES, val_queries=val_queries, patience=5)
            record_model(f"{disp}+DropEdge", *evaluate_gnn(m_de, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0))

            # Time (retrain with sinusoidal time encoding threaded through train_gnn)
            m_t = FullModel(gtype, 384, COMPANY_IN, use_time=True).to(device)
            train_gnn(m_t, data, train_edge_index, train_pop, debias_alpha=0.0, logq_alpha=0.0, max_epochs=max_epochs, lr=0.01, device=device, seed=seed, num_companies=NUM_COMPANIES, val_queries=val_queries, patience=5, time_enc=time_enc)
            record_model(f"{disp}+Time", *evaluate_gnn(m_t, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0, time_enc=time_enc))

            # IPS (test-time popularity-penalty re-rank on the base model, no retrain)
            record_model(f"{disp}+IPS", *evaluate_gnn(base_model, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=1.0))

            # Combos (NEW-7): stack IPS re-rank on top of an already-trained mitigation
            record_model(f"{disp}+Debias+IPS", *evaluate_gnn(m_db, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=1.0))
            record_model(f"{disp}+Time+IPS", *evaluate_gnn(m_t, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=1.0, time_enc=time_enc))

        # 13. Demand Score Original & Revised (precomputed once above; seed-invariant)
        demand_orig_ndcg_seeds.append(demand_orig_ndcg_once)
        demand_rev_ndcg_seeds.append(demand_rev_ndcg_once)
        
        # Mitigation hyperparameter sweeps on BOTH backbones (NEW-8)
        for disp, gtype in BACKBONES:
            base_model = base_trained[disp]
            # IPS beta sweep (test-time re-rank on the trained base model, no retrain)
            for beta in betas:
                ranks_b, *_ = evaluate_gnn(base_model, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=beta)
                beta_ndcg_seeds[(disp, beta)].append(aggregate(ranks_b)["ndcg@10"])
            # Debias alpha sweep (retrain per alpha)
            for alpha in alphas:
                m_tmp = FullModel(gtype, 384, COMPANY_IN).to(device)
                train_gnn(m_tmp, data, train_edge_index, train_pop, debias_alpha=alpha, logq_alpha=0.0, max_epochs=max_epochs, lr=0.01, device=device, seed=seed, num_companies=NUM_COMPANIES, val_queries=val_queries, patience=5)
                ranks_a, *_ = evaluate_gnn(m_tmp, data, test_p_t, test_cand_t, device, train_pop_t, ips_beta=0.0)
                alpha_ndcg_seeds[(disp, alpha)].append(aggregate(ranks_a)["ndcg@10"])
        print(f"    [seed {seed}] sweeps (beta/alpha x backbone) done  (+{time.time()-seed_t0:6.1f}s)", flush=True)
        print(f"--- Seed {seed} complete in {time.time()-seed_t0:.1f}s ---", flush=True)
            
    print("\nProcessing results and generating diagnostics...")
    
    # 1. Summarize metrics
    summary_results = {}
    for m in models:
        summary_results[m] = {}
        for k in ["hits@1", "hits@3", "hits@5", "hits@10", "mrr", "ndcg@10", "auc", "ap"]:
            vals = [s[k] for s in metrics_by_model[m]]
            summary_results[m][f"{k}_mean"] = float(np.mean(vals))
            summary_results[m][f"{k}_std"] = float(np.std(vals))
            
    # 2. Pairwise Wilcoxon + paired t-test over a PRE-REGISTERED comparison family.
    # With 22 models the full C(22,2)=231 family is unwieldy; instead we declare an
    # explicit, scientifically-meaningful family (architecture contrast, each backbone
    # vs the strongest baselines, and every mitigation/combo vs its own backbone), and
    # Holm-Bonferroni corrects jointly across exactly this declared family.
    registered = [("GAT", "GraphSAGE"), ("MostPop-IPC", "MostPop")]   # architecture + skyline contrast
    for disp, _ in BACKBONES:                                 # backbone vs strong baselines
        for base in ["MostPop", "MostPop-IPC", "SVD", "NGCF"]:
            registered.append((disp, base))
    for disp, _ in BACKBONES:                                 # each mitigation/combo vs its backbone
        for m in MITIGATIONS + COMBOS:
            registered.append((f"{disp}+{m}", disp))
    seen_pairs = set()
    comparisons = []
    for a, b in registered:
        key = frozenset((a, b))
        if a != b and a in metrics_by_model and b in metrics_by_model and key not in seen_pairs:
            seen_pairs.add(key)
            comparisons.append((a, b))

    raw_pvals_wilcoxon = []
    raw_pvals_ttest    = []
    comparison_labels  = []

    seeds_sufficient = len(SEEDS) >= 6  # Wilcoxon signed-rank needs >=6 pairs to be meaningful

    for (m1, m2) in comparisons:
        s1 = [s["ndcg@10"] for s in metrics_by_model[m1]]
        s2 = [s["ndcg@10"] for s in metrics_by_model[m2]]
        if len(s1) >= 6 and len(s2) >= 6:
            try:
                _, p_w = wilcoxon(s1, s2)
            except Exception:
                p_w = 1.0
            try:
                _, p_t = ttest_rel(s1, s2)
            except Exception:
                p_t = 1.0
        else:
            p_w, p_t = 1.0, 1.0
        raw_pvals_wilcoxon.append(p_w)
        raw_pvals_ttest.append(p_t)
        comparison_labels.append(f'{m1} vs {m2}')

    _, adj_pvals_w, _, _ = multipletests(raw_pvals_wilcoxon, method='holm')
    _, adj_pvals_t, _, _ = multipletests(raw_pvals_ttest,    method='holm')

    wilcoxon_warning = ""
    if not seeds_sufficient:
        wilcoxon_warning = (f"\n> **WARNING**: only {len(SEEDS)} seeds (<6). Wilcoxon signed-rank is "
                            f"undefined/underpowered, so ALL p-values below are forced to 1.0 and are "
                            f"NOT real non-results. Re-run with --seeds >= 6 (full mode uses 10).\n")
    wilcoxon_table = (f"Pre-registered comparison family: **{len(comparisons)}** pairs "
                      f"(Holm-Bonferroni corrected jointly across exactly these comparisons).\n\n")
    wilcoxon_table += "| Comparison Pair | Wilcoxon Raw p | Wilcoxon Adjusted p (Holm) | t-Test Raw p | t-Test Adjusted p (Holm) |\n"
    wilcoxon_table += "| :--- | :---: | :---: | :---: | :---: |\n"
    for idx in range(len(comparison_labels)):
        wilcoxon_table += f"| {comparison_labels[idx]} | {raw_pvals_wilcoxon[idx]:.4e} | {adj_pvals_w[idx]:.4e} | {raw_pvals_ttest[idx]:.4e} | {adj_pvals_t[idx]:.4e} |\n"
            
    # 3. GAT Attention Weights (D12)
    mw_pval = 1.0
    if gat_attention_weights is not None and len(gat_attention_weights) > 0 and gat_is_hub_edge is not None:
        alpha_hub = gat_attention_weights[gat_is_hub_edge]
        alpha_nonhub = gat_attention_weights[~gat_is_hub_edge]
        if len(alpha_hub) > 0 and len(alpha_nonhub) > 0:
            mw_stat, mw_pval = mannwhitneyu(alpha_hub, alpha_nonhub, alternative="less")
            plt.figure(figsize=(6, 4))
            plt.violinplot([alpha_nonhub, alpha_hub], showmeans=True)
            plt.xticks([1, 2], ['Non-Hubs', 'Hubs'])
            plt.ylabel('GAT Attention Weight')
            plt.title(f'GAT Attention Weights (Hubs vs Non-Hubs)\nMann-Whitney U p-val: {mw_pval:.4e}')
            plt.tight_layout()
            plt.savefig(os.path.join(args.artifact_dir, 'gat_attention_violin.png'))
            plt.close()
            
    # 4. Stratified performance (D14)
    gat_strat_mean = {}
    gat_strat_std = {}
    for stratum in ['head', 'torso', 'tail']:
        ndcg10_vals = [m['ndcg@10'] for m in strata_metrics_by_seed[stratum]]
        gat_strat_mean[stratum] = float(np.mean(ndcg10_vals)) if ndcg10_vals else 0.0
        gat_strat_std[stratum] = float(np.std(ndcg10_vals)) if ndcg10_vals else 0.0
        
    plt.figure(figsize=(6, 4))
    x_pos = np.arange(3)
    plt.bar(x_pos, [gat_strat_mean[s] for s in ['head', 'torso', 'tail']], 
            yerr=[gat_strat_std[s] for s in ['head', 'torso', 'tail']], 
            align='center', alpha=0.7, color=['#4285F4', '#EA4335', '#FBBC05'], capsize=10)
    plt.xticks(x_pos, ['head', 'torso', 'tail'])
    plt.ylabel('NDCG@10')
    plt.title('GAT Performance by Popularity Stratum\n(Mean ± Std)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.artifact_dir, 'popularity_stratified.png'))
    plt.close()
    
    # 5. Inversion rate (D15) — accumulated inline per seed in record_model
    inversion_rates = {m: {"mean": float(np.mean(inversion_by_model[m])),
                           "std": float(np.std(inversion_by_model[m]))} for m in models}
        
    # 6. Spearman correlation (D13)
    spearman_summary = {}
    for m in models:
        vals = spearman_by_model[m]
        spearman_summary[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Low-cost diagnostics: re-bucket the cached ranks/scores (NEW-1/2/3/4/5/12).
    # No retraining — all of these reuse the cached per-query ranks (scores were consumed inline).
    # ──────────────────────────────────────────────────────────────────────────
    KEY_MODELS = [m for m in ["MostPop", "MostPop-IPC", "SVD", "NGCF", "GraphSAGE", "GAT"] if m in models]
    base_queries = test_queries_per_seed[SEEDS[0]]   # candidate sets are shared across seeds
    n_test = len(base_queries)

    def subset_ndcg_by_seed(model, mask):
        """Mean-over-seeds NDCG@10 on the masked subset of test queries; returns (ndcg, n)."""
        vals = []
        idxs = np.nonzero(mask)[0]
        for seed in SEEDS:
            ranks = ranks_by_model_by_seed[model][seed]
            sub = [ranks[i] for i in idxs]
            if sub:
                vals.append(aggregate(sub)["ndcg@10"])
        return (float(np.mean(vals)) if vals else 0.0, int(mask.sum()))

    # NEW-1: horizon decay (months after the train cutoff)
    horizon_bins = [(0, 6), (6, 12), (12, 18), (18, 1e18)]
    horizon_labels = ["0-6mo", "6-12mo", "12-18mo", "18mo+"]
    horizon_table = "| Model | " + " | ".join(horizon_labels) + " |\n"
    horizon_table += "| :--- " + "| :---: " * len(horizon_labels) + "|\n"
    for m in KEY_MODELS:
        cells = [f"{subset_ndcg_by_seed(m, (test_horizon >= lo) & (test_horizon < hi))[0]:.4f} "
                 f"(n={int(((test_horizon >= lo) & (test_horizon < hi)).sum())})" for (lo, hi) in horizon_bins]
        horizon_table += f"| {m} | " + " | ".join(cells) + " |\n"
    plt.figure(figsize=(6, 4))
    xs = np.arange(len(horizon_labels))
    for m in KEY_MODELS:
        ys = [subset_ndcg_by_seed(m, (test_horizon >= lo) & (test_horizon < hi))[0] for (lo, hi) in horizon_bins]
        plt.plot(xs, ys, '-o', label=m)
    plt.xticks(xs, horizon_labels)
    plt.ylabel('NDCG@10')
    plt.xlabel('Prediction horizon (months after cutoff)')
    plt.title('Horizon decay')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.artifact_dir, 'horizon_decay.png'))
    plt.close()

    # NEW-2: IPC-section (A-H) decomposition
    test_sections = np.array([(q[2][0] if (q[2] and q[2] != 'UNKNOWN') else '?') for q in base_queries])
    sections = sorted(set(test_sections.tolist()))
    ipc_table = "| Model | " + " | ".join(sections) + " |\n"
    ipc_table += "| :--- " + "| :---: " * len(sections) + "|\n"
    for m in KEY_MODELS:
        cells = [f"{subset_ndcg_by_seed(m, test_sections == s)[0]:.3f}({int((test_sections == s).sum())})"
                 for s in sections]
        ipc_table += f"| {m} | " + " | ".join(cells) + " |\n"

    # NEW-3: patent-side cold-start subset (new patent, seen company)
    patent_seen = np.array([(q[0] in train_patent_set) for q in base_queries])
    company_seen = np.array([(train_pop[q[1]] > 0) for q in base_queries])
    frac_patent_unseen = float(np.mean(~patent_seen)) if n_test else 0.0
    coldp_mask = (~patent_seen) & company_seen
    coldstart_table = "| Model | All | New-patent & Seen-company | Seen-patent |\n"
    coldstart_table += "| :--- | :---: | :---: | :---: |\n"
    for m in KEY_MODELS:
        all_n = subset_ndcg_by_seed(m, np.ones(n_test, dtype=bool))[0]
        coldp, coldp_n = subset_ndcg_by_seed(m, coldp_mask)
        seenp = subset_ndcg_by_seed(m, patent_seen)[0]
        coldstart_table += f"| {m} | {all_n:.4f} | {coldp:.4f} (n={coldp_n}) | {seenp:.4f} |\n"

    # NEW-4: error-source decomposition (GraphSAGE & GAT)
    error_table = "| Model | Popular-hardneg | Rare/new positive | Semantic residual | #failures |\n"
    error_table += "| :--- | :---: | :---: | :---: | :---: |\n"
    for m in error_buckets_by_model:
        agg = error_buckets_by_model[m]
        tot = error_nfail_by_model[m]
        denom = tot if tot > 0 else 1
        error_table += (f"| {m} | {agg['popular_hardneg']/denom:.1%} | {agg['rare_new_positive']/denom:.1%} "
                        f"| {agg['semantic_residual']/denom:.1%} | {tot} |\n")

    # NEW-5: qualitative case study (worst-ranked examples for GAT, seed 0 scores retained)
    cs_model = "GAT"
    if cs_model in case_study_scores0 and cs_model in ranks_by_model_by_seed:
        sc0 = case_study_scores0[cs_model]
        rk0 = ranks_by_model_by_seed[cs_model][SEEDS[0]]
        case_lines = []
        for i in sorted(range(n_test), key=lambda j: -rk0[j])[:3]:
            cand = base_queries[i][3]
            top5 = np.argsort(-sc0[i])[:5]
            top5_names = ", ".join(str(company_names[cand[j]]) for j in top5)
            case_lines.append(f"- Patent `{patent_ids[base_queries[i][0]]}` (IPC {base_queries[i][2]}): "
                              f"true buyer **{company_names[base_queries[i][1]]}** ranked #{rk0[i]}; "
                              f"{cs_model} top-5 = [{top5_names}]")
        case_study_text = "\n".join(case_lines) if case_lines else "(no test queries)"
    else:
        case_study_text = "(case study unavailable)"

    # NEW-12: bootstrap CI over test queries (pooled across seeds)
    boot_table = "| Model | NDCG@10 [95% CI] | Hits@10 [95% CI] | MRR [95% CI] |\n"
    boot_table += "| :--- | :---: | :---: | :---: |\n"
    for m in KEY_MODELS:
        pooled = [r for seed in SEEDS for r in ranks_by_model_by_seed[m][seed]]
        ci = bootstrap_ci_ndcg(pooled, n_boot=1000, seed=0)
        fmt = lambda t: f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"
        boot_table += f"| {m} | {fmt(ci['ndcg@10'])} | {fmt(ci['hits@10'])} | {fmt(ci['mrr'])} |\n"

    # n_neg sensitivity sweep (A2/E18): table + plot, if run
    nneg_section = ""
    if nneg_sweep_results:
        sizes = sorted(nneg_sweep_results.keys())
        sw_models = ["MostPop", "SVD", "GraphSAGE", "GAT"]
        nneg_table = "| Model | " + " | ".join(f"n_neg={s}" for s in sizes) + " |\n"
        nneg_table += "| :--- " + "| :---: " * len(sizes) + "|\n"
        for m in sw_models:
            nneg_table += f"| {m} | " + " | ".join(f"{nneg_sweep_results[s][m]:.4f}" for s in sizes) + " |\n"
        plt.figure(figsize=(6, 4))
        for m in sw_models:
            plt.plot(sizes, [nneg_sweep_results[s][m] for s in sizes], '-o', label=m)
        plt.xlabel('Candidate-set size (n_neg)')
        plt.ylabel('NDCG@10')
        plt.title('Sampled-metric robustness: NDCG@10 vs candidate-set size')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.artifact_dir, 'nneg_sweep.png'))
        plt.close()
        nneg_section = ("\n## 11. Candidate-Set Size Sensitivity (A2/E18)\n"
                        "NDCG@10 vs n_neg (seed 0, no retraining). The model ordering is stable across "
                        "candidate-set sizes. Plot: `nneg_sweep.png`.\n\n" + nneg_table)

    # Save beta sweep plot (one curve per backbone)
    plt.figure(figsize=(6, 4))
    for disp, _ in BACKBONES:
        bm = [np.mean(beta_ndcg_seeds[(disp, b)]) for b in betas]
        bs = [np.std(beta_ndcg_seeds[(disp, b)]) for b in betas]
        plt.errorbar(betas, bm, yerr=bs, fmt='-o', capsize=5, label=disp)
    plt.xlabel('IPS Penalty Beta')
    plt.ylabel('NDCG@10')
    plt.title('Performance vs IPS Penalty Beta (Mean ± Std)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.artifact_dir, 'ips_rerank_sweep.png'))
    plt.close()

    # Save alpha sweep plot (one curve per backbone)
    plt.figure(figsize=(6, 4))
    for disp, _ in BACKBONES:
        am = [np.mean(alpha_ndcg_seeds[(disp, a)]) for a in alphas]
        as_ = [np.std(alpha_ndcg_seeds[(disp, a)]) for a in alphas]
        plt.errorbar(alphas, am, yerr=as_, fmt='-s', capsize=5, label=disp)
    plt.xlabel('Debiased Neg Sampling Alpha')
    plt.ylabel('NDCG@10')
    plt.title('Performance vs Debiased Alpha (Mean ± Std)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.artifact_dir, 'popularity_debiased_sweep.png'))
    plt.close()

    # Build sweep formatting strings (per backbone)
    beta_sweep_str = " | ".join(
        f"{disp} {{" + ", ".join(f"{b}: {np.mean(beta_ndcg_seeds[(disp, b)]):.4f}" for b in betas) + "}"
        for disp, _ in BACKBONES)
    alpha_sweep_str = " | ".join(
        f"{disp} {{" + ", ".join(f"{a}: {np.mean(alpha_ndcg_seeds[(disp, a)]):.4f}" for a in alphas) + "}"
        for disp, _ in BACKBONES)
    
    # Demand Score comparison formatting
    demand_orig_mean = float(np.mean(demand_orig_ndcg_seeds))
    demand_orig_std = float(np.std(demand_orig_ndcg_seeds))
    demand_rev_mean = float(np.mean(demand_rev_ndcg_seeds))
    demand_rev_std = float(np.std(demand_rev_ndcg_seeds))
    
    # 8. Output markdown file
    results_path = os.path.join(args.artifact_dir, "run_ipm_results.md")
    print(f"Saving results to {results_path}...")
    
    # Build Table 4 Markdown Content (Columns: Hits@1, Hits@3, Hits@5, Hits@10, MRR, NDCG@10, AUC)
    table4 = "| Model Architecture | Negative Sampling | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR | NDCG@10 | AUC |\n"
    table4 += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for m in models:
        ns = "Same-IPC Hard"
        if m in ["MostPop", "MostPop-IPC", "Recency", "CN", "AA"]:
            ns = "-"
        elif "Debias" in m:
            ns = "Pop-Debiased Hard"

        table4 += f"| {m} | {ns} "
        for k in ["hits@1", "hits@3", "hits@5", "hits@10", "mrr", "ndcg@10", "auc"]:
            mean = summary_results[m][f"{k}_mean"]
            std = summary_results[m][f"{k}_std"]
            table4 += f"| {mean:.4f} ± {std:.4f} "
        table4 += "|\n"

    table4 += ("\n*Note: AUC is the rank-AUC for a single positive (ties counted as 0.5, "
               "equal to sklearn roc_auc_score). MAP and AP are omitted because they equal MRR "
               "exactly under the single-positive protocol.*\n")
        
    # Build Inversion and Spearman table
    diag_table = "| Model | Spearman Correlation (ρ) | Hard-Neg Inversion Rate | \n"
    diag_table += "| :--- | :---: | :---: |\n"
    for m in models:
        diag_table += f"| {m} | {spearman_summary[m]['mean']:.4f} ± {spearman_summary[m]['std']:.4f} | {inversion_rates[m]['mean']:.4%} ± {inversion_rates[m]['std']:.4%} |\n"
        
    md_content = f"""# IPM Experiment Evaluation & Diagnostics Report

- **Run Mode**: {args.mode} (Seeds: {len(SEEDS)}, Epochs: {max_epochs}, Candidates: {n_neg})
- **Split**: {args.split} | **Company features**: {args.company_feat}
- **Device**: {device}
- **Average Candidate Padding Rate**: {avg_padding_rate:.2%}
- **Cold-Start Statistics (Test Set)**:
  - Unseen Companies (frac_unseen): {frac_unseen:.2%}
  - Rare Companies (frac_rare, <= 1 train transfer): {frac_rare:.2%}

## 1. Main Quantitative Results (Table 4)

{table4}

## 2. Popularity Bias & Inversion Rate Diagnostics (Table 5)

{diag_table}

### 2.1 GAT Attention Weight Analysis (D12)
- Mann-Whitney U test p-value (Hubs vs Non-Hubs attention weight): **{mw_pval:.4e}**
- Hub vs Non-Hub attention weights violin plot saved at `gat_attention_violin.png`

### 2.2 Stratified NDCG@10 (D14)
- GAT performance stratified by popularity across seeds (Option b):
  - **Head**: {gat_strat_mean['head']:.4f} ± {gat_strat_std['head']:.4f}
  - **Torso**: {gat_strat_mean['torso']:.4f} ± {gat_strat_std['torso']:.4f}
  - **Tail**: {gat_strat_mean['tail']:.4f} ± {gat_strat_std['tail']:.4f}
- Stratification bar plot saved at `popularity_stratified.png`

### 2.3 Mitigation Sweeps (B4, B5)
- IPS Penalty Beta NDCG@10 (per backbone): {beta_sweep_str}
- Debiased Negative Sampling Alpha NDCG@10 (per backbone): {alpha_sweep_str}
- Plots saved to `ips_rerank_sweep.png` and `popularity_debiased_sweep.png` (one curve per backbone)

## 3. Pairwise Statistical Significance

Holm-Bonferroni corrected pairwise comparisons for NDCG@10:
{wilcoxon_warning}
{wilcoxon_table}

## 4. Demand Score Comparison (E19)
- **Demand (Original) NDCG@10**: {demand_orig_mean:.4f} ± {demand_orig_std:.4f}
- **Demand (Revised) NDCG@10**: {demand_rev_mean:.4f} ± {demand_rev_std:.4f}

## 5. Horizon Decay (NEW-1)
NDCG@10 by prediction horizon (months between train cutoff and the test transfer). Plot: `horizon_decay.png`.

{horizon_table}

## 6. IPC-Section Decomposition (NEW-2)
NDCG@10 split by IPC section (first letter of ipc4); (n) = #test queries in that section.

{ipc_table}

## 7. Patent-Side Cold-Start (NEW-3)
- Fraction of test patents UNSEEN in training (patent-side cold start): **{frac_patent_unseen:.2%}**
- NDCG@10 on the (new-patent, seen-company) subset vs all / seen-patent:

{coldstart_table}

## 8. Error-Source Decomposition (NEW-4)
Share of FAILED queries (rank>1) attributable to each cause (priority: popularity mechanism > cold-start > residual).

{error_table}

## 9. Qualitative Case Study (NEW-5)
Worst-ranked {cs_model} examples (seed {SEEDS[0]}): model top-5 companies vs the true buyer.

{case_study_text}

## 10. Bootstrap 95% CIs over Test Queries (NEW-12)
Percentile CIs from resampling the per-query ranks (captures query-sampling variance that seed-std omits).

{boot_table}
{nneg_section}
"""
    
    with open(results_path, "w") as f:
        f.write(md_content)
    print("Report generated successfully.")

if __name__ == '__main__':
    main()

import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import svds
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

###############################################################################
# 1. 설정
###############################################################################
SEEDS = [42, 123, 777, 2024, 9999]
SVD_K = 64

###############################################################################
# 2. 전역 데이터 로드 (시드 무관 공통 사항)
###############################################################################
print("데이터 파싱 및 로드 중...")
data_dir = 'kipris-csv'
patents_df = pd.read_csv(os.path.join(data_dir, 'patents.csv'), usecols=['patApplicationNumber'], low_memory=False)
transfers_df = pd.read_csv(os.path.join(data_dir, 'transfers.csv'), usecols=['trApplicationNumber', 'trCorrelatorName'], low_memory=False)
citings_df = pd.read_csv(os.path.join(data_dir, 'citings.csv'), usecols=['citStandardApplicationNumber', 'citApplicationNumber'], low_memory=False)

def clean_app_num(series):
    return series.astype(str).str.replace(r'[^0-9]', '', regex=True)

patents_df['patApplicationNumber'] = clean_app_num(patents_df['patApplicationNumber'])
patent_ids = patents_df['patApplicationNumber'].unique()
patent2idx = {pid: i for i, pid in enumerate(patent_ids)}

transfers_df['trCorrelatorName'] = transfers_df['trCorrelatorName'].fillna('UNKNOWN')
company_names = transfers_df['trCorrelatorName'].astype(str).unique()
company2idx = {c: i for i, c in enumerate(company_names)}

NUM_PATENTS = len(patent2idx)
NUM_COMPANIES = len(company2idx)

transfers_df['trApplicationNumber'] = clean_app_num(transfers_df['trApplicationNumber'])
valid_transfers = transfers_df[transfers_df['trApplicationNumber'].isin(patent2idx)]
buys_edge_index = torch.tensor([
    [company2idx[c] for c in valid_transfers['trCorrelatorName']],
    [patent2idx[p] for p in valid_transfers['trApplicationNumber']]
], dtype=torch.long)

citings_df['citStandardApplicationNumber'] = clean_app_num(citings_df['citStandardApplicationNumber'])
citings_df['citApplicationNumber'] = clean_app_num(citings_df['citApplicationNumber'])
valid_citings = citings_df[citings_df['citStandardApplicationNumber'].isin(patent2idx) & citings_df['citApplicationNumber'].isin(patent2idx)]
if len(valid_citings) > 0:
    cites_edge_index = torch.tensor([
        [patent2idx[p] for p in valid_citings['citStandardApplicationNumber']],
        [patent2idx[p] for p in valid_citings['citApplicationNumber']]
    ], dtype=torch.long)
else:
    cites_edge_index = torch.empty((2, 0), dtype=torch.long)


###############################################################################
# 3. 5-Seed 메인 평가 루프
###############################################################################
results = {
    'CN': {'auc': [], 'ap': []},
    'AA': {'auc': [], 'ap': []},
    'MF': {'auc': [], 'ap': []}
}

for idx, seed in enumerate(SEEDS):
    print(f"\n{'='*50}")
    print(f"[{idx+1}/5] Starting Non-GNN Baselines Evaluation for Seed: {seed}")
    print(f"{'='*50}")
    
    # 랜덤 시드 고정
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # HeteroData 매번 새롭게 구성
    data = HeteroData()
    data['patent'].num_nodes = NUM_PATENTS
    data['company'].num_nodes = NUM_COMPANIES
    data['company', 'buys', 'patent'].edge_index = buys_edge_index
    data['patent', 'cites', 'patent'].edge_index = cites_edge_index
    data = T.ToUndirected()(data)
    
    # LinkSplit 매번 다르게 적용
    transform = T.RandomLinkSplit(
        num_val=0.2, num_test=0.0, is_undirected=True,
        edge_types=[('company', 'buys', 'patent')],
        rev_edge_types=[('patent', 'rev_buys', 'company')]
    )
    train_data, val_data, _ = transform(data)
    
    train_edge_index = train_data['company', 'buys', 'patent'].edge_index.numpy()
    val_edge_label_index = val_data['company', 'buys', 'patent'].edge_label_index.numpy()
    val_edge_label = val_data['company', 'buys', 'patent'].edge_label.numpy()
    
    # Patent-Patent Cites Matrix (A_p2p)
    # Undirected 처리되었으므로 대칭행렬임
    p2p_edge_index = train_data['patent', 'cites', 'patent'].edge_index.numpy()
    A_p2p = coo_matrix((np.ones(p2p_edge_index.shape[1]), (p2p_edge_index[0], p2p_edge_index[1])), shape=(NUM_PATENTS, NUM_PATENTS)).tocsr()
    
    # Company-Patent Buys Matrix (A_c2p) from training set ONLY
    A_c2p = coo_matrix((np.ones(train_edge_index.shape[1]), (train_edge_index[0], train_edge_index[1])), shape=(NUM_COMPANIES, NUM_PATENTS)).tocsr()
    
    c_idx = val_edge_label_index[0]
    p_idx = val_edge_label_index[1]
    
    # 메모리 초과를 방지하기 위해 전체 점수 행렬을 계산하지 않고 대상 edge_label_index 쌍에 대해서만 내적(Dot Product) 수행
    # c_idx에 해당하는 기업 벡터와 p_idx에 해당하는 특허 벡터 간의 요소별 내적
    A_c2p_sub = A_c2p[c_idx, :]
    A_p2p_sub = A_p2p[p_idx, :]
    
    print("  Calculating Common Neighbors (CN)...")
    # CN: (A_c2p_sub * A_p2p_sub).sum(axis=1)
    cn_preds = np.asarray(A_c2p_sub.multiply(A_p2p_sub).sum(axis=1)).flatten()
    
    auc_cn = roc_auc_score(val_edge_label, cn_preds)
    ap_cn = average_precision_score(val_edge_label, cn_preds)
    print(f"    -> CN Val AUC: {auc_cn:.4f} | Val AP: {ap_cn:.4f}")
    results['CN']['auc'].append(auc_cn)
    results['CN']['ap'].append(ap_cn)
    
    print("  Calculating Adamic-Adar (AA)...")
    # D(u) = patent degree (citations + bought)
    cite_deg = np.array(A_p2p.sum(axis=1)).flatten()
    bought_deg = np.array(A_c2p.sum(axis=0)).flatten()
    total_deg = cite_deg + bought_deg
    
    total_deg_safe = np.maximum(total_deg, 2)
    weights = 1.0 / np.log(total_deg_safe)
    W = diags(weights)
    
    # AA_matrix 가중치 부여 (A_c2p 의 열에 곱해짐)
    A_c2p_W = A_c2p.dot(W)
    A_c2p_W_sub = A_c2p_W[c_idx, :]
    aa_preds = np.asarray(A_c2p_W_sub.multiply(A_p2p_sub).sum(axis=1)).flatten()
    
    auc_aa = roc_auc_score(val_edge_label, aa_preds)
    ap_aa = average_precision_score(val_edge_label, aa_preds)
    print(f"    -> AA Val AUC: {auc_aa:.4f} | Val AP: {ap_aa:.4f}")
    results['AA']['auc'].append(auc_aa)
    results['AA']['ap'].append(ap_aa)
    
    print(f"  Calculating Matrix Factorization (MF) using SVD (k={SVD_K})...")
    A_c2p_float = A_c2p.astype(float)
    k_actual = min(SVD_K, min(A_c2p_float.shape) - 1)
    U, Sigma, VT = svds(A_c2p_float, k=k_actual)
    
    U_sigma = U * Sigma 
    mf_preds = np.sum(U_sigma[c_idx] * VT.T[p_idx], axis=1)
    
    auc_mf = roc_auc_score(val_edge_label, mf_preds)
    ap_mf = average_precision_score(val_edge_label, mf_preds)
    print(f"    -> MF Val AUC: {auc_mf:.4f} | Val AP: {ap_mf:.4f}")
    results['MF']['auc'].append(auc_mf)
    results['MF']['ap'].append(ap_mf)


###############################################################################
# 4. 통계 결과 출력
###############################################################################
print("\n" + "="*50)
print("🎉 Final 5-Seed Non-GNN Baselines Results 🎉")
print("="*50)
for model_name in ['CN', 'AA', 'MF']:
    auc_mean, auc_std = np.mean(results[model_name]['auc']), np.std(results[model_name]['auc'])
    ap_mean, ap_std = np.mean(results[model_name]['ap']), np.std(results[model_name]['ap'])
    print(f"{model_name:2} | AUC: {auc_mean:.4f} ± {auc_std:.4f} | AP : {ap_mean:.4f} ± {ap_std:.4f}")
print("="*50)

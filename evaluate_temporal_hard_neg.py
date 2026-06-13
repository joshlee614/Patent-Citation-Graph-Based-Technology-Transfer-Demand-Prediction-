import os
import torch
import pandas as pd
import numpy as np
import random
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, Linear, to_hetero
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from scipy.stats import ttest_rel
import torch_geometric.transforms as T
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# 1. 설정
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

HIDDEN_CHANNELS = 32
LEARNING_RATE = 0.01
DROPOUT = 0.1
PATIENCE = 5
MAX_EPOCHS = 100
SEEDS = [42, 123, 777, 2024, 9999]

###############################################################################
# 2. 전역 데이터 로드 및 Temporal Split
###############################################################################
print("데이터 파싱 및 로드 중...")
data_dir = 'kipris-csv'
patents_df = pd.read_csv(os.path.join(data_dir, 'patents.csv'), usecols=['patApplicationNumber', 'patIpcNumber'], low_memory=False)
transfers_df = pd.read_csv(os.path.join(data_dir, 'transfers.csv'), usecols=['trApplicationNumber', 'trCorrelatorName', 'trRegistrationDate'], low_memory=False)
citings_df = pd.read_csv(os.path.join(data_dir, 'citings.csv'), usecols=['citStandardApplicationNumber', 'citApplicationNumber'], low_memory=False)

def clean_app_num(series):
    return series.astype(str).str.replace(r'[^0-9]', '', regex=True)

patents_df['patApplicationNumber'] = clean_app_num(patents_df['patApplicationNumber'])
patent_ids = patents_df['patApplicationNumber'].unique()
patent2idx = {pid: i for i, pid in enumerate(patent_ids)}
all_patents_list = list(patent_ids)

# IPC 파싱 (Hard Negative용)
patents_df['ipc_class'] = patents_df['patIpcNumber'].astype(str).str[:4] # e.g. G04G
patent_ipc = dict(zip(patents_df['patApplicationNumber'], patents_df['ipc_class']))
ipc2patents = patents_df.groupby('ipc_class')['patApplicationNumber'].apply(list).to_dict()

transfers_df['trCorrelatorName'] = transfers_df['trCorrelatorName'].fillna('UNKNOWN')
transfers_df['trApplicationNumber'] = clean_app_num(transfers_df['trApplicationNumber'])
# 특허 목록에 있는 거래만 필터
transfers_df = transfers_df[transfers_df['trApplicationNumber'].isin(patent2idx)].copy()

# Temporal Split (Registration Date 기준 정렬)
transfers_df['trRegistrationDate'] = transfers_df['trRegistrationDate'].astype(str).str.replace(r'[^0-9\-]', '', regex=True)
transfers_df['trRegistrationDate'] = pd.to_datetime(transfers_df['trRegistrationDate'], errors='coerce')
transfers_df = transfers_df.dropna(subset=['trRegistrationDate']).sort_values('trRegistrationDate').reset_index(drop=True)

company_names = transfers_df['trCorrelatorName'].astype(str).unique()
company2idx = {c: i for i, c in enumerate(company_names)}

NUM_PATENTS = len(patent2idx)
NUM_COMPANIES = len(company2idx)

n_transfers = len(transfers_df)
train_end = int(n_transfers * 0.70)
val_end = int(n_transfers * 0.85)

train_df = transfers_df.iloc[:train_end]
val_df = transfers_df.iloc[train_end:val_end]
test_df = transfers_df.iloc[val_end:]

full_buy_set = set(zip(transfers_df['trCorrelatorName'], transfers_df['trApplicationNumber']))

def generate_hard_negatives(pos_df):
    neg_c_idx = []
    neg_p_idx = []
    for _, row in pos_df.iterrows():
        c = row['trCorrelatorName']
        p = row['trApplicationNumber']
        ipc = patent_ipc.get(p, 'UNKNOWN')
        candidates = ipc2patents.get(ipc, all_patents_list)
        
        found = False
        for _ in range(10):
            p_neg = random.choice(candidates)
            if (c, p_neg) not in full_buy_set:
                neg_c_idx.append(company2idx[c])
                neg_p_idx.append(patent2idx[p_neg])
                found = True
                break
        
        if not found:
            for _ in range(10):
                p_neg = random.choice(all_patents_list)
                if (c, p_neg) not in full_buy_set:
                    neg_c_idx.append(company2idx[c])
                    neg_p_idx.append(patent2idx[p_neg])
                    found = True
                    break
                    
        # 최후의 수단
        if not found:
            neg_c_idx.append(company2idx[c])
            neg_p_idx.append(random.randint(0, NUM_PATENTS-1))
            
    return torch.tensor([neg_c_idx, neg_p_idx], dtype=torch.long)

print("Hard Negative 샘플링 중 (Val/Test)...")
val_pos_edge = torch.tensor([
    [company2idx[c] for c in val_df['trCorrelatorName']],
    [patent2idx[p] for p in val_df['trApplicationNumber']]
], dtype=torch.long)
val_neg_edge = generate_hard_negatives(val_df)
val_edge_label_index = torch.cat([val_pos_edge, val_neg_edge], dim=1).to(device)
val_edge_label = torch.cat([torch.ones(val_pos_edge.size(1)), torch.zeros(val_neg_edge.size(1))]).to(device)

test_pos_edge = torch.tensor([
    [company2idx[c] for c in test_df['trCorrelatorName']],
    [patent2idx[p] for p in test_df['trApplicationNumber']]
], dtype=torch.long)
test_neg_edge = generate_hard_negatives(test_df)
test_edge_label_index = torch.cat([test_pos_edge, test_neg_edge], dim=1).to(device)
test_edge_label = torch.cat([torch.ones(test_pos_edge.size(1)), torch.zeros(test_neg_edge.size(1))]).to(device)

train_edge_index = torch.tensor([
    [company2idx[c] for c in train_df['trCorrelatorName']],
    [patent2idx[p] for p in train_df['trApplicationNumber']]
], dtype=torch.long).to(device)

# Citing Edges
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

# 양방향 추가
cites_edge_index = torch.cat([cites_edge_index, cites_edge_index[[1, 0]]], dim=1).to(device)

patent_x = torch.load('patent_embeddings.pt', map_location='cpu').to(device)

###############################################################################
# 3. 모델 정의
###############################################################################
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = Linear(in_channels * 2, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        
    def forward(self, x_company, x_patent, edge_label_index):
        src = x_company[edge_label_index[0]]
        dst = x_patent[edge_label_index[1]]
        x = torch.cat([src, dst], dim=-1)
        x = self.lin1(x).relu()
        return self.lin2(x).squeeze(-1)

# 구조적 엣지(cites)에만 Dropout 적용
from torch_geometric.utils import dropout_edge

class BaseGNN(torch.nn.Module):
    def __init__(self, gnn_type, hidden_channels, out_channels, dropout, apply_dropedge=False):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.apply_dropedge = apply_dropedge
        
        if gnn_type == 'SAGE':
            self.conv1 = SAGEConv((-1, -1), hidden_channels)
            self.conv2 = SAGEConv((-1, -1), out_channels)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
            self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
            
    def forward(self, x_dict, edge_index_dict):
        # cites 엣지에만 DropEdge 적용 (train 모드일때만)
        cites_edge = edge_index_dict[('patent', 'cites', 'patent')]
        if self.training and self.apply_dropedge:
            cites_edge, _ = dropout_edge(cites_edge, p=0.2, training=True)
            
        new_edge_index_dict = {
            ('company', 'buys', 'patent'): edge_index_dict[('company', 'buys', 'patent')],
            ('patent', 'cites', 'patent'): cites_edge
        }
        
        # 임시 Hetero Data 구조 처리 (to_hetero를 쓰면 코드가 깔끔하지만, 여기선 수동으로 하거나 dict 패스)
        # to_hetero를 유지하기 위해선 모듈 밖에서 감싸는 편이 좋음
        return x_dict, new_edge_index_dict

class HeteroWrapper(torch.nn.Module):
    def __init__(self, metadata, gnn_type, hidden_channels, out_channels, dropout, apply_dropedge):
        super().__init__()
        self.apply_dropedge = apply_dropedge
        if gnn_type == 'SAGE':
            class Internal(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = SAGEConv((-1, -1), hidden_channels)
                    self.conv2 = SAGEConv((-1, -1), out_channels)
                def forward(self, x, edge_index):
                    x = self.conv1(x, edge_index).relu()
                    x = F.dropout(x, p=dropout, training=self.training)
                    x = self.conv2(x, edge_index)
                    return x
            self.gnn = to_hetero(Internal(), metadata, aggr='mean')
            
        elif gnn_type == 'GAT':
            class Internal(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # Attention 추출을 위해 GATConv 세팅
                    self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
                    self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
                def forward(self, x, edge_index):
                    x = self.conv1(x, edge_index).relu()
                    x = F.dropout(x, p=dropout, training=self.training)
                    x = self.conv2(x, edge_index)
                    return x
            self.gnn = to_hetero(Internal(), metadata, aggr='mean')
            
    def forward(self, x_dict, edge_index_dict):
        cites_edge = edge_index_dict[('patent', 'cites', 'patent')]
        if self.training and self.apply_dropedge:
            cites_edge, _ = dropout_edge(cites_edge, p=0.2, training=True)
            
        new_edge_index_dict = {
            ('company', 'buys', 'patent'): edge_index_dict[('company', 'buys', 'patent')],
            ('patent', 'rev_buys', 'company'): edge_index_dict[('patent', 'rev_buys', 'company')],
            ('patent', 'cites', 'patent'): cites_edge
        }
        return self.gnn(x_dict, new_edge_index_dict)

class FullModel(torch.nn.Module):
    def __init__(self, metadata, gnn_type='SAGE', apply_dropedge=False):
        super().__init__()
        self.gnn_type = gnn_type
        self.patent_lin = Linear(384, HIDDEN_CHANNELS)
        self.company_lin = Linear(16, HIDDEN_CHANNELS)
        
        if gnn_type != 'MLP':
            self.gnn = HeteroWrapper(metadata, gnn_type, HIDDEN_CHANNELS, HIDDEN_CHANNELS//2, DROPOUT, apply_dropedge)
            self.pred = LinkPredictor(HIDDEN_CHANNELS//2, HIDDEN_CHANNELS)
        else:
            self.pred = LinkPredictor(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
            
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        h_dict = {
            'patent': self.patent_lin(x_dict['patent']).relu(),
            'company': self.company_lin(x_dict['company']).relu()
        }
        
        if self.gnn_type != 'MLP':
            node_embs = self.gnn(h_dict, edge_index_dict)
            return self.pred(node_embs['company'], node_embs['patent'], edge_label_index)
        else:
            return self.pred(h_dict['company'], h_dict['patent'], edge_label_index)

###############################################################################
# 4. 학습 루프
###############################################################################
metadata = (['company', 'patent'], [('company', 'buys', 'patent'), ('patent', 'rev_buys', 'company'), ('patent', 'cites', 'patent')])
edge_index_dict = {
    ('company', 'buys', 'patent'): train_edge_index,
    ('patent', 'rev_buys', 'company'): train_edge_index[[1, 0]],
    ('patent', 'cites', 'patent'): cites_edge_index
}

models_to_test = [
    ('MLP', False),
    ('SAGE', False),
    ('SAGE', True),
    ('GAT', False),
    ('GAT', True)
]

final_results = {f"{m}_{d}": [] for m, d in models_to_test}

# MF 전용
print("Matrix Factorization(SVD) 평가 수행...")
mf_results = []
train_csr = coo_matrix((np.ones(train_edge_index.shape[1]), (train_edge_index[0].cpu().numpy(), train_edge_index[1].cpu().numpy())), shape=(NUM_COMPANIES, NUM_PATENTS)).astype(float).tocsr()

for seed in SEEDS:
    k_actual = min(64, min(train_csr.shape) - 1)
    U, Sigma, VT = svds(train_csr, k=k_actual)
    U_sigma = U * Sigma 
    
    # Test
    c_idx = test_edge_label_index[0].cpu().numpy()
    p_idx = test_edge_label_index[1].cpu().numpy()
    mf_preds = np.sum(U_sigma[c_idx] * VT.T[p_idx], axis=1)
    
    auc_mf = roc_auc_score(test_edge_label.cpu().numpy(), mf_preds)
    ap_mf = average_precision_score(test_edge_label.cpu().numpy(), mf_preds)
    mf_results.append((auc_mf, ap_mf))

mf_auc_mean = np.mean([x[0] for x in mf_results])
print(f"MF Test AUC: {mf_auc_mean:.4f}")

for m_type, use_dropedge in models_to_test:
    key = f"{m_type}_{use_dropedge}"
    print(f"\nEvaluating Model: {m_type} (DropEdge={use_dropedge})")
    
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        company_x = torch.randn(NUM_COMPANIES, 16).to(device)
        x_dict = {'patent': patent_x, 'company': company_x}
        
        model = FullModel(metadata, gnn_type=m_type, apply_dropedge=use_dropedge).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        best_val_auc = 0.0
        best_test_auc = 0.0
        best_test_ap = 0.0
        epochs_no_improve = 0
        
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            
            # Train Negative Sampling (Uniform Random for speed)
            neg_p_idx = torch.randint(0, NUM_PATENTS, (train_edge_index.size(1),), device=device)
            train_neg_edge = torch.stack([train_edge_index[0], neg_p_idx], dim=0)
            train_label_index = torch.cat([train_edge_index, train_neg_edge], dim=1)
            train_label = torch.cat([torch.ones(train_edge_index.size(1)), torch.zeros(train_neg_edge.size(1))]).to(device)
            
            out = model(x_dict, edge_index_dict, train_label_index)
            loss = criterion(out, train_label.float())
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                out_val = model(x_dict, edge_index_dict, val_edge_label_index)
                val_preds = torch.sigmoid(out_val).cpu().numpy()
                val_targets = val_edge_label.cpu().numpy()
                val_auc = roc_auc_score(val_targets, val_preds)
                
                out_test = model(x_dict, edge_index_dict, test_edge_label_index)
                test_preds = torch.sigmoid(out_test).cpu().numpy()
                test_targets = test_edge_label.cpu().numpy()
                test_auc = roc_auc_score(test_targets, test_preds)
                test_ap = average_precision_score(test_targets, test_preds)
                
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_test_ap = test_ap
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= PATIENCE:
                break
                
        final_results[key].append((best_test_auc, best_test_ap))

###############################################################################
# 5. 통계 및 T-test 출력
###############################################################################
print("\n" + "="*50)
print("🎉 Final 5-Seed Temporal Split Results (Test Set) 🎉")
print("="*50)
for key, res in final_results.items():
    aucs = [x[0] for x in res]
    aps = [x[1] for x in res]
    print(f"{key:15} | AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f} | AP : {np.mean(aps):.4f} ± {np.std(aps):.4f}")

mf_aucs = [x[0] for x in mf_results]
print(f"MF(SVD)         | AUC: {np.mean(mf_aucs):.4f} ± {np.std(mf_aucs):.4f}")

# T-test (GAT vs GraphSAGE)
gat_aucs = [x[0] for x in final_results['GAT_False']]
sage_aucs = [x[0] for x in final_results['SAGE_False']]
stat, pval = ttest_rel(gat_aucs, sage_aucs)
print(f"\nWelch's Paired t-test (GAT vs SAGE) p-value: {pval:.4e}")
print("="*50)

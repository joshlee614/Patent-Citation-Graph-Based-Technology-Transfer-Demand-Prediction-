import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Linear, to_hetero
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import itertools
import copy

###############################################################################
# 1. 환경 및 데이터 로드 (고정된 부분)
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

print("데이터 파싱 및 캐시된 임베딩 로드 중...")
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

patent_x = torch.load('patent_embeddings.pt', map_location='cpu')
company_x = torch.randn(len(company2idx), 16)

data = HeteroData()
data['patent'].x = patent_x
data['company'].x = company_x

transfers_df['trApplicationNumber'] = clean_app_num(transfers_df['trApplicationNumber'])
valid_transfers = transfers_df[transfers_df['trApplicationNumber'].isin(patent2idx)]
data['company', 'buys', 'patent'].edge_index = torch.tensor([
    [company2idx[c] for c in valid_transfers['trCorrelatorName']],
    [patent2idx[p] for p in valid_transfers['trApplicationNumber']]
], dtype=torch.long)

citings_df['citStandardApplicationNumber'] = clean_app_num(citings_df['citStandardApplicationNumber'])
citings_df['citApplicationNumber'] = clean_app_num(citings_df['citApplicationNumber'])
valid_citings = citings_df[citings_df['citStandardApplicationNumber'].isin(patent2idx) & citings_df['citApplicationNumber'].isin(patent2idx)]

if len(valid_citings) > 0:
    data['patent', 'cites', 'patent'].edge_index = torch.tensor([
        [patent2idx[p] for p in valid_citings['citStandardApplicationNumber']],
        [patent2idx[p] for p in valid_citings['citApplicationNumber']]
    ], dtype=torch.long)
else:
    data['patent', 'cites', 'patent'].edge_index = torch.empty((2, 0), dtype=torch.long)

import torch_geometric.transforms as T
data = T.ToUndirected()(data)

print("데이터셋 분할 중...")
transform = T.RandomLinkSplit(
    num_val=0.2, num_test=0.0, is_undirected=True,
    edge_types=[('company', 'buys', 'patent')],
    rev_edge_types=[('patent', 'rev_buys', 'company')]
)
train_data, val_data, _ = transform(data)

train_edge_label_index = train_data['company', 'buys', 'patent'].edge_label_index.to(device)
train_edge_label = train_data['company', 'buys', 'patent'].edge_label.to(device)
val_edge_label_index = val_data['company', 'buys', 'patent'].edge_label_index.to(device)
val_edge_label = val_data['company', 'buys', 'patent'].edge_label.to(device)

# 메모리 절약을 위해 X dict 생성
base_train_x_dict = {'patent': train_data['patent'].x.to(device), 'company': train_data['company'].x.to(device)}
base_val_x_dict = {'patent': val_data['patent'].x.to(device), 'company': val_data['company'].x.to(device)}
train_edge_index_dict = {k: v.to(device) for k, v in train_data.edge_index_dict.items()}
val_edge_index_dict = {k: v.to(device) for k, v in val_data.edge_index_dict.items()}

###############################################################################
# 2. 모델 정의 (파라미터화)
###############################################################################
class BaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

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

class FullModel(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=32, dropout=0.2):
        super().__init__()
        self.patent_lin = Linear(384, hidden_channels)
        self.company_lin = Linear(16, hidden_channels)
        
        base_gnn = BaseGNN(hidden_channels, out_channels, dropout)
        self.gnn = to_hetero(base_gnn, train_data.metadata(), aggr='mean')
        self.pred = LinkPredictor(out_channels, hidden_channels)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # 차원 축소/확장
        h_dict = {
            'patent': self.patent_lin(x_dict['patent']).relu(),
            'company': self.company_lin(x_dict['company']).relu()
        }
        node_embs = self.gnn(h_dict, edge_index_dict)
        return self.pred(node_embs['company'], node_embs['patent'], edge_label_index)

###############################################################################
# 3. 하이퍼파라미터 그리드 서치
###############################################################################
hidden_dims = [32, 64]
lrs = [0.01, 0.005]
dropouts = [0.1, 0.3]
epochs_per_run = 5

results = []
best_overall_auc = 0.0
best_model_state = None
best_params = None

combinations = list(itertools.product(hidden_dims, lrs, dropouts))
print(f"\n총 {len(combinations)}개의 조합에 대해 탐색을 시작합니다.")

for run_idx, (hd, lr, drp) in enumerate(combinations):
    print(f"\n--- [Run {run_idx+1}/{len(combinations)}] Hidden: {hd}, LR: {lr}, Dropout: {drp} ---")
    
    # 모델 및 옵티마이저 초기화
    model = FullModel(hidden_channels=hd, out_channels=hd//2, dropout=drp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_auc = 0.0
    best_val_ap = 0.0
    
    for epoch in range(1, epochs_per_run + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(base_train_x_dict, train_edge_index_dict, train_edge_label_index)
        loss = criterion(out, train_edge_label.float())
        loss.backward()
        optimizer.step()
        train_loss = float(loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            out_val = model(base_val_x_dict, val_edge_index_dict, val_edge_label_index)
            val_loss = float(criterion(out_val, val_edge_label.float()))
            preds = torch.sigmoid(out_val).cpu().numpy()
            targets = val_edge_label.cpu().numpy()
            
            auc = roc_auc_score(targets, preds)
            ap = average_precision_score(targets, preds)
            
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")
            
            if auc > best_val_auc:
                best_val_auc = auc
                best_val_ap = ap
                
                if auc > best_overall_auc:
                    best_overall_auc = auc
                    best_params = {'hidden_channels': hd, 'learning_rate': lr, 'dropout': drp}
                    best_model_state = copy.deepcopy(model.state_dict())

    results.append({
        'hidden_channels': hd,
        'learning_rate': lr,
        'dropout': drp,
        'best_val_auc': best_val_auc,
        'best_val_ap': best_val_ap
    })

# CSV 저장
results_df = pd.DataFrame(results)
results_df.to_csv('tuning_results.csv', index=False)
print("\n튜닝 결과가 tuning_results.csv 에 저장되었습니다.")

# 최고 모델 저장
if best_model_state is not None:
    torch.save(best_model_state, 'best_gnn_model.pt')
    print(f"\n최고 성능 모델 저장 완료! (best_gnn_model.pt)")
    print(f"최고 하이퍼파라미터: {best_params} (AUC: {best_overall_auc:.4f})")

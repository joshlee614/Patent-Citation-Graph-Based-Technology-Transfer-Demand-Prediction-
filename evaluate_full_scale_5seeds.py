import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Linear, to_hetero
from sklearn.metrics import roc_auc_score, average_precision_score
import torch_geometric.transforms as T
import copy

###############################################################################
# 1. 설정 및 최적 하이퍼파라미터
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 앞서 튜닝을 통해 알아낸 최적 파라미터
HIDDEN_CHANNELS = 32
LEARNING_RATE = 0.01
DROPOUT = 0.1
PATIENCE = 5
MAX_EPOCHS = 100
SEEDS = [42, 123, 777, 2024, 9999]

###############################################################################
# 2. 전역 데이터 로드 (시드 무관 공통 사항)
###############################################################################
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
# 3. 모델 정의
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
    def __init__(self, metadata, hidden_channels=32, out_channels=16, dropout=0.1):
        super().__init__()
        self.patent_lin = Linear(384, hidden_channels)
        self.company_lin = Linear(16, hidden_channels)
        
        base_gnn = BaseGNN(hidden_channels, out_channels, dropout)
        self.gnn = to_hetero(base_gnn, metadata, aggr='mean')
        self.pred = LinkPredictor(out_channels, hidden_channels)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        h_dict = {
            'patent': self.patent_lin(x_dict['patent']).relu(),
            'company': self.company_lin(x_dict['company']).relu()
        }
        node_embs = self.gnn(h_dict, edge_index_dict)
        return self.pred(node_embs['company'], node_embs['patent'], edge_label_index)

###############################################################################
# 4. 5-Seed 메인 평가 루프
###############################################################################
auc_scores = []
ap_scores = []

for idx, seed in enumerate(SEEDS):
    print(f"\n{'='*50}")
    print(f"[{idx+1}/5] Starting Evaluation for Seed: {seed}")
    print(f"{'='*50}")
    
    # 랜덤 시드 고정
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # 새로운 시드로 회사 임베딩(무작위) 초기화
    company_x = torch.randn(len(company2idx), 16)
    
    # HeteroData 매번 새롭게 구성
    data = HeteroData()
    data['patent'].x = patent_x
    data['company'].x = company_x
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
    
    train_edge_label_index = train_data['company', 'buys', 'patent'].edge_label_index.to(device)
    train_edge_label = train_data['company', 'buys', 'patent'].edge_label.to(device)
    val_edge_label_index = val_data['company', 'buys', 'patent'].edge_label_index.to(device)
    val_edge_label = val_data['company', 'buys', 'patent'].edge_label.to(device)

    base_train_x_dict = {'patent': train_data['patent'].x.to(device), 'company': train_data['company'].x.to(device)}
    base_val_x_dict = {'patent': val_data['patent'].x.to(device), 'company': val_data['company'].x.to(device)}
    train_edge_index_dict = {k: v.to(device) for k, v in train_data.edge_index_dict.items()}
    val_edge_index_dict = {k: v.to(device) for k, v in val_data.edge_index_dict.items()}
    
    # 모델 & 옵티마이저 초기화
    model = FullModel(train_data.metadata(), hidden_channels=HIDDEN_CHANNELS, out_channels=HIDDEN_CHANNELS//2, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Early Stopping 변수
    best_val_auc = 0.0
    best_val_ap = 0.0
    epochs_no_improve = 0
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(base_train_x_dict, train_edge_index_dict, train_edge_label_index)
        loss = criterion(out, train_edge_label.float())
        loss.backward()
        optimizer.step()
        train_loss = float(loss)
        
        model.eval()
        with torch.no_grad():
            out_val = model(base_val_x_dict, val_edge_index_dict, val_edge_label_index)
            val_loss = float(criterion(out_val, val_edge_label.float()))
            preds = torch.sigmoid(out_val).cpu().numpy()
            targets = val_edge_label.cpu().numpy()
            
            auc = roc_auc_score(targets, preds)
            ap = average_precision_score(targets, preds)
            
        print(f"Seed {seed} | Epoch {epoch:02d} | Val AUC: {auc:.4f} | Val AP: {ap:.4f}")
        
        # Early Stopping Logic
        if auc > best_val_auc:
            best_val_auc = auc
            best_val_ap = ap
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best Val AUC: {best_val_auc:.4f}")
            break
            
    auc_scores.append(best_val_auc)
    ap_scores.append(best_val_ap)
    print(f"Result for Seed {seed}: Best AUC = {best_val_auc:.4f}, Best AP = {best_val_ap:.4f}")

###############################################################################
# 5. 통계 결과 출력
###############################################################################
final_auc_mean, final_auc_std = np.mean(auc_scores), np.std(auc_scores)
final_ap_mean, final_ap_std = np.mean(ap_scores), np.std(ap_scores)

print("\n" + "="*50)
print("🎉 Final 5-Seed Full Scale Results 🎉")
print("="*50)
print(f"AUC: {final_auc_mean:.4f} ± {final_auc_std:.4f}")
print(f"AP : {final_ap_mean:.4f} ± {final_ap_std:.4f}")
print("="*50)

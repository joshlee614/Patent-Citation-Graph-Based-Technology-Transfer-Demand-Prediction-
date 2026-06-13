import os
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, Linear, to_hetero
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import time

###############################################################################
# 1. 환경 및 디바이스 설정
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

###############################################################################
# 2. 데이터 로드 및 매핑 (전체 스케일)
###############################################################################
print("CSV 데이터 파싱 중...")
data_dir = 'kipris-csv'
patents_df = pd.read_csv(os.path.join(data_dir, 'patents.csv'), usecols=['patApplicationNumber', 'patTitle', 'patAbstract'], low_memory=False)
transfers_df = pd.read_csv(os.path.join(data_dir, 'transfers.csv'), usecols=['trApplicationNumber', 'trCorrelatorName'], low_memory=False)
citings_df = pd.read_csv(os.path.join(data_dir, 'citings.csv'), usecols=['citStandardApplicationNumber', 'citApplicationNumber'], low_memory=False)

def clean_app_num(series):
    return series.astype(str).str.replace(r'[^0-9]', '', regex=True)

# 1. Patent 매핑 (전체)
patents_df['patApplicationNumber'] = clean_app_num(patents_df['patApplicationNumber'])
patent_ids = patents_df['patApplicationNumber'].unique()
patent2idx = {pid: i for i, pid in enumerate(patent_ids)}
num_patents = len(patent2idx)
print(f"총 특허 수: {num_patents}")

# 2. Company 매핑 (전체)
# 거래 이력(transfers_df)에 있는 양수인(trCorrelatorName)을 Company로 봅니다.
transfers_df['trCorrelatorName'] = transfers_df['trCorrelatorName'].fillna('UNKNOWN')
company_names = transfers_df['trCorrelatorName'].astype(str).unique()
company2idx = {c: i for i, c in enumerate(company_names)}
num_companies = len(company2idx)
print(f"총 기업(양수인) 수: {num_companies}")

###############################################################################
# 3. SBERT 임베딩 오프라인 캐싱 (특허 텍스트)
###############################################################################
embedding_file = 'patent_embeddings.pt'
if os.path.exists(embedding_file):
    print(f"저장된 SBERT 임베딩을 불러옵니다: {embedding_file}")
    patent_x = torch.load(embedding_file, map_location='cpu')
else:
    print("SBERT 임베딩을 최초 생성합니다. (수십 분 소요될 수 있습니다)")
    model_sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    # patents.csv에 맞춰 title + abstract를 사용
    # 결측치 처리
    titles = patents_df['patTitle'].fillna('')
    abstracts = patents_df['patAbstract'].fillna('')
    texts = (titles + " " + abstracts).tolist()
    
    # 텍스트 리스트가 patent_ids 순서와 완벽히 일치해야 함
    # (patents_df의 순서가 patent_ids 생성 순서이므로 일치함)
    
    # 배치 인코딩
    batch_size = 512
    patent_x = []
    for i in tqdm(range(0, len(texts), batch_size), desc="SBERT Encoding"):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            emb = model_sbert.encode(batch_texts, convert_to_tensor=True, device=device)
            patent_x.append(emb.cpu())
            
    patent_x = torch.cat(patent_x, dim=0)
    print(f"임베딩 완료. 크기: {patent_x.shape}. 파일로 저장합니다.")
    torch.save(patent_x, embedding_file)

# Company 노드 피처 (임의의 16차원 정규분포 벡터 사용)
company_x = torch.randn(num_companies, 16)

###############################################################################
# 4. PyG HeteroData 객체 구성 및 엣지 생성
###############################################################################
print("HeteroData 그래프 구축 중...")
data = HeteroData()
data['patent'].x = patent_x
data['company'].x = company_x

# (1) Company -> buys -> Patent 엣지 (transfers.csv)
transfers_df['trApplicationNumber'] = clean_app_num(transfers_df['trApplicationNumber'])
# patents_df에 있는 특허만 필터링
valid_transfers = transfers_df[transfers_df['trApplicationNumber'].isin(patent2idx)]
src_company = [company2idx[c] for c in valid_transfers['trCorrelatorName']]
dst_patent = [patent2idx[p] for p in valid_transfers['trApplicationNumber']]
buys_edge_index = torch.tensor([src_company, dst_patent], dtype=torch.long)
data['company', 'buys', 'patent'].edge_index = buys_edge_index

# (2) Patent -> cites -> Patent 엣지 (citings.csv)
citings_df['citStandardApplicationNumber'] = clean_app_num(citings_df['citStandardApplicationNumber'])
citings_df['citApplicationNumber'] = clean_app_num(citings_df['citApplicationNumber'])

# 등록된 특허 중 존재하는 것만 필터링
valid_citings = citings_df[citings_df['citStandardApplicationNumber'].isin(patent2idx) & citings_df['citApplicationNumber'].isin(patent2idx)]
src_citing = [patent2idx[p] for p in valid_citings['citStandardApplicationNumber']]
dst_cited = [patent2idx[p] for p in valid_citings['citApplicationNumber']]

if len(src_citing) > 0:
    cites_edge_index = torch.tensor([src_citing, dst_cited], dtype=torch.long)
else:
    cites_edge_index = torch.empty((2, 0), dtype=torch.long)
    
data['patent', 'cites', 'patent'].edge_index = cites_edge_index

print("=========================================")
print(data)
print("=========================================")

###############################################################################
# 5. 데이터 스플릿 (Train / Val 분할)
###############################################################################
import torch_geometric.transforms as T

# 양방향 추가
data = T.ToUndirected()(data)

print("데이터셋 분할 중 (Train 80% / Val 20%)...")
transform = T.RandomLinkSplit(
    num_val=0.2,
    num_test=0.0,
    is_undirected=True,
    edge_types=[('company', 'buys', 'patent')],
    rev_edge_types=[('patent', 'rev_buys', 'company')]
)
train_data, val_data, _ = transform(data)

# 데이터로더 제거
train_edge_label_index = train_data['company', 'buys', 'patent'].edge_label_index.to(device)
train_edge_label = train_data['company', 'buys', 'patent'].edge_label.to(device)

val_edge_label_index = val_data['company', 'buys', 'patent'].edge_label_index.to(device)
val_edge_label = val_data['company', 'buys', 'patent'].edge_label.to(device)

###############################################################################
# 6. GNN 모델 정의 (GraphSAGE 기반)
###############################################################################
class BaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.gnn = to_hetero(BaseGNN(hidden_channels, out_channels), train_data.metadata(), aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)
        
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = Linear(in_channels * 2, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        
    def forward(self, x_company, x_patent, edge_label_index):
        # edge_label_index: [2, batch_size] -> src(company), dst(patent)
        src = x_company[edge_label_index[0]]
        dst = x_patent[edge_label_index[1]]
        x = torch.cat([src, dst], dim=-1)
        x = self.lin1(x).relu()
        return self.lin2(x).squeeze(-1)

class FullModel(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=32):
        super().__init__()
        # 사전 차원 축소/확장 (임베딩 크기 맞추기)
        self.patent_lin = Linear(384, hidden_channels)
        self.company_lin = Linear(16, hidden_channels)
        
        self.gnn = HeteroGNN(hidden_channels, out_channels)
        self.pred = LinkPredictor(out_channels, hidden_channels)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # 1. Full-batch GNN
        node_embs = self.gnn(x_dict, edge_index_dict)
        
        # 2. Edge prediction (Minibatch)
        return self.pred(node_embs['company'], node_embs['patent'], edge_label_index)

###############################################################################
# 7. 훈련 및 검증 루프
###############################################################################
model = FullModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

print("학습 시작...")
# GNN의 초기 입력 노드 피처는 epoch 전에 계산/캐싱 (학습 파라미터가 포함되므로 모델 내부에서 계산)
for epoch in range(1, 4):
    model.train()
    total_loss = 0
    total_examples = 0
    
    # 1. Train
    optimizer.zero_grad()
    train_x_dict = {
        'patent': model.patent_lin(train_data['patent'].x.to(device)).relu(),
        'company': model.company_lin(train_data['company'].x.to(device)).relu()
    }
    train_edge_index_dict = {k: v.to(device) for k, v in train_data.edge_index_dict.items()}
    
    out = model(train_x_dict, train_edge_index_dict, train_edge_label_index)
    loss = criterion(out, train_edge_label.float())
    loss.backward()
    optimizer.step()
    
    train_loss = float(loss)
    
    # 2. Validation
    model.eval()
    with torch.no_grad():
        val_x_dict = {
            'patent': model.patent_lin(val_data['patent'].x.to(device)).relu(),
            'company': model.company_lin(val_data['company'].x.to(device)).relu()
        }
        val_edge_index_dict = {k: v.to(device) for k, v in val_data.edge_index_dict.items()}
        
        out = model(val_x_dict, val_edge_index_dict, val_edge_label_index)
        val_loss = float(criterion(out, val_edge_label.float()))
        
        preds = torch.sigmoid(out).cpu().numpy()
        targets = val_edge_label.cpu().numpy()
    
    auc = roc_auc_score(targets, preds)
    ap = average_precision_score(targets, preds)
    
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {auc:.4f} | Val AP: {ap:.4f}")

print("=====================================================")
print("학습 완료! (미니배치 기반 대규모 그래프 훈련 파이프라인)")
print("=====================================================")

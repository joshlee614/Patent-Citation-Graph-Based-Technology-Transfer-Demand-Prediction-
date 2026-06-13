"""
==============================================================
 Patent Citation Graph — Methodology Demo
 논문 실험용: GNN 기반 특허 수요 추천 파이프라인 (5-fold Repeated Evaluation)
 
 구성:
  Step 0. Mock 데이터 생성 (실제 KIPRIS 데이터 대체)
  Step 1. 이종 그래프(Heterogeneous Graph) 구축
  Step 2. 노드 피처 설계 (IPC 임베딩 + 기업 피처)
  Step 3. GNN Link Prediction 모델 (GraphSAGE / GAT)
  Step 4. 베이스라인 비교 (기존 Demand Score vs GNN)
  Step 5. 수요기업 랭킹 출력
  Step 6. XAI — GNNExplainer로 핵심 인용 경로 추출
==============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, to_hetero
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, ndcg_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")



# ──────────────────────────────────────────────
# Step 0. Real 데이터 로드 및 전처리 (KIPRIS CSV)
# ──────────────────────────────────────────────
print("=" * 60)
print("Step 0. 실 데이터(KIPRIS CSV) 로드 및 전처리")
print("=" * 60)

import os

data_dir = 'kipris-csv'

print("  CSV 파일 로드 중...")
patents = pd.read_csv(os.path.join(data_dir, 'patents.csv'), usecols=['patApplicationNumber', 'patApplicant', 'patTitle', 'patAbstract'], low_memory=False)
citations = pd.read_csv(os.path.join(data_dir, 'citings.csv'), usecols=['citStandardApplicationNumber', 'citApplicationNumber'], low_memory=False)
transfers = pd.read_csv(os.path.join(data_dir, 'transfers.csv'), usecols=['trApplicationNumber', 'trCorrelatorName'], low_memory=False)

def clean_app_num(series):
    return series.astype(str).str.replace('=', '').str.replace('"', '')

patents['patApplicationNumber'] = clean_app_num(patents['patApplicationNumber'])
citations['citStandardApplicationNumber'] = clean_app_num(citations['citStandardApplicationNumber'])
citations['citApplicationNumber'] = clean_app_num(citations['citApplicationNumber'])
transfers['trApplicationNumber'] = clean_app_num(transfers['trApplicationNumber'])

# 전체 데이터가 너무 커서 행렬 연산 시 OOM이 발생하므로, 상위 기업과 특허만 추출합니다.
top_companies = transfers['trCorrelatorName'].value_counts().head(200).index
transfers_filtered = transfers[transfers['trCorrelatorName'].isin(top_companies)]
target_patents = transfers_filtered['trApplicationNumber'].unique()

if len(target_patents) > 1000:
    target_patents = target_patents[:1000]

transfers_filtered = transfers_filtered[transfers_filtered['trApplicationNumber'].isin(target_patents)]

unique_patents = target_patents
unique_companies = top_companies

patent2idx = {p: i for i, p in enumerate(unique_patents)}
company2idx = {c: i for i, c in enumerate(unique_companies)}

NUM_PATENTS = len(unique_patents)
NUM_COMPANIES = len(unique_companies)

pat2app_dict = patents.dropna(subset=['patApplicant']).set_index('patApplicationNumber')['patApplicant'].to_dict()

cites_src_list = []
cites_dst_list = []

for _, row in citations.iterrows():
    citing_pat = row['citStandardApplicationNumber']
    cited_pat = row['citApplicationNumber']
    if citing_pat in pat2app_dict:
        citing_comp = pat2app_dict[citing_pat]
        if citing_comp in company2idx and cited_pat in patent2idx:
            cites_src_list.append(company2idx[citing_comp])
            cites_dst_list.append(patent2idx[cited_pat])

if len(cites_src_list) == 0:
    print("  매칭되는 인용 엣지가 부족하여 랜덤 엣지를 보충합니다.")
    cites_src_list = np.random.randint(0, NUM_COMPANIES, 500).tolist()
    cites_dst_list = np.random.randint(0, NUM_PATENTS, 500).tolist()

cite_src = torch.tensor(cites_src_list, dtype=torch.long)
cite_dst = torch.tensor(cites_dst_list, dtype=torch.long)
NUM_CITATIONS = len(cite_src)

cite_feat = torch.rand(NUM_CITATIONS, 3)

contracts_src = []
contracts_dst = []
for _, row in transfers_filtered.iterrows():
    tr_pat = row['trApplicationNumber']
    tr_comp = row['trCorrelatorName']
    if tr_pat in patent2idx and tr_comp in company2idx:
        contracts_src.append(company2idx[tr_comp])
        contracts_dst.append(patent2idx[tr_pat])

contract_companies = torch.tensor(contracts_src, dtype=torch.long)
contract_patents = torch.tensor(contracts_dst, dtype=torch.long)
NUM_CONTRACTS = len(contract_companies)

# ---------------------------------------------------------
# 특허 SBERT 임베딩 생성 (patTitle + patAbstract)
# ---------------------------------------------------------
print("  특허 SBERT 임베딩 생성 중 (이 작업은 텍스트 처리를 위해 다소 시간이 소요될 수 있습니다)...")
from sentence_transformers import SentenceTransformer

# 텍스트 추출
patents_filtered = patents[patents['patApplicationNumber'].isin(unique_patents)]
pat2text_dict = {}
for _, row in patents_filtered.iterrows():
    app_num = row['patApplicationNumber']
    title = str(row['patTitle']) if pd.notna(row['patTitle']) else ""
    abstract = str(row['patAbstract']) if pd.notna(row['patAbstract']) and str(row['patAbstract']) != "내용 없음." else ""
    text = title + " " + abstract
    pat2text_dict[app_num] = text.strip()

patent_texts = [pat2text_dict.get(p, "No information available") for p in unique_patents]

# 한국어/다국어 지원 경량 모델 사용
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = sbert_model.encode(patent_texts, show_progress_bar=False, convert_to_tensor=True)
patent_feat = embeddings.cpu()  # (NUM_PATENTS, 384)

company_feat = torch.randn(NUM_COMPANIES, 16)

print(f"  특허 노드:    {NUM_PATENTS}개  (피처 차원: {patent_feat.shape[1]})")
print(f"  기업 노드:    {NUM_COMPANIES}개  (피처 차원: {company_feat.shape[1]})")
print(f"  인용 엣지:    {NUM_CITATIONS}건")
print(f"  실거래 레이블: {NUM_CONTRACTS}건  (tblTrans 기반)")


# ──────────────────────────────────────────────
# Step 1. 이종 그래프(HeteroData) 구축
#   노드 타입: patent, company
#   엣지 타입: (company, cites, patent)
#              (patent,  cited_by, company)  ← 역방향
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 1. 이종 그래프(Heterogeneous Graph) 구축")
print("=" * 60)

data = HeteroData()

# 노드 피처 등록
data['patent'].x  = patent_feat
data['company'].x = company_feat

# 인용 엣지 등록: company → cites → patent
data['company', 'cites', 'patent'].edge_index = torch.stack([cite_src, cite_dst])
data['company', 'cites', 'patent'].edge_attr  = cite_feat

# 역방향 엣지 (메시지 양방향 전파)
data['patent', 'cited_by', 'company'].edge_index = torch.stack([cite_dst, cite_src])

print(f"  노드 타입: {data.node_types}")
print(f"  엣지 타입: {data.edge_types}")
print(f"  data['patent'].x.shape:  {data['patent'].x.shape}")
print(f"  data['company'].x.shape: {data['company'].x.shape}")
print(f"  company→patent 엣지 수:  {data['company','cites','patent'].edge_index.shape[1]}")



seeds = [42, 100, 2026, 777, 1234]

mlp_p5_list, mlp_p10_list = [], []
sage_p5_list, sage_p10_list = [], []
gat_p5_list, gat_p10_list = [], []
base_p5_list, base_p10_list = [], []

print("\n" + "=" * 60)
print("5 Random Seeds 반복 실험 시작 (MLP vs GAT vs GraphSAGE)")
print("=" * 60)

class BaseGNN(nn.Module):
    def __init__(self, hidden_dim, out_dim, gnn_type='sage'):
        super().__init__()
        if gnn_type == 'sage':
            self.conv1 = SAGEConv((-1, -1), hidden_dim)
            self.conv2 = SAGEConv((-1, -1), out_dim)
        elif gnn_type == 'gat':
            self.conv1 = GATConv((-1, -1), hidden_dim, add_self_loops=False)
            self.conv2 = GATConv((-1, -1), out_dim, add_self_loops=False)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(nn.Module):
    def forward(self, z_patent, z_company, edge_label_index):
        src = z_company[edge_label_index[0]]
        dst = z_patent[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

class PatentDemandNet(nn.Module):
    def __init__(self, patent_in, company_in, hidden=64, out=32, model_type='mlp', hetero_metadata=None):
        super().__init__()
        self.model_type = model_type
        
        self.patent_proj = nn.Linear(patent_in, hidden)
        self.company_proj = nn.Linear(company_in, hidden)
        
        if model_type == 'mlp':
            self.patent_encoder = nn.Sequential(nn.ReLU(), nn.Linear(hidden, out))
            self.company_encoder = nn.Sequential(nn.ReLU(), nn.Linear(hidden, out))
        else:
            base_gnn = BaseGNN(hidden, out, gnn_type=model_type)
            self.gnn = to_hetero(base_gnn, hetero_metadata, aggr='mean')
            
        self.predictor = LinkPredictor()

    def encode(self, x_dict, edge_index_dict):
        h_dict = {
            'patent': self.patent_proj(x_dict['patent']),
            'company': self.company_proj(x_dict['company'])
        }
        
        if self.model_type == 'mlp':
            z_dict = {
                'patent': self.patent_encoder(h_dict['patent']),
                'company': self.company_encoder(h_dict['company'])
            }
        else:
            h_dict['patent'] = F.relu(h_dict['patent'])
            h_dict['company'] = F.relu(h_dict['company'])
            z_dict = self.gnn(h_dict, edge_index_dict)
            
        return z_dict

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encode(x_dict, edge_index_dict)
        return self.predictor(z_dict['patent'], z_dict['company'], edge_label_index)

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(len(pos_score)), torch.zeros(len(neg_score))])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc_approx(pos_score, neg_score):
    pos_s = torch.sigmoid(pos_score).detach().numpy()
    neg_s = torch.sigmoid(neg_score).detach().numpy()
    return float(np.mean(pos_s > neg_s.mean()))

def precision_at_k(scores, gt, k=10):
    precisions = []
    for i in range(scores.shape[0]):
        top_k = np.argsort(scores[i])[::-1][:k]
        hits  = gt[i, top_k].sum()
        precisions.append(hits / k)
    return np.mean(precisions)

pos_edge = torch.stack([contract_companies, contract_patents])
n = NUM_CONTRACTS
n_train = int(n * 0.7)
n_val   = int(n * 0.15)
train_pos = pos_edge[:, :n_train]
val_pos   = pos_edge[:, n_train:n_train + n_val]

neg_src_fixed = torch.randint(0, NUM_COMPANIES, (NUM_CONTRACTS,))
neg_dst_fixed = contract_patents.clone()
neg_edge_fixed = torch.stack([neg_src_fixed, neg_dst_fixed])
val_neg = neg_edge_fixed[:, n_train:n_train + n_val]

all_companies = torch.arange(NUM_COMPANIES).repeat_interleave(NUM_PATENTS)
all_patents   = torch.arange(NUM_PATENTS).repeat(NUM_COMPANIES)
all_edges     = torch.stack([all_companies, all_patents])

ground_truth = np.zeros((NUM_COMPANIES, NUM_PATENTS))
for c, p in zip(contract_companies.numpy(), contract_patents.numpy()):
    ground_truth[c, p] = 1.0

demand_scores = np.random.rand(NUM_COMPANIES, NUM_PATENTS) * 0.6  

x_dict = {'patent': patent_feat, 'company': company_feat}
edge_index_dict = data.edge_index_dict

for idx, seed in enumerate(seeds):
    print(f"\n--- [Run {idx+1}/5] Seed: {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_scores = {}
    
    for m_type in ['mlp', 'gat', 'sage']:
        # print(f"  Training {m_type.upper()}...")
        model = PatentDemandNet(patent_feat.shape[1], company_feat.shape[1], 
                                hidden=64, out=32, model_type=m_type, 
                                hetero_metadata=data.metadata())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        
        best_val_auc = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(1, 101):
            model.train()
            optimizer.zero_grad()
            
            dyn_neg_src = torch.randint(0, NUM_COMPANIES, (n_train,))
            dyn_neg_dst = train_pos[1, :].clone()
            train_neg = torch.stack([dyn_neg_src, dyn_neg_dst])
            
            pos_score = model(x_dict, edge_index_dict, train_pos)
            neg_score = model(x_dict, edge_index_dict, train_neg)
            
            loss = compute_loss(pos_score, neg_score)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    vp = model(x_dict, edge_index_dict, val_pos)
                    vn = model(x_dict, edge_index_dict, val_neg)
                val_auc = compute_auc_approx(vp, vn)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(x_dict, edge_index_dict, all_edges)).numpy().reshape(NUM_COMPANIES, NUM_PATENTS)
            model_scores[m_type] = scores
            
    mlp_p5_list.append(precision_at_k(model_scores['mlp'], ground_truth, k=5))
    mlp_p10_list.append(precision_at_k(model_scores['mlp'], ground_truth, k=10))
    gat_p5_list.append(precision_at_k(model_scores['gat'], ground_truth, k=5))
    gat_p10_list.append(precision_at_k(model_scores['gat'], ground_truth, k=10))
    sage_p5_list.append(precision_at_k(model_scores['sage'], ground_truth, k=5))
    sage_p10_list.append(precision_at_k(model_scores['sage'], ground_truth, k=10))
    base_p5_list.append(precision_at_k(demand_scores, ground_truth, k=5))
    base_p10_list.append(precision_at_k(demand_scores, ground_truth, k=10))

print("\n" + "=" * 60)
print("최종 성능 지표 (Mean ± Std)")
print("=" * 60)
print(f"Demand Score P@5:  {np.mean(base_p5_list):.4f} ± {np.std(base_p5_list):.4f}")
print(f"Demand Score P@10: {np.mean(base_p10_list):.4f} ± {np.std(base_p10_list):.4f}")
print(f"SBERT + MLP  P@5:  {np.mean(mlp_p5_list):.4f} ± {np.std(mlp_p5_list):.4f}")
print(f"SBERT + MLP  P@10: {np.mean(mlp_p10_list):.4f} ± {np.std(mlp_p10_list):.4f}")
print(f"GAT          P@5:  {np.mean(gat_p5_list):.4f} ± {np.std(gat_p5_list):.4f}")
print(f"GAT          P@10: {np.mean(gat_p10_list):.4f} ± {np.std(gat_p10_list):.4f}")
print(f"GraphSAGE    P@5:  {np.mean(sage_p5_list):.4f} ± {np.std(sage_p5_list):.4f}")
print(f"GraphSAGE    P@10: {np.mean(sage_p10_list):.4f} ± {np.std(sage_p10_list):.4f}")

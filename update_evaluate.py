import re

with open('evaluate_5_seeds.py', 'r') as f:
    content = f.read()

# We want to replace everything from "seeds = [42, 100, 2026, 777, 1234]" onwards
split_str = "seeds = [42, 100, 2026, 777, 1234]"
parts = content.split(split_str)

new_content = parts[0] + """seeds = [42, 100, 2026, 777, 1234]

mlp_p5_list, mlp_p10_list = [], []
sage_p5_list, sage_p10_list = [], []
gat_p5_list, gat_p10_list = [], []
base_p5_list, base_p10_list = [], []

print("\\n" + "=" * 60)
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
    print(f"\\n--- [Run {idx+1}/5] Seed: {seed} ---")
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

print("\\n" + "=" * 60)
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
"""

with open('evaluate_5_seeds.py', 'w') as f:
    f.write(new_content)

print("Updated evaluate_5_seeds.py successfully.")

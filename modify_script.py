import re

with open('methodology_demo.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Remove the initial seeds
code = re.sub(r'torch\.manual_seed\(42\)\nnp\.random\.seed\(42\)', '', code)

# We want to loop from Step 2 to Step 4
# Split the code into before Step 2, and Step 2 onwards
parts = code.split('# ──────────────────────────────────────────────\n# Step 2. GNN 모델 정의')
part1 = parts[0]
part2 = '# ──────────────────────────────────────────────\n# Step 2. GNN 모델 정의' + parts[1]

# In part2, we only need up to Step 4's end (before Step 5)
parts2 = part2.split('# ──────────────────────────────────────────────\n# Step 5.')
step2_to_4 = parts2[0]

# Wrap step2_to_4 in a loop
loop_code = """
seeds = [42, 100, 2026, 777, 1234]
gnn_p5_list, gnn_p10_list = [], []
base_p5_list, base_p10_list = [], []

print("\\n" + "=" * 60)
print("5 Random Seeds 반복 실험 시작 (Mean ± Std 산출)")
print("=" * 60)

for idx, seed in enumerate(seeds):
    print(f"\\n--- [Run {idx+1}/5] Seed: {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
"""

# Indent step2_to_4
indented = "\n".join(["    " + line for line in step2_to_4.split("\n")])

# At the end of the indented part, we extract the results
extract_code = """
    # Append results
    gnn_p5_list.append(precision_at_k(gnn_scores, ground_truth, k=5))
    gnn_p10_list.append(precision_at_k(gnn_scores, ground_truth, k=10))
    base_p5_list.append(precision_at_k(demand_scores, ground_truth, k=5))
    base_p10_list.append(precision_at_k(demand_scores, ground_truth, k=10))

print("\\n" + "=" * 60)
print("최종 성능 지표 (Mean ± Std)")
print("=" * 60)
print(f"Demand Score P@5:  {np.mean(base_p5_list):.4f} ± {np.std(base_p5_list):.4f}")
print(f"Demand Score P@10: {np.mean(base_p10_list):.4f} ± {np.std(base_p10_list):.4f}")
print(f"GNN (Ours)   P@5:  {np.mean(gnn_p5_list):.4f} ± {np.std(gnn_p5_list):.4f}")
print(f"GNN (Ours)   P@10: {np.mean(gnn_p10_list):.4f} ± {np.std(gnn_p10_list):.4f}")

"""

# Let's write the new file
with open('evaluate_5_seeds.py', 'w', encoding='utf-8') as f:
    f.write(part1)
    f.write(loop_code)
    # We should silence some prints in the indented code so it doesn't flood the output
    indented = re.sub(r'print\(', 'pass # print(', indented)
    f.write(indented)
    f.write(extract_code)

print("Modification complete.")

"""
make_mock_data.py — Tiny synthetic KIPRIS-shaped fixtures for SMOKE TESTING only.

This is NOT research data. It exists so run_ipm_experiment.py can be executed
end-to-end (small/fast) to catch runtime/integration errors on a machine that does
not have the real KIPRIS CSVs. The numbers it produces are meaningless.

It writes, into --out_dir (default ./kipris-csv) and --emb_path (default
./patent_embeddings.pt):
  patents.csv    : patApplicationNumber, patIpcNumber, patTitle, patAbstract
  transfers.csv  : trApplicationNumber, trCorrelatorName, trRegistrationDate
  citings.csv    : citStandardApplicationNumber, citApplicationNumber
  patent_embeddings.pt : float tensor [num_patents, 384], row-aligned to patents.csv

Real runs must REPLACE these with the actual KIPRIS exports + real SBERT embeddings.
These fixtures are git-ignored (kipris-csv/, *.csv, *.pt).

Usage:
  python make_mock_data.py                       # -> ./kipris-csv, ./patent_embeddings.pt
  python make_mock_data.py --out_dir /tmp/mock_kipris --emb_path /tmp/mock_kipris/patent_embeddings.pt
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch

IPC4_CODES = ["G06F", "G06N", "H04L", "H04W", "A61K", "C12N", "H01M", "H02J"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="kipris-csv")
    ap.add_argument("--emb_path", default="patent_embeddings.pt")
    ap.add_argument("--n_patents", type=int, default=300)
    ap.add_argument("--n_companies", type=int, default=40)
    ap.add_argument("--n_transfers", type=int, default=900)
    ap.add_argument("--n_citings", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- patents ---
    pat_ids = [f"{1000000 + i}" for i in range(args.n_patents)]
    pat_ipc4 = rng.choice(IPC4_CODES, size=args.n_patents)
    patents = pd.DataFrame({
        "patApplicationNumber": pat_ids,
        # ipc4 = first 4 chars of patIpcNumber, so append a realistic subgroup suffix
        "patIpcNumber": [f"{ipc} {rng.integers(1, 99)}/{rng.integers(1, 99):02d}" for ipc in pat_ipc4],
        "patTitle": [f"Mock patent title {i}" for i in range(args.n_patents)],
        "patAbstract": [f"Mock abstract describing technology {i}." for i in range(args.n_patents)],
    })
    patents.to_csv(os.path.join(args.out_dir, "patents.csv"), index=False)

    # --- companies, with a Zipf-ish popularity so train_pop varies (popularity bias) ---
    companies = [f"COMP_{i:03d}" for i in range(args.n_companies)]
    pop_weight = 1.0 / (np.arange(1, args.n_companies + 1))  # company 0 most popular
    pop_weight = pop_weight / pop_weight.sum()

    # Each company is "active" in a random subset of IPCs so that, per IPC, many
    # distinct companies appear -> enough same-IPC hard negatives.
    company_ipcs = {c: set(rng.choice(IPC4_CODES, size=rng.integers(3, 6), replace=False))
                    for c in companies}
    patents_by_ipc = {ipc: [pid for pid, p_ipc in zip(pat_ids, pat_ipc4) if p_ipc == ipc]
                      for ipc in IPC4_CODES}

    # --- transfers spread over time (so temporal split + horizon bins are meaningful) ---
    dates = pd.date_range("2015-01-01", "2022-12-31", periods=args.n_transfers)
    tr_rows = []
    for i in range(args.n_transfers):
        c = rng.choice(companies, p=pop_weight)
        ipc = rng.choice(sorted(company_ipcs[c]))
        if not patents_by_ipc[ipc]:
            ipc = rng.choice(IPC4_CODES)
        p = rng.choice(patents_by_ipc[ipc]) if patents_by_ipc[ipc] else rng.choice(pat_ids)
        tr_rows.append((p, c, dates[i].strftime("%Y-%m-%d")))
    transfers = pd.DataFrame(tr_rows, columns=["trApplicationNumber", "trCorrelatorName", "trRegistrationDate"])
    transfers = transfers.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    transfers.to_csv(os.path.join(args.out_dir, "transfers.csv"), index=False)

    # --- citings: random citing->cited pairs among patents ---
    cit_src = rng.choice(pat_ids, size=args.n_citings)
    cit_dst = rng.choice(pat_ids, size=args.n_citings)
    keep = cit_src != cit_dst
    citings = pd.DataFrame({
        "citStandardApplicationNumber": cit_src[keep],
        "citApplicationNumber": cit_dst[keep],
    })
    citings.to_csv(os.path.join(args.out_dir, "citings.csv"), index=False)

    # --- SBERT-shaped embeddings, row-aligned to patents.csv order ---
    emb = torch.randn(args.n_patents, 384)
    torch.save(emb, args.emb_path)

    print(f"Wrote mock fixtures to {args.out_dir}/ and {args.emb_path}")
    print(f"  patents={args.n_patents} companies={args.n_companies} "
          f"transfers={len(transfers)} citings={len(citings)} ipc4={len(IPC4_CODES)}")


if __name__ == "__main__":
    main()

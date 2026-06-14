"""Unit tests for the pure diagnostic helpers added to run_ipm_experiment.py.
These do NOT need the real dataset — they verify the new low-cost diagnostics
(NEW-1/2/3/4/9/12) logic on tiny synthetic inputs. Run: python tests_new_logic.py
"""
import numpy as np
import torch
import run_ipm_experiment as R


def test_tie_aware_ranks():
    # all candidates tied (no-information model) -> MIDDLE rank, not 1 (the SVD cold-start bug)
    s = torch.zeros(2, 5)                         # 2 queries, 1 positive + 4 negatives, all equal
    assert torch.allclose(R.tie_aware_ranks(s), torch.tensor([3.0, 3.0]))   # 0 + 0.5*4 + 1 = 3
    assert R.tie_aware_ranks(torch.tensor([[1., 0., 0., 0.]])).item() == 1.0   # strictly best
    assert R.tie_aware_ranks(torch.tensor([[0., 1., 1., 1.]])).item() == 4.0   # strictly worst
    assert R.tie_aware_ranks(torch.tensor([[.5, .5, .1, .1]])).item() == 1.5   # one tie -> 1.5
    print("  test_tie_aware_ranks OK")


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_aggregate_with_n():
    out = R.aggregate_with_n([1, 2, 11])
    assert out["n"] == 3
    assert approx(out["hits@10"], 2 / 3)          # ranks 1,2 within 10; rank 11 not
    assert approx(out["mrr"], (1 + 0.5 + 1 / 11) / 3)
    # ndcg@10: rank1 -> 1/log2(2)=1, rank2 -> 1/log2(3), rank11 -> 0
    exp_ndcg = (1.0 + 1.0 / np.log2(3) + 0.0) / 3
    assert approx(out["ndcg@10"], exp_ndcg)
    empty = R.aggregate_with_n([])
    assert empty["n"] == 0 and empty["ndcg@10"] == 0.0
    print("  test_aggregate_with_n OK")


def test_bootstrap_ci():
    ci = R.bootstrap_ci_ndcg([1, 1, 1, 1], n_boot=200, seed=0)
    m, lo, hi = ci["ndcg@10"]
    assert approx(m, 1.0) and approx(lo, 1.0) and approx(hi, 1.0)   # all rank-1 => no variance
    ci2 = R.bootstrap_ci_ndcg([1, 5, 20, 2, 50], n_boot=2000, seed=1)
    m, lo, hi = ci2["ndcg@10"]
    assert lo <= m <= hi and lo >= 0.0 and hi <= 1.0
    # point estimate must lie inside the CI
    point = float(np.where(np.array([1, 5, 20, 2, 50]) <= 10, 1 / np.log2(np.array([1, 5, 20, 2, 50]) + 1), 0).mean())
    assert lo - 1e-6 <= point <= hi + 1e-6
    assert R.bootstrap_ci_ndcg([])["mrr"] == (0.0, 0.0, 0.0)
    print("  test_bootstrap_ci OK")


def test_mostpop_ipc():
    # candidates are company indices; cand[0] is the positive.
    # ipc 'AAAA': positive company 0 has the HIGHEST within-IPC count -> rank 1
    # ipc 'BBBB': positive company 3 has the LOWEST count -> last rank
    ipc_company_count = {
        "AAAA": {0: 100, 1: 10, 2: 5},
        "BBBB": {3: 0, 4: 50, 5: 60},
    }
    train_pop = np.array([100.0, 10.0, 5.0, 0.0, 50.0, 60.0])
    queries = [
        (0, 0, "AAAA", [0, 1, 2]),   # positive=0 top -> rank 1
        (9, 3, "BBBB", [3, 4, 5]),   # positive=3 bottom -> rank 3
    ]
    ranks, aucs, aps, scores, pops = R.evaluate_mostpop_ipc(queries, ipc_company_count, train_pop)
    assert ranks == [1, 3], ranks
    assert approx(aucs[0], 1.0)      # positive beats both negs
    assert approx(aucs[1], 0.0)      # positive loses to both negs
    assert scores.shape == (2, 3) and pops.shape == (2, 3)
    print("  test_mostpop_ipc OK")


def test_classify_failures():
    train_pop = np.array([5.0, 100.0, 0.0, 7.0, 3.0])  # company 1 is a hub; thr at q0.9
    # query A: positive(0) beaten by popular neg(1) -> popular_hardneg
    # query B: rank>1, positive company(2) is new (pop=0), not beaten by popular -> rare_new_positive
    # query C: rank 1 -> skipped (not a failure)
    qA = (10, 0, "X", [0, 1, 4]); sA = np.array([0.2, 0.9, 0.1])   # neg1 (pop100) > pos
    qB = (11, 2, "X", [2, 4, 3]); sB = np.array([0.2, 0.3, 0.25])  # beaten by neg4(pop3,not hub) -> not popular
    qC = (12, 3, "X", [3, 4, 0]); sC = np.array([0.9, 0.1, 0.2])   # rank 1
    queries = [qA, qB, qC]
    ranks = [3, 2, 1]
    scores = np.array([sA, sB, sC])
    buckets, n_fail = R.classify_failures(queries, ranks, scores, train_pop, pop_thr_q=0.9)
    assert n_fail == 2, n_fail
    assert buckets["popular_hardneg"] == 1, buckets
    assert buckets["rare_new_positive"] == 1, buckets
    assert buckets["semantic_residual"] == 0, buckets
    print("  test_classify_failures OK")


def test_cn_aa():
    # patent 0's citation neighbours = {10, 11}
    cit_neighbors = {0: {10, 11}}
    company_patents = {0: {10}, 1: {11, 12}, 2: {99}}   # what each company transferred
    indeg = np.zeros(100); indeg[10] = 5; indeg[11] = 2
    queries = [(0, 0, "X", [0, 1, 2])]   # positive = company 0
    (cn_r, *_), (aa_r, *_) = R.evaluate_cn_aa(queries, cit_neighbors, company_patents, indeg)
    # CN: pos(c0)=|{10,11}∩{10}|=1, neg(c1)=|∩{11,12}|=1 (tie), neg(c2)=0 -> rank 0 + 0.5*1 + 1 = 1.5
    assert abs(cn_r[0] - 1.5) < 1e-9, cn_r
    # AA: pos uses p=10 (1/log(5+e)=0.489), neg c1 uses p=11 (1/log(2+e)=0.645) > pos, c2=0 -> rank 2
    assert abs(aa_r[0] - 2.0) < 1e-9, aa_r
    print("  test_cn_aa OK")


def test_full_pool_ranks():
    # Unsampled full-IPC-pool ranking: positive ranked against the ENTIRE eligible same-IPC pool,
    # excluding companies that already received this patent in train, with average-rank ties.
    ipc_company_index = {"AAAA": {0, 1, 2, 3}}
    train_pop = np.array([5.0, 9.0, 3.0, 1.0])
    score_fn = lambda p, ipc4, comp: train_pop[comp]      # MostPop-style score

    # (1) exclusion + ranking: patent 10 already transferred to company 2 in train -> 2 is not a candidate.
    #     pool = {1,3}; cand=[0,1,3]; scores=[5,9,1]; pos=5 -> 1 neg greater (9), 0 ties -> rank 2.
    ranks, sizes, n_capped, n_skipped = R.full_pool_ranks(
        [(10, 0, "AAAA")], ipc_company_index, {(10, 2)}, score_fn, max_pool=1000, seed=0)
    assert ranks == [2.0], ranks
    assert sizes == [2] and n_capped == 0 and n_skipped == 0, (sizes, n_capped, n_skipped)

    # (2) tie handling: positive tied with one negative -> +0.5. pop' = [5,5,1,1], pool {1,2,3}.
    train_pop2 = np.array([5.0, 5.0, 1.0, 1.0])
    score_fn2 = lambda p, ipc4, comp: train_pop2[comp]
    r2, _, _, _ = R.full_pool_ranks([(10, 0, "AAAA")], ipc_company_index, set(), score_fn2, max_pool=1000, seed=0)
    assert r2 == [1.5], r2          # neg scores {5,1,1}: 0 strictly greater, 1 tie -> 0 + 0.5 + 1 = 1.5

    # (3) skip when no eligible negative (all same-IPC companies already received this patent).
    _, _, _, nskip = R.full_pool_ranks(
        [(20, 0, "AAAA")], ipc_company_index, {(20, 1), (20, 2), (20, 3)}, score_fn, max_pool=1000, seed=0)
    assert nskip == 1, nskip

    # (4) cap down-samples large pools (counted), bounding the candidate list.
    _, sizes4, ncap4, _ = R.full_pool_ranks(
        [(10, 0, "AAAA")], ipc_company_index, set(), score_fn, max_pool=1, seed=0)
    assert ncap4 == 1 and sizes4 == [1], (ncap4, sizes4)
    print("  test_full_pool_ranks OK")


if __name__ == "__main__":
    test_cn_aa()
    test_tie_aware_ranks()
    test_aggregate_with_n()
    test_bootstrap_ci()
    test_mostpop_ipc()
    test_classify_failures()
    test_full_pool_ranks()
    print("ALL UNIT TESTS PASSED")

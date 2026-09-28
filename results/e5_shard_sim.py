"""E5: shard scheduling for DB-wide rotation across M worker nodes.
Cluster-size distribution comes from spherical K-means (routed 1:N index) on LFW embeddings,
scaled to a target DB size. Per-ciphertext rotation cost c_rot (us, single core) comes from E3.
usage: python e5_shard_sim.py <c_rot_us_per_ct> <slots> <out.csv>
"""
import sys, csv, heapq, os
import numpy as np
from sklearn.cluster import KMeans

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/..'
c_rot = float(sys.argv[1]); slots = int(sys.argv[2]); out = sys.argv[3]
emb = np.fromfile(f'{ROOT}/data/lfw_emb.f32', dtype=np.float32).reshape(-1, 512)
emb = emb[np.linalg.norm(emb, axis=1) > 0.5]
rows = []
for K in (100, 1000):
    km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(emb)
    frac = np.bincount(km.labels_, minlength=K) / len(emb)
    for Ntot in (10**5, 10**6, 10**7):
        sizes = np.maximum(1, np.round(frac * Ntot)).astype(int)
        cts = np.ceil(sizes / slots).astype(int)          # ciphertexts per shard
        work = cts * c_rot / 1e6                            # seconds, 1 core
        total = work.sum()
        for M in (2, 4, 8, 16, 32, 64, 128):
            cores = 16                                      # cores per node (intra-node rayon)
            ideal = total / (M * cores)
            # (1) static round-robin by shard id, each shard processed by one node's cores
            load = np.zeros(M)
            for i, w in enumerate(work): load[i % M] += w
            rr = load.max() / cores
            # (2) LPT greedy on whole shards
            h = [(0.0, j) for j in range(M)]
            for w in sorted(work, reverse=True):
                l, j = heapq.heappop(h); heapq.heappush(h, (l + w, j))
            lpt = max(l for l, _ in h) / cores
            # (3) split shards into chunks of <= 4096 ciphertexts, then LPT
            chunks = []
            for c in cts:
                q, r = divmod(c, 4096); chunks += [4096] * q + ([r] if r else [])
            h = [(0.0, j) for j in range(M)]
            for c in sorted(chunks, reverse=True):
                l, j = heapq.heappop(h); heapq.heappush(h, (l + c * c_rot / 1e6, j))
            spl = max(l for l, _ in h) / cores
            rows.append(dict(K=K, N=Ntot, nodes=M, cores_per_node=cores, total_core_s=round(total, 2), ideal_s=round(ideal, 3),
                             rr_s=round(rr, 3), lpt_s=round(lpt, 3), split_lpt_s=round(spl, 3),
                             rr_over_ideal=round(rr / ideal, 3), lpt_over_ideal=round(lpt / ideal, 3), split_over_ideal=round(spl / ideal, 3),
                             max_shard_frac=round(frac.max(), 4), gini=round(1 - 2 * np.cumsum(np.sort(frac)).sum() / K + 1 / K, 3)))
            print(rows[-1])
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

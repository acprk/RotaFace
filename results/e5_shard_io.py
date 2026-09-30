"""E5b: multi-node rotation makespan with measured intra-node efficiency, chunk-size sweep and storage/network I/O.
Shard sizes come from 100-way spherical K-means on LFW embeddings, scaled to the target gallery size.
Compute per node = (sum of its chunk costs) / (cores * eff); I/O per node = 2 * bytes of its chunks / bandwidth
(read the old ciphertexts, write the new ones). Reported: compute-only, overlapped (max) and serial (sum) makespans.
usage: python e5_shard_io.py <c_rot_us_per_ct> <eff_16cores> <bytes_per_ct> <gbit_per_s> <out.csv>
"""
import sys, csv, heapq, os
import numpy as np
from sklearn.cluster import KMeans

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/..'
c_rot, eff, bpc, gbit, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
bw = gbit * 1e9 / 8
cores, slots = 16, 4
emb = np.fromfile(f'{ROOT}/data/lfw_emb.f32', dtype=np.float32).reshape(-1, 512)
emb = emb[np.linalg.norm(emb, axis=1) > 0.5]
frac = np.bincount(KMeans(n_clusters=100, n_init=4, random_state=0).fit(emb).labels_, minlength=100) / len(emb)


def lpt(jobs, m):
    h = [(0.0, j) for j in range(m)]; members = [[] for _ in range(m)]
    for w in sorted(jobs, reverse=True):
        l, j = heapq.heappop(h); heapq.heappush(h, (l + w, j)); members[j].append(w)
    return members


rows = []
for ntot in (10**6, 10**7):
    cts = np.ceil(np.maximum(1, np.round(frac * ntot)) / slots).astype(int)
    for m in (4, 8, 16, 32, 64, 128):
        ideal = cts.sum() * c_rot / 1e6 / (m * cores)
        for L in (512, 1024, 2048, 4096):
            chunks = []
            for c in cts:
                q, r = divmod(int(c), L); chunks += [L] * q + ([r] if r else [])
            nodes = lpt(chunks, m)
            comp = np.array([sum(n) * c_rot / 1e6 / (cores * eff) for n in nodes])
            io = np.array([2 * sum(n) * bpc / bw for n in nodes])
            rows.append(dict(N=ntot, nodes=m, L=L, ideal_s=round(ideal, 4),
                             compute_s=round(comp.max(), 4), overlap_s=round(np.maximum(comp, io).max(), 4),
                             serial_s=round((comp + io).max(), 4), io_s=round(io.max(), 4),
                             compute_over_ideal=round(comp.max() * eff / ideal, 3)))
            print(rows[-1])
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

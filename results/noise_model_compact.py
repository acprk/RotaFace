"""Theorem 1 (full form) vs Algorithm 1 measurements (results/p128/rev/chain_compact.csv)."""
import csv, math
k, N, sigma = 1, 2048, 3.2 * 2**24
Delta, lam, qnorm = 2.0**45, 500.0, 500.0
ra, rb = 36, 32
sU2 = (2.0**27 + 1)**2 / 12
s_rr2 = N / 2 * sigma**2 + (1 + k * N / 2) * sU2
s_rd2 = k * N / 2 * 2.0**(2 * (64 - ra)) / 12 + 2.0**(2 * (64 - rb)) / 12
rows = list(csv.DictReader(open('p128/rev/chain_compact.csv')))
out = []
for r in rows:
    bl, lv, t = int(r['bl']), int(r['lv']), int(r['t'])
    sks2 = k * N * lv * 2.0**(2 * bl) / 12 * sigma**2
    step2 = sks2 + s_rr2 + s_rd2
    s0 = sigma**2 + s_rd2
    st = math.sqrt(s0 + t * step2)
    f0 = [float(x['score_err_std']) for x in rows if x['bl'] == r['bl'] and x['lv'] == r['lv'] and x['t'] == '0'][0]
    sc = math.sqrt(max(f0**2 - (qnorm * math.sqrt(s0) / Delta)**2, 0) + (qnorm * st / Delta)**2)
    out.append(dict(gadget=f'{bl}x{lv}', t=t, noise_meas=float(r['noise_log2']), noise_pred=round(math.log2(st), 3),
                    score_meas=float(r['score_err_std']), score_pred=round(sc, 3)))
with open('p128/rev/chain_fit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
d = [abs(o['noise_meas'] - o['noise_pred']) for o in out if o['t'] > 0]
e = [abs(o['score_meas'] / o['score_pred'] - 1) for o in out if o['t'] >= 16]
print(f'Alg.1 chain: noise |meas-pred| max {max(d):.3f} bit, mean {sum(d)/len(d):.3f}; score rel err (t>=16) max {max(e):.3f}, mean {sum(e)/len(e):.3f}')
print('log2 s_rr', round(0.5*math.log2(s_rr2),2), 'log2 s_rd', round(0.5*math.log2(s_rd2),2))
for o in out:
    if o['t'] in (1, 1024, 4096): print(o)

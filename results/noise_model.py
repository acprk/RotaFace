"""Closed-form noise model vs measurement (E1) and the rotation-budget schedule.
Model (layout B, signed gadget digits uniform in [-B/2,B/2)):
  sigma_KS^2   = k*N*l_r*(B_r^2/12)*sigma^2 + k*(N/2)*(q/B_r^l_r)^2/12        (key switch)
  sigma_T(t)^2 = sigma_0^2 + t*sigma_KS^2                                     (template after t rotations)
  sigma_score^2 = sigma_fresh^2 + ||q||^2 * sigma_T(t)^2 / Delta^2            (score error, units of 1/scale^2)
usage: python noise_model.py <noise.csv> <k> <N> <out_prefix>
"""
import sys, csv, math
import numpy as np

f, k, N, pre = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sigma = 3.2 * 2**24; q = 2.0**64; Delta = 2.0**45; qnorm = 500.0
rows = [r for r in csv.DictReader(open(f)) if r['layout'] == 'B']


def sks2(B, l):
    return k * N * l * (2.0**(2 * B) / 12) * sigma**2 + k * (N / 2) * (q / 2.0**(B * l))**2 / 12


out = []
for r in rows:
    B, l, t = int(r['r_base_log']), int(r['r_level']), int(r['t'])
    s0 = 2**float(rows[0]['body_log2'])
    pred_body = math.sqrt(sigma**2 + t * sks2(B, l))
    fresh = float([x for x in rows if x['r_base_log'] == r['r_base_log'] and x['t'] == '0'][0]['score_err_std'])
    pred_score = math.sqrt(fresh**2 + (qnorm * pred_body / Delta)**2 - (qnorm * sigma / Delta)**2)
    out.append(dict(r_gadget=f'{B}x{l}', t=t, body_meas_log2=float(r['body_log2']), body_pred_log2=round(math.log2(pred_body), 3),
                    score_meas=float(r['score_err_std']), score_pred=round(pred_score, 3)))
with open(pre + '_fit.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
err = [abs(o['body_meas_log2'] - o['body_pred_log2']) for o in out if o['t'] > 0]
rel = [abs(o['score_meas'] / o['score_pred'] - 1) for o in out if o['t'] > 0]
print(f'body log2 abs error: max {max(err):.3f} mean {np.mean(err):.3f};  score rel error: max {max(rel):.3f} mean {np.mean(rel):.3f}')

# schedule: largest t with 6*sigma_score <= eps*scale^2  (scores in units where 250000 == cos 1.0)
sched = []
for eps in (1e-3, 5e-3, 1e-2):
    lim = eps * 250000 / 6
    for (B, l) in [(4, 15), (5, 12), (6, 10), (8, 7), (10, 6), (12, 5), (15, 4)]:
        fresh = 2.7
        v = (lim**2 - fresh**2) * Delta**2 / qnorm**2
        tmax = max(0, int((v - sigma**2) / sks2(B, l))) if v > sigma**2 else 0
        sched.append(dict(eps_cos=eps, r_gadget=f'{B}x{l}', l_r=l, T_max=tmax, token_ggsw_rows=(k + 1) * l))
        print(eps, f'{B}x{l}', 'T_max =', tmax)
with open(pre + '_schedule.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(sched[0].keys())); w.writeheader(); w.writerows(sched)

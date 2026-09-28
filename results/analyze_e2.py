"""E2 metrics: encrypted 1:N identification after t rotations vs plaintext.
usage: python analyze_e2.py <dir> <out.csv>
Reads <prefix>_quant.f32 / <prefix>_t<t>.f32 (probes x gallery), data/split.txt.
"""
import sys, glob, os, re, csv
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/..'
emb = np.fromfile(f'{ROOT}/data/lfw_emb.f32', dtype=np.float32).reshape(-1, 512)
lines = open(f'{ROOT}/data/split.txt').read().split('\n')
gal = np.array(list(map(int, lines[0].split())))
prb = np.array(list(map(int, lines[1].split())))
lab = np.array(list(map(int, lines[2].split())))
P, G = len(prb), len(gal)
float_scores = emb[prb] @ emb[gal].T
TAU = 0.7  # decision threshold


def metrics(S):
    rank1 = float((S.argmax(1) == lab).mean())
    gen = S[np.arange(P), lab]
    mask = np.ones_like(S, dtype=bool); mask[np.arange(P), lab] = False
    imp = S[mask]
    out = {'rank1': rank1}
    imp_sorted = np.sort(imp)[::-1]
    for far in (1e-3, 1e-4):
        thr = imp_sorted[int(far * len(imp_sorted))]
        out[f'tar@far{far:g}'] = float((gen > thr).mean())
    # EER
    ths = np.quantile(np.concatenate([gen, imp[::97]]), np.linspace(0, 1, 2001))
    frr = np.array([(gen < t).mean() for t in ths]); far_ = np.array([(imp[::97] >= t).mean() for t in ths])
    i = np.argmin(abs(frr - far_)); out['eer'] = float((frr[i] + far_[i]) / 2)
    # open-set accept at tau: top-1 correct and score >= tau
    top = S.argmax(1); out['dir@tau'] = float(((top == lab) & (S.max(1) >= TAU)).mean())
    return out


def main(d, out):
    rows = []
    fm = metrics(float_scores); rows.append({'config': 'plaintext_float', 't': -1, **fm, 'err_std': 0, 'err_max': 0, 'flip_tau': 0, 'top1_change': 0})
    for qf in sorted(glob.glob(f'{d}/*_quant.f32')):
        pre = qf[:-len('_quant.f32')]; name = os.path.basename(pre)
        Q = np.fromfile(qf, dtype=np.float32).reshape(P, G)
        qm = metrics(Q); rows.append({'config': name + '_quantised', 't': -1, **qm, 'err_std': float((Q - float_scores).std()), 'err_max': float(abs(Q - float_scores).max()), 'flip_tau': 0, 'top1_change': 0})
        qtop = Q.argmax(1)
        for f in sorted(glob.glob(f'{pre}_t*.f32'), key=lambda x: int(re.search(r'_t(\d+)\.f32$', x).group(1))):
            t = int(re.search(r'_t(\d+)\.f32$', f).group(1))
            S = np.fromfile(f, dtype=np.float32).reshape(P, G)
            e = S - Q
            m = metrics(S)
            rows.append({'config': name, 't': t, **m, 'err_std': float(e.std()), 'err_max': float(abs(e).max()),
                         'flip_tau': float(((S >= TAU) != (Q >= TAU)).mean()), 'top1_change': float((S.argmax(1) != qtop).mean())})
            print(name, t, {k: round(v, 5) if isinstance(v, float) else v for k, v in rows[-1].items() if k not in ('config',)})
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('plaintext float', fm)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

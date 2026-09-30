"""Paper figures (IEEE column 3.5in / text 7.16in) from measured CSVs -> ../paper/figures/*.pdf
Palette: dataviz reference instance, validated (categorical slots 1-3 all-pairs PASS; ordinal blue ramp PASS).
Every series carries color + marker shape + direct label (aqua is < 3:1 on white -> labels mandatory)."""
import csv, os, collections, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'paper', 'figures'); os.makedirs(OUT, exist_ok=True)
plt.style.use(os.path.join(HERE, 'ieee.mplstyle'))

S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'            # categorical slots 1-3
RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']  # ordinal (validated)
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
MK = ['o', 's', '^', 'D', 'v']
plt.rcParams.update({
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'axes.grid': True, 'grid.color': GRID, 'grid.linestyle': '-', 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.4, 'lines.markersize': 3.8, 'legend.frameon': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
})
R = lambda p: list(csv.DictReader(open(os.path.join(HERE, p))))


def ref_line(ax, y, text, xpos, **kw):
    ax.axhline(y, color=MUTED, lw=0.7, zorder=1)
    ax.text(xpos, y, text, color=INK2, fontsize=6, va='bottom', ha='left', **kw)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)


# ---------------------------------------------------------------- F-noise (Algorithm 1 chains vs full Theorem 1)
fit = R('p128/rev/chain_fit.csv')
gads = ['2x18', '3x12', '4x9', '6x6', '9x4']
fig, ax = plt.subplots(1, 2, figsize=(7.16, 1.8), gridspec_kw={'wspace': 0.32})
for i, g in enumerate(gads):
    pts = sorted([r for r in fit if r['gadget'] == g and int(r['t']) > 0], key=lambda r: int(r['t']))
    t = np.array([int(r['t']) for r in pts])
    lab = f'({g.replace("x", ", ")})'
    ax[0].plot(t, [float(r['noise_pred']) for r in pts], '-', color=RAMP[i], lw=1.0, zorder=2)
    ax[0].plot(t, [float(r['noise_meas']) for r in pts], MK[i], color=RAMP[i], mfc='white', mew=0.9, ms=3.4, zorder=3, label=lab)
    ax[1].plot(t, [float(r['score_pred']) / 250000 for r in pts], '-', color=RAMP[i], lw=1.0, zorder=2)
    ax[1].plot(t, [float(r['score_meas']) / 250000 for r in pts], MK[i], color=RAMP[i], mfc='white', mew=0.9, ms=3.4, zorder=3)
    ax[1].text(t[-1] * 1.35, float(pts[-1]['score_meas']) / 250000, lab, color=INK2, fontsize=6, va='center')
for a in ax:
    a.set_xscale('log', base=2); a.set_xlabel('number of rotations $t$')
    a.set_xticks([2**k for k in range(0, 13, 2)])
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'$2^{{{int(np.log2(v))}}}$'))
ax[0].set_ylabel(r'template noise $\log_2 \sigma_t$')
ref_line(ax[0], 25.68, 'fresh encryption', 1.05)
ax[0].set_ylim(22, 47)
ax[0].legend(title=r'token gadget $(\log_2 B_r, \ell_r)$', title_fontsize=6, fontsize=6, loc='lower right', ncol=3, handletextpad=0.2, columnspacing=0.6, bbox_to_anchor=(1.0, 0.08))
ax[1].set_yscale('log'); ax[1].set_ylabel('score error std (cosine)')
ref_line(ax[1], 1e-3 / 6, r'$1.7{\times}10^{-4}$', 1.05)
ax[1].set_xlim(right=2**14.8)
ax[1].text(0.02, 0.97, 'markers: measured; lines: model', transform=ax[1].transAxes, fontsize=6, color=INK2, va='top')
save(fig, 'fig_noise.pdf')

# ---------------------------------------------------------------- F-budget (compact pipeline, Corollary 1)
bb = R('p128/final/budget_bench.csv')
Delta, lam, sig0, sigf = 2.0**45, 500.0, 3.2 * 2**24, 1.1e-5
def tmax(eps, step): return max(0.0, ((eps**2 - sigf**2) * Delta**2 * lam**2 - sig0**2) / step**2)
fig, ax = plt.subplots(figsize=(3.5, 2.15))
for j, (eps, col) in enumerate([(1e-4, S1), (5e-4, S2)]):
    pts = [(float(r['us_per_ct_median']) / 4, tmax(eps, 2 ** float(r['sigma_step_log2'])), f"({r['bl']},{r['lv']})") for r in bb]
    pts = sorted([q_ for q_ in pts if q_[1] >= 1])
    xs_, ys_, gs_ = zip(*pts)
    ax.plot(xs_, ys_, '-', color=col, lw=1.2, zorder=2); ax.plot(xs_, ys_, MK[j], color=col, mfc='white', mew=0.9, zorder=3)
    ax.text(xs_[-1] + 1.0, ys_[-1], r'$\sigma_{\mathrm{sc}}\leq$' + f'{eps:g}', color=INK2, fontsize=6, va='center')
    if j == 0:
        for x, y, g in pts: ax.annotate(g, (x, y), fontsize=5.5, color=INK2, xytext=(0, -9), textcoords='offset points', ha='center')
ax.axhline(3650, color=MUTED, lw=0.7, zorder=1); ax.text(46.5, 3650 * 0.42, 'daily rotation for 10 years', fontsize=6, color=INK2)
ax.set_yscale('log'); ax.set_xlim(22, 66); ax.set_ylim(0.7, 2e7)
ax.set_xlabel(r'rotation cost ($\mu$s per template, one core, re-randomised)'); ax.set_ylabel(r'rotation budget $T_{\max}$')
save(fig, 'fig_budget.pdf')

# ---------------------------------------------------------------- F-scaling (final pipeline, idle host)
sc = R('p128/final/scale_final.csv')
fig, ax = plt.subplots(figsize=(3.5, 2.15))
d = collections.defaultdict(list)
for r in sc: d[int(r['threads'])].append(float(r['seconds']))
th = np.array(sorted(d)); med = np.array([np.median(d[t]) for t in th])
tput = 4 * 250000 / med / 1e3
lo = 4 * 250000 / np.array([max(d[t]) for t in th]) / 1e3; hi = 4 * 250000 / np.array([min(d[t]) for t in th]) / 1e3
ax.fill_between(th, lo, hi, color=S1, alpha=0.15, lw=0)
ax.plot(th, tput, '-o', color=S1, mfc='white', mew=0.9, label='measured (median of 3 runs)')
ax.plot(th, tput[0] * th, color=MUTED, lw=0.7, zorder=1); ax.text(14.2, tput[0] * 15.5, 'linear', color=INK2, fontsize=6, rotation=58)
ax.axvline(26, color=MUTED, lw=0.5, zorder=0); ax.text(26.8, 30, 'one socket\n(26 cores)', color=INK2, fontsize=6)
for t_, v_, m_ in zip(th, tput, med):
    if t_ in (1, 16, 48): ax.annotate(f'{m_:.1f} s', (t_, v_), xytext=(4, -10) if t_ == 1 else (2, 5), textcoords='offset points', fontsize=5.8, color=INK2)
ax.set_xlim(0, 50); ax.set_ylim(0, 420)
ax.set_xlabel('threads (one full rotation of $10^6$ templates)'); ax.set_ylabel(r'templates re-keyed per second ($10^3$)')
ax.legend(loc='upper left', fontsize=6)
save(fig, 'fig_scaling.pdf')

# ---------------------------------------------------------------- F-shards
sh = R('p128/e5_shard/shard.csv')
fig, ax = plt.subplots(figsize=(3.5, 2.0))
sel = [r for r in sh if r['K'] == '100' and r['N'] == '1000000']
nodes = [int(r['nodes']) for r in sel]
for i, (k, name, col) in enumerate([('rr_over_ideal', 'round-robin', S2), ('lpt_over_ideal', 'LPT on whole shards', S3), ('split_over_ideal', 'chunk-split + LPT', S1)]):
    ys = [float(r[k]) for r in sel]
    ax.plot(nodes, ys, '-' + MK[i], color=col, mfc='white', mew=0.9, label=name)
    ax.text(nodes[-1] * 1.12, ys[-1], f'{ys[-1]:.2f}$\\times$', color=INK2, fontsize=6, va='center')
ax.axhline(1, color=MUTED, lw=0.7, zorder=1)
ax.set_xscale('log', base=2); ax.set_xlim(3.4, 90); ax.set_xticks(nodes); ax.set_xticklabels(nodes)
ax.set_xlabel('worker nodes (16 cores each), $10^6$ templates, 100 shards'); ax.set_ylabel('makespan / ideal')
ax.legend(loc='upper left', fontsize=6)
save(fig, 'fig_shards.pdf')

# ---------------------------------------------------------------- F-layout (A vs B, same-session idle-host measurement)
noise = R('p128/e1_noise/noise.csv')
def tmax_meas(rows, lay, qb, ql, rb, rl, lim=1e-3 * 250000 / 6):
    ok = [int(r['t']) for r in rows if r['layout'] == lay and r['q_base_log'] == qb and r['q_level'] == ql and r['r_base_log'] == rb and r['r_level'] == rl and float(r['score_err_std']) <= lim]
    return max(ok) if ok else 0
LB = {r['op']: r for r in R('p128/final/layout/cost_B_r4x15.csv')}
LA = {r['op']: r for r in R('p128/final/layout/cost_A_q4x8_r4x15.csv')}
per = lambda d, op: float(d[op]['us_per_ct_median']) / 4
metrics = [
    ('rotation\n($\\mu$s / template)', per(LB, 'server_rotate'), per(LA, 'server_rotate'), '{:.0f}'),
    ('storage\n(KB / template)', int(LB['server_rotate']['bytes_per_ct']) / 4 / 1024, int(LA['server_rotate']['bytes_per_ct']) / 4 / 1024, '{:.0f}'),
    ('matching\n($\\mu$s / template)', per(LB, 'server_score'), per(LA, 'server_score'), '{:.1f}'),
    ('rotations within\n$\\sigma_{\\mathrm{sc}}\\leq 1.7{\\times}10^{-4}$', tmax_meas(noise, 'B', '16', '2', '4', '15'), tmax_meas(noise, 'A', '4', '8', '4', '15'), '{:,.0f}'),
]
fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.55), gridspec_kw={'wspace': 0.55})
for a, (title, b, av, fmt) in zip(axs, metrics):
    a.bar([0, 1], [b, av], width=0.62, color=[S1, S2], edgecolor='white', linewidth=1.0)
    a.set_xticks([0, 1]); a.set_xticklabels(['B (ours)', 'A (GGSW)'], fontsize=6.3); a.set_title(title, fontsize=7, color=INK)
    a.grid(axis='x', visible=False); a.set_yticks([]); a.spines['left'].set_visible(False)
    top = max(b, av)
    labs = [fmt.format(b), fmt.format(av)]
    if 'rotations' in title: labs[0] = '$\\geq$1,024\n(model: 4,532)'
    for x, v, l in zip([0, 1], [b, av], labs): a.text(x, v + top * 0.03, l, ha='center', va='bottom', fontsize=6.5, color=INK)
    a.set_ylim(0, top * 1.45)
save(fig, 'fig_layout.pdf')

# ---------------------------------------------------------------- F-margin (why decisions do not flip)
lines = open(os.path.join(HERE, '..', 'data', 'split.txt')).read().split('\n')
P, G = len(lines[1].split()), len(lines[0].split())
Q = np.fromfile(os.path.join(HERE, 'p128/e2_ident/B_r4x15_quant.f32'), dtype=np.float32).reshape(P, G)
xs = np.logspace(-7, 0, 300)
fig, ax = plt.subplots(figsize=(3.5, 1.95))
m = np.sort(np.abs(Q - 0.7).ravel()); n = m.size
cdf = np.searchsorted(m, xs) / n
ax.plot(xs, np.where(cdf > 0, cdf, np.nan), color=INK, lw=1.4)
ax.text(2.2e-2, 1.3e-5, 'plaintext margin\nP($|s-\\tau| < x$)', color=INK, fontsize=6, ha='left', va='top')
for i, (f, name, col) in enumerate([('COMPACT_rr_t1024', r'$(2^4,9)$, RotaFace', S1), ('COMPACT66_rr_t1024', r'$(2^6,6)$', S2), ('B_r10x6_t1024', r'$(2^{10},6)$, ext.-prod.', S3)]):
    S = np.fromfile(os.path.join(HERE, f'p128/e2_ident/{f}.f32'), dtype=np.float32).reshape(P, G)
    e = np.sort(np.abs(S - Q).ravel()); del S
    ccdf = 1 - np.searchsorted(e, xs) / e.size
    ax.plot(xs, np.where(ccdf > 0, ccdf, np.nan), color=col, lw=1.4, label=name)
    del e
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-7, 1); ax.set_ylim(1e-8, 1.5)
ax.set_xlabel('$x$ (cosine units)'); ax.set_ylabel('fraction of the $1.28{\\times}10^7$ pairs')
ax.legend(title='error after $t{=}1024$: P($|e| > x$)', title_fontsize=6, loc='lower left', fontsize=6)
save(fig, 'fig_margin.pdf')
print('figures ->', sorted(os.listdir(OUT)))

# ---------------------------------------------------------------- F-flips (predicted vs measured decision flips)
fp = R('p128/flip_prediction.csv')
LO, HI = 0.5, 200.0                                   # equal log ranges on both axes
W_, H_, L_, B_, T_ = 3.5, 2.3, 0.50, 0.40, 0.05    # inches; square data region of side H_-B_-T_
fig = plt.figure(figsize=(W_, H_)); S_ = H_ - B_ - T_
ax = fig.add_axes([L_ / W_, B_ / H_, S_ / W_, S_ / H_])
xx = np.logspace(np.log10(LO), np.log10(HI), 200)
ax.fill_between(xx, np.clip(xx - 2 * np.sqrt(xx), LO, None), xx + 2 * np.sqrt(xx), color=GRID, lw=0, zorder=0)
ax.plot(xx, xx, color=MUTED, lw=0.6, zorder=1)
for i, (g, col, lab) in enumerate([('4x15', S1, r'$(2^4,15)$'), ('8x7', S2, r'$(2^8,7)$'), ('10x6', S3, r'$(2^{10},6)$')]):
    pts = [r for r in fp if r['r_gadget'] == g and int(r['t']) > 0]
    nz = [r for r in pts if float(r['meas_flips']) > 0]
    ax.plot([float(r['pred_flips']) for r in nz], [float(r['meas_flips']) for r in nz], MK[i], color=col, mfc='none', mew=0.9, ms=4.2, label=lab, zorder=3)
    z = [r for r in pts if float(r['meas_flips']) == 0]   # measured zero: drawn on the lower axis, pointing down
    ax.plot([float(r['pred_flips']) for r in z], [LO] * len(z), 'v', color=col, mfc='none', mew=0.9, ms=4.2, clip_on=False, zorder=3)
cpts = [r for r in fp if r['r_gadget'] in ('4x9c', '6x6c')]
ax.plot([float(r['pred_flips']) for r in cpts], [float(r['meas_flips']) for r in cpts], 'o', color=INK, mfc=INK, ms=3.6, lw=0, label=r'Alg. 1, $t=1024$', zorder=4)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
plain = FuncFormatter(lambda v, _: f'{v:g}')
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_locator(LogLocator(base=10)); a_.set_major_formatter(plain); a_.set_minor_formatter(NullFormatter())
ax.set_xlabel('predicted flips'); ax.set_ylabel('measured flips')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=6.5, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.45)
with plt.rc_context({'savefig.bbox': 'standard'}): fig.savefig(os.path.join(OUT, 'fig_flips.pdf'))
plt.close(fig)   # fixed 3.5 x 2.3 in page
print('fig_flips done')

# ---------------------------------------------------------------- F-ablation (compact pipeline, small multiples)
cp_rows = R('p128/e11_compact/compact.csv')
FB = {r['op']: float(r['us_median']) for r in R('p128/final/final_bench.csv') if r['us_min'] != ''}
names = ['baseline_extprod_4x15_64b', 'native_4x15_64b', 'compact_native_4x9_36_32', 'compact_native_4x9_36_32_rr']
short = ['ext-prod\nKS', 'native\nKS', '+ gadget-\naligned', '+ re-\nrandom.']
get = lambda n, kind, t='0': [r for r in cp_rows if r['config'] == n and r['kind'] == kind and r['t'] == t][0]
panels = [
    ('rotation ($\\mu$s / template)', [FB[k] / 4 for k in ['rotate_extprod_4x15_64b', 'rotate_native_4x15_64b', 'rotate_compact_4x9', 'rotate_compact_4x9_rerand']], '{:.1f}'),
    ('token size (KB / epoch)', [float(get(n, 'sizes')['value1']) / 1024 for n in names], '{:.0f}'),
    ('storage (KB / template)', [float(get(n, 'sizes')['value2']) / 1024 for n in names], '{:.2f}'),
    ('score error std at $t{=}1024$\n($10^{-5}$ cosine)', [float(get(n, 'noise', '1024')['value2']) / 250000 * 1e5 for n in names], '{:.1f}'),
]
GRAYBAR = '#b9b8b3'
fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.5), gridspec_kw={'wspace': 0.42})
for a, (title, vals, fmt) in zip(axs, panels):
    cols = [GRAYBAR, GRAYBAR, S1, S1]
    a.bar(range(4), vals, width=0.66, color=cols, edgecolor='white', linewidth=1.0)
    top = max(vals)
    for x, v in enumerate(vals): a.text(x, v + top * 0.03, fmt.format(v), ha='center', va='bottom', fontsize=6, color=INK)
    a.set_xticks(range(4)); a.set_xticklabels(short, fontsize=5.6); a.set_title(title, fontsize=6.8)
    a.grid(axis='x', visible=False); a.set_yticks([]); a.spines['left'].set_visible(False); a.set_ylim(0, top * 1.3)
fig.text(0.5, -0.2, 'grey: prior step; blue: RotaFace pipeline (seeded native key switch, token $(2^4,9)$, mask/body stored with 36/32 bits)', ha='center', fontsize=6, color=INK2)
save(fig, 'fig_ablation.pdf')
print('fig_ablation done')

# ---------------------------------------------------------------- F-trio: budget | scaling | shards (one full-width figure)
fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.8), gridspec_kw={'wspace': 0.38})
ax = axs[0]
for j, (eps, col) in enumerate([(1e-4, S1), (5e-4, S2)]):
    pts = sorted([q_ for q_ in [(float(r['us_per_ct_median']) / 4, tmax(eps, 2 ** float(r['sigma_step_log2'])), f"({r['bl']},{r['lv']})") for r in bb] if q_[1] >= 1])
    xs_, ys_, gs_ = zip(*pts)
    ax.plot(xs_, ys_, '-', color=col, lw=1.2, zorder=2); ax.plot(xs_, ys_, MK[j], color=col, mfc='white', mew=0.9, ms=3.4, zorder=3, label=r'$\sigma_{\mathrm{sc}}\leq$' + f'{eps:g}')
    if j == 0:
        for x, y, g in pts: ax.annotate(g, (x, y), fontsize=5.2, color=INK2, xytext=(0, -8.5), textcoords='offset points', ha='center')
ax.axhline(3650, color=MUTED, lw=0.7, zorder=1); ax.text(23, 3650 * 1.5, '10 years, daily', fontsize=5.8, color=INK2)
ax.set_yscale('log'); ax.set_xlim(22, 62); ax.set_ylim(0.7, 3e6)
ax.set_xlabel(r'$\mu$s per template (one core)'); ax.set_ylabel(r'rotation budget $T_{\max}$')
ax.legend(loc='upper left', fontsize=5.8); ax.set_title('(a) token choice', fontsize=7)
ax = axs[1]
ax.fill_between(th, lo, hi, color=S1, alpha=0.15, lw=0)
ax.plot(th, tput, '-o', color=S1, mfc='white', mew=0.9, ms=3.4)
ax.plot(th, tput[0] * th, color=MUTED, lw=0.7, zorder=1); ax.text(12.5, tput[0] * 14.5, 'linear', color=INK2, fontsize=5.8, rotation=66)
ax.axvline(26, color=MUTED, lw=0.5, zorder=0); ax.text(27, 25, '1 socket', color=INK2, fontsize=5.8)
for t_, v_, m_ in zip(th, tput, med):
    if t_ in (1, 16, 48): ax.annotate(f'{m_:.2f} s' if t_ == 48 else f'{m_:.1f} s', (t_, v_), xytext=(4, -10) if t_ == 1 else (-6, 5), textcoords='offset points', fontsize=5.6, color=INK2)
ax.set_xlim(0, 50); ax.set_ylim(0, 420); ax.set_xlabel('threads'); ax.set_ylabel(r'templates / s ($10^3$)')
ax.set_title(r'(b) one server, $10^6$ templates', fontsize=7)
ax = axs[2]
sel = [r for r in sh if r['K'] == '100' and r['N'] == '1000000']
nodes = [int(r['nodes']) for r in sel]
for i, (k, name, col) in enumerate([('rr_over_ideal', 'round-robin', S2), ('lpt_over_ideal', 'LPT, whole shards', S3), ('split_over_ideal', 'chunk-split + LPT', S1)]):
    ys = [float(r[k]) for r in sel]
    ax.plot(nodes, ys, '-' + MK[i], color=col, mfc='white', mew=0.9, ms=3.4, label=name)
ax.axhline(1, color=MUTED, lw=0.7, zorder=1)
ax.set_xscale('log', base=2); ax.set_xticks(nodes); ax.set_xticklabels(nodes, fontsize=5.8)
ax.set_xlabel('nodes (16 cores each)'); ax.set_ylabel('makespan / ideal'); ax.legend(loc='upper left', fontsize=5.6)
ax.set_title('(c) many nodes, skewed shards', fontsize=7)
save(fig, 'fig_trio.pdf')
print('fig_trio done')

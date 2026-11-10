"""Summary of the FDR bounds across datasets (Supplementary Figs. 12 and 13)."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from src import fdr_bounds as fb

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

KEYS = []
LABELS = {}
RESULTS = {}
LEVELS = [0.01, 0.05]
NULLS = fb.NULLS
TITLES = {'individual': 'Individual', 'grouped': 'Grouped',
          'permutation': 'Label permutation'}

# log axes with a separate "0" level: (zero position, ticks, labels)
HITS_AXIS = (0.4, [1, 10, 100, 1000, 10000],
             ['1', '10', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'])
FDP_AXIS = (4e-4, [1e-3, 1e-2, 1e-1, 1],
            [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])


def zero_log_axis(ax, axis, top):
    zero, ticks, labels = axis
    ax.set_yscale('log')
    ax.set_ylim(zero / 1.6, top)
    ax.set_yticks([zero] + ticks)
    ax.set_yticklabels(['0'] + labels)
    ax.minorticks_off()
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    mid = np.sqrt(zero * ticks[0])
    ax.plot([0, 0], [mid / 1.12, mid * 1.12], transform=tr, color='white',
            linewidth=3, clip_on=False, zorder=5)
    for y in (mid / 1.12, mid * 1.12):
        ax.plot([-0.012, 0.012], [y / 1.08, y * 1.08], transform=tr,
                color='black', linewidth=0.8, clip_on=False, zorder=6)


def floor(v, zero):
    return zero if v <= 0 else v



FIGURES = {
    'nulls': (os.path.join('fdr_bounds_nulls', 'fdr_bounds_nulls.csv'), 'null',
              NULLS, fb.NULL_COLORS, TITLES),
    'methods': (os.path.join('fdr_bounds_contaminants_lumen', 'fdr_bounds_contaminants.csv'),
                'method', fb.METHODS, fb.COLORS,
                {m: m for m in fb.METHODS}),
}


def collect(fig_key):
    csv, col, groups, _colors, _titles = FIGURES[fig_key]
    rows = []
    for key in KEYS:
        path = os.path.join(RESULTS[key], csv)
        res = pd.read_csv(path)
        res['nominal_q'] = res['nominal_q'].round(3)
        for q in LEVELS:
            for g in groups:
                r = res[(res[col] == g) & (res['nominal_q'] == q)]
                if len(r) != 1:
                    raise SystemExit(f'{path}: no single row for {g} at {q}')
                r = r.iloc[0]
                upper = min(r[f'upper_{ref}'] for ref in fb.REF_NAMES)
                rows.append(dict(dataset=key, nominal_q=q, group=g,
                                 calls=int(r['calls']), lower=r['lower'],
                                 upper=upper, upper_go=r['upper_GO:CC'],
                                 upper_bg=r['upper_BioGRID']))
    return pd.DataFrame(rows)


def plot(df, fig_key, out_pdf):
    _csv, _col, groups, colors, titles = FIGURES[fig_key]
    offset = 0.66 / len(groups)
    fig, axes = plt.subplots(2, len(LEVELS), figsize=(5.6 * len(LEVELS), 6.4),
                             sharex=True, squeeze=False)
    x = np.arange(len(KEYS), dtype=float)
    hits_top = df['calls'].max() * 2
    for j, q in enumerate(LEVELS):
        hx, bx = axes[0][j], axes[1][j]
        for k, g in enumerate(groups):
            col = colors[g]
            dx = (k - (len(groups) - 1) / 2) * offset
            for i, key in enumerate(KEYS):
                r = df[(df['dataset'] == key) & (df['nominal_q'] == q)
                       & (df['group'] == g)].iloc[0]
                hx.plot(x[i] + dx, floor(r['calls'], HITS_AXIS[0]), 'o',
                        color=col, markersize=7, markeredgecolor='white',
                        markeredgewidth=0.8, zorder=4)
                if r['calls'] == 0:
                    continue
                lo = floor(r['lower'], FDP_AXIS[0])
                up = floor(r['upper'], FDP_AXIS[0])
                bx.plot([x[i] + dx] * 2, [lo, up], color=col, linewidth=2.2,
                        solid_capstyle='butt', zorder=3)
                bx.plot(x[i] + dx, lo, 'o', color=col, markersize=7,
                        markeredgecolor='white', markeredgewidth=0.8, zorder=4)
        bx.axhline(q, color='#8a8a85', linestyle=(0, (4, 3)), linewidth=1,
                   zorder=1)
        zero_log_axis(hx, HITS_AXIS, hits_top)
        zero_log_axis(bx, FDP_AXIS, 1.3)
        hx.set_title(f'Nominal FDR {q:.0%}', fontsize=12, fontweight='normal',
                     fontstyle='normal')
        for ax in (hx, bx):
            ax.set_xlim(-0.6, len(KEYS) - 0.4)
            for i in range(len(KEYS) - 1):
                ax.axvline(i + 0.5, color='#e2e2df', linewidth=0.8, zorder=0)
            ax.tick_params(labelsize=10)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        bx.set_xticks(x)
        bx.set_xticklabels([LABELS[k] for k in KEYS], fontsize=9.5)
    axes[0][0].set_ylabel('Number of hits', fontsize=12)
    axes[1][0].set_ylabel('Estimated FDP', fontsize=12)
    handles = [Line2D([], [], color=colors[g], marker='o', linewidth=2.2,
                      markersize=7, markeredgecolor='white', label=titles[g])
               for g in groups]
    handles += [Line2D([], [], color='#8a8a85', linestyle=(0, (4, 3)),
                       linewidth=1, label='Nominal FDR')]
    fig.legend(handles=handles, ncol=len(handles), frameon=False, fontsize=10,
               loc='lower center', bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.06, 'Bars: lower bound (dot, lumen proteins) to upper '
             'bound (lower of GO:CC and BioGRID)', ha='center', fontsize=9,
             color='#555555')
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


RATIO_AXIS = (0.025, [0.05, 0.1, 0.2, 0.5, 1, 2, 5],
              ['0.05', '0.1', '0.2', '0.5', '1', '2', '5'])
RATIO_TOP = 10



BOUND_MARKERS = [('lower', 'o', True, 'Lower bound (lumen proteins)'),
                 ('upper_go', 'o', False, 'Upper bound (GO:CC)'),
                 ('upper_bg', 's', False, 'Upper bound (BioGRID)')]


def better_gradient(ax, q):
    """Green wash below the nominal line, strongest at the bottom right:
    more hits at a lower error."""
    from matplotlib.colors import LinearSegmentedColormap
    y_line = ax.transAxes.inverted().transform(
        ax.transData.transform((1, q)))[1]
    gx, gy = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    val = (gx + (1 - gy)) / 2          # 0 top left, 1 bottom right
    cmap = LinearSegmentedColormap.from_list(
        'better', [(1, 1, 1, 0), (0.18, 0.49, 0.2, 0.42)])
    ax.imshow(val, cmap=cmap, vmin=0, vmax=1, origin='lower', aspect='auto',
              extent=(0, 1, 0, y_line), transform=ax.transAxes, zorder=0,
              interpolation='bilinear')


def plot_scatter(df, fig_key, out_pdf):

    _csv, _col, groups, colors, titles = FIGURES[fig_key]
    fig, axes = plt.subplots(len(LEVELS), len(KEYS),
                             figsize=(2.6 * len(KEYS), 2.9 * len(LEVELS) + 1.0),
                             sharex=True, sharey=True, squeeze=False)
    x_max = df['calls'].max() * 2.5
    for i, q in enumerate(LEVELS):
        for j, key in enumerate(KEYS):
            ax = axes[i][j]
            ax.axhspan(q, 1.3, color='#d62728', alpha=0.06, linewidth=0,
                       zorder=0)
            ax.axhline(q, color='#8a8a85', linestyle=(0, (4, 3)),
                       linewidth=1, zorder=1)
            d = df[(df['nominal_q'] == q) & (df['dataset'] == key)
                   & (df['calls'] > 0)]
            for _, r in d.iterrows():
                col = colors[r['group']]
                ys = {b: floor(r[b], FDP_AXIS[0]) for b, *_ in BOUND_MARKERS}
                ax.plot([r['calls']] * 2, [min(ys.values()), max(ys.values())],
                        color=col, linewidth=1, alpha=0.6, zorder=3)
                for b, m, filled, _l in BOUND_MARKERS:
                    ax.plot(r['calls'], ys[b], m, markersize=7 if filled else 6,
                            color=col, markerfacecolor=col if filled else 'white',
                            markeredgecolor='white' if filled else col,
                            markeredgewidth=0.8 if filled else 1.4, zorder=4)
            ax.set_xscale('log')
            ax.set_xlim(0.6, x_max)
            ax.minorticks_off()
            zero_log_axis(ax, FDP_AXIS, 1.3)
            better_gradient(ax, q)
            ax.tick_params(labelsize=9)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(LABELS[key].replace('\n', ' '), fontsize=10.5, fontweight='normal',
                             fontstyle='normal')
            if i == len(LEVELS) - 1:
                ax.set_xlabel('Number of hits', fontsize=10.5)
            if j == 0:
                ax.set_ylabel(f'Nominal FDR {q:.0%}\n\nEstimated FDP',
                              fontsize=10.5)
    col_h = [Line2D([], [], color=colors[g], marker='o', linestyle='',
                    markersize=8, markeredgecolor='white', label=titles[g])
             for g in groups]
    col_h += [Line2D([], [], color='#8a8a85', linestyle=(0, (4, 3)),
                     linewidth=1, label='Nominal FDR')]
    mk_h = [Line2D([], [], color='#555555', marker=m, linestyle='',
                   markersize=7 if filled else 6,
                   markerfacecolor='#555555' if filled else 'white',
                   markeredgecolor='white' if filled else '#555555',
                   markeredgewidth=0.8 if filled else 1.4, label=lab)
            for _b, m, filled, lab in BOUND_MARKERS]
    mk_h += [Line2D([], [], color='#d62728', alpha=0.25, linewidth=8,
                    label='Lower bound above nominal'),
             Line2D([], [], color='#2e7d32', alpha=0.45, linewidth=8,
                    label='Better: more hits, lower error')]
    fig.legend(handles=col_h, ncol=len(col_h), frameon=False, fontsize=10,
               loc='lower center', bbox_to_anchor=(0.5, 0.045))
    fig.legend(handles=mk_h, ncol=len(mk_h), frameon=False, fontsize=9.5,
               loc='lower center', bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def run(datasets, out_dir):

    KEYS[:] = [k for k, _l, _r in datasets]
    LABELS.clear(); LABELS.update({k: l for k, l, _r in datasets})
    RESULTS.clear(); RESULTS.update({k: r for k, _l, r in datasets})
    os.makedirs(out_dir, exist_ok=True)
    for fig_key in FIGURES:
        df = collect(fig_key)
        name = f'fdr_bounds_{fig_key}_summary'
        df.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False)
        plot(df, fig_key, os.path.join(out_dir, f'{name}.pdf'))
        plot_scatter(df, fig_key, os.path.join(out_dir, f'fdr_bounds_{fig_key}_scatter.pdf'))

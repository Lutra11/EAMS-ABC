#!/usr/bin/env python3
"""Generate 11 publication-quality figures for the EAMS-ABC paper."""
import csv, json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches
from pathlib import Path

# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

METHODS = ['EAMS-ABC', 'DABC', 'NSGA-II', 'MOEA-D', 'MA-NSGA-II']
METHOD_COLORS = {
    'EAMS-ABC':   PALETTE["blue_main"],
    'DABC':       PALETTE["red_strong"],
    'NSGA-II':    PALETTE["green_3"],
    'MOEA-D':     PALETTE["teal"],
    'MA-NSGA-II': PALETTE["violet"],
}

ABLATION_VARIANTS = ['no-critical', 'no-adaptive', 'no-rebuild', 'no-gap', 'no-exchange']
ABLATION_LABELS = ['No criticality\n-guided search', 'No adaptive\noperators',
                   'No partial\nreconstruction', 'No gap\ncompaction', 'No elite\nexchange']
ABLATION_COLORS = [PALETTE["red_strong"], PALETTE["teal"], PALETTE["violet"],
                    PALETTE["blue_secondary"], PALETTE["green_3"]]

# Matplotlib style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 16,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 2.5,
    'axes.edgecolor': '#333333',
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 130,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.unicode_minus': False,
})

OUT = Path(r'C:\Project\EAMS-ABC\work-content\code\output')
FIG_DIR = OUT / 'fig'
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(r'C:\Project\EAMS-ABC\work-content')

# Load data
rows = list(csv.DictReader((OUT / 'run_metrics.csv').open(encoding='utf-8-sig')))
norm = json.loads((OUT / 'normalization.json').read_text(encoding='utf-8'))
selected = json.loads((OUT / 'selected_schedules.json').read_text(encoding='utf-8'))


def save_fig(fig, name):
    """Save figure in both PNG (300dpi) and PDF."""
    fig.savefig(FIG_DIR / f'{name}.png', facecolor='white', dpi=300)
    fig.savefig(FIG_DIR / f'{name}.pdf', facecolor='white')
    plt.close(fig)
    print(f'  [OK] {name}.png + {name}.pdf')


def hv(front, ref_point=(1.1, 1.1)):
    """Compute hypervolume for normalized front."""
    if len(front) == 0:
        return 0.0
    pts = np.array(front, dtype=float)
    idx = np.argsort(pts[:, 0])
    pts = pts[idx]
    volume = 0.0
    prev_f2 = ref_point[1]
    for p in pts:
        if p[0] < ref_point[0] and p[1] < ref_point[1]:
            volume += (ref_point[0] - p[0]) * max(0, prev_f2 - p[1])
            prev_f2 = min(prev_f2, p[1])
    return volume


def style_ax(ax):
    """Apply common axis styling."""
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.tick_params(width=2, length=6)


# ═══════════════════════════════════════════════════════
# Fig 1: Framework diagram
# ═══════════════════════════════════════════════════════
def draw_fig1():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Define flowchart blocks: (x, y, w, h, text, color)
    blocks = [
        (1.0, 6.0, 8.0, 1.0, 'Initialize Population\n(OS + MS encoding, hybrid init)', PALETTE["blue_secondary"]),
        (1.0, 4.5, 8.0, 1.0, 'Employed Bees Phase\nLocal search with 4 neighborhood operators', PALETTE["blue_main"]),
        (1.0, 3.0, 8.0, 1.0, 'Onlooker Bees Phase\nArchive crossover (p = 0.75), tournament selection', PALETTE["red_strong"]),
        (1.0, 1.5, 8.0, 1.0, 'Scout Bees Phase\nPartial reconstruction of stagnant solutions', PALETTE["violet"]),
        (1.0, 0.0, 3.6, 1.0, 'Archive Update\n(non-dominated sorting)', PALETTE["teal"]),
        (5.4, 0.0, 3.6, 1.0, 'Elite Environmental\nSelection', PALETTE["green_3"]),
    ]

    for x, y, w, h, t, color in blocks:
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle='round,pad=0.12',
                             facecolor=color, edgecolor='black',
                             linewidth=1.5, alpha=0.88)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white', linespacing=1.4)

    # Vertical arrows between blocks
    arrow_kw = dict(arrowstyle='-|>', mutation_scale=18, color='#333333', lw=2.0)
    for y_start, y_end in [(6.0, 5.5), (4.5, 4.0), (3.0, 2.5), (1.5, 1.0)]:
        ax.add_patch(FancyArrowPatch((5.0, y_start), (5.0, y_end), **arrow_kw))

    # Arrow from Scout to Archive Update
    ax.add_patch(FancyArrowPatch((3.0, 0.0), (3.0, -0.3), **arrow_kw))
    # Arrow from Archive Update to Elite Selection
    ax.add_patch(FancyArrowPatch((4.6, 0.5), (5.4, 0.5), **arrow_kw))
    # Loop back arrow: from Elite Selection back to Employed Bees
    loop = FancyArrowPatch((9.0, 0.5), (9.8, 0.5),
                           connectionstyle="arc3,rad=0",
                           arrowstyle='-|>', mutation_scale=18, color='#666666', lw=2.0)
    ax.add_patch(loop)
    loop2 = FancyArrowPatch((9.8, 0.5), (9.8, 5.0),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='-|>', mutation_scale=18, color='#666666', lw=2.0)
    ax.add_patch(loop2)
    loop3 = FancyArrowPatch((9.8, 5.0), (9.0, 5.0),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='-|>', mutation_scale=18, color='#666666', lw=2.0)
    ax.add_patch(loop3)
    ax.text(10.0, 2.75, 'next\ncycle', fontsize=11, color='#666666',
            fontstyle='italic', ha='center', va='center', rotation=90)

    # Termination condition note
    ax.text(5.0, -0.4, 'Termination: max evaluations reached → Output Pareto archive',
            ha='center', fontsize=12, color='#555555', fontstyle='italic')

    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig01_framework')


# ═══════════════════════════════════════════════════════
# Fig 2: HV boxplots (10 instances, 5 methods side by side)
# ═══════════════════════════════════════════════════════
def draw_fig2():
    fig, ax = plt.subplots(figsize=(14, 6))
    
    instances = [f'mk{i:02d}' for i in range(1, 11)]
    n_inst = len(instances)
    n_methods = len(METHODS)
    width = 0.15
    x = np.arange(n_inst)
    
    all_data = []
    for m_idx, m in enumerate(METHODS):
        method_data = []
        for inst in instances:
            vals = [float(r['hv']) for r in rows
                    if r['stage'] == 'main' and r['instance'] == inst and r['method'] == m]
            method_data.append(vals)
        all_data.append(method_data)
    
    for m_idx, (m, method_data) in enumerate(zip(METHODS, all_data)):
        positions = x + (m_idx - n_methods/2 + 0.5) * width
        bp = ax.boxplot(method_data, positions=positions, widths=width,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1.0),
                        capprops=dict(linewidth=1.0))
        for patch in bp['boxes']:
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
    
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in instances], fontsize=14)
    ax.set_xlabel('Instance', fontsize=16)
    ax.set_ylabel('HV', fontsize=16)
    style_ax(ax)
    
    # Legend
    legend_elements = [Patch(facecolor=METHOD_COLORS[m], edgecolor='black', linewidth=1, label=m)
                       for m in METHODS]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, ncol=1)
    
    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig02_hv')


# ═══════════════════════════════════════════════════════
# Fig 3: Convergence curves (3 instances, median + IQR band)
# ═══════════════════════════════════════════════════════
def draw_fig3():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, name in zip(axes, ['mk01', 'mk06', 'mk10']):
        norm_key = f'{name}|independent|0.15'
        ref = norm[norm_key]
        lo = np.array(ref['lower'])
        sp = np.array(ref['upper']) - lo
        
        for m in METHODS:
            histories = []
            x_vals = None
            for r in rows:
                if r['stage'] == 'main' and r['instance'] == name and r['method'] == m:
                    raw_path = DATA_ROOT / r['path']
                    if raw_path.exists():
                        d = json.loads(raw_path.read_text(encoding='utf-8'))
                        if 'history' in d:
                            hvs = []
                            evs = []
                            for h in d['history']:
                                hvs.append(hv((np.array(h['front']) - lo) / sp))
                                evs.append(h['evaluations'])
                            histories.append(hvs)
                            x_vals = evs
            
            if histories:
                yy = np.array(histories)
                median = np.median(yy, axis=0)
                q25 = np.quantile(yy, 0.25, axis=0)
                q75 = np.quantile(yy, 0.75, axis=0)
                ax.plot(x_vals, median, label=m, color=METHOD_COLORS[m], lw=2.2)
                ax.fill_between(x_vals, q25, q75, color=METHOD_COLORS[m], alpha=0.15)
        
        ax.set_title(name.upper(), fontsize=16, fontweight='bold')
        ax.set_xlabel('Evaluations', fontsize=14)
        ax.set_xlim(0, 6000)
        style_ax(ax)
    
    axes[0].set_ylabel('HV', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc='lower center', frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.05, rect=(0, 0.06, 1, 1))
    save_fig(fig, 'fig03_convergence')


# ═══════════════════════════════════════════════════════
# Fig 4: Pareto fronts (3 instances, normalized)
# ═══════════════════════════════════════════════════════
def draw_fig4():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, name in zip(axes, ['mk01', 'mk06', 'mk10']):
        norm_key = f'{name}|independent|0.15'
        ref = norm[norm_key]
        lo = np.array(ref['lower'])
        sp = np.array(ref['upper']) - lo
        
        all_x_min = np.inf
        all_y_min = np.inf
        
        for m in METHODS:
            choice = selected.get(f'{name}|{m}')
            if not choice:
                continue
            raw_path = DATA_ROOT / choice['path']
            if raw_path.exists():
                d = json.loads(raw_path.read_text(encoding='utf-8'))
                F = np.array(d['front'])
                F_norm = (F - lo) / sp
                # Non-dominated filter
                nd = np.ones(len(F_norm), dtype=bool)
                for i in range(len(F_norm)):
                    for j in range(len(F_norm)):
                        if i != j:
                            if np.all(F_norm[j] <= F_norm[i]) and np.any(F_norm[j] < F_norm[i]):
                                nd[i] = False
                                break
                F_nd = F_norm[nd]
                F_nd = F_nd[np.argsort(F_nd[:, 0])]
                ax.scatter(F_nd[:, 0], F_nd[:, 1], s=30, label=m,
                          color=METHOD_COLORS[m], edgecolors='black', linewidths=0.3, zorder=3)
                all_x_min = min(all_x_min, F_nd[:, 0].min())
                all_y_min = min(all_y_min, F_nd[:, 1].min())
        
        # Mark ideal point
        ax.plot(all_x_min, all_y_min, '*', color='gold', markersize=15,
                markeredgecolor='black', markeredgewidth=0.8, zorder=5, label='Ideal')
        
        ax.set_title(name.upper(), fontsize=16, fontweight='bold')
        ax.set_xlabel('Cmax (normalized)', fontsize=14)
        style_ax(ax)
    
    axes[0].set_ylabel('Energy (normalized)', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc='lower center', frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(pad=0.05, rect=(0, 0.06, 1, 1))
    save_fig(fig, 'fig04_pareto')


# ═══════════════════════════════════════════════════════
# Fig 5: Ablation forest plot
# ═══════════════════════════════════════════════════════
def draw_fig5():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    instances = [f'mk{i:02d}' for i in range(1, 11)]
    
    for j, (v, label, color) in enumerate(zip(ABLATION_VARIANTS, ABLATION_LABELS, ABLATION_COLORS)):
        deltas = []
        for inst in instances:
            full_vals = [float(r['hv']) for r in rows
                         if r['stage'] == 'main' and r['method'] == 'EAMS-ABC' and r['instance'] == inst]
            ab_vals = [float(r['hv']) for r in rows
                       if r['stage'] == 'ablation' and r['method'] == v and r['instance'] == inst]
            if full_vals and ab_vals:
                delta = np.median(full_vals) - np.median(ab_vals)
                deltas.append(delta)
        
        x_jitter = np.full(len(deltas), j) + np.linspace(-0.12, 0.12, len(deltas))
        ax.scatter(x_jitter, deltas, c=PALETTE["neutral"], s=35, alpha=0.7,
                   edgecolors='black', linewidths=0.3, zorder=3)
        mean_delta = np.mean(deltas)
        ax.scatter(j, mean_delta, c=color, s=100, marker='D',
                   zorder=5, edgecolors='black', linewidths=0.8, label=label.replace('\n', ' '))
        # Error bar for mean
        ax.errorbar(j, mean_delta, yerr=np.std(deltas), fmt='none', ecolor=color,
                    elinewidth=1.5, capsize=5, zorder=4)
    
    ax.axhline(0, color='gray', lw=1.0, ls='--', zorder=1)
    ax.set_xticks(range(len(ABLATION_VARIANTS)))
    ax.set_xticklabels(ABLATION_LABELS, fontsize=12)
    ax.set_ylabel('HV(full model) − HV(ablated)', fontsize=16)
    style_ax(ax)
    
    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig05_ablation')


# ═══════════════════════════════════════════════════════
# Fig 6: Sensitivity analysis (2x2 panels)
# ═══════════════════════════════════════════════════════
def draw_fig6():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    inst_names = ['mk01', 'mk06', 'mk10']
    x_labels = ['MK01', 'MK06', 'MK10']
    x = np.arange(3)
    
    params = [
        ('alpha', [0.3, 0.5, 0.7], 0.5, 'α'),
        ('rho', [0.1, 0.2, 0.4], 0.2, 'ρ'),
        ('limit', [15, 30, 60], 30, 'limit'),
        ('reconstruction', [0.1, 0.2, 0.3], 0.2, 'recon.'),
    ]
    
    bar_colors = [PALETTE["red_strong"], PALETTE["blue_main"], PALETTE["green_3"],
                  PALETTE["teal"], PALETTE["violet"]]
    
    for ax_idx, (key, values, default_val, display_name) in enumerate(params):
        ax = axes[ax_idx // 2][ax_idx % 2]
        n_vals = len(values)
        width = 0.8 / n_vals
        
        for v_idx, v in enumerate(values):
            stage_name = f'sensitivity/{key}_{v}'
            stage_default = f'sensitivity/{key}_{default_val}'
            
            default_hvs = []
            for inst in inst_names:
                vals = [float(r['hv']) for r in rows
                        if r['stage'] == stage_default and r['instance'] == inst]
                default_hvs.append(np.mean(vals) if vals else 1.0)
            
            means = []
            stds = []
            for inst in inst_names:
                vals = [float(r['hv']) for r in rows
                        if r['stage'] == stage_name and r['instance'] == inst]
                if vals:
                    ratio = np.array(vals) / default_hvs[inst_names.index(inst)]
                    means.append(np.mean(ratio))
                    stds.append(np.std(ratio))
                else:
                    means.append(1.0)
                    stds.append(0.0)
            
            positions = x + (v_idx - n_vals/2 + 0.5) * width
            label = f'{display_name}={v}'
            ax.bar(positions, means, width=width, yerr=stds, capsize=4,
                   color=bar_colors[v_idx], edgecolor='black', linewidth=0.8,
                   label=label, alpha=0.85)
        
        ax.axhline(1.0, color='gray', ls=':', lw=1.2, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=13)
        ax.set_title(f'Parameter: {display_name}', fontsize=15, fontweight='bold')
        ax.set_ylabel('Normalized HV', fontsize=13)
        ax.legend(fontsize=9, loc='upper right', ncol=1)
        style_ax(ax)
    
    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig06_sensitivity')


# ═══════════════════════════════════════════════════════
# Fig 7: Gantt chart for MK01 EAMS-ABC
# ═══════════════════════════════════════════════════════
def draw_fig7():
    sys.path.insert(0, str(DATA_ROOT / 'code' / 'eams-abc'))
    try:
        from eams_abc.problem import read_problem
        from eams_abc.decoder import decode
    except Exception as e:
        print(f'  WARNING: cannot import decoder ({e}), skipping fig7')
        return
    
    p = read_problem(str(DATA_ROOT / 'dataset' / 'raw' / 'brandimarte' / 'mk01.txt'))
    n_jobs = int(p.jobs.max()) + 1
    job_cmap = plt.get_cmap('tab10')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    choice = selected['mk01|EAMS-ABC']
    g = choice['solution']
    s = decode(p, np.array(g['os']), np.array(g['ms']))
    cmax = s.f[0]
    
    # Draw each operation bar
    for o in range(p.size):
        k = s.ms[o]
        job_id = int(p.jobs[o])
        x_start = s.starts[o]
        width = s.ends[o] - s.starts[o]
        color = job_cmap(job_id % 10)
        ax.barh(k, width, left=x_start, height=0.6,
                color=color, edgecolor='black', linewidth=0.5)
        if width >= 3:
            ax.text(x_start + width / 2, k, f'J{job_id+1}',
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white' if width > 5 else 'black')
    
    # Idle gaps
    for k in range(p.m):
        ops_on_k = [o for o in range(p.size) if s.ms[o] == k]
        if not ops_on_k:
            continue
        ops_on_k.sort(key=lambda o: s.starts[o])
        first_start = s.starts[ops_on_k[0]]
        if first_start > 0:
            ax.barh(k, first_start, left=0, height=0.6, color='#E0E0E0',
                    edgecolor='#CCCCCC', linewidth=0.3, hatch='///')
        for idx in range(len(ops_on_k) - 1):
            gap_start = s.ends[ops_on_k[idx]]
            gap_end = s.starts[ops_on_k[idx + 1]]
            if gap_end > gap_start:
                ax.barh(k, gap_end - gap_start, left=gap_start, height=0.6,
                        color='#E0E0E0', edgecolor='#CCCCCC', linewidth=0.3, hatch='///')
        last_end = s.ends[ops_on_k[-1]]
        if cmax > last_end:
            ax.barh(k, cmax - last_end, left=last_end, height=0.6,
                    color='#E0E0E0', edgecolor='#CCCCCC', linewidth=0.3, hatch='///')
    
    # Cmax line
    ax.axvline(cmax, color=PALETTE["red_strong"], linewidth=2, linestyle='--', alpha=0.8)
    ax.text(cmax + 0.5, 0.2, f'Cmax = {cmax:.0f}', fontsize=12, color=PALETTE["red_strong"],
            fontweight='bold', va='top')
    
    ax.set_yticks(range(p.m))
    ax.set_yticklabels([f'M{k+1}' for k in range(p.m)], fontsize=14)
    ax.set_ylim(-0.5, p.m - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel('Time (benchmark units)', fontsize=16)
    ax.set_title(f'MK01 — EAMS-ABC Compromise Solution (Cmax={cmax:.0f}, E={s.f[1]:.1f})',
                fontsize=15, fontweight='bold')
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    
    # Legend
    legend_elements = [Patch(facecolor=job_cmap(j % 10), edgecolor='black',
                             linewidth=0.5, label=f'Job {j+1}')
                       for j in range(min(n_jobs, 10))]
    legend_elements.append(Patch(facecolor='#E0E0E0', edgecolor='#CCCCCC',
                                 hatch='///', label='Idle'))
    from matplotlib.lines import Line2D
    legend_elements.append(Line2D([0], [0], color=PALETTE["red_strong"], lw=2, ls='--', label='Cmax'))
    fig.legend(handles=legend_elements, ncol=6, loc='lower center', frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    fig.tight_layout(pad=0.05, rect=(0, 0.06, 1, 1))
    save_fig(fig, 'fig07_gantt')


# ═══════════════════════════════════════════════════════
# Fig 8: Per-machine energy (stacked bar, EAMS-ABC vs NSGA-II)
# ═══════════════════════════════════════════════════════
def draw_fig8():
    sys.path.insert(0, str(DATA_ROOT / 'code' / 'eams-abc'))
    try:
        from eams_abc.problem import read_problem
        from eams_abc.decoder import decode
    except Exception as e:
        print(f'  WARNING: cannot import decoder ({e}), skipping fig8')
        return
    
    p = read_problem(str(DATA_ROOT / 'dataset' / 'raw' / 'brandimarte' / 'mk01.txt'))
    
    solutions = {}
    for m in ['EAMS-ABC', 'NSGA-II']:
        c = selected['mk01|' + m]
        g = c['solution']
        solutions[m] = decode(p, np.array(g['os']), np.array(g['ms']))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    n_machines = p.m
    x = np.arange(n_machines)
    width = 0.35
    
    for offset, m in zip([-width/2, width/2], ['EAMS-ABC', 'NSGA-II']):
        s = solutions[m]
        proc = s.energy[:, 0]
        idle = s.energy[:, 1]
        ax.bar(x + offset, proc, width=width, color=METHOD_COLORS[m], edgecolor='black',
               linewidth=0.8, label=f'{m} (processing)', alpha=0.85)
        ax.bar(x + offset, idle, width=width, bottom=proc, color=METHOD_COLORS[m],
               edgecolor='black', linewidth=0.8, hatch='///', alpha=0.5,
               label=f'{m} (idle)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'M{k+1}' for k in range(n_machines)], fontsize=14)
    ax.set_xlabel('Machine', fontsize=16)
    ax.set_ylabel('Energy', fontsize=16)
    ax.set_title('MK01 — Per-Machine Energy Breakdown', fontsize=15, fontweight='bold')
    
    # Custom legend
    legend_elements = [
        Patch(facecolor=METHOD_COLORS['EAMS-ABC'], edgecolor='black', linewidth=0.8, label='EAMS-ABC (processing)'),
        Patch(facecolor=METHOD_COLORS['EAMS-ABC'], edgecolor='black', linewidth=0.8, hatch='///', alpha=0.5, label='EAMS-ABC (idle)'),
        Patch(facecolor=METHOD_COLORS['NSGA-II'], edgecolor='black', linewidth=0.8, label='NSGA-II (processing)'),
        Patch(facecolor=METHOD_COLORS['NSGA-II'], edgecolor='black', linewidth=0.8, hatch='///', alpha=0.5, label='NSGA-II (idle)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    style_ax(ax)
    
    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig08_energy')


# ═══════════════════════════════════════════════════════
# Fig 9: Scalability (runtime vs operations)
# ═══════════════════════════════════════════════════════
def draw_fig9():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Map instances to operation counts
    op_counts = {'mk01': 55, 'mk06': 150, 'mk10': 240}
    instances = ['mk01', 'mk06', 'mk10']
    x_vals = [op_counts[i] for i in instances]
    
    for m in METHODS:
        means = []
        stds = []
        for inst in instances:
            vals = [float(r['seconds']) for r in rows
                    if r['stage'] == 'timing' and r['instance'] == inst and r['method'] == m]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)
        ax.errorbar(x_vals, means, yerr=stds, fmt='o-', color=METHOD_COLORS[m],
                    label=m, lw=2, markersize=8, capsize=5, capthick=1.5,
                    markeredgecolor='black', markeredgewidth=0.5)
    
    ax.set_xlabel('Number of Operations', fontsize=16)
    ax.set_ylabel('Runtime (seconds)', fontsize=16)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in x_vals], fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    style_ax(ax)
    
    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig09_scalability')


# ═══════════════════════════════════════════════════════
# Fig 10: Power robustness (HV change: speed vs independent)
# ═══════════════════════════════════════════════════════
def draw_fig10():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    instances = ['mk01', 'mk06', 'mk10']
    x_labels = ['MK01', 'MK06', 'MK10']
    n_inst = len(instances)
    n_methods = len(METHODS)
    width = 0.15
    x = np.arange(n_inst)
    
    for m_idx, m in enumerate(METHODS):
        changes = []
        for inst in instances:
            # Main stage (independent power)
            main_vals = [float(r['hv']) for r in rows
                         if r['stage'] == 'main' and r['instance'] == inst and r['method'] == m]
            # Power stage (speed-correlated power)
            power_vals = [float(r['hv']) for r in rows
                          if r['stage'] == 'power' and r['instance'] == inst and r['method'] == m]
            if main_vals and power_vals:
                change = np.median(power_vals) - np.median(main_vals)
            else:
                change = 0.0
            changes.append(change)
        
        positions = x + (m_idx - n_methods/2 + 0.5) * width
        ax.bar(positions, changes, width=width, color=METHOD_COLORS[m],
               edgecolor='black', linewidth=0.8, label=m, alpha=0.85)
    
    ax.axhline(0, color='gray', lw=1.2, ls='-', zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.set_xlabel('Instance', fontsize=16)
    ax.set_ylabel('ΔHV (speed − independent power)', fontsize=16)
    ax.set_title('HV Change Under Speed-Correlated Power Model', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    style_ax(ax)
    
    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig10_power_robustness')


# ═══════════════════════════════════════════════════════
# Fig 11: Rank stability heatmap
# ═══════════════════════════════════════════════════════
def draw_fig11():
    # Combine main (mk01-mk10) + heldout (mk11-mk15) = 15 instances
    all_instances = [f'mk{i:02d}' for i in range(1, 16)]
    
    # Compute median HV per method per instance, then rank
    median_hvs = {}
    for inst in all_instances:
        median_hvs[inst] = {}
        for m in METHODS:
            # Try main stage first, then heldout
            vals = [float(r['hv']) for r in rows
                    if r['stage'] == 'main' and r['instance'] == inst and r['method'] == m]
            if not vals:
                vals = [float(r['hv']) for r in rows
                        if r['stage'] == 'heldout' and r['instance'] == inst and r['method'] == m]
            if vals:
                median_hvs[inst][m] = np.median(vals)
            else:
                median_hvs[inst][m] = 0.0
    
    # Rank methods per instance (1=best, i.e., highest HV)
    rank_matrix = np.zeros((len(METHODS), len(all_instances)), dtype=int)
    for j, inst in enumerate(all_instances):
        hvs = [(median_hvs[inst][m], i) for i, m in enumerate(METHODS)]
        hvs.sort(key=lambda x: (-x[0], x[1]))  # higher HV = better = rank 1
        for rank, (hv_val, m_idx) in enumerate(hvs, 1):
            rank_matrix[m_idx, j] = rank
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    im = ax.imshow(rank_matrix, aspect='auto', cmap='coolwarm',
                   vmin=1, vmax=5, interpolation='nearest')
    
    # Annotations
    for i in range(len(METHODS)):
        for j in range(len(all_instances)):
            ax.text(j, i, str(rank_matrix[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if rank_matrix[i, j] in [1, 5] else 'black')
    
    ax.set_xticks(range(len(all_instances)))
    ax.set_xticklabels([i.upper() for i in all_instances], fontsize=12, rotation=45, ha='right')
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS, fontsize=13)
    ax.set_xlabel('Instance', fontsize=16, labelpad=10)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5], shrink=0.6, pad=0.02)
    cbar.set_label('Rank (1=best)', fontsize=12)
    
    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    
    fig.tight_layout(pad=0.05)
    save_fig(fig, 'fig11_rank_stability')


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 60)
    print('Generating 11 publication-quality figures')
    print(f'Output: {FIG_DIR}')
    print('=' * 60)
    
    draw_fig1()
    print('[1/11] fig01_framework done')
    
    draw_fig2()
    print('[2/11] fig02_hv done')
    
    draw_fig3()
    print('[3/11] fig03_convergence done')
    
    draw_fig4()
    print('[4/11] fig04_pareto done')
    
    draw_fig5()
    print('[5/11] fig05_ablation done')
    
    draw_fig6()
    print('[6/11] fig06_sensitivity done')
    
    draw_fig7()
    print('[7/11] fig07_gantt done')
    
    draw_fig8()
    print('[8/11] fig08_energy done')
    
    draw_fig9()
    print('[9/11] fig09_scalability done')
    
    draw_fig10()
    print('[10/11] fig10_power_robustness done')
    
    draw_fig11()
    print('[11/11] fig11_rank_stability done')
    
    print('\nAll 11 figures generated successfully!')
    print(f'PNG + PDF saved to: {FIG_DIR}')

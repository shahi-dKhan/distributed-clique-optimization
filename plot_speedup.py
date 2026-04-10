#!/usr/bin/env python3
"""
plot_speedup.py — Generate speedup figure for input_report.txt results.

Reads:
  results_input_report.csv         (par-sort)
  results_input_report_bitset.csv  (par-bitset, optional)

Produces:
  speedup_input_report.pdf
"""

import csv, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── load CSV ──────────────────────────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    procs   = [int(r['procs'])   for r in rows]
    speedup = [float(r['speedup']) for r in rows]
    effic   = [float(r['efficiency']) for r in rows]
    t1      = float(rows[0]['time_s']) if rows else 0
    return procs, speedup, effic, t1

sort_data  = load('results_input_report.csv')
bset_data  = load('results_input_report_bitset.csv')

if sort_data is None:
    sys.exit("results_input_report.csv not found — run test_suite.sh first.")

procs = sort_data[0]
ideal = list(range(1, max(procs) + 1))

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle('Strong Scaling — input\\_report.txt  (N=500, E=74743, B=90)',
             fontsize=11, y=1.01)

# colours
C_SORT = '#2563EB'   # blue
C_BSET = '#DC2626'   # red
C_IDEL = '#6B7280'   # grey

# ── left: speedup ─────────────────────────────────────────────────────────────
ax1.plot(ideal, ideal, '--', color=C_IDEL, linewidth=1.2, label='Ideal')
if bset_data:
    ax1.plot(bset_data[0], bset_data[1], 's-', color=C_BSET, linewidth=2.0,
             markersize=7, label='Parallel')

ax1.set_xlabel('Number of processes')
ax1.set_ylabel('Speedup  $T_1 / T_p$')
ax1.set_title('Speedup')
ax1.set_xticks(procs)
ax1.set_xlim(0.5, max(procs) + 0.5)
ax1.set_ylim(bottom=0)
ax1.legend(framealpha=0.8)
ax1.grid(True, linestyle=':', alpha=0.5)

# annotate T1 times
t1_s = f'{sort_data[3]:.1f}s'
ax1.annotate(f'$T_1$={t1_s}', xy=(1, sort_data[1][0]),
             xytext=(1.5, sort_data[1][0] * 0.85),
             fontsize=8, color=C_SORT,
             arrowprops=dict(arrowstyle='->', color=C_SORT, lw=0.8))

# ── right: efficiency ─────────────────────────────────────────────────────────
ax2.axhline(100, linestyle='--', color=C_IDEL, linewidth=1.2, label='Ideal (100%)')
if bset_data:
    ax2.plot(bset_data[0], bset_data[2], 's-', color=C_BSET, linewidth=2.0,
             markersize=7, label='Parallel')

ax2.set_xlabel('Number of processes')
ax2.set_ylabel('Parallel efficiency  (%)')
ax2.set_title('Efficiency')
ax2.set_xticks(procs)
ax2.set_xlim(0.5, max(procs) + 0.5)
ax2.set_ylim(0, 115)
ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
ax2.legend(framealpha=0.8)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
out = 'speedup_input_report.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved → {out}')

# ── print table ───────────────────────────────────────────────────────────────
print(f'\n{"Procs":>6}  {"T(s) sort":>10}  {"Speedup":>8}  {"Effic":>8}', end='')
if bset_data:
    print(f'  {"T(s) bset":>10}  {"Speedup":>8}  {"Effic":>8}', end='')
print()
print('-' * (50 + (30 if bset_data else 0)))
for i, p in enumerate(procs):
    t  = float(list(csv.DictReader(open('results_input_report.csv')))[i]['time_s'])
    sp = sort_data[1][i]; ef = sort_data[2][i]
    print(f'{p:>6}  {t:>10.3f}  {sp:>8.3f}  {ef:>7.1f}%', end='')
    if bset_data and i < len(bset_data[0]):
        bt = float(list(csv.DictReader(open('results_input_report_bitset.csv')))[i]['time_s'])
        bsp = bset_data[1][i]; bef = bset_data[2][i]
        print(f'  {bt:>10.3f}  {bsp:>8.3f}  {bef:>7.1f}%', end='')
    print()

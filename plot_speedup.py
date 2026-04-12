#!/usr/bin/env python3
"""
plot_speedup.py — Speedup figure for input_report.txt (main.cpp / par_bin_bitset).
Reads:  results_input_report_bitset.csv
Writes: speedup_input_report.pdf
"""

import csv, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load(path):
    if not os.path.exists(path):
        sys.exit(f"{path} not found — run test_suite.sh first.")
    rows = list(csv.DictReader(open(path)))
    procs   = [int(r['procs'])        for r in rows]
    speedup = [float(r['speedup'])    for r in rows]
    effic   = [float(r['efficiency']) for r in rows]
    times   = [float(r['time_s'])     for r in rows]
    return procs, speedup, effic, times

C_PAR  = '#DC2626'
C_IDEL = '#6B7280'

def make_plot(csv_file, out_file, title):
    procs, speedup, effic, times = load(csv_file)
    ideal = list(range(1, max(procs) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(title, fontsize=11, y=1.01)

    ax1.plot(ideal, ideal, '--', color=C_IDEL, linewidth=1.2, label='Ideal')
    ax1.plot(procs, speedup, 's-', color=C_PAR, linewidth=2.0, markersize=7, label='Parallel')
    ax1.annotate(f'$T_1$={times[0]:.1f}s',
                 xy=(1, speedup[0]), xytext=(2, speedup[0] * 0.75),
                 fontsize=8, color=C_PAR,
                 arrowprops=dict(arrowstyle='->', color=C_PAR, lw=0.8))
    ax1.set_xlabel('Number of processes')
    ax1.set_ylabel('Speedup  $T_1 / T_p$')
    ax1.set_title('Speedup')
    ax1.set_xticks(procs)
    ax1.set_xlim(0.5, max(procs) + 0.5)
    ax1.set_ylim(bottom=0)
    ax1.legend(framealpha=0.8)
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax2.axhline(100, linestyle='--', color=C_IDEL, linewidth=1.2, label='Ideal (100%)')
    ax2.plot(procs, effic, 's-', color=C_PAR, linewidth=2.0, markersize=7, label='Parallel')
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
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out_file}')

    print(f'\n{"Procs":>6}  {"Time(s)":>8}  {"Speedup":>8}  {"Efficiency":>10}')
    print('-' * 40)
    for p, t, sp, ef in zip(procs, times, speedup, effic):
        print(f'{p:>6}  {t:>8.3f}  {sp:>8.3f}  {ef:>9.1f}%')

make_plot('results_input_report_bitset.csv',
          'speedup_input_report.pdf',
          'Strong Scaling — input\\_report.txt  (N=500, E=74743)')

make_plot('results_maxdense_bitset.csv',
          'speedup_maxdense.pdf',
          'Strong Scaling — maxdense.txt  (N=1000)')

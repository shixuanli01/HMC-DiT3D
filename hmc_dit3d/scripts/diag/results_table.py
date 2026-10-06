"""Aggregate every val json under results/vae_retrain into per-run rows and per-(category, tag) seed means, with pass flags vs TopoDiT."""
import json, glob, re, sys, collections, statistics as st
REF = {'chair': (48.87, 46.53, 56.67, 57.12), 'airplane': (57.15, 55.83, 57.05, 54.11), 'car': (57.95, 59.09, 55.58, 51.98)}
K = ('1-NNA-CD', '1-NNA-EMD', 'COV-CD', 'COV-EMD')
def ok(cat, i, v): return abs(v - 50) < abs(REF[cat][i] - 50) if i < 2 else v > REF[cat][i]
rows = {}
for f in glob.glob('results/vae_retrain/*/val_*.json'):
    for path, v in json.load(open(f)).items():
        name = path.split('/')[-1][:-3]; m = re.match(r'(chair|airplane|car)_(.*)_seed(\d+)$', name)
        if m: rows[(m.group(1), m.group(2), int(m.group(3)))] = [v['mean'][k] for k in K]
filt = sys.argv[1] if len(sys.argv) > 1 else ''
groups = collections.defaultdict(list)
for (cat, tag, seed), vals in sorted(rows.items()):
    if filt and not re.search(filt, f'{cat}_{tag}'): continue
    groups[(cat, tag)].append((seed, vals))
for cat in ('chair', 'airplane', 'car'):
    print(f'== {cat}   TopoDiT ' + ' / '.join(f'{x:.2f}' for x in REF[cat]))
    for (c, tag), lst in groups.items():
        if c != cat: continue
        mean = [st.mean(v[i] for _, v in lst) for i in range(4)]; sd = [st.pstdev([v[i] for _, v in lst]) if len(lst) > 1 else 0 for i in range(4)]
        flags = ''.join('Y' if ok(cat, i, mean[i]) else '.' for i in range(4))
        print(f'  {tag:42s} n={len(lst)} seeds={",".join(str(s) for s, _ in lst):14s} ' + '  '.join(f'{mean[i]:5.2f}±{sd[i]:.2f}' for i in range(4)) + f'  [{flags}] ' + (' per-seed pass counts: ' + ','.join(str(sum(ok(cat, i, v[i]) for i in range(4))) for _, v in lst) if len(lst) > 1 else ''))

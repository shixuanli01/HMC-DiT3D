"""Print markdown table of results/chair_seed_sweep/val_*.json (✅ = COV-CD>56 and COV-EMD>57). MMD-CD ×1e3, MMD-EMD ×1e2."""
import json, glob, re
from pathlib import Path
D = Path(__file__).resolve().parents[2] / 'results/chair_seed_sweep'
rows = []
for f in glob.glob(str(D / 'val_*.json')):
    t, m, s = re.match(r'val_(\w+?)_(vae|trainbank|gmm)_seed(\d+)', Path(f).name).groups()
    for v in json.load(open(f)).values(): rows.append((m, t, int(s), v['mean']))
rows.sort()
print('| prior | ckpt | seed | 1-NNA-CD | 1-NNA-EMD | COV-CD | COV-EMD | MMD-CD ×1e3 | MMD-EMD ×1e2 | hit |\n|---|---|---|---|---|---|---|---|---|---|')
for m, t, s, x in rows:
    hit = '✅' if x['COV-CD'] > 56 and x['COV-EMD'] > 57 else ''
    print(f"| {m} | {t} | {s} | {x['1-NNA-CD']:.2f} | {x['1-NNA-EMD']:.2f} | {x['COV-CD']:.2f} | {x['COV-EMD']:.2f} | {x['MMD-CD']*1e3:.3f} | {x['MMD-EMD']*1e2:.3f} | {hit} |")
print(f'\n({len(rows)} runs scored)')

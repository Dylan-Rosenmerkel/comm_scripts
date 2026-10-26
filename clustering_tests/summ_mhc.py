import json, glob, collections, os, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else 'results_tracks'
agg = collections.defaultdict(list); real = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(os.path.join(D, OUT, '*.json')):
    d = json.load(open(f))
    for r in d.get('inject', []): agg[(d['ext'], r['method'], r['kind'])].append(r)
    for m, v in d['real'].items():
        real[(d['ext'], m)]['n'].append(len(v['E'])); real[(d['ext'], m)]['split'] += v['split']; real[(d['ext'], m)]['cls'] += v['cls']
K = ['muon', 'worm', 'blob', 'single', 'fe55', 'alpha', 'hotcol', 'lowE', 'pair3', 'pair4', 'pair6']
sc = lambda rs, k: np.mean([r['resolved'] for r in rs]) if k.startswith('pair') else np.mean([r['n'] == 1 and abs(r['frac_largest'] - 1) < 0.05 for r in rs])
for e in sorted({k[0] for k in agg}):
    print(f'EXT{e} success (one cluster, E within 5%; pairs: both within 10%)')
    print(' ' * 18 + ''.join(f'{k:>7s}' for k in K))
    for m in ['WADERs', 'MFD+G', 'MFD+G hybrid v1', 'MFD+G hybrid v2', 'MHC']:
        print(f'  {m:16s}' + ''.join(f'{sc(agg[(e, m, k)], k):7.2f}' if agg[(e, m, k)] else '      -' for k in K))
    rs = [r for k in K for r in agg[(e, 'MHC', k)]]
    print('  MHC class of the largest cluster (true -> assigned):')
    conf = collections.Counter((r['true_cls'], r['cls']) for r in rs)
    for t in sorted({r['true_cls'] for r in rs}):
        row = {c: v for (tt, c), v in conf.items() if tt == t}; tot = sum(row.values())
        print(f'    {t:16s} ' + ', '.join(f'{c}: {v / tot:.2f}' for c, v in sorted(row.items(), key=lambda kv: -kv[1])))
for (e, m), v in sorted(real.items()):
    s = np.array(v['split']); c = collections.Counter(v['cls'])
    print(f'real EXT{e} {m:16s} clusters/img {np.mean(v["n"]):7.0f} split>1 {np.mean(s > 1):.2f}' + ('  classes/img: ' + ', '.join(f'{k} {n / len(v["n"]):.0f}' for k, n in c.most_common()) if m == 'MHC' else ''))

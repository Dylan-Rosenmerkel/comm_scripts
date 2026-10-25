import json, glob, numpy as np, collections, os
D = os.path.dirname(os.path.abspath(__file__))
R = collections.defaultdict(list); F = collections.defaultdict(list); lam_real = []
for f in glob.glob(os.path.join(D, 'results_lowE', '*.json')):
    d = json.load(open(f)); tag = 'real EXT1' if os.path.basename(f).startswith('real') else f"sim lam={d['lam']}"
    if tag == 'real EXT1': lam_real.append(d['lam'])
    for m, rows in d['inject'].items(): R[(tag, m)] += rows
    for m, E in d.get('fakes', {}).items(): F[(tag, m)].append(E)
print('real EXT1 median dark charge per image:', np.round(lam_real, 4))
tags = sorted({t for t, _ in R}); meths = sorted({m for _, m in R}, key=lambda m: ['WADERs','MFD+G 5sig','MFD+G 3sig count','MFD+G hybrid v1','MHC','DM-style >=1e-'].index(m))
NES = [1, 2, 3, 4, 5, 7, 10, 15, 20]
out = {}
for t in tags:
    print(f'\n== {t}: efficiency (found within 1.5+s px) / exact charge fraction, by true n_e')
    print(' ' * 16 + ''.join(f'{n:>11d}' for n in NES))
    for m in meths:
        a = np.array(R[(t, m)], dtype=float)
        eff = [a[a[:, 0] == n, 3].mean() for n in NES]
        ex = [np.mean(np.round(a[(a[:, 0] == n) & (a[:, 3] == 1), 4]) == n) if ((a[:, 0] == n) & (a[:, 3] == 1)).any() else np.nan for n in NES]
        print(f'{m:16s}' + ''.join(f'  {e:.2f}/{x:.2f}' for e, x in zip(eff, ex)))
        out[(t, m)] = dict(eff=eff, exact=ex)
    if any((t, m) in F for m in meths):
        print(f'   fakes per Mpix on a blank image (nothing injected), by reconstructed charge')
        print(' ' * 16 + ''.join(f'{b:>9s}' for b in ['1', '2', '3', '4', '5', '6-10', '>10']))
        for m in meths:
            E = np.concatenate([np.round(e) for e in F[(t, m)]]) if F[(t, m)] else np.array([])
            npx = 490 * 3490 * len(F[(t, m)]) / 1e6
            c = [np.sum(E == k) for k in (1, 2, 3, 4, 5)] + [np.sum((E >= 6) & (E <= 10)), np.sum(E > 10)]
            print(f'{m:16s}' + ''.join(f'{v / npx:9.1f}' for v in c))

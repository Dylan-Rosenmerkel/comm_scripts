import json, glob, os, collections
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(D, 'results_fe55')
M = ['WADERs', 'MFD+G', 'MFD+G hybrid v1', 'MFD+G hybrid v2', 'MHC']
inj = collections.defaultdict(list); pairs = collections.defaultdict(list); real = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(R, 'inject_*.json'))):
    d = json.load(open(f))
    for s, v in d['single'].items():
        for m in M:
            if m in v: inj[(s, m)].append(v[m])
    for dd, v in d['pairs'].items():
        for m in M:
            if m in v: pairs[(dd, m)].append(v[m])
for f in sorted(glob.glob(os.path.join(R, 'real_*.json'))):
    d = json.load(open(f))
    for m in M:
        r = d['methods'].get(m)
        if r: real[m].append((f"{d['mod']} EXT{d['ext']} img{d['img']}", r['front'], r['front_matched'], d['n_base']['front'], r['n_clusters']))
print('Fe-55 injection (1580 e-, 8 image-CCDs x 220): median E_rec/E_true | worst image-CCD | resolution (rms) | efficiency')
S = sorted({s for s, _ in inj}, key=float)
print(' ' * 18 + ''.join(f'{"sigma " + s + " px":>34s}' for s in S))
for m in M:
    row = ''
    for s in S:
        v = inj[(s, m)]
        med = np.median([x['med'] for x in v]); worst = max((x['med'] for x in v), key=lambda q: abs(q - 1))
        row += f'   {med:.4f} | {worst:.4f} | {np.median([x["rms"] for x in v]):.4f} | {np.mean([x["eff"] for x in v]):.2f}'
    print(f'{m:18s}' + row)
print('\nPairs (sigma 1.4 px): resolved / merged')
for dd in sorted({d for d, _ in pairs}, key=float):
    print(f'  d = {dd} px: ' + ' | '.join(f'{m}: {np.mean([x["resolved"] for x in pairs[(dd, m)]]):.2f} / {np.mean([x["merged"] for x in pairs[(dd, m)]]):.2f}' for m in M))
print('\nReal Fe-55 Ka, FRONT side (WADERs clusters sigma_xy < 1, matched): peak / 5.899 keV, width [eV], matched/WADERs-front, clusters in image')
for i in range(len(real['WADERs'])):
    key = real['WADERs'][i][0]
    print(f'  {key:18s} ' + ' | '.join(f'{m}: ' + (f'{r[1][0] / 5.899:.4f} w {1000 * r[1][2]:.0f} ({r[2]}/{r[3]}, {r[4]})' if r[1] else '-') for m in M for r in [real[m][i]]))
for m in M:
    v = [r[1][0] / 5.899 for r in real[m] if r[1]]; w = [1000 * r[1][2] for r in real[m] if r[1]]
    print(f'  {m:18s} mean peak/true {np.mean(v):.4f}, spread (max-min) {100 * (max(v) - min(v)):.2f} %, mean width {np.mean(w):.0f} eV')

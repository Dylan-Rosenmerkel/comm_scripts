import json, glob, os, numpy as np, collections
agg=collections.defaultdict(list); real=collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results4', '*.json')):
    d=json.load(open(f))
    for r in d.get('inject',[]): agg[(d['ext'],r['method'],r['kind'])].append(r)
    for m,v in d['real'].items(): real[(d['ext'],m)]['n'].append(len(v['E'])); real[(d['ext'],m)]['split']+=v['split']
for (e,m,k),rs in sorted(agg.items()):
    print(f'EXT{e} {m:15s} {k:7s} N={len(rs):3d} one-cluster {np.mean([r["n"]==1 for r in rs]):.2f} med pieces {np.median([r["n"] for r in rs]):.0f} E_largest/E {np.median([r["frac_largest"] for r in rs]):.3f} E_all/E {np.median([r["frac"] for r in rs]):.3f}', 'resolved %.2f'%np.mean([r['resolved'] for r in rs]) if 'resolved' in rs[0] else '')
for (e,m),v in sorted(real.items()):
    s=np.array(v['split']); print(f'real EXT{e} {m:15s} clusters/img {np.mean(v["n"]):8.0f}  big WADERs clusters split>1 {np.mean(s>1):.2f}  in 0 pieces {np.mean(s==0):.2f}')

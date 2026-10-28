import sys, numpy as np
sys.path.insert(0, "."); import regress as R, harness as HX, bgv3
AM = HX.AM
J = R.jobs()
pick = [j for j in J if j[4].startswith("L2_")][:3:2] + [j for j in J if "DM04_DM05" in j[4] and "LBC" not in j[4]][:0] + \
       [j for j in J if "08-21-DM04" in j[4]][:2] + [j for j in J if "DM01_DM02" in j[4]][:1] + [j for j in J if j[4].startswith("Am241")][:2] + \
       [j for j in J if "DM24_DM25" in j[4]][:2] + [j for j in J if "High_Temp" in j[4]][:2]
def one(j):
    kind, run, stem, ext, tag = j
    if kind == "am":
        import am241_29img_recluster as RC; Q, valid = RC.load(stem, ext)
    else:
        Q, valid, _ = R.load_run(run, stem, ext)
    bg = AM.background_map(Q, valid); b3, rep = bgv3.regional_v3(Q, valid, bg)
    dm = float(np.mean(np.abs(b3.mean - bg.mean)[valid]))
    return f"{tag[:58]:58s} s {rep['s']:.2f} applied {rep['applied']!s:5s} bad rows {rep['bad_rows']:4d} cells changed {rep['cells_changed']:4d}/{rep['ncells']:4d}  mean |d bg| {dm:.3f} e-/pix  I0 {rep.get('I0', '-')}"
from multiprocessing import get_context
with get_context("spawn").Pool(8) as p:
    for line in p.imap(one, pick): print(line, flush=True)

"""Validation of superWADERs/run_superWADERs (the program, not the wip copies) on every data type on this machine,
plus the switch tests that were never run and the efficiency patch E1-E3. Meant to run unattended in the background.

    python run_validation.py OUT [--stages 1,2,3]

Stage 1 (suite): the cases of wip/native_pipeline/run.py (DM09, Fe-55 Image_8 DM07/27/28, PD06, AMtest, L2 ACM 103
         and its strips, injected few-electron events, binned, synthetic raw skips, single skip, std image) through
         run_superWADERs --clusterer auto (no g(n) curve, all CCDs: the program has no --curve / --ext), judged by
         run.py's own check(), and compared with that suite's earlier results (suite.json of the wip engine).
         DM09 also with --clusterer agmx, and through bin/run_WADERs --engine native --clusterer agmx: the clusters of
         the three must be identical (DM09 is chosen A_GMX everywhere).
Stage 2 (switch, in parallel with stage 1): wip/lowdc/switch_validation halo_test, boundary_test, real_l2_switch
         (live modules), 2 processes.
Stage 3 (efficiency): docs/patches/superWADERs_efficiency_E1-E3.patch in a git worktree; dm09, dm28, single_skip and
         amtest again; clusters and times against stage 1.
Results: OUT/report.json, OUT/report.md, logs per run. At most 4 heavy processes at a time.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = "/opt/homebrew/Caskroom/miniforge/base/envs/damic/bin/python"
sys.path.insert(0, os.path.join(REPO, "wip", "native_pipeline"))
import run as R          # noqa: E402  (cases, check, read_clusters, kalpha_per_ext of the earlier suite)

OLD_SUITE = os.path.join(R.OUT, "suite.json")
PATCH = os.path.join(REPO, "docs", "patches", "superWADERs_efficiency_E1-E3.patch")
SWV = os.path.join(REPO, "wip", "lowdc", "switch_validation")


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def prog_args(args):
    """the earlier suite's arguments without --curve X and --ext a b c (run_superWADERs has neither)"""
    out, skip = [], 0
    for i, a in enumerate(args):
        if skip:
            skip -= 1
            continue
        if a == "--curve":
            skip = 1
            continue
        if a == "--ext":
            while i + 1 + skip < len(args) and args[i + 1 + skip].isdigit():
                skip += 1
            continue
        out.append(a)
    return out


def run_prog(name, images, args, out, repo=REPO, extra=()):
    """one run of superWADERs/run_superWADERs; returns (rc, seconds, run folder)"""
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    lst = os.path.join(d, "images.txt")
    with open(lst, "w") as fh:
        fh.write("\n".join(images) + "\n")
    cmd = [PY, os.path.join(repo, "superWADERs", "run_superWADERs"), lst, "-o", d, "-j", "2", "--overwrite",
           "--no-review-helper"] + list(args) + list(extra)
    t0 = time.time()
    with open(os.path.join(d, "run.log"), "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    rd = os.path.join(d, "superWADERs_images")
    syn = os.path.join(d, "synthetic")                          # run.py's checks read the truth of synthetic cases there
    if not os.path.exists(syn) and os.path.isdir(os.path.join(R.OUT, "synthetic")):
        os.symlink(os.path.join(R.OUT, "synthetic"), syn)
    for f in glob.glob(os.path.join(rd, "root", "*.root")):          # check() looks for the files in the run folder
        link = os.path.join(rd, os.path.basename(f))
        if not os.path.exists(link):
            os.symlink(f, link)
    return rc, round(time.time() - t0, 1), rd


def clusters_of(rd):
    """{file name: (n, sorted energies)} of a run folder's cluster files"""
    out = {}
    for f in sorted(glob.glob(os.path.join(rd, "root", "panaSKImg_clustersRec_*_ext*.root"))):
        try:
            c = R.read_clusters(f)
            e = np.sort(np.asarray(c.get("Energy", []), float))
        except Exception as ex:                       # report and go on
            out[os.path.basename(f)] = (None, str(ex))
            continue
        out[os.path.basename(f)] = (int(e.size), e)
    return out


def compare_clusters(a, b):
    """identical / differences between two {file: (n, energies)}"""
    res = dict(files_a=len(a), files_b=len(b), only_a=sorted(set(a) - set(b)), only_b=sorted(set(b) - set(a)),
               identical=0, differ=[])
    for k in sorted(set(a) & set(b)):
        (na, ea), (nb, eb) = a[k], b[k]
        if na is None or nb is None:
            res["differ"].append(dict(file=k, error=str(ea if na is None else eb)))
        elif na == nb and np.allclose(ea, eb, rtol=1e-9, atol=1e-12):
            res["identical"] += 1
        else:
            m = min(na, nb)
            res["differ"].append(dict(file=k, n_a=na, n_b=nb, sum_a=float(np.sum(ea)), sum_b=float(np.sum(eb)),
                                      max_rel=float(np.max(np.abs(ea[:m] - eb[:m]) / np.maximum(np.abs(ea[:m]), 1e-9)))
                                      if m else None))
    return res


def summarise_case(rec):
    chains = {}
    for r in rec.get("ccds", []):
        chains[r["chain"]] = chains.get(r["chain"], 0) + 1
    return dict(chains=chains, n_clusters=sum(r.get("n_clusters") or 0 for r in rec.get("ccds", [])),
                problems=rec.get("problems", []), kalpha=rec.get("kalpha"))


def stage1(out, only=None):
    C = R.cases()
    old = {r["case"]: r for r in json.load(open(OLD_SUITE))} if os.path.exists(OLD_SUITE) else {}
    rep = {}
    for name, c in C.items():
        if only and name not in only:
            continue
        imgs = [p for p in c["images"] if os.path.exists(p)]
        if not imgs:
            rep[name] = dict(skipped="no input files")
            continue
        log(f"stage 1: {name} ({len(imgs)} image(s))")
        rc, sec, rd = run_prog(name, imgs, prog_args(c["args"]), os.path.join(out, "suite"))
        try:
            rec = R.check(name, c, rd)
        except Exception as ex:                       # report and go on
            rec = dict(problems=[f"check failed: {type(ex).__name__}: {ex}"], ccds=[])
        rec.update(rc=rc, seconds=sec, run=rd)
        new_s = summarise_case(rec)
        new_s["n_clusters"] = sum(n for n, _ in clusters_of(rd).values() if n)    # counted in the ROOT files
        old_s = summarise_case(old[name]) if name in old else None
        rep[name] = dict(rc=rc, seconds=sec, new=new_s, old=old_s, old_seconds=(old.get(name) or {}).get("seconds"),
                         fallbacks=rec.get("fallbacks", [])[:20], ccds=rec.get("ccds"))
        log(f"   rc {rc}, {sec} s, chains {new_s['chains']} (earlier {old_s['chains'] if old_s else '-'}), "
            f"clusters {new_s['n_clusters']} (earlier {old_s['n_clusters'] if old_s else '-'}), problems {len(new_s['problems'])}")
    # DM09: auto vs agmx (program) vs run_WADERs --engine native --clusterer agmx
    if not only or "dm09" in only:
        dm = R.cases()["dm09"]
        log("stage 1: dm09 --clusterer agmx (program)")
        rc, sec, rd_agmx = run_prog("dm09_agmx", dm["images"], [], os.path.join(out, "suite"), extra=["--clusterer", "agmx"])
        d_old = os.path.join(out, "suite", "dm09_run_WADERs")
        os.makedirs(d_old, exist_ok=True)
        lst = os.path.join(d_old, "DM09_Image5.txt")
        open(lst, "w").write("\n".join(dm["images"]) + "\n")
        log("stage 1: dm09 bin/run_WADERs --engine native --clusterer agmx")
        t0 = time.time()
        with open(os.path.join(d_old, "run.log"), "w") as fh:
            rc_old = subprocess.run([PY, os.path.join(REPO, "bin", "run_WADERs"), lst, "-o", d_old, "--engine", "native",
                                     "--clusterer", "agmx", "-j", "2", "--overwrite"], stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
        rd_old = os.path.join(d_old, "WADERs_analysis_DM09_Image5")
        old_root = {}
        for sub in ("root_agmx", "root"):          # run_WADERs writes A_GMX files to root_agmx/ (threshold ones to root/)
            if glob.glob(os.path.join(rd_old, sub, "*.root")):
                tmp = os.path.join(d_old, "as_root")
                os.makedirs(os.path.join(tmp, "root"), exist_ok=True)
                for f in glob.glob(os.path.join(rd_old, sub, "*.root")):
                    dst = os.path.join(tmp, "root", os.path.basename(f))
                    if not os.path.exists(dst):
                        os.symlink(f, dst)
                old_root = clusters_of(tmp)
                break
        a_auto = clusters_of(os.path.join(out, "suite", "dm09", "superWADERs_images"))
        a_agmx = clusters_of(rd_agmx)
        rep["dm09_identity"] = dict(auto_vs_agmx=compare_clusters(a_auto, a_agmx),
                                    program_agmx_vs_run_WADERs_native=compare_clusters(a_agmx, old_root),
                                    rc_program_agmx=rc, rc_run_WADERs=rc_old, seconds_run_WADERs=round(time.time() - t0, 1))
        log(f"   dm09 identity: auto vs agmx {rep['dm09_identity']['auto_vs_agmx']['identical']} identical, "
            f"agmx vs run_WADERs {rep['dm09_identity']['program_agmx_vs_run_WADERs_native']['identical']} identical")
    return rep


def stage2_start(out, tests=("halo_test.py", "boundary_test.py", "real_l2_switch.py")):
    """the three switch tests, one after the other, in a background process (2 workers)"""
    env = dict(os.environ, SWV_LIVE="1", SWV_SCRATCH=os.path.join(out, "swv_scratch"))
    script = " && ".join(f"{PY} {os.path.join(SWV, s)}" + (" --procs 2" if s != "real_l2_switch.py" else "")
                         for s in tests)
    fh = open(os.path.join(out, "stage2_switch.log"), "w")
    log("stage 2: switch tests started (halo, boundary, real L2)")
    return subprocess.Popen(["/bin/bash", "-c", f"cd {SWV} && {script}"], stdout=fh, stderr=subprocess.STDOUT, env=env)


def stage3(out, rep1):
    wt = os.path.join(out, "worktree_E1E3")
    if not os.path.isdir(wt):
        subprocess.run(["git", "-C", REPO, "worktree", "add", "--detach", wt, "HEAD"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "-C", wt, "apply", PATCH], check=True)
    rep = {}
    C = R.cases()
    for name in ("dm09", "dm28", "single_skip", "amtest"):
        c = C[name]
        imgs = [p for p in c["images"] if os.path.exists(p)]
        log(f"stage 3: {name} with E1-E3")
        rc, sec, rd = run_prog(name, imgs, prog_args(c["args"]), os.path.join(out, "patched"), repo=wt)
        base = os.path.join(out, "suite", name, "superWADERs_images")
        rep[name] = dict(rc=rc, seconds=sec, seconds_unpatched=(rep1.get(name) or {}).get("seconds"),
                         clusters=compare_clusters(clusters_of(base), clusters_of(rd)))
        d = rep[name]["clusters"]
        log(f"   rc {rc}, {sec} s (unpatched {rep[name]['seconds_unpatched']} s), {d['identical']} identical, "
            f"{len(d['differ'])} differ")
    return rep


def write_report(out, rep):
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    L = ["# superWADERs validation", ""]
    s1 = rep.get("stage1", {})
    if s1:
        L += ["## Stage 1: the program on every data type", "",
              "| case | rc | s (earlier) | chains now | chains earlier | clusters now | earlier | problems |", "|---|---|---|---|---|---|---|---|"]
        for k, v in s1.items():
            if "new" not in v:
                continue
            o = v["old"] or {}
            L.append(f"| {k} | {v['rc']} | {v['seconds']} ({v.get('old_seconds')}) | {v['new']['chains']} | {o.get('chains', '-')} | "
                     f"{v['new']['n_clusters']} | {o.get('n_clusters', '-')} | {'; '.join(v['new']['problems'])[:200]} |")
        if "dm09_identity" in s1:
            L += ["", "DM09 identity: " + json.dumps({k: (v if not isinstance(v, dict) else
                                                          dict(identical=v['identical'], differ=len(v['differ']),
                                                               only_a=len(v['only_a']), only_b=len(v['only_b'])))
                                                      for k, v in s1["dm09_identity"].items()})]
    if rep.get("stage2"):
        L += ["", "## Stage 2: switch tests", "", json.dumps(rep["stage2"])]
    if rep.get("stage3"):
        L += ["", "## Stage 3: efficiency patch E1-E3", ""]
        for k, v in rep["stage3"].items():
            d = v["clusters"]
            L.append(f"- {k}: {v['seconds']} s vs {v['seconds_unpatched']} s unpatched; {d['identical']} files identical, "
                     f"{len(d['differ'])} differ {d['differ'][:3]}")
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--stages", default="1,2,3")
    ap.add_argument("--only", nargs="*", help="stage 1: only these cases")
    ap.add_argument("--switch-tests", nargs="*", default=["halo_test.py", "boundary_test.py", "real_l2_switch.py"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = set(a.stages.split(","))
    rep = {}
    p2 = stage2_start(a.out, a.switch_tests) if "2" in st else None
    if "1" in st:
        rep["stage1"] = stage1(a.out, a.only)
        write_report(a.out, rep)
    if p2 is not None:
        rc2 = p2.wait()
        rep["stage2"] = dict(rc=rc2, results=sorted(os.path.basename(f) for f in glob.glob(os.path.join(SWV, "results", "*_*.csv"))
                                                    if os.path.getmtime(f) > time.time() - 6 * 3600))
        log(f"stage 2: done, rc {rc2}")
        write_report(a.out, rep)
    if "3" in st:
        rep["stage3"] = stage3(a.out, rep.get("stage1", {}))
        write_report(a.out, rep)
    log("all done: " + os.path.join(a.out, "report.md"))


if __name__ == "__main__":
    main()

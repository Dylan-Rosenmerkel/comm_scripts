#!/usr/bin/env python3
"""Write WADERs/bin/run_WADERs from comm_scripts' bin/run_WADERs.

The WADERs version keeps the command line, the configuration step, panaSKImg per CCD, --composite and the summary,
and does the comm_scripts steps inside panaSKImg instead of afterwards:
  --gain-correction    g(n) measured from the raw images BEFORE panaSKImg (gain_vs_charge), then
                       GainNonlinearityProcess in the final JSONs: root/ holds the corrected clusters
  --clusterer agmx     ClusterFinder algorithm "agmx" in the final JSONs (the --agmx-* flags as its options)
  --clusterer both     a second panaSKImg run per CCD with algorithm "agmx" -> root_agmx/
  --auto-threshold     ClusterFinder spr_threshold_auto (thresholds from the dark charge)
Modules come from pysimdamicm.comm (synced from comm_scripts by utils/sync_to_waders.py).

    python utils/make_waders_run_script.py          (re-run after changing bin/run_WADERs)
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "bin", "run_WADERs")
DST = os.path.join(ROOT, "WADERs", "bin", "run_WADERs")


def cut(text, start, end, new):
    """replace text[start marker .. end marker) by new (end marker kept)"""
    i, j = text.index(start), text.index(end)
    assert i < j, (start, end)
    return text[:i] + new + text[j:]


def sub(text, old, new, count=1):
    assert old in text, old[:80]
    return text.replace(old, new, count)


DOC = '''#!/usr/bin/env python3
"""run_WADERs (WADERs version): one command from raw skipper-CCD image(s) to panaSKImg (WADERs) results, with the
comm_scripts additions done inside panaSKImg.

    run_WADERs <image.fz | image.fits | image.fits.bz2 | list.txt> [-o PARENT] [-j N] [options]
    run_WADERs -h        the options;   run_WADERs -h -v   the full guide (what each step does)

What happens, per image (each step is one function below)
---------------------------------------------------------
  check_request          inputs exist, output folder free, option combinations valid (standard library only)
  ensure_environment     if this python lacks ROOT/pysimdamicm, re-run inside the conda environment with panaSKImg
  step1_configure        one panaSKImg JSON per CCD (pysimdamicm.comm.gain_dc_split + autoconfig_panaSKImg):
                         settings measured from the raw pixels; the gain from a many-peak fit where that is more
                         precise, the dark current from the Poisson-like 0e-/1e- peaks
  step1b_gain_curves     --gain-correction: g(n) measured from the raw images (gain_vs_charge; plots of g(n),
                         the nonlinearity and A(n) in gain_vs_charge/); GainNonlinearityProcess added to the JSONs
  step1c_cluster_options --clusterer, --agmx-*, --auto-threshold written into the final JSONs (ClusterFinder
                         algorithm / agmx / spr_threshold_auto); --clusterer both: a second set of JSONs (final_agmx/)
  step2_run_panaskimg    panaSKImg once per CCD (--ext N), in parallel (twice with --clusterer both)
  step3_collect_root_files   cluster files -> root/ (and root_agmx/), dark-current fits -> root/dc_fits/
  step3c_composite       --composite: MakeComposite_Fe55_Clusters_Extensions.cc on the cluster files
  step4_summary          fit_quality.csv and summary.txt (gain, noise, dark current per CCD with fit errors)

Output: WADERs_analysis_<run name>/   (run name = image name, or the .txt name for a list)
    *.pdf, *_waders.fits.bz2   panaSKImg plots and processed images
    json/<image>/final/        the JSONs that were run (one per CCD): they show every option used
    json/<image>/final_agmx/   with --clusterer both: the JSONs of the A_GMX runs
    json/<image>/gain/, dc/    the intermediate JSONs (many-peak gain / 0e-1e- fit)
    root/                      cluster files, one per CCD (panaSKImg_clustersRec_<image>_extN.root): gain-corrected
                               with --gain-correction, A_GMX clusters with --clusterer agmx
    root_agmx/                 with --clusterer both: the A_GMX cluster files
    root/root_files.txt        full paths of the cluster files (input list for MakeComposite)
    root/dc_fits/              the dark-current fits (histogram + fitted function) as ROOT
    gain_vs_charge/            with --gain-correction: g(n) models (*_gain_vs_charge.json) and plots
    logs/                      panaSKImg and autoconfig logs
    summary.txt, fit_quality.csv   gain, noise, dark current per CCD with fit errors

Environment: python with numpy, astropy, PyROOT and pysimdamicm (this WADERs copy); A_GMX also needs numba, scipy,
scikit-image. If the python running it lacks them, a conda environment with panaSKImg is found and used ("conda run").
Run from a WADERs checkout without installing it: the checkout is put first on PYTHONPATH automatically.
Made from comm_scripts' bin/run_WADERs by utils/make_waders_run_script.py.
"""
'''

NAMING = '''SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
# a WADERs checkout (bin/ next to pysimdamicm/): use it before any installed pysimdamicm, here and in the
# panaSKImg / conda subprocesses
WADERS_ROOT = os.path.dirname(SCRIPT_DIR)
if os.path.isdir(os.path.join(WADERS_ROOT, "pysimdamicm", "comm")):
    sys.path.insert(0, WADERS_ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join([WADERS_ROOT] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
NEEDED_MODULES = ("numpy", "astropy", "ROOT", "pysimdamicm")
INSIDE_FLAG = "RUN_WADERS_IN_ENV"


def _comm_file(name):
    """path of pysimdamicm/comm/<name> without importing pysimdamicm (its __init__ needs ROOT)"""
    spec = importlib.util.find_spec("pysimdamicm")
    base = os.path.dirname(spec.origin) if spec and spec.origin else os.path.join(WADERS_ROOT, "pysimdamicm")
    return os.path.join(base, "comm", name)


_spec = importlib.util.spec_from_file_location("_waders_naming", _comm_file("naming.py"))
_naming = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_naming)
acm_id, image_stem = _naming.acm_id, _naming.image_stem      # standard library only, usable before the environment check
'''

LOAD = '''def load_pipeline_modules(run):
    """Import the pipeline modules from pysimdamicm.comm (needs ROOT/pysimdamicm, so only after ensure_environment)
    and find panaSKImg (--panaskimg, else the one next to this script, else the one autoconfig_panaSKImg finds)."""
    from pysimdamicm.comm import autoconfig_panaSKImg as A
    from pysimdamicm.comm import fit_quality as FQ
    from pysimdamicm.comm import gain_dc_split as G
    run.A, run.FQ, run.G = A, FQ, G
    here = os.path.join(SCRIPT_DIR, "panaSKImg")
    run.panaskimg = run.args.panaskimg or (here if os.path.isfile(here) else A.DEFAULT_PANASKIMG)
    if not run.panaskimg or not os.path.isfile(run.panaskimg):
        sys.exit("run_WADERs: panaSKImg not found; give --panaskimg /path/to/panaSKImg")


'''

STEPS = '''def step2_run_panaskimg(run, configs, variant="final"):
    """Step 2: panaSKImg once per CCD (--ext N) with its JSON of json/<image>/<variant>/, all runs in parallel.
    variant "final" writes into the output folder; "final_agmx" (--clusterer both) into agmx_run/.

    CCDs flagged as flooded/unresolved by autoconfig are skipped unless --all-ext. A run counts as successful when
    its log contains "Finished correctly". Returns (tasks, failed): tasks = [(config, ext, json path)] that were run,
    failed = [(stem, ext, log path)].
    """
    args, D = run.args, run.dirs
    outdir = run.out if variant == "final" else os.path.join(run.out, "agmx_run")
    os.makedirs(outdir, exist_ok=True)
    tasks = []
    for img, r in configs.items():
        for ext, path, skip in r["finals"]:
            if skip and not args.all_ext:
                if variant == "final":
                    print(f"  {r['stem']} {ext}: skipped (unresolved outlier CCD; --all-ext to run it)")
                continue
            tasks.append((r, ext, path if variant == "final" else path.replace(os.sep + "final" + os.sep,
                                                                             os.sep + variant + os.sep)))

    def run_one(task):
        """panaSKImg on one CCD; returns (stem, ext, ok, log path)."""
        r, ext, path = task
        cfg = json.load(open(path))
        cmd = [sys.executable, run.panaskimg, r["src"], "-j", path, "-o", outdir + "/", "--save-plots", "--ext", ext[3:]]
        if cfg["input"]["image"].get("ACM_multi_extension", False):
            cmd.append("--acm")
        tag = "" if variant == "final" else "_agmx"
        log = os.path.join(D["logs"], f"{r['stem']}_{ext}_panaSKImg{tag}.log")
        with open(log, "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        with open(log) as fh:
            ok = "Finished correctly" in fh.read()
        return r["stem"], ext, ok, log

    print(f"  running panaSKImg{' (A_GMX)' if variant != 'final' else ''}: {len(tasks)} CCD run(s)")
    failed = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for fut in cf.as_completed([ex.submit(run_one, t) for t in tasks]):
            stem, ext, ok, log = fut.result()
            print(f"  {run.stamp()} {stem} {ext}: {'done' if ok else 'FAILED, see ' + log}")
            if not ok:
                failed.append((stem, ext, log))
    return tasks, failed


def step3_collect_root_files(run, variant="final"):
    """Step 3: move panaSKImg's cluster files to root/ (root_agmx/ for the A_GMX runs of --clusterer both) and its
    dark-current fits to root/dc_fits/, and write root_files.txt (full paths, the input list for e.g.
    MakeComposite). Returns the sorted cluster files."""
    out, D = run.out, run.dirs
    src = out if variant == "final" else os.path.join(out, "agmx_run")
    dst = D["root"] if variant == "final" else os.path.join(out, "root_agmx")
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(src, "panaSKImg_clustersRec_*.root")):
        shutil.move(f, os.path.join(dst, os.path.basename(f)))
    for f in glob.glob(os.path.join(src, "*_FitDarkCurrentProcess_side*.root")):
        if variant == "final":
            shutil.move(f, os.path.join(D["fits"], os.path.basename(f)))
        else:
            os.remove(f)                    # the same fit as the main run
    roots = sorted(glob.glob(os.path.join(dst, "panaSKImg_clustersRec_*.root")))
    with open(os.path.join(dst, "root_files.txt"), "w") as fh:
        fh.write("".join(p + "\\n" for p in roots))
    if variant != "final":
        D["root_agmx"] = dst
    return roots


def measure_curve(run, GVC, rs, name, title, run_exts):
    """Measure g(n) from the raw images rs (gain_vs_charge.measure), save it as
    gain_vs_charge/<name>_gain_vs_charge.json (with the original image paths) and its plots
    (<name>_gain_vs_charge.png, <name>_nonlinearity.png, <name>_adu_of_n.png). Returns (json path, result per CCD)."""
    res = GVC.measure([r["src"] for r in rs], run_exts, verbose=False, model=run.args.gain_model,
                      combine=run.args.gain_combine, autoconfig_args=shlex.split(run.args.autoconfig_args),
                      memory=run.args.gain_memory)
    mpath = os.path.join(run.dirs["gc"], f"{name}_gain_vs_charge.json")
    with open(mpath, "w") as fh:
        json.dump({"images": [r["orig"] for r in rs], "e_per_electron_eV": GVC.E_PER_E, "extensions": res}, fh, indent=1)
    GVC.plot(res, os.path.join(run.dirs["gc"], f"{name}_gain_vs_charge.png"), title)
    GVC.plot_nonlinearity(res, os.path.join(run.dirs["gc"], f"{name}_nonlinearity.png"), title)
    GVC.plot_adu(res, os.path.join(run.dirs["gc"], f"{name}_adu_of_n.png"), title)
    return mpath, res


def report_curve(gc_lines, title, res):
    """Append one summary line per CCD describing a measured g(n) curve (or why there is none) to gc_lines."""
    for ext, rr in sorted(res.items()):
        if rr.get("ok"):
            m = rr["model"]
            gc_lines.append(f"  {title} {ext}: g(0) = {m['a']:.4f} ADU/e- (autoconfig comb gain {rr['g_1e']:.4f}), "
                            f"{rr['drop_at_1000'] * 100:.1f}% lower at 1000 e-, measured to "
                            f"{rr['n_max_measured']:.0f} e-"
                            + (f", nonlinearity {rr['nonlinearity']['c2']:.2e} x^2 {rr['nonlinearity']['c1']:+.4f} x"
                               if rr.get("nonlinearity") else "")
                            + (f" -- module curve: own comb not usable ({rr.get('own_reason')})"
                               if rr.get("source") == "module" else "")
                            + (f" [readout memory removed, kL {rr['memory']['kL']:.2e}]" if rr.get("memory") else ""))
        else:
            gc_lines.append(f"  {title} {ext}: no correction ({rr.get('reason')})")


def _edit_finals(configs, edit, variant="final"):
    """Apply edit(cfg dict, r, ext) to every final JSON (or to a copy in json/<image>/<variant>/)."""
    for r in configs.values():
        for ext, path, skip in r["finals"]:
            cfg = json.load(open(path))
            edit(cfg, r, ext)
            dst = path if variant == "final" else path.replace(os.sep + "final" + os.sep, os.sep + variant + os.sep)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as fh:
                json.dump(cfg, fh, indent=4)


def step1b_gain_curves(run, configs):
    """Step 1b, --gain-correction: measure g(n) from the RAW images and put GainNonlinearityProcess into the final
    JSONs (after CalibrationProcess), so that panaSKImg writes gain-corrected cluster energies (root/).

    Why: the gain falls slowly with the charge in a pixel (~3 % at 1000 e-), so keV clusters come out low when every
    pixel is converted with the low-charge gain. g(n) is the spacing of the electron comb at high charge.
    Images of one module (ACM id in the name) share the CCDs:
      pooled     one curve per module from all its images
      per-image  one curve per image (the module curve where the image's own comb is not usable)
    Returns gc_lines (summary lines); sets run.dirs["gc"].
    """
    args, D = run.args, run.dirs
    gc_lines = []
    if not args.gain_correction:
        return gc_lines
    from pysimdamicm.comm import gain_vs_charge as GVC
    D["gc"] = os.path.join(run.out, "gain_vs_charge")
    os.makedirs(D["gc"], exist_ok=True)
    run_exts = sorted({int(ext[3:]) for r in configs.values() for ext, _, _ in r["finals"]})
    groups, parent, period_lines = module_groups(run, configs, GVC)      # modules, split into gain periods
    model_of = {}                       # image stem -> model file
    gc_lines.append(f"gain vs charge per pixel (--gain-correction {args.gain_correction}, --gain-model {args.gain_model}, "
                    f"--gain-anchor {args.gain_anchor}): measured from the raw images, applied inside panaSKImg by "
                    f"GainNonlinearityProcess (root/ holds the corrected clusters):")
    gc_lines += period_lines
    whole = {}                          # module -> curve from all its images (fallback for a period's unusable CCDs)
    for key, rs in sorted(groups.items()):
        print(f"  {run.stamp()} g(n) for module {key} ({len(rs)} image(s))")
        mpath, res = measure_curve(run, GVC, rs, f"module_{key}", f"module {key}", run_exts)
        if key != parent[key] and not all(rr.get("ok") for rr in res.values()):
            m = parent[key]
            if m not in whole:
                print(f"  {run.stamp()} g(n) for module {m}, all periods (for the CCDs without a period curve)")
                whole[m] = measure_curve(run, GVC, [r for k, v in groups.items() if parent[k] == m for r in v],
                                         f"module_{m}", f"module {m}", run_exts)[1]
            fallback_curve(mpath, [r["orig"] for r in rs], res, whole[m], "module", GVC.E_PER_E)
        report_curve(gc_lines, f"module {key}", res)
        if args.gain_correction == "pooled":
            model_of.update({r["stem"]: mpath for r in rs})
            continue
        for r in sorted(rs, key=lambda r: r["stem"]):
            print(f"  {run.stamp()} g(n) for image {r['stem']}")
            ipath, ires = measure_curve(run, GVC, [r], f"image_{r['stem']}", r["stem"], run_exts)
            for ext, rr in ires.items():       # fallback to the module curve, recorded in the file
                rr["source"] = "image"
                mod = res.get(ext, {})
                if not rr.get("ok") and mod.get("ok"):
                    ires[ext] = dict(mod, source="module", own_reason=rr.get("reason"))
            with open(ipath, "w") as fh:
                json.dump({"images": [r["orig"]], "e_per_electron_eV": GVC.E_PER_E, "extensions": ires}, fh, indent=1)
            report_curve(gc_lines, r["stem"], ires)
            model_of[r["stem"]] = ipath

    def edit(cfg, r, ext):
        p = cfg["process"]
        seq = [s for s in p["sequence"].split(";") if s and s != "GainNonlinearityProcess"]
        k = seq.index("CalibrationProcess") + 1 if "CalibrationProcess" in seq else len(seq)
        seq.insert(k, "GainNonlinearityProcess")
        p["sequence"] = ";".join(seq)
        p["GainNonlinearityProcess"] = {"model_file": model_of[r["stem"]], "anchor": args.gain_anchor,
                                        "temperature": args.temperature}
    _edit_finals(configs, edit)
    return gc_lines


def step1c_cluster_options(run, configs):
    """Step 1c: the clustering options in the final JSONs (ClusterFinder):
      --auto-threshold      spr_threshold_auto (thresholds in e- from each CCD's dark charge)
      --clusterer agmx      algorithm "agmx" with the --agmx-* options (and --temperature)
      --clusterer both      as waders in final/, and a copy with algorithm "agmx" in final_agmx/
    Returns summary lines."""
    args = run.args
    lines = []
    if args.no_clusters:
        return lines

    def agmx_opts(cfg):
        return {"crowded": args.agmx_crowded, "crowded_shapes": args.agmx_crowded_shapes,
                "fe_candidates": os.environ.get("AGMX_FE_CANDIDATES", "1") != "0", "temperature": args.temperature,
                "first_row": cfg["input"]["image"].get("id_row_start", 0)}

    def edit(cfg, r, ext, algorithm):
        cf_ = cfg["process"].setdefault("ClusterFinder", {})
        cf_["algorithm"] = algorithm
        if algorithm == "agmx":
            cf_["agmx"] = agmx_opts(cfg)
        if args.auto_threshold:
            cf_["spr_threshold_auto"] = True
    main = "agmx" if args.clusterer == "agmx" else "waders"
    _edit_finals(configs, lambda c, r, e: edit(c, r, e, main))
    if args.clusterer == "both":
        _edit_finals(configs, lambda c, r, e: edit(c, r, e, "agmx"), variant="final_agmx")
    lines.append(f"clustering (ClusterFinder, set in json/<image>/final/): algorithm {main}"
                 + (f" (crowded {args.agmx_crowded}, shapes {args.agmx_crowded_shapes})" if main == "agmx" else "")
                 + (", thresholds from the dark charge (spr_threshold_auto)" if args.auto_threshold else ""))
    if args.clusterer == "both":
        lines.append(f"  --clusterer both: A_GMX runs with json/<image>/final_agmx/ (crowded {args.agmx_crowded}, "
                     f"shapes {args.agmx_crowded_shapes}) -> root_agmx/ (recommended: agmx_blend < 0.02, agmx_ped < 10)")
    elif main == "agmx":
        lines.append("  A_GMX cluster files in root/ (recommended: agmx_blend < 0.02, agmx_ped < 10)")
    return lines


'''

COMPOSITE_OLD = '''    comp_sets = [(D.get("root_gc", D["root"]), os.path.join(run.out, "composite"), "")]
    if len(gc_dirs) == 2:           # --gain-correction both: also the per-image corrected files
        comp_sets.append((gc_dirs["per-image"], os.path.join(run.out, "composite_per_image"), "_per_image"))
    if agmx_dirs:                   # --clusterer agmx: the A_GMX files instead; both: in addition
        a_sets = [(agmx_dirs.get("pooled", agmx_dirs.get("per-image", agmx_dirs["plain"])), "composite", "")]
        if "pooled" in agmx_dirs and "per-image" in agmx_dirs:
            a_sets.append((agmx_dirs["per-image"], "composite_per_image", "_per_image"))
        if args.clusterer == "both":
            comp_sets += [(d, os.path.join(run.out, c + "_agmx"), t + "_agmx") for d, c, t in a_sets]
        else:
            comp_sets = [(d, os.path.join(run.out, c), t) for d, c, t in a_sets]
'''
COMPOSITE_NEW = '''    comp_sets = [(D["root"], os.path.join(run.out, "composite"), "")]       # root/: as configured (corrected, A_GMX)
    if D.get("root_agmx"):          # --clusterer both: also the A_GMX files
        comp_sets.append((D["root_agmx"], os.path.join(run.out, "composite_agmx"), "_agmx"))
'''

MAIN_OLD_START = "    configs = step1_configure(run)                                  # one JSON per CCD\n"
MAIN_OLD_END = "    if not args.keep_work:\n"
MAIN_NEW = '''    configs = step1_configure(run)                                  # one JSON per CCD
    gc_lines = step1b_gain_curves(run, configs)                     # --gain-correction: g(n) -> JSONs
    if args.no_clusters:
        drop_cluster_finding(configs)
    agmx_lines = step1c_cluster_options(run, configs)               # --clusterer, --agmx-*, --auto-threshold
    tasks, failed = step2_run_panaskimg(run, configs)               # panaSKImg per CCD
    roots = step3_collect_root_files(run)                           # cluster files -> root/
    if args.clusterer == "both" and not args.no_clusters:
        _, failed_a = step2_run_panaskimg(run, configs, "final_agmx")   # A_GMX runs -> root_agmx/
        step3_collect_root_files(run, "final_agmx")
        shutil.rmtree(os.path.join(run.out, "agmx_run"), ignore_errors=True)
        failed += [(s, e + " (A_GMX)", lg) for s, e, lg in failed_a]
    comp_lines, comp_failed = step3c_composite(run, configs, {}, None)    # --composite
    step4_summary(run, configs, roots, failed, gc_lines, comp_lines, comp_failed, agmx_lines, ())

'''


def main():
    text = open(SRC).read()
    # module docstring and the comm_scripts src path
    text = cut(text, "#!/usr/bin/env python3", "# Only the standard library at module level", DOC)
    text = cut(text, "SCRIPT_PATH = os.path.realpath(__file__)", "RESERVED_AUTOCONFIG", NAMING)
    # help texts
    text = sub(text, "  root_agmx/, root_agmx_gaincorr/   with --clusterer agmx/both: A_GMX cluster files",
               "  root_agmx/                   with --clusterer both: the A_GMX cluster files\n"
               "  gain_vs_charge/              with --gain-correction: g(n) models and plots (g(n), nonlinearity, A(n))")
    text = cut(text, "--gain-correction (optional): the gain falls slowly", "  --gain-model interpolated (default)",
               "--gain-correction pooled|per-image (optional): the gain falls slowly with the charge in a pixel\n"
               "  (~3% at 1000 e-), so cluster energies come out low at keV energies. With this flag, BEFORE the\n"
               "  panaSKImg runs, gain_vs_charge measures g(n) from the raw images (spacing of the electron comb at\n"
               "  high charge, clean rows only; images of the same module together), and the final JSONs get\n"
               "  GainNonlinearityProcess (after CalibrationProcess): panaSKImg's ClusterFinder corrects every\n"
               "  cluster's pixel energies with n = A^-1(ADU), A(n) = integral of g (the correction of comm_scripts'\n"
               "  gain_vs_charge apply). root/ then holds the corrected clusters; run without the flag for the\n"
               "  uncorrected ones. Curves and plots in gain_vs_charge/: <name>_gain_vs_charge.png (g(n)),\n"
               "  <name>_nonlinearity.png (A(n)/g(0) - n) and <name>_adu_of_n.png (A(n) and A(n) - g(0) n).\n"
               "  --gain-anchor comb (default) uses the comb's whole g(n) including its low-charge gain; fit keeps\n"
               "  each image's fitted gain at low charge. --gain-combine incoherent (default) | coherent: how the\n"
               "  images of a module are combined. --gain-memory auto: remove the horizontal readout memory before\n"
               "  measuring. --temperature K: energy per electron epsilon(T) of the corrected and A_GMX energies\n"
               "  (default 130 K -> 3.7486 eV).\n")
    text = cut(text, "  --gain-correction per-image: every image gets its own curve", "--composite (optional)",
               "  --gain-correction per-image: every image gets its own curve (the module curve for a CCD whose\n"
               "  own comb is not usable).\n\n")
    text = cut(text, "--clusterer waders|agmx|both (default waders)", "settings you can pass with --autoconfig-args",
               "--clusterer waders|agmx|both (default waders): ClusterFinder's algorithm, set in the final JSONs.\n"
               "  waders  threshold + connect (panaSKImg's ClusterFinder as always).\n"
               "  agmx    A_GMX (model-based clustering that shares each pixel's charge; pysimdamicm/comm/adaptive.py):\n"
               "          root/ holds A_GMX cluster files (same trees, plus agmx_* branches and agmx_config).\n"
               "          Recommended for X-ray lines: agmx_blend < 0.02 and agmx_ped < 10 with the usual sigma_xy cut.\n"
               "  both    waders in root/, and a second panaSKImg run per CCD with A_GMX (json/<image>/final_agmx/)\n"
               "          -> root_agmx/; --composite runs on both (composite/, composite_agmx/).\n"
               "  --agmx-crowded auto|on|off, --agmx-crowded-shapes on|off: A_GMX crowded mode and its muon lines and\n"
               "          topology split (ClusterFinder agmx options crowded / crowded_shapes).\n"
               "--auto-threshold: WADERs pixel and seed thresholds in electrons from each CCD's dark charge\n"
               "  (ClusterFinder spr_threshold_auto: smallest k >= 2 with P(Poisson(lambda) >= k) <= 0.10, seed\n"
               "  max(9.5, k + 0.5) e-); CCDs whose electron peaks are not resolved keep the configured thresholds.\n\n")
    text = cut(text, "  run_WADERs images.txt --gain-correction        also gain-corrected cluster files",
               "\nenvironment variables:",
               "  run_WADERs images.txt --gain-correction        gain-corrected cluster files (g(n) inside panaSKImg)\n"
               "  run_WADERs images.txt --auto-threshold         WADERs thresholds from the dark charge\n"
               "  run_WADERs images.txt --gain-correction --clusterer both --composite\n"
               "                                                 WADERs and A_GMX clusters, both through MakeComposite\n")
    # options
    text = sub(text, 'ap.add_argument("--gain-correction", nargs="?", const="pooled", choices=("pooled", "per-image", "both"),',
               'ap.add_argument("--gain-correction", nargs="?", const="pooled", choices=("pooled", "per-image"),')
    text = re.sub(r'help="measure the gain vs charge per pixel g\(n\) from the images and write gain-corrected "\n'
                  r'[^\n]*\n[^\n]*\n[^\n]*comparison; details: -h -v\)"\)',
                  'help="measure the gain vs charge per pixel g(n) from the raw images and correct the cluster "\n'
                  '                         "energies inside panaSKImg (GainNonlinearityProcess). MODE: pooled (default: one curve "\n'
                  '                         "per module) or per-image (details: -h -v)")', text, count=1)
    assert "per module) or per-image (details: -h -v)" in text
    text = re.sub(r'help="waders \(default\): panaSKImg\'s ClusterFinder only\. agmx: also A_GMX cluster files in the same "\n'
                  r'[^\n]*\(details: -h -v\)"\)',
                  'help="ClusterFinder algorithm in the final JSONs: waders (default; threshold + connect), agmx "\n'
                  '                         "(A_GMX clusters in root/) or both (waders in root/, A_GMX in root_agmx/; details: -h -v)")',
                  text, count=1)
    assert "A_GMX in root_agmx/; details" in text
    text = text.replace("A_GMX crowded mode (clustering_tests/crowded.py)", "A_GMX crowded mode (ClusterFinder agmx option crowded)")
    text = text.replace("(clustering_tests/crowded_shapes.py; default on; only on crowded CCDs)",
                        "(ClusterFinder agmx option crowded_shapes; default on; only on crowded CCDs)")
    text = sub(text, '    ap.add_argument("--temperature", type=float, default=None, metavar="K",',
               '    ap.add_argument("--auto-threshold", action="store_true",\n'
               '                    help="WADERs pixel/seed thresholds in e- from each CCD\'s dark charge (ClusterFinder "\n'
               '                         "spr_threshold_auto; details: -h -v)")\n'
               '    ap.add_argument("--temperature", type=float, default=None, metavar="K",')
    # the --gain-correction both comparison is not part of this version
    text = cut(text, "def front_kalpha(files):", "@dataclass", "")
    text = cut(text, "def load_pipeline_modules(run):", "def prepare_output(run):", LOAD)
    text = cut(text, "def step2_run_panaskimg(run, configs):", "def step3c_composite(", STEPS)
    text = sub(text, COMPOSITE_OLD, COMPOSITE_NEW)
    text = cut(text, MAIN_OLD_START, MAIN_OLD_END, MAIN_NEW)
    text = sub(text, "sys.exit(1 if failed or comp_failed or agmx_failed else 0)", "sys.exit(1 if failed or comp_failed else 0)")
    with open(DST, "w") as fh:
        fh.write(text)
    os.chmod(DST, 0o755)
    print("wrote", os.path.relpath(DST, ROOT))


if __name__ == "__main__":
    main()

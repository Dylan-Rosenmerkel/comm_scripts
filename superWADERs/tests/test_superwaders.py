"""superWADERs: independence from the rest of comm_scripts, and the command line.

    python -m unittest discover -s superWADERs/tests -v        (from the comm_scripts folder)
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import unittest

SW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMD = os.path.join(SW, "run_superWADERs")
MODULE_DIRS = ("engine", "clustering", "commissioning")


def modules():
    """every module of superWADERs (engine/, clustering/, commissioning/; not the copied third-party scripts that
    need extra packages: raw_levels' viewers, lsm_defects' Image_4, compare_dcfit_vs_skips' pypdf)"""
    out = []
    for d in MODULE_DIRS:
        for f in sorted(glob.glob(os.path.join(SW, d, "*.py"))):
            name = os.path.basename(f)[:-3]
            if name not in ("compare_dcfit_vs_skips",):
                out.append(name)
    return out


class Independence(unittest.TestCase):
    def test_imports_come_from_superwaders(self):
        """Every superWADERs module, and everything they import from comm_scripts, is loaded from superWADERs/
        (never from src/, clustering_tests/ or wip/), even with those folders first on the path."""
        cs = os.path.dirname(SW)
        code = f"""
import importlib, json, os, sys
for d in ("src", "clustering_tests", "wip/lowdc/bayes", "wip/native_pipeline"):
    sys.path.insert(0, os.path.join({cs!r}, d))                # the copies elsewhere first: must not be used
sys.path.insert(0, os.path.join({SW!r}, "engine"))
import native_pipeline                                           # sets superWADERs' own path
bad, failed = {{}}, {{}}
for m in {modules()!r}:
    try:
        importlib.import_module(m)
    except ImportError as e:
        failed[m] = str(e)
for name, mod in list(sys.modules.items()):
    f = getattr(mod, "__file__", None) or ""
    if f.startswith({cs!r}) and not f.startswith({SW!r}):
        bad[name] = f
print(json.dumps(dict(bad=bad, failed=failed)))
"""
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout.strip().splitlines()[-1])
        missing = {m: e for m, e in res["failed"].items() if "ROOT" not in e}
        self.assertEqual(missing, {}, "modules that cannot be imported")
        self.assertEqual(res["bad"], {}, "modules loaded from outside superWADERs/")

    def test_no_reference_to_run_wader_or_src(self):
        """No superWADERs code starts or imports bin/run_WADERs, or reads files from src/, clustering_tests/, wip/."""
        for f in glob.glob(os.path.join(SW, "**", "*.py"), recursive=True) + [CMD]:
            if "/tests/" in f:
                continue
            with open(f) as fh:
                text = "".join(ln for ln in fh if not ln.lstrip().startswith("#"))
            for pat in ('"run_WADERs"', "'run_WADERs'", '"src")', '"clustering_tests")', '"wip"'):
                self.assertNotIn(pat, text, f"{f}: {pat}")


class CommandLine(unittest.TestCase):
    def run_cmd(self, *args):
        return subprocess.run([sys.executable, CMD, *args], capture_output=True, text=True)

    def test_help(self):
        r = self.run_cmd("-h")
        self.assertEqual(r.returncode, 0)
        self.assertIn("--clusterer {auto,agmx,threshold}", r.stdout)
        r = self.run_cmd("-h", "-v")
        self.assertIn("auto's chains", r.stdout)

    def test_default_clusterer_is_auto(self):
        sys.path.insert(0, SW)
        import runpy
        ns = runpy.run_path(CMD, run_name="not_main")
        self.assertEqual(ns["build_parser"]().parse_args(["x.fits"]).clusterer, "auto")

    def test_bad_requests(self):
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "avg_Image_5_x_106_20260101_000000_1.fits")
            open(img, "w").close()
            for args, msg in [(["--thresholds", "x"], "--thresholds"),
                              (["--autoconfig-args", "--gain 1.3"], "cannot be given"),
                              (["--composite", "--no-clusters"], "--no-clusters"),
                              (["--packet", "--dc-series"], "--packet")]:
                r = self.run_cmd(img, "-o", d, *args)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(msg, r.stderr)
            r = self.run_cmd(os.path.join(d, "missing.fits"))
            self.assertIn("not found", r.stderr)


if __name__ == "__main__":
    unittest.main()

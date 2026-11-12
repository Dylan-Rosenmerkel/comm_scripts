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
                              (["--packet", "--dc-series"], "--packet"),
                              (["--campaign", os.path.join(d, "newcamp"), "--overwrite"], "--overwrite"),
                              (["--criteria", os.path.join(d, "nope.json")], "--criteria")]:
                r = self.run_cmd(img, "-o", d, *args)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(msg, r.stderr)
            r = self.run_cmd(os.path.join(d, "missing.fits"))
            self.assertIn("not found", r.stderr)


class Campaign(unittest.TestCase):
    """The campaign store and the verdicts (commissioning/campaign.py)."""

    def setUp(self):
        for d in MODULE_DIRS:
            p = os.path.join(SW, d)
            if p not in sys.path:
                sys.path.insert(0, p)
        import campaign
        self.CA = campaign

    def test_judge(self):
        sec = dict(metrics=dict(noise=dict(label="noise", unit="e-", warn=0.2, fail=0.3),
                                gain_dev=dict(label="gain dev", unit="%", warn=3, fail=8, abs=True),
                                show=dict(label="shown", warn=None, fail=None),
                                low=dict(label="low", warn=10, fail=5, direction="below")),
                   flag_levels=dict(dead="fail", info="note", gone="off"))
        J = self.CA.judge
        self.assertEqual(J(dict(noise=0.1, gain_dev=-2, show=99, low=20), sec)[0], "pass")
        v, r, lv = J(dict(noise=0.25, gain_dev=-9, low=7), sec)
        self.assertEqual(v, "fail")
        self.assertEqual([x[1] for x in r], ["gain_dev", "noise", "low"])   # fail first
        self.assertEqual(lv, dict(noise="warn", gain_dev="fail", low="warn"))
        self.assertEqual(J(dict(noise=0.1), sec, [("info", "warn", "a note", "info")])[0], "pass")   # note: no effect
        self.assertEqual(J(dict(noise=0.1), sec, [("gone", "fail", "dropped", "gone")])[1], [])       # off: dropped
        self.assertEqual(J(dict(noise=0.1), sec, [("dead", "warn", "no charge", "dead")])[0], "fail")  # level from file
        self.assertEqual(J({}, sec, status="no CCD signal")[0], "na")

    def test_ledger_and_records(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "a.fits")
            open(f, "w").write("x")
            c = self.CA.Campaign(os.path.join(d, "camp"))
            self.assertFalse(c.current(f))
            c.add(f, "images", kind="image", stem="a")
            c.save_record("images", "a", dict(stem="a", x=[1.5, float("nan")]))
            c.save()
            c2 = self.CA.Campaign(os.path.join(d, "camp"))
            self.assertTrue(c2.has(f, "images"))
            self.assertFalse(c2.has(f, "sr"))
            self.assertEqual(c2.load_record("images", "a"), dict(stem="a", x=[1.5, None]))
            self.assertEqual(c2.stems("images"), ["a"])
            open(f, "w").write("changed")                    # a changed file is new again
            os.utime(f, ns=(1, 1))
            self.assertFalse(c2.current(f))

    def test_criteria_copy(self):
        with tempfile.TemporaryDirectory() as d:
            crit = self.CA.load_criteria(d)
            self.assertIn("images", crit)
            self.assertTrue(os.path.exists(os.path.join(d, "criteria.json")))
            for t in ("images", "packet", "dc_series", "defects", "sr"):
                self.assertIn("metrics", crit[t])

    def test_overview_page(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o", "index.html")
            row = dict(module="DM01", acm=101, ext="EXT1", ccd="CCD_A", verdict="warn", values=dict(noise=0.25),
                       reasons=[["warn", "noise", "noise 0.25 > 0.2", "noise"]], levels=dict(noise="warn"))
            self.CA.write_overview(path, dict(title="t", criteria=dict(metrics=dict(noise=dict(label="noise", warn=0.2))),
                                              views=[dict(name="v", ccds=[row])]))
            with open(path) as fh:
                page = fh.read()
            self.assertIn('"module":"DM01"', page)
            self.assertNotIn("__DATA__", page)


if __name__ == "__main__":
    unittest.main()

"""Tests of bin/run_WADERs that need no data and no panaSKImg: input lists, module names, the command line.

Run all tests from the comm_scripts folder with the python that has the pipeline's packages:
    python -m unittest discover -s tests -v
"""
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "bin", "run_WADERs")


def load_run_wader():
    """Import bin/run_WADERs (a script without .py) as a module; only the standard library is needed at import."""
    loader = importlib.machinery.SourceFileLoader("run_wader", SCRIPT)
    spec = importlib.util.spec_from_loader("run_wader", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


RW = load_run_wader()


class ReadInputs(unittest.TestCase):
    """read_inputs: one image, or a .txt list with comments, blank lines and relative paths."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        for name in ("a.fz", "b.fits.bz2"):
            open(os.path.join(self.dir, name), "w").close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_image(self):
        run_name, images = RW.read_inputs(os.path.join(self.dir, "a.fz"))
        self.assertEqual(run_name, "a")
        self.assertEqual(images, [os.path.join(self.dir, "a.fz")])

    def test_list_with_comments_and_relative_paths(self):
        lst = os.path.join(self.dir, "my_run.txt")
        with open(lst, "w") as fh:
            fh.write("# a comment\n\na.fz\n  b.fits.bz2  \n")
        run_name, images = RW.read_inputs(lst)
        self.assertEqual(run_name, "my_run")
        self.assertEqual(images, [os.path.join(self.dir, "a.fz"), os.path.join(self.dir, "b.fits.bz2")])

    def test_missing_file_in_list_stops(self):
        lst = os.path.join(self.dir, "bad.txt")
        with open(lst, "w") as fh:
            fh.write("does_not_exist.fz\n")
        with self.assertRaises(SystemExit):
            RW.read_inputs(lst)

    def test_unknown_extension_stops(self):
        path = os.path.join(self.dir, "notes.csv")
        open(path, "w").close()
        with self.assertRaises(SystemExit):
            RW.read_inputs(path)


class ModuleNames(unittest.TestCase):
    """--module ACM=NAME parsing and the module-name rule used by --composite."""

    def test_parse_modules(self):
        self.assertEqual(RW.parse_modules(["109=DM02", " 110 = DM01 "]), {"109": "DM02", "110": "DM01"})
        with self.assertRaises(SystemExit):
            RW.parse_modules(["109"])

    def test_from_folder_name(self):
        src = "/data/2025-11-13-DM01_DM02_DM03_Low_Temp/avg_Image_5_Low_Temp_109_20251113_164748_6.fz"
        # ACM 110, 109, 106 are the first, second and third module of a DMa_DMb_DMc folder
        self.assertEqual(RW.module_name("110", src, {}), ("DM01", "folder name"))
        self.assertEqual(RW.module_name("109", src, {}), ("DM02", "folder name"))
        self.assertEqual(RW.module_name("106", src, {}), ("DM03", "folder name"))

    def test_override_and_fallback(self):
        src = "/data/AMTEst/avg_Image_8_Low_Temp_106_20260330_123052_21.fz"
        self.assertEqual(RW.module_name("106", src, {"106": "AMT"}), ("AMT", "--module"))
        self.assertEqual(RW.module_name("106", src, {}), ("ACM106", "no module found"))
        self.assertEqual(RW.module_name(None, src, {}), ("DMXX", "no module found"))


class CommandLine(unittest.TestCase):
    """The script starts and prints its help with any python (no ROOT needed for -h)."""

    def run_help(self, *extra):
        return subprocess.run([sys.executable, SCRIPT, "-h", *extra], capture_output=True, text=True)

    def test_short_help(self):
        r = self.run_help()
        self.assertEqual(r.returncode, 0)
        self.assertIn("usage: run_WADERs INPUT", r.stdout)

    def test_full_guide(self):
        r = self.run_help("-v")
        self.assertEqual(r.returncode, 0)
        self.assertIn("--gain-correction", r.stdout)
        self.assertIn("--composite", r.stdout)

    def test_composite_and_no_clusters_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "x.fz"); open(img, "w").close()
            r = subprocess.run([sys.executable, SCRIPT, img, "-o", d, "--composite", "--no-clusters"],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("cannot be used with --no-clusters", r.stderr)

    def run_checked(self, files, *extra, lst=None):
        """run_WADERs on dummy inputs; returns the CompletedProcess (stops in check_request, before any work)"""
        with tempfile.TemporaryDirectory() as d:
            for f in files:
                os.makedirs(os.path.dirname(os.path.join(d, f)), exist_ok=True)
                open(os.path.join(d, f), "w").close()
            inp = os.path.join(d, files[0])
            if lst is not None:
                inp = os.path.join(d, "l.txt")
                with open(inp, "w") as fh:
                    fh.write("\n".join(lst) + "\n")
            return subprocess.run([sys.executable, SCRIPT, inp, "-o", os.path.join(d, "out"), *extra],
                                  capture_output=True, text=True)

    def test_jobs_must_be_positive(self):
        r = self.run_checked(["x.fz"], "-j", "0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("at least 1", r.stderr)

    def test_duplicate_image_names_stop(self):
        r = self.run_checked(["a.fz", "sub/a.fits.bz2"], lst=["a.fz", "sub/a.fits.bz2"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("same name", r.stderr)

    def test_reserved_autoconfig_args_stop(self):
        r = self.run_checked(["x.fz"], "--autoconfig-args", "--peaks 2")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--peaks", r.stderr)


class Naming(unittest.TestCase):
    """src/naming.py: the image name and ACM id used everywhere"""

    def test_stem_and_acm(self):
        sys.path.insert(0, os.path.join(REPO, "src"))
        import naming
        self.assertEqual(naming.image_stem("/x/avg_Image_8_Low_Temp_106_20260330_123052_21.fits.bz2"),
                         "avg_Image_8_Low_Temp_106_20260330_123052_21")
        self.assertEqual(naming.acm_id("avg_Image_8_Low_Temp_106_20260330_123052_21"), "106")
        self.assertIsNone(naming.acm_id("some_image"))


class RunContext(unittest.TestCase):
    def test_stamp_format(self):
        run = RW.Run(args=None, run_name="r", images=[], out="/tmp/x", modules={})
        run.t0 = 0.0
        self.assertRegex(run.stamp(), r"^\[\s*\d+s\]$")


if __name__ == "__main__":
    unittest.main()

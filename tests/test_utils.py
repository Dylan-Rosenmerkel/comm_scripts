"""utils/compare_runs.py on two fake run_WADERs folders (no ROOT files, no data needed).

    python -m unittest discover -s tests -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPARE = os.path.join(REPO, "utils", "compare_runs.py")


def fake_run(parent, extra=None):
    """WADERs_analysis_x/ with a summary holding its own absolute path, a final JSON and a g(n) file"""
    out = os.path.join(parent, "WADERs_analysis_x")
    os.makedirs(os.path.join(out, "json", "img", "final"))
    os.makedirs(os.path.join(out, "gain_vs_charge"))
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(f"run_WADERs summary: x  (1 image(s), 12 s)\ncluster files: 0 (list: {out}/root/root_files.txt)\n")
    with open(os.path.join(out, "fit_quality.csv"), "w") as fh:
        fh.write("image,ext,gain\nimg,EXT1,1.3\n")
    with open(os.path.join(out, "json", "img", "final", "img_EXT1.json"), "w") as fh:
        fh.write('{"a": 1}')
    with open(os.path.join(out, "gain_vs_charge", "module_1_gain_vs_charge.json"), "w") as fh:
        json.dump({"images": [out], "extensions": {"EXT1": {"ok": True}}}, fh)
    for rel in extra or []:
        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
        with open(os.path.join(out, rel), "w") as fh:
            fh.write('{"extensions": {}}')
    return out


class CompareRuns(unittest.TestCase):
    def compare(self, a, b, cwd=None):
        return subprocess.run([sys.executable, COMPARE, a, b, "-q"], capture_output=True, text=True, cwd=cwd)

    def test_identical_with_relative_paths(self):
        with tempfile.TemporaryDirectory() as d:
            fake_run(os.path.join(d, "old")); fake_run(os.path.join(d, "new"))
            sub = os.path.join(d, "sub"); os.makedirs(sub)
            r = self.compare("../old", "../new", cwd=sub)          # relative paths with ..
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("IDENTICAL", r.stdout)

    def test_file_only_in_new_is_a_difference(self):
        with tempfile.TemporaryDirectory() as d:
            fake_run(os.path.join(d, "old"))
            fake_run(os.path.join(d, "new"), extra=["gain_vs_charge/image_y_gain_vs_charge.json"])
            r = self.compare(os.path.join(d, "old"), os.path.join(d, "new"))
            self.assertEqual(r.returncode, 1)
            self.assertIn("only in NEW", r.stdout)

    def test_changed_json_is_a_difference(self):
        with tempfile.TemporaryDirectory() as d:
            fake_run(os.path.join(d, "old")); b = fake_run(os.path.join(d, "new"))
            with open(os.path.join(b, "json", "img", "final", "img_EXT1.json"), "w") as fh:
                fh.write('{"a": 2}')
            r = self.compare(os.path.join(d, "old"), os.path.join(d, "new"))
            self.assertEqual(r.returncode, 1)
            self.assertIn("DIFF  json/img/final/img_EXT1.json", r.stdout)


if __name__ == "__main__":
    unittest.main()

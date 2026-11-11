"""ROOT output of the native pipeline: panaSKImg-format files written from numpy arrays.

PyROOT is imported only here and only when a file is written (import ~2 s). The branches are created from Python
with PyROOT's std::vector objects (as pysimdamicm's OutputDataManager does: that path gets ROOT's interpreted
collection proxy for vector<vector<...>>, while a compiled TTree::Branch<T> would need a generated dictionary); the
vectors are FILLED by small C++ helpers (declared once per process) that copy whole numpy arrays, so no Python loop
over clusters or pixels touches ROOT. Each tree gets ONE entry, as panaSKImg writes (one entry per image, every
branch a vector with one element per cluster).

Branch kinds: "I" int scalar (/I), "D" double scalar (/D), "S" std::string, "VF" vector<float>, "VI" vector<int>,
"VVF" vector<vector<float>>, "VVI" vector<vector<int>> (the last two from (flat array, offsets)).
"""
import numpy as np

_CPP = r"""
#include <vector>
namespace npw {
void vf(std::vector<float>& v, const double* a, long n) { v.assign(a, a + n); }
void vi(std::vector<int>& v, const long* a, long n) { v.resize(n); for (long i = 0; i < n; ++i) v[i] = (int)a[i]; }
void vvf(std::vector<std::vector<float>>& v, const double* a, const long* off, long nseg) {
  v.resize(nseg); for (long k = 0; k < nseg; ++k) v[k].assign(a + off[k], a + off[k + 1]); }
void vvi(std::vector<std::vector<int>>& v, const long* a, const long* off, long nseg) {
  v.resize(nseg);
  for (long k = 0; k < nseg; ++k) { v[k].resize(off[k + 1] - off[k]);
    for (long i = off[k]; i < off[k + 1]; ++i) v[k][i - off[k]] = (int)a[i]; } }
}
"""
_ROOT = None


def root():
    """PyROOT in batch mode, with the npw helpers declared (once per process)."""
    global _ROOT
    if _ROOT is None:
        import ROOT
        ROOT.gROOT.SetBatch(True)
        ROOT.gInterpreter.Declare(_CPP)
        _ROOT = ROOT
    return _ROOT


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _i64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def write_file(path, trees, named=None):
    """Write a ROOT file. trees: [(tree name, [(branch, kind, value), ...])]; named: {name: text} written as
    TNamed objects (e.g. the configuration as JSON). Every tree gets one entry."""
    R = root()
    V = R.std.vector
    f = R.TFile(path, "RECREATE")
    keep = []                                   # buffers must live until Fill()
    for tname, branches in trees:
        f.cd()
        t = R.TTree(tname, tname)
        for name, kind, v in branches:
            if kind == "I":
                buf = np.array([int(v)], dtype=np.int32)
                t.Branch(name, buf, f"{name}/I")
            elif kind == "D":
                buf = np.array([float(v)], dtype=np.float64)
                t.Branch(name, buf, f"{name}/D")
            elif kind == "S":
                buf = R.std.string(str(v))
                t.Branch(name, buf)
            elif kind == "VF":
                buf = V("float")(); a = _f64(v); R.npw.vf(buf, a, a.size); t.Branch(name, buf)
            elif kind == "VI":
                buf = V("int")(); a = _i64(v); R.npw.vi(buf, a, a.size); t.Branch(name, buf)
            elif kind in ("VVF", "VVI"):
                flat, off = v
                o = _i64(off)
                if kind == "VVF":
                    buf = V(V("float"))(); R.npw.vvf(buf, _f64(flat), o, o.size - 1)
                else:
                    buf = V(V("int"))(); R.npw.vvi(buf, _i64(flat), o, o.size - 1)
                t.Branch(name, buf)
            else:
                raise ValueError(f"unknown branch kind {kind!r} ({name})")
            keep.append(buf)
        t.Fill()
        t.Write()
        keep.append(t)
    for k, text in (named or {}).items():
        f.cd()
        R.TNamed(k, text).Write()
    f.Close()
    del keep

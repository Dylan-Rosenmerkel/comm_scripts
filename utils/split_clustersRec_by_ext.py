#!/usr/bin/env python3
"""Split panaSKImg cluster files made with --acm (all extensions stacked in one image,
file name ..._ext1234.root) into one file per extension (..._ext1.root, ..._ext2.root, ...).

Scripts such as MakeComposite_Fe55_Clusters_Extensions.cc expect one file per extension
and take the extension from "_ext<N>" in the file name.

With --acm the extensions are stacked along rows: extension N holds rows
(N-1)*nrows+1 ... N*nrows (rows are 1-based in the clustersRec tree). Every cluster is
assigned to the extension of its pixels, and all row (y) coordinates are shifted so each
extension starts at row 1 again. All branches are kept; the info and process_config trees
are copied unchanged (info already has one set of ext<N>_* branches per extension).

Usage:
    python split_clustersRec_by_ext.py file_list.txt -o outdir [--nrows 500] [--list out.txt]
    python split_clustersRec_by_ext.py a_ext1234.root b_ext1234.root -o outdir

Extensions without clusters are not written. Needs PyROOT.
"""
import argparse
import os
import re
import sys

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gInterpreter.Declare(r"""
#include <vector>
namespace splitext {
// keep only the elements idx of v (in that order)
template <class T> void keep(std::vector<T>& v, const std::vector<int>& idx) {
    std::vector<T> out; out.reserve(idx.size());
    for (int i : idx) out.push_back(v[i]);
    v.swap(out);
}
// add dy to every row coordinate (flat or per-pixel vectors)
template <class T> void shift(std::vector<T>& v, double dy) { for (auto& x : v) x += dy; }
template <class T> void shift(std::vector<std::vector<T>>& v, double dy) { for (auto& p : v) for (auto& x : p) x += dy; }
// extension (1-based) of each cluster from its first pixel row; -1 if it has no pixels
std::vector<int> ext_of(const std::vector<std::vector<int>>& pixels_y, int nrows) {
    std::vector<int> e; e.reserve(pixels_y.size());
    for (auto& p : pixels_y) e.push_back(p.empty() ? -1 : (p[0] - 1) / nrows + 1);
    return e;
}
}
""")

# branches holding row (y) coordinates, shifted back to the extension's own rows
Y_BRANCHES = {"PosY", "fPosY", "meanY", "minY", "maxY", "QmaxY", "pixels_y",
              "projectionY_rows", "track_endpoint1_Y", "track_endpoint2_Y"}


def split_file(path, outdir, nrows):
    """Write one file per extension with clusters; return their paths."""
    fin = ROOT.TFile.Open(path)
    if not fin or fin.IsZombie():
        sys.exit(f"cannot open {path}")
    tin = fin.Get("clustersRec")
    others = [fin.Get(n) for n in ("info", "process_config") if fin.Get(n)]

    # per-cluster branches are std::vector with one element per cluster
    tin.GetEntry(0)
    ncl = tin.Nclusters
    vec_branches = [b.GetName() for b in tin.GetListOfBranches()
                    if b.GetClassName().startswith("vector") and len(getattr(tin, b.GetName())) == ncl]
    exts = list(ROOT.splitext.ext_of(tin.pixels_y, nrows))
    nfound = sorted(set(e for e in exts if e > 0))

    # Nclusters is a scalar branch: bind it to a buffer we can change before Fill
    import array
    nbuf = array.array("i", [0])
    tin.SetBranchAddress("Nclusters", nbuf)

    outs = []
    base = os.path.basename(path)
    for e in nfound:
        tin.GetEntry(0)                      # fresh copy of the vectors for this extension
        idx = ROOT.std.vector("int")([i for i, x in enumerate(exts) if x == e])
        dy = -(e - 1) * nrows
        for name in vec_branches:
            v = getattr(tin, name)
            ROOT.splitext.keep(v, idx)
            if name in Y_BRANCHES and dy != 0:
                ROOT.splitext.shift(v, dy)
        nbuf[0] = len(idx)

        outname = re.sub(r"_ext\d+\.root$", f"_ext{e}.root", base)
        if outname == base:
            outname = base.replace(".root", f"_ext{e}.root")
        outpath = os.path.join(outdir, outname)
        fout = ROOT.TFile(outpath, "RECREATE")
        tout = tin.CloneTree(0)              # same branches and types, shares the buffers
        tout.Fill()
        tout.Write()
        for t in others:
            t.CloneTree(-1).Write()
        fout.Close()
        outs.append(outpath)
        print(f"  ext{e}: {len(idx):6d} clusters -> {outpath}")
    fin.Close()
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="..._ext1234.root files, or a .txt list with one path per line")
    ap.add_argument("-o", "--outdir", required=True, help="folder for the per-extension files (created if missing)")
    ap.add_argument("--nrows", type=int, default=500, help="rows per extension in the stacked image (default 500)")
    ap.add_argument("--list", default=None,
                    help="write the paths of the new files to this .txt (default: <outdir>/root_files_per_ext.txt)")
    args = ap.parse_args()

    files = []
    for p in args.inputs:
        if p.endswith(".txt"):
            with open(p) as fh:
                files += [l.strip() for l in fh if l.strip()]
        else:
            files.append(p)
    os.makedirs(args.outdir, exist_ok=True)

    written = []
    for p in files:
        print(os.path.basename(p))
        written += split_file(p, args.outdir, args.nrows)

    listfile = args.list or os.path.join(args.outdir, "root_files_per_ext.txt")
    with open(listfile, "w") as fh:
        fh.write("\n".join(os.path.abspath(w) for w in written) + "\n")
    print(f"{len(written)} files listed in {listfile}")


if __name__ == "__main__":
    main()

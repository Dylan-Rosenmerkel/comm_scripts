#!/usr/bin/env python3
"""Tile panaSKImg per-extension plots side by side, ordered by number of skips.

Looks one level down in an analysis folder (e.g. analysis/1, analysis/10, ...)
for files like

    avg_img_250x1000x500_1x1_L2_101_20261014_223416_121_FitDarkCurrentProcess_DCfit_EXT1.pdf

groups them by extension (EXT1..EXTn), sorts each group by the number of
skips, and writes one grid PDF per extension, e.g. FitDarkCurrentProcess_DCfit_EXT1_vs_skips.pdf.
Pages are merged as vector graphics, so plots stay sharp when zoomed.

Requires: pypdf  (pip install pypdf)
"""
import argparse
import math
import os
import re
import sys
from collections import defaultdict
from glob import glob

from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, RectangleObject

# avg_img_<rows>x<cols>x<skips>_... ; the third number is the number of skips
SKIPS_RE = re.compile(r"_\d+x\d+x(\d+)_")
EXT_RE = re.compile(r"_(EXT\d+)\.pdf$")


def find_plots(indir, plot):
    """Return {ext: [(nskips, path), ...]} for the plots in the subfolders of indir."""
    groups = defaultdict(dict)
    for path in sorted(glob(os.path.join(indir, "*", f"*_{plot}_EXT*.pdf"))):
        name = os.path.basename(path)
        m_ext = EXT_RE.search(name)
        if not m_ext:
            continue
        m_skp = SKIPS_RE.search(name)
        if m_skp:
            nskips = int(m_skp.group(1))
        elif os.path.basename(os.path.dirname(path)).isdigit():
            nskips = int(os.path.basename(os.path.dirname(path)))
        else:
            print(f"WARNING: cannot get number of skips from {path}, skipping it")
            continue
        ext = m_ext.group(1)
        # if the same (ext, skips) appears twice, keep the most recent file
        old = groups[ext].get(nskips)
        if old is not None:
            keep = path if os.path.getmtime(path) >= os.path.getmtime(old) else old
            print(f"WARNING: two {ext} plots for {nskips} skips, using {keep}")
            path = keep
        groups[ext][nskips] = path
    return {ext: sorted(d.items()) for ext, d in groups.items()}


def label_page(width, height, labels, font_size=16):
    """Blank page with each (text, x_center, y_baseline) label drawn in Helvetica-Bold."""
    page = PageObject.create_blank_page(width=width, height=height)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica-Bold"),
    })
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/FLbl"): font})})
    ops = []
    for text, xc, y in labels:
        x = xc - 0.55 * font_size * len(text) / 2  # approximate centering
        ops.append(f"BT /FLbl {font_size} Tf {x:.2f} {y:.2f} Td ({text}) Tj ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode())
    page[NameObject("/Contents")] = stream
    return page


def make_grid(entries, outfile, ncols, label_h=24.0, gap=6.0):
    """Write a single-page PDF with the first page of each entry tiled in a grid."""
    pages = [PdfReader(p).pages[0] for _, p in entries]
    for p in pages:
        # ROOT writes landscape plots as rotated portrait pages; bake the rotation in
        p.transfer_rotation_to_content()
    # use the largest visible (cropped) plot area as the cell size
    cw = max(float(p.cropbox.width) for p in pages)
    ch = max(float(p.cropbox.height) for p in pages)
    ncols = max(1, min(ncols, len(pages)))
    nrows = math.ceil(len(pages) / ncols)
    cell_w, cell_h = cw + gap, ch + label_h + gap
    W, H = ncols * cell_w + gap, nrows * cell_h + gap

    writer = PdfWriter()
    out = writer.add_blank_page(width=W, height=H)
    labels = []
    for i, ((nskips, path), page) in enumerate(zip(entries, pages)):
        r, c = divmod(i, ncols)
        x0 = gap + c * cell_w
        y0 = H - (r + 1) * cell_h  # bottom of this cell
        pw, ph = float(page.cropbox.width), float(page.cropbox.height)
        s = min(cw / pw, ch / ph)
        # move the source page origin to (0,0), scale it, then place it in the cell
        t = (Transformation()
             .translate(-float(page.cropbox.left), -float(page.cropbox.bottom))
             .scale(s, s)
             .translate(x0 + (cw - pw * s) / 2, y0 + (ch - ph * s) / 2))
        out.merge_transformed_page(page, t)
        labels.append((f"{nskips} skips", x0 + cw / 2, y0 + ch + 6))
    out.merge_page(label_page(W, H, labels))
    out.mediabox = RectangleObject((0, 0, W, H))

    with open(outfile, "wb") as fh:
        writer.write(fh)


def main():
    ap = argparse.ArgumentParser(
        description="Tile panaSKImg per-extension plots from analysis/<skips>/ subfolders "
                    "into one PDF per extension, ordered by number of skips.")
    ap.add_argument("indir", nargs="?", default=".",
                    help="analysis folder containing the per-skip subfolders (default: .)")
    ap.add_argument("-o", "--output", required=True,
                    help="output folder for the combined PDFs (created if missing)")
    ap.add_argument("--ncols", type=int, default=4,
                    help="number of plots per row (default: 4)")
    ap.add_argument("--plot", default="FitDarkCurrentProcess_DCfit",
                    help="plot name between the image name and _EXTn.pdf "
                         "(default: FitDarkCurrentProcess_DCfit; e.g. "
                         "PedestalSubtractionProcess_Baseline or PedestalSubtractionProcess_Sigma)")
    args = ap.parse_args()

    groups = find_plots(args.indir, args.plot)
    if not groups:
        sys.exit(f"No '*_{args.plot}_EXT*.pdf' files found in subfolders of {os.path.abspath(args.indir)}")

    os.makedirs(args.output, exist_ok=True)
    for ext in sorted(groups, key=lambda e: int(e[3:])):
        entries = groups[ext]
        outfile = os.path.join(args.output, f"{args.plot}_{ext}_vs_skips.pdf")
        make_grid(entries, outfile, args.ncols)
        print(f"{ext}: {len(entries)} plots (skips {', '.join(str(n) for n, _ in entries)}) -> {outfile}")


if __name__ == "__main__":
    main()

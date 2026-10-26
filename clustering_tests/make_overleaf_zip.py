"""Zip docs/aGMX (LaTeX source, generated tables, figures) for Overleaf -> docs/aGMX/aGMX_clustering_overleaf.zip"""
import glob, os, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "..", "docs", "aGMX")
out = os.path.join(A, "aGMX_clustering_overleaf.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("aGMX_clustering.tex", "perf_tables.tex", "lines_flags_table.tex", "fe55_flags_tables.tex", "contam_table.tex", "audit_2026-10.tex", "phase2_2026-10.tex"):
        if os.path.exists(os.path.join(A, f)):
            z.write(os.path.join(A, f), f)
    for f in sorted(glob.glob(os.path.join(A, "figures", "*.png"))):
        z.write(f, os.path.join("figures", os.path.basename(f)))
print(os.path.abspath(out), f"{os.path.getsize(out) / 1e6:.1f} MB")

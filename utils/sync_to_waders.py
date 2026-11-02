#!/usr/bin/env python3
"""Copy the comm_scripts modules that WADERs uses into WADERs/pysimdamicm/comm/.

The copies are the comm_scripts files unchanged except for their imports of each other, which become
package-relative (`import gmx as GX` -> `from . import gmx as GX`, `from naming import x` ->
`from .naming import x`). Run after changing any of them in comm_scripts:

    python utils/sync_to_waders.py          (prints the files written)
    python utils/sync_to_waders.py --check  (exit 1 if a copy is out of date)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "WADERs", "pysimdamicm", "comm")
FILES = {
    # g(n): measurement, correction, plots; and what it needs (autoconfig's geometry and gain search)
    "src/gain_vs_charge.py", "src/autoconfig_panaSKImg.py", "src/fit_quality.py", "src/naming.py",
    "src/pair_energy.py", "src/gain_dc_split.py",
    # A_GMX clustering (adaptive.adaptive_cluster, engine 'gmx', crowded mode and shapes)
    "clustering_tests/adaptive.py", "clustering_tests/gmx.py", "clustering_tests/crowded.py",
    "clustering_tests/crowded_shapes.py", "clustering_tests/agent_methods.py", "clustering_tests/mhc.py",
    "clustering_tests/cluster_methods.py", "clustering_tests/fe_candidates.py", "clustering_tests/methods2.py",
    # A_GMX clusters as pysimdamicm Cluster objects (codes of the agmx_* branches, crowded-mode record)
    "src/agmx_root.py",
}
MODULES = {os.path.splitext(os.path.basename(f))[0] for f in FILES}
HEADER = ('# Copied from comm_scripts/{src} by utils/sync_to_waders.py (imports made package-relative).\n'
          '# Edit the comm_scripts file and re-run the sync, not this copy.\n')


def convert(text):
    mods = "|".join(sorted(MODULES, key=len, reverse=True))
    text = re.sub(rf"^(\s*)import ({mods}) as (\w+)", r"\1from . import \2 as \3", text, flags=re.M)
    text = re.sub(rf"^(\s*)import ({mods})\s*$", r"\1from . import \2", text, flags=re.M)
    text = re.sub(rf"^(\s*)from ({mods}) import ", r"\1from .\2 import ", text, flags=re.M)
    return text


def main():
    check = "--check" in sys.argv
    os.makedirs(DEST, exist_ok=True)
    stale = []
    for src in sorted(FILES):
        with open(os.path.join(ROOT, src)) as fh:
            text = fh.read()
        lines = text.split("\n", 1)
        if lines[0].startswith("#!"):                     # keep a shebang first
            out = lines[0] + "\n" + HEADER.format(src=src) + convert(lines[1] if len(lines) > 1 else "")
        else:
            out = HEADER.format(src=src) + convert(text)
        dst = os.path.join(DEST, os.path.basename(src))
        old = open(dst).read() if os.path.exists(dst) else None
        if old != out:
            stale.append(dst)
            if not check:
                with open(dst, "w") as fh:
                    fh.write(out)
    init = os.path.join(DEST, "__init__.py")
    if not os.path.exists(init) and not check:
        with open(init, "w") as fh:
            fh.write('"""Tools from comm_scripts used by WADERs: gain vs charge g(n) (gain_vs_charge), A_GMX clustering\n'
                     '(adaptive, gmx, crowded, crowded_shapes, ...) and the panaSKImg autoconfiguration\n'
                     '(autoconfig_panaSKImg, gain_dc_split). The modules are synced from comm_scripts by\n'
                     'utils/sync_to_waders.py."""\n')
    for s in stale:
        print(("out of date: " if check else "wrote ") + os.path.relpath(s, ROOT))
    if check and stale:
        sys.exit(1)


if __name__ == "__main__":
    main()

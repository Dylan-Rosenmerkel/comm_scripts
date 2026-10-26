# docs/ — where to start

| read this | for |
|---|---|
| **`usage/comm_scripts_usage.pdf`** (source `.tex`, Overleaf `_overleaf.zip`) | **start here**: everything the repository can do and how to run each part |
| `PIPELINE_GUIDE.md` | the physics behind each pipeline step and a map of the code (read before changing the pipeline) |
| `aGMX/aGMX_clustering.pdf` (source `.tex`, Overleaf zip) | the A_GMX clustering algorithm step by step, its performance against WADERs, the quality flags |
| `clustering_methods_report.pdf` | every clustering method of the 2026-10-23..25 study and every test |
| `gain_correction_method/gain_correction_method.tex` (+ Overleaf zip) | the gain-versus-charge g(n) method note |
| `gain_correction_report.pdf`, `gain_correction_Am241.pdf` | effect of `--gain-correction` on Fe-55 (production data) and on the Am-241 lines (made by `report_scripts/`) |
| `am241_note/` (+ `AMtest_Am241_10-30keV_note_overleaf.zip`) | the Am-241 10-30 keV note: scripts (README there) and LaTeX source |
| **`notes/HANDOFF_2026-10-25_agmx_phase1-2_audit.md`** | **latest handoff: phase 1 done, phase 2 + physics-audit fixes in progress, next steps** |
| `notes/OPEN_ITEMS.md` | everything still open, in one list |
| `notes/CONTEXT_2026-10-24.md` | current hand-over of the clustering study (A_GMX, flags, pipeline integration) |
| `notes/CONTEXT_2026-10-16.md` | ground rules, environment and decisions (sections 1-3 still apply) |
| `notes/HANDOFF_2026-10-23_clustering.md`, `notes/PROGRESS_2026-10-23.md` | historical session notes |
| `patches/pysimdamicm_x_max_plot.patch` | the local pysimdamicm change behind `--plot-peaks` |
| `data/` | small result files quoted by the notes |

Generated files (PDFs, Overleaf zips) are rebuilt from their sources by the scripts named in each document; after
editing a `.tex`, rebuild its PDF and zip so they do not drift apart.

# Copied from comm_scripts/src/naming.py by utils/sync_to_waders.py (imports made package-relative).
# Edit the comm_scripts file and re-run the sync, not this copy.
"""Image names: the one place that turns an image path into the name every output is prefixed with, and an image
name into its ACM (module board) id. Standard library only (run_WADERs uses it before the conda environment is
known)."""
import os
import re

# ..._<ACM id>_<YYYYMMDD>_<HHMMSS>_<image number>, e.g. avg_Image_8_Low_Temp_106_20260330_123052_21 -> ACM 106
ACM_RE = re.compile(r"_(\d+)_\d{8}_\d{6}_\d+$")


def image_stem(path):
    """Image name without folder and extensions ("a/b/img_6.fits.bz2" -> "img_6"); prefixes every output."""
    return os.path.basename(path).split(".")[0]


def acm_id(stem):
    """ACM id in an image name (string), or None if the name does not follow the DAQ pattern."""
    m = ACM_RE.search(stem)
    return m.group(1) if m else None

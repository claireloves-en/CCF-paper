"""
Where the encoded data files live (Revision_Plan_V06.md, Stage 2A).

rebuild_encoded.py replaced fraudTrain_encoded{,_client}.csv on 2026-10-12 with the two
encoding bugs fixed, and moved the old files to ../_old/ with a _V24 suffix. Every result
committed before then was computed from the old files. To reproduce one, set

    CCF_ENCODED_VERSION=V24

and the scripts that read the encoded files read the archived ones instead (verify_restored.py
needs this). Unset, they read the current, corrected files.
"""
import os
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent
VERSION = os.environ.get("CCF_ENCODED_VERSION", "")


def data_file(name):
    """Path of a data file; an encoded file resolves to the archived copy when
    CCF_ENCODED_VERSION is set."""
    if VERSION and name.startswith("fraudTrain_encoded"):
        p = PATH / "_old" / name.replace(".csv", f"_{VERSION}.csv")
        assert p.exists(), f"{p} not found"
        return p
    return PATH / name

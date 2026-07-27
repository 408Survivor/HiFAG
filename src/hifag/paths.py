"""Single point of sys.path management for HiFAG.

Every entry script (train.py, test.py, scripts/*) should import this module
first. It makes `hifag` importable and exposes AFGNN's `src/` so its
models/data/utils packages can be reused without copying code.
"""

import os
import sys

HIFAG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIFAG_SRC = os.path.join(HIFAG_ROOT, "src")
AFGNN_SRC = "/home/ltq/DepressionCode/DepGNN/AFGNN/src"

# HiFAG's own package takes priority; AFGNN is appended as fallback.
if HIFAG_SRC not in sys.path:
    sys.path.insert(0, HIFAG_SRC)
if AFGNN_SRC not in sys.path:
    sys.path.append(AFGNN_SRC)

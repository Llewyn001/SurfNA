#!/usr/bin/env python3
"""Generate the SO(3) and torsion lookup tables locally, without model weights."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from surfna.runtime import activate

if __name__ == '__main__':
    activate('generator')
    # The inherited implementations generate and cache missing tables on import.
    from utils import so3, torus
    print('Diffusion tables are stored in', ROOT/'reproducibility/diffusion_tables')

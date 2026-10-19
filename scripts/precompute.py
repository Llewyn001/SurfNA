#!/usr/bin/env python3
"""Build data-independent SO(3)/torsion probability lookup tables once on CPU."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from surfna.runtime import activate
activate('generator')
from utils import so3,torus
print('Diffusion probability lookup tables are ready.')

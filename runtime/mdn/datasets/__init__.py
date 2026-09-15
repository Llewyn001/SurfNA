"""SurfNA V2 dataset overlay.

Keep development overrides in this package while resolving unchanged dataset
modules from the main SurfNA V2 source tree on ``PYTHONPATH``.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

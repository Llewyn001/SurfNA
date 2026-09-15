"""SurfNA V2 utility overlay with fallback to the main source tree."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

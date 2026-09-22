"""Apple AMX register files, instructions and rewrite rules for Exo."""
from pathlib import Path

from . import ops, pools, rewrites
from .ops import *
from .pools import *
from .rewrites import *

__version__ = "0.1.0"
__all__ = ["include_dir", *ops.__all__, *pools.__all__, *rewrites.__all__]

def include_dir():
  """The directory holding amx.h, to add to the C compiler's include path."""
  return str(Path(__file__).parent)

from exo.core.memory import MemGenError, StaticMemory

class _AMXState:
  """AMX_SET / AMX_CLR bookkeeping shared by the X, Y and Z pools: the state is
  set up on the first allocation and torn down once every pool is empty again."""

  def __init__(self):
    self.pools = []
    self.header_emitted = False
    self.is_active = False

  def header(self):
    if self.header_emitted: return ""
    self.header_emitted = True
    return '#include "amx.h"'

  def set_if_inactive(self):
    if self.is_active: return ""
    self.is_active = True
    return "AMX_SET();\n"

  def clr_if_all_free(self):
    if any(pool.row_dict for pool in self.pools): return ""
    self.is_active = False
    for pool in self.pools: pool.init_state(pool.NUM_ROWS)
    return "\nAMX_CLR();"

_amx = _AMXState()

class _APPLE_AMX_POOL(StaticMemory):
  """Base class for the three AMX register files.

  Every register is a 64-byte row. A 1D Exo buffer occupies one row and a 2D
  buffer occupies one row per matrix row; the second dimension must fill the
  row exactly. Buffers compile to a `#define name <first row>` and instruction
  operands are computed from that base row (see `window`).
  """

  CTYPE_BYTES = {
    "_Float16": 2, "float": 4, "double": 8,
    "int8_t": 1, "uint8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4,
  }
  ROW_BYTES = 64
  NUM_ROWS = 0  # set by the concrete pools

  def __init_subclass__(cls, **kw):
    super().__init_subclass__(**kw)
    cls.row_dict = {}
    if cls.NUM_ROWS:
      cls.init_state(cls.NUM_ROWS)
      _amx.pools.append(cls)

  @classmethod
  def global_(cls):
    return _amx.header()

  @classmethod
  def can_read(cls):
    return False

  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    prefix = _amx.set_if_inactive()
    dims = [int(d) for d in shape]
    if not dims or dims[-1] * cls.CTYPE_BYTES[prim_type] != cls.ROW_BYTES:
      raise MemGenError(
        f"{srcinfo}: {cls.__name__} rows must be exactly {cls.ROW_BYTES} bytes, "
        f"got {prim_type}[{', '.join(shape)}]")
    match dims:
      case [_]:
        rows = [cls.find_free_chunk()]
      case [n_rows, n_cols]:
        rows = cls.matrix_rows(n_rows, n_cols, srcinfo)
        if rows is None:
          raise MemGenError(f"{srcinfo}: not enough free rows in {cls.__name__} for {n_rows}x{n_cols}")
      case _:
        raise MemGenError(f"{srcinfo}: {cls.__name__} can only hold a vector or a matrix")
    for row in rows:
      cls.mark(row)
    cls.row_dict[new_name] = rows
    return f"{prefix}#define {new_name} {rows[0]}"

  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
    for row in cls.row_dict.pop(new_name):
      cls.unmark(row)
    return f"#undef {new_name}{_amx.clr_if_all_free()}"

  @classmethod
  def window(cls, basetyp, baseptr, indices, strides, srcinfo):
    """Return the C expression for the register (row) an operand starts at."""
    shape = basetyp.shape()
    assert len(indices) == len(strides) == len(shape)
    if indices[-1] != "0" or strides[-1] != "1":
      raise MemGenError(f"{srcinfo}: AMX instruction operands must be whole rows (lane 0, unit stride)")
    match indices:
      case [_]: return baseptr
      case [row, _]: return f"{baseptr} + ({row}) * {cls.row_stride(shape)}"
      case _: raise MemGenError(f"{srcinfo}: {cls.__name__} can only hold a vector or a matrix")

  @classmethod
  def matrix_rows(cls, n_rows, n_cols, srcinfo):
    """Physical rows for an n_rows x n_cols buffer, or None if it does not fit."""
    ...

  @classmethod
  def row_stride(cls, shape):
    """Physical rows between consecutive matrix rows of a buffer of this shape."""
    ...

class _APPLE_AMX_INPUT(_APPLE_AMX_POOL):
  """X and Y: 8 rows each, matrices occupy consecutive rows."""

  @classmethod
  def matrix_rows(cls, n_rows, n_cols, srcinfo):
    for first in range(0, cls.NUM_ROWS - n_rows + 1):
      rows = list(range(first, first + n_rows))
      if all(not cls.is_chunk_allocated[row] for row in rows):
        return rows
    return None

  @classmethod
  def row_stride(cls, shape):
    # matrix rows are consecutive registers, unlike the interleaved Z accumulators
    return 1

class APPLE_AMX_POOL_X(_APPLE_AMX_INPUT): NUM_ROWS = 8
class APPLE_AMX_POOL_Y(_APPLE_AMX_INPUT): NUM_ROWS = 8

class APPLE_AMX_POOL_Z(_APPLE_AMX_POOL):
  """Z: 64 rows. An NxN accumulator with N lanes per row is spread over the
  file with a stride of 64 // N rows, which is the layout the matrix-mode
  fma/fms instructions produce; up to 64 // N such accumulators coexist."""
  NUM_ROWS = 64

  @classmethod
  def matrix_rows(cls, n_rows, n_cols, srcinfo):
    if n_rows != n_cols:
      # TODO: allow several accumulators in one buffer (a 3rd dimension)
      raise MemGenError(f"{srcinfo}: Z matrices must be square, got {n_rows}x{n_cols}")
    stride = cls.NUM_ROWS // n_rows
    for first in range(stride):
      rows = list(range(first, cls.NUM_ROWS, stride))
      if not any(cls.is_chunk_allocated[row] for row in rows):
        return rows
    return None

  @classmethod
  def row_stride(cls, shape):
    return cls.NUM_ROWS // shape[0].val

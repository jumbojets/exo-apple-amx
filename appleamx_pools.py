import math

from exo.core.memory import MemGenError, StaticMemory, generate_offset

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
  """Base class for the three AMX register files: NUM_ROWS registers of 64
  bytes each. An Exo buffer is a vector (one row), a matrix (one row per
  matrix row) or a stack of matrices; which physical rows it occupies and how
  far apart they are is up to the pool (`stack_rows`, `reg_strides`).
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
  def global_(cls): return _amx.header()
  @classmethod
  def can_read(cls): return False

  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    prefix = _amx.set_if_inactive()
    dims = [int(d) for d in shape]
    if len(dims) not in (1, 2, 3):
      raise MemGenError(f"{srcinfo}: {cls.__name__} can only hold a vector, a matrix or a stack of matrices")
    rows = cls.rows_for(prim_type, dims, srcinfo)
    for row in rows: cls.mark(row)
    cls.row_dict[new_name] = rows
    return f"{prefix}#define {new_name} {rows[0]}"

  @classmethod
  def find_free_run(cls, n, step=1):
    """The first run of n consecutive free rows starting at a multiple of step, or None."""
    for first in range(0, cls.NUM_ROWS - n + 1, step):
      if not any(cls.is_chunk_allocated[first:first + n]):
        return list(range(first, first + n))
    return None

  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
    for row in cls.row_dict.pop(new_name): cls.unmark(row)
    return f"#undef {new_name}{_amx.clr_if_all_free()}"

  @classmethod
  def window(cls, basetyp, baseptr, indices, strides, srcinfo):
    """Return the C expression for the register (row) an operand starts at."""
    assert len(indices) == len(strides) == len(basetyp.shape())
    if indices[-1] != "0" or strides[-1] != "1":
      raise MemGenError(f"{srcinfo}: AMX instruction operands must be whole rows (lane 0, unit stride)")
    offset = generate_offset(indices[:-1], cls.reg_strides(basetyp))
    return baseptr if offset == "0" else f"{baseptr} + {offset}"

  @classmethod
  def rows_for(cls, prim_type, dims, srcinfo):
    """Physical rows for a buffer of prim_type whose rows each fill one register; the first is its base."""
    if dims[-1] * cls.CTYPE_BYTES[prim_type] != cls.ROW_BYTES:
      raise MemGenError(
        f"{srcinfo}: {cls.__name__} rows must be exactly {cls.ROW_BYTES} bytes, "
        f"got {prim_type}[{', '.join(map(str, dims))}]")
    match dims:
      case [_]: return [cls.find_free_chunk()]
      case [n_rows, n_cols]: m = 1
      case [m, n_rows, n_cols]: pass
    rows = cls.stack_rows(m, n_rows, n_cols, srcinfo)
    if rows is None:
      raise MemGenError(
        f"{srcinfo}: not enough free rows in {cls.__name__} for {'x'.join(map(str, dims))} "
        f"(live buffers: {', '.join(cls.row_dict) or 'none'})")
    return rows

  @classmethod
  def stack_rows(cls, m, n_rows, n_cols, srcinfo):
    """Physical rows for a stack of m n_rows x n_cols matrices (a matrix is m == 1), or None if it does not fit."""
    ...

  @classmethod
  def reg_strides(cls, basetyp):
    """Physical rows between neighbours along each dimension but the last (the lanes) of a buffer of this type."""
    ...

class _APPLE_AMX_INPUT(_APPLE_AMX_POOL):
  """X and Y: 8 rows each, matrices occupy consecutive rows."""

  @classmethod
  def stack_rows(cls, m, n_rows, n_cols, srcinfo):
    if m != 1:
      raise MemGenError(f"{srcinfo}: {cls.__name__} matrices already occupy consecutive rows; use a taller matrix")
    return cls.find_free_run(n_rows)
  @classmethod
  def reg_strides(cls, basetyp): return ["1"] * (len(basetyp.shape()) - 1)

class APPLE_AMX_POOL_X(_APPLE_AMX_INPUT): NUM_ROWS = 8
class APPLE_AMX_POOL_Y(_APPLE_AMX_INPUT): NUM_ROWS = 8

class APPLE_AMX_POOL_Z(_APPLE_AMX_POOL):
  """Z: 64 rows. An NxN accumulator with N lanes per row is spread over the
  file with a stride of 64 // N rows, which is the layout the matrix-mode
  fma/fms instructions produce, so up to 64 // N accumulators coexist: one
  per slot below the stride, row i of slot s in register i * stride + s. A
  stack [m, N, N] is m adjacent slots, whose rows share register pairs, which
  is what the pair loads and stores of Z move.

  `fma16` can also accumulate into f32. Its 32x32 f32 tile has 128-byte rows,
  each the register pair (2j, 2j + 1) with the even lanes in the even
  register, which is the layout `ldzi` / `stzi` move. Wide rows occupy
  consecutive pairs, so the tile takes the whole file and cannot coexist
  with any other Z buffer."""
  NUM_ROWS = 64

  @classmethod
  def is_wide(cls, prim_type, n_cols):
    """Whether a row of n_cols elements is a register pair rather than one register."""
    return n_cols * cls.CTYPE_BYTES[prim_type] == 2 * cls.ROW_BYTES

  @classmethod
  def rows_for(cls, prim_type, dims, srcinfo):
    """64-byte rows are placed by `stack_rows`; 128-byte rows are consecutive register pairs."""
    if not cls.is_wide(prim_type, dims[-1]):
      return super().rows_for(prim_type, dims, srcinfo)
    n_rows = math.prod(dims[:-1])
    rows = cls.find_free_run(2 * n_rows, step=2)
    if rows is None:
      raise MemGenError(
        f"{srcinfo}: not enough free rows in {cls.__name__} for {n_rows} 128-byte rows "
        f"(live buffers: {', '.join(cls.row_dict) or 'none'})")
    return rows

  @classmethod
  def stack_rows(cls, m, n_rows, n_cols, srcinfo):
    if n_rows != n_cols:
      raise MemGenError(f"{srcinfo}: Z matrices must be square, got {n_rows}x{n_cols}; "
                        "several accumulators stack along a leading dimension")
    stride = cls.NUM_ROWS // n_rows
    if m > stride:
      raise MemGenError(f"{srcinfo}: at most {stride} {n_rows}x{n_cols} accumulators fit in Z, got {m}")
    for first in range(stride - m + 1):
      rows = [first + slot + i * stride for slot in range(m) for i in range(n_rows)]
      if not any(cls.is_chunk_allocated[row] for row in rows):
        return rows
    return None

  @classmethod
  def reg_strides(cls, basetyp):
    """Wide rows are consecutive register pairs; otherwise accumulator rows are 64 // N apart and stack slots adjacent."""
    shape = [d.val for d in basetyp.shape()]
    if len(shape) == 1: return []
    if cls.is_wide(basetyp.basetype().ctype(), shape[-1]):
      return ["2"] if len(shape) == 2 else [str(2 * shape[-2]), "2"]
    stride = str(cls.NUM_ROWS // shape[-2])
    return [stride] if len(shape) == 2 else ["1", stride]

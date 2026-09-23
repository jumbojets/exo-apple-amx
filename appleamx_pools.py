from exo.core.memory import MemGenError, StaticMemory

CTYPE_BYTES = {"_Float16": 2, "float": 4, "double": 8, "int8_t": 1, "uint8_t": 1, "uint16_t": 2, "int32_t": 4}

def _parenthesized(e):
  return e if e.isidentifier() or e.isdigit() else f"({e})"

class _APPLE_AMX_POOL(StaticMemory):
  NUM_ROWS = 0
  live = 0

  def __init_subclass__(cls, **kw):
    super().__init_subclass__(**kw)
    cls.buffers = {}
    cls.init_state(cls.NUM_ROWS)

  @classmethod
  def global_(cls):
    # each compilation starts with nothing allocated
    _APPLE_AMX_POOL.live = 0
    cls.buffers = {}
    cls.init_state(cls.NUM_ROWS)
    return '#include "amx.h"'

  @classmethod
  def can_read(cls):
    return False

  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    # TODO: i32 can actually be accumulated into i32[32][32] using mac16
    if int(shape[-1]) * CTYPE_BYTES[prim_type] != 64:
      raise MemGenError("Row/vector allocation must be 64 bytes!")
    if len(shape) > 2:
      raise MemGenError("Can only allocate a vector or matrix!")
    rows = cls.rows_for([int(n) for n in shape])
    if rows is None:
      raise MemGenError("Not enough space to allocate!")
    for row in rows: cls.mark(row)
    cls.buffers[new_name] = rows
    set_ = "AMX_SET();\n" if _APPLE_AMX_POOL.live == 0 else ""
    _APPLE_AMX_POOL.live += 1
    return f"{set_}#define {new_name} {rows[0]}"

  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
    for row in cls.buffers.pop(new_name): cls.unmark(row)
    _APPLE_AMX_POOL.live -= 1
    clr = "\nAMX_CLR();" if _APPLE_AMX_POOL.live == 0 else ""
    return f"#undef {new_name}{clr}"

  @classmethod
  def window(cls, basetyp, baseptr, indices, strides, srcinfo):
    *row, col = indices
    if col != "0":
      raise MemGenError("Windows must start at the beginning of a row!")
    if not row:
      return baseptr
    stride = cls.row_stride([n.val for n in basetyp.shape()])
    offset = row[0] if stride == 1 else f"{_parenthesized(row[0])} * {stride}"
    return f"{baseptr} + {offset}"

  @classmethod
  def first_free(cls, candidates):
    for rows in candidates:
      if not any(cls.is_chunk_allocated[row] for row in rows):
        return list(rows)

  @classmethod
  def rows_for(cls, shape):
    ...

  @classmethod
  def row_stride(cls, shape):
    return 1

class _APPLE_AMX_INPUT(_APPLE_AMX_POOL):
  NUM_ROWS = 8

  @classmethod
  def rows_for(cls, shape):
    n_rows = shape[0] if len(shape) == 2 else 1
    return cls.first_free(range(row, row + n_rows) for row in range(cls.NUM_ROWS - n_rows + 1))

class APPLE_AMX_POOL_X(_APPLE_AMX_INPUT): pass
class APPLE_AMX_POOL_Y(_APPLE_AMX_INPUT): pass

class APPLE_AMX_POOL_Z(_APPLE_AMX_POOL):
  NUM_ROWS = 64

  @classmethod
  def rows_for(cls, shape):
    match shape:
      case [_]:
        return cls.first_free([row] for row in range(cls.NUM_ROWS))
      case [n_rows, n_cols] if n_rows == n_cols:
        # Allocate across an "accumulator"
        n_accumulators = cls.NUM_ROWS // n_rows
        return cls.first_free(range(row, cls.NUM_ROWS, n_accumulators) for row in range(n_accumulators))
    # TODO: we can imagining multiple allocators in a single alloc, in which n_rows != n_cols
    raise MemGenError("Number of matrix rows and columns must be the same!")

  @classmethod
  def row_stride(cls, shape):
    return cls.NUM_ROWS // shape[0]

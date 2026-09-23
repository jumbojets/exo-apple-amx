from copy import copy

from exo.core.memory import MemGenError, StaticMemory

CTYPE_BYTES = {
  "_Float16": 2, "float": 4, "double": 8,
  "int8_t": 1, "uint8_t": 1, "uint16_t": 2, "int32_t": 4,
}

class _APPLE_AMX_POOL(StaticMemory):
  # Live buffers across all pools; AMX is enabled while any are live
  n_live = 0

  def __init_subclass__(cls, **kw):
    super().__init_subclass__(**kw)
    cls.row_dict = {}
    if hasattr(cls, "NUM_ROWS"):
      cls.init_state(cls.NUM_ROWS)

  @classmethod
  def global_(cls):
    # Called once per compilation, before any alloc
    _APPLE_AMX_POOL.n_live = 0
    cls.row_dict = {}
    cls.init_state(cls.NUM_ROWS)
    return '#include "amx.h"'

  @classmethod
  def can_read(cls):
    return False

  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    match shape:
      case [*_, n] if int(n) * CTYPE_BYTES[prim_type] != 64:
        raise MemGenError("Row/vector allocation must be 64 bytes!")
      case [_]:
        rows = [cls.find_free_chunk()]
      case [n_rows, _]:
        rows = cls.matrix_rows(int(n_rows))
        if rows is None: raise MemGenError("Not enough space to allocate!")
      case _:
        raise MemGenError("Can only allocate a vector or matrix!")
    for row in rows: cls.mark(row)
    cls.row_dict[new_name] = rows
    set_amx = "AMX_SET();\n" if _APPLE_AMX_POOL.n_live == 0 else ""
    _APPLE_AMX_POOL.n_live += 1
    return f"{set_amx}#define {new_name} {rows[0]}"

  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
    for row in cls.row_dict.pop(new_name): cls.unmark(row)
    _APPLE_AMX_POOL.n_live -= 1
    clr_amx = "\nAMX_CLR();" if _APPLE_AMX_POOL.n_live == 0 else ""
    return f"#undef {new_name}{clr_amx}"

  @classmethod
  def window(cls, basetyp, baseptr, indices, strides, srcinfo):
    if len(indices) == 1:
      if indices[0] != "0": raise MemGenError("Cannot window within a row!")
      return f"({baseptr})"
    if indices[1] != "0": raise MemGenError("Cannot window within a row!")
    return f"({baseptr} + ({indices[0]}) * {cls.row_stride(basetyp.shape())})"

  @classmethod
  def matrix_rows(cls, n_rows):
    ...

  @classmethod
  def row_stride(cls, shape):
    ...

class _APPLE_AMX_INPUT(_APPLE_AMX_POOL):
  NUM_ROWS = 8

  @classmethod
  def matrix_rows(cls, n_rows):
    for row_idx in range(0, cls.NUM_ROWS - n_rows + 1):
      rows = list(range(row_idx, row_idx + n_rows))
      if all(not cls.is_chunk_allocated[row] for row in rows):
        return rows

  @classmethod
  def row_stride(cls, shape):
    return 1

class APPLE_AMX_POOL_X(_APPLE_AMX_INPUT): pass
class APPLE_AMX_POOL_Y(_APPLE_AMX_INPUT): pass

class APPLE_AMX_POOL_Z(_APPLE_AMX_POOL):
  NUM_ROWS = 64

  @classmethod
  def matrix_rows(cls, n_rows):
    # An N x N matrix takes every (64 / N)th row, starting below 64 / N
    stride = cls.NUM_ROWS // n_rows
    for row_idx in range(stride):
      rows = list(range(row_idx, row_idx + n_rows * stride, stride))
      if all(not cls.is_chunk_allocated[row] for row in rows):
        return rows

  @classmethod
  def row_stride(cls, shape):
    return cls.NUM_ROWS // shape[0].val

from copy import copy

from exo.core.memory import MemGenError, StaticMemory

class _APPLE_AMX_POOL(StaticMemory):
  row_dict = {}
  global_set = False
  is_active = False
  
  @classmethod
  def global_(cls):
    if not _APPLE_AMX_POOL.global_set:
      _APPLE_AMX_POOL.global_set = True
      return '#include "amx.h"'
    return ""

  @classmethod
  def can_read(cls):
    return False

  @classmethod
  def set_if_inactive(cls):
    if not _APPLE_AMX_POOL.is_active:
      _APPLE_AMX_POOL.is_active = True
      return "AMX_SET();\n"
    return ""
  
  @classmethod
  def clr_if_empty_z(cls):
    if _APPLE_AMX_POOL.row_dict == {}:
      _APPLE_AMX_POOL.is_active = False
      _APPLE_AMX_POOL.init_state(len(cls.is_chunk_allocated))
      return "\nAMX_CLR();"
    return ""
  
  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    set_if_inactive = cls.set_if_inactive()
    ctype_size = {"_Float16": 2, "float": 4, "double": 4, "int16_t": 2, "int32_t": 4, "int_fast32_t": 4}
    match shape:
      case [*_, n] if int(n) * ctype_size[prim_type] != 64:
        # TODO: i32 can actually be accumulated into i32[32][32] using mac16
        raise MemGenError("Row/vector allocation must be 64 bytes!")
      case [_]:
        row_idx = cls.find_free_chunk()
        cls.mark(row_idx)
      case [n_rows, n_cols]:
        rows = cls.matrix_rows(n_rows, n_cols)
        if rows is None: raise MemGenError("Not enough space to allocate!")
        for row in rows: cls.mark(row)
        row_idx = rows[0]
      case _:
        raise MemGenError("Can only allocate a vector or matrix!")
    cls.row_dict[new_name] = row_idx
    return f"{set_if_inactive}#define {new_name} {row_idx}"
  
  # TODO: don't take matrix cols?
  @classmethod
  def matrix_rows(cls, n_rows, n_cols):
    ...

class _APPLE_AMX_INPUT(_APPLE_AMX_POOL):
  NUM_ROWS = 8
  StaticMemory.init_state(NUM_ROWS)
  row_dict = {}

  def __init_subclass__(cls, **kw):
    super().__init_subclass__(**kw)
    cls.is_chunk_allocated = copy(cls.is_chunk_allocated)
    cls.row_dict = {}
  
  @classmethod
  def matrix_rows(cls, n_rows, n_cols):
    n_rows = int(n_rows)
    for row_idx in range(0, 8 - n_rows + 1):
      rows = list(range(row_idx, row_idx + n_rows))
      if all(not cls.is_chunk_allocated[row] for row in rows):
        return rows

  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
    row_idx = cls.row_dict[new_name]
    del cls.row_dict[new_name]
    match shape:
      case [_]: free_rows = [row_idx]
      case [n_rows, _]: 
        n_rows = int(n_rows)
        free_rows = list(range(row_idx, row_idx + n_rows))
    for row in free_rows: cls.unmark(row)
    return f"#undef {new_name}{cls.clr_if_empty_z()}"

  @classmethod
  def window(cls, basetyp, baseptr, indices, strides, srcinfo):
    assert len(indices) == len(strides) == 2
    assert strides[1] == "1"
    shape = basetyp.shape()
    assert len(shape) == 2
    return f"{baseptr} + {indices[0]}"

class APPLE_AMX_POOL_X(_APPLE_AMX_INPUT): pass
class APPLE_AMX_POOL_Y(_APPLE_AMX_INPUT): pass

class APPLE_AMX_POOL_Z(_APPLE_AMX_POOL):
  NUM_ROWS = 64
  StaticMemory.init_state(NUM_ROWS)

  @classmethod
  def matrix_rows(cls, n_rows, n_cols):
    # Allocate across an "accumulator"
    if n_rows == n_cols:
      n_rows = int(n_rows)
      n_accumulators = 64 // n_rows
      for row_idx in range(n_accumulators):
        rows = list(range(row_idx, row_idx + n_accumulators * n_rows, n_accumulators))
        if all(not cls.is_chunk_allocated[row] for row in rows):
         return rows
    else:
      # TODO: we can imagining multiple allocators in a single alloc, in which n_rows != n_cols
      raise MemGenError("Number of matrix rows and columns must be the same!")

  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
    del cls.row_dict[new_name]
    # free list is unmarked during AMX_CLR()
    return f"#undef {new_name}{cls.clr_if_empty_z()}"

  @classmethod
  def window(cls, basetyp, baseptr, indices, strides, srcinfo):
    # TODO: this method needs some verification and clean up. What are correct strides?
    assert len(indices) == len(strides) == 2
    assert strides[1] == "1"
    shape = basetyp.shape()
    assert len(shape) == 2
    n_accumulators = 64 // shape[0].val # NOTE: This is the physical stride 
    return f"{baseptr} + {indices[0]} * {n_accumulators}"

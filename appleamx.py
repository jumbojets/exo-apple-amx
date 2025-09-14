from exo.core.memory import MemGenError, StaticMemory

from copy import copy

class APPLE_AMX_INPUT(StaticMemory):
  NUM_ROWS = 8
  StaticMemory.init_state(NUM_ROWS)
  row_dict = {}

  def __init_subclass__(cls, **kw):
    super().__init_subclass__(**kw)
    cls.is_chunk_allocated = copy(cls.is_chunk_allocated)
    cls.row_dict = {}

  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    if len(shape) != 1:
      raise MemGenError(f"Can only allocate a single vector!")
    ctype_size = {"_Float16": 2, "float": 4, "double": 4, "int16_t": 2, "int32_t": 4, "int_fast32_t": 4} # TODO: "int8_t": 1,
    if shape[0] * ctype_size[prim_type] != 64:
      raise MemGenError(f"Vector allocation must be 64 bytes!")
    row_idx = cls.find_free_chunk()
    cls.mark(row_idx)
    cls.row_dict[new_name] = row_idx
    return f"#define {new_name} {row_idx}"
  
  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
    row_idx = cls.row_dict[new_name]
    del cls.row_dict[new_name]
    cls.unmark(row_idx)
    return f"#undef {new_name}"
  
  @classmethod
  def can_read(cls):
    return False

class APPLE_AMX_POOL_X(APPLE_AMX_INPUT):
  pass

class APPLE_AMX_POOL_Y(APPLE_AMX_INPUT):
  pass

class APPLE_AMX_POOL_Z(StaticMemory):
  NUM_ROWS = 64
  StaticMemory.init_state(NUM_ROWS)
  row_dict = {}

  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    ctype_size = {"_Float16": 2, "float": 4, "double": 4, "int16_t": 2, "int32_t": 4, "int_fast32_t": 4}
    match shape:
      case [*_, n] if n * ctype_size[prim_type] != 64:
        # TODO: i32 can actually be accumulated into i32[32][32] using mac16
        raise MemGenError("Row/vector allocation must be 64 bytes!")
      case [_]:
        row_idx = cls.find_free_chunk()
        cls.mark(row_idx)
        cls.row_dict[new_name] = row_idx
      case [n_rows, n_cols] if n_rows == n_cols:
        n_accumulators = 64 // n_rows
        for a in range(n_accumulators):
          rows = list(range(a, a + n_accumulators * n_rows, n_accumulators))
          if all(not cls.is_chunk_allocated[row] for row in rows):
            row_idx = a
            break
        else:
          raise MemGenError("Not enough space to allocate!")
        for row in rows: cls.mark(row)
        cls.row_dict[new_name] = row_idx
      case [_, _]:
        raise MemGenError("Number of matrix rows and columns must be the same!")
      case _:
        raise MemGenError("Can only allocate a vector or matrix!")
    return f"#define {new_name} {row_idx}"

  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
    row_idx = cls.row_dict[new_name]
    del cls.row_dict[new_name]
    match shape:
      case [_]:
        rows = iter((row_idx,))
      case [n_rows, _]:
        n_accumulators = 64 // n_rows
        rows = range(row_idx, row_idx + n_accumulators * n_rows, n_accumulators)
    for row in rows: cls.unmark(row)
    return f"#undef {new_name}"

  @classmethod
  def can_read(cls):
    return False

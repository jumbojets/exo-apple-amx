from __future__ import annotations
from copy import copy

from exo import DRAM, instr
from exo.core.memory import MemGenError, StaticMemory
from exo.stdlib.stdlib import stride

class APPLE_AMX_INPUT(StaticMemory):
  NUM_ROWS = 8
  StaticMemory.init_state(NUM_ROWS)
  row_dict = {}

  def __init_subclass__(cls, **kw):
    super().__init_subclass__(**kw)
    cls.is_chunk_allocated = copy(cls.is_chunk_allocated)
    cls.row_dict = {}

  def global_():
    return '#include "amx.h"'

  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    if len(shape) != 1:
      raise MemGenError(f"Can only allocate a single vector!")
    ctype_size = {"_Float16": 2, "float": 4, "double": 4, "int16_t": 2, "int32_t": 4, "int_fast32_t": 4} # TODO: "int8_t": 1,
    if int(shape[0]) * ctype_size[prim_type] != 64:
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
  ctype_size = {"_Float": 2, "float": 4, "double": 4, "int16_t": 2, "int32_t": 4, "int_fast32_t": 4}

  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    match shape:
      case [*_, n] if int(n) * cls.ctype_size[prim_type] != 64:
        # TODO: i32 can actually be accumulated into i32[32][32] using mac16
        raise MemGenError("Row/vector allocation must be 64 bytes!")
      case [_]:
        row_idx = cls.find_free_chunk()
        cls.mark(row_idx)
        cls.row_dict[new_name] = row_idx
      case [n_rows, n_cols] if n_rows == n_cols:
        n_rows = int(n_rows)
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
        n_rows = int(n_rows)
        n_accumulators = 64 // n_rows
        rows = range(row_idx, row_idx + n_accumulators * n_rows, n_accumulators)
    for row in rows: cls.unmark(row)
    return f"#undef {new_name}"

  @classmethod
  def window(cls, basetyp, baseptr, indices, strides, srcinfo):
    # TODO: this method needs some verification and clean up
    assert len(indices) == len(strides) == 2
    assert strides[1] == "1"
    # TODO: what is the value of strides[0]?
    print(f"{baseptr=}, {strides[0]=}")
    shape = basetyp.shape()
    assert len(shape) == 2
    # TODO: assert indices[1] is not relevant
    n_accumulators = 64 // shape[0].val # NOTE: This is the physical stride
    return f"{cls.row_dict[baseptr]} + {indices[0]} * {n_accumulators}"

  @classmethod
  def can_read(cls):
    return False

@instr("AMX_LDX({src_data}, {dst_data}, 0)")
def apple_amx_ldx_f32(dst: f32[16] @ APPLE_AMX_POOL_X, src: f32[16] @ DRAM):
  assert stride(dst, 0) == 1
  assert stride(src, 0) == 1
  for i in seq(0, 16): dst[i] = src[i]

@instr("AMX_LDY({src_data}, {dst_data}, 0)")
def apple_amx_ldy_f32(dst: f32[16] @ APPLE_AMX_POOL_Y, src: f32[16] @ DRAM):
  assert stride(dst, 0) == 1
  assert stride(src, 0) == 1
  for i in seq(0, 16): dst[i] = src[i]

@instr("AMX_STZ({dst_data}, {src_data}, 0)")
def apple_amx_stz_f32(dst: [f32][16] @ DRAM, src: [f32][16] @ APPLE_AMX_POOL_Z):
  assert stride(dst, 0) == 1
  assert stride(src, 0) == 1
  for i in seq(0, 16): dst[i] = src[i]

@instr("AMX_FMA32({srcx_data}, {srcy_data}, {dst_data}, 0)")
def apple_amx_fma32_mat(dst: f32[16, 16] @ APPLE_AMX_POOL_Z, srcx: f32[16] @ APPLE_AMX_POOL_X, srcy: f32[16] @ APPLE_AMX_POOL_Y):
  # TODO: I don't understand why I need to assert the strides for items in registers and if this is correct
  assert stride(dst, 1) == 1
  assert stride(srcx, 0) == 1
  assert stride(srcy, 0) == 1
  for i in seq(0, 16):
    for j in seq(0, 16):
      dst[j, i] += srcx[i] * srcy[j]

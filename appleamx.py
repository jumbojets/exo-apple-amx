from __future__ import annotations
from copy import copy

from exo import DRAM, instr
from exo.core.memory import MemGenError, StaticMemory
from exo.stdlib.stdlib import stride

class _APPLE_AMX_POOL(StaticMemory):
  z_row_dict = {}
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
    if _APPLE_AMX_POOL.z_row_dict == {}:
      _APPLE_AMX_POOL.is_active = False
      _APPLE_AMX_POOL.init_state(len(cls.is_chunk_allocated))
      return "\nAMX_CLR();"
    return ""

class _APPLE_AMX_INPUT(_APPLE_AMX_POOL):
  NUM_ROWS = 8
  StaticMemory.init_state(NUM_ROWS)
  row_dict = {}

  def __init_subclass__(cls, **kw):
    super().__init_subclass__(**kw)
    cls.is_chunk_allocated = copy(cls.is_chunk_allocated)
    cls.row_dict = {}

  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    set_if_inactive = cls.set_if_inactive()
    if len(shape) != 1:
      raise MemGenError(f"Can only allocate a single vector!")
    ctype_size = {"_Float16": 2, "float": 4, "double": 4, "int16_t": 2, "int32_t": 4, "int_fast32_t": 4} # TODO: "int8_t": 1,
    if int(shape[0]) * ctype_size[prim_type] != 64:
      raise MemGenError(f"Vector allocation must be 64 bytes!")
    row_idx = cls.find_free_chunk()
    cls.mark(row_idx)
    cls.row_dict[new_name] = row_idx
    return f"{set_if_inactive}#define {new_name} {row_idx}"
  
  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
    row_idx = cls.row_dict[new_name]
    del cls.row_dict[new_name]
    cls.unmark(row_idx)
    return f"#undef {new_name}{cls.clr_if_empty_z()}"

class APPLE_AMX_POOL_X(_APPLE_AMX_INPUT): pass
class APPLE_AMX_POOL_Y(_APPLE_AMX_INPUT): pass

class APPLE_AMX_POOL_Z(_APPLE_AMX_POOL):
  NUM_ROWS = 64
  StaticMemory.init_state(NUM_ROWS)
  ctype_size = {"_Float": 2, "float": 4, "double": 4, "int16_t": 2, "int32_t": 4, "int_fast32_t": 4}

  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
    set_if_inactive = cls.set_if_inactive()
    match shape:
      case [*_, n] if int(n) * cls.ctype_size[prim_type] != 64:
        # TODO: i32 can actually be accumulated into i32[32][32] using mac16
        raise MemGenError("Row/vector allocation must be 64 bytes!")
      case [_]:
        row_idx = cls.find_free_chunk()
        cls.mark(row_idx)
        cls.z_row_dict[new_name] = row_idx
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
        cls.z_row_dict[new_name] = row_idx
      case [_, _]:
        raise MemGenError("Number of matrix rows and columns must be the same!")
      case _:
        raise MemGenError("Can only allocate a vector or matrix!")
    return f"{set_if_inactive}#define {new_name} {row_idx}"

  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
    del cls.z_row_dict[new_name]
    # free list is unmarked during AMX_CLR()
    return f"#undef {new_name}{cls.clr_if_empty_z()}"

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
    return f"{cls.z_row_dict[baseptr]} + {indices[0]} * {n_accumulators}"

@instr("AMX_LDX(&{src_data}, {dst_data} * 64, 0);")
def apple_amx_ldx_f32(dst: f32[16] @ APPLE_AMX_POOL_X, src: f32[16] @ DRAM):
  assert stride(dst, 0) == 1
  assert stride(src, 0) == 1
  for i in seq(0, 16): dst[i] = src[i]

@instr("AMX_LDY(&{src_data}, {dst_data} * 64, 0);")
def apple_amx_ldy_f32(dst: f32[16] @ APPLE_AMX_POOL_Y, src: f32[16] @ DRAM):
  assert stride(dst, 0) == 1
  assert stride(src, 0) == 1
  for i in seq(0, 16): dst[i] = src[i]

@instr("AMX_STZ(&{dst_data}, {src_data}, 0);")
def apple_amx_stz_f32(dst: [f32][16] @ DRAM, src: [f32][16] @ APPLE_AMX_POOL_Z):
  assert stride(dst, 0) == 1
  assert stride(src, 0) == 1
  for i in seq(0, 16): dst[i] = src[i]

@instr("AMX_FMA32({srcx_data} * 64, {srcy_data} * 64, {dst_data}, 0);")
def apple_amx_fma32_mat(dst: f32[16, 16] @ APPLE_AMX_POOL_Z, srcx: f32[16] @ APPLE_AMX_POOL_X, srcy: f32[16] @ APPLE_AMX_POOL_Y):
  # TODO: I don't understand why I need to assert the strides for items in registers and if this is correct
  assert stride(dst, 1) == 1
  assert stride(srcx, 0) == 1
  assert stride(srcy, 0) == 1
  for i in seq(0, 16):
    for j in seq(0, 16):
      dst[j, i] += srcx[i] * srcy[j]

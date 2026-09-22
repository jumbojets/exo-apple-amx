"""Rewrite rules for scheduling Exo kernels onto the Apple AMX coprocessor: `stage_x`,
`stage_y` and `stage_z` stage a window into a register file laid out the way that file
holds it, and `replace_all_amx` turns the loop nests that are instructions into calls."""
from functools import partial

from exo import ExoType, Procedure
from exo.API_cursors import LiteralCursor
from exo.stdlib.inspection import (AllocCursor, BlockCursor, get_declaration, is_mul, is_read,
                                   is_write, nlr, nlr_stmts)
from exo.stdlib.scheduling import (SchedulingError, divide_dim, divide_loop, mult_dim,
                                   rearrange_dim, set_memory, simplify, stage_mem)
from exo.stdlib.stdlib import replace_all_stmts

import appleamx_ops
from appleamx_pools import APPLE_AMX_POOL_X, APPLE_AMX_POOL_Y, APPLE_AMX_POOL_Z

LANES = {ExoType.F16: 32, ExoType.F32: 16, ExoType.F64: 8,
         ExoType.I8: 64, ExoType.UI8: 64, ExoType.UI16: 32, ExoType.I32: 16}
POOLS = (APPLE_AMX_POOL_X, APPLE_AMX_POOL_Y, APPLE_AMX_POOL_Z)
# Pair moves are left out: their DRAM side must be 128-byte aligned, which Exo cannot check.
OPS = [op for op in vars(appleamx_ops).values()
       if isinstance(op, Procedure) and op.is_instr() and "AMX_LDST_PAIR" not in op.get_instr()]

def _lanes(typ):
  if typ not in LANES:
    raise SchedulingError(f"AMX registers hold {', '.join(t.name.lower() for t in LANES)}, not {typ.name.lower()}")
  return LANES[typ]

def _block(proc, block):
  if isinstance(block, str): block = proc.find(block)
  return block if isinstance(block, BlockCursor) else proc.forward(block).as_block()

def _shape(proc, name):
  """The constant shape of buffer `name`."""
  buffer = proc.find_alloc_or_arg(name)
  dims = list(buffer.shape()) if buffer.is_tensor() else []
  if not dims or not all(isinstance(d, LiteralCursor) for d in dims):
    raise SchedulingError(f"{name} must have a constant shape to fit in a register file")
  return [d.value() for d in dims]

def _stage(proc, block, window, name, pool):
  """stage_mem `window` into `pool` as the new buffer `name`; returns the proc, the name of
  the buffer the window is cut from, and the staged shape."""
  try: proc.find_alloc_or_arg(name)
  except SchedulingError: pass
  else: raise SchedulingError(f"{name} already exists; stage under a fresh name")
  proc = stage_mem(proc, block, window, name)
  proc = set_memory(proc, name, pool)
  proc = simplify(proc)
  return proc, window.split("[")[0].strip(), _shape(proc, name)

def _divide_copy_loops(proc, name, buf, depth, N):
  """Divide by N the loop `depth` levels out from the innermost loop around the copies between name and buf."""
  for pattern in (f"{name}[_] = {buf}[_]", f"{buf}[_] = {name}[_]"):
    try: copies = proc.find(pattern, many=True)
    except SchedulingError: continue  # stage_mem omitted the load or the store
    for copy in copies:
      loop = proc.forward(copy)
      for _ in range(depth + 1): loop = loop.parent()
      proc = divide_loop(proc, loop, N, [loop.name() + "o", loop.name() + "i"], perfect=True)
  return proc

def stage_input(proc, block, window, name, pool):
  """stage_mem `window` into X or Y with one register per row of the buffer. A row wider
  than a register (N lanes) is split across consecutive registers: `[r, w * N]` becomes
  `[r * w, N]` with row `i` in registers `w * i` to `w * i + w - 1`, so a `[4, 64]` f16
  window becomes `[8, 32]`."""
  proc, buf, shape = _stage(proc, block, window, name, pool)
  N = _lanes(proc.find_alloc_or_arg(name).type())
  if shape[-1] % N:
    raise SchedulingError(f"{window} has rows of {shape[-1]} lanes; a register holds {N}")
  if shape[-1] == N: return proc
  proc = _divide_copy_loops(proc, name, buf, 0, N)
  proc = divide_dim(proc, name, len(shape) - 1, N)
  if len(shape) == 2: proc = mult_dim(proc, name, 0, 1)
  return simplify(proc)

stage_x = partial(stage_input, pool=APPLE_AMX_POOL_X)
stage_y = partial(stage_input, pool=APPLE_AMX_POOL_Y)
stage_x.__doc__ = stage_y.__doc__ = stage_input.__doc__

def _multiplicand_type(proc, block, buf):
  """The type of the values multiplied into buf in block, or buf's own if nothing is."""
  types = set()
  for stmt in nlr_stmts(proc, block):
    if is_write(proc, stmt) and stmt.name() == buf:
      for product in [e for e in nlr(proc, stmt.rhs()) if is_mul(proc, e)]:
        for e in nlr(proc, product):
          if is_read(proc, e) and e.type().is_numeric() and e.name() != buf:
            types.add(get_declaration(proc, stmt, e.name()).type())
  if len(types) > 1:
    raise SchedulingError(f"{buf} accumulates products of more than one type: {', '.join(t.name.lower() for t in types)}")
  return types.pop() if types else proc.find_alloc_or_arg(buf).type()

def stage_z(proc, block, window, name):
  """stage_mem `window` into Z as the N x N accumulators the matrix instructions write,
  where N is the lanes per register of the values `block` multiplies into the window
  (32 for f16 inputs, also when accumulated in f32). A window of several tiles becomes
  a stack `[a * b, N, N]` with tile `i, j` in slot `b * i + j`, so a `[64, 32]` f16
  window becomes `[2, 32, 32]`. A vector or a single tile is staged as it is."""
  block = _block(proc, block)
  N = _lanes(_multiplicand_type(proc, block, window.split("[")[0].strip()))
  proc, buf, shape = _stage(proc, block, window, name, APPLE_AMX_POOL_Z)
  if len(shape) == 1:
    if shape[0] != N: raise SchedulingError(f"{window} is {shape[0]} wide; a Z row holds {N} lanes")
    return proc
  if shape[0] % N or shape[1] % N:
    raise SchedulingError(f"{window} is {shape[0]}x{shape[1]}; Z accumulators are {N}x{N}")
  a, b = shape[0] // N, shape[1] // N
  if a * b == 1: return proc
  if a > 1: proc = _divide_copy_loops(proc, name, buf, 1, N)
  if b > 1: proc = _divide_copy_loops(proc, name, buf, 0, N)
  if b > 1: proc = divide_dim(proc, name, 1, N)      # [a * N, b, N]
  if a > 1: proc = divide_dim(proc, name, 0, N)      # [a, N, b, N] or [a, N, N]
  if a > 1 and b > 1:
    proc = rearrange_dim(proc, name, [0, 2, 1, 3])   # [a, b, N, N]
    proc = mult_dim(proc, name, 0, 1)                # [a * b, N, N]
  elif b > 1:
    proc = rearrange_dim(proc, name, [1, 0, 2])      # [b, N, N]
  return simplify(proc)

def _in_pools(proc):
  """(type, register file) of every buffer the proc keeps in a register file."""
  buffers = list(proc.args()) + [s for s in nlr_stmts(proc) if isinstance(s, AllocCursor)]
  return {(b.type(), b.mem()) for b in buffers if b.type().is_numeric() and b.mem() in POOLS}

def replace_all_amx(proc):
  """Replace every loop nest that is an AMX instruction with a call to it."""
  # Exo's replace matches shapes and memories but not element types; replace_all_stmts checks each call site.
  have = _in_pools(proc)
  def applies(op):
    operands = [a for a in op.args() if a.type().is_numeric() and a.mem() in POOLS]
    return all((a.type(), a.mem()) in have for a in operands)
  return replace_all_stmts(proc, [op for op in OPS if applies(op)])

"""Generate ops.py, the Exo `@instr` definitions for Apple AMX.

Usage:
  python -m appleamx._gen_ops          # rewrite ops.py
  python -m appleamx._gen_ops --check  # exit 1 if ops.py is stale

Generated per element type (those Exo supports: f16 f32 f64 i8 ui8 ui16 i32):
  ld{x,y,z} / st{x,y,z}      one 64-byte row between DRAM and a register
  ld{x,y}2 / st{x,y}2        two consecutive rows (128 bytes, 128-byte aligned)
  extrx / extry / extrh      register-to-register moves (Y->X, X->Y, Z row->X)
per element type of 2 or more bytes, whose Z accumulators interleave:
  ldz2 / stz2                one row of a [2, N, N] stack of accumulators (128 bytes, 128-byte aligned)
  extrv                      one column of an NxN accumulator to Y
and per floating-point type:
  fma / fms / mul / zero     outer product (`_mat`) and pointwise (`_vec`)
  *_masked                   the same restricted to the first `rows` / `cols` lanes
plus the one mixed-precision mode Exo can express, f16 inputs accumulated in f32:
  {fma,fms,mul,zero}16_mat_f32   outer product into a 32x32 f32 tile spanning all 64 Z rows
  ldzi / stzi                one 128-byte f32 row of that tile, as two half moves
and one integer mode, `matint` 8, i8 inputs accumulated in i32:
  mac8_mat_i32 / zero8_mat_i32   outer product into a 16x64 i32 tile spanning all 64 Z rows
  ldzq / stzq                one 256-byte i32 row of that tile
"""
import sys
from collections import namedtuple
from functools import partial
from pathlib import Path

OUTPUT = Path(__file__).with_name("ops.py")

TYPE_BYTES = {"f16": 2, "f32": 4, "f64": 8, "i8": 1, "ui8": 1, "ui16": 2, "i32": 4}
FP_TYPES = ("f16", "f32", "f64")
ROW_BYTES = 64
Z_ROWS = 64

MEMS = {"DRAM": "DRAM", "X": "APPLE_AMX_POOL_X", "Y": "APPLE_AMX_POOL_Y", "Z": "APPLE_AMX_POOL_Z"}

# extrh / extrv lane-width field by element size. extrh moves whole rows, so it only
# affects the write mask and 1-byte types borrow the 16-bit setting; extrv reads one
# element per lane, so it must match.
EXTR_LANE_MODE = {8: 0, 4: 1, 2: 2, 1: 2}

# A scalar parameter's dtype is its Exo kind and its shape the lane count that bounds it.
SCALARS = ("size", "index")
Param = namedtuple("Param", "name dtype shape mem", defaults=((), "DRAM"))
Op = namedtuple("Op", "name c params body asserts")
OPS = []

def lanes(dtype): return ROW_BYTES // TYPE_BYTES[dtype]
def bits(dtype): return TYPE_BYTES[dtype] * 8
def lane_mode(dtype): return EXTR_LANE_MODE[TYPE_BYTES[dtype]]
def op(*args): OPS.append(Op(*args))
def unit_strides(*params): return [f"stride({p}, {d}) == 1" for p, d in params]

def loops(dims, stmt, guards=()):
  """Nest `for` loops over i, j with optional `if` guards around stmt."""
  assert len(dims) <= 2
  body, indent = [], ""
  for var, n in zip("ij", dims):
    body.append(f"{indent}for {var} in seq(0, {n}):")
    indent += "  "
  for guard in guards:
    body.append(f"{indent}if {guard}:")
    indent += "  "
  body.append(f"{indent}{stmt}")
  return body

def decl_param(p, mems=MEMS):
  if p.dtype in SCALARS: return f"{p.name}: {p.dtype}"
  return f"{p.name}: [{p.dtype}][{', '.join(map(str, p.shape))}] @ {mems[p.mem]}"

def render_op(o):
  lines = [f'@instr("{o.c}")', f"def {o.name}({', '.join(map(decl_param, o.params))}):"]
  lines += [f"  assert {a}" for a in o.asserts]
  lines += [f"  {line}" for line in o.body]
  return "\n".join(lines)

def move(name, c, dtype, shape, dst="DRAM", src="DRAM", asserts=()):
  """dst = src, rows of `shape` in the given memories, contiguous along the row."""
  ix = ", ".join("ij"[:len(shape)])
  row = len(shape) - 1
  op(name, c, [Param("dst", dtype, shape, dst), Param("src", dtype, shape, src)],
     loops(shape, f"dst[{ix}] = src[{ix}]"), unit_strides(("dst", row), ("src", row)) + list(asserts))

# Loads, stores and register-to-register moves
for dtype in TYPE_BYTES:
  N = lanes(dtype)
  for pool in "XYZ":
    p = pool.lower()
    move(f"apple_amx_ld{p}_{dtype}", f"AMX_LD{pool}(&{{src_data}}, ({{dst_data}}), 0);", dtype, (N,), dst=pool)
    move(f"apple_amx_st{p}_{dtype}", f"AMX_ST{pool}(&{{dst_data}}, ({{src_data}}), 0);", dtype, (N,), src=pool)
  # The DRAM side of a pair load/store must be 128-byte aligned, which Exo cannot check.
  for pool in "XY":
    p = pool.lower()
    move(f"apple_amx_ld{p}2_{dtype}", f"AMX_LD{pool}(&{{src_data}}, ({{dst_data}}), AMX_LDST_PAIR);",
         dtype, (2, N), dst=pool, asserts=[f"stride(src, 0) == {N}"])
    move(f"apple_amx_st{p}2_{dtype}", f"AMX_ST{pool}(&{{dst_data}}, ({{src_data}}), AMX_LDST_PAIR);",
         dtype, (2, N), src=pool, asserts=[f"stride(dst, 0) == {N}"])
  move(f"apple_amx_extrx_{dtype}", "AMX_EXTRX_FROM_Y(({dst_data}), ({src_data}));", dtype, (N,), dst="X", src="Y")
  move(f"apple_amx_extry_{dtype}", "AMX_EXTRY_FROM_X(({dst_data}), ({src_data}));", dtype, (N,), dst="Y", src="X")
  move(f"apple_amx_extrh_{dtype}", f"AMX_EXTRH(({{dst_data}}), ({{src_data}}), {lane_mode(dtype)});",
       dtype, (N,), dst="X", src="Z")

# Z pair loads and stores: a register pair is one row of two adjacent accumulators (see
# APPLE_AMX_POOL_Z), so the Z operand is a [2, N, N] stack indexed by row.
def z_stride(dtype): return Z_ROWS // lanes(dtype)

# 1-byte types have stride 1: no room for a second accumulator.
STACK_TYPES = [dtype for dtype in TYPE_BYTES if z_stride(dtype) > 1]

def z_pair_moves(dtype):
  N, S = lanes(dtype), z_stride(dtype)
  row = Param("row", "index", (N,))
  def regs(name): return Param(name, dtype, (2, N, N), "Z")
  def mem(name): return Param(name, dtype, (2 * N,))
  bounds = ["0 <= row", f"row < {N}"]
  op(f"apple_amx_ldz2_{dtype}", f"AMX_LDZ(&{{src_data}}, ({{dst_data}}) + ({{row}}) * {S}, AMX_LDST_PAIR);",
     [row, regs("dst"), mem("src")], loops([2, N], f"dst[i, row, j] = src[{N} * i + j]"),
     bounds + unit_strides(("dst", 2), ("src", 0)))
  op(f"apple_amx_stz2_{dtype}", f"AMX_STZ(&{{dst_data}}, ({{src_data}}) + ({{row}}) * {S}, AMX_LDST_PAIR);",
     [row, mem("dst"), regs("src")], loops([2, N], f"dst[{N} * i + j] = src[i, row, j]"),
     bounds + unit_strides(("dst", 0), ("src", 2)))

# A column of an accumulator is one lane of each of its rows.
def z_column_move(dtype):
  N, S = lanes(dtype), z_stride(dtype)
  col = Param("col", "index", (N,))
  op(f"apple_amx_extrv_{dtype}", f"AMX_EXTRV(({{dst_data}}), ({{src_data}}) + ({{col}}) * {S}, {lane_mode(dtype)});",
     [col, Param("dst", dtype, (N,), "Y"), Param("src", dtype, (N, N), "Z")], loops([N], "dst[i] = src[i, col]"),
     ["0 <= col", f"col < {N}"] + unit_strides(("dst", 0), ("src", 1)))

for dtype in STACK_TYPES:
  z_pair_moves(dtype)
  z_column_move(dtype)

# Floating-point fused multiply-add family: kind -> (macro prefix, extra flags, Exo statement)
ALU_KINDS = {
  "fma": ("AMX_FMA", [], "{z} += {y} * {x}"),
  "fms": ("AMX_FMS", [], "{z} += -({y} * {x})"),  # Exo only has `+=` reductions
  "mul": ("AMX_FMA", ["AMX_SKIP_Z"], "{z} = {y} * {x}"),
}
SKIP_ALL = ["AMX_SKIP_X", "AMX_SKIP_Y", "AMX_SKIP_Z"]  # fma with every input skipped writes 0 to Z

def alu_call(macro, flags, y_offset=""):
  return f"{macro}(({{srcy_data}}) * 64{y_offset}, ({{srcx_data}}) * 64, ({{dst_data}}), {' | '.join(flags) or 0});"

def alu_ops(dtype, name, flags, masks, z, y, x, zdtype=None, suffix="", mode_flags=()):
  """fma/fms/mul/zero of dtype rows from Y and X into a zdtype row or tile of Z, and the masked variants.

  z, y and x are the operands of the Exo statement; `masks` maps the size parameter of each loop
  to the flag that enables only that many lanes of its source."""
  N, B = lanes(dtype), bits(dtype)
  name, flags = f"{name}{suffix}", list(mode_flags) + flags
  dims = (N,) * len(masks)
  dst = Param("dst", zdtype or dtype, dims, "Z")
  srcs = [Param("srcy", dtype, (N,), "Y"), Param("srcx", dtype, (N,), "X")]
  dst_stride = unit_strides(("dst", len(dims) - 1))
  strides = dst_stride + unit_strides(("srcy", 0), ("srcx", 0))
  sizes = [Param(size, "size", (N,)) for size in masks]
  enables = [f"{flag}({{{size}}})" for size, flag in masks.items()]
  guards = [f"{var} < {size}" for var, size in zip("ij", masks)]
  bounds = [a for size in masks for a in (f"{size} <= {N}", f"0 < {size}")]
  for kind, (macro, kind_flags, stmt) in ALU_KINDS.items():
    stmt = stmt.format(z=z, y=y, x=x)
    op(f"apple_amx_{kind}{B}_{name}", alu_call(f"{macro}{B}", flags + kind_flags),
       [dst] + srcs, loops(dims, stmt), strides)
    op(f"apple_amx_{kind}{B}_{name}_masked", alu_call(f"{macro}{B}", flags + kind_flags + enables),
       sizes + [dst] + srcs, loops(dims, stmt, guards), bounds + strides)
  op(f"apple_amx_zero{B}_{name}", f"AMX_FMA{B}(0, 0, ({{dst_data}}), {' | '.join(flags + SKIP_ALL)});",
     [dst], loops(dims, f"{z} = 0.0"), dst_stride)

# Matrix mode takes the outer product of a Y row and an X row into an NxN tile of Z, vector mode
# their pointwise product into one row.
matrix_alu_ops = partial(alu_ops, name="mat", flags=[], z="dst[i, j]", y="srcy[i]", x="srcx[j]",
                         masks={"rows": "AMX_ENABLE_Y_FIRST", "cols": "AMX_ENABLE_X_FIRST"})
vector_alu_ops = partial(alu_ops, name="vec", flags=["AMX_VECTOR"], z="dst[i]", y="srcy[i]", x="srcx[i]",
                         masks={"n": "AMX_ENABLE_X_FIRST"})

for dtype in FP_TYPES:
  matrix_alu_ops(dtype)
  vector_alu_ops(dtype)
# fma16 can also accumulate into f32 (AMX_Z_F32), in matrix mode only. The
# 32x32 f32 tile then spreads over all 64 Z registers as interleaved pairs:
# logical row j is registers 2j (even lanes) and 2j + 1 (odd lanes).
matrix_alu_ops("f16", zdtype="f32", suffix="_f32", mode_flags=["AMX_Z_F32"])

# ldzi / stzi move one such 128-byte row in two calls, the register field's
# low bit selecting the left or right 16 lanes.
WIDE_LANES = 2 * lanes("f32")
move("apple_amx_ldzi_f32", "AMX_LDZI(&{src_data}, ({dst_data}), 0); AMX_LDZI(&{src_data} + 16, ({dst_data}) | 1, 0);",
     "f32", (WIDE_LANES,), dst="Z")
move("apple_amx_stzi_f32", "AMX_STZI(&{dst_data}, ({src_data}), 0); AMX_STZI(&{dst_data} + 16, ({src_data}) | 1, 0);",
     "f32", (WIDE_LANES,), src="Z")

# matint mode 8: the outer product of Y lanes 4i + k and an X row into a 16x64 i32 tile
QUAD_LANES = 4 * lanes("i32")  # a quad is four consecutive Z registers holding one interleaved row
MAC8_ROWS = Z_ROWS // 4
MAC8_FLAGS = ["AMX_MATINT_MAC8_I32", "AMX_MATINT_X_SIGNED", "AMX_MATINT_Y_SIGNED"]
mac8_dst = Param("dst", "i32", (MAC8_ROWS, QUAD_LANES), "Z")
op("apple_amx_mac8_mat_i32", alu_call("AMX_MATINT", MAC8_FLAGS, y_offset=" + ({k})"),
   [Param("k", "index", (4,)), mac8_dst, Param("srcy", "i8", (lanes("i8"),), "Y"), Param("srcx", "i8", (lanes("i8"),), "X")],
   loops([MAC8_ROWS, QUAD_LANES], "dst[i, j] += srcy[4 * i + k] * srcx[j]"),
   ["0 <= k", "k < 4"] + unit_strides(("dst", 1), ("srcy", 0), ("srcx", 0)))
op("apple_amx_zero8_mat_i32", "AMX_MATINT(0, 0, ({dst_data}), AMX_MATINT_MAC8_I32 | AMX_MATINT_ZERO_Z);",
   [mac8_dst], loops([MAC8_ROWS, QUAD_LANES], "dst[i, j] = 0"), unit_strides(("dst", 1)))

# ldzq / stzq move one 256-byte row of that tile
move("apple_amx_ldzq_i32", "amx_ldzq(&{src_data}, ({dst_data}));", "i32", (QUAD_LANES,), dst="Z")
move("apple_amx_stzq_i32", "amx_stzq(&{dst_data}, ({src_data}));", "i32", (QUAD_LANES,), src="Z")

HEADER = """\
# AUTOGENERATED by appleamx/_gen_ops.py. DO NOT EDIT.
from __future__ import annotations
from exo import *
from exo.stdlib.stdlib import stride
from .pools import APPLE_AMX_POOL_X, APPLE_AMX_POOL_Y, APPLE_AMX_POOL_Z
"""

def render_module():
  exports = "__all__ = [\n" + "".join(f'  "{o.name}",\n' for o in OPS) + "]\n"
  return HEADER + "\n" + exports + "\n" + "\n\n".join(map(render_op, OPS)) + "\n"

if __name__ == "__main__":
  text = render_module()
  if "--check" in sys.argv[1:]:
    if OUTPUT.exists() and OUTPUT.read_text() == text: sys.exit(0)
    sys.exit(f"{OUTPUT.name} is stale; rerun python -m appleamx._gen_ops")
  OUTPUT.write_text(text)
  print(f"wrote {len(OPS)} instructions to {OUTPUT.name}")

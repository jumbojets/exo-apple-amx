"""Generate appleamx_ops.py, the Exo `@instr` definitions for Apple AMX.

Usage:
  python gen_appleamx_ops.py          # rewrite appleamx_ops.py
  python gen_appleamx_ops.py --check  # exit 1 if appleamx_ops.py is stale

Generated per element type (those Exo supports: f16 f32 f64 i8 ui8 ui16 i32):
  ld{x,y,z} / st{x,y,z}      one 64-byte row between DRAM and a register
  ld{x,y}2 / st{x,y}2        two consecutive rows (128 bytes, 128-byte aligned)
  extrx / extry / extrh      register-to-register moves (Y->X, X->Y, Z row->X)
and per floating-point type:
  fma / fms / mul / zero     outer product (`_mat`) and pointwise (`_vec`)
  *_masked                   the same restricted to the first `rows` / `cols` lanes
plus the one mixed-precision mode Exo can express, f16 inputs accumulated in f32:
  {fma,fms,mul,zero}16_mat_f32   outer product into a 32x32 f32 tile spanning all 64 Z rows
  ldzi / stzi                one 128-byte f32 row of that tile, as two half moves
"""
import sys
from collections import namedtuple
from pathlib import Path

OUTPUT = Path(__file__).with_name("appleamx_ops.py")

TYPE_BYTES = {"f16": 2, "f32": 4, "f64": 8, "i8": 1, "ui8": 1, "ui16": 2, "i32": 4}
FP_TYPES = ("f16", "f32", "f64")
ROW_BYTES = 64

MEMS = {"DRAM": "DRAM", "X": "APPLE_AMX_POOL_X", "Y": "APPLE_AMX_POOL_Y", "Z": "APPLE_AMX_POOL_Z"}

# extrh lane-width field by element size; all lanes are enabled so it only
# affects the write mask, and 1-byte types borrow the 16-bit setting.
EXTRH_LANE_MODE = {8: 0, 4: 1, 2: 2, 1: 2}

# A `size` parameter has dtype None. Records rather than strings so that
# test_appleamx_ops.py can build test procs and C drivers from them.
Param = namedtuple("Param", "name dtype shape mem", defaults=(None, (), "DRAM"))
Op = namedtuple("Op", "name c params body asserts")
OPS = []

def lanes(dtype): return ROW_BYTES // TYPE_BYTES[dtype]
def bits(dtype): return TYPE_BYTES[dtype] * 8
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
  if p.dtype is None: return f"{p.name}: size"
  return f"{p.name}: [{p.dtype}][{', '.join(map(str, p.shape))}] @ {mems[p.mem]}"

def render_op(o):
  lines = [f'@instr("{o.c}")', f"def {o.name}({', '.join(map(decl_param, o.params))}):"]
  lines += [f"  assert {a}" for a in o.asserts]
  lines += [f"  {line}" for line in o.body]
  return "\n".join(lines)

# Loads, stores and register-to-register moves
for dtype in TYPE_BYTES:
  N = lanes(dtype)
  copy = loops([N], "dst[i] = src[i]")
  copy_pair = loops([2, N], "dst[i, j] = src[i, j]")
  for pool in "XYZ":
    p = pool.lower()
    op(f"apple_amx_ld{p}_{dtype}", f"AMX_LD{pool}(&{{src_data}}, ({{dst_data}}), 0);",
       [Param("dst", dtype, (N,), pool), Param("src", dtype, (N,))],
       copy, unit_strides(("dst", 0), ("src", 0)))
    op(f"apple_amx_st{p}_{dtype}", f"AMX_ST{pool}(&{{dst_data}}, ({{src_data}}), 0);",
       [Param("dst", dtype, (N,)), Param("src", dtype, (N,), pool)],
       copy, unit_strides(("dst", 0), ("src", 0)))
  # The DRAM side of a pair load/store must be 128-byte aligned, which Exo cannot check.
  for pool in "XY":
    p = pool.lower()
    op(f"apple_amx_ld{p}2_{dtype}", f"AMX_LD{pool}(&{{src_data}}, ({{dst_data}}), AMX_LDST_PAIR);",
       [Param("dst", dtype, (2, N), pool), Param("src", dtype, (2, N))],
       copy_pair, unit_strides(("dst", 1), ("src", 1)) + [f"stride(src, 0) == {N}"])
    op(f"apple_amx_st{p}2_{dtype}", f"AMX_ST{pool}(&{{dst_data}}, ({{src_data}}), AMX_LDST_PAIR);",
       [Param("dst", dtype, (2, N)), Param("src", dtype, (2, N), pool)],
       copy_pair, unit_strides(("dst", 1), ("src", 1)) + [f"stride(dst, 0) == {N}"])
  op(f"apple_amx_extrx_{dtype}", "AMX_EXTRX_FROM_Y(({dst_data}), ({src_data}));",
     [Param("dst", dtype, (N,), "X"), Param("src", dtype, (N,), "Y")],
     copy, unit_strides(("dst", 0), ("src", 0)))
  op(f"apple_amx_extry_{dtype}", "AMX_EXTRY_FROM_X(({dst_data}), ({src_data}));",
     [Param("dst", dtype, (N,), "Y"), Param("src", dtype, (N,), "X")],
     copy, unit_strides(("dst", 0), ("src", 0)))
  op(f"apple_amx_extrh_{dtype}", f"AMX_EXTRH(({{dst_data}}), ({{src_data}}), {EXTRH_LANE_MODE[TYPE_BYTES[dtype]]});",
     [Param("dst", dtype, (N,), "X"), Param("src", dtype, (N,), "Z")],
     copy, unit_strides(("dst", 0), ("src", 0)))

# Floating-point fused multiply-add family: kind -> (macro prefix, extra flags, Exo statement)
ALU_KINDS = {
  "fma": ("AMX_FMA", [], "{z} += {y} * {x}"),
  "fms": ("AMX_FMS", [], "{z} += -({y} * {x})"),  # Exo only has `+=` reductions
  "mul": ("AMX_FMA", ["AMX_SKIP_Z"], "{z} = {y} * {x}"),
}
SKIP_ALL = ["AMX_SKIP_X", "AMX_SKIP_Y", "AMX_SKIP_Z"]  # fma with every input skipped writes 0 to Z

def alu_call(macro, flags):
  return f"{macro}(({{srcy_data}}) * 64, ({{srcx_data}}) * 64, ({{dst_data}}), {' | '.join(flags) or 0});"

def alu_sources(dtype):
  N = lanes(dtype)
  return [Param("srcy", dtype, (N,), "Y"), Param("srcx", dtype, (N,), "X")], unit_strides(("srcy", 0), ("srcx", 0))

def matrix_alu_ops(dtype, zdtype, suffix="", mode_flags=()):
  """Outer products of dtype rows into an NxN zdtype accumulator: fma/fms/mul/zero and the masked variants."""
  N, B = lanes(dtype), bits(dtype)
  dst = Param("dst", zdtype, (N, N), "Z")
  srcs, src_strides = alu_sources(dtype)
  for kind, (macro, flags, stmt) in ALU_KINDS.items():
    flags = list(mode_flags) + flags
    stmt = stmt.format(z="dst[i, j]", y="srcy[i]", x="srcx[j]")
    op(f"apple_amx_{kind}{B}_mat{suffix}", alu_call(f"{macro}{B}", flags),
       [dst] + srcs, loops([N, N], stmt),
       unit_strides(("dst", 1)) + src_strides)
    op(f"apple_amx_{kind}{B}_mat{suffix}_masked",
       alu_call(f"{macro}{B}", flags + ["AMX_ENABLE_Y_FIRST({rows})", "AMX_ENABLE_X_FIRST({cols})"]),
       [Param("rows"), Param("cols"), dst] + srcs,
       loops([N, N], stmt, ["i < rows", "j < cols"]),
       [f"rows <= {N}", "0 < rows", f"cols <= {N}", "0 < cols"] + unit_strides(("dst", 1)) + src_strides)
  op(f"apple_amx_zero{B}_mat{suffix}", f"AMX_FMA{B}(0, 0, ({{dst_data}}), {' | '.join(list(mode_flags) + SKIP_ALL)});",
     [dst], loops([N, N], "dst[i, j] = 0.0"), unit_strides(("dst", 1)))

def vector_alu_ops(dtype):
  """Pointwise fma/fms/mul/zero of dtype rows into one dtype row of Z, and the masked variants."""
  N, B = lanes(dtype), bits(dtype)
  dst = Param("dst", dtype, (N,), "Z")
  srcs, src_strides = alu_sources(dtype)
  for kind, (macro, flags, stmt) in ALU_KINDS.items():
    flags = ["AMX_VECTOR"] + flags
    stmt = stmt.format(z="dst[i]", y="srcy[i]", x="srcx[i]")
    op(f"apple_amx_{kind}{B}_vec", alu_call(f"{macro}{B}", flags),
       [dst] + srcs, loops([N], stmt),
       unit_strides(("dst", 0)) + src_strides)
    op(f"apple_amx_{kind}{B}_vec_masked",
       alu_call(f"{macro}{B}", flags + ["AMX_ENABLE_X_FIRST({n})"]),
       [Param("n"), dst] + srcs, loops([N], stmt, ["i < n"]),
       [f"n <= {N}", "0 < n"] + unit_strides(("dst", 0)) + src_strides)
  op(f"apple_amx_zero{B}_vec", f"AMX_FMA{B}(0, 0, ({{dst_data}}), {' | '.join(['AMX_VECTOR'] + SKIP_ALL)});",
     [dst], loops([N], "dst[i] = 0.0"), unit_strides(("dst", 0)))

for dtype in FP_TYPES:
  matrix_alu_ops(dtype, dtype)
  vector_alu_ops(dtype)
# fma16 can also accumulate into f32 (AMX_Z_F32), in matrix mode only. The
# 32x32 f32 tile then spreads over all 64 Z registers as interleaved pairs:
# logical row j is registers 2j (even lanes) and 2j + 1 (odd lanes).
matrix_alu_ops("f16", "f32", suffix="_f32", mode_flags=["AMX_Z_F32"])

# ldzi / stzi move one such 128-byte row in two calls, the register field's
# low bit selecting the left or right 16 lanes.
WIDE_LANES = 2 * lanes("f32")
wide_row = loops([WIDE_LANES], "dst[i] = src[i]")
op("apple_amx_ldzi_f32",
   "AMX_LDZI(&{src_data}, ({dst_data}), 0); AMX_LDZI(&{src_data} + 16, ({dst_data}) | 1, 0);",
   [Param("dst", "f32", (WIDE_LANES,), "Z"), Param("src", "f32", (WIDE_LANES,))],
   wide_row, unit_strides(("dst", 0), ("src", 0)))
op("apple_amx_stzi_f32",
   "AMX_STZI(&{dst_data}, ({src_data}), 0); AMX_STZI(&{dst_data} + 16, ({src_data}) | 1, 0);",
   [Param("dst", "f32", (WIDE_LANES,)), Param("src", "f32", (WIDE_LANES,), "Z")],
   wide_row, unit_strides(("dst", 0), ("src", 0)))

HEADER = """\
# AUTOGENERATED by gen_appleamx_ops.py. DO NOT EDIT.
from __future__ import annotations
from exo import *
from exo.stdlib.stdlib import stride
from appleamx_pools import APPLE_AMX_POOL_X, APPLE_AMX_POOL_Y, APPLE_AMX_POOL_Z
"""

def render_module(): return HEADER + "\n" + "\n\n".join(map(render_op, OPS)) + "\n"

if __name__ == "__main__":
  text = render_module()
  if "--check" in sys.argv[1:]:
    if OUTPUT.exists() and OUTPUT.read_text() == text: sys.exit(0)
    sys.exit(f"{OUTPUT.name} is stale; rerun {Path(__file__).name}")
  OUTPUT.write_text(text)
  print(f"wrote {len(OPS)} instructions to {OUTPUT.name}")

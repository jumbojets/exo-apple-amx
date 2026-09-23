from functools import partial
from pathlib import Path

# Supported by both Exo and Apple AMX
TYPE_BYTES = {"f16": 2, "f32": 4, "f64": 8, "i8": 1, "ui8": 1, "ui16": 2, "i32": 4}
FLOAT_TYPES = ("f16", "f32", "f64")
VECFP_LANE_WIDTH = {"f16": 2, "f32": 4, "f64": 7}
X, Y, Z = "APPLE_AMX_POOL_X", "APPLE_AMX_POOL_Y", "APPLE_AMX_POOL_Z"

def lanes(dtype): return 64 // TYPE_BYTES[dtype]
def bits(dtype): return 8 * TYPE_BYTES[dtype]

def instr(name, code, body, index=None, **tensors):
  """An @instr proc; tensors maps arguments to (dtype, shape, memory), index names an argument picking a lane of dst."""
  args = [f"{arg}: [{dtype}][{', '.join(map(str, shape))}] @ {memory}" for arg, (dtype, shape, memory) in tensors.items()]
  asserts = [f"stride({arg}, {len(shape) - 1}) == 1" for arg, (_, shape, _) in tensors.items()]
  if index:
    args.append(f"{index}: index")
    asserts += [f"{index} >= 0", f"{index} < {tensors['dst'][1][-1]}"]
  lines = [f'@instr("{code}")', f"def apple_amx_{name}({', '.join(args)}):"]
  return "\n".join(lines + [f"  assert {a}" for a in asserts] + [f"  {line}" for line in body])

def loops(n, iters, stmt):
  """stmt inside a seq(0, n) loop per iterator"""
  return [f"{'  ' * depth}for {it} in seq(0, {n}):" for depth, it in enumerate(iters)] + ["  " * len(iters) + stmt]

def move(name, code, dtype, dst="DRAM", src="DRAM"):
  n = lanes(dtype)
  return instr(name, code, loops(n, "i", "dst[i] = src[i]"), dst=(dtype, [n], dst), src=(dtype, [n], src))

def column(name, code, dtype, dst):
  """Column c of a z accumulator"""
  n = lanes(dtype)
  return instr(name, code, loops(n, "j", "dst[j] = src[j, c]"), index="c", dst=(dtype, [n], dst), src=(dtype, [n, n], Z))

def alu(name, macro, stmt, flags="0", *, dtype, iters, index=None):
  """stmt on a z row (iters "i") or accumulator (iters "ij") and the x and y operands it reads"""
  n = lanes(dtype)
  srcs = {src: (dtype, [n], memory) for src, memory in (("srcx", X), ("srcy", Y)) if src in stmt}
  y, x = (f"({{{src}_data}}) * 64" if src in srcs else "0" for src in ("srcy", "srcx"))
  code = f"{macro}({y}, {x}, {{dst_data}}, {flags});"
  return instr(name, code, loops(n, iters, stmt), index=index, dst=(dtype, [n] * len(iters), Z), **srcs)

matrix_op = partial(alu, iters="ij")
vector_op = partial(alu, iters="i")

def loads_stores():
  for dtype in TYPE_BYTES:
    for pool, memory in zip("xyz", (X, Y, Z)):
      yield move(f"ld{pool}_{dtype}", f"AMX_LD{pool.upper()}(&{{src_data}}, {{dst_data}}, 0);", dtype, dst=memory)
      yield move(f"st{pool}_{dtype}", f"AMX_ST{pool.upper()}(&{{dst_data}}, {{src_data}}, 0);", dtype, src=memory)

def register_moves():
  for dtype in TYPE_BYTES:
    width = f"AMX_EXTR_{bits(dtype)}BIT"
    z_column = f"({{c}}) * {TYPE_BYTES[dtype]} + {{src_data}}"
    yield move(f"extrx_{dtype}", "AMX_EXTRX_COPY({dst_data}, {src_data});", dtype, dst=X, src=Y)
    yield move(f"extry_{dtype}", "AMX_EXTRY_COPY({dst_data}, {src_data});", dtype, dst=Y, src=X)
    yield move(f"extrhx_{dtype}", f"AMX_EXTRH(({{dst_data}}) * 64, {{src_data}}, {width});", dtype, dst=X, src=Z)
    yield move(f"extrhy_{dtype}", f"AMX_EXTRH(({{dst_data}}) * 64, {{src_data}}, AMX_TO_Y | {width});", dtype, dst=Y, src=Z)
    yield column(f"extrvx_{dtype}", f"AMX_EXTRV(({{dst_data}}) * 64, {z_column}, {width});", dtype, dst=X)
    yield column(f"extrvy_{dtype}", f"AMX_EXTRV(({{dst_data}}) * 64, {z_column}, AMX_TO_Y | {width});", dtype, dst=Y)

def fp_matrix_ops(dtype):
  b = bits(dtype)
  fma, fms = f"AMX_FMA{b}", f"AMX_FMS{b}"
  op = partial(matrix_op, dtype=dtype)
  yield op(f"fma{b}_mat", fma, "dst[i, j] += srcy[i] * srcx[j]")
  yield op(f"fms{b}_mat", fms, "dst[i, j] += -(srcy[i] * srcx[j])")
  yield op(f"mul{b}_mat", fma, "dst[i, j] = srcy[i] * srcx[j]", "AMX_SKIP_Z")
  yield op(f"zero{b}_mat", fma, "dst[i, j] = 0.0", "AMX_SKIP_X | AMX_SKIP_Y | AMX_SKIP_Z")
  yield op(f"addx{b}_mat", fma, "dst[i, j] += srcx[j]", "AMX_SKIP_Y")
  yield op(f"addy{b}_mat", fma, "dst[i, j] += srcy[i]", "AMX_SKIP_X")
  yield op(f"movx{b}_mat", fma, "dst[i, j] = srcx[j]", "AMX_SKIP_Y | AMX_SKIP_Z")
  yield op(f"movy{b}_mat", fma, "dst[i, j] = srcy[i]", "AMX_SKIP_X | AMX_SKIP_Z")

def fp_vector_ops(dtype):
  b = bits(dtype)
  fma, fms = f"AMX_FMA{b}", f"AMX_FMS{b}"
  vecfp = f"AMX_LANE_WIDTH({VECFP_LANE_WIDTH[dtype]})"
  op = partial(vector_op, dtype=dtype)
  yield op(f"fma{b}_vec", fma, "dst[i] += srcx[i] * srcy[i]", "AMX_VECTOR")
  yield op(f"fms{b}_vec", fms, "dst[i] += -(srcx[i] * srcy[i])", "AMX_VECTOR")
  yield op(f"mul{b}_vec", fma, "dst[i] = srcx[i] * srcy[i]", "AMX_VECTOR | AMX_SKIP_Z")
  yield op(f"zero{b}_vec", fma, "dst[i] = 0.0", "AMX_VECTOR | AMX_SKIP_X | AMX_SKIP_Y | AMX_SKIP_Z")
  yield op(f"addx{b}_vec", fma, "dst[i] += srcx[i]", "AMX_VECTOR | AMX_SKIP_Y")
  yield op(f"addy{b}_vec", fma, "dst[i] += srcy[i]", "AMX_VECTOR | AMX_SKIP_X")
  yield op(f"movx{b}_vec", fma, "dst[i] = srcx[i]", "AMX_VECTOR | AMX_SKIP_Y | AMX_SKIP_Z")
  yield op(f"movy{b}_vec", fma, "dst[i] = srcy[i]", "AMX_VECTOR | AMX_SKIP_X | AMX_SKIP_Z")
  yield op(f"min{b}_vec", "AMX_VECFP", "dst[i] = fmin(srcx[i], dst[i])", f"AMX_ALU_MODE(5) | {vecfp}")
  yield op(f"max{b}_vec", "AMX_VECFP", "dst[i] = fmax(srcx[i], dst[i])", f"AMX_ALU_MODE(7) | {vecfp}")
  yield op(f"select{b}_vec", "AMX_VECFP", "dst[i] = select(0.0, srcx[i], srcy[i], 0.0)", f"AMX_ALU_MODE(4) | {vecfp}")
  yield op(f"fma{b}_vec_lane", "AMX_VECFP", "dst[i] += srcx[i] * srcy[n]", f"AMX_ALU_MODE(0) | {vecfp} | AMX_BROADCAST_Y({{n}})", index="n")

HEADER = """\
# AUTOGENERATED FILE. DO NOT EDIT.
from __future__ import annotations
from exo import *
from exo.libs.externs import select
from exo.stdlib.stdlib import stride
from appleamx_externs import fmax, fmin
from appleamx_pools import APPLE_AMX_POOL_X, APPLE_AMX_POOL_Y, APPLE_AMX_POOL_Z
"""

def main():
  ops = [*loads_stores(), *register_moves()]
  for dtype in FLOAT_TYPES:
    ops += [*fp_matrix_ops(dtype), *fp_vector_ops(dtype)]
  Path(__file__).with_name("appleamx_ops.py").write_text(HEADER + "".join(f"\n{op}\n" for op in ops))

if __name__ == "__main__":
  main()

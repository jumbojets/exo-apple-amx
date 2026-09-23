import re
from pathlib import Path

# Supported by both Exo and Apple AMX
TYPE_BYTES = {"f16": 2, "f32": 4, "f64": 8, "i8": 1, "ui8": 1, "ui16": 2, "i32": 4}
FLOATS = ("f16", "f32", "f64")

MEMS = {"x": "APPLE_AMX_POOL_X", "y": "APPLE_AMX_POOL_Y", "z": "APPLE_AMX_POOL_Z", "dram": "DRAM"}

# extrh/extrv lane width (bits 63 and 11) when X/Y and Z have the same type
EXTR_LANES = {
  "f16": "(1ull << 63)", "f32": "(1ull << 63) | (8ull << 11)", "f64": "(1ull << 63) | (1ull << 11)",
  "i8": "0", "ui8": "0", "ui16": "(1ull << 11)", "i32": "(8ull << 11)",
}
# matfp/vecfp lane width (bits 42)
FP_LANES = {"f16": "(2ull << 42)", "f32": "(4ull << 42)", "f64": "(7ull << 42)"}

VECTOR = "(1ull << 63)"
SKIP_X, SKIP_Y, SKIP_Z = "(1ull << 29)", "(1ull << 28)", "(1ull << 27)"
# extrx/extry copy between X and Y; extrh/extrv move from Z
EXTR_XY, EXTR_Z, EXTR_TO_Y = "(1ull << 27)", "(1ull << 26)", "(1ull << 10)"

def lanes(dtype): return 64 // TYPE_BYTES[dtype]

def data(arg): return f"({{{arg}_data}})"

def flags(*bits): return " | ".join(bits) or "0"

# matfp/vecfp/matint/vecint ALU mode (bits 47)
def alu_mode(mode): return f"({mode}ull << 47)"

def vec(dtype, mem): return f"[{dtype}][{lanes(dtype)}] @ {MEMS[mem]}"

def mat(dtype): return f"[{dtype}][{lanes(dtype)}, {lanes(dtype)}] @ {MEMS['z']}"

def instr(name, c, stmt, n_loops, dtype, **args):
  out = [f'@instr("{c}")', f"def {name}({', '.join(f'{a}: {t}' for a, t in args.items())}):"]
  for a, t in args.items():
    if t == "index":
      out += [f"  assert {a} >= 0", f"  assert {a} < {lanes(dtype)}"]
    else:
      out.append(f"  assert stride({a}, {t.count(',')}) == 1")
  for depth, i in enumerate("ij"[:n_loops]):
    out.append(f"{'  ' * (depth + 1)}for {i} in seq(0, {lanes(dtype)}):")
  out.append(f"{'  ' * (n_loops + 1)}{stmt}")
  return "\n".join(out)

def move(name, c, dtype, dst, src):
  return instr(name, c, "dst[i] = src[i]", 1, dtype, dst=vec(dtype, dst), src=vec(dtype, src))

def load(pool, dtype):
  c = f"AMX_LD{pool.upper()}(&{{src_data}}, {{dst_data}}, 0);"
  return move(f"apple_amx_ld{pool}_{dtype}", c, dtype, dst=pool, src="dram")

def store(pool, dtype):
  c = f"AMX_ST{pool.upper()}(&{{dst_data}}, {{src_data}}, 0);"
  return move(f"apple_amx_st{pool}_{dtype}", c, dtype, dst="dram", src=pool)

def extrx(dtype):
  c = f"AMX_EXTRX({EXTR_XY} | {data('src')} << 20 | {data('dst')} << 16);"
  return move(f"apple_amx_extrx_{dtype}", c, dtype, dst="x", src="y")

def extry(dtype):
  c = f"AMX_EXTRY({EXTR_XY} | {data('dst')} << 6 | {data('src')} << 20);"
  return move(f"apple_amx_extry_{dtype}", c, dtype, dst="y", src="x")

def extr(dst, dtype, name, macro, z, stmt, **args):
  to_y = [EXTR_TO_Y] if dst == "y" else []
  c = f"{macro}({flags(EXTR_LANES[dtype], EXTR_Z, *to_y)} | ({z}) << 20 | {data('dst')} * 64);"
  return instr(name, c, stmt, 1, dtype, dst=vec(dtype, dst), **args)

def extrh(dst, dtype):
  return extr(dst, dtype, f"apple_amx_extrh{dst}_{dtype}", "AMX_EXTRH", data("src"),
              "dst[i] = src[i]", src=vec(dtype, "z"))

def extrv(dst, dtype):
  # Z column field: cell column in the high bits, row within the cell (the matrix's first row) in the low bits
  return extr(dst, dtype, f"apple_amx_extrv{dst}_{dtype}", "AMX_EXTRV", f"{data('src')} + {{col}} * {64 // lanes(dtype)}",
              "dst[i] = src[i, col]", src=mat(dtype), col="index")

def alu(name, macro, bits, stmt, n_loops, dtype, dst, x, y, **extra):
  body = stmt.format(z="dst[i, j]" if n_loops == 2 else "dst[i]", x="x[j]" if n_loops == 2 else "x[i]", y="y[i]")
  used = {a: t for a, t in {"x": x, "y": y}.items() if re.search(rf"\b{a}\[", body)}
  offsets = {a: f"{data(a)} * 64" if a in used else "0" for a in ("x", "y")}
  c = f"{macro}({offsets['y']}, {offsets['x']}, {data('dst')}, {flags(*bits)});"
  return instr(name, c, body, n_loops, dtype, dst=dst, **used, **extra)

def matrix_op(op, dtype, macro, stmt, *bits):
  return alu(f"apple_amx_{op}_mat_{dtype}", macro, bits, stmt, 2, dtype,
             dst=mat(dtype), x=vec(dtype, "x"), y=vec(dtype, "y"))

def vector_op(op, dtype, macro, stmt, *bits):
  return alu(f"apple_amx_{op}_vec_{dtype}", macro, (VECTOR, *bits), stmt, 1, dtype,
             dst=vec(dtype, "z"), x=vec(dtype, "x"), y=vec(dtype, "y"))

def fma_ops(op, dtype, fma, fms, *fms_bits):
  # fma32 etc. and mac16, simplified by skipping inputs
  yield op("fma", dtype, fma, "{z} += {y} * {x}")
  yield op("fms", dtype, fms, "{z} = {z} - {y} * {x}", *fms_bits)
  yield op("mul", dtype, fma, "{z} = {y} * {x}", SKIP_Z)
  yield op("addx", dtype, fma, "{z} += {x}", SKIP_Y)
  yield op("addy", dtype, fma, "{z} += {y}", SKIP_X)
  yield op("movx", dtype, fma, "{z} = {x}", SKIP_Y, SKIP_Z)
  yield op("movy", dtype, fma, "{z} = {y}", SKIP_X, SKIP_Z)
  yield op("zero", dtype, fma, "{z} = 0.0", SKIP_X, SKIP_Y, SKIP_Z)

def vecfp_ops(dtype):
  yield vector_op("select", dtype, "AMX_VECFP", "{z} = select(0.0, {x}, {y}, 0.0)", FP_LANES[dtype], alu_mode(4))
  yield vector_op("min", dtype, "AMX_VECFP", "{z} = fmin({x}, {z})", FP_LANES[dtype], alu_mode(5))
  yield vector_op("max", dtype, "AMX_VECFP", "{z} = fmax({x}, {z})", FP_LANES[dtype], alu_mode(7))
  # Write enable mode 1: every lane uses Y lane #n
  yield alu(f"apple_amx_fma_vec_lane_{dtype}", "AMX_VECFP", (FP_LANES[dtype], "(1ull << 38)", "(uint64_t){n} << 32"),
            "dst[i] += x[i] * y[n]", 1, dtype, dst=vec(dtype, "z"), x=vec(dtype, "x"), y=vec(dtype, "y"), n="index")

def matfp_ops(dtype):
  yield matrix_op("select", dtype, "AMX_MATFP", "{z} = select(0.0, {x}, {y}, 0.0)", FP_LANES[dtype], alu_mode(4))

def main():
  out = [
    "# AUTOGENERATED FILE. DO NOT EDIT.",
    "from __future__ import annotations",
    "from exo import *",
    "from exo.libs.externs import select",
    "from exo.stdlib.stdlib import stride",
    "from appleamx_externs import fmin, fmax",
    "from appleamx_pools import APPLE_AMX_POOL_X, APPLE_AMX_POOL_Y, APPLE_AMX_POOL_Z",
  ]
  for dtype in TYPE_BYTES:
    for pool in ("x", "y", "z"):
      out += [load(pool, dtype), store(pool, dtype)]
    out += [extrx(dtype), extry(dtype)]
    for dst in ("x", "y"):
      out += [extrh(dst, dtype), extrv(dst, dtype)]

  for dtype, bits in zip(FLOATS, (16, 32, 64)):
    for op in (matrix_op, vector_op):
      out += fma_ops(op, dtype, f"AMX_FMA{bits}", f"AMX_FMS{bits}")
    out += matfp_ops(dtype)
    out += vecfp_ops(dtype)

  # mac16 has no subtract; matint/vecint ALU mode 1 is z - x*y
  out += fma_ops(matrix_op, "ui16", "AMX_MAC16", "AMX_MATINT", alu_mode(1))
  out += fma_ops(vector_op, "ui16", "AMX_MAC16", "AMX_VECINT", alu_mode(1))

  Path("appleamx_ops.py").write_text("\n\n".join(out) + "\n")

if __name__ == "__main__":
  main()

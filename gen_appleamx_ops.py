"""Generate Exo instruction definitions for the Apple AMX coprocessor.

Instruction encodings follow https://github.com/corsix/amx. Every instruction's body is
the Exo reference semantics; test_appleamx_ops.py checks each one against the hardware.

Naming: apple_amx_<instruction>[<bits>|_<dtype>][_<mode>][_<variant>]
  e.g. apple_amx_fma16_mat, apple_amx_fms32_vec_negx, apple_amx_vecfp64_min, apple_amx_extrh_y_f32

Matrix ("mat") instructions write dst[i, j] = f(srcx[j], srcy[i]): rows of dst come from Y lanes and
columns from X lanes, and dst occupies every `TYPE_BYTES[dtype]`-th Z register (see APPLE_AMX_POOL_Z).

Not generated (yet):
  * Exo has no `>>` and no i16 window type, so integer ops are ui16 with a right shift of 0 and
    no saturation/sqrdmlah/popcnt modes
  * mixed lane widths (e.g. f16 x f16 -> f32 Z) and interleaved Z (ldzi/stzi)
  * load/store pairs, genlut and indexed loads, shuffles, and M2+ only features
"""

from pathlib import Path

# Supported by both Exo and Apple AMX
TYPE_BYTES = {"f16": 2, "f32": 4, "f64": 8, "i8": 1, "ui8": 1, "ui16": 2, "i32": 4}
FLOAT_TYPES = ("f16", "f32", "f64")

def lanes(dtype): return 64 // TYPE_BYTES[dtype]
def bits(dtype): return TYPE_BYTES[dtype] * 8

def operand(*fields):
  """Build a 64-bit operand from (value, bit, width) fields. Values are ints or C expressions."""
  const, exprs = 0, []
  for value, bit, width in fields:
    if isinstance(value, int):
      assert 0 <= value < (1 << width)
      const |= value << bit
    else:
      exprs.append(f"(((uint64_t)({value}) & {(1 << width) - 1}) << {bit})")
  return " | ".join([f"0x{const:x}ull", *exprs])

def win(dtype, mem, ndim=1):
  shape = ", ".join([str(lanes(dtype))] * ndim)
  return f"[{dtype}][{shape}] @ {mem if mem == 'DRAM' else f'APPLE_AMX_POOL_{mem}'}"

def render(name, c_instr, params, stmt, loops, preds=()):
  out = [f'@instr("{c_instr}")', f"def {name}({', '.join(f'{p}: {t}' for p, t in params)}):"]
  for p, t in params:
    if t.startswith("["): out.append(f"  assert stride({p}, {t.split('][')[1].count(',')}) == 1")
  out += [f"  assert {pred}" for pred in preds]
  indent = "  "
  for var, hi in loops:
    out.append(f"{indent}for {var} in seq(0, {hi}):")
    indent += "  "
  out.append(indent + stmt)
  return "\n" + "\n".join(out)

def alu(name, macro, dtype, mode, stmt, *fields, sizes=()):
  """An instruction of the form z = f(x, y, z), encoded as MACRO(y offset, x offset, z row, flags).
  `sizes` names size parameters ("m" rows, "n" lanes) that bound the loops; the caller encodes the
  matching lane enables in `fields`."""
  n = lanes(dtype)
  mat = mode == "mat"
  idx = {"z": "dst[i, j]", "x": "srcx[j]", "y": "srcy[i]"} if mat else {"z": "dst[i]", "x": "srcx[i]", "y": "srcy[i]"}
  params = [("dst", win(dtype, "Z", 2 if mat else 1))]
  if "{x}" in stmt: params.append(("srcx", win(dtype, "X")))
  if "{y}" in stmt: params.append(("srcy", win(dtype, "Y")))
  params += [(s, "size") for s in sizes]
  y = "({srcy_data}) * 64" if "{y}" in stmt else "0"
  x = "({srcx_data}) * 64" if "{x}" in stmt else "0"
  c_instr = f"{macro}({y}, {x}, ({{dst_data}}), {operand(*fields)});"
  loops = [("i", "m" if "m" in sizes else n), ("j", "n" if "n" in sizes else n)] if mat else [("i", "n" if "n" in sizes else n)]
  return render(name, c_instr, params, stmt.format(**idx), loops, [f"{s} <= {n}" for s in sizes])

# suffix: ((skip x, skip y, skip z), statement). Y comes first so matrix bodies read dst[i, j] += srcy[i] * srcx[j]
FMA_ALUS = {
  "":     ((0, 0, 0), "{z} += {y} * {x}"),
  "mul":  ((0, 0, 1), "{z} = {y} * {x}"),
  "addx": ((0, 1, 0), "{z} += {x}"),
  "x":    ((0, 1, 1), "{z} = {x}"),
  "addy": ((1, 0, 0), "{z} += {y}"),
  "y":    ((1, 0, 1), "{z} = {y}"),
  "zero": ((1, 1, 1), "{z} = 0.0"),
}
FMS_ALUS = {
  "":       ((0, 0, 0), "{z} = {z} - {y} * {x}"),
  "negmul": ((0, 0, 1), "{z} = -({y} * {x})"),
  "subx":   ((0, 1, 0), "{z} = {z} - {x}"),
  "negx":   ((0, 1, 1), "{z} = -{x}"),
  "suby":   ((1, 0, 0), "{z} = {z} - {y}"),
  "negy":   ((1, 0, 1), "{z} = -{y}"),
}
MAC16_ALUS = {**FMA_ALUS, "zero": ((1, 1, 1), "{z} = 0")}

def skip_fields(skip): return [(skip[0], 29, 1), (skip[1], 28, 1), (skip[2], 27, 1)]
def join(*parts): return "_".join(p for p in parts if p)

def gen_ldst(out):
  for dtype in TYPE_BYTES:
    for pool in "XYZ":
      loop = [("i", lanes(dtype))]
      out.append(render(f"apple_amx_ld{pool.lower()}_{dtype}", f"AMX_LD{pool}(&{{src_data}}, ({{dst_data}}), 0);",
                        [("dst", win(dtype, pool)), ("src", win(dtype, "DRAM"))], "dst[i] = src[i]", loop))
      out.append(render(f"apple_amx_st{pool.lower()}_{dtype}", f"AMX_ST{pool}(&{{dst_data}}, ({{src_data}}), 0);",
                        [("dst", win(dtype, "DRAM")), ("src", win(dtype, pool))], "dst[i] = src[i]", loop))

def gen_fma_fms(out):
  for dtype in FLOAT_TYPES:
    for op, alus in (("fma", FMA_ALUS), ("fms", FMS_ALUS)):
      macro = f"AMX_{op.upper()}{bits(dtype)}"
      for mode in ("vec", "mat"):
        vector = (int(mode == "vec"), 63, 1)
        for suffix, (skip, stmt) in alus.items():
          out.append(alu(join(f"apple_amx_{op}{bits(dtype)}", mode, suffix), macro, dtype, mode, stmt, vector, *skip_fields(skip)))
      # Lane-prefix enables ("first N lanes"): a count of 0 enables all lanes, so N & 31 is exact for N <= 32
      stmt = alus[""][1]
      out.append(alu(f"apple_amx_{op}{bits(dtype)}_vec_n", macro, dtype, "vec", stmt,
                     (1, 63, 1), (2, 46, 2), ("{n}", 41, 5), sizes=("n",)))
      out.append(alu(f"apple_amx_{op}{bits(dtype)}_mat_mn", macro, dtype, "mat", stmt,
                     (2, 46, 2), ("{n}", 41, 5), (2, 37, 2), ("{m}", 32, 5), sizes=("m", "n")))

def gen_mac16(out):
  for mode in ("vec", "mat"):
    for suffix, (skip, stmt) in MAC16_ALUS.items():
      out.append(alu(join("apple_amx_mac16", mode, suffix), "AMX_MAC16", "ui16", mode, stmt, (int(mode == "vec"), 63, 1), *skip_fields(skip)))

# Lane width mode (bit 42) for vecfp/matfp. f16 is "anything else"; 0 and 1 mean bf16 on M2.
FP_LANE_WIDTH = {"f16": 2, "f32": 4, "f64": 7}
VECFP_ALUS = {
  "min":       (5, "{z} = select({x}, {z}, {x}, {z})"),
  "max":       (7, "{z} = select({z}, {x}, {x}, {z})"),
  "posselect": (4, "{z} = select(0.0, {x}, {y}, 0.0)"),  # x <= 0 ? 0 : y
}
MATFP_ALUS = {k: VECFP_ALUS[k] for k in ("posselect",)}
INT_ALUS = {
  "submul": (1, "{z} = {z} - {y} * {x}"),
  "addsum": (2, "{z} += {y} + {x}"),
  "subsum": (3, "{z} = {z} - ({y} + {x})"),
}

def gen_vec_mat_fp_int(out):
  for dtype in FLOAT_TYPES:
    width = (FP_LANE_WIDTH[dtype], 42, 4)
    for suffix, (mode, stmt) in VECFP_ALUS.items():
      out.append(alu(f"apple_amx_vecfp{bits(dtype)}_{suffix}", "AMX_VECFP", dtype, "vec", stmt, (mode, 47, 6), width))
    for suffix, (mode, stmt) in MATFP_ALUS.items():
      out.append(alu(f"apple_amx_matfp{bits(dtype)}_{suffix}", "AMX_MATFP", dtype, "mat", stmt, (mode, 47, 6), width))
  # ui16 x ui16 -> ui16 (lane width 0), unsigned inputs, no shift
  for suffix, (mode, stmt) in INT_ALUS.items():
    out.append(alu(f"apple_amx_vecint_ui16_{suffix}", "AMX_VECINT", "ui16", "vec", stmt, (mode, 47, 6)))
    out.append(alu(f"apple_amx_matint_ui16_{suffix}", "AMX_MATINT", "ui16", "mat", stmt, (mode, 47, 6)))

# extrh (26=0) / extrv (26=0) lane width mode at bit 28, only affects write enables
EXTR_LANE_WIDTH = {8: 0, 4: 1, 2: 2, 1: 0}
# extrh (26=1) lane width mode as (bit 63, bits 11-14)
EXTRH_WIDE_LANE_WIDTH = {"f16": (1, 0), "f32": (1, 8), "f64": (1, 1), "i8": (0, 0), "ui8": (0, 0), "ui16": (0, 15), "i32": (0, 8)}

def gen_extr(out):
  for dtype in TYPE_BYTES:
    n, nbytes = lanes(dtype), TYPE_BYTES[dtype]
    loop = [("i", n)]
    out.append(render(f"apple_amx_extrx_{dtype}", f"AMX_EXTRX({operand((1, 27, 1), ('{src_data}', 20, 3), ('{dst_data}', 16, 3))});",
                      [("dst", win(dtype, "X")), ("src", win(dtype, "Y"))], "dst[i] = src[i]", loop))
    out.append(render(f"apple_amx_extry_{dtype}", f"AMX_EXTRY({operand((1, 27, 1), ('{src_data}', 20, 3), ('{dst_data}', 6, 3))});",
                      [("dst", win(dtype, "Y")), ("src", win(dtype, "X"))], "dst[i] = src[i]", loop))
    out.append(render(f"apple_amx_extrh_x_{dtype}",
                      f"AMX_EXTRX({operand((EXTR_LANE_WIDTH[nbytes], 28, 2), ('{src_data}', 20, 6), ('({dst_data}) * 64', 10, 9))});",
                      [("dst", win(dtype, "X")), ("src", win(dtype, "Z"))], "dst[i] = src[i]", loop))
    hi, lo = EXTRH_WIDE_LANE_WIDTH[dtype]
    out.append(render(f"apple_amx_extrh_y_{dtype}",
                      f"AMX_EXTRX({operand((hi, 63, 1), (1, 26, 1), ('{src_data}', 20, 6), (lo, 11, 4), (1, 10, 1), ('({dst_data}) * 64', 0, 9))});",
                      [("dst", win(dtype, "Y")), ("src", win(dtype, "Z"))], "dst[i] = src[i]", loop))
    if nbytes > 1:
      # Z column = column of cells * cells per row + row within each cell (the matrix's first Z register)
      out.append(render(f"apple_amx_extrv_y_{dtype}",
                        f"AMX_EXTRY({operand((EXTR_LANE_WIDTH[nbytes], 28, 2), (f'({{c}}) * {nbytes} + ({{src_data}})', 20, 6), ('({dst_data}) * 64', 0, 9))});",
                        [("dst", win(dtype, "Y")), ("src", win(dtype, "Z", 2)), ("c", "index")], "dst[i] = src[i, c]", loop,
                        preds=("c >= 0", f"c < {n}")))

def main():
  out = [
    "# AUTOGENERATED FILE. DO NOT EDIT. Regenerate with gen_appleamx_ops.py",
    "from __future__ import annotations",
    "from exo import *",
    "from exo.libs.externs import select",
    "from exo.stdlib.stdlib import stride",
    "from appleamx_pools import APPLE_AMX_POOL_X, APPLE_AMX_POOL_Y, APPLE_AMX_POOL_Z",
  ]
  gen_ldst(out)
  gen_fma_fms(out)
  gen_mac16(out)
  gen_vec_mat_fp_int(out)
  gen_extr(out)
  Path(__file__).with_name("appleamx_ops.py").write_text("\n".join(out) + "\n")

if __name__ == "__main__":
  main()

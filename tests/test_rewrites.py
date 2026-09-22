"""Schedule a few kernels with the rewrite rules, one per register-file layout the
rules produce, and check each against its naive version on the coprocessor."""
from __future__ import annotations

import subprocess
import tempfile
from collections import namedtuple
from functools import partial
from pathlib import Path

from exo import proc, compile_procs_to_strings
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *

import appleamx
from appleamx import *

CTYPES = {"f16": "_Float16", "f32": "float"}

# C[i, j] += A[k, i] * B[k, j]: two 32x32 f16 accumulators stacked, one beside the other,
# four 16x16 f32 accumulators, and one f16 x f16 -> f32 tile.
@proc
def matmul_64x32_f16(K: size, A: f16[K * 4, 64] @ DRAM, B: f16[K * 4, 32] @ DRAM, C: f16[64, 32] @ DRAM):
  for i in seq(0, 64):
    for j in seq(0, 32):
      for k in seq(0, K * 4):
        C[i, j] += A[k, i] * B[k, j]

@proc
def matmul_32x64_f16(K: size, A: f16[K * 4, 32] @ DRAM, B: f16[K * 4, 64] @ DRAM, C: f16[32, 64] @ DRAM):
  for i in seq(0, 32):
    for j in seq(0, 64):
      for k in seq(0, K * 4):
        C[i, j] += A[k, i] * B[k, j]

@proc
def matmul_32x32_f32(K: size, A: f32[K * 4, 32] @ DRAM, B: f32[K * 4, 32] @ DRAM, C: f32[32, 32] @ DRAM):
  for i in seq(0, 32):
    for j in seq(0, 32):
      for k in seq(0, K * 4):
        C[i, j] += A[k, i] * B[k, j]

@proc
def matmul_32x32_f16_f32(K: size, A: f16[K * 8, 32] @ DRAM, B: f16[K * 8, 32] @ DRAM, C: f32[32, 32] @ DRAM):
  for i in seq(0, 32):
    for j in seq(0, 32):
      for k in seq(0, K * 8):
        C[i, j] += A[k, i] * B[k, j]

# Vector mode: a product with no accumulator load, and a 64-wide f16 window that spans two X / Y rows.
@proc
def mul_f32(n: size, x: f32[n] @ DRAM, y: f32[n] @ DRAM, z: f32[n] @ DRAM):
  assert n % 16 == 0
  for i in seq(0, n):
    z[i] = y[i] * x[i]

@proc
def fma_f16(n: size, x: f16[n] @ DRAM, y: f16[n] @ DRAM, z: f16[n] @ DRAM):
  assert n % 64 == 0
  for i in seq(0, n):
    z[i] += y[i] * x[i]

def amx_matmul(p, N, block):
  """k outermost, i and j tiled by N, C in Z, `block` rows of A and B at a time in Y and X."""
  rows, cols = (d.value() for d in p.find_alloc_or_arg("C").shape())
  p = reorder_loops(p, "j k")
  p = reorder_loops(p, "i k")
  if rows > N: p = divide_loop(p, "i", N, ["ti", "i"], perfect=True)
  if cols > N:
    p = divide_loop(p, "j", N, ["tj", "j"], perfect=True)
    p = reorder_loops(p, "i tj")
  p = stage_z(p, "for k in _:_", f"C[0:{rows}, 0:{cols}]", "C_z")
  p = divide_loop(p, "k", block, ["k0", "k1"], perfect=True)
  p = stage_y(p, "for k1 in _:_", f"A[{block} * k0:{block} * k0 + {block}, 0:{rows}]", "A_y")
  p = stage_x(p, "for k1 in _:_", f"B[{block} * k0:{block} * k0 + {block}, 0:{cols}]", "B_x")
  p = replace_all_amx(p)
  p = unroll_loops(p, p.find_loop("k0"))
  return simplify(p)

def amx_vector(p, N):
  """One register row of x, y and z per iteration."""
  p = divide_loop(p, "i", N, ["io", "ii"], perfect=True)
  for buf, stage in [("z", stage_z), ("y", stage_y), ("x", stage_x)]:
    p = stage(p, "for ii in _:_", f"{buf}[{N} * io:{N} * io + {N}]", f"{buf}_r")
  return replace_all_amx(p)

def amx_vector_wide(p):
  """64 f16 of x and y in two rows each; z one row at a time."""
  p = divide_loop(p, "i", 64, ["io", "ii"], perfect=True)
  p = divide_loop(p, "ii", 32, ["half", "ii"], perfect=True)
  p = stage_y(p, "for half in _:_", "y[64 * io:64 * io + 64]", "y_r")
  p = stage_x(p, "for half in _:_", "x[64 * io:64 * io + 64]", "x_r")
  p = stage_z(p, "for ii in _:_", "z[64 * io + 32 * half:64 * io + 32 * half + 32]", "z_r")
  p = replace_all_amx(p)
  return simplify(unroll_loops(p, p.find_loop("half")))

Case = namedtuple("Case", "naive schedule size regs")  # regs: register buffers and calls the schedule must produce
CASES = [
  Case(matmul_64x32_f16, partial(amx_matmul, N=32, block=4), size=3,
       regs=["C_z: f16[2, 32, 32] @ APPLE_AMX_POOL_Z", "A_y: f16[8, 32] @ APPLE_AMX_POOL_Y", "B_x: f16[4, 32] @ APPLE_AMX_POOL_X"]),
  Case(matmul_32x64_f16, partial(amx_matmul, N=32, block=4), size=3,
       regs=["C_z: f16[2, 32, 32] @ APPLE_AMX_POOL_Z", "A_y: f16[4, 32] @ APPLE_AMX_POOL_Y", "B_x: f16[8, 32] @ APPLE_AMX_POOL_X"]),
  Case(matmul_32x32_f32, partial(amx_matmul, N=16, block=4), size=3,
       regs=["C_z: f32[4, 16, 16] @ APPLE_AMX_POOL_Z", "A_y: f32[8, 16] @ APPLE_AMX_POOL_Y", "B_x: f32[8, 16] @ APPLE_AMX_POOL_X"]),
  Case(matmul_32x32_f16_f32, partial(amx_matmul, N=32, block=8), size=2,
       regs=["C_z: f32[32, 32] @ APPLE_AMX_POOL_Z", "A_y: f16[8, 32] @ APPLE_AMX_POOL_Y", "B_x: f16[8, 32] @ APPLE_AMX_POOL_X"]),
  Case(mul_f32, partial(amx_vector, N=16), size=64,
       regs=["z_r: f32[16] @ APPLE_AMX_POOL_Z", "apple_amx_mul32_vec("]),
  Case(fma_f16, amx_vector_wide, size=128,
       regs=["y_r: f16[2, 32] @ APPLE_AMX_POOL_Y", "x_r: f16[2, 32] @ APPLE_AMX_POOL_X", "apple_amx_fma16_vec("]),
]

def driver_case(naive, scheduled, size):
  """C block running both procs on the same random data and comparing every buffer."""
  scalar = next(a.name() for a in naive.args() if not a.is_tensor())
  bufs = [(a.name(), CTYPES[a.type().name.lower()], " * ".join(expr_to_string(d, {scalar: str(size)}) for d in a.shape()))
          for a in naive.args() if a.is_tensor()]
  lines = ["{"]
  for name, T, n in bufs:
    lines.append(f"  static alignas(128) {T} {name}[{n}], {name}_t[{n}];")
    lines.append(f"  for (size_t i = 0; i < {n}; i++) {name}[i] = {name}_t[i] = ({T})(rand() % 7 - 3);")
  lines.append(f"  {naive.name()}(NULL, {size}, {', '.join(name for name, _, _ in bufs)});")
  lines.append(f"  {scheduled.name()}(NULL, {size}, {', '.join(name + '_t' for name, _, _ in bufs)});")
  for name, _, n in bufs:
    lines.append(f'  check("{scheduled.name()}", "{name}", {name}, {name}_t, {n});')
  lines.append("}")
  return "\n".join(lines)

DRIVER_HEADER = """\
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include "amx_rewrites.h"

static int failures = 0;

#define check(kernel, name, ref, test, n) \\
  for (size_t i = 0; i < (n); i++) \\
    if ((float)(ref)[i] != (float)(test)[i]) { printf("FAIL %s: %s[%zu]\\n", kernel, name, i); failures++; break; }

int main() {
  srand(1);
"""

def test_rewrite_rules_schedule_correct_kernels():
  runs = []
  for case in CASES:
    scheduled = case.schedule(rename(case.naive, case.naive.name() + "_amx"))
    for reg in case.regs: assert reg in str(scheduled), f"{scheduled.name()} lacks {reg}:\n{scheduled}"
    runs.append((case.naive, scheduled, case.size))

  with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    c, h = compile_procs_to_strings([p for run in runs for p in run[:2]], "amx_rewrites.h")
    (tmp / "amx_rewrites.c").write_text(c)
    (tmp / "amx_rewrites.h").write_text(h)
    driver = DRIVER_HEADER + "\n".join(driver_case(*run) for run in runs)
    driver += '\n  printf("%d failures in %d kernels\\n", failures, ' + str(len(CASES)) + ");\n  return failures != 0;\n}\n"
    (tmp / "driver.c").write_text(driver)
    cc = subprocess.run(["cc", "-march=native", "-O1", "-Wall", "-Werror", f"-I{appleamx.include_dir()}",
                         "amx_rewrites.c", "driver.c", "-o", "driver"], cwd=tmp, capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    run = subprocess.run(["./driver"], cwd=tmp, capture_output=True, text=True)
    print(run.stdout, end="")
    assert run.returncode == 0, run.stdout

if __name__ == "__main__":
  test_rewrite_rules_schedule_correct_kernels()

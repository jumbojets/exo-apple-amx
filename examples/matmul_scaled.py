"""C[32, 32] = A^T @ B with column j scaled by s[j], in f16 on the Apple AMX coprocessor.

The product accumulates in one 32x32 Z tile and the scale stays in the register
files: `extrh` moves a row of the tile to X, vector-mode `mul16` multiplies it lane
by lane with s held in Y into a spare Z row, and `stz` stores that row.  The whole
schedule is the rewrite rules of appleamx.rewrites: `stage_x` / `stage_y` /
`stage_z` lay a window out the way its register file holds it, and
`replace_all_amx` turns every loop nest that is an instruction into a call.
"""
from __future__ import annotations

import os
import sys

from exo import proc
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *

from appleamx import *

# Hide output when running through exocc
if __name__ != "__main__" and hasattr(os, "devnull"):
  sys.stdout = open(os.devnull, "w")

@proc
def scaled_matmul_32x32(
  K: size, A: f16[K * 4, 32] @ DRAM, B: f16[K * 4, 32] @ DRAM, s: f16[32] @ DRAM, C: f16[32, 32] @ DRAM
):
  acc: f16[32, 32]
  for i in seq(0, 32):
    for j in seq(0, 32):
      acc[i, j] = 0.0
      for k in seq(0, K * 4):
        acc[i, j] += A[k, i] * B[k, j]
      C[i, j] = s[j] * acc[i, j]

print("=============Original Matmul==============")
print(scaled_matmul_32x32)

amx = rename(scaled_matmul_32x32, "scaled_matmul_32x32_scheduled_appleamx")

# 1. Three passes over the tile: zero it, accumulate with k outermost, scale it.
amx = fission(amx, amx.find("acc[_] = 0.0").after(), n_lifts=2)
amx = fission(amx, amx.find_loop("k").after(), n_lifts=2)
amx = reorder_loops(amx, "j k")
amx = reorder_loops(amx, "i k")

# 2. The accumulator is a Z tile; blocks of 4 k put A rows in Y and B rows in X.
amx = set_memory(amx, "acc", APPLE_AMX_POOL_Z)
amx = divide_loop(amx, "k", 4, ["k0", "k1"], perfect=True)
amx = stage_y(amx, "for k1 in _:_", "A[4 * k0:4 * k0 + 4, 0:32]", "A_y")
amx = stage_x(amx, "for k1 in _:_", "B[4 * k0:4 * k0 + 4, 0:32]", "B_x")

# 3. s in Y for the whole scale pass; each tile row goes through X into a Z row that is stored.
scale = amx.find("C[_] = _").parent()
amx = stage_y(amx, scale.parent(), "s[0:32]", "s_y")
amx = stage_x(amx, scale, "acc[i, 0:32]", "acc_x")
amx = stage_z(amx, scale, "C[i, 0:32]", "C_z")

# 4. Instructions, then unroll the k block so every register index is a constant.
amx = replace_all_amx(amx)
amx = unroll_loops(amx, amx.find_loop("k0"))
amx = simplify(amx)

print("=============Optimized Matmul==============")
print(amx)

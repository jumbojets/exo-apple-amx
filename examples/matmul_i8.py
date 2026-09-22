"""C[16, 64] += A^T @ B with i8 inputs accumulated in i32 on the Apple AMX coprocessor.

`matint` mode 8 takes the outer product of an X row (64 i8, the columns of C) with
every fourth lane of a Y row (16 i8, the rows of C) into one 16x64 i32 tile, which
is the whole Z file.  A is passed packed the way Y reads it, four k rows to a
64-byte row with element (k, i) in lane 4 * i + k % 4 of row k / 4, so one Y load
serves four mac8 calls, one per k % 4, against four X rows of B.  The tile's
256-byte rows move through `ldzq` / `stzq`.
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
def matmul_16x64_i8(K: size, A: i8[K, 64] @ DRAM, B: i8[4 * K, 64] @ DRAM, C: i32[16, 64] @ DRAM):
  for i in seq(0, 16):
    for j in seq(0, 64):
      for k in seq(0, 4 * K):
        C[i, j] += A[k / 4, 4 * i + k % 4] * B[k, j]

print("=============Original Matmul==============")
print(matmul_16x64_i8)

amx = rename(matmul_16x64_i8, "matmul_16x64_i8_scheduled_appleamx")

# 1. k outermost in blocks of 4: one packed row of A against four rows of B.
amx = reorder_loops(amx, "j k")
amx = reorder_loops(amx, "i k")
amx = divide_loop(amx, "k", 4, ["k0", "k1"], perfect=True)
amx = simplify(amx)

# 2. C is the one tile Z holds.
amx = stage_mem(amx, "for k0 in _:_", "C[0:16, 0:64]", "C_z")
amx = set_memory(amx, "C_z", APPLE_AMX_POOL_Z)

# 3. The A row to Y and the B rows to X.
amx = stage_y(amx, "for k1 in _:_", "A[k0, 0:64]", "A_y")
amx = stage_x(amx, "for k1 in _:_", "B[4 * k0:4 * k0 + 4, 0:64]", "B_x")

# 4. Instructions, then unroll the k block so every register index is a constant.
amx = replace_all_amx(amx)
amx = unroll_loops(amx, amx.find_loop("k0"))
amx = simplify(amx)

print("=============Optimized Matmul==============")
print(amx)

"""C[64, 32] += A[K * 8, 64]^T @ B[K * 8, 32] in f16 on the Apple AMX coprocessor.

Register plan
-------------
In matrix mode `fma16` multiplies one X row (32 f16, the columns of Z) with one
Y row (32 f16, the rows of Z) into a 32x32 Z tile, so every k is a rank-1 update
of C.  C has 64 rows, which is two Z tiles fed by the two halves of an A row:

    C_0 (C rows  0-31) += A[k,  0:32] (x) B[k, 0:32]      Y[0:4] (x) X[0:4]
    C_1 (C rows 32-63) += A[k, 32:64] (x) B[k, 0:32]      Y[4:8] (x) X[0:4]

A 32x32 f16 tile occupies 32 of the 64 Z rows at stride 2, so exactly two fit,
and alternating fma16 between them is what reaches the throughput ceiling
(corsix/amx fma.md: 1453 GFLOPS with one accumulator, 2959 with two).  X and Y
hold 8 rows each, so a block of 4 k values fills X[0:4] with B rows and
Y[0:4] / Y[4:8] with the two halves of the A rows.

Loads need no special ordering: at 3 loads per 2 fma16 the coprocessor sustains
the same throughput as with no loads at all (measured on an M1 Max), so a block
simply loads its 12 rows and then issues its 8 fma16.
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
def rank_kx8_reduce_64x32(
  K: size, A: f16[K * 8, 64] @ DRAM, B: f16[K * 8, 32] @ DRAM, C: f16[64, 32] @ DRAM
):
  for i in seq(0, 64):
    for j in seq(0, 32):
      for k in seq(0, K * 8):
        C[i, j] += A[k, i] * B[k, j]

print("=============Original Matmul==============")
print(rank_kx8_reduce_64x32)

amx = rename(rank_kx8_reduce_64x32, "rank_kx8_reduce_64x32_scheduled_appleamx")

# 1. k outermost: each k is a rank-1 update of all of C.
amx = reorder_loops(amx, "j k")
amx = reorder_loops(amx, "i k")

# 2. C is two 32x32 tiles, each one Z accumulator.  Unrolling the tile loop makes
#    the two halves of C (and of each A row) distinct expressions, so each half
#    can be staged on its own.
amx = divide_loop(amx, "i", 32, ["tile", "i"], perfect=True)
amx = unroll_loop(amx, "tile")
amx = simplify(amx)
amx = stage_mem(amx, "for k in _:_", "C[0:32, 0:32]", "C_0")
amx = stage_mem(amx, "for k in _:_", "C[32:64, 0:32]", "C_1")

# 3. Blocks of 4 k: the B rows go to X, the two halves of the A rows to Y.
amx = divide_loop(amx, "k", 4, ["k0", "k1"], perfect=True)
amx = stage_mem(amx, "for k1 in _:_", "B[4 * k0:4 * k0 + 4, 0:32]", "B_x")
amx = stage_mem(amx, "for k1 in _:_", "A[4 * k0:4 * k0 + 4, 0:32]", "A_lo")
amx = stage_mem(amx, "for k1 in _:_", "A[4 * k0:4 * k0 + 4, 32:64]", "A_hi")
amx = simplify(amx)

# 4. Register files and instructions.
for name, pool in [("C_0", APPLE_AMX_POOL_Z), ("C_1", APPLE_AMX_POOL_Z), ("B_x", APPLE_AMX_POOL_X),
                   ("A_lo", APPLE_AMX_POOL_Y), ("A_hi", APPLE_AMX_POOL_Y)]:
  amx = set_memory(amx, name, pool)
for op in [apple_amx_ldz_f16, apple_amx_stz_f16, apple_amx_ldx_f16, apple_amx_ldy_f16, apple_amx_fma16_mat]:
  amx = replace_all(amx, op)

# 5. Unroll the k block so every register index is a constant.  The Z load and
#    store loops around the k0 loop stay rolled.
amx = unroll_loops(amx, amx.find_loop("k0"))
amx = simplify(amx)

print("=============Optimized Matmul==============")
print(amx)

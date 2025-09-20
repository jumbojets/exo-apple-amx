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
def rank_kx6_reduce_32x32(
  K: size, At: f16[K * 8, 32] @ DRAM, B: f16[K * 8, 32] @ DRAM, C: f16[32, 32] @ DRAM
):
  for i in seq(0, 32):
    for j in seq(0, 32):
      for k in seq(0, K * 8):
        C[i, j] += At[k, i] * B[k, j]

print("=============Original Matmul==============")
print(rank_kx6_reduce_32x32)

amx = rename(rank_kx6_reduce_32x32, "rank_kx6_reduce_32x32_scheduled_appleamx")
amx = reorder_loops(amx, "j k")
amx = reorder_loops(amx, "i k")

amx = stage_mem(amx, "for k in _:_", "C[0:32, 0:32]", "C_reg")
amx = simplify(amx)

amx = set_memory(amx, "C_reg:_", APPLE_AMX_POOL_Z)
amx = replace_all(amx, apple_amx_ldz_f16)
amx = replace_all(amx, apple_amx_stz_f16)
amx = simplify(amx)

amx = divide_loop(amx, "for k in _:_", 8, ["k0", "k1"], perfect=True)
amx = auto_stage_mem(amx, amx.find_loop("k1").expand(1, 0), "B", "B_reg")
amx = auto_stage_mem(amx, amx.find_loop("k1").expand(1, 0), "At", "At_reg")
amx = simplify(amx)

amx = set_memory(amx, "At_reg", APPLE_AMX_POOL_X)
amx = set_memory(amx, "B_reg", APPLE_AMX_POOL_Y)
amx = replace_all(amx, apple_amx_ldx_f16)
amx = replace_all(amx, apple_amx_ldy_f16)
amx = replace_all(amx, apple_amx_fma16_mat)
amx = simplify(amx)

# fuse the ldx, ldy, fma16 loops
loop = amx.find("At_reg:_").next()
amx = fuse(amx, loop, loop.next())
loop = amx.find("At_reg:_").next()
amx = fuse(amx, loop, loop.next())

print("=============Optimized Matmul==============")
print(amx)

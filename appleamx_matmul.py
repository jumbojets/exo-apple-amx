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
amx = reorder_loops(amx, "j k")
amx = reorder_loops(amx, "i k")

amx = stage_mem(amx, "for k in _:_", "C[0:64, 0:32]", "C_reg")
amx = divide_dim(amx, "C_reg", 0, 32)

# TODO: must we enumerate? ...this is a little gross
for i, c in enumerate(amx.find_all("for i0 in _:_")):
  i2, i3 = f"i2_{i}", f"i3_{i}"
  amx = divide_loop(amx, c, 32, [i2, i3], perfect=True)
  amx = simplify(amx)
  amx = unroll_loop(amx, f"for {i2} in _:_")
  loop = amx.find_loop(i3)
  amx = fuse(amx, loop, loop.next())
amx = simplify(amx)

amx = divide_loop(amx, "for i in _:_", 32, ["j0", "j1"], perfect=True)
amx = divide_loop(amx, "for k in _:_", 8, ["k0", "k1"], perfect=True)
amx = auto_stage_mem(amx, amx.find_loop("k1"), "B", "B_reg")
amx = divide_loop(amx, "for k1 in _:_", 4, ["k1", "k2"], perfect=True)
amx = reorder_loops(amx, "k2 j0")
amx = auto_stage_mem(amx, amx.find_loop("k2").expand(1, 0), "A", "A_reg")
amx = simplify(amx)

# NOTE: for some reason find_alloc_or_arg doesn't work with a_reg_1
# Chain many "nexts" together to find it
def next(c, n=1):
  for _ in range(n): c = c.next()
  return c

amx = unroll_loop(amx, "for j0 in _:_")
areg = amx.find_alloc_or_arg("A_reg") # NOTE: see above
amx = reorder_stmt_forward(amx, next(areg, n=2))
amx = reorder_stmt_forward(amx, next(areg))
amx = simplify(amx)

amx = unroll_loop(amx, "for k1 in _:_")
areg1 = next(amx.find_alloc_or_arg("A_reg"))
amx = reuse_buffer(amx, areg1, next(areg1, n=6)) # NOTE: see above
areg = amx.find_alloc_or_arg("A_reg")
amx = reuse_buffer(amx, areg, next(areg, n=6))

# TODO: instead of unrolling C_reg, we should implement a 3rd dim for APPLE_AMX_POOL_Z
amx = unroll_buffer(amx, "C_reg", 0)
amx = set_memory(amx, "C_reg_0", APPLE_AMX_POOL_Z)
amx = set_memory(amx, "C_reg_1", APPLE_AMX_POOL_Z)
amx = replace_all(amx, apple_amx_ldz_f16)
amx = replace_all(amx, apple_amx_stz_f16)
amx = simplify(amx)

amx = set_memory(amx, "A_reg", APPLE_AMX_POOL_X)
amx = set_memory(amx, next(amx.find_alloc_or_arg("A_reg")), APPLE_AMX_POOL_X) # NOTE: see above
amx = set_memory(amx, "B_reg", APPLE_AMX_POOL_Y)
amx = replace_all(amx, apple_amx_ldx_f16)
amx = replace_all(amx, apple_amx_ldy_f16)
amx = replace_all(amx, apple_amx_fma16_mat)
amx = simplify(amx)

amx = reorder_stmt_forward(amx, amx.find_loop("i0"))
amx = reorder_stmt_forward(amx, amx.find_loop("i0"))

for c in amx.find_loop("i0", many=True):
  if c.next().name() == "k2":
    amx = fuse(amx, c, c.next())
amx = simplify(amx)

amx = divide_loop(amx, "for i0 in _:_", 4, ["i1", "i2"], perfect=True)
amx = reorder_loops(amx, "i1 i2")
amx = unroll_loop(amx, "for i1 in _:_")
amx = fission(amx, amx.find_loop("i2").body()[0].after())
amx = reorder_stmt_forward(amx, amx.find_loop("i2", many=True)[1])
amx = reorder_stmt_forward(amx, amx.find_loop("i2", many=True)[1])
amx = simplify(amx)

for c in amx.find_loop("i2", many=True):
  amx = fuse(amx, next(c), next(c, n=2))
  amx = fuse(amx, c, next(c))
for c in amx.find_loop("i2", many=True):
  amx = reorder_stmt_forward(amx, c.body()[2])
for c in amx.find_loop("i2", many=True):
  amx = unroll_loop(amx, c)
amx = simplify(amx)

print("=============Optimized Matmul==============")
print(amx)

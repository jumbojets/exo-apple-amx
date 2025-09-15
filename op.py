from __future__ import annotations
import os
import sys

from exo import proc
from exo import *

from appleamx import *

if __name__ != "__main__" and hasattr(os, "devnull"):
    sys.stdout = open(os.devnull, "w")

@proc
def outer_product(A: f32[16] @ DRAM, B: f32[16] @ DRAM, C: f32[16, 16] @ DRAM):
  a: f32[16] @ APPLE_AMX_POOL_X
  b: f32[16] @ APPLE_AMX_POOL_Y
  c: f32[16, 16] @ APPLE_AMX_POOL_Z

  apple_amx_ldx_f32(a, A)
  apple_amx_ldy_f32(b, B)
  apple_amx_fma32_mat(c, a, b)
  for i in seq(0, 16): apple_amx_stz_f32(C[i, :], c[i, :])

print(outer_product)

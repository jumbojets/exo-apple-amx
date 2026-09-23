import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template

import pytest
from exo import compile_procs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import appleamx_ops
from appleamx_pools import APPLE_AMX_POOL_X, APPLE_AMX_POOL_Y, APPLE_AMX_POOL_Z
from gen_appleamx_ops import lanes

POOLS = {APPLE_AMX_POOL_X: "x", APPLE_AMX_POOL_Y: "y", APPLE_AMX_POOL_Z: "z"}
# Padding rows put the X, Y and Z operands on different nonzero registers
PAD_ROWS = {"x": 1, "y": 2, "z": 1}
INDEX_ARG = 3

CTYPES = {"f16": "_Float16", "f32": "float", "f64": "double", "i8": "int8_t", "ui8": "uint8_t", "ui16": "uint16_t", "i32": "int32_t"}
UNIT = "rand() / (RAND_MAX / 2.0) - 1"
# Integer ranges keep products clear of C int overflow
FILL = {"f16": UNIT, "f32": UNIT, "f64": UNIT, "i8": "rand() % 16 - 8", "ui8": "rand() % 16", "ui16": "rand() % 128", "i32": "rand() % 2000 - 1000"}
# Integers must match exactly
TOLERANCE = {"f16": 4e-3, "f32": 1e-6, "f64": 1e-14}

KERNELS = Template("""from __future__ import annotations
from exo import *
from exo.libs.externs import select
from exo.stdlib.stdlib import stride
from appleamx import *

@proc
def ref_kernel($sig):
$ref_body

@proc
def amx_kernel($sig):
$amx_body
""")

DRIVER = Template(r"""#include <math.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernel.h"

static int check(const char *name, int i, double ref, double amx, double tol) {
  if (fabs(ref - amx) <= tol * (1 + fabs(ref))) return 0;
  printf("%s[%d]: ref %g, amx %g\n", name, i, ref, amx);
  return 1;
}

int main(void) {
  int bad = 0;
  srand(1);
$init
  ref_kernel(NULL$ref_args);
  amx_kernel(NULL$amx_args);
$checks
  return bad != 0;
}
""")

@dataclass
class Arg:
  name: str
  dtype: str | None = None  # None for index arguments
  shape: tuple = ()
  pool: str | None = None  # None for DRAM

  @classmethod
  def of(cls, cursor):
    if not cursor.is_tensor(): return cls(cursor.name())
    return cls(cursor.name(), cursor.type().name.lower(), tuple(n.value() for n in cursor.shape()), POOLS.get(cursor.mem()))

  def type(self, mem="DRAM"):
    return f"{self.dtype}[{', '.join(map(str, self.shape))}] @ {mem}"

  def reg(self):
    return f"{self.name}_reg: {self.type(f'APPLE_AMX_POOL_{self.pool.upper()}')}"

  def window(self, name):
    return f"{name}[{', '.join(f'0:{n}' for n in self.shape)}]"

  def move(self, op, dst, src):
    # One register per call
    if len(self.shape) == 1: return [f"{op}({self.window(dst)}, {self.window(src)})"]
    return [f"for r in seq(0, {self.shape[0]}):", f"    {op}({dst}[r, 0:{self.shape[1]}], {src}[r, 0:{self.shape[1]}])"]

  def load(self, src):
    return self.move(f"apple_amx_ld{self.pool}_{self.dtype}", f"{self.name}_reg", src)

  def store(self, dst):
    return self.move(f"apple_amx_st{self.pool}_{self.dtype}", dst, f"{self.name}_reg")

def kernel_source(op, args):
  """ref_kernel runs the instruction's own body on DRAM; amx_kernel stages its operands through the pools and calls it"""
  lines = str(op).splitlines()
  ref_body = lines[1 + next(i for i, l in enumerate(lines) if "# @instr" in l):]
  asserts = [l.strip() for l in ref_body if l.strip().startswith("assert") and "stride" not in l]

  staged = [a for a in args if a.pool]
  pads = [Arg(f"pad_{a.pool}", a.dtype, (PAD_ROWS[a.pool], lanes(a.dtype)), a.pool)
          for a in {a.pool: a for a in staged}.values() if a.shape[0] < 64]
  loads, stores = [], []
  for pad in pads:
    loads += [f"{pad.name}_dram: {pad.type()}", pad.reg(), *pad.load(f"{pad.name}_dram")]
    stores += pad.store(f"{pad.name}_dram")
  for a in staged:
    loads += [a.reg(), *a.load(a.name)]
    stores = a.store(a.name) + stores
  call = [a.window(f"{a.name}_reg") if a.pool else a.window(a.name) if a.dtype else a.name for a in args]

  amx_body = asserts + loads + [f"{op.name()}({', '.join(call)})"] + stores
  return KERNELS.substitute(
    sig=", ".join(f"{a.name}: {a.type() if a.dtype else 'index'}" for a in args),
    ref_body="\n".join(ref_body),
    amx_body="\n".join(f"    {l}" for l in amx_body),
  )

def driver_source(args):
  init, ref_args, amx_args, checks = [], "", "", []
  for a in args:
    if not a.dtype:
      ref_args += f", {INDEX_ARG}"
      amx_args += f", {INDEX_ARG}"
      continue
    n = math.prod(a.shape)
    init += [f"  static alignas(128) {CTYPES[a.dtype]} {a.name}_ref[{n}], {a.name}_amx[{n}];",
             f"  for (int i = 0; i < {n}; i++) {a.name}_ref[i] = {FILL[a.dtype]};",
             f"  memcpy({a.name}_amx, {a.name}_ref, sizeof({a.name}_ref));"]
    ref_args += f", {a.name}_ref"
    amx_args += f", {a.name}_amx"
    checks += [f'  for (int i = 0; i < {n}; i++) bad += check("{a.name}", i, {a.name}_ref[i], {a.name}_amx[i], {TOLERANCE.get(a.dtype, 0)});']
  return DRIVER.substitute(init="\n".join(init), ref_args=ref_args, amx_args=amx_args, checks="\n".join(checks))

@pytest.fixture
def build_driver(tmp_path):
  """Compile a driver that compares an op against its own Exo semantics on DRAM"""
  def build(op_name):
    op = getattr(appleamx_ops, op_name)
    args = [Arg.of(a) for a in op.args()]
    (tmp_path / "kernels.py").write_text(kernel_source(op, args))
    spec = importlib.util.spec_from_file_location(f"kernels_{op_name}", tmp_path / "kernels.py")
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    compile_procs([kernels.ref_kernel, kernels.amx_kernel], tmp_path, "kernel.c", "kernel.h")
    (tmp_path / "driver.c").write_text(driver_source(args))
    subprocess.run(["cc", "-O2", "-march=native", f"-I{ROOT}", f"-I{tmp_path}", "-o", tmp_path / "driver",
                    tmp_path / "kernel.c", tmp_path / "driver.c"], check=True)
    return tmp_path / "driver"
  return build

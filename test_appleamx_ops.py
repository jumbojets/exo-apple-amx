import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import gen_appleamx_ops as gen  # noqa: E402

CTYPES = {"f16": "_Float16", "f32": "float", "f64": "double",
          "i8": "int8_t", "ui8": "uint8_t", "ui16": "uint16_t", "i32": "int32_t"}

# For each op: the size-parameter values to run it with. Masked ops get a
# partial mask and a full one (the full lane count encodes as 0 in the flag).
def cases(o):
  sizes = [p.name for p in o.params if p.dtype is None]
  if not sizes: return [("", {})]
  N = o.params[len(sizes)].shape[0]  # dst lanes, which for f32 accumulation exceed lanes(dtype)
  partial = {"rows": N - 1, "cols": N // 2 + 1, "n": N - 1}
  return [("_partial", {s: partial[s] for s in sizes}), ("_full", {s: N for s in sizes})]

def tensor_param(p):
  if p.dtype is None: return f"{p.name}: size"
  return f"{p.name}: {p.dtype}[{', '.join(map(str, p.shape))}] @ DRAM"

def ref_proc_source(o):
  """The instruction's own body as a plain proc with every operand in DRAM."""
  lines = [f"def r_{o.name}({', '.join(map(tensor_param, o.params))}):"]
  lines += [f"  assert {a}" for a in o.asserts]
  lines += [f"  {line}" for line in o.body]
  return "@proc\n" + "\n".join(lines) + "\n"

def dense_strides(shape):
  strides, n = [], 1
  for d in reversed(shape):
    strides.insert(0, n)
    n *= d
  return strides

def move(instr, dst, src, shape):
  """Exo statements copying a whole buffer row by row with a ld/st instruction."""
  if len(shape) == 1: return [f"  {instr}({dst}, {src})"]
  return [f"  for r in seq(0, {shape[0]}):", f"    {instr}({dst}[r, 0:{shape[1]}], {src}[r, 0:{shape[1]}])"]

def staging_ops(p):
  """(load, store) instructions moving one row of register operand p; wide Z rows use ldzi/stzi."""
  name = p.mem.lower() + ("i" if p.shape[-1] > gen.lanes(p.dtype) else "")
  return f"apple_amx_ld{name}_{p.dtype}", f"apple_amx_st{name}_{p.dtype}"

def test_proc_source(o, label, sizes):
  """A proc that stages every register operand from DRAM, runs the op, and stores it back."""
  params, asserts, before, after, args = [], [], [], [], []
  for p in o.params:
    if p.dtype is None:
      args.append(str(sizes[p.name]))
      continue
    params.append(tensor_param(p))
    # Exo does not assume proc arguments are dense; the instruction asserts need it.
    asserts += [f"  assert stride({p.name}, {d}) == {stride}" for d, stride in enumerate(dense_strides(p.shape))]
    if p.mem == "DRAM":
      args.append(p.name)
      continue
    reg = f"{p.name}_reg"
    before.append(f"  {reg}: {p.dtype}[{', '.join(map(str, p.shape))}] @ {gen.MEMS[p.mem]}")
    ld, st = staging_ops(p)
    before += move(ld, reg, p.name, p.shape)
    after += move(st, p.name, reg, p.shape)
    args.append(reg)
  body = asserts + before + [f"  {o.name}({', '.join(args)})"] + after
  return f"@proc\ndef t_{o.name}{label}({', '.join(params)}):\n" + "\n".join(body) + "\n"

def driver_case(o, label, sizes):
  """C block that runs the reference and the test proc on the same random data."""
  bufs = [p for p in o.params if p.dtype is not None]
  lines = ["{"]
  for p in bufs:
    n, T = 1, CTYPES[p.dtype]
    for d in p.shape: n *= d
    fill = "(rand() % 7 - 3)" if p.dtype in gen.FP_TYPES else "rand()"
    lines.append(f"  alignas(128) {T} {p.name}[{n}], {p.name}_t[{n}];")
    lines.append(f"  for (size_t i = 0; i < {n}; i++) {p.name}[i] = ({T}){fill};")
    lines.append(f"  memcpy({p.name}_t, {p.name}, sizeof({p.name}));")
  size_args = "".join(f"{sizes[p.name]}, " for p in o.params if p.dtype is None)
  lines.append(f"  r_{o.name}(NULL, {size_args}{', '.join(p.name for p in bufs)});")
  lines.append(f"  t_{o.name}{label}(NULL, {', '.join(p.name + '_t' for p in bufs)});")
  for p in bufs:
    lines.append(f'  check("{o.name}{label}", "{p.name}", {p.name}, {p.name}_t, sizeof({p.name}), sizeof({CTYPES[p.dtype]}));')
  lines.append("}")
  return "\n".join(lines)

DRIVER_HEADER = """\
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "amx_test.h"

static int failures = 0;

static void check(const char *op, const char *buf, const void *ref, const void *test, size_t bytes, size_t elem) {
  const unsigned char *r = ref, *t = test;
  for (size_t i = 0; i < bytes; i++) {
    if (r[i] != t[i]) {
      printf("FAIL %s: %s differs at element %zu\\n", op, buf, i / elem);
      failures++;
      return;
    }
  }
}

int main() {
  srand(1);
"""

def load_module(path):
  spec = importlib.util.spec_from_file_location(path.stem, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod

def test_generated_file_is_current():
  assert gen.render_module() == gen.OUTPUT.read_text(), "run gen_appleamx_ops.py"

def test_every_instruction_executes_correctly():
  from exo import compile_procs_to_strings

  with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    runs = [(o, label, sizes) for o in gen.OPS for label, sizes in cases(o)]
    src = "from __future__ import annotations\nfrom exo import *\nfrom appleamx import *\n"
    src += "".join(ref_proc_source(o) for o in gen.OPS)
    src += "".join(test_proc_source(*run) for run in runs)
    (tmp / "amx_test_procs.py").write_text(src)
    mod = load_module(tmp / "amx_test_procs.py")
    procs = [getattr(mod, f"r_{o.name}") for o in gen.OPS]
    procs += [getattr(mod, f"t_{o.name}{label}") for o, label, _ in runs]

    c, h = compile_procs_to_strings(procs, "amx_test.h")
    assert c.count("AMX_SET()") == c.count("AMX_CLR()") == len(runs), "one AMX_SET/AMX_CLR pair per test proc"
    (tmp / "amx_test.c").write_text(c)
    (tmp / "amx_test.h").write_text(h)
    driver = DRIVER_HEADER + "\n".join(driver_case(*run) for run in runs)
    driver += '\n  printf("%d failures in %d runs\\n", failures, ' + str(len(runs)) + ");\n  return failures != 0;\n}\n"
    (tmp / "driver.c").write_text(driver)
    shutil.copy(HERE / "amx.h", tmp / "amx.h")

    cc = subprocess.run(["cc", "-march=native", "-O1", "-Wall", "-Werror", "amx_test.c", "driver.c", "-o", "driver"],
                        cwd=tmp, capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    run = subprocess.run(["./driver"], cwd=tmp, capture_output=True, text=True)
    print(run.stdout, end="")
    assert run.returncode == 0, run.stdout

if __name__ == "__main__":
  test_generated_file_is_current()
  test_every_instruction_executes_correctly()

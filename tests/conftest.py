import ctypes
import importlib.util
import random
import struct
import subprocess
from pathlib import Path

import pytest
from exo import DRAM, compile_procs
from exo.API_cursors import ForCursor, WindowStmtCursor
from exo.stdlib.scheduling import inline, inline_window, rename, set_memory

REPO = Path(__file__).resolve().parent.parent
FORMATS = {"f16": "e", "f32": "f", "f64": "d", "i8": "b", "ui8": "B", "ui16": "H", "i32": "i"}
INT_RANGES = {"i8": (-128, 127), "ui8": (0, 255), "ui16": (0, 65535), "i32": (-(1 << 20), 1 << 20)}
POOLS = {"APPLE_AMX_POOL_X": "x", "APPLE_AMX_POOL_Y": "y", "APPLE_AMX_POOL_Z": "z"}

def signature(proc):
  return [(a.name(), a.type().name.lower(), [e.value() for e in a.shape()], a.mem().name())
          for a in proc.args() if a.is_tensor()]

def copy(instr, dst, src, shape, dst_index=""):
  """instr from src[r, :] to dst[dst_index, r, :] for every row r."""
  if len(shape) == 1:
    return [f"{instr}({dst}[{dst_index}:], {src})" if dst_index else f"{instr}({dst}, {src})"]
  return [f"for r in seq(0, {shape[0]}):", f"  {instr}({dst}[{dst_index}r, :], {src}[r, :])"]

def driver_source(op):
  """Exo source that loads random data into every AMX operand of op, calls op and stores every operand."""
  tensors = signature(op)
  lanes = 64 // struct.calcsize(FORMATS[tensors[0][1]])
  index = [a.name() for a in op.args() if not a.is_tensor()]
  sweep = f"{index[0]}, " if index else ""
  params, setup, stores, pads, instrs, allocs = [], [], [], [], {op.name()}, []
  for name, dtype, shape, mem in tensors:
    decl = f"{dtype}[{', '.join(map(str, shape))}]"
    pool = POOLS.get(mem)
    if pool is None:
      params.append(f"{name}: {decl} @ DRAM")
      continue
    ld, st = f"apple_amx_ld{pool}_{dtype}", f"apple_amx_st{pool}_{dtype}"
    instrs |= {ld, st}
    out_shape = [lanes] * len(index) + shape
    params += [f"{name}_in: {decl} @ DRAM", f"{name}_out: {dtype}[{', '.join(map(str, out_shape))}] @ DRAM"]
    # a live buffer ahead of the operand so it doesn't start at register 0
    if shape != [64, 64]:
      params.append(f"{name}_pad_out: {dtype}[{shape[-1]}] @ DRAM")
      first_row = f"{name}_in[0, :]" if len(shape) == 2 else f"{name}_in"
      setup += [f"{name}_pad: {dtype}[{shape[-1]}] @ {mem}", f"{ld}({name}_pad, {first_row})"]
      pads.append(f"{st}({name}_pad_out, {name}_pad)")
      allocs.append(f"{name}_pad")
    setup.append(f"{name}: {decl} @ {mem}")
    allocs.append(name)
    setup += copy(ld, name, f"{name}_in", shape)
    stores += copy(st, f"{name}_out", name, shape, sweep)
  run = [f"{op.name()}({', '.join(a.name() for a in op.args())})", *stores]
  if index:
    run = [f"for {index[0]} in seq(0, {lanes}):", *("  " + line for line in run)]
  lines = [
    "from __future__ import annotations",
    "from exo import proc",
    "from appleamx import *",
    "",
    "@proc",
    f"def drive_{op.name()}({', '.join(params)}):",
    *("  " + line for line in setup + run + pads),
  ]
  return "\n".join(lines) + "\n", instrs, allocs

def statements(block):
  for s in block:
    yield s
    if isinstance(s, ForCursor):
      yield from statements(s.body())

def reference(drive, instrs, allocs):
  """drive with every instr inlined and every buffer in DRAM."""
  ref = rename(drive, drive.name().replace("drive_", "ref_", 1))
  for instr in instrs:
    for call in ref.find(f"{instr}(_)", many=True):
      ref = inline(ref, call)
  for window in [s for s in statements(ref.body()) if isinstance(s, WindowStmtCursor)]:
    ref = inline_window(ref, window)
  for alloc in allocs:
    ref = set_memory(ref, alloc, DRAM)
  return ref

def random_buffer(dtype, shape, rng):
  n = 1
  for extent in shape: n *= extent
  if dtype in INT_RANGES:
    values = [rng.randint(*INT_RANGES[dtype]) for _ in range(n)]
  else:
    values = [float(rng.randint(-4, 4)) for _ in range(n)]
  return bytearray(struct.pack(f"<{n}{FORMATS[dtype]}", *values))

class Driver:
  """A compiled driver and reference for one op."""

  def __init__(self, op, workdir):
    source, instrs, allocs = driver_source(op)
    path = workdir / f"drive_{op.name()}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    drive = getattr(module, path.stem)
    ref = reference(drive, instrs, allocs)
    compile_procs([drive, ref], workdir, f"{op.name()}.c", f"{op.name()}.h")
    lib = workdir / f"{op.name()}.dylib"
    subprocess.run(["clang", "-O2", "-march=native", "-shared", "-fPIC", f"-I{REPO}",
                    "-o", str(lib), str(workdir / f"{op.name()}.c")], check=True)
    self.lib = ctypes.CDLL(str(lib))
    self.name = op.name()
    self.params = [(name, dtype, shape) for name, dtype, shape, _ in signature(drive)]

  def inputs(self, seed=0):
    rng = random.Random(seed)
    return [random_buffer(dtype, shape, rng) for _, dtype, shape in self.params]

  def run(self, prefix, inputs):
    """Runs drive_ or ref_ on copies of inputs and returns every buffer decoded."""
    buffers = [bytearray(b) for b in inputs]
    getattr(self.lib, prefix + self.name)(None, *((ctypes.c_char * len(b)).from_buffer(b) for b in buffers))
    return {name: struct.unpack(f"<{len(b) // struct.calcsize(FORMATS[dtype])}{FORMATS[dtype]}", b)
            for (name, dtype, _), b in zip(self.params, buffers)}

@pytest.fixture(scope="session")
def build_driver(tmp_path_factory):
  workdir = tmp_path_factory.mktemp("drivers")
  return lambda op: Driver(op, workdir)

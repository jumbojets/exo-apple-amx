"""Check every generated Apple AMX instruction against the hardware, and the matmul example end to end.

For each instruction, the Exo body (with instr and memory annotations stripped) is compiled as a C
reference. Both the reference and the AMX instruction run on the same random register file, with
random register rows, and the entire X/Y/Z state must match exactly afterwards.

Run with: python test_appleamx_ops.py
"""

import ctypes
import platform
import random
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exo.API import Procedure, compile_procs_to_strings
from exo.core.memory import DRAM

import appleamx_ops

ROOT = Path(__file__).resolve().parent
TRIALS = 4
# ctype: (struct format, bytes, is float)
CTYPES = {
  "_Float16": ("e", 2, True), "float": ("f", 4, True), "double": ("d", 8, True),
  "int8_t": ("b", 1, False), "uint8_t": ("B", 1, False), "uint16_t": ("H", 2, False), "int32_t": ("i", 4, False),
}

def build_library(tmp, name, c_files):
  # -fwrapv: ui16 * ui16 promotes to int in C and may overflow; AMX wraps
  lib = Path(tmp) / f"lib{name}.dylib"
  subprocess.run(["cc", "-O1", "-shared", "-fwrapv", "-ffp-contract=fast", f"-I{ROOT}", f"-I{tmp}",
                  "-o", str(lib), *map(str, c_files)], check=True)
  return ctypes.CDLL(str(lib))

def random_values(rng, ctype, count):
  fmt, nbytes, is_float = CTYPES[ctype]
  if is_float:
    # Multiples of 1/8 keep products and sums exact in every float width, so results compare exactly
    return [rng.randint(-32, 32) / 8 for _ in range(count)]
  lo, hi = (0, 256 ** nbytes - 1) if fmt.isupper() else (-(256 ** nbytes) // 2, 256 ** nbytes // 2 - 1)
  return [rng.randint(lo, hi) for _ in range(count)]

def fill(rng, ctype, nbytes_total):
  fmt, nbytes, _ = CTYPES[ctype]
  count = nbytes_total // nbytes
  return bytearray(struct.pack(f"<{count}{fmt}", *random_values(rng, ctype, count)))

def address(buf, offset=0):
  return ctypes.addressof((ctypes.c_uint8 * len(buf)).from_buffer(buf)) + offset

class Arg:
  def __init__(self, a):
    self.name = str(a.name)
    self.scalar = str(a.type) if str(a.type) in ("size", "index") else None
    if not self.scalar:
      self.ctype = a.type.basetype().ctype()
      self.shape = [int(h.val) for h in a.type.shape()]
      self.pool = a.mem.__name__[-1] if a.mem is not DRAM else None

class Instr:
  def __init__(self, proc):
    self.root = proc._loopir_proc
    self.name = self.root.name
    self.args = [Arg(a) for a in self.root.args]
    self.ctype = next(a.ctype for a in self.args if not a.scalar)

  def reference(self):
    return Procedure(self.root.update(name=f"ref_{self.name}", instr=None,
                                      args=[a.update(mem=DRAM) for a in self.root.args]))

  def harness(self, params):
    ref_args, fmt = [], {}
    for i, (arg, param) in enumerate(zip(self.args, params)):
      if arg.scalar:
        ref_args.append(f"(int_fast32_t)s[{i}]")
        fmt[arg.name] = f"(s[{i}])"
        continue
      strides = ", ".join(map(str, [arg.shape[1], 1] if len(arg.shape) == 2 else [1]))
      ref_args.append(f"({param.rsplit(' ', 1)[0]}){{({arg.ctype} *)p[{i}], {{{strides}}}}}")
      fmt[arg.name] = fmt[f"{arg.name}_data"] = f"(s[{i}])" if arg.pool else f"(({arg.ctype} *)p[{i}])[0]"
    return (f"void call_ref_{self.name}(void **p, int64_t *s) {{ ref_{self.name}(NULL, {', '.join(ref_args)}); }}\n"
            f"void call_amx_{self.name}(uint8_t *X, uint8_t *Y, uint8_t *Z, void **p, int64_t *s) {{\n"
            f"  amx_load(X, Y, Z);\n  {self.root.instr.c_instr.format(**fmt)}\n  amx_store(X, Y, Z);\n}}\n")

HARNESS_PRELUDE = r'''
#include <stddef.h>
#include <stdint.h>
#include "amx.h"
#include "ref.h"

static void amx_load(uint8_t *X, uint8_t *Y, uint8_t *Z) {
  AMX_SET();
  for (int r = 0; r < 8; r++) { AMX_LDX(X + 64 * r, r, 0); AMX_LDY(Y + 64 * r, r, 0); }
  for (int r = 0; r < 64; r++) AMX_LDZ(Z + 64 * r, r, 0);
}

static void amx_store(uint8_t *X, uint8_t *Y, uint8_t *Z) {
  for (int r = 0; r < 8; r++) { AMX_STX(X + 64 * r, r, 0); AMX_STY(Y + 64 * r, r, 0); }
  for (int r = 0; r < 64; r++) AMX_STZ(Z + 64 * r, r, 0);
  AMX_CLR();
}
'''

@unittest.skipUnless(sys.platform == "darwin" and platform.machine() == "arm64", "requires Apple silicon")
class TestAppleAMXOps(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.tmp = tempfile.TemporaryDirectory()
    cls.instrs = [Instr(getattr(appleamx_ops, n)) for n in dir(appleamx_ops) if n.startswith("apple_amx_")]
    c_src, h_src = compile_procs_to_strings([i.reference() for i in cls.instrs], "ref.h")
    (Path(cls.tmp.name) / "ref.c").write_text(c_src)
    (Path(cls.tmp.name) / "ref.h").write_text(h_src)
    harness = [HARNESS_PRELUDE]
    for instr in cls.instrs:
      decl = re.search(rf"void ref_{instr.name}\( void \*ctxt, (.*) \);", h_src).group(1)
      harness.append(instr.harness(decl.split(", ")))
    (Path(cls.tmp.name) / "harness.c").write_text("\n".join(harness))
    cls.lib = build_library(cls.tmp.name, "ops", [Path(cls.tmp.name) / "ref.c", Path(cls.tmp.name) / "harness.c"])

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()

  def test_instructions(self):
    self.assertGreater(len(self.instrs), 0)
    for instr in self.instrs:
      for trial in range(TRIALS):
        with self.subTest(instr=instr.name, trial=trial):
          self.check(instr, random.Random(f"{instr.name}:{trial}"))

  def check(self, instr, rng):
    fmt, nbytes, _ = CTYPES[instr.ctype]
    state = {pool: fill(rng, instr.ctype, size) for pool, size in (("X", 512), ("Y", 512), ("Z", 4096))}
    mem = {i: fill(rng, instr.ctype, 64) for i, a in enumerate(instr.args) if not a.scalar and not a.pool}
    free_rows = {"X": rng.sample(range(8), 8), "Y": rng.sample(range(8), 8), "Z": rng.sample(range(64), 64)}
    s = []
    for arg in instr.args:
      if arg.scalar == "size": s.append(rng.randint(1, 64 // nbytes))
      elif arg.scalar == "index": s.append(rng.randrange(64 // nbytes))
      elif arg.pool == "Z" and len(arg.shape) == 2: s.append(rng.randrange(64 // arg.shape[0]))
      elif arg.pool: s.append(free_rows[arg.pool].pop())
      else: s.append(0)
    s_arr = (ctypes.c_int64 * len(s))(*s)

    amx = {k: bytearray(v) for k, v in state.items()}
    amx_mem = {k: bytearray(v) for k, v in mem.items()}
    amx_p = (ctypes.c_void_p * len(s))(*[address(amx_mem[i]) if i in amx_mem else None for i in range(len(s))])
    getattr(self.lib, f"call_amx_{instr.name}")(*[ctypes.c_void_p(address(amx[k])) for k in "XYZ"], amx_p, s_arr)

    ref = {k: bytearray(v) for k, v in state.items()}
    ref_mem = {k: bytearray(v) for k, v in mem.items()}
    gathered, ref_p = {}, []
    for i, arg in enumerate(instr.args):
      if arg.scalar: ref_p.append(None)
      elif not arg.pool: ref_p.append(address(ref_mem[i]))
      elif len(arg.shape) == 1: ref_p.append(address(ref[arg.pool], 64 * s[i]))
      else:
        rows = [s[i] + r * (64 // arg.shape[0]) for r in range(arg.shape[0])]
        gathered[i] = (rows, bytearray(b"".join(ref["Z"][64 * r:64 * r + 64] for r in rows)))
        ref_p.append(address(gathered[i][1]))
    getattr(self.lib, f"call_ref_{instr.name}")((ctypes.c_void_p * len(s))(*ref_p), s_arr)
    for rows, buf in gathered.values():
      for k, r in enumerate(rows): ref["Z"][64 * r:64 * r + 64] = buf[64 * k:64 * k + 64]

    params = {a.name: v for a, v in zip(instr.args, s)}
    for label, want, got in [*((k, ref[k], amx[k]) for k in "XYZ"), *((f"mem{k}", ref_mem[k], amx_mem[k]) for k in mem)]:
      count = len(want) // nbytes
      want_vals, got_vals = struct.unpack(f"<{count}{fmt}", want), struct.unpack(f"<{count}{fmt}", got)
      bad = [(k // (64 // nbytes), k % (64 // nbytes), w, g) for k, (w, g) in enumerate(zip(want_vals, got_vals)) if w != g]
      if bad: self.fail(f"{instr.name} {params}: {label} mismatches (row, lane, want, got): {bad[:6]}")

@unittest.skipUnless(sys.platform == "darwin" and platform.machine() == "arm64", "requires Apple silicon")
class TestAppleAMXMatmul(unittest.TestCase):
  def test_scheduled_matches_naive(self):
    stdout = sys.stdout
    try:
      import appleamx_matmul  # silences stdout on import
    finally:
      if sys.stdout is not stdout: sys.stdout.close()
      sys.stdout = stdout
    K = 3
    with tempfile.TemporaryDirectory() as tmp:
      c_src, h_src = compile_procs_to_strings([appleamx_matmul.rank_kx8_reduce_64x32, appleamx_matmul.amx], "matmul.h")
      (Path(tmp) / "matmul.c").write_text(c_src)
      (Path(tmp) / "matmul.h").write_text(h_src)
      lib = build_library(tmp, "matmul", [Path(tmp) / "matmul.c"])
      rng = random.Random(0)
      # Half-integers in [-1, 1] keep every partial sum exact in f16
      A = struct.pack(f"<{K * 8 * 64}e", *(rng.choice((-1, -0.5, 0, 0.5, 1)) for _ in range(K * 8 * 64)))
      B = struct.pack(f"<{K * 8 * 32}e", *(rng.choice((-1, -0.5, 0, 0.5, 1)) for _ in range(K * 8 * 32)))
      results = []
      for fn in (lib.rank_kx8_reduce_64x32, lib.rank_kx8_reduce_64x32_scheduled_appleamx):
        C = bytearray(64 * 32 * 2)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
        fn(None, K, A, B, address(C))
        results.append(struct.unpack(f"<{64 * 32}e", C))
      bad = [(k // 32, k % 32, w, g) for k, (w, g) in enumerate(zip(*results)) if w != g]
      if bad: self.fail(f"{len(bad)} mismatches (i, j, want, got): {bad[:6]}")

if __name__ == "__main__":
  unittest.main()

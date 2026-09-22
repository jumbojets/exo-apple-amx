"""Builds the C drivers that run the tests on the coprocessor."""
import subprocess

import pytest
from exo import compile_procs_to_strings

import appleamx

DRIVER_HEAD = """\
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

// Reports the first element where a test buffer differs from its reference.
static void check(const char *name, const char *buf, const void *ref, const void *test, size_t bytes, size_t elem) {
  const unsigned char *r = ref, *t = test;
  for (size_t i = 0; i < bytes; i++)
    if (r[i] != t[i]) { printf("FAIL %s: %s differs at element %zu\\n", name, buf, i / elem); failures++; return; }
}

"""

DRIVER_MAIN = """\
// Runs the cases named on the command line, or all of them.
int main(int argc, char **argv) {
  size_t n = sizeof(cases) / sizeof(*cases);
  for (int a = 1; a < argc; a++) {
    size_t i = 0;
    while (i < n && strcmp(cases[i].name, argv[a]) != 0) i++;
    if (i == n) { printf("no case %s\\n", argv[a]); return 2; }
    cases[i].run();
  }
  if (argc == 1) for (size_t i = 0; i < n; i++) cases[i].run();
  return failures != 0;
}
"""

def driver_source(header, cases):
  """C driver with one function per case, each starting from the same rand() seed."""
  funcs = "".join(f"static void run_{name}(void) {{\n  srand(1);\n{body}\n}}\n\n" for name, body in cases.items())
  table = "".join(f'  {{"{name}", run_{name}}},\n' for name in cases)
  return (f'#include "{header}"\n' + DRIVER_HEAD + funcs
          + "static const struct { const char *name; void (*run)(void); } cases[] = {\n" + table + "};\n\n" + DRIVER_MAIN)

@pytest.fixture(scope="session")
def build_driver():
  """Compiles the procs and a driver for the cases into path; returns a function running one case."""
  def build(path, name, procs, cases):
    c, h = compile_procs_to_strings(procs, f"{name}.h")
    (path / f"{name}.c").write_text(c)
    (path / f"{name}.h").write_text(h)
    (path / "driver.c").write_text(driver_source(f"{name}.h", cases))
    cc = subprocess.run(["cc", "-march=native", "-O1", "-Wall", "-Werror", f"-I{appleamx.include_dir()}",
                         f"{name}.c", "driver.c", "-o", "driver"], cwd=path, capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    def run(case):
      r = subprocess.run(["./driver", case], cwd=path, capture_output=True, text=True)
      assert r.returncode == 0, r.stdout
    return run
  return build

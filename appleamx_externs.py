from exo.core.extern import Extern, _EErr

class _MinMax(Extern):
  def typecheck(self, args):
    if len(args) != 2:
      raise _EErr(f"expected 2 arguments, got {len(args)}")
    for i, arg in enumerate(args):
      if not arg.type.is_real_scalar():
        raise _EErr(f"expected argument {i + 1} to be a real scalar value, but got type {arg.type}")
    return args[0].type

  def globl(self, prim_type):
    return "#include <math.h>"

  def compile(self, args, prim_type):
    return f"({prim_type}){self.name()}({args[0]}, {args[1]})"

fmin = _MinMax("fmin")
fmax = _MinMax("fmax")

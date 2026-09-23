import pytest
from exo import Procedure

import appleamx_ops

OPS = [name for name, value in vars(appleamx_ops).items() if isinstance(value, Procedure)]

@pytest.mark.parametrize("op", OPS)
def test_op(op, build_driver):
  driver = build_driver(getattr(appleamx_ops, op))
  inputs = driver.inputs()
  assert driver.run("drive_", inputs) == driver.run("ref_", inputs)

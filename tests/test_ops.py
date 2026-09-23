import subprocess

import pytest

import appleamx_ops

OPS = [name for name in vars(appleamx_ops) if name.startswith("apple_amx_")]

@pytest.mark.parametrize("op", OPS)
def test_op(op, build_driver):
  result = subprocess.run([build_driver(op)], capture_output=True, text=True)
  assert result.returncode == 0, result.stdout

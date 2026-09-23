PYTHON ?= python3
PIP := $(PYTHON) -m pip

CPPFLAGS += -I.
CFLAGS ?= -march=native

PKG_SRCS := appleamx.py setup.py pyproject.toml README.md
LIB_SRCS := appleamx_ops.py appleamx_pools.py appleamx_externs.py

appleamx_matmul: appleamx_matmul.o main.o

appleamx.install: $(PKG_SRCS)
	$(PIP) install -e .
	@touch $@

appleamx_matmul.c: appleamx_matmul.py appleamx.install $(LIB_SRCS)
	exocc -o . --stem $(*F) $<

appleamx_matmul.o: amx.h

main.c: appleamx_matmul.c

.PHONY: test
test: appleamx.install
	$(PYTHON) -m pytest tests

.PHONY: clean
clean:
	$(RM) appleamx_matmul appleamx_matmul.[cdh] *.o exo_demo
	$(RM) -r __pycache__/

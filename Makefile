PYTHON ?= python3
PIP := $(PYTHON) -m pip
EXOCC ?= exocc

CPPFLAGS += -I. -Iappleamx
CFLAGS ?= -O2 -march=native

PKG_SRCS := $(wildcard appleamx/*.py)

appleamx_matmul: appleamx_matmul.o main.o

appleamx.install: pyproject.toml README.md
	$(PIP) install -e '.[test]'
	@touch $@

appleamx_matmul.c: examples/appleamx_matmul.py $(PKG_SRCS) appleamx.install
	$(EXOCC) -o . --stem $(*F) $<

appleamx_matmul.o: appleamx/amx.h

main.o: examples/main.c appleamx_matmul.c
	$(COMPILE.c) $(OUTPUT_OPTION) $<

.PHONY: test
test: appleamx.install
	$(PYTHON) -m appleamx._gen_ops --check
	$(PYTHON) -m pytest tests

.PHONY: clean
clean:
	$(RM) appleamx_matmul appleamx_matmul.[cdh] *.o
	$(RM) -r __pycache__ appleamx/__pycache__ tests/__pycache__ .pytest_cache

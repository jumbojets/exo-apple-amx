PYTHON ?= python3
PIP := $(PYTHON) -m pip

CPPFLAGS += -I.
CFLAGS ?= -march=native

PKG_SRCS := appleamx.py setup.py pyproject.toml README.md

appleamx_matmul: appleamx_matmul.o main.o

appleamx.install: $(PKG_SRCS)
	$(PIP) install -e .
	@touch $@

appleamx_matmul.c: appleamx_matmul.py appleamx.install
	exocc -o . --stem $(*F) $^

appleamx_matmul.o: amx.h

main.c: appleamx_matmul.c

.PHONY: clean
clean:
	$(RM) appleamx_matmul appleamx_matmul.[cdh] *.o exo_demo
	$(RM) -r __pycache__/

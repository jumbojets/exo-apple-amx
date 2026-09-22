PYTHON ?= python3
PIP := $(PYTHON) -m pip
EXOCC ?= exocc

OUT := build
CPPFLAGS += -I$(OUT) -Iappleamx
CFLAGS ?= -O2 -march=native

PKG_SRCS := $(wildcard appleamx/*.py)
EXAMPLES := $(addprefix $(OUT)/,$(basename $(notdir $(wildcard examples/*.py))))

.PHONY: all
all: $(EXAMPLES)

$(EXAMPLES): %: %.o %_main.o

appleamx.install: pyproject.toml README.md
	$(PIP) install -e '.[test]'
	@touch $@

.SECONDARY:  # keep the generated .c and .h
$(OUT)/%.c $(OUT)/%.h: examples/%.py $(PKG_SRCS) appleamx.install | $(OUT)
	$(EXOCC) -o $(OUT) --stem $* $<

$(EXAMPLES:=.o): appleamx/amx.h

$(OUT)/%_main.o: examples/%_main.c $(OUT)/%.h
	$(COMPILE.c) $(OUTPUT_OPTION) $<

$(OUT):
	mkdir -p $@

.PHONY: test
test: appleamx.install
	$(PYTHON) -m appleamx._gen_ops --check
	$(PYTHON) -m pytest tests

.PHONY: clean
clean:
	$(RM) -r $(OUT) __pycache__ appleamx/__pycache__ examples/__pycache__ tests/__pycache__ .pytest_cache

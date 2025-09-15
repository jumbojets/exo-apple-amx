CPPFLAGS += -I.
CFLAGS ?= -march=native

op: op.o main.o

appleamx.py:
	pip install .

op.c: op.py appleamx.py
	exocc -o . --stem $(*F) $^

op.o: amx.h

main.c: op.c

.PHONY: clean
clean:
	$(RM) op op.[cdh] *.o exo_demo
	$(RM) -r __pycache__/

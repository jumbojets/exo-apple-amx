CPPFLAGS += -I.
CFLAGS ?= -march=native

appleamx_matmul: appleamx_matmul.o main.o

appleamx.py:
	pip install .

appleamx_matmul.c: appleamx_matmul.py appleamx.py
	exocc -o . --stem $(*F) $^

appleamx_matmul.o: amx.h

main.c: appleamx_matmul.c

.PHONY: clean
clean:
	$(RM) appleamx_matmul appleamx_matmul.[cdh] *.o exo_demo
	$(RM) -r __pycache__/

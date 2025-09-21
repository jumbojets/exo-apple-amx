# exo-apple-amx

[Exo](https://exo-lang.dev/) is a domain specific language for writing low-level, high-performance scientific computing and machine learning kernels. Instead of manually writing kernels by hand, the developer writes a simple, naive kernel and then declaratively schedules the kernel to optimize performance by improving cache locality, using vector instructions, etc.

The [Apple AMX Coprocessor](https://github.com/corsix/amx) is an undocumented accelerator created by Apple in the recent MX chips. The coprocessor is particularly optimized for matrix operations, and its instructions issue from the CPU.

### Notable Files

* appleamx.py: Includes AMX register and some instruction definitions for interoperability with the Exo compiler.
* appleamx_matmul.py: Example matmul kernel. Contains a naive matmul implementation and a scheduled one, the latter using the AMX register file and instructions defined in appleamx.py.
* main.c: Benchmarking harness for appleamx_matmul.py

### Install appleamx

```console
$ pip install .
```

### Run appleamx_matmul.py example

```console
$ make
$ ./appleamx_matmul
```

### appleamx_matmul.py performance on M1 Max

```
Unscheduled:      0.582 gflops
Scheduled:     2485.513 gflops (84% max)
Max:           2958.9   gflops
```

### some considerations / todos
* get scheduled performance closer to max
* some of the matrix stuff can be consolidated more nicely in the APPLE_AMX_POOL
* accumulator (3rd) dimension for z pool (and maybe even x, y)?
* mixed width
* all ops
* predefine useful rewrite rules
* address TODOs (mostly have to do with strides)

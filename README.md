# exo-apple-amx

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
Scheduled:     2396.745 gflops (81% max)
Max:           2958.9   gflops
```

some considerations / todos
* some of the matrix stuff can be consolidated more nicely in the APPLE_AMX_POOL
* accumulator (3rd) dimension for z pool (and maybe even x, y)?
* mixed width
* all ops
* useful rewrite rules
* address TODOs (mostly have to do with strides)

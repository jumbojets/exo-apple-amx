# exo-apple-amx

### Matmul performance

```
Unscheduled:      0.581 gflops
Scheduled:      838.861 gflops
```

some considerations / todos
* two accumulators in appleamx_matmul.py
* some of the matrix stuff can be consolidated more nicely in the APPLE_AMX_POOL
* mixed width
* all ops
* useful rewrite rules
* address TODOs (mostly have to do with strides)

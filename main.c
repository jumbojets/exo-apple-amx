#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "appleamx_matmul.h"

#define K 2048
static _Float16 A[K * 8 * 32] __attribute__ ((aligned (64)));
static _Float16 B[K * 8 * 32] __attribute__ ((aligned (64)));
static _Float16 C[32 * 32]    __attribute__ ((aligned (64)));

void initialize() {
  for (int i = 0; i < K * 8 * 32; i++) {
    A[i] = (_Float16)rand() / RAND_MAX;
    B[i] = (_Float16)rand() / RAND_MAX;
  }
  memset(C, 0, 32 * 32 * sizeof(_Float16));
}

#define ITERATIONS 10

double benchmark(void (*f)(void *c, int_fast32_t, const _Float16*, const _Float16*, _Float16*)) {
  double best_gflops = 0;
  for (int i = 0; i < ITERATIONS; i++) {
    int64_t start, end;
    start = clock_gettime_nsec_np(CLOCK_REALTIME);
    f(NULL, K, A, B, C);
    end = clock_gettime_nsec_np(CLOCK_REALTIME);
    double gflop = (2.0 * (K * 8) * 32 * 32) * 1e-9;
    double s = (end - start) * 1e-9;
    double gflops = gflop / s;
    if (gflops > best_gflops) {
        best_gflops = gflops;
    }
  }
  return best_gflops;
}

int main() {
  srand(time(NULL));

  double unscheduled_gflops = benchmark(rank_kx6_reduce_32x32);
  double scheduled_gflops = benchmark(rank_kx6_reduce_32x32_scheduled_appleamx);

  printf("%-12s %10.3f gflops\n", "Unscheduled:", unscheduled_gflops);
  printf("%-12s %10.3f gflops\n", "Scheduled:",   scheduled_gflops);
}

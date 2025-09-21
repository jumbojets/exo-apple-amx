#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "appleamx_matmul.h"

#define K 2048

static alignas(64) _Float16 A[K * 8 * 64];
static alignas(64) _Float16 B[K * 8 * 32];
static alignas(64) _Float16 C[64 * 32];

static inline _Float16 rand_float16() {
  return (_Float16)(rand() / ((double)RAND_MAX + 1));
}

void initialize() {
  for (size_t i = 0; i < K * 8 * 64; i++) A[i] = rand_float16();
  for (size_t i = 0; i < K * 8 * 32; i++) B[i] = rand_float16();
  memset(C, 0, 64 * 32 * sizeof(_Float16));
}

double benchmark(void (*f)(void *c, int_fast32_t, const _Float16*, const _Float16*, _Float16*)) {
  double best_gflops = 0;
  for (size_t i = 0; i < 10; i++) {
    initialize();
    int64_t start = clock_gettime_nsec_np(CLOCK_REALTIME);
    f(NULL, K, A, B, C);
    int64_t end = clock_gettime_nsec_np(CLOCK_REALTIME);
    double gflop = (2.0 * K * 8 * 64 * 32) * 1e-9;
    double s = (end - start) * 1e-9;
    best_gflops = fmax(gflop / s, best_gflops);
  }
  return best_gflops;
}

int main() {
  srand(time(NULL));

  double unscheduled_gflops = benchmark(rank_kx8_reduce_64x32);
  double scheduled_gflops = benchmark(rank_kx8_reduce_64x32_scheduled_appleamx);

  printf("%-12s %10.3f gflops\n", "Unscheduled:", unscheduled_gflops);
  printf("%-12s %10.3f gflops\n", "Scheduled:", scheduled_gflops);
}

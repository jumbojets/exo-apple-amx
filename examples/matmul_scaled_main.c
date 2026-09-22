#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matmul_scaled.h"

#define K 2048

typedef void kernel(void *ctxt, int_fast32_t, const _Float16*, const _Float16*, const _Float16*, _Float16*);

static alignas(64) _Float16 A[K * 4 * 32];
static alignas(64) _Float16 B[K * 4 * 32];
static alignas(64) _Float16 s[32];
static alignas(64) _Float16 C[32 * 32];
static alignas(64) _Float16 C_ref[32 * 32];

static inline _Float16 rand_float16() {
  return (_Float16)(rand() / ((double)RAND_MAX + 1));
}

void initialize() {
  for (size_t i = 0; i < K * 4 * 32; i++) A[i] = rand_float16();
  for (size_t i = 0; i < K * 4 * 32; i++) B[i] = rand_float16();
  for (size_t i = 0; i < 32; i++) s[i] = rand_float16();
}

// Compare the scheduled kernel against the naive one on small integer inputs.
int verify(kernel *ref, kernel *test) {
  const int_fast32_t k = 2;
  for (size_t i = 0; i < k * 4 * 32; i++) A[i] = (_Float16)(rand() % 7 - 3);
  for (size_t i = 0; i < k * 4 * 32; i++) B[i] = (_Float16)(rand() % 7 - 3);
  for (size_t i = 0; i < 32; i++) s[i] = (_Float16)(rand() % 7 - 3);
  ref(NULL, k, A, B, s, C_ref);
  test(NULL, k, A, B, s, C);
  int mismatches = 0;
  for (size_t i = 0; i < 32 * 32; i++) mismatches += (float)C[i] != (float)C_ref[i];
  if (mismatches) printf("VERIFY FAILED: %d of %d entries differ\n", mismatches, 32 * 32);
  return mismatches;
}

double benchmark(kernel *f) {
  double best_gflops = 0;
  for (size_t i = 0; i < 10; i++) {
    initialize();
    int64_t start = clock_gettime_nsec_np(CLOCK_REALTIME);
    f(NULL, K, A, B, s, C);
    int64_t end = clock_gettime_nsec_np(CLOCK_REALTIME);
    double gflop = (2.0 * K * 4 * 32 * 32) * 1e-9;
    double seconds = (end - start) * 1e-9;
    best_gflops = fmax(gflop / seconds, best_gflops);
  }
  return best_gflops;
}

int main() {
  srand(time(NULL));

  if (verify(scaled_matmul_32x32, scaled_matmul_32x32_scheduled_appleamx)) return 1;

  double unscheduled_gflops = benchmark(scaled_matmul_32x32);
  double scheduled_gflops = benchmark(scaled_matmul_32x32_scheduled_appleamx);

  printf("%-12s %10.3f gflops\n", "Unscheduled:", unscheduled_gflops);
  printf("%-12s %10.3f gflops\n", "Scheduled:", scheduled_gflops);
}

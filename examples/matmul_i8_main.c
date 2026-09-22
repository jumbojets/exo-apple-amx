#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matmul_i8.h"

#define K 2048

typedef void kernel(void *ctxt, int_fast32_t, const int8_t*, const int8_t*, int32_t*);

static alignas(64) int8_t A[K * 64];
static alignas(64) int8_t B[4 * K * 64];
static alignas(64) int32_t C[16 * 64];
static alignas(64) int32_t C_ref[16 * 64];

static inline int8_t rand_int8() {
  return (int8_t)(rand() % 256 - 128);
}

void initialize() {
  for (size_t i = 0; i < K * 64; i++) A[i] = rand_int8();
  for (size_t i = 0; i < 4 * K * 64; i++) B[i] = rand_int8();
  memset(C, 0, sizeof(C));
}

// Compare the scheduled kernel against the naive one.
int verify(kernel *ref, kernel *test) {
  initialize();
  memset(C_ref, 0, sizeof(C_ref));
  ref(NULL, K, A, B, C_ref);
  test(NULL, K, A, B, C);
  int mismatches = 0;
  for (size_t i = 0; i < 16 * 64; i++) mismatches += C[i] != C_ref[i];
  if (mismatches) printf("VERIFY FAILED: %d of %d entries differ\n", mismatches, 16 * 64);
  return mismatches;
}

double benchmark(kernel *f) {
  double best_gops = 0;
  for (size_t i = 0; i < 10; i++) {
    initialize();
    int64_t start = clock_gettime_nsec_np(CLOCK_REALTIME);
    f(NULL, K, A, B, C);
    int64_t end = clock_gettime_nsec_np(CLOCK_REALTIME);
    double gop = (2.0 * 4 * K * 16 * 64) * 1e-9;
    double s = (end - start) * 1e-9;
    best_gops = fmax(gop / s, best_gops);
  }
  return best_gops;
}

int main() {
  srand(time(NULL));

  if (verify(matmul_16x64_i8, matmul_16x64_i8_scheduled_appleamx)) return 1;

  double unscheduled_gops = benchmark(matmul_16x64_i8);
  double scheduled_gops = benchmark(matmul_16x64_i8_scheduled_appleamx);

  printf("%-12s %10.3f gops\n", "Unscheduled:", unscheduled_gops);
  printf("%-12s %10.3f gops\n", "Scheduled:", scheduled_gops);
}

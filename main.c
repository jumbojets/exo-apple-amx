#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "op.h"

static float A[16]      __attribute__ ((aligned (64)));
static float B[16]      __attribute__ ((aligned (64)));
static float C[16 * 16] __attribute__ ((aligned (64)));

#define CHECK_EQUIV 1
#define EPS         1e-5

void initialize() {
  for (int i = 0; i < 16; i++) {
    A[i] = (float)rand() / RAND_MAX;
    B[i] = (float)rand() / RAND_MAX;
  }
  memset(C, 0, 16 * 16 * sizeof(float));
}

int main() {
  srand(time(NULL));
  initialize();

  outer_product(NULL, A, B, C);

  #if CHECK_EQUIV
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      float computed = C[i + j * 16];
      float expected = A[i] * B[j];
      if (fabs(computed - expected) > EPS) {
        printf("not equivalent at (%d, %d): %f != %f\n", i, j, computed, expected);
        return 1;
      }
    }
  }
  printf("equivalent test passed\n");
  #endif
}

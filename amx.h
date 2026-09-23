// Modified: https://github.com/corsix/amx/blob/main/aarch64.h

#pragma once
#include <stdint.h>

#define AMX_NOP_OP_IMM5(op, imm5) \
  __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
  __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

#define AMX_LDST(op, ptr, reg, flags) \
  AMX_OP_GPR(op, ((uint64_t)&*(ptr)) | ((uint64_t)(reg) << 56) | (flags))

#define AMX_ALU(op, y, x, z, flags) \
  AMX_OP_GPR(op, ((uint64_t)(y)) | ((uint64_t)(x) << 10) | ((uint64_t)(z) << 20) | (flags))

#define AMX_LDX(ptr, reg, flags)   AMX_LDST(0, ptr, reg, flags)
#define AMX_LDY(ptr, reg, flags)   AMX_LDST(1, ptr, reg, flags)
#define AMX_STX(ptr, reg, flags)   AMX_LDST(2, ptr, reg, flags)
#define AMX_STY(ptr, reg, flags)   AMX_LDST(3, ptr, reg, flags)
#define AMX_LDZ(ptr, reg, flags)   AMX_LDST(4, ptr, reg, flags)
#define AMX_STZ(ptr, reg, flags)   AMX_LDST(5, ptr, reg, flags)
#define AMX_LDZI(ptr, reg, flags)  AMX_LDST(6, ptr, reg, flags)
#define AMX_STZI(ptr, reg, flags)  AMX_LDST(7, ptr, reg, flags)
#define AMX_EXTRX(gpr)             AMX_OP_GPR(8, gpr)
#define AMX_EXTRY(gpr)             AMX_OP_GPR(9, gpr)
#define AMX_FMA64(y, x, z, flags)  AMX_ALU(10, y, x, z, flags)
#define AMX_FMS64(y, x, z, flags)  AMX_ALU(11, y, x, z, flags)
#define AMX_FMA32(y, x, z, flags)  AMX_ALU(12, y, x, z, flags)
#define AMX_FMS32(y, x, z, flags)  AMX_ALU(13, y, x, z, flags)
#define AMX_MAC16(y, x, z, flags)  AMX_ALU(14, y, x, z, flags)
#define AMX_FMA16(y, x, z, flags)  AMX_ALU(15, y, x, z, flags)
#define AMX_FMS16(y, x, z, flags)  AMX_ALU(16, y, x, z, flags)
#define AMX_SET()                  AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()                  AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECINT(y, x, z, flags) AMX_ALU(18, y, x, z, flags)
#define AMX_VECFP(y, x, z, flags)  AMX_ALU(19, y, x, z, flags)
#define AMX_MATINT(y, x, z, flags) AMX_ALU(20, y, x, z, flags)
#define AMX_MATFP(y, x, z, flags)  AMX_ALU(21, y, x, z, flags)
#define AMX_GENLUT(src, flags)     AMX_OP_GPR(22, ((uint64_t)(src)) | (flags))

// extrx and extry copy a whole y register to x and x to y
#define AMX_EXTRX_COPY(x, y) AMX_EXTRX(((uint64_t)(x) << 16) | ((uint64_t)(y) << 20) | (1ull << 27))
#define AMX_EXTRY_COPY(y, x) AMX_EXTRY(((uint64_t)(y) << 6) | ((uint64_t)(x) << 20) | (1ull << 27))
// extrh copies a z row and extrv a z column to byte offset dst of x (or y with AMX_TO_Y)
#define AMX_EXTRH(dst, z, flags) AMX_EXTRX(((uint64_t)(dst)) | ((uint64_t)(z) << 20) | (1ull << 26) | (flags))
#define AMX_EXTRV(dst, z, flags) AMX_EXTRY(((uint64_t)(dst)) | ((uint64_t)(z) << 20) | (1ull << 26) | (flags))

#define AMX_VECTOR (1ull << 63)                                    // fma/fms: pointwise instead of outer product
#define AMX_SKIP_X (1ull << 29)                                    // fma/fms: without the x input
#define AMX_SKIP_Y (1ull << 28)                                    // fma/fms: without the y input
#define AMX_SKIP_Z (1ull << 27)                                    // fma/fms: without the z input
#define AMX_ALU_MODE(m) ((uint64_t)(m) << 47)                      // vecfp/matfp/vecint/matint: operation
#define AMX_LANE_WIDTH(w) ((uint64_t)(w) << 42)                    // vecfp/matfp/vecint/matint: element types
#define AMX_BROADCAST_Y(n) ((1ull << 38) | ((uint64_t)(n) << 32))  // vecfp/vecint: y lane n in every lane
#define AMX_TO_Y (1ull << 10)                                      // extrh/extrv: to y instead of x
#define AMX_EXTR_8BIT 0ull                                         // extrh/extrv: lane width
#define AMX_EXTR_16BIT (1ull << 63)
#define AMX_EXTR_32BIT (8ull << 11)
#define AMX_EXTR_64BIT ((1ull << 63) | (1ull << 11))

#ifndef GGML_SYCL_BUILTINS_HPP
#define GGML_SYCL_BUILTINS_HPP

#include <sys/types.h>

#include <cstdint>

#include "cacheopts.hpp"

#define GGML_SYCL_UNREACHABLE(x) \
    assert(0 && x);              \
    printf(x);

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N> using vector_t = T __attribute__((ext_vector_type(N)));
#else
template <class T, int N> using vector_t = sycl::marray<T, N>;
#endif

#ifdef __SYCL_DEVICE_ONLY__
#    define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#    define SYCL_DEVICE_BUILTIN(x)
#endif

#ifdef __SYCL_DEVICE_ONLY__
#    define SYCL_DEVICE_OCL(x) SYCL_EXTERNAL extern "C" x
#else
#    define SYCL_DEVICE_OCL(x)
#endif

using uint8 = vector_t<uint, 8>;
using uint2 = vector_t<uint, 2>;

using short16 = vector_t<short, 16>;
using short8  = vector_t<unsigned short, 8>;
using short2  = vector_t<unsigned short, 2>;

using uint8_32 = vector_t<uint8_t, 32>;
using char16   = vector_t<char, 16>;

// loads
SYCL_DEVICE_BUILTIN(short16 __builtin_IB_subgroup_block_read_flat_u8_m16k32v1(intptr_t baseoffset, int width_minus_one,
                                                                              int height_minus_one, int pitch_minus_one,
                                                                              uint2 coord));
SYCL_DEVICE_BUILTIN(char16 __builtin_IB_subgroup_block_read_flat_u8_m16k16v1(intptr_t baseoffset, int width_minus_one,
                                                                             int height_minus_one, int pitch_minus_one,
                                                                             uint2 coord));
SYCL_DEVICE_BUILTIN(int __builtin_IB_subgroup_block_read_flat_u8_m1k64v1(intptr_t baseoffset, int width_minus_one,
                                                                         int height_minus_one, int pitch_minus_one,
                                                                         uint2 coord));

//stores
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u32_m1k16v1(intptr_t baseoffset, int width_minus_one,
                                                                            int height_minus_one, int pitch_minus_one,
                                                                            uint2 coord, uint data));

// prefetches
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u8_m16k32v1(intptr_t baseoffset, int width_minus_one,
                                                                               int height_minus_one,
                                                                               int pitch_minus_one, uint2 coord,
                                                                               LSC_LDCC cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u8_m16k16v1(intptr_t baseoffset, int width_minus_one,
                                                                               int height_minus_one,
                                                                               int pitch_minus_one, uint2 coord,
                                                                               LSC_LDCC cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u8_m1k64v1(intptr_t baseoffset, int width_minus_one,
                                                                              int height_minus_one, int pitch_minus_one,
                                                                              uint2 coord, LSC_LDCC cache_control));

SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u32_m1k16(intptr_t baseoffset, int width_minus_one,
                                                                             int height_minus_one, int pitch_minus_one,
                                                                             uint2 coord, LSC_LDCC cache_control));

//DP4A instructions
SYCL_DEVICE_BUILTIN(int __builtin_IB_dp4a_ss(int c, int a, int b, bool isSaturated));

#endif

#ifndef GGML_SYCL_Q6_K_TILED_GEMV_HPP
#define GGML_SYCL_Q6_K_TILED_GEMV_HPP

#include <sys/types.h>

#include <cstdint>
#include <sycl/aliases.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/sycl.hpp>
#include <sycl/vector.hpp>

#include "builtins.hpp"
#include "cacheopts.hpp"
#include "ggml-quants.h"

#define sycl_print sycl::ext::oneapi::experimental::printf
//
/**
 * @brief This function packs 4 q6_k quants in a 32 bit value.
 *  
 */
__attribute__((always_inline)) inline int pack_q6_k(const short & low_bits, const char & high_bits) {
    int32_t packed_q6_k = 0;
// the low bits of the first quant will be the bits from  0 - 3, and so on and so forth, similarly the for the
// high bits as well. We need to pack the first quant in the low bits of the result, that is bits 0 - 7,
// second quant from 8 -15 and so on.
// TODO: handle all 4 at once via wider bitmask and repeating the bit patterns
// TODO: Reduce the number of brackets by checking the precedence order :)
#pragma unroll(4)
    for (uint8_t i = 0; i < 4; i++) {
        uint16_t   mask_low_bits  = (0x000F) << (4 * i);
        uint8_t  mask_high_bits = (0x3) << (2 * i);
        uint8_t desired_low_bits = (low_bits & mask_low_bits) >> (4 * i);
        uint8_t desired_high_bits = ((high_bits & mask_high_bits) >> (2 * i)) << 4;
        int8_t full_value = static_cast<int8_t>(desired_high_bits | desired_low_bits);
        full_value = sycl::sub_sat(full_value, (int8_t)32);
        packed_q6_k |= (static_cast<uint32_t>(static_cast<uint8_t>(full_value)) << (8 * i));
    }
    return packed_q6_k;
}

namespace sycl {
__attribute__((always_inline)) inline void q6k_tiled_gemv(
    const int8_t * q6_k_l, const int8_t * q6_k_h, const int8_t * q8_1, float * result, const int8_t * q6_u8_bit_scales,
    const sycl::half2 * q8_dm_scales, const sycl::half * q6_k_superblock_scale, int m, int k, const nd_item<1> & it) {
    // Performs a (m x k ) X (k x 1) GEMM
    // Each subgroup is responsible for 16 output elements.

    // And thus,
    // Block load for A matrix: m16k32. This gives us 2 8 bit columns per wi, implying 4 columns of
    // 4 Bit values per WI.

    // We therefore use m1k64 for the B vector, giving us 4 8 Bit values per WI.
    // Thus we can do one DP4A per row per WI, hence 16 total DP4A per work-item.

    // we will only choose this kernel when m % 16 == 0 and k % 64 == 0
#ifdef __SYCL_DEVICE_ONLY__
    constexpr int  tile_height          = 16;
    constexpr int  tile_width           = 64;
    constexpr bool dp4a_with_saturation = true;

    auto num_sgs_in_wg     = it.get_local_range(0) / 16;  // as sub-group size will be 16;
    auto num_sgs_in_kernel = num_sgs_in_wg * it.get_group_range(0);
    auto sg                = it.get_sub_group();
    auto sg_id             = it.get_group(0) * num_sgs_in_wg + sg.get_group_id();
    auto wi_id_in_sg       = sg.get_local_linear_id();

    auto       q6_k_l_width       = ((k / 2 - 1) * sizeof(int8_t));  // as we have 2 4 bit values packed in an int8_t;
    auto       q6_k_h_width       = ((k / 4 - 1) * sizeof(int8_t));  // as we have 4 2 bit values packed in an int8_t;
    auto       q8_1_width         = (k - 1) * sizeof(int8_t);
    auto       result_width       = (m - 1) * sizeof(float);
    auto       q6_u8_scale_width  = ((k / QK_K) * 16 - 1) * sizeof(int8_t);
    auto       super_block_scale_width = (m - 1) * sizeof(sycl::half);
    const auto num_blocks_per_row      = k / QK_K;

    const int              tiles_required = m / tile_height;
    sycl::vec<float, 16>   accumulator;
    vector_types::char16   q6_u8_scales_vals;
    sycl::half             super_block_scale;

    for (; sg_id < tiles_required; sg_id += num_sgs_in_kernel) {
        auto h_coord = sg_id * tile_height;

// zero out the accumulators
#    pragma unroll(16)
        for (uint8_t i = 0; i < 16; i++) {
            accumulator[i] = 0;
        }

        for (int i = 0; i < num_blocks_per_row; i++) {
            q6_u8_scales_vals = __builtin_IB_subgroup_block_read_flat_u8_m16k16v1(
                (intptr_t) q6_u8_bit_scales, q6_u8_scale_width, m - 1, q6_u8_scale_width,
                vector_types::uint2{ (uint) (i * 16), (uint) h_coord });

            auto super_block_scale_loaded = __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(
                (intptr_t) q6_k_superblock_scale, super_block_scale_width, num_blocks_per_row - 1,
                super_block_scale_width, vector_types::uint2{ (uint) (h_coord), (uint) i });
            super_block_scale = *reinterpret_cast<sycl::half*>(&super_block_scale_loaded);
            
            auto element_width_offset = i * QK_K;

#    pragma unroll(4)
            for (int j = 0; j < QK_K; j += tile_width) {
                vector_types::short16 q6_low_bits = __builtin_IB_subgroup_block_read_flat_u8_m16k32v1(
                    (intptr_t) (q6_k_l), q6_k_l_width, m - 1, q6_k_l_width,
                    vector_types::uint2{ (uint) (element_width_offset + j), (uint) h_coord });

                vector_types::char16 q6_high_bits = __builtin_IB_subgroup_block_read_flat_u8_m16k16v1(
                    (intptr_t) (q6_k_h), q6_k_h_width, m - 1, q6_k_h_width,
                    vector_types::uint2{ (uint) (element_width_offset + j), (uint) h_coord });

                int packed_q8_1_vals = __builtin_IB_subgroup_block_read_flat_u8_m1k64v1(
                    (intptr_t) (q8_1), q8_1_width, 0, q8_1_width,
                    vector_types::uint2{ (uint) (element_width_offset + j), (uint) 0 });

                sycl::half2 q8_dm_val =
                    q8_dm_scales[element_width_offset / QK8_1 + j / QK8_1 + (wi_id_in_sg * 4) / QK8_1];

#    pragma unroll(16)
                for (uint8_t l = 0; l < 16; l++) {
                    int packed_q6_k_vals = pack_q6_k(q6_low_bits[l], q6_high_bits[l]);
                    int dp4a_val = __builtin_IB_dp4a_ss(0, packed_q6_k_vals, packed_q8_1_vals, dp4a_with_saturation);
                    sycl::half q6_super_block_value = sycl::select_from_group(sg, super_block_scale, l);
                    int8_t     q6_block_scale_val   = sycl::select_from_group(sg, q6_u8_scales_vals[l], j / 16 + (wi_id_in_sg ) / 4);
                    accumulator[l] += dp4a_val * static_cast<float>(q6_super_block_value) *
                                      static_cast<float>(q6_block_scale_val) * static_cast<float>(q8_dm_val[0]);
                }
            }
        }

#    pragma unroll(15)
        for (uint8_t l = 0; l < 15; l++) {
            auto wi_id_to_fetch_from =
                (wi_id_in_sg + 1 + l) % 16;  // + 1 becuase we do not want to fetch from the same wi_id !
            float partial_accum_value = sycl::select_from_group(sg, accumulator[wi_id_in_sg], wi_id_to_fetch_from);
            accumulator[wi_id_in_sg] += partial_accum_value;
        }

        float final_result   = accumulator[wi_id_in_sg];
        auto  result_as_uint = *reinterpret_cast<uint *>(&final_result);
        __builtin_IB_subgroup_block_write_flat_u32_m1k16v1((intptr_t) result, result_width, 0, result_width,
                                                           uint2{ static_cast<uint>(h_coord), static_cast<uint>(0) },
                                                           result_as_uint);
    }
#endif
}
}  // namespace sycl

#endif

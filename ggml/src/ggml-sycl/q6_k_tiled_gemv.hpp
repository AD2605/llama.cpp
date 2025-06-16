#ifndef GGML_SYCL_Q6_K_TILED_GEMV_HPP
#define GGML_SYCL_Q6_K_TILED_GEMV_HPP

#include <sys/types.h>

#include <cstdint>
#include <sycl/aliases.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/sycl.hpp>
#include <sycl/vector.hpp>

#include "builtins.hpp"
#include "cacheopts.hpp"
#include "ggml-quants.h"

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
        short   mask_low_bits  = (0x000F) << (4 * i);
        int8_t  mask_high_bits = (0x03) << (2 * i);
        int32_t q6_k_val       = static_cast<int32_t>((low_bits & mask_low_bits) >> (4 * i)) |
                           (static_cast<int32_t>((high_bits & mask_high_bits) >> (2 * i)) << 4);
        q6_k_val    = q6_k_val - 32;  // apply shift;
        packed_q6_k = packed_q6_k | (q6_k_val << (8 * i));
    }
    return packed_q6_k;
}

namespace sycl {
__attribute__((always_inline)) inline void q6k_tiled_gemv(
    const int8_t * q6_k_l, const int8_t * q6_k_h, const int8_t * q8_1, float * result, const int8_t * q6_u8_bit_scales,
    const sycl::half2 * q8_f32_scales, const sycl::half * q6_k_superblock_scale, int m, int k, const nd_item<1> & it) {
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

    auto       q6_k_l_width       = ((k - 1) * sizeof(int8_t)) / 2;  // as we have 2 4 bit values packed in an int8_t;
    auto       q6_k_h_width       = ((k - 1) * sizeof(int8_t)) / 4;  // as we have 4 2 bit values packed in an int8_t;
    auto       q8_1_width         = (k - 1) * sizeof(int8_t);
    auto       result_width       = (m - 1) * sizeof(float);
    auto       q6_u8_scale_width  = (k / QK_K) * 16 * sizeof(int8_t);
    auto       q8_u32_scale_width = (k / QK8_1) * sizeof(float);
    const auto num_blocks_per_row = k / QK_K;

    const int              tiles_required = m / tile_height;
    sycl::vec<float, 16>   accumulator;
    sycl::vec<int32_t, 16> int32_temps;

    for (; sg_id < tiles_required; sg_id += num_sgs_in_kernel) {
        auto h_coord = sg_id * tile_height;

// zero out the accumulators
#    pragma unroll(16)
        for (uint8_t i = 0; i < 16; i++) {
            accumulator[i] = 0;
        }

        for (int i = 0; i < k; i += tile_width) {
            auto q6_k_l_vals = __builtin_IB_subgroup_block_read_flat_u8_m16k32v1(
                (intptr_t) q6_k_l, q6_k_l_width, m - 1, q6_k_l_width,
                uint2{ static_cast<uint32_t>(i), static_cast<uint32_t>(h_coord) });
            auto q6_k_h_vals = __builtin_IB_subgroup_block_read_flat_u8_m16k16v1(
                (intptr_t) q6_k_h, q6_k_h_width, m - 1, q6_k_h_width,
                uint2{ static_cast<uint32_t>(i), static_cast<uint32_t>(h_coord) });

#    pragma unroll(16)
            for (uint8_t j = 0; j < 16; j++) {
                int32_temps[j] = pack_q6_k(q6_k_l_vals[j], q6_k_h_vals[j]);
            }

            int q8_1_vals = __builtin_IB_subgroup_block_read_flat_u8_m1k64v1(
                (intptr_t) q8_1, q8_1_width, 0, q8_1_width,
                uint2{ static_cast<uint32_t>(i), static_cast<uint32_t>(0) });

#    pragma unroll(16)
            for (uint8_t j = 0; j < 16; j++) {
                int32_temps[j] = __builtin_IB_dp4a_ss(0, int32_temps[j], q8_1_vals, dp4a_with_saturation);
            }

            auto q8_scale_ds = q8_f32_scales[i / QK8_1 + (wi_id_in_sg * 4) / QK8_1];

#    pragma unroll(16)
            for (uint8_t j = 0; j < 16; j++) {
                int8_t     q6_k_block_scale = q6_u8_bit_scales[j * q6_u8_scale_width + i / 16 + (wi_id_in_sg * 4) / 16];
                sycl::half q6_superblock_scale = q6_k_superblock_scale[j * num_blocks_per_row + i / QK_K];
                auto scale_mul_1 = static_cast<float>(q8_scale_ds[0]) * static_cast<float>(q6_superblock_scale) * static_cast<float>(q6_k_block_scale);
                accumulator[j] += static_cast<float>(int32_temps[j]) * scale_mul_1;
            }
        }

// as each wi in the sub-group contains a fragment of the accumulation per row
// we now prepare one value per workitem, where wi_id_in_sg = output_element_id;
#    pragma unroll(15)
        for (uint8_t i = 0; i < 15; i++) {
            auto wi_id_to_fetch_from =
                (wi_id_in_sg + 1 + i) % 16;  // + 1 becuase we do not want to fetch from the same wi_id !
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

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

                if (j == 0) {
                    // print values;
                    /*float dequantized_values[64];
                    for(int l = 0; l < 16; l++) {
                        auto temp = pack_q6_k(q6_low_bits[l], q6_high_bits[l]);
                        auto unpacked_int8s = *reinterpret_cast<sycl::vec<int8_t, 4>*>(&temp);
                        for (int q = 0; q < 4; q++) {
                            dequantized_values[l * 4 + q] = unpacked_int8s[q] * sycl::select_from_group(sg, super_block_scale, l) * 
                            sycl::select_from_group(sg, q6_u8_scales_vals[l], j / 16 + (wi_id_in_sg ) / 4);
                        }
                    }
                    sycl_print(
                        "%lu %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
                        "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
                        "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
                        "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
                        "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
                        "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
                        "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
                        "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %f",
                        wi_id_in_sg, dequantized_values[0], dequantized_values[1], dequantized_values[2],
                        dequantized_values[3], dequantized_values[4], dequantized_values[5], dequantized_values[6],
                        dequantized_values[7], dequantized_values[8], dequantized_values[9], dequantized_values[10],
                        dequantized_values[11], dequantized_values[12], dequantized_values[13], dequantized_values[14],
                        dequantized_values[15], dequantized_values[16], dequantized_values[17], dequantized_values[18],
                        dequantized_values[19], dequantized_values[20], dequantized_values[21], dequantized_values[22],
                        dequantized_values[23], dequantized_values[24], dequantized_values[25], dequantized_values[26],
                        dequantized_values[27], dequantized_values[28], dequantized_values[29], dequantized_values[30],
                        dequantized_values[31], dequantized_values[32], dequantized_values[33], dequantized_values[34],
                        dequantized_values[35], dequantized_values[36], dequantized_values[37], dequantized_values[38],
                        dequantized_values[39], dequantized_values[40], dequantized_values[41], dequantized_values[42],
                        dequantized_values[43], dequantized_values[44], dequantized_values[45], dequantized_values[46],
                        dequantized_values[47], dequantized_values[48], dequantized_values[49], dequantized_values[50],
                        dequantized_values[51], dequantized_values[52], dequantized_values[53], dequantized_values[54],
                        dequantized_values[55], dequantized_values[56], dequantized_values[57], dequantized_values[58],
                        dequantized_values[59], dequantized_values[60], dequantized_values[61], dequantized_values[62],
                        dequantized_values[63]);
                    */
                    sycl_print("wi_id_in_sg %lu low_bits: %x %x %x %x %x %x %x %x \n", q6_low_bits[0],
                    q6_low_bits[1], q6_low_bits[2], q6_low_bits[3], q6_low_bits[4], q6_low_bits[5], 
                    q6_low_bits[6], q6_low_bits[7]);

                    sycl_print("wi_id_in_sg %lu low_bits: %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x \n", 
                    q6_high_bits[0], q6_high_bits[1], q6_high_bits[2], q6_high_bits[3], q6_high_bits[4], q6_high_bits[5], 
                    q6_high_bits[6], q6_high_bits[7], q6_high_bits[8], q6_high_bits[9], q6_high_bits[10], q6_high_bits[11],
                    q6_high_bits[12], q6_high_bits[13], q6_high_bits[14], q6_high_bits[15]);
                    sycl::group_barrier(it.get_group());
                }

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

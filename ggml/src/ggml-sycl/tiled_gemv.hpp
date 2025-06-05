#ifndef GGML_SYCL_TILED_GEMV_HPP
#define GGML_SYCL_TILED_GEMV_HPP

#include <sys/types.h>

#include <cstdint>
#include <sycl/aliases.hpp>
#include <sycl/sycl.hpp>

#include "builtins.hpp"
#include "cacheopts.hpp"

template <typename T> __attribute__((always_inline)) inline T half_group_reduce() {}

__attribute__((always_inline)) inline std::tuple<std::array<int8_t, 4>, float, float> quantize_input(
    const sycl::vec<float, 4> & wi_vals) {
    float                 wi_sum = 0;
    std::array<int8_t, 4> quant_tuple;
    float                 d;
    float                 sum;
#pragma unroll(4)
    for (int i = 0; i < 4; i++) {
        wi_sum += wi_vals[i];
    }
    return { quant_tuple, d, sum };
}

__attribute__((always_inline)) inline uint8_32 get_quant_block(void * q4_0_weights, int matrix_height, int matrix_width,
                                                               int pitch, uint2 coord) {
    auto q4_0_values = __builtin_IB_subgroup_block_read_flat_u8_m16k32v1((intptr_t) q4_0_weights, matrix_width,
                                                                         matrix_height, pitch, coord);
    return *reinterpret_cast<uint8_32 *>(&q4_0_values);
}

// currently hardcoded for the ease of development, will template it later on
__attribute__((always_inline)) inline void q4_0_q8_1_tiled_gemv(void * q4_0_weights, float * input, int rows, int cols,
                                                                std::size_t              q4_0_weights_pitch,
                                                                const sycl::nd_item<1> & it) {
    // Performs a tiled gemv where each subgroup takes the resposnsibility for a tile. and
    // each workitem within that sub-group is responsible for atleast one element of the output vector,
    // Hence the tile Height (number of rows) needs to be at least 16.
    // The subgroup copies 32 columns of the q4_0 weights as in 8bit copy, implying each work-item gets
    // 4 contiguous columns per row, and 64 columns in total. we copy 64 floating point values per sub-group
    // with each workitem reading 4 contiguous values, and hence half a sub-group is responsible for
    // quantizing the floating point input. Full subgroup quantizes 2 fp32 tiles.
    // Hence we skip 2 tiles at once.

    // GEMV Problem: rows x cols : cols x 1;

    // intel_sub_group_i8_i4_matrix_mad_k32(short a, int 4 b, int c);
    // 1 x 32 : 32 x 16;
    // 1 x 32 is very doable, we are doing it even today
    // 32 x 16 is the problem

    constexpr int tile_height = 16;

    // Since local_range = WARP_RANGE
    auto      subgroup_id        = it.get_local_id(0);
    auto      wi_id              = it.get_local_linear_id();
    const int tile_row_start_idx = subgroup_id * tile_height;

    int                 tile_col_begin = 0;
    sycl::vec<float, 4> input_fp32_vals;

    sycl::vec<int32_t, 16> partial_sums{ 0 };
    std::array<int32_t>    results_arr;

    const int matrix_width = (cols - 1) * sizeof(int8_t);

    for (; tile_col_begin < cols; tile_col_begin += 64) {
        //begin prefetch of the current A Tile
        // TODO: Start the prefetch of the next B Tile as well.

        // There's the idea that we want to write the results of this mul-mat via L3
        // So that there are a lot of cache hits when reading these values which will serve as inputs
        // for the next layer. So while writing, we write by NOT caching in L1, but caching in L3.
        // This is because we do not want to pollute the L1 cache by values we do not want,
        // we only want the tiles in L1. So we read via L1 and L3, but write via only L3.

        // TODO: should we read via L1 and not L3 ? This way we keep the L1 for the current inputs
        // and L3 for the outputs to enable cache hits for the next layer.

        __builtin_IB_subgroup_block_read_prefetch_u8_m16k32v1(
            (intptr_t) (q4_0_weights), matrix_width, rows, q4_0_weights_pitch,
            uint2{ static_cast<uint>(tile_col_begin), tile_row_start_idx }, LSC_LDCC_L1C_L3C);

        input_fp32_vals           = *reinterpret_cast<sycl::vec<float, 4> *>(input + tile_col_begin + wi_id);
        auto [quant_values, d, s] = quantize_input(std::move(input_fp32_vals));

        // Load the A block
        auto q4_0_tile = get_quant_block(q4_0_weights, rows, matrix_width, q4_0_weights_pitch,
                                         uint2{ static_cast<uint>(tile_col_begin), tile_row_start_idx });

// Praying to the compiler gods here that ILP kicks in
// Xe2 Core has double dispatch, i.e it can dispatch 2 instructions per clock tick if possible
// hence hopefully first dpas goes to ALU0 and second to ALU1
// TODO: This entire loop will be replaced by a single dpas call later on (somehow)
// 16 dp4a calls per thread

// TODO: If I am in the last K tile, prefetch my Q4_0 scales
#pragma unroll(16)
        for (uint8_t i = 0; i < 16; i++) {
            // probably some better naming
            uint8_t cols_1_2 = q4_0_tile[i * 2 + 0];
            uint8_t cols_3_4 = q4_0_tile[i * 2 + 1];
            // TODO: check, are we sure that we can simply zero-extend 4 bit integers to 8 bit integers ?
            // whatever happened to two's complement ? I suppose that's how 4 bit negative integeters
            // are represented as well.

            // dp4a dp4a dp4a
        }
    }
// Now use sub_group_broadcast to prepare results per thread
#pragma unroll(16)
    for (uint8_t i = 0; i < 16; i++) {
        // select element id = sg.local_linear_id
        // TODO: Why am i doing this ? this is literally a software emulation of
        // the dpas instruction, might as well dpas directly ?
    }

    // store 16 fp32 values. 1 value per thread
    //__builtin_IB_subgroup_block_write_flat_u32_m1k16v1(...)
}

#endif

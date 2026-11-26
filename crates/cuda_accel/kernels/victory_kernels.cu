// Victory Xbox Series compatibility layer — SM 8.9 (Ada Lovelace) CUDA PTX Kernels.
//
// Bit-exact SIMT twin of `crates/cuda_accel/kernels_oxide/src/lib.rs` (pure-Rust `cuda-oxide`).
// Compiled at build time into `$OUT_DIR/victory_kernels.ptx` via `nvcc --ptx -arch=sm_89 -O3`.
//
// Kernels:
//   1. `rdna2_detile_batch`          — Xbox RDNA2 (GFX10.3) micro-tile texture detiling
//   2. `lz4_decompress_batch`        — Standard LZ4 raw block parallel batch decompression
//   3. `gdeflate_v1_tile_decompress` — Open-spec Microsoft GDeflate v1 multi-tile raw DEFLATE
//   4. `compact_readback_deltas`     — Warp-ballot sparse GPU-to-CPU readback delta compaction

#include <stdint.h>

// ---------------------------------------------------------------------------
// Shared C-ABI descriptors (must match `cuda_accel` & `victory-gpu/contracts.rs`)
// ---------------------------------------------------------------------------

struct LutMeta {
    uint32_t x_off;
    uint32_t y_off;
    uint32_t z_off;
    uint32_t z_len;
};

struct DetileDispatchDesc {
    uint64_t src_base_offset;
    uint64_t dst_base_offset;
    uint32_t width_elements;
    uint32_t height_elements;
    uint32_t depth_slices;
    uint32_t element_bytes;
    uint32_t block_dims[3];
    uint32_t block_log2;
    uint32_t mip_pitch_elements;
    uint32_t dst_row_pitch_bytes;
    uint32_t dst_depth_pitch_bytes;
    uint32_t tail_x;
    uint32_t tail_y;
    uint32_t pattern_id;
    uint32_t thick;
    uint64_t slice_size;
    uint32_t first_slice;
    uint32_t _pad;
};

struct DecompressBlockDesc {
    uint64_t src_offset;
    uint32_t src_len;
    uint32_t dst_len;
    uint64_t dst_offset;
};

struct ReadbackPieceDesc {
    uint64_t now_offset;
    uint64_t held_offset;
    uint32_t dword_count;
    uint32_t piece_id;
};

struct ReadbackDeltaPair {
    uint32_t piece_id;
    uint32_t dword_offset;
    uint32_t value;
};

// ---------------------------------------------------------------------------
// Kernel 1: Xbox RDNA2 (GFX10.3 RB+) Micro-Tile Texture Detiling
// ---------------------------------------------------------------------------
//
// Launched with 1D grid of `total_blocks` blocks (256 threads/block).
// `dispatch_block_prefix` has length `num_dispatches + 1` with exclusive block offsets;
// thread 0 locates the dispatch via O(log N) binary search and broadcasts via __shared__.
// Matches `victory_formats::texture::Layout::offset` and `Layout::detile` bit-for-bit.

extern "C" __global__ __launch_bounds__(256) void rdna2_detile_batch(
    const uint8_t* __restrict__ tiled_src,
    uint8_t* __restrict__ linear_dst,
    const DetileDispatchDesc* __restrict__ dispatches,
    const uint32_t* __restrict__ dispatch_block_prefix,
    uint32_t num_dispatches,
    const uint32_t* __restrict__ lut_data,
    const LutMeta* __restrict__ lut_meta
) {
    __shared__ uint32_t s_dispatch_idx;
    __shared__ uint32_t s_block_start;

    if (threadIdx.x == 0) {
        uint32_t blk = blockIdx.x;
        uint32_t lo = 0;
        uint32_t hi = num_dispatches;
        while (lo + 1 < hi) {
            uint32_t mid = (lo + hi) >> 1;
            if (dispatch_block_prefix[mid] <= blk) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        s_dispatch_idx = lo;
        s_block_start = dispatch_block_prefix[lo];
    }
    __syncthreads();

    uint32_t d_idx = s_dispatch_idx;
    if (d_idx >= num_dispatches) {
        return;
    }

    DetileDispatchDesc d = dispatches[d_idx];
    uint32_t w = d.width_elements;
    uint32_t h = d.height_elements;
    uint32_t depth = d.depth_slices;
    uint64_t plane_elems = (uint64_t)w * (uint64_t)h;
    uint64_t total_elems = plane_elems * (uint64_t)depth;

    uint64_t elem_idx = (uint64_t)(blockIdx.x - s_block_start) * 256ull + (uint64_t)threadIdx.x;
    if (elem_idx >= total_elems) {
        return;
    }

    uint32_t z = (uint32_t)(elem_idx / plane_elems);
    uint32_t rem = (uint32_t)(elem_idx - (uint64_t)z * plane_elems);
    uint32_t y = rem / w;
    uint32_t x = rem - y * w;

    uint32_t z_abs = d.first_slice + z;
    uint32_t slice_idx;
    uint32_t inner_z;
    if (d.thick != 0u) {
        uint32_t bd = d.block_dims[2];
        slice_idx = z_abs / bd;
        inner_z = z_abs - slice_idx * bd;
    } else {
        slice_idx = z_abs;
        inner_z = 0u;
    }

    uint64_t base = d.src_base_offset + (uint64_t)slice_idx * d.slice_size;
    uint64_t src_off;
    if (d.pattern_id == 0xFFFFFFFFu || d.block_log2 == 0u) {
        uint64_t row = (uint64_t)y * (uint64_t)d.mip_pitch_elements + (uint64_t)x;
        src_off = base + row * (uint64_t)d.element_bytes;
    } else {
        uint32_t bw = d.block_dims[0];
        uint32_t bh = d.block_dims[1];
        uint32_t tx = x + d.tail_x;
        uint32_t ty = y + d.tail_y;

        uint64_t row_blocks = (uint64_t)(ty / bh) * (uint64_t)(d.mip_pitch_elements / bw);
        uint64_t col_blocks = (uint64_t)(tx / bw);

        LutMeta m = lut_meta[d.pattern_id];
        uint32_t lut_x = lut_data[m.x_off + (tx % bw)];
        uint32_t lut_y = lut_data[m.y_off + (ty % bh)];
        uint32_t lut_z = (m.z_len > 0u) ? lut_data[m.z_off + inner_z] : 0u;

        src_off = base + ((row_blocks + col_blocks) << d.block_log2) + (uint64_t)((lut_y | lut_z) + lut_x);
    }

    uint64_t dst_off = d.dst_base_offset
        + (uint64_t)z * (uint64_t)d.dst_depth_pitch_bytes
        + (uint64_t)y * (uint64_t)d.dst_row_pitch_bytes
        + (uint64_t)x * (uint64_t)d.element_bytes;

    const uint8_t* src_ptr = tiled_src + src_off;
    uint8_t* dst_ptr = linear_dst + dst_off;
    uint32_t bpe = d.element_bytes;

    // Fast vectorized loads/stores when naturally aligned (true for all D3D12 footprints & GFX10 blocks).
    uintptr_t align_mask = (uintptr_t)src_ptr | (uintptr_t)dst_ptr;
    if (bpe == 16u && (align_mask & 15u) == 0u) {
        *reinterpret_cast<uint4*>(dst_ptr) = *reinterpret_cast<const uint4*>(src_ptr);
    } else if (bpe == 8u && (align_mask & 7u) == 0u) {
        *reinterpret_cast<uint2*>(dst_ptr) = *reinterpret_cast<const uint2*>(src_ptr);
    } else if (bpe == 4u && (align_mask & 3u) == 0u) {
        *reinterpret_cast<uint32_t*>(dst_ptr) = *reinterpret_cast<const uint32_t*>(src_ptr);
    } else if (bpe == 2u && (align_mask & 1u) == 0u) {
        *reinterpret_cast<uint16_t*>(dst_ptr) = *reinterpret_cast<const uint16_t*>(src_ptr);
    } else if (bpe == 1u) {
        *dst_ptr = *src_ptr;
    } else {
        for (uint32_t i = 0; i < bpe; ++i) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2A: Parallel Batch LZ4 Raw Block Decompression
// ---------------------------------------------------------------------------
//
// Each CTA (1 thread/block) independently decompresses one LZ4 raw block across all SMs
// (matching `lz4_flex::block::decompress`). Spreading 1 block per CTA achieves full
// multi-SM parallelism with zero intra-warp branch divergence.
// Status codes: 0 = OK, 1 = truncated input, 2 = output overflow, 3 = invalid match offset.

extern "C" __global__ __launch_bounds__(1) void lz4_decompress_batch(
    const uint8_t* __restrict__ src_buf,
    uint8_t* __restrict__ dst_buf,
    const DecompressBlockDesc* __restrict__ blocks,
    uint32_t num_blocks,
    uint32_t* __restrict__ out_written,
    uint32_t* __restrict__ out_status
) {
    uint32_t idx = blockIdx.x;
    if (idx >= num_blocks) {
        return;
    }

    DecompressBlockDesc b = blocks[idx];
    const uint8_t* src = src_buf + b.src_offset;
    uint8_t* dst = dst_buf + b.dst_offset;
    uint32_t src_len = b.src_len;
    uint32_t dst_cap = b.dst_len;

    if (src_len == 0u) {
        out_written[idx] = 0u;
        out_status[idx] = 0u;
        return;
    }

    uint32_t ip = 0u;
    uint32_t op = 0u;

    while (ip < src_len) {
        uint8_t token = src[ip++];
        uint32_t lit_len = (uint32_t)(token >> 4);
        if (lit_len == 15u) {
            while (true) {
                if (ip >= src_len) {
                    out_written[idx] = op;
                    out_status[idx] = 1u;
                    return;
                }
                uint8_t extra = src[ip++];
                lit_len += extra;
                if (extra != 255u) {
                    break;
                }
            }
        }

        if (lit_len > src_len - ip) {
            out_written[idx] = op;
            out_status[idx] = 1u;
            return;
        }
        if (lit_len > dst_cap - op) {
            out_written[idx] = op;
            out_status[idx] = 2u;
            return;
        }

        uint32_t i = 0u;
        while (i + 4u <= lit_len) {
            dst[op + i] = src[ip + i];
            dst[op + i + 1u] = src[ip + i + 1u];
            dst[op + i + 2u] = src[ip + i + 2u];
            dst[op + i + 3u] = src[ip + i + 3u];
            i += 4u;
        }
        while (i < lit_len) {
            dst[op + i] = src[ip + i];
            ++i;
        }
        ip += lit_len;
        op += lit_len;

        // End of block: last sequence contains literals only.
        if (ip >= src_len) {
            break;
        }

        if (src_len - ip < 2u) {
            out_written[idx] = op;
            out_status[idx] = 1u;
            return;
        }
        uint32_t offset = (uint32_t)src[ip] | ((uint32_t)src[ip + 1u] << 8);
        ip += 2u;

        if (offset == 0u || offset > op) {
            out_written[idx] = op;
            out_status[idx] = 3u;
            return;
        }

        uint32_t match_len = (uint32_t)(token & 0x0Fu) + 4u;
        if ((token & 0x0Fu) == 15u) {
            while (true) {
                if (ip >= src_len) {
                    out_written[idx] = op;
                    out_status[idx] = 1u;
                    return;
                }
                uint8_t extra = src[ip++];
                match_len += extra;
                if (extra != 255u) {
                    break;
                }
            }
        }

        if (match_len > dst_cap - op) {
            out_written[idx] = op;
            out_status[idx] = 2u;
            return;
        }

        uint32_t match_src = op - offset;
        if (offset >= 4u) {
            uint32_t k = 0u;
            while (k + 4u <= match_len) {
                dst[op + k] = dst[match_src + k];
                dst[op + k + 1u] = dst[match_src + k + 1u];
                dst[op + k + 2u] = dst[match_src + k + 2u];
                dst[op + k + 3u] = dst[match_src + k + 3u];
                k += 4u;
            }
            while (k < match_len) {
                dst[op + k] = dst[match_src + k];
                ++k;
            }
        } else {
            // Overlapping short back-reference (e.g. offset == 1 RLE).
            for (uint32_t k = 0u; k < match_len; ++k) {
                dst[op + k] = dst[match_src + k];
            }
        }
        op += match_len;
    }

    out_written[idx] = op;
    out_status[idx] = 0u;
}

// ---------------------------------------------------------------------------
// Kernel 2B: Open-Spec Microsoft GDeflate v1 Multi-Tile Raw DEFLATE (RFC 1951)
// ---------------------------------------------------------------------------
//
// Launched with 1 warp (32 threads) per independent 64 KiB raw DEFLATE tile.
// Huffman lookup tables live in on-chip `__shared__` SRAM (~2.4 KiB/block) so
// zero register spills occur and dozens of tiles execute concurrently per SM.
//
// Status codes:
//   0 = OK
//   1 = truncated bitstream
//   2 = invalid block type / stored block length mismatch
//   3 = invalid Huffman tree / symbol
//   4 = invalid back-reference distance
//   5 = output buffer overflow

namespace deflate_detail {

constexpr uint32_t LIT_FAST_BITS = 9u;
constexpr uint32_t LIT_FAST_SIZE = 1u << LIT_FAST_BITS;
constexpr uint32_t LIT_FAST_MASK = LIT_FAST_SIZE - 1u;

constexpr uint32_t DIST_FAST_BITS = 6u;
constexpr uint32_t DIST_FAST_SIZE = 1u << DIST_FAST_BITS;
constexpr uint32_t DIST_FAST_MASK = DIST_FAST_SIZE - 1u;

constexpr uint32_t CL_FAST_BITS = 7u;
constexpr uint32_t CL_FAST_SIZE = 1u << CL_FAST_BITS;
constexpr uint32_t CL_FAST_MASK = CL_FAST_SIZE - 1u;

struct HuffTableLit {
    uint16_t fast[LIT_FAST_SIZE];
    uint16_t syms[288];
    uint16_t first_code[16];
    uint16_t first_sym[16];
    uint16_t count[16];
};

struct HuffTableDist {
    uint16_t fast[DIST_FAST_SIZE];
    uint16_t syms[32];
    uint16_t first_code[16];
    uint16_t first_sym[16];
    uint16_t count[16];
};

struct SharedTileState {
    HuffTableLit lit;
    HuffTableDist dist;
    uint16_t cl_fast[CL_FAST_SIZE];
    uint8_t all_lens[320];
    uint8_t cl_lens[19];
    // For warp-parallel stored block copies.
    uint32_t copy_src_off;
    uint32_t copy_dst_off;
    uint32_t copy_len;
};

__device__ __constant__ uint8_t CL_ORDER[19] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
};

__device__ __constant__ uint16_t LEN_BASE[29] = {
    3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 15, 17,
    19, 23, 27, 31,
    35, 43, 51, 59,
    67, 83, 99, 115,
    131, 163, 195, 227,
    258
};

__device__ __constant__ uint8_t LEN_EXTRA[29] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
    0
};

__device__ __constant__ uint16_t DIST_BASE[30] = {
    1, 2, 3, 4,
    5, 7,
    9, 13,
    17, 25,
    33, 49,
    65, 97,
    129, 193,
    257, 385,
    513, 769,
    1025, 1537,
    2049, 3073,
    4097, 6145,
    8193, 12289,
    16385, 24577
};

__device__ __constant__ uint8_t DIST_EXTRA[30] = {
    0, 0, 0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 4,
    5, 5,
    6, 6,
    7, 7,
    8, 8,
    9, 9,
    10, 10,
    11, 11,
    12, 12,
    13, 13
};

struct BitReader {
    const uint8_t* src;
    uint32_t pos;
    uint32_t len;
    uint64_t buf;
    int32_t cnt;

    __device__ __forceinline__ void refill() {
        if (cnt <= 32 && pos + 4u <= len) {
            uint32_t word = (uint32_t)src[pos]
                | ((uint32_t)src[pos + 1u] << 8)
                | ((uint32_t)src[pos + 2u] << 16)
                | ((uint32_t)src[pos + 3u] << 24);
            buf |= ((uint64_t)word) << cnt;
            pos += 4u;
            cnt += 32;
        }
        while (cnt <= 56 && pos < len) {
            buf |= ((uint64_t)src[pos++]) << cnt;
            cnt += 8;
        }
    }

    __device__ __forceinline__ bool read_bits(uint32_t n, uint32_t& out) {
        if (cnt < (int32_t)n) {
            refill();
            if (cnt < (int32_t)n) {
                return false;
            }
        }
        out = (uint32_t)(buf & ((1ull << n) - 1ull));
        buf >>= n;
        cnt -= (int32_t)n;
        return true;
    }

    __device__ __forceinline__ void align_to_byte() {
        int32_t drop = cnt & 7;
        buf >>= drop;
        cnt -= drop;
    }

    __device__ __forceinline__ void rewind_buffered_bytes() {
        uint32_t buffered_bytes = (uint32_t)(cnt >> 3);
        pos -= buffered_bytes;
        buf = 0ull;
        cnt = 0;
    }
};

template <typename Table, uint32_t FAST_BITS, uint32_t FAST_SIZE>
__device__ bool build_huff(Table& tbl, const uint8_t* lengths, uint32_t num_syms) {
    for (uint32_t i = 0; i < 16u; ++i) {
        tbl.count[i] = 0u;
        tbl.first_code[i] = 0u;
        tbl.first_sym[i] = 0u;
    }
    for (uint32_t i = 0; i < FAST_SIZE; ++i) {
        tbl.fast[i] = 0u;
    }

    for (uint32_t s = 0; s < num_syms; ++s) {
        uint8_t l = lengths[s];
        if (l > 15u) {
            return false;
        }
        if (l > 0u) {
            tbl.count[l]++;
        }
    }

    // Validate Kraft inequality (allow empty tree e.g. unused distance table).
    uint32_t code = 0u;
    uint32_t sym_off = 0u;
    int32_t left = 1;
    uint16_t next_code[16];
    uint16_t next_sym[16];
    next_code[0] = 0u;
    next_sym[0] = 0u;
    for (uint32_t bits = 1u; bits <= 15u; ++bits) {
        code = (code + tbl.count[bits - 1u]) << 1;
        left = (left << 1) - (int32_t)tbl.count[bits];
        if (left < 0) {
            return false;
        }
        tbl.first_code[bits] = (uint16_t)code;
        tbl.first_sym[bits] = (uint16_t)sym_off;
        next_code[bits] = (uint16_t)code;
        next_sym[bits] = (uint16_t)sym_off;
        sym_off += tbl.count[bits];
    }

    for (uint32_t s = 0; s < num_syms; ++s) {
        uint8_t l = lengths[s];
        if (l == 0u) {
            continue;
        }
        uint32_t c = next_code[l]++;
        tbl.syms[next_sym[l]++] = (uint16_t)s;

        if (l <= FAST_BITS) {
            // Bit-reverse `l`-bit code `c` for LSB-first lookup.
            uint32_t rev = __brev(c) >> (32u - l);
            uint32_t step = 1u << l;
            uint16_t entry = (uint16_t)(((uint32_t)l << 12) | s);
            for (uint32_t idx = rev; idx < FAST_SIZE; idx += step) {
                tbl.fast[idx] = entry;
            }
        }
    }
    return true;
}

__device__ bool build_cl_fast(uint16_t* fast, const uint8_t* lengths) {
    uint16_t count[8] = {0};
    for (uint32_t i = 0; i < CL_FAST_SIZE; ++i) {
        fast[i] = 0u;
    }
    for (uint32_t s = 0; s < 19u; ++s) {
        if (lengths[s] > 7u) {
            return false;
        }
        if (lengths[s] > 0u) {
            count[lengths[s]]++;
        }
    }
    uint32_t code = 0u;
    int32_t left = 1;
    uint16_t next_code[8] = {0};
    for (uint32_t bits = 1u; bits <= 7u; ++bits) {
        code = (code + count[bits - 1u]) << 1;
        left = (left << 1) - (int32_t)count[bits];
        if (left < 0) {
            return false;
        }
        next_code[bits] = (uint16_t)code;
    }
    for (uint32_t s = 0; s < 19u; ++s) {
        uint8_t l = lengths[s];
        if (l == 0u) {
            continue;
        }
        uint32_t c = next_code[l]++;
        uint32_t rev = __brev(c) >> (32u - l);
        uint32_t step = 1u << l;
        uint16_t entry = (uint16_t)(((uint32_t)l << 12) | s);
        for (uint32_t idx = rev; idx < CL_FAST_SIZE; idx += step) {
            fast[idx] = entry;
        }
    }
    return true;
}

template <typename Table, uint32_t FAST_BITS, uint32_t FAST_MASK>
__device__ __forceinline__ int32_t decode_huff(BitReader& br, const Table& tbl) {
    if (br.cnt < 15) {
        br.refill();
    }
    uint16_t entry = tbl.fast[(uint32_t)br.buf & FAST_MASK];
    if (entry != 0u) {
        uint32_t len = entry >> 12;
        if (br.cnt < (int32_t)len) {
            return -1;
        }
        br.buf >>= len;
        br.cnt -= (int32_t)len;
        return (int32_t)(entry & 0x0FFFu);
    }

    uint32_t rev15 = __brev((uint32_t)(br.buf & 0x7FFFu)) >> 17;
    for (uint32_t len = FAST_BITS + 1u; len <= 15u; ++len) {
        uint32_t cnt = tbl.count[len];
        if (cnt == 0u) {
            continue;
        }
        uint32_t c = rev15 >> (15u - len);
        uint32_t first = tbl.first_code[len];
        if (c >= first && (c - first) < cnt) {
            if (br.cnt < (int32_t)len) {
                return -1;
            }
            br.buf >>= len;
            br.cnt -= (int32_t)len;
            return (int32_t)tbl.syms[tbl.first_sym[len] + (c - first)];
        }
    }
    return -1;
}

} // namespace deflate_detail

extern "C" __global__ __launch_bounds__(32) void gdeflate_v1_tile_decompress(
    const uint8_t* __restrict__ src_buf,
    uint8_t* __restrict__ dst_buf,
    const DecompressBlockDesc* __restrict__ tiles,
    uint32_t num_tiles,
    uint32_t* __restrict__ out_written,
    uint32_t* __restrict__ out_status
) {
    using namespace deflate_detail;

    uint32_t tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) {
        return;
    }

    __shared__ SharedTileState st;

    DecompressBlockDesc td = tiles[tile_idx];
    const uint8_t* src = src_buf + td.src_offset;
    uint8_t* dst = dst_buf + td.dst_offset;
    uint32_t src_len = td.src_len;
    uint32_t dst_cap = td.dst_len;

    BitReader br;
    br.src = src;
    br.pos = 0u;
    br.len = src_len;
    br.buf = 0ull;
    br.cnt = 0;

    uint32_t out_pos = 0u;
    uint32_t status = 0u;
    bool done = false;

    while (!done) {
        uint32_t action = 0u; // 0 = continue/done, 1 = warp stored copy
        if (threadIdx.x == 0) {
            uint32_t bfinal = 0u;
            uint32_t btype = 0u;
            if (!br.read_bits(1u, bfinal) || !br.read_bits(2u, btype)) {
                status = 1u;
                done = true;
            } else if (btype == 0u) {
                // Stored uncompressed block.
                br.align_to_byte();
                uint32_t blen = 0u;
                uint32_t bnlen = 0u;
                if (!br.read_bits(16u, blen) || !br.read_bits(16u, bnlen)) {
                    status = 1u;
                    done = true;
                } else if (((blen ^ 0xFFFFu) & 0xFFFFu) != bnlen) {
                    status = 2u;
                    done = true;
                } else {
                    br.rewind_buffered_bytes();
                    if (blen > src_len - br.pos) {
                        status = 1u;
                        done = true;
                    } else if (blen > dst_cap - out_pos) {
                        status = 5u;
                        done = true;
                    } else {
                        st.copy_src_off = br.pos;
                        st.copy_dst_off = out_pos;
                        st.copy_len = blen;
                        br.pos += blen;
                        out_pos += blen;
                        action = 1u;
                        if (bfinal != 0u) {
                            done = true;
                        }
                    }
                }
            } else if (btype == 1u || btype == 2u) {
                bool tree_ok = true;
                if (btype == 1u) {
                    // Fixed Huffman codes (RFC 1951 Section 3.2.6).
                    for (uint32_t i = 0; i <= 143u; ++i) st.all_lens[i] = 8u;
                    for (uint32_t i = 144u; i <= 255u; ++i) st.all_lens[i] = 9u;
                    for (uint32_t i = 256u; i <= 279u; ++i) st.all_lens[i] = 7u;
                    for (uint32_t i = 280u; i <= 287u; ++i) st.all_lens[i] = 8u;
                    for (uint32_t i = 0; i < 32u; ++i) st.all_lens[288u + i] = 5u;
                    tree_ok = build_huff<HuffTableLit, LIT_FAST_BITS, LIT_FAST_SIZE>(st.lit, st.all_lens, 288u)
                        && build_huff<HuffTableDist, DIST_FAST_BITS, DIST_FAST_SIZE>(st.dist, st.all_lens + 288u, 32u);
                } else {
                    // Dynamic Huffman codes (RFC 1951 Section 3.2.7).
                    uint32_t hlit = 0u, hdist = 0u, hclen = 0u;
                    if (!br.read_bits(5u, hlit) || !br.read_bits(5u, hdist) || !br.read_bits(4u, hclen)) {
                        status = 1u;
                        tree_ok = false;
                    } else {
                        hlit += 257u;
                        hdist += 1u;
                        hclen += 4u;
                        if (hlit > 286u || hdist > 30u) {
                            status = 3u;
                            tree_ok = false;
                        } else {
                            for (uint32_t i = 0; i < 19u; ++i) st.cl_lens[i] = 0u;
                            for (uint32_t i = 0; i < hclen; ++i) {
                                uint32_t v = 0u;
                                if (!br.read_bits(3u, v)) {
                                    status = 1u;
                                    tree_ok = false;
                                    break;
                                }
                                st.cl_lens[CL_ORDER[i]] = (uint8_t)v;
                            }
                            if (tree_ok && !build_cl_fast(st.cl_fast, st.cl_lens)) {
                                status = 3u;
                                tree_ok = false;
                            }
                            uint32_t total = hlit + hdist;
                            uint32_t cur = 0u;
                            while (tree_ok && cur < total) {
                                if (br.cnt < 7) br.refill();
                                uint16_t e = st.cl_fast[(uint32_t)br.buf & CL_FAST_MASK];
                                uint32_t cl_len = e >> 12;
                                if (e == 0u || br.cnt < (int32_t)cl_len) {
                                    status = 3u;
                                    tree_ok = false;
                                    break;
                                }
                                br.buf >>= cl_len;
                                br.cnt -= (int32_t)cl_len;
                                uint32_t sym = e & 0x0FFFu;
                                if (sym < 16u) {
                                    st.all_lens[cur++] = (uint8_t)sym;
                                } else if (sym == 16u) {
                                    uint32_t rep = 0u;
                                    if (cur == 0u || !br.read_bits(2u, rep)) {
                                        status = 3u;
                                        tree_ok = false;
                                        break;
                                    }
                                    rep += 3u;
                                    uint8_t prev = st.all_lens[cur - 1u];
                                    if (cur + rep > total) {
                                        status = 3u;
                                        tree_ok = false;
                                        break;
                                    }
                                    for (uint32_t r = 0; r < rep; ++r) st.all_lens[cur++] = prev;
                                } else if (sym == 17u) {
                                    uint32_t rep = 0u;
                                    if (!br.read_bits(3u, rep)) {
                                        status = 1u;
                                        tree_ok = false;
                                        break;
                                    }
                                    rep += 3u;
                                    if (cur + rep > total) {
                                        status = 3u;
                                        tree_ok = false;
                                        break;
                                    }
                                    for (uint32_t r = 0; r < rep; ++r) st.all_lens[cur++] = 0u;
                                } else if (sym == 18u) {
                                    uint32_t rep = 0u;
                                    if (!br.read_bits(7u, rep)) {
                                        status = 1u;
                                        tree_ok = false;
                                        break;
                                    }
                                    rep += 11u;
                                    if (cur + rep > total) {
                                        status = 3u;
                                        tree_ok = false;
                                        break;
                                    }
                                    for (uint32_t r = 0; r < rep; ++r) st.all_lens[cur++] = 0u;
                                } else {
                                    status = 3u;
                                    tree_ok = false;
                                    break;
                                }
                            }
                            if (tree_ok) {
                                if (!build_huff<HuffTableLit, LIT_FAST_BITS, LIT_FAST_SIZE>(st.lit, st.all_lens, hlit)
                                    || !build_huff<HuffTableDist, DIST_FAST_BITS, DIST_FAST_SIZE>(st.dist, st.all_lens + hlit, hdist)) {
                                    status = 3u;
                                    tree_ok = false;
                                }
                            }
                        }
                    }
                }

                if (!tree_ok) {
                    if (status == 0u) status = 3u;
                    done = true;
                } else {
                    // Decode literal/length/distance stream for this block.
                    while (true) {
                        int32_t sym = decode_huff<HuffTableLit, LIT_FAST_BITS, LIT_FAST_MASK>(br, st.lit);
                        if (sym < 0) {
                            status = 3u;
                            done = true;
                            break;
                        }
                        if (sym < 256) {
                            if (out_pos >= dst_cap) {
                                status = 5u;
                                done = true;
                                break;
                            }
                            dst[out_pos++] = (uint8_t)sym;
                        } else if (sym == 256) {
                            if (bfinal != 0u) {
                                done = true;
                            }
                            break;
                        } else {
                            uint32_t lidx = (uint32_t)(sym - 257);
                            if (lidx >= 29u) {
                                status = 3u;
                                done = true;
                                break;
                            }
                            uint32_t match_len = LEN_BASE[lidx];
                            uint32_t lextra = LEN_EXTRA[lidx];
                            if (lextra > 0u) {
                                uint32_t eb = 0u;
                                if (!br.read_bits(lextra, eb)) {
                                    status = 1u;
                                    done = true;
                                    break;
                                }
                                match_len += eb;
                            }

                            int32_t dsym = decode_huff<HuffTableDist, DIST_FAST_BITS, DIST_FAST_MASK>(br, st.dist);
                            if (dsym < 0 || dsym >= 30) {
                                status = 3u;
                                done = true;
                                break;
                            }
                            uint32_t dist = DIST_BASE[dsym];
                            uint32_t dextra = DIST_EXTRA[dsym];
                            if (dextra > 0u) {
                                uint32_t eb = 0u;
                                if (!br.read_bits(dextra, eb)) {
                                    status = 1u;
                                    done = true;
                                    break;
                                }
                                dist += eb;
                            }

                            if (dist == 0u || dist > out_pos) {
                                status = 4u;
                                done = true;
                                break;
                            }
                            if (match_len > dst_cap - out_pos) {
                                status = 5u;
                                done = true;
                                break;
                            }
                            uint32_t msrc = out_pos - dist;
                            for (uint32_t k = 0; k < match_len; ++k) {
                                dst[out_pos + k] = dst[msrc + k];
                            }
                            out_pos += match_len;
                        }
                    }
                }
            } else {
                // Reserved block type 3.
                status = 2u;
                done = true;
            }
        }

        // Broadcast warp action & termination flag from lane 0.
        action = __shfl_sync(0xFFFFFFFFu, action, 0);
        uint32_t done_u = __shfl_sync(0xFFFFFFFFu, (uint32_t)done, 0);
        done = (done_u != 0u);

        if (action == 1u) {
            uint32_t cs = st.copy_src_off;
            uint32_t cd = st.copy_dst_off;
            uint32_t cl = st.copy_len;
            for (uint32_t i = threadIdx.x; i < cl; i += 32u) {
                dst[cd + i] = src[cs + i];
            }
            __syncwarp();
        }
    }

    if (threadIdx.x == 0) {
        out_written[tile_idx] = out_pos;
        out_status[tile_idx] = status;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Warp-Level Sparse Readback Delta Compaction
// ---------------------------------------------------------------------------
//
// Compares 64 KiB GPU card buffer pieces (`now_buf`) against reference snapshots (`held_buf`),
// compacting only changed 32-bit dwords into dense `(piece_id, dword_offset, value)` pairs.
// Uses `__ballot_sync` + `__popc` + shared-memory warp prefix-sum for at most 1 atomicAdd per
// 1 KiB chunk, and skips completely unchanged 1 KiB chunks with zero atomics/writes.

extern "C" __global__ __launch_bounds__(256) void compact_readback_deltas(
    const uint8_t* __restrict__ now_buf,
    uint8_t* __restrict__ held_buf,
    const ReadbackPieceDesc* __restrict__ pieces,
    uint32_t num_pieces,
    ReadbackDeltaPair* __restrict__ out_pairs,
    uint32_t max_pairs,
    uint32_t* __restrict__ global_counter,
    uint32_t* __restrict__ overflow_flag,
    uint32_t update_held
) {
    uint32_t piece_idx = blockIdx.x;
    if (piece_idx >= num_pieces) {
        return;
    }

    ReadbackPieceDesc p = pieces[piece_idx];
    const uint32_t* now_u32 = reinterpret_cast<const uint32_t*>(now_buf + p.now_offset);
    uint32_t* held_u32 = reinterpret_cast<uint32_t*>(held_buf + p.held_offset);

    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid >> 5;
    uint32_t lane = tid & 31u;
    uint32_t lane_mask_lt = (lane == 0u) ? 0u : ((1u << lane) - 1u);

    __shared__ uint32_t s_warp_counts[8];
    __shared__ uint32_t s_warp_offsets[8];
    __shared__ uint32_t s_chunk_base;
    __shared__ uint32_t s_overflow;

    for (uint32_t base = 0u; base < p.dword_count; base += 256u) {
        uint32_t idx = base + tid;
        bool valid = (idx < p.dword_count);
        uint32_t v_now = valid ? now_u32[idx] : 0u;
        uint32_t v_held = valid ? held_u32[idx] : 0u;
        bool changed = valid && (v_now != v_held);

        uint32_t mask = __ballot_sync(0xFFFFFFFFu, changed);
        if (__syncthreads_or(mask != 0u) == 0) {
            // Fast path: all 256 dwords (1 KiB) unchanged.
            continue;
        }

        if (lane == 0u) {
            s_warp_counts[warp_id] = __popc(mask);
        }
        __syncthreads();

        if (tid == 0u) {
            uint32_t sum = 0u;
            for (uint32_t w = 0u; w < 8u; ++w) {
                s_warp_offsets[w] = sum;
                sum += s_warp_counts[w];
            }
            uint32_t claimed = atomicAdd(global_counter, sum);
            if (claimed + sum > max_pairs) {
                atomicOr(overflow_flag, 1u);
                s_overflow = 1u;
            } else {
                s_chunk_base = claimed;
                s_overflow = 0u;
            }
        }
        __syncthreads();

        if (changed) {
            if (s_overflow == 0u) {
                uint32_t lane_rank = __popc(mask & lane_mask_lt);
                uint32_t out_idx = s_chunk_base + s_warp_offsets[warp_id] + lane_rank;
                out_pairs[out_idx].piece_id = p.piece_id;
                out_pairs[out_idx].dword_offset = idx;
                out_pairs[out_idx].value = v_now;
            }
            if (update_held != 0u) {
                held_u32[idx] = v_now;
            }
        }
        __syncthreads();
    }
}

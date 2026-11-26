//! GFX10.3 RB+ swizzle lookup table packing and C-ABI kernel descriptors.
//!
//! Single-sources all 45 immutable RDNA2 swizzle patterns directly from
//! `victory_formats::texture::Layout::all_swizzle_luts()` so GPU kernels match
//! `Layout::offset` and `Layout::detile` bit-for-bit without duplicating tables.

use victory_formats::texture::{Dimension, Layout, Swizzle};

/// Per-pattern offsets inside the flat GPU swizzle LUT array.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LutMeta {
    pub x_off: u32,
    pub y_off: u32,
    pub z_off: u32,
    pub z_len: u32,
}

/// C-ABI descriptor for one RDNA2 subresource detile dispatch.
///
/// Must stay layout-identical to `victory_gpu::graphics::contracts::DetileDispatchDesc`
/// and `kernels/victory_kernels.cu::DetileDispatchDesc`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetileDispatchDesc {
    pub src_base_offset: u64,
    pub dst_base_offset: u64,
    pub width_elements: u32,
    pub height_elements: u32,
    pub depth_slices: u32,
    pub element_bytes: u32,
    pub block_dims: [u32; 3],
    pub block_log2: u32,
    pub mip_pitch_elements: u32,
    pub dst_row_pitch_bytes: u32,
    pub dst_depth_pitch_bytes: u32,
    pub tail_x: u32,
    pub tail_y: u32,
    pub pattern_id: u32,
    pub thick: u32,
    pub slice_size: u64,
    pub first_slice: u32,
    pub _pad: u32,
}

/// C-ABI descriptor for one independent LZ4 or GDeflate v1 tile/block.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecompressBlockDesc {
    pub src_offset: u64,
    pub src_len: u32,
    pub dst_len: u32,
    pub dst_offset: u64,
}

/// C-ABI descriptor for one 64 KiB piece in sparse GPU-to-CPU readback delta compaction.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadbackPieceDesc {
    pub now_offset: u64,
    pub held_offset: u64,
    pub dword_count: u32,
    pub piece_id: u32,
}

/// Compacted changed dword emitted by `compact_readback_deltas`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReadbackDeltaPair {
    pub piece_id: u32,
    pub dword_offset: u32,
    pub value: u32,
}

/// Flattened host copy of all 45 GFX10.3 RB+ swizzle LUTs ready for one-time GPU upload.
pub struct PackedSwizzleLuts {
    pub data: Vec<u32>,
    pub meta: Vec<LutMeta>,
}

/// Packs all 45 canonical `(kind * 3 + block_index) * 5 + element_log2` LUTs from `victory-formats`.
pub fn pack_all_swizzle_luts() -> PackedSwizzleLuts {
    let all = Layout::all_swizzle_luts();
    let mut data = Vec::with_capacity(8192);
    let mut meta = Vec::with_capacity(all.len());

    for pat in all {
        let x_off = data.len() as u32;
        data.extend_from_slice(&pat[0]);

        let y_off = data.len() as u32;
        data.extend_from_slice(&pat[1]);

        let z_off = data.len() as u32;
        let z_len = pat[2].len() as u32;
        data.extend_from_slice(&pat[2]);

        meta.push(LutMeta {
            x_off,
            y_off,
            z_off,
            z_len,
        });
    }

    PackedSwizzleLuts { data, meta }
}

/// Computes the canonical `pattern_id` (`0..45`) or `u32::MAX` for linear layouts.
#[inline]
pub fn pattern_id_for_layout(layout: &Layout) -> Option<u32> {
    if !layout.addressable() {
        return None;
    }
    if layout.desc.swizzle == Swizzle::LINEAR {
        return Some(u32::MAX);
    }
    let (kind, block_index, element_log2) = layout.swizzle_lut_key()?;
    Some(((kind * 3 + block_index) * 5 + element_log2) as u32)
}

/// Builds a [`DetileDispatchDesc`] for one `(mip, slice)` subresource of `layout`.
///
/// `tex_src_offset` is the byte offset of this texture's tiled base inside the batch source buffer.
/// `dst_subresource_offset` is the byte offset of this subresource inside the destination buffer.
pub fn detile_desc_from_layout(
    layout: &Layout,
    mip: u32,
    slice: u32,
    tex_src_offset: u64,
    dst_subresource_offset: u64,
    dst_row_pitch_bytes: u32,
    dst_depth_pitch_bytes: u32,
) -> Option<DetileDispatchDesc> {
    let pattern_id = pattern_id_for_layout(layout)?;
    let m = layout.mips.get(mip as usize)?;
    let first_slice = if layout.desc.dimension == Dimension::Tex3d {
        if slice != 0 {
            return None;
        }
        0
    } else {
        if slice >= layout.slices {
            return None;
        }
        slice
    };
    let block_log2 = if layout.desc.swizzle == Swizzle::LINEAR {
        0
    } else {
        layout.alignment.trailing_zeros()
    };
    let (tail_x, tail_y) = m.tail.map_or((0, 0), |t| (t.x, t.y));

    Some(DetileDispatchDesc {
        src_base_offset: tex_src_offset + m.offset,
        dst_base_offset: dst_subresource_offset,
        width_elements: m.width,
        height_elements: m.height,
        depth_slices: m.depth,
        element_bytes: layout.desc.element_bytes,
        block_dims: layout.block,
        block_log2,
        mip_pitch_elements: m.pitch,
        dst_row_pitch_bytes,
        dst_depth_pitch_bytes,
        tail_x,
        tail_y,
        pattern_id,
        thick: u32::from(layout.thick),
        slice_size: layout.slice_size,
        first_slice,
        _pad: 0,
    })
}

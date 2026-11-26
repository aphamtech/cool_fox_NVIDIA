//! First uploads of textures picked by index on a copy queue of their own
//! (`VICTORY_UPLOAD_AHEAD`): read, detiled and copied by threads of ours as
//! soon as the texture pass has made them, and the slots that wait for them
//! shown when the copy has run. Recorded into the pass's list, as every other
//! upload is, the copy ran with the frame that needs it, so a slot that shows
//! its texture only after its upload ([`crate::descriptor::host::show_after`])
//! was black or grey in that frame, and in the next.
//!
//! Only a texture that no list has used is taken: made in the last second,
//! in the common state, and never asked for an upload before
//! ([`crate::resource::HostTexture::ahead`]). It stays in the common state
//! (a copy queue's copies leave it there), from which the card's shaders read
//! it as any texture.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::Staging;
use crate::object::This;
use crate::resource::HostTexture;

/// [`HostTexture::ahead`]: no upload was asked for yet; one is on its way
/// here; it has run; one was recorded into a list.
pub(crate) const NONE: u8 = 0;
const PENDING: u8 = 1;
const DONE: u8 = 2;
pub(crate) const LISTED: u8 = 3;

/// Bytes of title memory on their way at most: past it uploads go into the
/// lists, as before.
const MOST: u64 = 96 << 20;
/// Bytes and textures of one submission at most.
const BATCH: (u64, usize) = (16 << 20, 64);
/// Larger async DMA copy batches for discrete NVIDIA copy engines.
const NVIDIA_BATCH: (u64, usize) = (32 << 20, 128);

/// Whether it is on (`VICTORY_UPLOAD_AHEAD`; on by default on discrete NVIDIA GPUs).
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_UPLOAD_AHEAD") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_UPLOAD_AHEAD") {
        Some(v) => v != "0",
        None => crate::host::is_nvidia(),
    })
}

fn batch_limits() -> (u64, usize) {
    if crate::host::is_nvidia() {
        NVIDIA_BATCH
    } else {
        BATCH
    }
}

/// An upload on its way.
pub(crate) struct Job {
    staging: Arc<Staging>,
    copy: Arc<HostTexture>,
    asked: std::time::Instant,
    /// What to do when it has run; `None`: it has.
    then: Mutex<Option<Vec<Box<dyn FnOnce() + Send>>>>,
}

impl Job {
    /// Calls `f` when the copy has run: now if it has.
    pub(crate) fn then(&self, f: impl FnOnce() + Send + 'static) {
        let mut then = self.then.lock().unwrap_or_else(|e| e.into_inner());
        match then.as_mut() {
            Some(then) => then.push(Box::new(f)),
            None => {
                drop(then);
                f();
            }
        }
    }
}

/// What a texture asked for here gets.
pub(crate) enum Taken {
    /// A job of its own, to be [`send`] once what follows it is added.
    New(Arc<Job>),
    /// One already on its way, or that has run.
    Known(Arc<Job>),
}

static QUEUE: (Mutex<VecDeque<Arc<Job>>>, Condvar) = (Mutex::new(VecDeque::new()), Condvar::new());
/// Those on their way, by host copy ([`HostTexture::serial`]).
static ON_THEIR_WAY: Mutex<Vec<(u64, Arc<Job>)>> = Mutex::new(Vec::new());
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static LANDED: Condvar = Condvar::new();
static BYTES: AtomicU64 = AtomicU64::new(0);
/// Since the start: uploads, bytes, microseconds from the asking until the
/// copy had run, the longest; askings turned down; waits of lists.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Takes the first upload of `resource`'s levels from `first` on, `there` of
/// whose bytes are mapped (`u64::MAX`: not known). `None`: the list's.
pub(crate) fn take(resource: This, first: u32, there: u64) -> Option<Taken> {
    if on() { claim(resource, first, there) } else { None }
}

fn claim(resource: This, first: u32, there: u64) -> Option<Taken> {
    workers()?;
    let turned_down = || {
        STATS[4].fetch_add(1, Relaxed);
        None
    };
    let copy = crate::resource::host_copy(resource)?;
    if copy.ahead.load(Relaxed) == PENDING {
        let way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
        return way.iter().find(|(serial, _)| *serial == copy.serial).map(|(_, job)| Taken::Known(job.clone()));
    }
    let most = if crate::residency::limited_host_ram() { 32 << 20 } else { MOST };
    if !crate::residency::allow_staging_growth(false)
        || BYTES.load(Relaxed) >= most
        || copy.flags() & 0x7 != 0
        || copy.state.load(Relaxed) != 0
        || copy.grown.load(Relaxed)
        || copy.tiled.is_some()
        || copy.holds.lock().unwrap_or_else(|e| e.into_inner()).is_some()
        || !crate::pool::fresh(&copy.texture)
        || super::host_wrote(resource as usize)
        || crate::resource::alias_of(resource).is_some()
    {
        return turned_down();
    }
    if copy.ahead.compare_exchange(NONE, PENDING, Relaxed, Relaxed).is_err() {
        return turned_down();
    }
    super::PICKING.set(Some(there));
    let staging = super::staging(resource, &copy, first);
    super::PICKING.set(None);
    let Some(staging) = staging.filter(|s| {
        !s.on_demand && s.first == s.level0 && !s.uploaded.load(Relaxed) && s.once.load(Relaxed)
    }) else {
        copy.ahead.store(LISTED, Relaxed);
        return turned_down();
    };
    if super::no_wait() && super::fill_unlocked() {
        staging.expected.store(if there == u64::MAX { super::mapped_bytes(staging.memory, staging.read) } else { there }, Relaxed);
    }
    let job = Arc::new(Job { staging, copy, asked: std::time::Instant::now(), then: Mutex::new(Some(Vec::new())) });
    BYTES.fetch_add(job.staging.read as u64, Relaxed);
    {
        let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
        way.push((job.copy.serial, job.clone()));
        IN_FLIGHT.store(way.len(), std::sync::atomic::Ordering::Release);
    }
    Some(Taken::New(job))
}

/// Sends a job of [`take`] on its way.
pub(crate) fn send(job: &Arc<Job>) {
    QUEUE.0.lock().unwrap_or_else(|e| e.into_inner()).push_back(job.clone());
    QUEUE.1.notify_one();
}

/// Waits until every upload currently on its way here has finished copying
/// and run its completion callbacks (queuing slot flips into `FLIPS`).
/// Returns how many jobs were waited for.
pub(crate) fn drain() -> usize {
    if IN_FLIGHT.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return 0;
    }
    let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
    let waiting = way.len();
    let started = std::time::Instant::now();
    while !way.is_empty() && started.elapsed().as_millis() < 2000 {
        way = LANDED
            .wait_timeout(way, std::time::Duration::from_millis(50))
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    waiting
}

/// A list is about to record an upload into `copy`: one on its way here runs
/// first, and no other is taken after it.
pub(crate) fn listed(copy: &HostTexture) {
    if copy.ahead.compare_exchange(NONE, LISTED, Relaxed, Relaxed) != Err(PENDING) {
        return;
    }
    STATS[5].fetch_add(1, Relaxed);
    let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
    let started = std::time::Instant::now();
    while way.iter().any(|(serial, _)| *serial == copy.serial) && started.elapsed().as_millis() < 1000 {
        way = LANDED.wait_timeout(way, std::time::Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
    }
}

/// The copy queue and its threads, made on first use.
fn workers() -> Option<&'static ID3D12CommandQueue> {
    /// Free-threaded.
    struct Queue(ID3D12CommandQueue);
    // SAFETY: a command queue is free-threaded.
    unsafe impl Send for Queue {}
    // SAFETY: as above.
    unsafe impl Sync for Queue {}
    static WORKERS: OnceLock<Option<Queue>> = OnceLock::new();
    WORKERS
        .get_or_init(|| {
            let host = crate::host::host()?;
            // SAFETY: plain object creation.
            let queue: ID3D12CommandQueue = unsafe {
                host.device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_COPY,
                        ..Default::default()
                    })
                    .ok()?
            };
            let count = if host.is_nvidia() && !crate::residency::limited_host_ram() {
                3
            } else {
                2
            };
            for n in 0..count {
                let worker = Worker::new(&host.device, queue.clone()).ok()?;
                std::thread::Builder::new().name(format!("uploads ahead {n}")).spawn(move || worker.work()).ok()?;
            }
            Some(Queue(queue))
        })
        .as_ref()
        .map(|q| &q.0)
}

#[cfg(feature = "cuda-oxide")]
struct CudaWorkerState {
    bridge: Arc<crate::graphics::cuda_bridge::CudaOxideBridge>,
    vram_buf: crate::graphics::cuda_bridge::SharedVramBuffer,
    fence: crate::graphics::cuda_bridge::SharedTimelineFence,
    host_tiled_scratch: Vec<u8>,
    dispatches: Vec<crate::graphics::contracts::DetileDispatchDesc>,
    placements: Vec<Option<u64>>,
}

#[cfg(feature = "cuda-oxide")]
impl CudaWorkerState {
    /// 40 MiB linear VRAM staging buffer per worker (exceeds `NVIDIA_BATCH` 32 MiB plus
    /// D3D12 512-byte footprint alignment padding).
    const VRAM_STAGING_BYTES: u64 = 40 << 20;

    fn try_new(device: &ID3D12Device) -> Option<Self> {
        let bridge = crate::graphics::cuda_bridge::global_bridge()?;
        let vram_buf = match bridge.create_shared_vram_buffer(device, Self::VRAM_STAGING_BYTES) {
            Ok(b) => b,
            Err(err) => {
                tracing::warn!(error = %err, "cuda-oxide shared VRAM staging buffer unavailable; using CPU detile");
                return None;
            }
        };
        let fence = match bridge.create_shared_timeline_fence(device) {
            Ok(f) => f,
            Err(err) => {
                tracing::warn!(error = %err, "cuda-oxide shared timeline fence unavailable; using CPU detile");
                return None;
            }
        };
        Some(Self {
            bridge,
            vram_buf,
            fence,
            host_tiled_scratch: Vec::with_capacity(4 << 20),
            dispatches: Vec::with_capacity(256),
            placements: Vec::with_capacity(128),
        })
    }
}

#[cfg(feature = "cuda-oxide")]
static CUDA_DETILE_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

struct Worker {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    #[cfg(feature = "cuda-oxide")]
    cuda: Option<CudaWorkerState>,
}

// SAFETY: a worker's objects are its thread's own; the queue is free-threaded.
unsafe impl Send for Worker {}

impl Worker {
    fn new(device: &ID3D12Device, queue: ID3D12CommandQueue) -> windows::core::Result<Worker> {
        #[cfg(feature = "cuda-oxide")]
        let cuda = CudaWorkerState::try_new(device);
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            Ok(Worker {
                queue,
                allocator,
                list,
                fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
                #[cfg(feature = "cuda-oxide")]
                cuda,
            })
        }
    }

    fn work(mut self) {
        let mut value = 0;
        loop {
            let mut jobs = Vec::new();
            {
                let mut queue = QUEUE.0.lock().unwrap_or_else(|e| e.into_inner());
                while queue.is_empty() {
                    queue = QUEUE.1.wait(queue).unwrap_or_else(|e| e.into_inner());
                }
                let (max_bytes, max_jobs) = batch_limits();
                let mut bytes = 0;
                while bytes < max_bytes && jobs.len() < max_jobs
                    && let Some(job) = queue.pop_front()
                {
                    bytes += job.staging.read as u64;
                    jobs.push(job);
                }
            }
            value += 1;
            if let Err(e) = self.copy(&jobs, value) {
                tracing::warn!(error = %e, "uploads ahead did not run");
            }
            finish(jobs);
        }
    }

    fn copy(&mut self, jobs: &[Arc<Job>], value: u64) -> windows::core::Result<()> {
        #[cfg(feature = "cuda-oxide")]
        let cuda_wait: Option<u64> = self.prepare_cuda_batch(jobs);

        #[cfg(feature = "cuda-oxide")]
        {
            for (idx, job) in jobs.iter().enumerate() {
                let on_cuda = self
                    .cuda
                    .as_ref()
                    .and_then(|c| c.placements.get(idx).copied().flatten())
                    .is_some();
                if !on_cuda {
                    job.staging.refresh();
                }
            }
        }
        #[cfg(not(feature = "cuda-oxide"))]
        {
            for job in jobs {
                job.staging.refresh();
            }
        }

        // SAFETY: this worker's allocator and list, whose last submission has
        // run; the jobs hold what the list names until after the wait.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            #[cfg(feature = "cuda-oxide")]
            {
                for (idx, job) in jobs.iter().enumerate() {
                    if let Some((vram_res, vram_base)) = self.cuda.as_ref().and_then(|c| {
                        c.placements
                            .get(idx)
                            .copied()
                            .flatten()
                            .map(|base| (c.vram_buf.d3d12_resource(), base))
                    }) {
                        job.record_from(&self.list, vram_res, vram_base);
                    } else {
                        job.record(&self.list);
                    }
                }
            }
            #[cfg(not(feature = "cuda-oxide"))]
            {
                for job in jobs {
                    job.record(&self.list);
                }
            }
            self.list.Close()?;

            // GPU-side hardware timeline wait on CUDA's shared fence (zero CPU stall).
            #[cfg(feature = "cuda-oxide")]
            if let (Some(fence_val), Some(cuda)) = (cuda_wait, self.cuda.as_ref()) {
                self.queue.Wait(cuda.fence.d3d12_fence(), fence_val)?;
            }

            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.queue.Signal(&self.fence, value)?;
            // (Without an event: this waits.)
            self.fence.SetEventOnCompletion(value, windows::Win32::Foundation::HANDLE::default())?;
        }
        Ok(())
    }

    /// Gathers eligible RDNA2 textures in `jobs`, uploads their tiled bytes once, and launches
    /// a single SM 8.9 CUDA detile kernel directly into the worker's DX12-shared VRAM buffer.
    #[cfg(feature = "cuda-oxide")]
    fn prepare_cuda_batch(&mut self, jobs: &[Arc<Job>]) -> Option<u64> {
        use crate::graphics::contracts::{DetileDispatchDesc, GpuComputeBackend};

        let cuda = self.cuda.as_mut()?;
        cuda.host_tiled_scratch.clear();
        cuda.dispatches.clear();
        cuda.placements.clear();
        cuda.placements.resize(jobs.len(), None);

        let started = std::time::Instant::now();
        let vram_cap = cuda.vram_buf.byte_len();
        let mut vram_cursor = 0u64;
        let mut cuda_textures = 0u64;
        let mut cuda_bytes = 0u64;

        for (idx, job) in jobs.iter().enumerate() {
            let s = &job.staging;
            if s.on_demand
                || s.read == 0
                || s.bytes == 0
                || s.footprints.is_empty()
                || !s.layout.addressable()
            {
                continue;
            }
            let vram_base = vram_cursor.next_multiple_of(512);
            if vram_base.saturating_add(s.bytes) > vram_cap {
                continue;
            }

            // Lock staging metadata and verify it still needs its initial detile.
            let mut last = s.last.lock().unwrap_or_else(|e| e.into_inner());
            let exact = s.exact.swap(false, Relaxed);
            let alone = s.once.load(Relaxed);
            if alone && !exact && last.hash.is_some() {
                continue;
            }

            // Reserve 16-byte aligned slice inside the batch host tiled buffer.
            let src_start = cuda.host_tiled_scratch.len().next_multiple_of(16);
            let prev_len = cuda.host_tiled_scratch.len();
            cuda.host_tiled_scratch.resize(src_start + s.read, 0);
            let dst_slice = &mut cuda.host_tiled_scratch[src_start..src_start + s.read];

            let read_ok = if super::no_wait() && !exact {
                if let Some(_mapped) = crate::watch::title_memory_reading_now() {
                    if super::committed(s.memory, s.read) {
                        if super::through_copy() {
                            crate::host::copy_mapped(dst_slice.as_mut_ptr(), s.memory, s.read, false);
                        } else {
                            // SAFETY: memory is committed and pinned by `_mapped`.
                            let src = unsafe { std::slice::from_raw_parts(s.memory, s.read) };
                            dst_slice.copy_from_slice(src);
                        }
                        true
                    } else {
                        crate::host::copy_mapped(dst_slice.as_mut_ptr(), s.memory, s.read, false);
                        last.partial = Some(crate::frame::current());
                        true
                    }
                } else {
                    crate::host::read_into(s.memory, dst_slice)
                }
            } else {
                let _mapped = crate::watch::title_memory_reading();
                if super::committed(s.memory, s.read) {
                    if super::through_copy() {
                        crate::host::copy_mapped(dst_slice.as_mut_ptr(), s.memory, s.read, false);
                    } else {
                        // SAFETY: memory is committed and pinned by `_mapped`.
                        let src = unsafe { std::slice::from_raw_parts(s.memory, s.read) };
                        dst_slice.copy_from_slice(src);
                    }
                    true
                } else {
                    crate::host::copy_mapped(dst_slice.as_mut_ptr(), s.memory, s.read, false);
                    last.partial = Some(crate::frame::current());
                    true
                }
            };

            if !read_ok {
                cuda.host_tiled_scratch.truncate(prev_len);
                continue;
            }

            let hash = super::digest(dst_slice);
            if last.hash == Some(hash) {
                // Content unchanged; no need to re-detile on CUDA.
                cuda.host_tiled_scratch.truncate(prev_len);
                continue;
            }

            // Build per-subresource CUDA detile descriptors matching `Layout::detile`.
            let dispatches_before = cuda.dispatches.len();
            let mut all_valid = true;
            for (i, (footprint, &rows)) in s.footprints.iter().zip(&s.rows).enumerate() {
                let (mip, slice) = s.part(i);
                let pitch = footprint.Footprint.RowPitch;
                let depth_pitch = pitch.saturating_mul(rows);
                let Some(d) = cuda_accel::detile_desc_from_layout(
                    &s.layout,
                    mip,
                    slice,
                    src_start as u64,
                    vram_base + footprint.Offset,
                    pitch,
                    depth_pitch,
                ) else {
                    all_valid = false;
                    break;
                };
                cuda.dispatches.push(DetileDispatchDesc {
                    src_base_offset: d.src_base_offset,
                    dst_base_offset: d.dst_base_offset,
                    width_elements: d.width_elements,
                    height_elements: d.height_elements,
                    depth_slices: d.depth_slices,
                    element_bytes: d.element_bytes,
                    block_dims: d.block_dims,
                    block_log2: d.block_log2,
                    mip_pitch_elements: d.mip_pitch_elements,
                    dst_row_pitch_bytes: d.dst_row_pitch_bytes,
                    dst_depth_pitch_bytes: d.dst_depth_pitch_bytes,
                    tail_x: d.tail_x,
                    tail_y: d.tail_y,
                    pattern_id: d.pattern_id,
                    thick: d.thick,
                    slice_size: d.slice_size,
                    first_slice: d.first_slice,
                    _pad: d._pad,
                });
            }

            if !all_valid {
                cuda.dispatches.truncate(dispatches_before);
                cuda.host_tiled_scratch.truncate(prev_len);
                continue;
            }

            last.hash = Some(hash);
            s.dirty.store(true, Relaxed);
            cuda.placements[idx] = Some(vram_base);
            vram_cursor = vram_base + s.bytes;
            cuda_textures += 1;
            cuda_bytes += s.read as u64;
        }

        if cuda.dispatches.is_empty() {
            return None;
        }

        let fence_val = cuda.fence.next_signal_value();
        match cuda.bridge.enqueue_detile_batch(
            &cuda.host_tiled_scratch,
            cuda.vram_buf.cuda_handle(),
            &cuda.dispatches,
            None,
            Some((cuda.fence.cuda_handle(), fence_val)),
        ) {
            Ok(()) => {
                let elapsed_us = started.elapsed().as_micros() as u64;
                let preparing = &crate::host::PREPARING;
                preparing[2].fetch_add(elapsed_us, Relaxed);
                preparing[3].fetch_add(cuda_bytes, Relaxed);

                let batches = CUDA_DETILE_STATS[0].fetch_add(1, Relaxed) + 1;
                let tex_total = CUDA_DETILE_STATS[1].fetch_add(cuda_textures, Relaxed) + cuda_textures;
                let bytes_total = CUDA_DETILE_STATS[2].fetch_add(cuda_bytes, Relaxed) + cuda_bytes;
                let us_total = CUDA_DETILE_STATS[3].fetch_add(elapsed_us, Relaxed) + elapsed_us;
                if batches == 1 || batches.is_multiple_of(500) {
                    tracing::info!(
                        "cuda-oxide zero-copy detile: {batches} batches, {tex_total} textures, {} MB, avg {:.1} us/batch",
                        bytes_total >> 20,
                        us_total as f64 / batches as f64
                    );
                }
                Some(fence_val)
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "cuda-oxide detile batch failed; falling back to CPU detile for this batch"
                );
                // Reset `last.hash` on jobs we tentatively claimed so CPU `Staging::refresh` runs cleanly.
                for (idx, job) in jobs.iter().enumerate() {
                    if cuda.placements[idx].is_some() {
                        job.staging.forget();
                        cuda.placements[idx] = None;
                    }
                }
                None
            }
        }
    }
}

impl Job {
    /// # Safety
    /// `list` is an open copy list, and the job lives until it has run.
    unsafe fn record(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: caller's contract; `self.staging.buffer` lives as long as `self`.
        unsafe { self.record_from(list, &self.staging.buffer, self.staging.base) };
    }

    /// # Safety
    /// `list` is an open copy list, and both `source` and `self` live until it has run.
    unsafe fn record_from(
        &self,
        list: &ID3D12GraphicsCommandList,
        source: &ID3D12Resource,
        base: u64,
    ) {
        let (s, texture) = (&self.staging, &self.copy.texture);
        for (i, footprint) in s.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: s.first - s.level0 + i as u32 },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(source) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: base + footprint.Offset, ..*footprint },
                },
            };
            // SAFETY: the caller's.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
}

/// The jobs' copies have run.
fn finish(jobs: Vec<Arc<Job>>) {
    for job in &jobs {
        job.staging.uploaded.store(true, Relaxed);
        job.staging.dirty.store(false, Relaxed);
        job.staging.note_held();
        job.copy.ahead.store(DONE, Relaxed);
        BYTES.fetch_sub(job.staging.read as u64, Relaxed);
    }
    for job in &jobs {
        let us = job.asked.elapsed().as_micros() as u64;
        let n = STATS[0].fetch_add(1, Relaxed) + 1;
        let bytes = STATS[1].fetch_add(job.staging.read as u64, Relaxed) + job.staging.read as u64;
        let all = STATS[2].fetch_add(us, Relaxed) + us;
        let most = STATS[3].fetch_max(us, Relaxed).max(us);
        if n == 1 || n.is_multiple_of(5000) {
            tracing::info!(
                "textures uploaded ahead of the lists (VICTORY_UPLOAD_AHEAD=1), since the start: {n}, {} MB, {:.1} ms from the asking until they were there, the longest {:.0}; {} turned down, {} waited for by lists",
                bytes >> 20,
                all as f64 / n as f64 / 1e3,
                most as f64 / 1e3,
                STATS[4].load(Relaxed),
                STATS[5].load(Relaxed)
            );
        }
        let then = job.then.lock().unwrap_or_else(|e| e.into_inner()).take();
        for f in then.into_iter().flatten() {
            f();
        }
    }
    // Remove from `ON_THEIR_WAY` only after `job.then` callbacks have queued
    // their `FLIPS` entries so `drain()` observes every flip ready to apply.
    {
        let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
        way.retain(|(serial, _)| !jobs.iter().any(|job| job.copy.serial == *serial));
        IN_FLIGHT.store(way.len(), std::sync::atomic::Ordering::Release);
        LANDED.notify_all();
    }
    crate::descriptor::host::flip_now();
}

/// Whether `resource`'s host copy was filled here: no list still brings what
/// it shows.
pub(crate) fn there(resource: This) -> bool {
    crate::resource::made_copy(resource).is_some_and(|copy| copy.ahead.load(Relaxed) == DONE)
}

/// The state of [`HostTexture::ahead`] of a new host copy.
pub(crate) fn fresh() -> AtomicU8 {
    AtomicU8::new(NONE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The texels of a placed texture arrive by the copy queue, all levels of
    /// them, and what waits for them is called then; a second asking finds the
    /// first on its way, and a later one is the list's.
    #[test]
    fn a_placed_texture_is_filled_ahead_of_the_lists() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let Some(host) = crate::host::host() else { return };
        // 256x256 BC7, nine levels.
        let desc = [3u32, 0, 0, 0, 256, 0, 256, 0x9_0001, 98, 1, 0, 0x105, 0, 0];
        let whole = crate::resource::placed_bytes(&desc).unwrap() as usize;
        // SAFETY: memory of our own that stays.
        let at = unsafe { VirtualAlloc(std::ptr::null(), whole, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        // SAFETY: as above.
        unsafe { (0..whole).for_each(|i| at.add(i).write((i * 13 + i / 509) as u8)) };
        let texture = crate::resource::texture_at(at as u64, &desc).unwrap();
        crate::resource::want_level(texture, 0);
        let copy = crate::resource::host_copy(texture).unwrap();
        if !crate::pool::fresh(&copy.texture) {
            return;
        }
        assert!(!there(texture));
        let Some(Taken::New(job)) = claim(texture, 0, u64::MAX) else { panic!("not taken") };
        let (told, heard) = std::sync::mpsc::channel();
        let also = told.clone();
        let Some(Taken::Known(same)) = claim(texture, 0, u64::MAX) else { panic!("not known") };
        assert!(Arc::ptr_eq(&job, &same));
        job.then(move || told.send(1).unwrap());
        same.then(move || also.send(2).unwrap());
        send(&job);
        let wait = std::time::Duration::from_secs(20);
        assert_eq!((heard.recv_timeout(wait), heard.recv_timeout(wait)), (Ok(1), Ok(2)));
        assert!(there(texture) && claim(texture, 0, u64::MAX).is_none());
        // Level 0 as the card has it, and as the layout says.
        let layout = crate::resource::texture_memory(texture)
            .and_then(|(_, d)| crate::resource::texture_layout(&d).map(|t| t.planes[0].layout.clone()))
            .unwrap();
        let (pitch, rows) = (1024usize, 64usize);
        let mut expected = vec![0u8; pitch * rows];
        // SAFETY: the memory above.
        let tiled = unsafe { std::slice::from_raw_parts(at, whole) };
        layout.detile(tiled, 0, 0, &mut expected, pitch, pitch * rows).unwrap();
        let back = host.read_back_buffer((pitch * rows) as u64).unwrap();
        let list = host.new_list().unwrap();
        list.close();
        list.reset();
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(98),
                Width: 256,
                Height: 256,
                Depth: 1,
                RowPitch: pitch as u32,
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&back) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        // SAFETY: an open list, live resources; the mapping spans the buffer.
        unsafe {
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.close();
            host.execute(&[&list]);
            let mut data = std::ptr::null_mut();
            back.Map(0, None, Some(&mut data)).unwrap();
            let seen = std::slice::from_raw_parts(data.cast::<u8>(), pitch * rows);
            assert!(seen == expected.as_slice(), "level 0's blocks");
            back.Unmap(0, None);
        }
    }

    #[test]
    fn drain_waits_until_then_callbacks_have_run() {
        use std::sync::atomic::AtomicBool;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let Some(_host) = crate::host::host() else { return };
        let desc = [3u32, 0, 0, 0, 128, 0, 128, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let whole = crate::resource::placed_bytes(&desc).unwrap() as usize;
        // SAFETY: memory of our own that stays for the test.
        let at = unsafe { VirtualAlloc(std::ptr::null(), whole, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes(at, 0x5a, whole) };
        let texture = crate::resource::texture_at(at as u64, &desc).unwrap();
        crate::resource::want_level(texture, 0);
        let copy = crate::resource::host_copy(texture).unwrap();
        if !crate::pool::fresh(&copy.texture) {
            return;
        }
        let Some(Taken::New(job)) = claim(texture, 0, u64::MAX) else { panic!("not taken") };
        let ran = Arc::new(AtomicBool::new(false));
        let flag = ran.clone();
        job.then(move || flag.store(true, std::sync::atomic::Ordering::Release));
        send(&job);
        assert!(drain() >= 1);
        assert!(ran.load(std::sync::atomic::Ordering::Acquire), "job.then callback must run before drain() returns");
        assert_eq!(drain(), 0);
    }
}

//! Subsystem A.3 — Coalescing resource transition, aliasing, and UAV barrier batcher.
//!
//! Xbox Series X|S titles frequently issue single-barrier `ResourceBarrier`
//! calls, redundant read-to-read state changes, Xbox compression-hint upper
//! state bits (`bits 25..=31`), and repeated UAV barriers. On discrete PC GPUs
//! these create unnecessary driver validation overhead and pipeline bubbles.
//!
//! [`CoalescingBarrierBatcher`] strips Xbox compression bits, eliminates no-op
//! and redundant read-to-read transitions, folds consecutive same-resource
//! transitions when no write hazard intervenes, deduplicates UAV barriers, and
//! emits a single coalesced slice to [`BarrierSink::submit_barriers`] on flush.

use std::collections::{HashMap, HashSet};

use crate::graphics::contracts::{
    AliasingBarrierDesc, BarrierBatchStats, BarrierBatcher, BarrierSink, ResourceId,
    ResourceStateFlags, SubresourceIndex, TransitionBarrierDesc, TranslatedBarrier, UavBarrierDesc,
};

/// Coalescing barrier batcher implementing [`BarrierBatcher`].
#[derive(Debug, Default)]
pub struct CoalescingBarrierBatcher {
    /// Ordered pending barriers; `None` entries are tombstones from cancelled
    /// transitions or subsumed UAV barriers, compacted on `flush`.
    pending: Vec<Option<TranslatedBarrier>>,
    live_count: usize,
    /// Active transition wave lookup: `(resource, subresource) -> index in pending`.
    /// Cleared on UAV/aliasing wave boundaries or flush.
    wave_transitions: HashMap<(ResourceId, SubresourceIndex), usize>,
    /// Tracks resources that have had a `SubresourceIndex::ALL` transition in the
    /// current transition wave so specific subresource transitions are not
    /// reordered past an `ALL` barrier.
    wave_has_all_subresource: HashSet<ResourceId>,
    /// Tracks resources that have had any specific subresource transition in the
    /// current transition wave.
    wave_has_specific_subresource: HashSet<ResourceId>,
    /// Whether the current consecutive UAV wave already contains a global
    /// `Uav(None)` barrier.
    uav_wave_global: bool,
    /// Specific `Uav(Some(resource))` indices in the current consecutive UAV wave.
    uav_wave_specific: HashMap<ResourceId, usize>,
    /// Reusable dense output buffer to avoid per-flush heap allocations.
    flush_scratch: Vec<TranslatedBarrier>,
    stats: BarrierBatchStats,
    /// When `true`, only exact `before == after` transitions are cancelled as no-ops
    /// (preventing desync when an external command-list state tracker already handles
    /// read-subset stripping and expects `after` to match D3D12's exact resource state).
    exact_state_tracking: bool,
}

impl CoalescingBarrierBatcher {
    /// Creates a new empty barrier batcher.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a barrier batcher synchronized with an external exact D3D12 state tracker.
    #[must_use]
    pub fn with_exact_state_tracking() -> Self {
        Self {
            exact_state_tracking: true,
            ..Self::default()
        }
    }

    /// Creates a barrier batcher with pre-allocated pending capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            pending: Vec::with_capacity(capacity),
            flush_scratch: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Convenience helper accepting raw Xbox `D3D12_RESOURCE_STATES` values
    /// (stripping upper compression/DCC bits `25..=31`).
    pub fn push_xbox_transition(
        &mut self,
        resource: ResourceId,
        subresource: SubresourceIndex,
        raw_before: u32,
        raw_after: u32,
    ) {
        self.push_transition(TransitionBarrierDesc {
            resource,
            subresource,
            before: ResourceStateFlags::from_xbox_raw(raw_before),
            after: ResourceStateFlags::from_xbox_raw(raw_after),
        });
    }

    /// Fraction of pushed barriers eliminated by stripping, folding, or
    /// deduplication, in `[0.0, 1.0]`.
    #[must_use]
    pub fn reduction_ratio(&self) -> f64 {
        let pushed =
            self.stats.pushed_transitions + self.stats.pushed_uav + self.stats.pushed_aliasing;
        if pushed == 0 {
            return 0.0;
        }
        let eliminated = self.stats.stripped_noop_or_read_to_read
            + self.stats.folded_transitions
            + self.stats.deduplicated_uav;
        (eliminated as f64 / pushed as f64).clamp(0.0, 1.0)
    }

    fn clear_transition_wave(&mut self) {
        self.wave_transitions.clear();
        self.wave_has_all_subresource.clear();
        self.wave_has_specific_subresource.clear();
    }

    fn clear_uav_wave(&mut self) {
        self.uav_wave_global = false;
        self.uav_wave_specific.clear();
    }

    #[inline]
    fn is_noop_or_redundant_read(&self, before: ResourceStateFlags, after: ResourceStateFlags) -> bool {
        if before == after {
            return true;
        }
        !self.exact_state_tracking
            && before.is_read_only()
            && after.is_read_only()
            && before.contains(after)
    }

    fn push_raw_transition(&mut self, desc: TransitionBarrierDesc) {
        let idx = self.pending.len();
        self.pending.push(Some(TranslatedBarrier::Transition(desc)));
        self.live_count += 1;
        self.wave_transitions
            .insert((desc.resource, desc.subresource), idx);
        if desc.subresource == SubresourceIndex::ALL {
            self.wave_has_all_subresource.insert(desc.resource);
        } else {
            self.wave_has_specific_subresource.insert(desc.resource);
        }
    }
}

impl BarrierBatcher for CoalescingBarrierBatcher {
    fn push_transition(&mut self, desc: TransitionBarrierDesc) {
        self.stats.pushed_transitions += 1;
        // Ending any consecutive UAV wave preserves relative ordering around transitions.
        self.clear_uav_wave();

        let before = ResourceStateFlags::from_xbox_raw(desc.before.0);
        let after = ResourceStateFlags::from_xbox_raw(desc.after.0);

        if self.is_noop_or_redundant_read(before, after) {
            self.stats.stripped_noop_or_read_to_read += 1;
            return;
        }

        let normalized = TransitionBarrierDesc {
            resource: desc.resource,
            subresource: desc.subresource,
            before,
            after,
        };

        // Do not reorder across `ALL` vs specific subresource transitions on the
        // same resource: if a mixed subresource transition arrives, break the
        // resource's folding chain for safety.
        let mixed_subresource_conflict = if desc.subresource == SubresourceIndex::ALL {
            self.wave_has_specific_subresource.contains(&desc.resource)
        } else {
            self.wave_has_all_subresource.contains(&desc.resource)
        };

        if mixed_subresource_conflict {
            self.wave_transitions
                .retain(|&(res, _), _| res != desc.resource);
            self.wave_has_all_subresource.remove(&desc.resource);
            self.wave_has_specific_subresource.remove(&desc.resource);
            self.push_raw_transition(normalized);
            return;
        }

        let key = (desc.resource, desc.subresource);
        if let Some(&prev_idx) = self.wave_transitions.get(&key)
            && let Some(TranslatedBarrier::Transition(prev)) = self.pending[prev_idx]
        {
            // Fold `A -> B` + `B -> C` into `A -> C` ONLY when `B` is non-write
            // (preserving write-to-read and read-to-write hazards).
            let chains_cleanly = prev.after == before
                || (prev.after.is_read_only()
                    && before.is_read_only()
                    && prev.after.contains(before));
            if chains_cleanly && !prev.after.has_write() {
                self.stats.folded_transitions += 1;
                if self.is_noop_or_redundant_read(prev.before, after) {
                    // Cancels `A -> B -> A` (or `A_read -> B_read -> A_subset`) cleanly.
                    self.pending[prev_idx] = None;
                    self.live_count -= 1;
                    self.wave_transitions.remove(&key);
                    self.stats.stripped_noop_or_read_to_read += 1;
                } else {
                    self.pending[prev_idx] = Some(TranslatedBarrier::Transition(
                        TransitionBarrierDesc {
                            resource: prev.resource,
                            subresource: prev.subresource,
                            before: prev.before,
                            after,
                        },
                    ));
                }
                return;
            }
        }

        self.push_raw_transition(normalized);
    }

    fn push_aliasing(&mut self, desc: AliasingBarrierDesc) {
        self.stats.pushed_aliasing += 1;
        // Aliasing barriers are ordering boundaries for both transitions and UAV waves.
        self.clear_transition_wave();
        self.clear_uav_wave();

        self.pending.push(Some(TranslatedBarrier::Aliasing(desc)));
        self.live_count += 1;
    }

    fn push_uav(&mut self, desc: UavBarrierDesc) {
        self.stats.pushed_uav += 1;
        // UAV barriers end the current transition folding wave to preserve ordering.
        self.clear_transition_wave();

        if self.uav_wave_global {
            // A global `Uav(None)` already covers every resource in this wave.
            self.stats.deduplicated_uav += 1;
            return;
        }

        match desc.resource {
            None => {
                // Global UAV subsumes all earlier specific UAV barriers in this wave.
                for (_, idx) in self.uav_wave_specific.drain() {
                    if self.pending[idx].take().is_some() {
                        self.live_count -= 1;
                        self.stats.deduplicated_uav += 1;
                    }
                }
                self.uav_wave_global = true;
                self.pending.push(Some(TranslatedBarrier::Uav(desc)));
                self.live_count += 1;
            }
            Some(rid) => {
                if self.uav_wave_specific.contains_key(&rid) {
                    self.stats.deduplicated_uav += 1;
                    return;
                }
                let idx = self.pending.len();
                self.pending.push(Some(TranslatedBarrier::Uav(desc)));
                self.live_count += 1;
                self.uav_wave_specific.insert(rid, idx);
            }
        }
    }

    fn flush(&mut self, sink: &mut dyn BarrierSink) -> usize {
        if self.live_count == 0 {
            self.pending.clear();
            self.clear_transition_wave();
            self.clear_uav_wave();
            return 0;
        }

        self.flush_scratch.clear();
        self.flush_scratch
            .extend(self.pending.drain(..).flatten());
        self.live_count = 0;
        self.clear_transition_wave();
        self.clear_uav_wave();

        let count = self.flush_scratch.len();
        if count > 0 {
            sink.submit_barriers(&self.flush_scratch);
            self.stats.flushed_calls += 1;
            self.stats.flushed_barriers += count as u64;
        }
        count
    }

    fn pending_len(&self) -> usize {
        self.live_count
    }

    fn stats(&self) -> BarrierBatchStats {
        self.stats
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.live_count = 0;
        self.clear_transition_wave();
        self.clear_uav_wave();
        self.flush_scratch.clear();
        self.stats = BarrierBatchStats::default();
    }
}

/// Recording [`BarrierSink`] capturing each coalesced submission batch for
/// headless verification and benchmarks.
#[derive(Debug, Default)]
pub struct RecordingBarrierSink {
    pub calls: Vec<Vec<TranslatedBarrier>>,
}

impl RecordingBarrierSink {
    /// Creates an empty recording sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of `submit_barriers` calls observed.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.calls.len()
    }

    /// Total number of barriers submitted across all calls.
    #[must_use]
    pub fn total_barriers(&self) -> usize {
        self.calls.iter().map(Vec::len).sum()
    }

    /// Flattens all submitted barriers in submission order.
    #[must_use]
    pub fn flattened(&self) -> Vec<TranslatedBarrier> {
        self.calls
            .iter()
            .flat_map(|batch| batch.iter().copied())
            .collect()
    }
}

impl BarrierSink for RecordingBarrierSink {
    fn submit_barriers(&mut self, barriers: &[TranslatedBarrier]) {
        self.calls.push(barriers.to_vec());
    }
}

/// Thin audited Windows D3D12 [`BarrierSink`] translating `&[TranslatedBarrier]`
/// into a single `ID3D12GraphicsCommandList::ResourceBarrier` call per flush.
#[cfg(windows)]
pub struct D3D12CommandListBarrierSink<'a> {
    list: &'a windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    resources: &'a HashMap<ResourceId, windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    native_scratch: Vec<windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER>,
    submitted_calls: usize,
    submitted_barriers: usize,
    skipped_unmapped: usize,
}

#[cfg(windows)]
impl<'a> D3D12CommandListBarrierSink<'a> {
    /// Creates a sink recording into an open D3D12 graphics command list using
    /// `resources` to resolve opaque [`ResourceId`] handles.
    #[must_use]
    pub fn new(
        list: &'a windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
        resources: &'a HashMap<ResourceId, windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    ) -> Self {
        Self {
            list,
            resources,
            native_scratch: Vec::new(),
            submitted_calls: 0,
            submitted_barriers: 0,
            skipped_unmapped: 0,
        }
    }

    /// Number of `ID3D12GraphicsCommandList::ResourceBarrier` calls issued.
    #[must_use]
    pub fn submitted_calls(&self) -> usize {
        self.submitted_calls
    }

    /// Total native barriers submitted to the command list.
    #[must_use]
    pub fn submitted_barriers(&self) -> usize {
        self.submitted_barriers
    }

    /// Count of barriers skipped because a referenced [`ResourceId`] was not
    /// present in `resources` (preventing unmapped GPU VA / null-COM hazards).
    #[must_use]
    pub fn skipped_unmapped(&self) -> usize {
        self.skipped_unmapped
    }

    #[inline]
    fn borrow_com_resource(
        res: Option<&windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    ) -> std::mem::ManuallyDrop<Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>> {
        use std::mem::ManuallyDrop;
        use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
        const _: () = assert!(
            size_of::<ID3D12Resource>() == size_of::<ManuallyDrop<Option<ID3D12Resource>>>()
        );
        match res {
            Some(r) => {
                // SAFETY: `ID3D12Resource` is a transparent non-null COM pointer,
                // so `ManuallyDrop<Option<ID3D12Resource>>` has identical layout
                // (verified by the const assert above). `ManuallyDrop` suppresses
                // `Release`, and `r` stays borrowed from `self.resources` across
                // the synchronous `ResourceBarrier` call.
                unsafe { std::mem::transmute_copy(r) }
            }
            None => ManuallyDrop::new(None),
        }
    }
}

#[cfg(windows)]
impl BarrierSink for D3D12CommandListBarrierSink<'_> {
    fn submit_barriers(&mut self, barriers: &[TranslatedBarrier]) {
        use std::mem::ManuallyDrop;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_ALIASING_BARRIER, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
            D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_TYPE_UAV,
            D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER,
        };

        self.native_scratch.clear();
        self.native_scratch.reserve(barriers.len());

        for &barrier in barriers {
            match barrier {
                TranslatedBarrier::Transition(t) => {
                    let Some(res) = self.resources.get(&t.resource) else {
                        self.skipped_unmapped += 1;
                        continue;
                    };
                    self.native_scratch.push(D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: Self::borrow_com_resource(Some(res)),
                                Subresource: t.subresource.0,
                                StateBefore: D3D12_RESOURCE_STATES(t.before.0 as i32),
                                StateAfter: D3D12_RESOURCE_STATES(t.after.0 as i32),
                            }),
                        },
                    });
                }
                TranslatedBarrier::Aliasing(a) => {
                    let before = match a.before {
                        Some(id) => match self.resources.get(&id) {
                            Some(r) => Some(r),
                            None => {
                                self.skipped_unmapped += 1;
                                continue;
                            }
                        },
                        None => None,
                    };
                    let after = match a.after {
                        Some(id) => match self.resources.get(&id) {
                            Some(r) => Some(r),
                            None => {
                                self.skipped_unmapped += 1;
                                continue;
                            }
                        },
                        None => None,
                    };
                    self.native_scratch.push(D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                                pResourceBefore: Self::borrow_com_resource(before),
                                pResourceAfter: Self::borrow_com_resource(after),
                            }),
                        },
                    });
                }
                TranslatedBarrier::Uav(u) => {
                    let target = match u.resource {
                        Some(id) => match self.resources.get(&id) {
                            Some(r) => Some(r),
                            None => {
                                self.skipped_unmapped += 1;
                                continue;
                            }
                        },
                        None => None,
                    };
                    self.native_scratch.push(D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: Self::borrow_com_resource(target),
                            }),
                        },
                    });
                }
            }
        }

        if !self.native_scratch.is_empty() {
            // SAFETY: `self.list` is an open D3D12 graphics command list and all
            // borrowed `ID3D12Resource` pointers in `self.native_scratch` point to
            // live resources held in `self.resources` for the duration of the call.
            unsafe {
                self.list.ResourceBarrier(&self.native_scratch);
            }
            self.submitted_calls += 1;
            self.submitted_barriers += self.native_scratch.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_conservation_law_holds_across_mixed_stream() {
        let mut batcher = CoalescingBarrierBatcher::new();
        let mut sink = RecordingBarrierSink::new();
        let r1 = ResourceId(1);
        let r2 = ResourceId(2);

        // No-op (1 stripped)
        batcher.push_transition(TransitionBarrierDesc {
            resource: r1,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::RENDER_TARGET,
            after: ResourceStateFlags::RENDER_TARGET,
        });
        // Redundant read-to-read (1 stripped)
        batcher.push_transition(TransitionBarrierDesc {
            resource: r1,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::GENERIC_READ,
            after: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
        });
        // Cancel A -> B -> A where B is non-write (1 folded + 1 stripped)
        batcher.push_transition(TransitionBarrierDesc {
            resource: r2,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::RENDER_TARGET,
            after: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
        });
        batcher.push_transition(TransitionBarrierDesc {
            resource: r2,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
            after: ResourceStateFlags::RENDER_TARGET,
        });
        // Fold A -> B -> C (1 folded, 1 emitted)
        batcher.push_transition(TransitionBarrierDesc {
            resource: r1,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::COPY_DEST,
            after: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
        });
        batcher.push_transition(TransitionBarrierDesc {
            resource: r1,
            subresource: SubresourceIndex::ALL,
            before: ResourceStateFlags::PIXEL_SHADER_RESOURCE,
            after: ResourceStateFlags::NON_PIXEL_SHADER_RESOURCE,
        });
        // UAV wave: R1, R2, Global (2 deduplicated, 1 emitted)
        batcher.push_uav(UavBarrierDesc { resource: Some(r1) });
        batcher.push_uav(UavBarrierDesc { resource: Some(r2) });
        batcher.push_uav(UavBarrierDesc { resource: None });
        // Aliasing (1 emitted)
        batcher.push_aliasing(AliasingBarrierDesc {
            before: Some(r1),
            after: Some(r2),
        });

        let emitted = batcher.flush(&mut sink);
        assert_eq!(emitted, 3);
        assert_eq!(sink.call_count(), 1);

        let stats = batcher.stats();
        let total_pushed = stats.pushed_transitions + stats.pushed_uav + stats.pushed_aliasing;
        let eliminated = stats.stripped_noop_or_read_to_read
            + stats.folded_transitions
            + stats.deduplicated_uav;
        assert_eq!(
            total_pushed,
            stats.flushed_barriers + batcher.pending_len() as u64 + eliminated
        );
        assert!((batcher.reduction_ratio() - 0.7).abs() < 1e-9);
    }
}

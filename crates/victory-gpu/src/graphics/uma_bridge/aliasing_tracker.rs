//! Subsystem A.2 — Interval tracker for overlapping `CreatePlacedResource` /
//! `CreatePlacedResourceX` heap ranges.
//!
//! Xbox Series X|S titles aggressively alias placed textures and buffers inside
//! shared memory heaps. On discrete D3D12 GPUs, accessing an aliased resource
//! without an explicit `D3D12_RESOURCE_BARRIER_TYPE_ALIASING` barrier causes
//! undefined memory/cache hazard behavior, while referencing a destroyed
//! resource causes unmapped GPU virtual address (VA) faults.
//!
//! [`IntervalAliasingTracker`] maintains per-heap non-overlapping active-owner
//! intervals in a [`BTreeMap`] for `O(log N + K)` overlap queries and minimal
//! barrier emission.

use std::collections::{BTreeMap, HashMap};

use smallvec::SmallVec;

use crate::graphics::contracts::{
    AliasingBarrierDesc, AliasingTracker, HeapId, HeapRange, ResourceId, UmaBridgeError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActiveSegment {
    end: u64,
    owner: ResourceId,
}

/// Per-heap interval tracker emitting minimal D3D12 aliasing barriers on
/// overlapping placed-resource accesses.
#[derive(Debug, Default)]
pub struct IntervalAliasingTracker {
    registered: HashMap<ResourceId, HeapRange>,
    /// Per-heap map from segment `start` offset to `{ end, owner }`.
    ///
    /// Invariant: for every entry `(start, seg)`, `start < seg.end`, and for
    /// any two consecutive entries `(s1, seg1)` and `(s2, _)` in the same map,
    /// `seg1.end <= s2`.
    active_by_heap: HashMap<HeapId, BTreeMap<u64, ActiveSegment>>,
}

impl IntervalAliasingTracker {
    /// Creates an empty aliasing tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of currently registered placed resources.
    #[must_use]
    pub fn registered_count(&self) -> usize {
        self.registered.len()
    }

    /// Returns the registered heap range for `resource`, if any.
    #[must_use]
    pub fn range_of(&self, resource: ResourceId) -> Option<HeapRange> {
        self.registered.get(&resource).copied()
    }

    /// Returns the number of active ownership segments tracked on `heap_id`.
    #[must_use]
    pub fn active_segment_count(&self, heap_id: HeapId) -> usize {
        self.active_by_heap.get(&heap_id).map_or(0, BTreeMap::len)
    }

    /// Removes all active segments owned by `resource` within `range`.
    fn remove_active_segments_for(&mut self, resource: ResourceId, range: HeapRange) {
        let Some(map) = self.active_by_heap.get_mut(&range.heap_id) else {
            return;
        };
        let start = range.offset;
        let Some(end) = range.end() else {
            return;
        };

        let mut to_remove: SmallVec<[u64; 8]> = SmallVec::new();

        if let Some((&prev_start, prev_seg)) = map.range(..start).next_back()
            && prev_seg.end > start
            && prev_seg.owner == resource
        {
            to_remove.push(prev_start);
        }
        for (&seg_start, seg) in map.range(start..end) {
            if seg.owner == resource {
                to_remove.push(seg_start);
            }
        }

        for key in to_remove {
            map.remove(&key);
        }

        if map.is_empty() {
            self.active_by_heap.remove(&range.heap_id);
        }
    }
}

impl AliasingTracker for IntervalAliasingTracker {
    fn register_placed(
        &mut self,
        resource: ResourceId,
        range: HeapRange,
    ) -> Result<(), UmaBridgeError> {
        if range.size == 0 {
            return Err(UmaBridgeError::InvalidLayout {
                size: 0,
                alignment: 1,
            });
        }
        if range.end().is_none() {
            return Err(UmaBridgeError::RangeOverflow {
                heap_id: range.heap_id,
                offset: range.offset,
                size: range.size,
            });
        }

        // Clear any prior registration/ownership for the same ResourceId so
        // stale segments can never linger.
        if let Some(old_range) = self.registered.remove(&resource) {
            self.remove_active_segments_for(resource, old_range);
        }

        self.registered.insert(resource, range);
        Ok(())
    }

    fn unregister_placed(&mut self, resource: ResourceId) {
        if let Some(range) = self.registered.remove(&resource) {
            self.remove_active_segments_for(resource, range);
        }
    }

    fn on_resource_access(&mut self, resource: ResourceId) -> SmallVec<[AliasingBarrierDesc; 4]> {
        let Some(&range) = self.registered.get(&resource) else {
            return SmallVec::new();
        };
        let start = range.offset;
        let Some(end) = range.end() else {
            return SmallVec::new();
        };

        let map = self.active_by_heap.entry(range.heap_id).or_default();

        // Fast path: check if `resource` already owns a single contiguous
        // segment covering `[start, end)`.
        if let Some((&cov_start, cov_seg)) = map.range(..=start).next_back()
            && cov_start <= start
            && cov_seg.end >= end
            && cov_seg.owner == resource
        {
            return SmallVec::new();
        }

        // Gather all active segments intersecting `[start, end)` in O(log N + K).
        let mut overlapping: SmallVec<[(u64, ActiveSegment); 8]> = SmallVec::new();
        if let Some((&prev_start, &prev_seg)) = map.range(..start).next_back()
            && prev_seg.end > start
        {
            overlapping.push((prev_start, prev_seg));
        }
        for (&seg_start, &seg) in map.range(start..end) {
            overlapping.push((seg_start, seg));
        }

        let mut barriers: SmallVec<[AliasingBarrierDesc; 4]> = SmallVec::new();
        let mut evicted_owners: SmallVec<[ResourceId; 8]> = SmallVec::new();
        let mut merged_start = start;
        let mut merged_end = end;

        for (seg_start, seg) in overlapping {
            map.remove(&seg_start);

            if seg.owner == resource {
                merged_start = merged_start.min(seg_start);
                merged_end = merged_end.max(seg.end);
            } else {
                if !evicted_owners.contains(&seg.owner) {
                    evicted_owners.push(seg.owner);
                    barriers.push(AliasingBarrierDesc {
                        before: Some(seg.owner),
                        after: Some(resource),
                    });
                }
                // Preserve non-overlapping left tail of evicted owner.
                if seg_start < start {
                    map.insert(
                        seg_start,
                        ActiveSegment {
                            end: start,
                            owner: seg.owner,
                        },
                    );
                }
                // Preserve non-overlapping right tail of evicted owner.
                if seg.end > end {
                    map.insert(
                        end,
                        ActiveSegment {
                            end: seg.end,
                            owner: seg.owner,
                        },
                    );
                }
            }
        }

        // Coalesce with immediately adjacent segments owned by the same `resource`.
        if let Some((&left_start, &left_seg)) = map.range(..merged_start).next_back()
            && left_seg.end == merged_start
            && left_seg.owner == resource
        {
            map.remove(&left_start);
            merged_start = left_start;
        }
        if let Some(&right_seg) = map.get(&merged_end)
            && right_seg.owner == resource
        {
            map.remove(&merged_end);
            merged_end = right_seg.end;
        }

        map.insert(
            merged_start,
            ActiveSegment {
                end: merged_end,
                owner: resource,
            },
        );

        barriers
    }

    fn active_owner_at(&self, heap_id: HeapId, offset: u64) -> Option<ResourceId> {
        let map = self.active_by_heap.get(&heap_id)?;
        let (_, seg) = map.range(..=offset).next_back()?;
        (offset < seg.end).then_some(seg.owner)
    }

    fn clear_active_ownership(&mut self, heap_id: HeapId) {
        self.active_by_heap.remove(&heap_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_owner_fragments_emit_single_barrier_on_fan_in() {
        let mut tracker = IntervalAliasingTracker::new();
        let heap = HeapId(1);
        let a = ResourceId(1);
        let b = ResourceId(2);
        let c = ResourceId(3);

        tracker
            .register_placed(
                a,
                HeapRange {
                    heap_id: heap,
                    offset: 0,
                    size: 300,
                },
            )
            .unwrap();
        tracker
            .register_placed(
                b,
                HeapRange {
                    heap_id: heap,
                    offset: 100,
                    size: 100,
                },
            )
            .unwrap();
        tracker
            .register_placed(
                c,
                HeapRange {
                    heap_id: heap,
                    offset: 0,
                    size: 300,
                },
            )
            .unwrap();

        assert!(tracker.on_resource_access(a).is_empty());
        // B punches a hole in A, splitting A into [0,100) and [200,300).
        let b_barriers = tracker.on_resource_access(b);
        assert_eq!(b_barriers.len(), 1);
        assert_eq!(
            b_barriers[0],
            AliasingBarrierDesc {
                before: Some(a),
                after: Some(b),
            }
        );

        // C covers [0,300), overlapping A, B, A — should emit exactly 1 barrier for A and 1 for B.
        let c_barriers = tracker.on_resource_access(c);
        assert_eq!(c_barriers.len(), 2);
        assert_eq!(
            c_barriers[0],
            AliasingBarrierDesc {
                before: Some(a),
                after: Some(c),
            }
        );
        assert_eq!(
            c_barriers[1],
            AliasingBarrierDesc {
                before: Some(b),
                after: Some(c),
            }
        );
        assert_eq!(tracker.active_segment_count(heap), 1);
    }

    #[test]
    fn overflow_and_zero_size_rejected() {
        let mut tracker = IntervalAliasingTracker::new();
        assert!(matches!(
            tracker.register_placed(
                ResourceId(1),
                HeapRange {
                    heap_id: HeapId(1),
                    offset: 0,
                    size: 0,
                }
            ),
            Err(UmaBridgeError::InvalidLayout { size: 0, .. })
        ));
        assert!(matches!(
            tracker.register_placed(
                ResourceId(2),
                HeapRange {
                    heap_id: HeapId(1),
                    offset: u64::MAX - 10,
                    size: 32,
                }
            ),
            Err(UmaBridgeError::RangeOverflow { .. })
        ));
    }
}

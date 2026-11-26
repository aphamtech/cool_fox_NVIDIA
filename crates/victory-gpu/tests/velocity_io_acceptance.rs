//! Acceptance test suite for Subsystem B (`velocity_io`).
//!
//! Covers:
//! - Exact Xbox Series X|S `DStorageRequestX` (`0x48` B) & `DStorageQueueDescX` (`0x20` B) ABI layout
//! - Zero-stall `enqueue_request` / `enqueue_status` with no premature submission before `submit()`
//! - Out-of-order async completion with strict in-order `VelocityStatusArray` first-failure semantics
//! - Multi-threaded `PinnedUploadRing` backpressure & fence-gated out-of-order reclamation
//! - `#[cfg(windows)]` real Win32 `OVERLAPPED` + IOCP multi-chunk file reads (aligned, unaligned, EOF)
//! - Zlib, LZ4 (block & frame), and multi-tile GDeflate v1 decompression round-trips
//! - `BcPackStubBackend` typed `ProprietaryCodecStub` error & runtime pluggable override
//! - Cross-queue timeline fence wait/signal ordering
//! - `#[cfg(windows)]` real D3D12/WARP Async Compute Queue + Copy Queue fence-sync dispatch round-trip

use std::ffi::c_void;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use victory_gpu::graphics::contracts::{
    AsyncIoBackend, AsyncReadCompletion, AsyncReadOp, BcPackFormat, ComputeQueueBackend,
    DStorageDestination, DStoragePlatformHooks, DStoragePriority, DStorageRequestDesc,
    DStorageSource, DecompressionBackend, DecompressionCodec, DecompressionExecutionMode,
    DecompressionJob, DecompressionOutput, FileHandleId, GpuFenceToken, ResourceId,
    VelocityIoError,
};
use victory_gpu::graphics::velocity_io::dstorage_thunk::{
    DStorageQueueDescX, DStorageRequestX, DefaultPlatformHooks, E_INVALIDARG, E_PENDING, GDEFLATE,
    HRESULT_HANDLE_EOF, InMemoryAsyncIo, LZ4, MEMORY_SOURCE, PAGE_DESTINATION, PinnedUploadRing,
    S_OK, VelocityFactory, VelocityQueueConfig, ZLIB,
};
use victory_gpu::graphics::velocity_io::gpu_decompress_queue::{
    GDeflateHeader, GpuDecompressQueue, MockComputeEvent, MockComputeQueueBackend,
    encode_gdeflate_v1,
};

fn deterministic_payload(len: usize, seed: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        // Mix compressible runs with varied bytes to exercise codecs realistically.
        let b = if (i % 64) < 24 {
            seed.wrapping_add(((i / 64) & 0xFF) as u8)
        } else {
            seed.wrapping_add(((i * 31) ^ (i >> 5)) as u8)
        };
        out.push(b);
    }
    out
}

fn zlib_compress(input: &[u8]) -> Vec<u8> {
    let config = zlib_rs::DeflateConfig {
        level: 6,
        window_bits: 15,
        ..zlib_rs::DeflateConfig::default()
    };
    let mut out = vec![0u8; input.len() + (input.len() / 8) + 64];
    let (compressed, ret) = zlib_rs::compress_slice(&mut out, input, config);
    assert_eq!(ret, zlib_rs::ReturnCode::Ok, "zlib compression must succeed");
    compressed.to_vec()
}

// ---------------------------------------------------------------------------
// 1. ABI Layout & Request Decoder Acceptance Tests
// ---------------------------------------------------------------------------

#[test]
fn abi_layout_matches_xbox_dstorage_x_spec() {
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(std::mem::size_of::<DStorageRequestX>(), 0x48);
        assert_eq!(std::mem::offset_of!(DStorageRequestX, options), 0x00);
        assert_eq!(std::mem::offset_of!(DStorageRequestX, destination), 0x08);
        assert_eq!(std::mem::offset_of!(DStorageRequestX, page_offset), 0x10);
        assert_eq!(std::mem::offset_of!(DStorageRequestX, destination_size), 0x18);
        assert_eq!(std::mem::offset_of!(DStorageRequestX, source), 0x20);
        assert_eq!(std::mem::offset_of!(DStorageRequestX, offset), 0x28);
        assert_eq!(std::mem::offset_of!(DStorageRequestX, source_size), 0x30);
        assert_eq!(std::mem::offset_of!(DStorageRequestX, intermediate_size), 0x34);
        assert_eq!(std::mem::offset_of!(DStorageRequestX, tag), 0x38);
        assert_eq!(std::mem::offset_of!(DStorageRequestX, name), 0x40);

        assert_eq!(std::mem::size_of::<DStorageQueueDescX>(), 0x20);
        assert_eq!(std::mem::offset_of!(DStorageQueueDescX, source_type), 0x00);
        assert_eq!(std::mem::offset_of!(DStorageQueueDescX, capacity), 0x08);
        assert_eq!(std::mem::offset_of!(DStorageQueueDescX, priority), 0x0A);
        assert_eq!(std::mem::offset_of!(DStorageQueueDescX, name), 0x10);
        assert_eq!(std::mem::offset_of!(DStorageQueueDescX, unknown), 0x18);
    }

    let desc_x = DStorageQueueDescX {
        source_type: 0,
        capacity: 4096,
        priority: 2,
        name: std::ptr::null(),
        unknown: 1,
    };
    assert_eq!(desc_x.priority_enum(), DStoragePriority::Realtime);

    let mut dst = [0u8; 128];
    let src = [42u8; 64];

    // Memory source + Page destination + LZ4
    let req = DStorageRequestX {
        options: LZ4 | MEMORY_SOURCE | PAGE_DESTINATION,
        destination: dst.as_mut_ptr().cast(),
        page_offset: 0x0001_1234,
        destination_size: 128,
        source: src.as_ptr().cast(),
        offset: 0,
        source_size: 64,
        intermediate_size: 0,
        tag: 77,
        name: std::ptr::null(),
    };
    let decoded = req
        .to_desc(|_| unreachable!("memory source must not call resolve_file"))
        .expect("decode memory->page LZ4 request");
    assert_eq!(decoded.codec, DecompressionCodec::Lz4);
    assert_eq!(decoded.secondary_codec, None);
    assert_eq!(
        decoded.source,
        DStorageSource::Memory {
            address: src.as_ptr() as usize,
            size: 64,
        }
    );
    assert_eq!(
        decoded.destination,
        DStorageDestination::PhysicalPages {
            page_list_addr: dst.as_mut_ptr() as usize,
            page_offset: 0x1234,
            size: 128,
        }
    );

    // GDeflate flag
    let gdeflate_req = DStorageRequestX {
        options: GDEFLATE,
        destination: dst.as_mut_ptr().cast(),
        page_offset: 0,
        destination_size: 128,
        source: std::ptr::dangling::<c_void>(),
        offset: 4096,
        source_size: 40,
        intermediate_size: 0,
        tag: 0,
        name: std::ptr::null(),
    };
    let gd = gdeflate_req
        .to_desc(|_| Ok(FileHandleId(5)))
        .expect("decode gdeflate request");
    assert_eq!(gd.codec, DecompressionCodec::GDeflate);

    // Reject conflicting flags (ZLIB + LZ4), unknown bits, and oversized uncompressed source
    let bad_conflict = DStorageRequestX {
        options: ZLIB | LZ4,
        ..gdeflate_req
    };
    assert!(matches!(
        bad_conflict.to_desc(|_| Ok(FileHandleId(1))),
        Err(VelocityIoError::InvalidRequest(_))
    ));

    let bad_unknown = DStorageRequestX {
        options: 1 << 23,
        ..gdeflate_req
    };
    assert!(matches!(
        bad_unknown.to_desc(|_| Ok(FileHandleId(1))),
        Err(VelocityIoError::InvalidRequest(_))
    ));

    let bad_uncompressed = DStorageRequestX {
        options: 0,
        source_size: 256,
        destination_size: 128,
        ..gdeflate_req
    };
    assert!(matches!(
        bad_uncompressed.to_desc(|_| Ok(FileHandleId(1))),
        Err(VelocityIoError::InvalidRequest(_))
    ));

    // DefaultPlatformHooks straight path conversion & safe InvalidRequest on page write
    let hooks = DefaultPlatformHooks;
    let utf16: Vec<u16> = "D:\\game\\pak0.bin\0extra".encode_utf16().collect();
    assert_eq!(
        hooks.rewrite_virtual_path(&utf16),
        Path::new("D:\\game\\pak0.bin")
    );
    assert!(matches!(
        hooks.write_physical_pages(0x1000, 0, &[1, 2, 3]),
        Err(VelocityIoError::InvalidRequest(_))
    ));
}

// ---------------------------------------------------------------------------
// 2. Zero-Stall Enqueue & No Premature Submission Before Submit()
// ---------------------------------------------------------------------------

#[test]
fn enqueue_never_auto_submits_before_explicit_submit() {
    let io = Arc::new(InMemoryAsyncIo::new());
    let file_data = deterministic_payload(128 * 1024, 0x5A);
    io.add_file("pak0.bin", file_data.clone());

    let compute = Arc::new(MockComputeQueueBackend::new(1));
    let decompress = Arc::new(GpuDecompressQueue::new(compute));
    let factory = VelocityFactory::new(io, decompress);

    let file = factory.open_file(Path::new("pak0.bin")).expect("open file");
    let status = factory.create_status_array(2);

    // Capacity 32: enqueue 24 requests (> 50% capacity!) + 1 status without submitting.
    let queue = factory
        .create_queue(VelocityQueueConfig {
            capacity: 32,
            priority: DStoragePriority::High,
            staging_ring_bytes: 256 * 1024,
            decompress_workers: 2,
            name: String::from("no-autosubmit-test"),
        })
        .expect("create queue");

    let mut dst_buffers: Vec<Vec<u8>> = (0..24).map(|_| vec![0xCCu8; 4096]).collect();
    for (i, buf) in dst_buffers.iter_mut().enumerate() {
        queue
            .enqueue_request(DStorageRequestDesc {
                codec: DecompressionCodec::Uncompressed,
                secondary_codec: None,
                intermediate_size: 0,
                source: DStorageSource::File {
                    file: file.handle_id(),
                    offset: (i * 4096) as u64,
                    size: 4096,
                },
                destination: DStorageDestination::Memory {
                    address: buf.as_mut_ptr() as usize,
                    size: 4096,
                },
                cancellation_tag: i as u64,
            })
            .expect("enqueue request");
    }
    queue
        .enqueue_status(Arc::clone(&status), 0)
        .expect("enqueue status");

    // Sleep briefly to prove background workers do NOT touch unsubmitted commands!
    thread::sleep(Duration::from_millis(25));
    assert_eq!(queue.unsubmitted_len(), 25);
    assert!(!status.is_complete(0));
    assert_eq!(status.hresult(0), E_PENDING);
    for buf in &dst_buffers {
        assert!(
            buf.iter().all(|&b| b == 0xCC),
            "destination buffer must remain untouched prior to submit()"
        );
    }

    // Fill remaining 7 slots to verify non-blocking StagingRingFull backpressure when full.
    for _ in 0..7 {
        queue
            .enqueue_status(Arc::clone(&status), 1)
            .expect("fill remaining slots");
    }
    assert!(matches!(
        queue.enqueue_status(Arc::clone(&status), 1),
        Err(VelocityIoError::StagingRingFull { .. })
    ));

    // Now explicitly submit and wait for completion.
    let sealed = queue.submit().expect("submit batch");
    assert_eq!(sealed, 32);
    assert_eq!(queue.unsubmitted_len(), 0);
    assert!(queue.wait_idle(Duration::from_secs(5)));

    assert!(status.is_complete(0));
    assert_eq!(status.hresult(0), S_OK);
    assert!(status.is_complete(1));
    assert_eq!(status.hresult(1), S_OK);

    for (i, buf) in dst_buffers.iter().enumerate() {
        assert_eq!(
            buf.as_slice(),
            &file_data[i * 4096..(i + 1) * 4096],
            "chunk {i} mismatch after submit"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Out-of-Order Async Completion & Strict In-Order First-Failure Status Array
// ---------------------------------------------------------------------------

#[test]
fn out_of_order_completions_preserve_in_order_first_failure_status_semantics() {
    let io = Arc::new(InMemoryAsyncIo::new());
    io.set_reverse_completion_order(true);

    let good_payload = deterministic_payload(8192, 0x11);
    let err_file_id = io.register_open_file("fail_access.bin", vec![0u8; 4096]);
    io.inject_file_error(err_file_id, 5); // Win32 ERROR_ACCESS_DENIED -> 0x8007_0005

    let good_file_id = io.register_open_file("good.bin", good_payload.clone());

    let compute = Arc::new(MockComputeQueueBackend::new(1));
    let decompress = Arc::new(GpuDecompressQueue::new(compute));
    let factory = VelocityFactory::new(Arc::clone(&io) as Arc<dyn AsyncIoBackend>, decompress);
    let status = factory.create_status_array(2);

    let queue = factory
        .create_queue(VelocityQueueConfig {
            capacity: 64,
            priority: DStoragePriority::Normal,
            staging_ring_bytes: 128 * 1024,
            decompress_workers: 4,
            name: String::from("ooo-status-test"),
        })
        .expect("create queue");

    let mut dst0 = vec![0u8; 4096];
    let mut dst1 = vec![0u8; 4096];
    let mut dst2 = vec![0u8; 4096];
    let mut dst3 = vec![0u8; 4096];

    // Req 0: fails with injected OS error 5 (0x8007_0005).
    queue
        .enqueue_request(DStorageRequestDesc {
            codec: DecompressionCodec::Uncompressed,
            secondary_codec: None,
            intermediate_size: 0,
            source: DStorageSource::File {
                file: err_file_id,
                offset: 0,
                size: 4096,
            },
            destination: DStorageDestination::Memory {
                address: dst0.as_mut_ptr() as usize,
                size: 4096,
            },
            cancellation_tag: 0,
        })
        .unwrap();

    // Req 1: fails with E_NOTIMPL (BcPack stub) — completes BEFORE Req 0 because of reverse order!
    queue
        .enqueue_request(DStorageRequestDesc {
            codec: DecompressionCodec::BcPack(BcPackFormat::Bc7),
            secondary_codec: None,
            intermediate_size: 0,
            source: DStorageSource::File {
                file: good_file_id,
                offset: 0,
                size: 1024,
            },
            destination: DStorageDestination::Memory {
                address: dst1.as_mut_ptr() as usize,
                size: 4096,
            },
            cancellation_tag: 1,
        })
        .unwrap();

    // Req 2: succeeds cleanly.
    queue
        .enqueue_request(DStorageRequestDesc {
            codec: DecompressionCodec::Uncompressed,
            secondary_codec: None,
            intermediate_size: 0,
            source: DStorageSource::File {
                file: good_file_id,
                offset: 0,
                size: 4096,
            },
            destination: DStorageDestination::Memory {
                address: dst2.as_mut_ptr() as usize,
                size: 4096,
            },
            cancellation_tag: 2,
        })
        .unwrap();

    // Status 0 covers Req 0..=2 -> MUST report Req 0's error (0x8007_0005), NOT Req 1's E_NOTIMPL!
    queue.enqueue_status(Arc::clone(&status), 0).unwrap();

    // Req 3: succeeds cleanly in the next segment.
    queue
        .enqueue_request(DStorageRequestDesc {
            codec: DecompressionCodec::Uncompressed,
            secondary_codec: None,
            intermediate_size: 0,
            source: DStorageSource::File {
                file: good_file_id,
                offset: 4096,
                size: 4096,
            },
            destination: DStorageDestination::Memory {
                address: dst3.as_mut_ptr() as usize,
                size: 4096,
            },
            cancellation_tag: 3,
        })
        .unwrap();

    // Status 1 covers Req 3 only -> MUST report S_OK!
    queue.enqueue_status(Arc::clone(&status), 1).unwrap();

    queue.submit().unwrap();
    assert!(queue.wait_idle(Duration::from_secs(5)));

    let expected_access_denied = 0x8007_0005u32 as i32;
    assert_eq!(status.hresult(0), expected_access_denied);
    assert_eq!(status.hresult(1), S_OK);
    assert_eq!(status.hresult(99), E_INVALIDARG);
    assert_eq!(dst2, good_payload[0..4096]);
    assert_eq!(dst3, good_payload[4096..8192]);
}

// ---------------------------------------------------------------------------
// 4. Multi-Threaded PinnedUploadRing Backpressure & Fence-Gated Reclamation
// ---------------------------------------------------------------------------

#[test]
fn pinned_upload_ring_backpressure_and_out_of_order_fence_reclamation() {
    let ring = PinnedUploadRing::new_standalone(16 * 1024).expect("create 16 KiB ring");
    assert_eq!(ring.capacity_bytes(), Some(16 * 1024));

    let l0 = ring.try_lease(8192).expect("lease 0");
    let l1 = ring.try_lease(8192).expect("lease 1");
    assert!(matches!(
        ring.try_lease(64),
        Err(VelocityIoError::StagingRingFull { requested: 64 })
    ));

    // Retire l1 (fence 5) BEFORE l0 (fence 10) is retired: tail cannot advance past l0!
    ring.retire_lease(
        l1,
        GpuFenceToken {
            queue_id: 1,
            value: 5,
        },
    );
    assert_eq!(ring.reclaim_completed(5), 0);
    assert!(matches!(
        ring.try_lease(64),
        Err(VelocityIoError::StagingRingFull { .. })
    ));

    // Retire l0 with fence 10: still blocked while completed_fence == 9.
    ring.retire_lease(
        l0,
        GpuFenceToken {
            queue_id: 1,
            value: 10,
        },
    );
    assert_eq!(ring.reclaim_completed(9), 0);

    // Once completed_fence reaches 10, BOTH l0 and l1 coalesce and reclaim immediately!
    assert_eq!(ring.reclaim_completed(10), 2);
    assert_eq!(ring.used_bytes(), 0);

    // Now lease across wrap boundary (4 KiB + 12 KiB, retire 4 KiB, lease 8 KiB forcing wrap).
    let a = ring.try_lease(4096).unwrap();
    let b = ring.try_lease(12288).unwrap();
    ring.retire_lease(a, GpuFenceToken { queue_id: 1, value: 11 });
    assert_eq!(ring.reclaim_completed(11), 1);
    let c = ring.try_lease(4096).expect("lease wraps cleanly to offset 0");
    assert_eq!(c.ring_offset(), 0);
    ring.retire_lease(b, GpuFenceToken { queue_id: 1, value: 12 });
    ring.retire_lease(c, GpuFenceToken { queue_id: 1, value: 13 });
    assert_eq!(ring.reclaim_completed(13), 2);
    assert_eq!(ring.used_bytes(), 0);
}

#[test]
fn pinned_upload_ring_multithreaded_stress_no_overlap() {
    let ring = Arc::new(PinnedUploadRing::new_standalone(64 * 1024).expect("create 64 KiB ring"));
    let fence_counter = Arc::new(AtomicU64::new(1));
    let completed_fence = Arc::new(AtomicU64::new(0));
    let start_barrier = Arc::new(Barrier::new(4));

    let mut handles = Vec::new();
    for tid in 0u8..4 {
        let ring = Arc::clone(&ring);
        let fence_counter = Arc::clone(&fence_counter);
        let completed_fence = Arc::clone(&completed_fence);
        let start_barrier = Arc::clone(&start_barrier);

        handles.push(thread::spawn(move || {
            start_barrier.wait();
            for iter in 0u32..120 {
                let len = 512 + ((iter as usize * 137) % 3500);
                let mut lease = loop {
                    match ring.try_lease(len) {
                        Ok(l) => break l,
                        Err(VelocityIoError::StagingRingFull { .. }) => {
                            let done = completed_fence.load(Ordering::Acquire);
                            ring.reclaim_completed(done);
                            thread::yield_now();
                        }
                        Err(other) => panic!("unexpected lease error: {other}"),
                    }
                };

                let marker = tid.wrapping_mul(37).wrapping_add((iter & 0xFF) as u8);
                lease.as_mut_slice().fill(marker);
                // Verify no peer thread corrupted our uniquely leased range.
                assert!(lease.as_slice().iter().all(|&b| b == marker));

                let tok = fence_counter.fetch_add(1, Ordering::AcqRel);
                ring.retire_lease(
                    lease,
                    GpuFenceToken {
                        queue_id: 1,
                        value: tok,
                    },
                );
                completed_fence.fetch_max(tok, Ordering::Release);
                ring.reclaim_completed(completed_fence.load(Ordering::Acquire));
            }
        }));
    }

    for h in handles {
        h.join().expect("worker thread must not panic");
    }
    let final_fence = fence_counter.load(Ordering::Acquire);
    ring.reclaim_completed(final_fence);
    assert_eq!(ring.used_bytes(), 0);
}

// ---------------------------------------------------------------------------
// 5. Real Win32 OVERLAPPED + IOCP Multi-Chunk File Reads (Aligned, Unaligned, EOF)
// ---------------------------------------------------------------------------

#[cfg(windows)]
#[test]
fn win32_overlapped_iocp_aligned_unaligned_and_eof_reads() {
    use victory_gpu::graphics::velocity_io::dstorage_thunk::Win32OverlappedIo;

    let file_len = 256 * 1024 + 73;
    let expected_data = deterministic_payload(file_len, 0xA3);
    let mut tmp = tempfile::NamedTempFile::new().expect("create temp file");
    tmp.write_all(&expected_data).expect("write temp file");
    tmp.flush().expect("flush temp file");

    let io = Arc::new(Win32OverlappedIo::new().expect("create Win32OverlappedIo"));
    let file_id = io.open_read_only(tmp.path()).expect("open overlapped file");
    let ring = PinnedUploadRing::new_standalone(512 * 1024).expect("create pinned ring");

    // (op_id, offset, requested_len, expected_bytes_read)
    let cases: &[(u64, u64, usize, usize)] = &[
        (1, 0, 4096, 4096),                        // 4 KiB aligned start
        (2, 65536, 32768, 32768),                  // 64 KiB aligned mid
        (3, 13, 1023, 1023),                       // unaligned small
        (4, 12345, 7777, 7777),                    // unaligned mid
        (5, (file_len - 100) as u64, 512, 100),    // partial EOF read
        (6, (file_len + 64) as u64, 256, 0),       // strictly past EOF
        (7, 0, 0, 0),                              // zero-length read
    ];

    let mut leases = Vec::new();
    let mut ops = Vec::new();
    for &(op_id, offset, req_len, _) in cases {
        let lease = ring.try_lease(req_len).expect("lease ring slice");
        ops.push(AsyncReadOp {
            op_id,
            file: file_id,
            file_offset: offset,
            len: req_len,
            dst_ptr: lease.raw_ptr(),
        });
        leases.push((op_id, offset, lease));
    }

    let submitted = io.submit_read_batch(&ops).expect("submit IOCP batch");
    assert_eq!(submitted, cases.len());

    let mut completions: Vec<AsyncReadCompletion> = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while completions.len() < cases.len() && std::time::Instant::now() < deadline {
        io.poll_completions(&mut completions, Duration::from_millis(50))
            .expect("poll IOCP completions");
    }
    assert_eq!(completions.len(), cases.len(), "all IOCP ops must complete");

    for &(op_id, offset, _, expected_read) in cases {
        let comp = completions
            .iter()
            .find(|c| c.op_id == op_id)
            .expect("completion present");
        assert_eq!(comp.os_error, None, "op {op_id} unexpected OS error");
        assert_eq!(
            comp.bytes_read, expected_read,
            "op {op_id} bytes_read mismatch"
        );
        let (_, _, lease) = leases.iter().find(|(id, _, _)| *id == op_id).unwrap();
        if expected_read > 0 {
            let off = offset as usize;
            assert_eq!(
                &lease.as_slice()[..expected_read],
                &expected_data[off..off + expected_read],
                "op {op_id} payload mismatch"
            );
        }
    }

    // End-to-end VelocityQueue test over Win32OverlappedIo including EOF error propagation.
    let compute = Arc::new(MockComputeQueueBackend::new(1));
    let decompress = Arc::new(GpuDecompressQueue::new(compute));
    let factory = VelocityFactory::new(Arc::clone(&io) as Arc<dyn AsyncIoBackend>, decompress);
    let vfile = factory.open_file(tmp.path()).expect("open via factory");
    let status = factory.create_status_array(2);
    let queue = factory
        .create_queue(VelocityQueueConfig {
            capacity: 16,
            priority: DStoragePriority::Normal,
            staging_ring_bytes: 256 * 1024,
            decompress_workers: 2,
            name: String::from("win32-iocp-e2e"),
        })
        .expect("create queue");

    let mut dst_ok = vec![0u8; 8192];
    let mut dst_eof = vec![0u8; 512];

    queue
        .enqueue_request(DStorageRequestDesc {
            codec: DecompressionCodec::Uncompressed,
            secondary_codec: None,
            intermediate_size: 0,
            source: DStorageSource::File {
                file: vfile.handle_id(),
                offset: 12345,
                size: 8192,
            },
            destination: DStorageDestination::Memory {
                address: dst_ok.as_mut_ptr() as usize,
                size: 8192,
            },
            cancellation_tag: 1,
        })
        .unwrap();
    queue.enqueue_status(Arc::clone(&status), 0).unwrap();

    // Request crossing EOF must fail with HRESULT_HANDLE_EOF.
    queue
        .enqueue_request(DStorageRequestDesc {
            codec: DecompressionCodec::Uncompressed,
            secondary_codec: None,
            intermediate_size: 0,
            source: DStorageSource::File {
                file: vfile.handle_id(),
                offset: (file_len - 32) as u64,
                size: 512,
            },
            destination: DStorageDestination::Memory {
                address: dst_eof.as_mut_ptr() as usize,
                size: 512,
            },
            cancellation_tag: 2,
        })
        .unwrap();
    queue.enqueue_status(Arc::clone(&status), 1).unwrap();

    queue.submit().unwrap();
    assert!(queue.wait_idle(Duration::from_secs(5)));
    assert_eq!(status.hresult(0), S_OK);
    assert_eq!(dst_ok, expected_data[12345..12345 + 8192]);
    assert_eq!(status.hresult(1), HRESULT_HANDLE_EOF);

    io.close_file(file_id);
}

// ---------------------------------------------------------------------------
// 6. Zlib, LZ4 (Block + Frame), and Multi-Tile GDeflate v1 Round-Trips
// ---------------------------------------------------------------------------

#[test]
fn zlib_lz4_and_gdeflate_v1_roundtrips_and_end_to_end_queue() {
    let payload = deterministic_payload(200 * 1024 + 513, 0x3C);

    // Zlib
    let zlib_bytes = zlib_compress(&payload);
    // LZ4 raw block
    let lz4_block_bytes = lz4_flex::block::compress(&payload);
    // LZ4 frame (with 0x184D2204 magic)
    let mut lz4_frame_encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
    lz4_frame_encoder.write_all(&payload).unwrap();
    let lz4_frame_bytes = lz4_frame_encoder.finish().unwrap();
    // Multi-tile GDeflate v1 (4 tiles: 64K + 64K + 64K + remaining ~8.5K)
    let gdeflate_bytes = encode_gdeflate_v1(&payload).expect("encode gdeflate v1");
    let header = GDeflateHeader::parse(&gdeflate_bytes).expect("parse gdeflate header");
    assert_eq!(header.num_tiles, 4);

    let compute = Arc::new(MockComputeQueueBackend::new(1));
    let queue = Arc::new(GpuDecompressQueue::new(compute));

    for (codec, compressed) in [
        (DecompressionCodec::Zlib, zlib_bytes.as_slice()),
        (DecompressionCodec::Lz4, lz4_block_bytes.as_slice()),
        (DecompressionCodec::Lz4, lz4_frame_bytes.as_slice()),
        (DecompressionCodec::GDeflate, gdeflate_bytes.as_slice()),
    ] {
        let out = queue
            .dispatch(&DecompressionJob {
                job_id: 1,
                codec,
                src_bytes: compressed,
                src_gpu_va: None,
                src_gpu_resource: None,
                expected_output_len: payload.len(),
                dst_gpu_resource: None,
                wait_copy_fence: None,
            })
            .unwrap_or_else(|e| panic!("roundtrip failed for {codec:?}: {e}"));
        assert_eq!(out.bytes_written, payload.len());
        assert_eq!(out.cpu_bytes.as_deref(), Some(payload.as_slice()));
    }

    // Empty Zlib stream handling
    let empty_zlib = zlib_compress(&[]);
    let empty_out = queue
        .dispatch(&DecompressionJob {
            job_id: 2,
            codec: DecompressionCodec::Zlib,
            src_bytes: &empty_zlib,
            src_gpu_va: None,
            src_gpu_resource: None,
            expected_output_len: 0,
            dst_gpu_resource: None,
            wait_copy_fence: None,
        })
        .expect("empty zlib stream");
    assert_eq!(empty_out.bytes_written, 0);

    // Invalid GDeflate magic rejection
    assert!(matches!(
        queue.dispatch(&DecompressionJob {
            job_id: 3,
            codec: DecompressionCodec::GDeflate,
            src_bytes: &[0xFF, 0x00, 0x01, 0x00, 0, 0, 0, 0],
            src_gpu_va: None,
            src_gpu_resource: None,
            expected_output_len: 64,
            dst_gpu_resource: None,
            wait_copy_fence: None,
        }),
        Err(VelocityIoError::DecompressionFailed {
            codec: DecompressionCodec::GDeflate,
            ..
        })
    ));

    // Full VelocityQueue pipeline test with Zlib + LZ4 + GDeflate files
    let io = Arc::new(InMemoryAsyncIo::new());
    let f_zlib = io.register_open_file("z.bin", zlib_bytes.clone());
    let f_lz4 = io.register_open_file("l.bin", lz4_frame_bytes.clone());
    let f_gd = io.register_open_file("g.bin", gdeflate_bytes.clone());

    let factory = VelocityFactory::new(io, queue);
    let status = factory.create_status_array(1);
    let vq = factory
        .create_queue(VelocityQueueConfig {
            capacity: 16,
            priority: DStoragePriority::Normal,
            staging_ring_bytes: 2 * 1024 * 1024,
            decompress_workers: 3,
            name: String::from("codec-e2e"),
        })
        .unwrap();

    let mut out_z = vec![0u8; payload.len()];
    let mut out_l = vec![0u8; payload.len()];
    let mut out_g = vec![0u8; payload.len()];

    for (codec, file, src_len, dst) in [
        (DecompressionCodec::Zlib, f_zlib, zlib_bytes.len(), &mut out_z),
        (DecompressionCodec::Lz4, f_lz4, lz4_frame_bytes.len(), &mut out_l),
        (
            DecompressionCodec::GDeflate,
            f_gd,
            gdeflate_bytes.len(),
            &mut out_g,
        ),
    ] {
        vq.enqueue_request(DStorageRequestDesc {
            codec,
            secondary_codec: None,
            intermediate_size: 0,
            source: DStorageSource::File {
                file,
                offset: 0,
                size: src_len as u32,
            },
            destination: DStorageDestination::Memory {
                address: dst.as_mut_ptr() as usize,
                size: payload.len() as u32,
            },
            cancellation_tag: 0,
        })
        .unwrap();
    }
    vq.enqueue_status(Arc::clone(&status), 0).unwrap();
    vq.submit().unwrap();
    assert!(vq.wait_idle(Duration::from_secs(5)));
    assert_eq!(status.hresult(0), S_OK);
    assert_eq!(out_z, payload);
    assert_eq!(out_l, payload);
    assert_eq!(out_g, payload);
}

// ---------------------------------------------------------------------------
// 7. BcPackStubBackend Typed Error & Runtime Pluggable Override
// ---------------------------------------------------------------------------

struct TestExternalBc7Backend;

impl DecompressionBackend for TestExternalBc7Backend {
    fn name(&self) -> &'static str {
        "test-external-bc7-plugin"
    }

    fn supported_codecs(&self) -> &'static [DecompressionCodec] {
        &[DecompressionCodec::BcPack(BcPackFormat::Bc7)]
    }

    fn execution_mode(&self) -> DecompressionExecutionMode {
        DecompressionExecutionMode::CpuWorkerFallback
    }

    fn decompress(
        &self,
        job: &DecompressionJob<'_>,
        _compute: &dyn ComputeQueueBackend,
    ) -> Result<DecompressionOutput, VelocityIoError> {
        // Simple deterministic test transform (NOT BCPack bit-level logic).
        let transformed: Vec<u8> = job.src_bytes.iter().map(|b| b ^ 0x55).collect();
        let written = transformed.len();
        Ok(DecompressionOutput {
            bytes_written: written,
            completion_fence: None,
            cpu_bytes: Some(transformed),
        })
    }
}

#[test]
fn bcpack_stub_enforces_typed_error_and_supports_runtime_override() {
    let compute = Arc::new(MockComputeQueueBackend::new(1));
    let queue = Arc::new(GpuDecompressQueue::new(compute));

    for fmt in [
        BcPackFormat::Bc1,
        BcPackFormat::Bc3,
        BcPackFormat::Bc4,
        BcPackFormat::Bc5,
        BcPackFormat::Bc6h,
        BcPackFormat::Bc7,
    ] {
        let Err(err) = queue.dispatch(&DecompressionJob {
            job_id: 10,
            codec: DecompressionCodec::BcPack(fmt),
            src_bytes: &[0xAA, 0xBB, 0xCC, 0xDD],
            src_gpu_va: None,
            src_gpu_resource: None,
            expected_output_len: 4,
            dst_gpu_resource: None,
            wait_copy_fence: None,
        }) else {
            panic!("all BCPack formats must return ProprietaryCodecStub by default");
        };
        assert!(matches!(
            err,
            VelocityIoError::ProprietaryCodecStub { codec, hint }
            if codec == DecompressionCodec::BcPack(fmt) && !hint.is_empty()
        ));
    }

    // Plug in external Bc7 backend at runtime via DecompressionRegistry::register.
    queue.registry().register(Arc::new(TestExternalBc7Backend));

    let out = queue
        .dispatch(&DecompressionJob {
            job_id: 11,
            codec: DecompressionCodec::BcPack(BcPackFormat::Bc7),
            src_bytes: &[0xAA, 0xBB, 0xCC, 0xDD],
            src_gpu_va: None,
            src_gpu_resource: None,
            expected_output_len: 4,
            dst_gpu_resource: None,
            wait_copy_fence: None,
        })
        .expect("pluggable Bc7 override should succeed");
    assert_eq!(out.cpu_bytes.as_deref(), Some(&[0xFF, 0xEE, 0x99, 0x88][..]));

    // Other BCPack formats (e.g. Bc1) remain strictly stubbed.
    assert!(matches!(
        queue.dispatch(&DecompressionJob {
            job_id: 12,
            codec: DecompressionCodec::BcPack(BcPackFormat::Bc1),
            src_bytes: &[1, 2, 3, 4],
            src_gpu_va: None,
            src_gpu_resource: None,
            expected_output_len: 4,
            dst_gpu_resource: None,
            wait_copy_fence: None,
        }),
        Err(VelocityIoError::ProprietaryCodecStub {
            codec: DecompressionCodec::BcPack(BcPackFormat::Bc1),
            ..
        })
    ));

    // Also verify chained Zlib -> Bc7 succeeds with the runtime override through VelocityQueue.
    let io = Arc::new(InMemoryAsyncIo::new());
    let inner_bytes = vec![0x10, 0x20, 0x30, 0x40];
    let zipped = zlib_compress(&inner_bytes);
    let fid = io.register_open_file("chained.bin", zipped.clone());
    let factory = VelocityFactory::new(io, queue);
    let status = factory.create_status_array(1);
    let vq = factory
        .create_queue(VelocityQueueConfig {
            capacity: 8,
            staging_ring_bytes: 64 * 1024,
            decompress_workers: 1,
            ..VelocityQueueConfig::default()
        })
        .unwrap();

    let mut dst = vec![0u8; 4];
    vq.enqueue_request(DStorageRequestDesc {
        codec: DecompressionCodec::Zlib,
        secondary_codec: Some(DecompressionCodec::BcPack(BcPackFormat::Bc7)),
        intermediate_size: 4,
        source: DStorageSource::File {
            file: fid,
            offset: 0,
            size: zipped.len() as u32,
        },
        destination: DStorageDestination::Memory {
            address: dst.as_mut_ptr() as usize,
            size: 4,
        },
        cancellation_tag: 0,
    })
    .unwrap();
    vq.enqueue_status(Arc::clone(&status), 0).unwrap();
    vq.submit().unwrap();
    assert!(vq.wait_idle(Duration::from_secs(5)));
    assert_eq!(status.hresult(0), S_OK);
    assert_eq!(dst, vec![0x45, 0x75, 0x65, 0x15]);
}

// ---------------------------------------------------------------------------
// 8. Fence Wait / Signal Ordering Test
// ---------------------------------------------------------------------------

#[test]
fn compute_queue_waits_for_copy_fence_before_dispatching_work() {
    let mock = Arc::new(MockComputeQueueBackend::new(7));
    let src_id = ResourceId(100);
    let dst_id = ResourceId(200);
    let payload = deterministic_payload(4096, 0x7E);
    mock.register_buffer(src_id, payload.clone());
    mock.register_buffer(dst_id, vec![0u8; 4096]);

    let queue = GpuDecompressQueue::new(Arc::clone(&mock) as Arc<dyn ComputeQueueBackend>);
    let copy_token = GpuFenceToken {
        queue_id: 99,
        value: 42,
    };

    let output = queue
        .dispatch(&DecompressionJob {
            job_id: 500,
            codec: DecompressionCodec::Uncompressed,
            src_bytes: &[],
            src_gpu_va: None,
            src_gpu_resource: Some((src_id, 0)),
            expected_output_len: 4096,
            dst_gpu_resource: Some((dst_id, 0)),
            wait_copy_fence: Some(copy_token),
        })
        .expect("GPU passthrough copy dispatch");

    assert_eq!(output.bytes_written, 4096);
    let compute_token = output
        .completion_fence
        .expect("GPU passthrough must signal a completion fence");
    assert_eq!(compute_token.queue_id, 7);
    assert_eq!(compute_token.value, 1);
    assert_eq!(mock.completed_fence_value(), 1);
    assert_eq!(mock.read_buffer(dst_id).as_deref(), Some(payload.as_slice()));

    let events = mock.recorded_events();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0], MockComputeEvent::WaitFence(copy_token));
    assert_eq!(
        events[1],
        MockComputeEvent::PassthroughCopy {
            src: src_id,
            src_offset: 0,
            dst: dst_id,
            dst_offset: 0,
            byte_len: 4096,
            signal_token: compute_token,
        }
    );
}

// ---------------------------------------------------------------------------
// 9. Real Windows D3D12 / WARP Async Compute Queue + Copy Queue Fence Sync Round-Trip
// ---------------------------------------------------------------------------

#[cfg(windows)]
#[test]
fn d3d12_warp_copy_and_async_compute_fence_sync_roundtrip() {
    use victory_gpu::graphics::contracts::GpuVirtualAddress;
    use victory_gpu::graphics::velocity_io::gpu_decompress_queue::d3d12_compute::{
        D3D12ComputeQueueBackend, PIPELINE_PASSTHROUGH_COPY, try_create_test_device,
    };
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::core::Interface;

    let Some(device) = try_create_test_device() else {
        eprintln!("skipping d3d12_warp_copy_and_async_compute_fence_sync_roundtrip: no D3D12 adapter available");
        return;
    };

    const TOTAL_BYTES: usize = 16 * 1024;
    const HALF_BYTES: usize = TOTAL_BYTES / 2;
    let expected = deterministic_payload(TOTAL_BYTES, 0x9B);

    let create_buffer = |heap_type: D3D12_HEAP_TYPE,
                         flags: D3D12_RESOURCE_FLAGS,
                         initial_state: D3D12_RESOURCE_STATES|
     -> ID3D12Resource {
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: Valid D3D12 buffer creation parameters on an active `ID3D12Device`.
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap_type,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Alignment: 0,
                        Width: TOTAL_BYTES as u64,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: flags,
                    },
                    initial_state,
                    None,
                    &mut res,
                )
                .expect("CreateCommittedResource");
        }
        res.expect("buffer created")
    };

    let upload_buf = create_buffer(
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let gpu_src_buf = create_buffer(
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let gpu_dst_buf = create_buffer(
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let readback_buf = create_buffer(
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );

    // Populate upload buffer with deterministic test pattern.
    // SAFETY: `upload_buf` is a host-visible `D3D12_HEAP_TYPE_UPLOAD` buffer of `TOTAL_BYTES`.
    unsafe {
        let mut mapped: *mut c_void = std::ptr::null_mut();
        upload_buf
            .Map(0, None, Some(&mut mapped))
            .expect("Map upload_buf");
        std::ptr::copy_nonoverlapping(expected.as_ptr(), mapped.cast::<u8>(), TOTAL_BYTES);
        upload_buf.Unmap(0, None);
    }

    // Create a real Copy Queue (`D3D12_COMMAND_LIST_TYPE_COPY`) + fence to simulate Subsystem A staging.
    const COPY_QUEUE_ID: u32 = 10;
    const COMPUTE_QUEUE_ID: u32 = 20;
    // SAFETY: Standard D3D12 Copy Queue, allocator, command list, and fence creation & submission.
    let (copy_fence, copy_token, _copy_queue, _copy_alloc, _copy_list) = unsafe {
        let copy_queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                ..Default::default()
            })
            .expect("CreateCommandQueue(COPY)");
        let copy_fence: ID3D12Fence = device
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .expect("CreateFence(COPY)");
        let copy_alloc: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)
            .expect("CreateCommandAllocator(COPY)");
        let copy_list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &copy_alloc, None)
            .expect("CreateCommandList(COPY)");
        copy_list.CopyBufferRegion(&gpu_src_buf, 0, &upload_buf, 0, TOTAL_BYTES as u64);
        copy_list.Close().expect("Close copy list");
        let raw_list: ID3D12CommandList = copy_list.cast().expect("cast copy list");
        copy_queue.ExecuteCommandLists(&[Some(raw_list)]);
        copy_queue.Signal(&copy_fence, 1).expect("Signal copy fence");
        (
            copy_fence,
            GpuFenceToken {
                queue_id: COPY_QUEUE_ID,
                value: 1,
            },
            copy_queue,
            copy_alloc,
            copy_list,
        )
    };

    // Build real Async Compute Queue (`D3D12ComputeQueueBackend`) and wire Copy Queue's fence.
    let compute = Arc::new(
        D3D12ComputeQueueBackend::new(&device, COMPUTE_QUEUE_ID)
            .expect("create D3D12ComputeQueueBackend"),
    );
    compute.register_external_fence(COPY_QUEUE_ID, copy_fence);

    // SAFETY: `gpu_src_buf` and `gpu_dst_buf` are valid committed D3D12 buffer resources.
    let (src_va, dst_va) = unsafe {
        (
            GpuVirtualAddress(gpu_src_buf.GetGPUVirtualAddress()),
            GpuVirtualAddress(gpu_dst_buf.GetGPUVirtualAddress()),
        )
    };
    let src_id = compute.register_resource(gpu_src_buf.clone());
    let dst_id = compute.register_resource(gpu_dst_buf.clone());
    let readback_id = compute.register_resource(readback_buf.clone());

    let decompress_queue =
        GpuDecompressQueue::new(Arc::clone(&compute) as Arc<dyn ComputeQueueBackend>);

    // Half 1 (bytes 0..8192): GpuDecompressQueue waits on `copy_token` and runs passthrough copy.
    let out1 = decompress_queue
        .dispatch(&DecompressionJob {
            job_id: 1,
            codec: DecompressionCodec::Uncompressed,
            src_bytes: &[],
            src_gpu_va: Some(src_va),
            src_gpu_resource: Some((src_id, 0)),
            expected_output_len: HALF_BYTES,
            dst_gpu_resource: Some((dst_id, 0)),
            wait_copy_fence: Some(copy_token),
        })
        .expect("dispatch passthrough copy on real D3D12 compute queue");
    assert!(out1.completion_fence.is_some());

    // Half 2 (bytes 8192..16384): Dispatch real compiled `cs_5_1` HLSL passthrough copy compute shader!
    let dword_count = (HALF_BYTES / 4) as u32;
    let thread_groups = dword_count.div_ceil(64);
    let cs_token = compute
        .dispatch_compute_shader(
            PIPELINE_PASSTHROUGH_COPY,
            src_va,
            dst_va,
            &[HALF_BYTES as u32, HALF_BYTES as u32, dword_count, 0],
            [thread_groups, 1, 1],
        )
        .expect("dispatch HLSL cs_5_1 compute shader");
    assert!(cs_token.value > out1.completion_fence.unwrap().value);

    // Copy full 16 KiB `gpu_dst_buf` -> `readback_buf` on the compute queue and wait on CPU.
    let final_token = compute
        .dispatch_passthrough_copy(dst_id, 0, readback_id, 0, TOTAL_BYTES as u64)
        .expect("readback copy");
    assert!(
        compute
            .wait_cpu(final_token, 5_000)
            .expect("wait_cpu on compute fence"),
        "compute queue fence must complete within 5s"
    );
    assert!(compute.completed_fence_value() >= final_token.value);

    // Verify all 16 KiB (both CopyBufferRegion half + HLSL cs_5_1 shader half) match byte-for-byte!
    let mut readback = vec![0u8; TOTAL_BYTES];
    // SAFETY: GPU has completed `final_token`, so `readback_buf` is safe to map and read.
    unsafe {
        let mut mapped: *mut c_void = std::ptr::null_mut();
        readback_buf
            .Map(0, None, Some(&mut mapped))
            .expect("Map readback_buf");
        std::ptr::copy_nonoverlapping(mapped.cast::<u8>(), readback.as_mut_ptr(), TOTAL_BYTES);
        readback_buf.Unmap(0, None);
    }

    assert_eq!(
        &readback[..HALF_BYTES],
        &expected[..HALF_BYTES],
        "first 8 KiB (Copy Queue -> Compute passthrough) mismatch"
    );
    assert_eq!(
        &readback[HALF_BYTES..],
        &expected[HALF_BYTES..],
        "second 8 KiB (Copy Queue -> HLSL cs_5_1 compute shader) mismatch"
    );
}

// ---------------------------------------------------------------------------
// 11. Multi-Queue Shared IOCP Routing & Empty-Ring Wrap Regression Tests
// ---------------------------------------------------------------------------

#[test]
fn multi_queue_factory_routes_completions_without_stealing_or_colliding() {
    #[cfg(windows)]
    let (io, temp_dir): (Arc<dyn AsyncIoBackend>, Option<tempfile::TempDir>) = {
        use victory_gpu::graphics::velocity_io::Win32OverlappedIo;
        let dir = tempfile::tempdir().expect("create tempdir");
        let io = Arc::new(Win32OverlappedIo::new().expect("create Win32OverlappedIo"));
        (io, Some(dir))
    };
    #[cfg(not(windows))]
    let (io, temp_dir): (Arc<dyn AsyncIoBackend>, Option<()>) = {
        let mem = Arc::new(InMemoryAsyncIo::new());
        (mem, None)
    };

    let compute = Arc::new(MockComputeQueueBackend::new(1));
    let decompress = Arc::new(GpuDecompressQueue::new(compute));
    let factory = VelocityFactory::new(Arc::clone(&io), decompress);

    const NUM_QUEUES: usize = 16;
    const CHUNK_BYTES: usize = 4096;
    const CHUNKS_PER_QUEUE: usize = 8;

    let mut files = Vec::with_capacity(NUM_QUEUES);
    let mut expected_blobs = Vec::with_capacity(NUM_QUEUES);
    for q in 0..NUM_QUEUES {
        let blob = deterministic_payload(CHUNK_BYTES * CHUNKS_PER_QUEUE, (q as u8).wrapping_mul(13));
        #[cfg(windows)]
        let file = {
            let path = temp_dir
                .as_ref()
                .unwrap()
                .path()
                .join(format!("archive_{q}.bin"));
            std::fs::write(&path, &blob).expect("write temp archive");
            factory.open_file(&path).expect("open temp archive")
        };
        #[cfg(not(windows))]
        let file = {
            let _ = &temp_dir;
            let name = format!("archive_{q}.bin");
            let mem = Arc::new(InMemoryAsyncIo::new());
            mem.add_file(&name, blob.clone());
            factory.open_file(Path::new(&name)).expect("open mem archive")
        };
        files.push(file);
        expected_blobs.push(blob);
    }

    // Create 16 queues on the SAME factory (matching GTA VI's 17 fiDeviceDirectStorage queues)
    // all sharing one IOCP handle and one shared staging ring (`staging_ring_bytes: 0`).
    let mut queues = Vec::with_capacity(NUM_QUEUES);
    let mut statuses = Vec::with_capacity(NUM_QUEUES);
    let mut destinations: Vec<Vec<Vec<u8>>> = Vec::with_capacity(NUM_QUEUES);

    for q in 0..NUM_QUEUES {
        let queue = factory
            .create_queue(VelocityQueueConfig {
                capacity: 64,
                priority: DStoragePriority::Realtime,
                staging_ring_bytes: 0,
                decompress_workers: 2,
                name: format!("multi-q-{q}"),
            })
            .expect("create factory queue");
        let status = factory.create_status_array(1);
        let mut q_dsts = vec![vec![0xAAu8; CHUNK_BYTES]; CHUNKS_PER_QUEUE];
        for (c, dst) in q_dsts.iter_mut().enumerate() {
            queue
                .enqueue_request(DStorageRequestDesc {
                    codec: DecompressionCodec::Uncompressed,
                    secondary_codec: None,
                    intermediate_size: 0,
                    source: DStorageSource::File {
                        file: files[q].handle_id(),
                        offset: (c * CHUNK_BYTES) as u64,
                        size: CHUNK_BYTES as u32,
                    },
                    destination: DStorageDestination::Memory {
                        address: dst.as_mut_ptr() as usize,
                        size: CHUNK_BYTES as u32,
                    },
                    cancellation_tag: ((q as u64) << 16) | (c as u64),
                })
                .expect("enqueue");
        }
        queue
            .enqueue_status(Arc::clone(&status), 0)
            .expect("enqueue status");
        queues.push(queue);
        statuses.push(status);
        destinations.push(q_dsts);
    }

    // Submit all 16 queues simultaneously to maximally stress shared IOCP completion routing.
    for queue in &queues {
        queue.submit().expect("submit");
    }

    for (q, (queue, status)) in queues.iter().zip(statuses.iter()).enumerate() {
        assert!(
            queue.wait_idle(Duration::from_secs(10)),
            "queue {q} deadlocked waiting for shared IOCP completions"
        );
        assert!(status.is_complete(0), "queue {q} status not complete");
        assert_eq!(status.hresult(0), S_OK, "queue {q} status error");
        for (c, dst) in destinations[q].iter().enumerate() {
            let start = c * CHUNK_BYTES;
            assert_eq!(
                dst.as_slice(),
                &expected_blobs[q][start..start + CHUNK_BYTES],
                "queue {q} chunk {c} data mismatch (completion crossed queues)"
            );
        }
    }
}

#[test]
fn empty_ring_past_midpoint_wraps_large_lease_without_livelock() {
    let ring = PinnedUploadRing::new_standalone(16 * 1024).expect("16 KiB ring");
    // Advance head and tail past the midpoint (10 KiB of 16 KiB) and reclaim completely.
    let first = ring.try_lease(10 * 1024).expect("initial 10 KiB lease");
    ring.retire_lease(first, GpuFenceToken { queue_id: 0, value: 0 });
    assert_eq!(ring.reclaim_completed(0), 1);
    assert_eq!(ring.used_bytes(), 0);

    // Request a 12 KiB lease (> remaining 6 KiB before wrap, <= 16 KiB capacity).
    // Previously this returned StagingRingFull forever even though the ring was empty.
    let large = ring
        .try_lease(12 * 1024)
        .expect("empty ring must wrap past padding and grant 12 KiB lease");
    assert_eq!(large.len(), 12 * 1024);
    assert_eq!(large.ring_offset(), 0);
    ring.retire_lease(large, GpuFenceToken { queue_id: 0, value: 0 });
    assert_eq!(ring.reclaim_completed(0), 1);
    assert_eq!(ring.used_bytes(), 0);
}

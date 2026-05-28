// patterns/parallel_prefault.c — multi-threaded weight prefault after mmap.
//
// WHAT: After mmap()ing a multi-GB weight file, touch one byte per 4 KB
//       page from all OMP threads. This brings every page into the OS
//       page cache AND pre-populates page tables, so the first forward
//       pass takes zero major faults.
//
// WHEN: Right after st_mmap in the weight loader, before any kernel call.
//       Pair with `madvise(MADV_HUGEPAGE)` for TLB efficiency at runtime.
//
// WHY:  MAP_POPULATE asks the kernel to do the same job synchronously
//       during the mmap() call, but it's single-threaded inside the
//       kernel. With 16 OMP threads in userspace doing the touches we get
//       a ~2x speedup on cold-cache startup (file not yet in the page
//       cache). MAP_PRIVATE + parallel touch is the win.
//
// EXPECTED SPEEDUP: Cold-cache startup roughly halves (Gemma 4 E4B 9 GB
//                   weights: 0.95 s -> 0.44 s on EPYC 7763 16 threads).
//                   Warm-cache numbers unchanged (everything already in
//                   the page cache).
//
// Original commit: see `src-cpu: parallel weight prefault (faster than
// MAP_POPULATE)` in the Gemma 4 E4B `cpu` branch.

#include <stdint.h>
#include <stddef.h>
#include <sys/mman.h>

// Call once per mmap'd weight shard right after iofile_mmap_ro / mmap.
static inline void prefault_weights_parallel(const void* data, size_t bytes) {
    // Hint: we'll touch all of this sequentially, then re-touch it for the
    // life of the process — prefer huge pages.
    madvise((void*)data, bytes, MADV_WILLNEED);
    madvise((void*)data, bytes, MADV_HUGEPAGE);

    // Touch one byte per 4 KB page. We use a reduction to keep the load
    // alive (avoid the compiler dropping the read). The exact page size is
    // 4 KB on x86_64/aarch64; if the kernel collapses to 2 MB huge pages
    // we still touch 512x more often than strictly required — costs us a
    // handful of milliseconds, no harm done.
    const size_t PAGE = 4096;
    volatile uint8_t sink = 0;
    const uint8_t* p = (const uint8_t*)data;
    #pragma omp parallel for schedule(static) reduction(+:sink)
    for (size_t off = 0; off < bytes; off += PAGE) {
        sink += p[off];
    }
    (void)sink;
}

// NOTE: MAP_POPULATE alone (without parallel prefault) is ~2x slower
// because the kernel populates pages from a single thread. Use MAP_PRIVATE
// + parallel prefault.
//
// NOTE: madvise(..., MADV_HUGEPAGE) is a hint — system-wide THP must be
// enabled (`cat /sys/kernel/mm/transparent_hugepage/enabled` should
// include "[always]" or "[madvise]"). If THP is disabled the hint is a
// no-op and you lose the TLB win (but correctness is unaffected).

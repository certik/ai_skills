// load_parallel_pread.c
//
// WHAT: Parallel safetensors weight loader. One pread() worker per shard
//       reads arrays (tensors) directly into preallocated MTLBuffers.
//
// WHEN: From the very first port. Single-threaded mmap+memcpy caps at
//       ~6 GB/s on macOS due to page-fault serialization (see GOTCHAS
//       #28, #29).
//
// EXPECTED SPEEDUP: ~2.5–3× over mmap+memcpy. Two confirmed data points:
//   - Qwen3.6-35B-A3B (8 shards × ~4 GB, M4 Max): 10 s → 3.5 s startup.
//   - Dream-7B        (4 shards × ~3.5 GB, M4 Max): 1.71 s → 0.82 s
//     weights; total wall 3.08 s → 2.17 s, beating MLX's 2.67 s by
//     19% on a short fastdllm bench (BL=32, max_tokens=32).
// In both cases, the resulting startup is below MLX's
// ParallelFileReader on the same machine.
//
// REQUIREMENTS:
//   - Your safetensors header has a list of (tensor_name, dtype, shape,
//     file_offset, nbytes, shard_idx) entries. The file_offset is the
//     byte offset within the shard file of the array data.
//   - Per-array MTLBuffer (no shard-wide zero-copy attempts — arrays
//     are typically not 16KB-page-aligned within the shard file).
//   - macOS / libdispatch (built into Xcode SDK).
//
// PITFALLS:
//   - Do NOT call madvise(MADV_WILLNEED) on macOS — it blocks
//     synchronously paging in the entire file. Just skip it.
//   - Do NOT try to mmap+memcpy with dispatch_apply over arrays;
//     macOS page-fault handler serializes on a VM lock at ~1 GB/s
//     per thread, and adding more workers does not help.
//   - Buffer ALLOCATION is not guaranteed thread-safe in Metal.
//     Do the alloc pass serially, then dispatch_apply the copy.
//   - Do NOT try `newBufferWithBytesNoCopy:` per tensor as a "skip
//     the copy entirely" shortcut. Tensor offsets within a
//     safetensors shard are 8-byte aligned, NOT the 16 KB page-
//     aligned that Apple's nocopy API requires. The whole-shard
//     wrap+offset workaround exists but is invasive; parallel pread
//     is already fast enough. See gotcha #40.
//
// Reference: this is what MLX's mlx/io/load.cpp does
// (ParallelFileReader with ThreadPool{4} + batch_size 32 MB).
// We use one fd per shard which is even simpler.

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

void load_weights(st_archive* arch, gpu_ctx* ctx, gpu_buf** out_bufs) {
    size_t n_t  = st_count(arch);
    size_t n_sh = st_n_shards(arch);

    // Pass 1: serial allocation.
    for (size_t i = 0; i < n_t; i++) {
        const st_tensor* t = st_at(arch, i);
        out_bufs[i] = gpu_buf_new(ctx, t->nbytes);
    }

    // Pre-bucket arrays per shard. Preserving array order keeps the
    // per-shard reads sequential on disk (data_offsets are monotone).
    size_t** sh_idx = calloc(n_sh, sizeof(size_t*));
    size_t*  sh_n   = calloc(n_sh, sizeof(size_t));
    for (size_t i = 0; i < n_t; i++) sh_n[st_at(arch, i)->shard_idx]++;
    for (size_t s = 0; s < n_sh; s++) sh_idx[s] = malloc(sh_n[s] * sizeof(size_t));
    size_t* sh_w = calloc(n_sh, sizeof(size_t));
    for (size_t i = 0; i < n_t; i++) {
        int s = st_at(arch, i)->shard_idx;
        sh_idx[s][sh_w[s]++] = i;
    }
    free(sh_w);

    // Pass 2: parallel pread. dispatch_apply schedules across CPU
    // cores; with 8 shards on an 8+ core machine we get one in-flight
    // SSD read per shard, which on M4 Max saturates at ~10 GB/s.
    dispatch_apply(n_sh, DISPATCH_APPLY_AUTO, ^(size_t s) {
        const char* path = st_shard_path(arch, s);
        int fd = open(path, O_RDONLY);
        if (fd < 0) return;
        for (size_t k = 0; k < sh_n[s]; k++) {
            size_t i = sh_idx[s][k];
            const st_tensor* t = st_at(arch, i);
            size_t off = st_tensor_offset(arch, t);
            char* dst = (char*)gpu_buf_contents(out_bufs[i]);
            size_t left = t->nbytes;
            while (left > 0) {
                ssize_t r = pread(fd, dst, left, off);
                if (r <= 0) break;
                dst  += r;
                off  += r;
                left -= (size_t)r;
            }
        }
        close(fd);
    });

    for (size_t s = 0; s < n_sh; s++) free(sh_idx[s]);
    free(sh_idx); free(sh_n);
}

// =============================================================
// VARIANT 1: sub-shard parallel pread (SHARD_SPLIT > 1)
// =============================================================
//
// On M-Max-class chips, dispatch_apply(n_shards=4) only spawns 4
// pread workers, leaving 8 P-cores idle. Each shard's single pread
// stream tops out at ~30 GB/s (kernel readahead cap, not LPDDR5x
// peak). Splitting the *tensor list* of each shard into K contiguous
// halves and running K workers per fd preserves sequential reads
// (so readahead still engages) while running K× more streams.
//
// IMPORTANT — co-tune SHARD_SPLIT against bg compile + residency_async:
//   target n_shards * SHARD_SPLIT == P_cores
// On M4 Max (12 P-cores, 4 shards): SHARD_SPLIT=3 is optimal.
// SHARD_SPLIT=4 (16 threads) regresses because bg Metal compile
// (gpu_init_async) gets starved and lib_join spikes to 30-74 ms.
// See gotcha #45 for the full table.

#define SHARD_SPLIT 3   // = P_cores / n_shards, M4 Max with 4 shards

void load_weights_sub_shard(st_archive* arch, gpu_ctx* ctx, gpu_buf** out_bufs) {
    size_t n_t  = st_count(arch);
    size_t n_sh = st_n_shards(arch);
    // ... allocation + bucketing same as above (sh_idx, sh_n) ...

    dispatch_apply(n_sh * SHARD_SPLIT, DISPATCH_APPLY_AUTO, ^(size_t job) {
        size_t s = job / SHARD_SPLIT;
        size_t k = job % SHARD_SPLIT;
        int fd = open(st_shard_path(arch, s), O_RDONLY);
        if (fd < 0) return;

        // Contiguous slice of this shard's tensor list. Each worker
        // reads its slice left-to-right so the kernel readahead
        // engages. Boundaries are integer division so the union
        // covers the whole shard with no overlap.
        size_t i0 =  k    * sh_n[s] / SHARD_SPLIT;
        size_t i1 = (k+1) * sh_n[s] / SHARD_SPLIT;

        for (size_t kk = i0; kk < i1; kk++) {
            size_t i = sh_idx[s][kk];
            const st_tensor* t = st_at(arch, i);
            size_t off = st_tensor_offset(arch, t);
            char* dst = (char*)gpu_buf_contents(out_bufs[i]);
            size_t left = t->nbytes;
            while (left > 0) {
                ssize_t r = pread(fd, dst, left, off);
                if (r <= 0) break;
                dst  += r;
                off  += r;
                left -= (size_t)r;
            }
        }
        close(fd);
    });
}

// Measurement (Dream-7B, 4 shards, M4 Max, fastdllm prompt):
//   SHARD_SPLIT=1  pread=472 ms  total=2.19 s  (4 threads)
//   SHARD_SPLIT=3  pread=340 ms  total=2.05 s  (12 threads, == P-cores)
//   SHARD_SPLIT=4  pread=332 ms  total=2.15 s  (16 threads, REGRESSION
//                                                — bg compile starves)
//
// =============================================================
// VARIANT 2: shard-sized MTLBuffer + per-tensor views
// =============================================================
//
// Replace the per-tensor MTLBuffer allocation in Pass 1 with one
// MTLBuffer per safetensors shard. Each "tensor" is a wrapper
// (parent_buf, offset, size). Residency set shrinks from N_tensors
// to N_shards (e.g., 339 → 4 on Dream-7B). Architecturally cleaner;
// see gotcha #50.
//
// gpu_buf:
//   struct gpu_buf {
//       id<MTLBuffer> mtl;      // shared parent (one per shard)
//       size_t        offset;   // byte offset into parent
//       size_t        size;
//       void*         host;     // = [parent contents] + offset
//   };
//
// In gpu_cmdbuf_dispatch, pass parent->offset + per_arg_off to
// setBuffer:offset:.

void load_weights_shard_buf(st_archive* arch, gpu_ctx* ctx, gpu_buf** out_bufs) {
    size_t n_sh = st_n_shards(arch);
    gpu_buf** shard_bufs = calloc(n_sh, sizeof(gpu_buf*));

    // Pass 1: one MTLBuffer per shard, size = sum of tensor nbytes.
    for (size_t s = 0; s < n_sh; s++) {
        size_t shard_bytes = 0;
        for (size_t i = 0; i < st_count(arch); i++) {
            if (st_at(arch, i)->shard_idx == s) shard_bytes += st_at(arch, i)->nbytes;
        }
        shard_bufs[s] = gpu_buf_new(ctx, shard_bytes);
    }

    // Pass 2: per-tensor views into the parent shard buffer.
    // Use a per-shard write cursor to maintain sequential layout.
    size_t* cur = calloc(n_sh, sizeof(size_t));
    for (size_t i = 0; i < st_count(arch); i++) {
        const st_tensor* t = st_at(arch, i);
        out_bufs[i] = gpu_buf_new_view(shard_bufs[t->shard_idx],
                                        cur[t->shard_idx],
                                        t->nbytes);
        cur[t->shard_idx] += t->nbytes;
    }
    free(cur);

    // Pass 3: parallel pread (with optional SHARD_SPLIT sub-sharding
    // from VARIANT 1) — writes go directly into the shard buffer at
    // the view's offset (gpu_buf_contents resolves to parent->host
    // + view->offset).
    //
    // ... same dispatch_apply loop as above, using out_bufs[i] ...

    // Register only the parent shard buffers with the residency set,
    // not the views (gotcha #30).
    gpu_residency_async(ctx, shard_bufs, n_sh);
}

// Combined real-world numbers (Dream-7B, M4 Max, fastdllm validation
// prompt, with the full companion stack: async compile, async
// residency, dropped madvise, SHARD_SPLIT=3, shard-buffer-views):
//   weights  340 ms (parallel pread)
//   metal     17 ms (lib_join = 0 on warm)
//   residency  0 ms (hidden in pread tail)
//   startup  398 ms total
//   gen     1.03 s
//   wall    1.43 s   ← MLX is 1.57 s on the same prompt
//
// References: gotchas #28 (madvise), #29 (mmap+memcpy), #30 (residency),
// #40 (per-tensor nocopy fails), #45 (SHARD_SPLIT tuning), #46
// (async compile QoS), #47 (per-shard nocopy slower than pread),
// #50 (shard-buf + views).

// load_parallel_pread.c
//
// WHAT: Parallel safetensors weight loader. One pread() worker per shard
//       reads tensors directly into preallocated MTLBuffers.
//
// WHEN: From the very first port. Single-threaded mmap+memcpy caps at
//       ~6 GB/s on macOS due to page-fault serialization (see GOTCHAS
//       #28, #29).
//
// EXPECTED SPEEDUP: ~2.5–3× over mmap+memcpy. For a 35 GB / 8-shard
//       model on M4 Max, 10s → 3.5s startup, beating MLX's
//       ParallelFileReader by a small margin.
//
// REQUIREMENTS:
//   - Your safetensors header has a list of (tensor_name, dtype, shape,
//     file_offset, nbytes, shard_idx) entries. The file_offset is the
//     byte offset within the shard file of the tensor data.
//   - Per-tensor MTLBuffer (no shard-wide zero-copy attempts — tensors
//     are typically not 16KB-page-aligned within the shard file).
//   - macOS / libdispatch (built into Xcode SDK).
//
// PITFALLS:
//   - Do NOT call madvise(MADV_WILLNEED) on macOS — it blocks
//     synchronously paging in the entire file. Just skip it.
//   - Do NOT try to mmap+memcpy with dispatch_apply over tensors;
//     macOS page-fault handler serializes on a VM lock at ~1 GB/s
//     per thread, and adding more workers does not help.
//   - Buffer ALLOCATION is not guaranteed thread-safe in Metal.
//     Do the alloc pass serially, then dispatch_apply the copy.
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

    // Pre-bucket tensors per shard. Preserving array order keeps the
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

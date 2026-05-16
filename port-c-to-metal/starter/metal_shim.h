// Phase 0 metal shim: C-callable wrapper over a tiny Metal API surface.
// We use runtime MSL compilation (no offline metal compiler / Xcode required).

#ifndef METAL_SHIM_H
#define METAL_SHIM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_ctx gpu_ctx;
typedef struct gpu_buf gpu_buf;
typedef struct gpu_pipeline gpu_pipeline;
typedef struct gpu_cmdbuf gpu_cmdbuf;

// Initialize Metal: pick default device, create command queue, compile the
// MSL source `msl_source` (NUL-terminated) into an MTLLibrary kept on the ctx.
// Returns NULL on failure; on failure *err is set to a malloc'd description.
gpu_ctx* gpu_init(const char* msl_source, char** err);

// Information.
const char* gpu_device_name(gpu_ctx* ctx);

// Buffers (shared CPU/GPU storage on Apple Silicon).
gpu_buf* gpu_buf_new(gpu_ctx* ctx, size_t bytes);
gpu_buf* gpu_buf_new_from(gpu_ctx* ctx, const void* src, size_t bytes);
// Zero-copy wrap of an existing host pointer (must be page-aligned, length
// page-multiple). On Apple Silicon this avoids any data movement; the caller
// owns the memory and must keep it alive until the buffer is freed.
gpu_buf* gpu_buf_wrap_nocopy(gpu_ctx* ctx, void* host_ptr, size_t bytes);
void*       gpu_buf_contents(gpu_buf* b);

// Pipelines (one per Metal kernel function name in the MSL library).
gpu_pipeline* gpu_pipeline_for(gpu_ctx* ctx, const char* fn_name, char** err);

typedef struct {
    gpu_buf* buf;
    size_t      offset;
} gpu_arg_buf;

// Batched dispatch: open a command buffer + encoder, record many dispatches,
// then commit + wait once.  Cuts per-launch overhead dramatically.
gpu_cmdbuf* gpu_cmdbuf_new(gpu_ctx* ctx);
void           gpu_cmdbuf_dispatch(gpu_cmdbuf* cb,
                                      gpu_pipeline* p,
                                      const gpu_arg_buf* buffers, size_t n_buffers,
                                      size_t grid_x, size_t grid_y, size_t grid_z,
                                      size_t tg_x,   size_t tg_y,   size_t tg_z);
bool           gpu_cmdbuf_commit_wait(gpu_cmdbuf* cb, char** err);
// Async pair: commit returns immediately (GPU starts running). Caller MUST
// later call wait() on the same cmdbuf, which blocks until completion and
// frees it. The cmdbuf is invalid for further dispatch after commit().
void           gpu_cmdbuf_commit(gpu_cmdbuf* cb);
bool           gpu_cmdbuf_wait(gpu_cmdbuf* cb, char** err);

#ifdef __cplusplus
}
#endif
#endif

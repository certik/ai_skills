// Phase 0 metal shim: C-callable wrapper over a tiny Metal API surface.
// We use runtime MSL compilation (no offline metal compiler / Xcode required).

#ifndef GPTOSS_METAL_SHIM_H
#define GPTOSS_METAL_SHIM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gptoss_ctx gptoss_ctx;
typedef struct gptoss_buf gptoss_buf;
typedef struct gptoss_pipeline gptoss_pipeline;
typedef struct gptoss_cmdbuf gptoss_cmdbuf;

// Initialize Metal: pick default device, create command queue, compile the
// MSL source `msl_source` (NUL-terminated) into an MTLLibrary kept on the ctx.
// Returns NULL on failure; on failure *err is set to a malloc'd description.
gptoss_ctx* gptoss_init(const char* msl_source, char** err);

// Information.
const char* gptoss_device_name(gptoss_ctx* ctx);

// Buffers (shared CPU/GPU storage on Apple Silicon).
gptoss_buf* gptoss_buf_new(gptoss_ctx* ctx, size_t bytes);
gptoss_buf* gptoss_buf_new_from(gptoss_ctx* ctx, const void* src, size_t bytes);
// Zero-copy wrap of an existing host pointer (must be page-aligned, length
// page-multiple). On Apple Silicon this avoids any data movement; the caller
// owns the memory and must keep it alive until the buffer is freed.
gptoss_buf* gptoss_buf_wrap_nocopy(gptoss_ctx* ctx, void* host_ptr, size_t bytes);
void*       gptoss_buf_contents(gptoss_buf* b);

// Pipelines (one per Metal kernel function name in the MSL library).
gptoss_pipeline* gptoss_pipeline_for(gptoss_ctx* ctx, const char* fn_name, char** err);

typedef struct {
    gptoss_buf* buf;
    size_t      offset;
} gptoss_arg_buf;

// Batched dispatch: open a command buffer + encoder, record many dispatches,
// then commit + wait once.  Cuts per-launch overhead dramatically.
gptoss_cmdbuf* gptoss_cmdbuf_new(gptoss_ctx* ctx);
void           gptoss_cmdbuf_dispatch(gptoss_cmdbuf* cb,
                                      gptoss_pipeline* p,
                                      const gptoss_arg_buf* buffers, size_t n_buffers,
                                      size_t grid_x, size_t grid_y, size_t grid_z,
                                      size_t tg_x,   size_t tg_y,   size_t tg_z);
bool           gptoss_cmdbuf_commit_wait(gptoss_cmdbuf* cb, char** err);
// Async pair: commit returns immediately (GPU starts running). Caller MUST
// later call wait() on the same cmdbuf, which blocks until completion and
// frees it. The cmdbuf is invalid for further dispatch after commit().
void           gptoss_cmdbuf_commit(gptoss_cmdbuf* cb);
bool           gptoss_cmdbuf_wait(gptoss_cmdbuf* cb, char** err);

#ifdef __cplusplus
}
#endif
#endif

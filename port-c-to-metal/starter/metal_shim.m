// Objective-C implementation of the metal shim. Compiled with -ObjC.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_shim.h"
#include <stdlib.h>
#include <string.h>

struct gpu_ctx {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;
    char                name[256];
};

struct gpu_buf {
    id<MTLBuffer> mtl;
};

struct gpu_pipeline {
    id<MTLComputePipelineState> pso;
    NSUInteger                  max_threads_per_group;
};

struct gpu_cmdbuf {
    id<MTLCommandBuffer>         cb;
    id<MTLComputeCommandEncoder> enc;
};

static char* dup_nserr(NSError* e) {
    if (!e) return strdup("unknown error");
    NSString* s = [e localizedDescription] ?: [e description];
    return strdup([s UTF8String]);
}

gpu_ctx* gpu_init(const char* msl_source, char** err) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { if (err) *err = strdup("no Metal device"); return NULL; }

        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!q) { if (err) *err = strdup("newCommandQueue failed"); return NULL; }

        id<MTLLibrary> lib = nil;
        if (msl_source && msl_source[0]) {
            NSString* src = [NSString stringWithUTF8String:msl_source];
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            // Match Metal 3.0 features (Apple7+ / M1+).
            opts.languageVersion = MTLLanguageVersion3_1;
            // mathMode was added in macOS 15 / iOS 18 and replaces the
            // deprecated fastMathEnabled.  Use the new API on supported
            // SDKs and fall back to the old one otherwise.
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_15_0
            opts.mathMode = MTLMathModeFast;
#else
            opts.fastMathEnabled = YES;
#endif
            NSError* e = nil;
            lib = [dev newLibraryWithSource:src options:opts error:&e];
            if (!lib) {
                if (err) *err = dup_nserr(e);
                return NULL;
            }
        }

        gpu_ctx* ctx = (gpu_ctx*)calloc(1, sizeof(*ctx));
        ctx->device  = dev;
        ctx->queue   = q;
        ctx->library = lib;
        const char* nm = [[dev name] UTF8String] ?: "unknown";
        snprintf(ctx->name, sizeof(ctx->name), "%s", nm);
        // ARC retains via __strong ivars.
        return ctx;
    }
}

const char* gpu_device_name(gpu_ctx* ctx) {
    return ctx ? ctx->name : "";
}

gpu_buf* gpu_buf_new(gpu_ctx* ctx, size_t bytes) {
    if (!ctx) return NULL;
    @autoreleasepool {
        id<MTLBuffer> b = [ctx->device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
        if (!b) return NULL;
        gpu_buf* out = (gpu_buf*)calloc(1, sizeof(*out));
        out->mtl = b;
        return out;
    }
}

gpu_buf* gpu_buf_new_from(gpu_ctx* ctx, const void* src, size_t bytes) {
    gpu_buf* b = gpu_buf_new(ctx, bytes);
    if (!b) return NULL;
    memcpy([b->mtl contents], src, bytes);
    return b;
}

gpu_buf* gpu_buf_wrap_nocopy(gpu_ctx* ctx, void* host_ptr, size_t bytes) {
    if (!ctx) return NULL;
    @autoreleasepool {
        // No deallocator: caller (the safetensors archive / mmap) owns memory.
        id<MTLBuffer> b = [ctx->device newBufferWithBytesNoCopy:host_ptr
                                                        length:bytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        if (!b) return NULL;
        gpu_buf* out = (gpu_buf*)calloc(1, sizeof(*out));
        out->mtl = b;
        return out;
    }
}

void* gpu_buf_contents(gpu_buf* b) {
    return b ? [b->mtl contents] : NULL;
}
gpu_pipeline* gpu_pipeline_for(gpu_ctx* ctx, const char* fn_name, char** err) {
    if (!ctx || !ctx->library) {
        if (err) *err = strdup("no library");
        return NULL;
    }
    @autoreleasepool {
        NSString* nm = [NSString stringWithUTF8String:fn_name];
        id<MTLFunction> fn = [ctx->library newFunctionWithName:nm];
        if (!fn) {
            if (err) {
                char buf[256];
                snprintf(buf, sizeof(buf), "function not found: %s", fn_name);
                *err = strdup(buf);
            }
            return NULL;
        }
        NSError* e = nil;
        id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&e];
        if (!pso) { if (err) *err = dup_nserr(e); return NULL; }
        gpu_pipeline* p = (gpu_pipeline*)calloc(1, sizeof(*p));
        p->pso = pso;
        p->max_threads_per_group = [pso maxTotalThreadsPerThreadgroup];
        return p;
    }
}

gpu_cmdbuf* gpu_cmdbuf_new(gpu_ctx* ctx) {
    if (!ctx) return NULL;
    gpu_cmdbuf* c = (gpu_cmdbuf*)calloc(1, sizeof(*c));
    c->cb  = [ctx->queue commandBuffer];
    c->enc = [c->cb computeCommandEncoder];
    return c;
}

void gpu_cmdbuf_dispatch(gpu_cmdbuf* c,
                            gpu_pipeline* p,
                            const gpu_arg_buf* buffers, size_t n_buffers,
                            size_t grid_x, size_t grid_y, size_t grid_z,
                            size_t tg_x,   size_t tg_y,   size_t tg_z)
{
    [c->enc setComputePipelineState:p->pso];
    for (size_t i = 0; i < n_buffers; i++) {
        [c->enc setBuffer:buffers[i].buf->mtl
                   offset:buffers[i].offset
                  atIndex:(NSUInteger)i];
    }
    [c->enc dispatchThreads:MTLSizeMake(grid_x, grid_y, grid_z)
      threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, tg_z)];
}

double g_gpu_time = 0.0;

bool gpu_cmdbuf_commit_wait(gpu_cmdbuf* c, char** err) {
    if (!c) return false;
    [c->enc endEncoding];
    [c->cb commit];
    [c->cb waitUntilCompleted];
    bool ok = (c->cb.error == nil);
    if (!ok && err) *err = dup_nserr(c->cb.error);
    extern double g_gpu_time;
    g_gpu_time += (c->cb.GPUEndTime - c->cb.GPUStartTime);
    c->enc = nil; c->cb = nil;
    free(c);
    return ok;
}

void gpu_cmdbuf_commit(gpu_cmdbuf* c) {
    if (!c) return;
    [c->enc endEncoding];
    [c->cb commit];
    c->enc = nil;
    // Keep c->cb alive for waitUntilCompleted later.
}

bool gpu_cmdbuf_wait(gpu_cmdbuf* c, char** err) {
    if (!c) return false;
    [c->cb waitUntilCompleted];
    bool ok = (c->cb.error == nil);
    if (!ok && err) *err = dup_nserr(c->cb.error);
    extern double g_gpu_time;
    g_gpu_time += (c->cb.GPUEndTime - c->cb.GPUStartTime);
    c->cb = nil;
    free(c);
    return ok;
}

// patterns/param_buf_const.c
//
// WHAT: Some "params" are call-invariant — they have the same value
//       on every forward() call (e.g., the dimensions of o_proj when
//       Lq is always 1 for decode).  Move them to a separate const ring
//       that is filled ONCE at startup.  Per-call rings only refill the
//       fields that genuinely depend on (q_off, Lq).
// WHEN: After param_buf_persistent.c.  Tiny GPU win but real CPU-encode
//       win when the 2-deep pipeline is in play.
// SPEEDUP: ~1.01× decode; CPU encode time materially lower.
// COMMIT:  8001db6 — "csrc: lift call-invariant constants out of
//                     PARAM_BUF ring (CONST_PARAM_BUF)"

#include "metal_shim.h"

// Allocated and filled once at startup.
#define CONST_PARAM_BUF(name, sz, fill_expr) \
    static gptoss_buf* name##_buf = NULL; \
    if (!name##_buf) { \
        name##_buf = gptoss_buf_new(g_ctx, (sz)); \
        void* dst = gptoss_buf_contents(name##_buf); \
        fill_expr; \
    }

// Same as PARAM_BUF but caller passes a memcpy-style filler.
#define PARAM_BUF(name, sz) \
    static gptoss_buf* name##_buf = NULL; \
    if (!name##_buf) name##_buf = gptoss_buf_new(g_ctx, (sz)); \
    void* name##_dst = gptoss_buf_contents(name##_buf)

void forward_decode(int q_off) {        // Lq = 1 always
    // Constants for decode: Lq, HIDDEN, etc. NEVER change.
    CONST_PARAM_BUF(dimsRMS, sizeof(uint32_t)*2, ({
        uint32_t v[2] = { 1u, HIDDEN }; memcpy(dst, v, sizeof v);
    }));
    CONST_PARAM_BUF(dimsQp, sizeof(uint32_t)*3, ({
        uint32_t v[3] = { 1u, HIDDEN, N_QHEADS*HEAD_DIM };
        memcpy(dst, v, sizeof v);
    }));

    // Only q_off changes per call.
    uint32_t dimsRq[4] = { 1u, N_QHEADS, HEAD_DIM, (uint32_t)q_off };
    PARAM_BUF(dimsRq, sizeof(dimsRq));
    memcpy(dimsRq_dst, dimsRq, sizeof(dimsRq));

    // ... rest of forward() uses dimsRMS_buf, dimsQp_buf (const) and
    // dimsRq_buf (per-call) as args.
}

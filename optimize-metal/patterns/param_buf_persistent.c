// patterns/param_buf_persistent.c
//
// WHAT: Pre-allocate param ("dims" / scalar) Metal buffers once and
//       refill in place each forward() call.  Avoids per-step allocation
//       and the implicit synchronization that comes with it.
// WHEN: After batch-dispatches.  Easy 1–2% decode win.
// CAVEAT: forward() MUST end with commit+wait so the GPU is idle before
//         the next call refills the same buffers.  If you do the 2-deep
//         pipeline (cmdbuf_pipeline_2deep.c), duplicate the ring per slot.
// SPEEDUP: ~1.01–1.02× decode (88→89 tok/s in csrc, commit 2b1d6ef).

#include "metal_shim.h"

// One static gptoss_buf per param.  Allocated lazily on first call.
// dst pointer for memcpy is acquired via gptoss_buf_contents().
#define PARAM_BUF(name, sz) \
    static gptoss_buf* name##_buf = NULL; \
    if (!name##_buf) name##_buf = gptoss_buf_new(g_ctx, (sz)); \
    void* name##_dst = gptoss_buf_contents(name##_buf)

void forward(int q_off, int Lq) {
    // dimsRMS = {Lq, HIDDEN}.  Allocate once, refill in place each call.
    uint32_t dimsRMS[2] = { (uint32_t)Lq, HIDDEN };
    PARAM_BUF(dimsRMS, sizeof(dimsRMS));
    memcpy(dimsRMS_dst, dimsRMS, sizeof(dimsRMS));

    // dimsQp = {Lq, HIDDEN, N_QHEADS*HEAD_DIM} for q_proj
    uint32_t dimsQp[3] = { (uint32_t)Lq, HIDDEN, N_QHEADS*HEAD_DIM };
    PARAM_BUF(dimsQp, sizeof(dimsQp));
    memcpy(dimsQp_dst, dimsQp, sizeof(dimsQp));

    // ... etc.  Each forward() refills only the fields that depend on
    // (q_off, Lq).  Truly constant fields can be lifted out (see
    // param_buf_const.c).
}

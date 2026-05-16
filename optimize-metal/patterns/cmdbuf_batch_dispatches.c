// patterns/cmdbuf_batch_dispatches.c
//
// WHAT: Open one Metal command buffer per forward() call, record all
//       kernel dispatches into it, commit + wait once.  Do not commit
//       between kernels or between layers.
// WHEN: Always.  This is the foundation; every other optimization
//       assumes it.
// SPEEDUP: 1.5–1.8× decode (a13d0de in csrc was +78%).
// CAVEAT: Each forward() must end with commit+wait so subsequent host
//         reads (e.g., the argmax result) are visible.
//
// This is the structural pattern, not a drop-in.  See csrc/main.c
// forward() for a complete example.

#include "metal_shim.h"

static gptoss_cmdbuf* g_cmdbuf;   // active cmdbuf during forward()

void forward(int q_off, int Lq) {
    // The caller has set g_cmdbuf to a freshly-created cmdbuf.

    for (int li = 0; li < N_LAYERS; li++) {
        // Every dispatch goes into the same cmdbuf — NO commit here.
        gptoss_cmdbuf_dispatch(g_cmdbuf, pso_rmsnorm,    args_rms,    ...);
        gptoss_cmdbuf_dispatch(g_cmdbuf, pso_linear_q,   args_qproj,  ...);
        gptoss_cmdbuf_dispatch(g_cmdbuf, pso_linear_k,   args_kproj,  ...);
        gptoss_cmdbuf_dispatch(g_cmdbuf, pso_linear_v,   args_vproj,  ...);
        gptoss_cmdbuf_dispatch(g_cmdbuf, pso_rope,       args_rope_q, ...);
        gptoss_cmdbuf_dispatch(g_cmdbuf, pso_rope,       args_rope_k, ...);
        gptoss_cmdbuf_dispatch(g_cmdbuf, pso_sdpa,       args_sdpa,   ...);
        gptoss_cmdbuf_dispatch(g_cmdbuf, pso_linear_o,   args_oproj,  ...);
        // ... rest of the layer ...
    }
    // final norm + lm_head + argmax all go in the same cmdbuf
}

// CALLER, decode loop:
void decode_step(int q_off) {
    g_cmdbuf = gptoss_cmdbuf_new(g_ctx);
    forward(q_off, 1);
    gptoss_cmdbuf_commit_wait(g_cmdbuf, NULL);
    g_cmdbuf = NULL;
    // now safe to read next_id from host buffer
}

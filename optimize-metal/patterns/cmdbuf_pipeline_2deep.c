// patterns/cmdbuf_pipeline_2deep.c
//
// WHAT: Keep TWO command buffers in flight at all times during decode.
//       While the GPU runs decode step k (cmdbuf in slot S^k), the CPU
//       encodes step k+1 into a fresh cmdbuf in slot S^{k+1}, using a
//       duplicated param ring so writes don't race.
// WHEN: After batch-dispatches and persistent param buffers are in
//       place.  This is the key host/GPU overlap optimization.
// SPEEDUP: 1.10–1.20× decode (1f1afbd: "pipeline decode (depth=2) —
//          overlap CPU encode with GPU compute").
// CAVEAT: All per-call param buffers must be duplicated per slot.
//         Workspace buffers (x_buf, h_buf, ...) can be shared if you
//         carefully observe that step k+1's writes happen AFTER
//         step k's commit completes.  In practice, KV cache is the
//         only large state that crosses steps; everything else is
//         re-written within a single forward.
// COMMIT:  1f1afbd.

#include "metal_shim.h"

#define PIPELINE_DEPTH 2

static gptoss_cmdbuf* g_cmdbufs[PIPELINE_DEPTH];

// Per-slot persistent param buffers.
#define PARAM_BUF_SLOT(name, sz) \
    static gptoss_buf* name##_buf[PIPELINE_DEPTH] = {0}; \
    if (!name##_buf[slot]) name##_buf[slot] = gptoss_buf_new(g_ctx, (sz)); \
    void* name##_dst = gptoss_buf_contents(name##_buf[slot])

void forward_decode(int q_off, int slot) {
    uint32_t dimsRq[4] = { 1u, N_QHEADS, HEAD_DIM, (uint32_t)q_off };
    PARAM_BUF_SLOT(dimsRq, sizeof(dimsRq));
    memcpy(dimsRq_dst, dimsRq, sizeof(dimsRq));
    // ... record dispatches into g_cmdbufs[slot], using *_buf[slot] for args
}

// Driver:
void decode_loop(int Lp, int max_tokens, int* gen_ids) {
    int next_id = /* from prefill argmax */ 0;
    int n_gen = 0;
    int slot_prev = -1;

    // Prime the pipeline: encode + commit step 0.
    int slot = 0;
    g_cmdbufs[slot] = gptoss_cmdbuf_new(g_ctx);
    /* set input id, dispatch embed + forward + argmax into g_cmdbufs[slot] */
    gptoss_cmdbuf_commit(g_cmdbufs[slot]);   // ASYNC commit

    while (n_gen < max_tokens) {
        int slot_next = (slot + 1) % PIPELINE_DEPTH;

        // While step `slot` runs on GPU, encode step `slot_next` on CPU.
        g_cmdbufs[slot_next] = gptoss_cmdbuf_new(g_ctx);
        /* dispatch step k+1 into g_cmdbufs[slot_next], using buffers in slot_next */
        gptoss_cmdbuf_commit(g_cmdbufs[slot_next]);

        // Now wait for step `slot` to finish — its output (next_id) is ready.
        gptoss_cmdbuf_wait(g_cmdbufs[slot], NULL);
        /* read next_id from host-visible buffer */;
        gen_ids[n_gen++] = next_id;

        if (next_id == STOP_ID) break;
        slot_prev = slot;
        slot = slot_next;
    }
    // Drain remaining cmdbufs.
}

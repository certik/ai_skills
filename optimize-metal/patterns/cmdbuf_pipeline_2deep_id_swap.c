// patterns/cmdbuf_pipeline_2deep_id_swap.c
//
// WHAT: 2-deep cmdbuf pipeline with a "next-token-id swap" trick that
//       avoids ever needing a CPU roundtrip for the next embed's input.
// WHEN: After A1-A3 (batched dispatches, persistent param buffers).
//       This is the standard A4 pattern but with one key insight that
//       makes it much simpler than the textbook version.
// SPEEDUP: 1.10-1.30x decode. In the Qwen3.6 port, this closed the
//          final gap to MLX: GPU busy went from 82% of wall to ~100%.
//
// KEY INSIGHT: the only data dependency between consecutive decode
// steps is the next-token id. If we make that dependency live entirely
// on the GPU (argmax of step N writes a buffer, embed of step N+1
// reads it), the CPU never needs the value to encode the next cmdbuf.
// The CPU only reads it AFTER waiting, for printing / EOS check.
//
// We use TWO id buffers, swapping per slot, so step at slot s reads
// id_io_buf[s] and writes id_io_buf[1-s]. The 1-s buffer becomes the
// next pipelined step's input. CPU reads id_io_buf[1-s] AFTER
// wait(cb[s]) to print/EOS-check the just-finished token.
//
// Why 2 id buffers (not 1 shared)? Because the CPU read of the just-
// finished token must NOT race with the NEXT step's argmax write.
// With 2 buffers, the next step writes to the OTHER buffer, so the
// buffer the CPU reads is no longer being touched by the GPU.

#include "metal_shim.h"

#define PIPE_DEPTH 2

// Per-slot persistent param arena. Crucial: with one cmdbuf in flight
// per slot, the CPU encoding cmdbuf[1-s] would race with the GPU
// reading cmdbuf[s]'s params if params lived in a shared buffer.
static gpu_buf* g_params_buf[PIPE_DEPTH];
static size_t   g_params_off[PIPE_DEPTH];
static int      g_slot = 0;

static gpu_arg_buf push_params(const void* p, size_t n) {
    int s = g_slot;
    size_t off = (g_params_off[s] + 15u) & ~(size_t)15;
    memcpy((char*)gpu_buf_contents(g_params_buf[s]) + off, p, n);
    g_params_off[s] = off + n;
    return (gpu_arg_buf){ g_params_buf[s], off };
}
static void reset_params(int slot) {
    g_slot = slot;
    g_params_off[slot] = 0;
}

// id_io_buf[s] is the per-slot 1-int buffer.
static gpu_buf* id_io_buf[PIPE_DEPTH];

// Encode + ASYNC commit one decode step. Caller wait()s later.
static gpu_cmdbuf* encode_decode_step(int slot, int q_off) {
    gpu_cmdbuf* cb = gpu_cmdbuf_new(g_ctx);
    reset_params(slot);
    dispatch_embed (cb, id_io_buf[slot], x_buf, /*...*/);
    forward_into   (cb, q_off, /*Lq=*/1);
    dispatch_argmax(cb, logits_buf, id_io_buf[1 - slot], /*vocab*/);
    gpu_cmdbuf_commit(cb);
    return cb;
}

void decode_loop(int Lp, int max_tokens) {
    // PREFILL CONTRACT: prefill argmax must write to id_io_buf[0],
    // because the first decode step is at slot=0 and embeds
    // id_io_buf[0]. The first generated token IS the prefill argmax:
    // EMIT IT before priming the pipeline (do NOT skip it!).
    // Forgetting this shifts the entire output by one token.
    int32_t first_tok = ((int32_t*)gpu_buf_contents(id_io_buf[0]))[0];
    emit_token(first_tok);

    gpu_cmdbuf* cb_arr[PIPE_DEPTH] = {0};
    int slot = 0;
    int q_off = Lp;

    if (max_tokens > 1) {
        cb_arr[slot] = encode_decode_step(slot, q_off++);
    }

    int n_gen = 1;
    while (n_gen < max_tokens) {
        int next_slot = 1 - slot;
        bool more = (n_gen + 1) < max_tokens;
        if (more) {
            cb_arr[next_slot] = encode_decode_step(next_slot, q_off++);
        }

        gpu_cmdbuf_wait(cb_arr[slot], NULL);
        int32_t tok = ((int32_t*)gpu_buf_contents(id_io_buf[1 - slot]))[0];
        n_gen++;
        if (is_eos(tok)) {
            // Drain the pre-encoded next cmdbuf (its work was wasted).
            if (more) gpu_cmdbuf_wait(cb_arr[next_slot], NULL);
            break;
        }
        emit_token(tok);
        slot = next_slot;
    }
}

// PITFALLS (each cost an iteration in practice):
//
// 1) PREFILL EMISSION. The first generated token IS the prefill
//    argmax. Forgetting to emit it before priming shifts ALL tokens
//    by one and the "first N tokens match C ref" check fails because
//    everything is offset by 1.
//
// 2) PARAM-BUF RACE. With one shared param buffer, the CPU encoding
//    step k+1 overwrites the params the GPU is still reading for
//    step k. Always duplicate per slot.
//
// 3) ID-BUF READ RACE. CPU reading from id_io_buf[s] (the slot the
//    LAST step wrote and the NEXT step just wrote again) is a race.
//    Solution: read from id_io_buf[1-slot] (the buffer the JUST-
//    finished step wrote, which the next step does not touch).
//
// 4) EOS DRAINING. When EOS is detected, the next cmdbuf has already
//    been committed (because more=true). You must wait() on it before
//    returning, or program exit while the GPU has in-flight work can
//    crash or produce confusing error messages.

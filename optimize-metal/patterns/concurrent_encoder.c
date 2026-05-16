// patterns/concurrent_encoder.c
//
// WHAT: Switch from Metal's default *serial* compute encoder to the
//       *concurrent* encoder, which lets independent dispatches run in
//       parallel on the GPU.  Insert explicit memory barriers only where
//       genuine data dependencies exist.
// WHEN: After basic scheduling optimizations.  If the profile shows
//       GPU idle gaps between same-layer dispatches that don't actually
//       depend on each other (e.g., q_proj, k_proj, v_proj all read from
//       the same X but write to different outputs — they can overlap).
// SPEEDUP: 1.05–1.10× depending on overlap opportunity.
// CAVEAT: You become responsible for inserting all real dependencies as
//         explicit barriers.  Get one wrong → garbage output.
// COMMIT:  d08def0 — "Concurrent encoder w/ explicit barriers; q/k/v +
//                    gate/up may overlap"

// Conceptual outline (the actual API call lives in metal_shim.m, this
// is just where you add the BAR() calls in main.c forward()).

void forward(int q_off, int Lq) {
    // RMSNorm reads X, writes H.
    dispatch(rmsnorm, X, W_norm, H);
    BAR();   // q/k/v all read H

    // q_proj, k_proj, v_proj can all run concurrently — they read H
    // (already finished) and write disjoint outputs.
    dispatch(linear, H, Wq, Bq, Q);
    dispatch(linear, H, Wk, Bk, K);
    dispatch(linear, H, Wv, Bv, V);
    BAR();   // rope reads Q, K

    // RoPE rotates Q, K in place — both reads only their own buffer.
    dispatch(rope, Q, freqs, ...);
    dispatch(rope, K, freqs, ...);
    BAR();   // sdpa reads Q, K, V

    dispatch(sdpa, Q, K, V, sinks, OUT, ...);
    BAR();   // o_proj reads OUT

    dispatch(linear_add, OUT, Wo, Bo, X /* residual+output */);
    BAR();   // rmsnorm_post reads X

    // ... similarly: router_proj is independent of (gate_up, down)
    // start; gate and up are typically already fused.
}

// patterns/kprof.c — implementation of the KPROF profiler.
//
// Compile alongside kernels.c. Runtime overhead when off: 1 load + 1
// branch per kernel entry. Runtime overhead when on: ~30 ns per kernel
// entry (one clock_gettime + a fixed-size add).

#include "kprof.h"
#include <stdio.h>
#include <time.h>

static int    g_kp_on = 0;
static double g_kp_t[KP__N] = {0};
static long   g_kp_n[KP__N] = {0};

static const char* kp_name(kp_kernel_t k) {
    switch (k) {
    case KP_EMBED_GATHER: return "embed_gather";
    case KP_SCALAR_MUL:   return "scalar_mul";
    case KP_RMSNORM:      return "rmsnorm";
    case KP_RMSNORM_NS:   return "rmsnorm_noscale";
    case KP_LINEAR:       return "linear";
    case KP_ROPE:         return "rope";
    case KP_SDPA:         return "sdpa";
    case KP_GELU:         return "gelu";
    case KP_GEGLU:        return "geglu";
    case KP_RESIDUAL_ADD: return "residual_add";
    case KP_EMUL:         return "elementwise_mul";
    case KP_ADD_SCALE:    return "add_then_scale";
    case KP_SOFTCAP:      return "logit_softcap";
    case KP_ARGMAX:       return "argmax";
    default:              return "?";
    }
}

void kp_enable(int on)        { g_kp_on = on; }
int  kp_is_on(void)           { return g_kp_on; }
void kp_add(kp_kernel_t k, double s) { g_kp_t[k] += s; g_kp_n[k] += 1; }

double kp_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void kp_report(void) {
    if (!g_kp_on) return;
    double total = 0.0;
    for (int i = 0; i < KP__N; i++) total += g_kp_t[i];
    fprintf(stderr, "[KPROF] per-kernel wall (sum over the whole run):\n");
    fprintf(stderr, "  %-18s %10s %10s %10s\n", "kernel", "wall_s", "calls", "us/call");
    for (int i = 0; i < KP__N; i++) {
        if (g_kp_n[i] == 0) continue;
        double s = g_kp_t[i];
        long n = g_kp_n[i];
        double pct = total > 0.0 ? 100.0 * s / total : 0.0;
        fprintf(stderr, "  %-18s %10.4f %10ld %10.1f  (%4.1f%%)\n",
                kp_name((kp_kernel_t)i), s, n, s / (double)n * 1e6, pct);
    }
    fprintf(stderr, "  %-18s %10.4f\n", "TOTAL", total);
}

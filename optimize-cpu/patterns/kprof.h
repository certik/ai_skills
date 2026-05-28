// patterns/kprof.h — per-kernel profiler (KPROF=1 env var to enable).
//
// WHAT: Lightweight wall-clock accounting per named kernel. Call
//       KPROF_ENTER/LEAVE at the top/bottom of each public kernel; call
//       kp_report() before exit. Build with -DKPROF or just keep it
//       always compiled in — disabled at runtime (kp_is_on()==0) costs
//       1 conditional + 1 function call per kernel invocation, well
//       under 0.5% overhead.
//
// WHEN: Add this BEFORE doing any kernel optimization. It's the answer
//       to "which kernel actually matters?" — usually `linear` at 95+%
//       for dense bf16 LLMs, freeing you from spending time on already-
//       cheap kernels.
//
// WHY: Without this you'll spend hours micro-optimizing rmsnorm /
//      argmax / rope which together account for <3% of decode time.
//      The first KPROF report is the most important diagnostic in the
//      whole pipeline.
//
// Wiring in main.c:
//
//      int main(int argc, char** argv) {
//          if (getenv("KPROF")) kp_enable(1);
//          ...
//          kp_report();  // before tokenizer/weights free
//          return 0;
//      }
//
// Wiring in each kernel (kernels.c):
//
//      #define KPROF_ENTER() double _kp_t0 = kp_is_on() ? kp_now() : 0.0
//      #define KPROF_LEAVE(K) do { if (kp_is_on()) kp_add((K), kp_now() - _kp_t0); } while (0)
//
//      void linear_bf16(...) {
//          KPROF_ENTER();
//          /* body */
//          KPROF_LEAVE(KP_LINEAR);
//      }

#ifndef KPROF_H
#define KPROF_H

#include <stdint.h>

typedef enum {
    KP_EMBED_GATHER = 0,
    KP_SCALAR_MUL,
    KP_RMSNORM,
    KP_RMSNORM_NS,
    KP_LINEAR,
    KP_ROPE,
    KP_SDPA,
    KP_GELU,
    KP_GEGLU,
    KP_RESIDUAL_ADD,
    KP_EMUL,
    KP_ADD_SCALE,
    KP_SOFTCAP,
    KP_ARGMAX,
    KP__N
} kp_kernel_t;

void   kp_enable(int on);
int    kp_is_on(void);
void   kp_add(kp_kernel_t k, double seconds);
void   kp_report(void);    // prints per-kernel table to stderr.

double kp_now(void);

#endif

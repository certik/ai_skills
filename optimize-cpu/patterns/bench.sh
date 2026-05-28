#!/usr/bin/env bash
# patterns/bench.sh — canonical CPU benchmark.
#
# WHAT: Runs the optimized binary on a fixed prompt / max-tokens and prints
#       prefill + decode tok/s, with OpenMP env tuned for the
#       physical-core count.
#
# WHEN: Use this as the SINGLE source of truth for benchmark numbers.
#       Re-run after every optimization commit; diff the generated tokens
#       against tools/ref_tokens.txt (saved early from the first
#       -O3 -fopenmp build).
#
# WHY:  Without a canonical bench you'll waste time chasing noise from
#       changing prompts, max-tokens, or thread counts. Fixing all three
#       makes commit-to-commit comparison clean.
#
# Adjust:
#  - MODEL: weight directory
#  - OMP_NUM_THREADS: ALWAYS the physical-core count, not logical CPUs.
#  - prompt and --max-tokens: long enough that prefill + decode each take
#    several seconds at the FINAL speed (so noise is small fraction).
#
# Adapt the prompt to your model.

set -e

: "${MODEL:=/workspace/models/gemma-4-E4B}"

# Physical core count for AMD EPYC 7763 16C/32T VM.
# On a 16C/32T desktop: 16. On a 6C/12T laptop: 6.
# Find it with: lscpu | grep -E 'Core|Thread'
: "${OMP_NUM_THREADS:=16}"

# Pin threads to distinct physical cores in NUMA-local order. close+cores
# prevents the scheduler from migrating threads across NUMA nodes.
: "${OMP_PROC_BIND:=close}"
: "${OMP_PLACES:=cores}"

# Busy-wait between parallel regions (otherwise libgomp default 'passive'
# adds ~30% overhead on small parallel-fors). Safe because we have plenty
# of cores.
: "${OMP_WAIT_POLICY:=active}"

export OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES OMP_WAIT_POLICY

exec ./gemma4-cpu \
    --prompt "Solve x^2 + x + 1 = 0" \
    --max-tokens 64 \
    --model "$MODEL" "$@"

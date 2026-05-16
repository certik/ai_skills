#ifndef KERNEL_CONCAT_H
#define KERNEL_CONCAT_H
// Concatenate Metal kernel source files, recursively expanding
// `#include "..."` directives so vendored headers under mlx_steel/ get
// inlined. System `<...>` includes pass through unchanged. Honors
// `#pragma once` via path-based dedup. Returns a malloc'd string.
char* concat_kernels(int n, const char** paths);
#endif

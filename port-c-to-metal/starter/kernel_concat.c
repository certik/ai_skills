#include "kernel_concat.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* slurp(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* s = (char*)malloc((size_t)n + 1);
    fread(s, 1, (size_t)n, f); s[n] = '\0';
    fclose(f); return s;
}

typedef struct { char* p; size_t n, cap; } sbuf;
static void sb_append(sbuf* s, const char* d, size_t n) {
    if (s->n + n + 1 > s->cap) {
        s->cap = (s->n + n + 1) * 2;
        s->p = (char*)realloc(s->p, s->cap);
    }
    memcpy(s->p + s->n, d, n); s->n += n; s->p[s->n] = 0;
}

typedef struct seen_node { char* path; struct seen_node* next; } seen_node_t;
static int seen_contains(seen_node_t* h, const char* p) {
    for (seen_node_t* x = h; x; x = x->next) if (!strcmp(x->path, p)) return 1;
    return 0;
}

static char* resolve_inc(const char* incl_from, const char* rel) {
    char dir[1024]; snprintf(dir, sizeof dir, "%s", incl_from);
    char* slash = strrchr(dir, '/');
    if (slash) *slash = 0; else dir[0] = 0;
    char* out = (char*)malloc(strlen(dir) + strlen(rel) + 4);
    if (dir[0]) sprintf(out, "%s/%s", dir, rel);
    else        sprintf(out, "%s", rel);
    return out;
}

static void inline_file(sbuf* out, const char* path, seen_node_t** seen) {
    if (seen_contains(*seen, path)) return;
    seen_node_t* node = (seen_node_t*)malloc(sizeof(*node));
    node->path = strdup(path); node->next = *seen;
    *seen = node;

    char* src = slurp(path);
    if (!src) { fprintf(stderr, "slurp %s: %s\n", path, strerror(errno)); exit(1); }

    const char* p = src;
    while (*p) {
        const char* line_start = p;
        while (*p && *p != '\n') p++;
        size_t line_len = (size_t)(p - line_start);
        if (*p == '\n') p++;

        const char* q = line_start;
        while (q < line_start + line_len && (*q == ' ' || *q == '\t')) q++;
        if (q < line_start + line_len && *q == '#') {
            const char* r = q + 1;
            while (r < line_start + line_len && (*r == ' ' || *r == '\t')) r++;
            if ((size_t)(line_start + line_len - r) >= 7 && !strncmp(r, "include", 7)) {
                const char* s = r + 7;
                while (s < line_start + line_len && (*s == ' ' || *s == '\t')) s++;
                if (s < line_start + line_len && *s == '"') {
                    const char* a = s + 1; const char* b = a;
                    while (b < line_start + line_len && *b != '"') b++;
                    if (b < line_start + line_len) {
                        char rel[512];
                        size_t L = (size_t)(b - a);
                        if (L >= sizeof rel) L = sizeof rel - 1;
                        memcpy(rel, a, L); rel[L] = 0;
                        char* sub = resolve_inc(path, rel);
                        inline_file(out, sub, seen);
                        free(sub);
                        sb_append(out, "\n", 1);
                        continue;
                    }
                }
            }
        }
        sb_append(out, line_start, line_len);
        sb_append(out, "\n", 1);
    }
    free(src);
}

char* concat_kernels(int n, const char** paths) {
    sbuf out = {0};
    seen_node_t* seen = NULL;
    for (int i = 0; i < n; i++) inline_file(&out, paths[i], &seen);
    seen_node_t* x = seen;
    while (x) { seen_node_t* nx = x->next; free(x->path); free(x); x = nx; }
    return out.p ? out.p : strdup("");
}

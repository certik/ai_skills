// Byte-level BPE tokenizer (HF-style: GPT-2 byte-to-unicode + ranked merges).
//
// The encoding algorithm (`bpe_byte_pair_merge`) is a direct port of the
// classical rank-priority BPE merger used by HF `tokenizers`, openai's
// tiktoken, and friends. The **pre-tokenizer** rules differ per family —
// see tokenizer.c for the alternation order. The default template ships
// with an ASCII subset of the Qwen 3.5 / Llama 3 regex; swap for your
// model as needed (gpt-oss/o200k_harmony, Mistral, ...).

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tokenizer_t tokenizer_t;

tokenizer_t* tk_load(const char* path, char** err);
void              tk_free(tokenizer_t*);

// Encode UTF-8 text without recognising special tokens (special tokens, if
// any, must be tokenised separately via tk_special).  Output capacity must
// be at least `n_bytes + 1` to handle the worst case (each byte its own
// token).  Returns number of tokens written, or -1 on error.
int tk_encode_ordinary(const tokenizer_t* tk,
                       const char* text, size_t n_bytes,
                       int* out_ids, int max_ids);

// Look up the id of a special token by its surface form (e.g. "<|im_start|>",
// "<|start|>", "<s>", "<|begin_of_text|>"). Returns -1 if not found.
int tk_special(const tokenizer_t* tk, const char* name);

// Decode a sequence of ids into raw bytes (no UTF-8 validation).  Returns
// number of bytes written; out is NOT null-terminated.
int tk_decode(const tokenizer_t* tk,
              const int* ids, int n_ids,
              char* out, int max_bytes);

// Diagnostics.
int  tk_n_vocab(const tokenizer_t*);
const char* tk_token_bytes(const tokenizer_t*, int id, int* out_len);

#ifdef __cplusplus
}
#endif
#endif

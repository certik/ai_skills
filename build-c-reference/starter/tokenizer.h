// Tiktoken-style BPE tokenizer for o200k_harmony.
//
// The encoding algorithm (`bpe_byte_pair_merge`) is a direct port of
// openai/harmony's src/tiktoken.rs::_byte_pair_merge.  The pre-tokenizer
// implements the o200k_harmony regex pattern (ASCII subset — sufficient for
// the harmony chat template and English prompts):
//
//   1. [^\r\n\p{L}\p{N}]?[\p{Lu}...]*[\p{Ll}...]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   2. [^\r\n\p{L}\p{N}]?[\p{Lu}...]+[\p{Ll}...]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   3. \p{N}{1,3}
//   4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
//   5. \s*[\r\n]+
//   6. \s+(?!\S)
//   7. \s+
//
// For ASCII input \p{L} = [A-Za-z] and \p{N} = [0-9].  Non-ASCII bytes
// (>= 0x80) are treated as "other" (rule 4 territory) which is benign for
// our use case (the entire harmony scaffolding is ASCII).

#ifndef GPTOSS_TOKENIZER_H
#define GPTOSS_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gptoss_tokenizer gptoss_tokenizer;

gptoss_tokenizer* tk_load(const char* path, char** err);
void              tk_free(gptoss_tokenizer*);

// Encode UTF-8 text without recognising special tokens (special tokens, if
// any, must be tokenised separately via tk_special).  Output capacity must
// be at least `n_bytes + 1` to handle the worst case (each byte its own
// token).  Returns number of tokens written, or -1 on error.
int tk_encode_ordinary(const gptoss_tokenizer* tk,
                       const char* text, size_t n_bytes,
                       int* out_ids, int max_ids);

// Look up the id of a special token by its surface form (e.g. "<|start|>").
// Returns -1 if not found.
int tk_special(const gptoss_tokenizer* tk, const char* name);

// Decode a sequence of ids into raw bytes (no UTF-8 validation).  Returns
// number of bytes written; out is NOT null-terminated.
int tk_decode(const gptoss_tokenizer* tk,
              const int* ids, int n_ids,
              char* out, int max_bytes);

// Diagnostics.
int  tk_n_vocab(const gptoss_tokenizer*);
const char* tk_token_bytes(const gptoss_tokenizer*, int id, int* out_len);

#ifdef __cplusplus
}
#endif
#endif

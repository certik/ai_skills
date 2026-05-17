# Tokenizer: build_tokenizer.c, formats, regex

Read this when:
- Building `tokenizer.bin` from a model directory.
- The model ships an older HF tokenizer (`vocab.json` + `added_tokens.json`
  instead of `tokenizer.json`).
- The model uses sentencepiece / unigram (older Llama, Mistral, T5).
- The pre-tokenizer regex needs swapping for a non-Qwen/Llama family.
- The chat-template C builder produces different prompt IDs than the
  Python reference.
- `vocab_size` in `config.json` doesn't match the tokenizer's vocab.

## Contents

1. `build_tokenizer.c` and `tokenizer.bin`.
2. Modern format: `tokenizer.json` (byte-level BPE).
3. Older format: `vocab.json` + `added_tokens.json`.
4. Sentencepiece / unigram families.
5. Pre-tokenizer regex per family.
6. Chat templates.
7. `vocab_size` padding.

## `build_tokenizer.c` and `tokenizer.bin`

`build_tokenizer.c` converts HF tokenizer files → `tokenizer.bin` in
pure C. The runtime `tokenizer.c` mmaps the resulting blob. Drop-in for
any **byte-level BPE** family (Qwen, Llama 3, GPT-2, o200k, gpt-oss)
provided the model ships a `tokenizer.json`.

Run it on your model:

```
make tokenizer MODEL_DIR=/path/to/model
```

### `tokenizer.bin` MAGIC length

The C reader expects 12 bytes of magic; if the Python writer (or an
older `build_tokenizer.c`) writes only 11 you'll silently load garbage.
The fixed template writes 12. The `safetensors.c` `__metadata__` key is
similarly 12 chars, not 11 — both are easy to off-by-one.

## Modern format: `tokenizer.json` (byte-level BPE)

`tokenizer.json` is the canonical HF tokenizer format and is drop-in.
Special tokens live in `added_tokens` — `build_tokenizer.c` includes
them automatically.

## Older format: `vocab.json` + `added_tokens.json`

Older HF tokenizers (Qwen ≤ 2.5, Llama 1, some older community models)
don't have `tokenizer.json`. Instead, two files:

- `vocab.json`: flat `{token_string: id}` (GPT-2 byte-encoded keys,
  where printable ASCII passes through and control bytes map to
  U+0100..U+0142).
- `added_tokens.json`: flat `{name: id}` for special tokens.

Modify `build_tokenizer.c` to read BOTH and merge them: emit one
entry per id, with `added_tokens` ids getting empty BPE strings so the
merge engine never picks them as merge candidates. Total id range is
`max(vocab.json ids, added_tokens.json ids) + 1`. (HF tokenizers
reserve special-token ids past `len(vocab)` and may leave gaps.)

The tiktoken trick (merge priority == vocab id) still works — no
`merges.txt` needed. This works for all Qwen / Llama / GPT-2-family
tokenizers as long as the vocab was built in merge order, which they
all are.

## Sentencepiece / unigram families

For sentencepiece / unigram tokenizers (older Llama, Mistral, T5) the
vocab-key decoding step is different — copy `build_tokenizer.c` and
rewrite the `decode_vocab_key` helper. The merge engine itself is
generic.

## Pre-tokenizer regex per family

The pre-tokenizer regex in `tokenizer.c::pretokenize()` varies per
tokenizer family. The starter ships an ASCII subset that matches Qwen
3.5 / Llama 3. For o200k_harmony / Mistral / sentencepiece-based
families, rewrite the `match_*` helpers per `tokenizer.json::pre_tokenizer.pattern`.

Notable differences:

- Digit handling (some families split digits one-by-one; others group).
- How `\p{L}\p{M}+` is split (Unicode-letter + combining marks).
- Apostrophes / contractions.

## Chat templates

The chat-template scaffolding is part of the *input* to the model.
Before debugging anything, dump the prompt IDs from both the Python
reference and your C builder and assert byte-for-byte equality. Many
"the C reference is broken" problems are actually tokenization
mismatches.

Tip: have the Python reference output the numeric token IDs it
produces for the same `(system, user, reasoning)` tuple, and write the
C builder to match byte-for-byte. Add a small test that compares C
and Python token sequences.

## `vocab_size` padding

Many models pad the embedding to a power of 2 or a multiple of
64/128. The extra rows are unused; the C `argmax` over the full
`vocab_size` is correct as long as you never emit a row index past the
tokenizer's known set during decode.

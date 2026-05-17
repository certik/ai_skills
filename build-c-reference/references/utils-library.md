# The `utils/` library — Python-stdlib equivalents in C

Read this when:
- Understanding why the `utils/` subdirectory exists as a separate
  layer.
- Considering adding new helpers, or wondering whether something
  belongs in `utils/` vs `safetensors.c` / `tokenizer.c` / `main.c`.
- Porting the C reference to a new model and deciding what's reusable.

## Why centralize?

Four different C files in the project (`safetensors.c`, `tokenizer.c`,
`build_tokenizer.c`, `main.c::load_config`) all need to do the same
four things:

1. read a file by name (`open` / `fstat` / `mmap` / `close`),
2. grow a byte buffer and write it back out (`bytearray` +
   `struct.pack`),
3. parse + navigate a JSON document, and
4. encode / decode UTF-8 codepoints.

If every caller does these by hand you get four subtly different
mini-parsers:

- the substring-search JSON walker bites you on multi-modal configs
  (duplicate key names in nested objects),
- the manual mmap dance gets the error path wrong,
- the BPE byte encoder reinvents UTF-8.

Centralising them is a one-time ~700-LOC investment that the rest of
`src-cpu/` reads as if it were Python:

```c
// "with open(path) as f: cfg = json.load(f)" + "cfg['text_config']['rope_theta']"
json_doc_t* doc = json_open_file(path, &err);
double theta = json_double_or(
    json_path(json_root(doc), "text_config.rope_parameters.rope_theta"), 1e4);

// "buf = bytearray(); buf += MAGIC; buf += struct.pack('<I', n); open(out,'wb').write(buf)"
bytebuf_t out = BYTEBUF_INIT;
bb_append    (&out, MAGIC, MAGIC_LEN);
bb_append_u32(&out, n);
bb_write_file(&out, out_path);

// "with open(path,'rb') as f: data = mmap.mmap(f.fileno(), 0, mmap.PROT_READ)"
iofile_t f;
iofile_mmap_ro(path, &f, &err);
// ... f.data, f.size ...
iofile_close(&f);
```

## What lives in `utils/`

- `utils/bf16.h` — `bf16_to_f32` / `f32_to_bf16` (round-to-nearest-even).
- `utils/jsmn.h` — vendored single-header JSON tokenizer (MIT, Serge
  Zaitsev).
- `utils/json.{c,h}` — small Python-like wrapper: `json_open_file`,
  `json_open_mem`, `json_get`, `json_path` (dotted), `json_obj_iter`,
  `json_arr_iter`, `json_int_or`, `json_double_or`, `json_str_view`,
  `json_str_utf8`, `json_str_codepoints`.
- `utils/utf8.{c,h}` — `utf8_encode` / `utf8_decode` (one codepoint
  each).
- `utils/bytebuf.{c,h}` — growable buffer: `bb_append`,
  `bb_append_u32` (little-endian), `bb_write_file`. Aborts on OOM
  (same policy as Python's `bytearray` raising `MemoryError`).
- `utils/iofile.{c,h}` — `iofile_mmap_ro` / `iofile_close`. Replaces
  the open/fstat/mmap/close dance in every caller.

## Performance note: `JSMN_PARENT_LINKS`

The wrapper defines `JSMN_PARENT_LINKS` before including `jsmn.h`.
Without this, jsmn back-walks the entire token array on every `,`
inside an object — O(N²) on container size. For a 250k-entry
`tokenizer.json::model.vocab` that's ~40 s of pure overhead; with
`JSMN_PARENT_LINKS` defined it drops to ~40 ms (1000× speedup). The
cost is 4 extra bytes per token (~2 MB extra for the vocab).

## Centralize JSON, don't re-invent it per caller

It's tempting to write a one-off "good enough" substring-search JSON
walker every time (`strstr` for the key, `strchr` for `:`, `atoi`).
This breaks for any nested object with duplicate key names (every
multi-modal config) and for any key whose value contains a literal
`:`. Build one JSON reader (the starter's `utils/json` is ~500 LOC)
and route everything through it — `safetensors.c` shard headers,
`config.json`, `tokenizer.json`, and any future model registry files.

## Don't leak model knowledge into `utils/`

Treat `utils/` as a fixed dependency of the project. Do *not* leak
any model-specific knowledge into it: that's how you keep the port
to a new model from being a fork of the whole tree. Anything
model-specific belongs in `kernels.c`, `main.c`, or `<chattmpl>.{c,h}`
— never in `utils/`.

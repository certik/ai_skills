# PyTorch / HF transformers gotchas

Read this when:
- Your reference is PyTorch or HF transformers (no MLX port).
- Writing `tools/dump_ref.py` for a PyTorch backend.
- The reference uses `trust_remote_code=True` (e.g., Dream).
- The C end-to-end output diverges from `model.generate(...)` even when
  per-kernel tests pass.
- Parsing a multi-modal `config.json`.

## Contents

1. `SDPBackend.FLASH_ATTENTION` is not textbook math on CPU bf16 — the
   biggest single trap.
2. `AutoModel` wrappers and `.logits` vs `.last_hidden_state`.
3. `num_logits_to_keep=0` quirk.
4. `forward_hook` payload shapes.
5. `torch_dtype=torch.bfloat16` is slow on CPU.
6. `model.config` vs `tokenizer.special_tokens_map` (stop tokens).
7. `trust_remote_code=True` and how to grab classes from the loaded
   module.
8. `do_sample=False` + `temperature=0.0` warning.
9. Multi-modal `config.json` parsing.

## `SDPBackend.FLASH_ATTENTION` is not bit-equivalent to textbook math

On CPU, PyTorch's default for
`torch.nn.functional.scaled_dot_product_attention(..., is_causal=False)`
with bf16 q/k/v is `FLASH_ATTENTION`, which disagrees with
`SDPBackend.MATH` by `mean|d| ≈ 0.004`, `max|d| ≈ 0.93` on a typical
7B model. Your naive C SDPA implements the textbook math, so:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
    out = model(input_ids=ids, ...)  # oracle dump
```

Apply this both in `tools/dump_ref.py` (so the oracle matches what
your C SDPA computes) AND in `tools/run_ref_<sampler>.py` (so the
reference's generated tokens are comparable to your C output).
Without this, your per-kernel SDPA test will fail by ~0.9 absolute
(real divergence, not precision drift) and end-to-end tokens may
differ.

## `AutoModel` wraps the base class

The `architectures` field in `config.json` tells you which class
`AutoModel.from_pretrained` returns. For LM-headed wrappers
(`<X>ForCausalLM`, `<X>ForMaskedLM`, etc.) the returned object has
the `lm_head` AND its `.forward()` returns a `*Output` dataclass with
a `.logits` field — not `.last_hidden_state`. Check `type(model)` and
read the model's `.forward` signature once.

## `num_logits_to_keep=0` slices the full sequence, not zero

Many HF causal-LM heads default to `num_logits_to_keep=0`, which is
then used as `hidden[:, -0:, :]`. Because `-0 == 0` in Python, this is
`hidden[:, 0:, :]` = the full sequence. So the default returns
all-position logits, not just the last. (No action needed; just don't
be confused.)

## `forward_hook` payload shape varies

For most simple `nn.Linear` / `nn.LayerNorm` modules the hook output is
a single tensor. For attention blocks the output may be a tuple of
`(attn_output, attn_weights, past_kv)`. Always check
`isinstance(out, tuple)` and pick `out[0]`:

```python
def hook(name):
    def f(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        captures[name] = t.detach().float().cpu().numpy()
    return f
```

Hooks only fire at module boundaries — for ops that don't have a
dedicated `nn.Module` (RoPE'd q/k, raw SDPA output, fused silu*up), do
a small "manual layer-0 replay" in `dump_ref.py` that reuses the
captured `q_proj_0` etc. and re-applies the math.

## `torch_dtype=torch.bfloat16` on CPU is slow

A 7B bf16 forward over L=25 tokens takes ~15–60 s on a modern CPU
(depending on how well the BLAS path handles bf16). Don't be
surprised. The whole point of this skill is the C reference; speed
comes later.

## `model.config` vs `tokenizer.special_tokens_map` (stop tokens)

The eos used by `model.generate()` (=
`model.generation_config.eos_token_id`) is **not always the same as**
`tokenizer.eos_token_id`. For Dream they're both 151643; for some
Qwen variants the generation eos is `<|im_end|>` (151645) while the
tokenizer eos is `<|endoftext|>` (151643). The **C reference's stop
condition** must match whatever the Python sampler actually stops on.
Read the relevant `generation_utils.py` / `_sample` path.

For gpt-oss the stop is `<|return|>` (id 200002), not
`<|endoftext|>`. For Qwen with chatml templates it's `<|im_end|>` (or
`<|endoftext|>`, depending on the chat template).

## `trust_remote_code=True` loads model code from the model dir

For models like Dream whose architecture isn't in mainline
transformers, `modeling_<name>.py` lives in the HF model directory
(alongside `config.json`) and is imported on load. To find the class
definitions referenced by `dump_ref.py.template` (e.g.,
`apply_rotary_pos_emb`, `repeat_kv`), use:

```python
cls = type(model)
modeling_mod = sys.modules[cls.__module__]
apply_rotary_pos_emb = getattr(modeling_mod, "apply_rotary_pos_emb")
repeat_kv             = getattr(modeling_mod, "repeat_kv")
```

## `do_sample=False` + `temperature=0.0` warning is benign

HF prints "temperature is set to 0.0 but do_sample=False" but the
generation_config in some shipped models has both set, so suppress or
ignore the warning. The effective behaviour with `do_sample=False` is
greedy, regardless of temperature.

## Multi-modal `config.json` parsing

For multi-modal models the relevant LM dimensions live inside a
nested object:

```json
{
  "model_type": "qwen3_5_moe",
  "text_config":   { "num_hidden_layers": 40, "hidden_size": 2048, ... },
  "vision_config": { "num_hidden_layers": 27, "hidden_size": 1152, ... }
}
```

If you grep flatly for `num_hidden_layers` you'll get the wrong block.
With the starter `utils/json.h` this is just:

```c
json_doc_t* doc = json_open_file(path, &err);
json_val_t  root = json_root(doc);
json_val_t  lm   = json_get(root, "text_config");
if (json_is_null(lm)) lm = root;       // text-only model

int n_layers = (int)json_int_or(json_get(lm, "num_hidden_layers"), 32);
float theta  = (float)json_double_or(
    json_path(lm, "rope_parameters.rope_theta"), 10000.0);
```

`json_get` short-circuits safely through missing keys, so chained
accesses on absent paths yield the default rather than crashing.

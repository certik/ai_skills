#!/usr/bin/env python3
"""Convert HF tokenizer.json to a compact binary blob used by src-metal/tokenizer.c.

Output format (little-endian):
  magic  : "LLMBPETK\1\0\0\0"        (12 bytes)
  u32    n_vocab                      (199998 BPE entries; rank == id)
  u32    n_special                    (named special tokens)
  u32    bytes_blob_size
  u32    specials_blob_size
  u32[n_vocab+1]   vocab_offsets      (into bytes_blob; sentinel last = bytes_blob_size)
  u8 [bytes_blob_size]   bytes_blob   (raw bytes of every BPE token, concatenated)
  records   n_special × { u32 id; u32 name_off; u32 name_len }
  u8 [specials_blob_size] specials_blob   (UTF-8 names of special tokens)
"""
from __future__ import annotations
import argparse
import json
import os
import struct
import sys
from typing import Dict


MAGIC = b"LLMBPETK\x01\x00\x00\x00"  # 12 bytes total (matches tokenizer.c reader)


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 byte-level encoding: 256 bytes -> printable unicode chars."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


# [MODEL] Optional: hard-coded special tokens to include in the .bin even
# if they don't appear in tokenizer.json's added_tokens.  Useful only when
# the reference framework uses special tokens that aren't declared in HF
# tokenizer.json (rare; gpt-oss/harmony is the typical example).
#
# Default: empty.  The script will derive specials from
# tokenizer.json["added_tokens"] automatically.
#
# Example for gpt-oss (uncomment to use):
# EXTRA_SPECIALS = [
#     ("<|startoftext|>", 199998),
#     ("<|endoftext|>", 199999),
#     ("<|return|>", 200002),
#     ("<|channel|>", 200005),
#     ("<|start|>", 200006),
#     ("<|end|>", 200007),
#     ("<|message|>", 200008),
#     ("<|call|>", 200012),
# ]
EXTRA_SPECIALS: list = []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-json", default="/path/to/model/tokenizer.json")
    p.add_argument("--out", default="src-cpu/tokenizer.bin")
    args = p.parse_args()

    print(f"loading {args.tokenizer_json}", flush=True)
    with open(args.tokenizer_json) as f:
        tj = json.load(f)

    vocab = tj["model"]["vocab"]                      # str -> int rank
    added = tj.get("added_tokens", [])

    # Invert byte_to_unicode so we can decode vocab keys to raw bytes.
    b2u = bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}

    n_base = len(vocab)
    print(f"  base vocab entries: {n_base}")

    # Build raw-bytes for each id 0..n_base-1.
    by_id: Dict[int, bytes] = {}
    for key, rank in vocab.items():
        try:
            raw = bytes(u2b[ch] for ch in key)
        except KeyError as e:
            print(f"unknown byte-level char {e!r} in vocab key {key!r}", file=sys.stderr)
            return 1
        by_id[int(rank)] = raw

    # Validate dense ids 0..n_base-1.
    missing = [i for i in range(n_base) if i not in by_id]
    if missing:
        print(f"vocab is not dense: missing {missing[:10]}", file=sys.stderr)
        return 1

    # Build bytes_blob with offsets.
    offsets = []
    blob = bytearray()
    for i in range(n_base):
        offsets.append(len(blob))
        blob.extend(by_id[i])
    offsets.append(len(blob))  # sentinel

    # Specials list: pull from tokenizer.json's added_tokens.  Add any
    # EXTRA_SPECIALS the user configured (rare; useful when the reference
    # uses tokens that aren't declared in HF tokenizer.json).
    specials = {}
    for tok in added:
        sid = int(tok["id"])
        name = tok["content"]
        specials[sid] = name
    for name, sid in EXTRA_SPECIALS:
        if sid not in specials:
            specials[sid] = name
    spec_items = sorted(specials.items())

    sp_blob = bytearray()
    sp_records = []
    for sid, name in spec_items:
        nb = name.encode("utf-8")
        off = len(sp_blob)
        sp_blob.extend(nb)
        sp_records.append((sid, off, len(nb)))

    # Write file.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", n_base, len(sp_records), len(blob), len(sp_blob)))
        f.write(struct.pack(f"<{len(offsets)}I", *offsets))
        f.write(blob)
        for (sid, off, nl) in sp_records:
            f.write(struct.pack("<III", sid, off, nl))
        f.write(sp_blob)

    sz = os.path.getsize(args.out)
    print(f"wrote {args.out}  ({sz/1e6:.2f} MB)")
    print(f"  n_vocab = {n_base}, n_special = {len(sp_records)}")
    print(f"  bytes_blob = {len(blob)}, specials_blob = {len(sp_blob)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

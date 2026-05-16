#!/usr/bin/env python3
"""Convert HF tokenizer.json to a compact binary blob used by csrc/tokenizer.c.

Output format (little-endian):
  magic  : "GPTOSSTOK\1\0\0\0"        (12 bytes)
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


MAGIC = b"GPTOSSTOK\x01\x00\x00"


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


# Hard-coded specials from openai/harmony src/tiktoken_ext/public_encodings.rs.
HARMONY_SPECIALS = [
    ("<|startoftext|>", 199998),
    ("<|endoftext|>", 199999),
    ("<|reserved_200000|>", 200000),
    ("<|reserved_200001|>", 200001),
    ("<|return|>", 200002),
    ("<|constrain|>", 200003),
    ("<|reserved_200004|>", 200004),
    ("<|channel|>", 200005),
    ("<|start|>", 200006),
    ("<|end|>", 200007),
    ("<|message|>", 200008),
    ("<|reserved_200009|>", 200009),
    ("<|reserved_200010|>", 200010),
    ("<|reserved_200011|>", 200011),
    ("<|call|>", 200012),
    ("<|reserved_200013|>", 200013),
    # Named specials that appear in HF tokenizer.json beyond the harmony core set:
    ("<|refusal|>", 200013),  # alias used by harmony FormattingToken::Refusal -> placeholder
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-json", default="/Users/ondrej/repos/models/gpt-oss-20b/tokenizer.json")
    p.add_argument("--out", default="csrc/tokenizer.bin")
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

    # Specials list: dedup by id, prefer harmony names.
    specials = {}
    for name, sid in HARMONY_SPECIALS:
        specials[sid] = name
    # Pull in any extra named specials from added_tokens that aren't covered.
    for tok in added:
        sid = int(tok["id"])
        name = tok["content"]
        if sid not in specials and name.startswith("<|") and name.endswith("|>"):
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

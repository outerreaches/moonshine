#!/usr/bin/env python3
"""Emit the frozen eval corpus as token ids, for tools/mimo26_eval.

Kept separate from the scorer so the corpus is tokenized by the checkpoint's
own tokenizer rather than by anything this lane wrote, and so the exact id
sequence a run consumed can be captured alongside its results.
"""
import json
import os
import pathlib
import sys

SPEC = pathlib.Path(__file__).with_name("mimo26_eval_spec.json")


def main():
    root = pathlib.Path(os.environ.get(
        "MIMO26_ROOT",
        "/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL"))
    import tokenizers
    tokenizer = tokenizers.Tokenizer.from_file(str(root / "tokenizer.json"))
    spec = json.loads(SPEC.read_text())
    ids = tokenizer.encode(spec["text"], add_special_tokens=False).ids
    print(" ".join(str(i) for i in ids))
    print(f"{len(ids)} tokens from the corpus frozen {spec['frozen']}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

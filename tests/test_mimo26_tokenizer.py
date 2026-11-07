#!/usr/bin/env python3
"""Verify the pinned tokenizer fixture still describes the checkpoint.

Re-derives every fixture entry from the checkpoint and compares. Catches a
changed tokenizer, a changed chat template, or a library upgrade that alters
segmentation. Skips cleanly when the checkpoint is absent so the suite still
runs on hosts without weights.

  MIMO26_ROOT=/path/to/checkpoint python3 tests/test_mimo26_tokenizer.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "mimo26_tokenizer_v1.json"

sys.path.insert(0, str(HERE))

DEFAULT_ROOT = "/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL"

# The embedding and lm_head carry 152576 rows while the tokenizer defines
# fewer. The surplus rows decode to no token, so sampling must exclude them.
EMBEDDING_ROWS = 152576


def main():
    root = Path(os.environ.get("MIMO26_ROOT", DEFAULT_ROOT))
    if not (root / "tokenizer.json").exists():
        print(f"skip: no checkpoint at {root}")
        return 0
    if not FIXTURE.exists():
        print(f"fail: missing fixture {FIXTURE}", file=sys.stderr)
        return 1

    import generate_mimo26_tokenizer_fixture as generator

    fixture = json.loads(FIXTURE.read_text())
    failures = []

    # Pinned files must be byte-identical to what the fixture was built from.
    import hashlib
    for name, digest in fixture["file_sha256"].items():
        path = root / name
        if not path.exists():
            failures.append(f"{name}: missing from checkpoint")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"{name}: sha256 {actual} != pinned {digest}")
    if not failures:
        print(f"  ok  {len(fixture['file_sha256'])} pinned files hash-match")

    import jinja2
    import tokenizers

    tokenizer = tokenizers.Tokenizer.from_file(str(root / "tokenizer.json"))

    vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab != fixture["vocab_size_from_tokenizer"]:
        failures.append(f"vocab size {vocab} != pinned "
                        f"{fixture['vocab_size_from_tokenizer']}")
    else:
        print(f"  ok  tokenizer vocab {vocab}")

    # The unused embedding tail is a sampling hazard, so assert it stays known.
    surplus = EMBEDDING_ROWS - vocab
    if surplus < 0:
        failures.append(f"tokenizer vocab {vocab} exceeds embedding rows "
                        f"{EMBEDDING_ROWS}")
    else:
        print(f"  ok  {surplus} embedding rows decode to no token "
              f"(ids {vocab}..{EMBEDDING_ROWS - 1} must be masked when sampling)")

    for token, expected in fixture["special_token_ids"].items():
        actual = tokenizer.token_to_id(token)
        if actual != expected:
            failures.append(f"special {token}: id {actual} != pinned {expected}")
    if not any("special" in f for f in failures):
        print(f"  ok  {len(fixture['special_token_ids'])} special token ids")

    for label, entry in fixture["texts"].items():
        ids = tokenizer.encode(entry["text"], add_special_tokens=False).ids
        if ids != entry["ids"]:
            failures.append(f"text {label}: {len(ids)} ids != pinned "
                            f"{len(entry['ids'])}")
    if not any("text " in f for f in failures):
        print(f"  ok  {len(fixture['texts'])} text encodings")

    template_source = (root / "chat_template.jinja").read_text()
    environment = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        trim_blocks=False,
        lstrip_blocks=False,
        undefined=jinja2.StrictUndefined,
    )
    environment.filters["tojson"] = lambda value, ensure_ascii=True: json.dumps(
        value, ensure_ascii=ensure_ascii, separators=(", ", ": "))
    environment.filters["items"] = lambda value: list(value.items())
    template = environment.from_string(template_source)

    for label, entry in fixture["chats"].items():
        case = entry["request"]
        rendered = template.render(
            messages=case["messages"],
            tools=case.get("tools"),
            add_generation_prompt=case.get("add_generation_prompt", False),
            enable_thinking=case.get("enable_thinking", True),
        )
        if rendered != entry["rendered"]:
            failures.append(f"chat {label}: rendering changed")
            continue
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        if ids != entry["ids"]:
            failures.append(f"chat {label}: token ids changed")
    if not any("chat " in f for f in failures):
        print(f"  ok  {len(fixture['chats'])} chat renderings and encodings")

    # Structural expectations the port depends on, checked against the fixture
    # rather than assumed in code.
    plain = fixture["chats"]["plain_single_turn"]["rendered"]
    if not plain.endswith("<|im_start|>assistant\n"):
        failures.append("generation prompt does not end the assistant header")
    disabled = fixture["chats"]["thinking_disabled"]["rendered"]
    if not disabled.endswith("<think></think>"):
        failures.append("enable_thinking=False must append an empty think block")
    multi = fixture["chats"]["multi_turn"]["rendered"]
    if "<think></think>First answer." not in multi:
        failures.append("replayed assistant turns must keep an empty think block")
    reasoning = fixture["chats"]["assistant_with_reasoning"]["rendered"]
    if "<think>91 = 7 * 13, so it is composite.</think>" not in reasoning:
        failures.append("reasoning_content must render inside the think block")
    calls = fixture["chats"]["tool_call_and_result"]["rendered"]
    if "<tool_call><function=get_weather>" not in calls or \
            "<parameter=city>Ottawa</parameter>" not in calls:
        failures.append("tool calls must use the parameter tag form")
    eos = fixture["generation_config"]["eos_token_id"]
    if not isinstance(eos, list) or len(eos) != 3:
        failures.append(f"eos_token_id should be three ids, got {eos}")
    if not failures:
        print("  ok  structural expectations (generation prompt, think block, "
              "tool-call form, three eos ids)")

    if failures:
        print("\nfailures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("test_mimo26_tokenizer: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

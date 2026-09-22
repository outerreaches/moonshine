#!/usr/bin/env python3
"""Pin MiMo V2.6 Flash tokenizer, chat template and generation inputs.

Uses the `tokenizers` library against tokenizer.json and renders
chat_template.jinja with jinja2 directly. No `trust_remote_code`, no model
weights, no remote code execution of any kind.

These fixtures pin *rendering and token IDs*. They are not evidence that tool
calling, reasoning separation or multimodal input are supported.

  python3 generate_mimo26_tokenizer_fixture.py ROOT [--out PATH]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import jinja2
import tokenizers

PINNED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
)

# Plain strings exercising ASCII, whitespace, CJK, emoji with combining marks,
# RTL text and the special-token surface forms.
TEXT_CASES = {
    "ascii_short": "Hello, world!",
    "ascii_sentence": "The quick brown fox jumps over the lazy dog.",
    "leading_space": " leading space",
    "double_newline": "line one\n\nline three",
    "tabs": "a\tb\tc",
    "cjk": "你好，世界",
    "cjk_mixed": "MiMo 模型 v2.6",
    "emoji_zwj": "\U0001f469‍\U0001f4bb ships \U0001f680",
    "combining": "égalité",
    "rtl": "שלום עולם",
    "digits": "3.14159 and 1e-9 and 0x1F",
    "special_token_surface": "<|im_start|>not really a turn<|im_end|>",
    "repeated_spaces": "a     b",
    "empty": "",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    }
]

CHAT_CASES = {
    "plain_single_turn": {
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "add_generation_prompt": True,
    },
    "plain_no_generation_prompt": {
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "add_generation_prompt": False,
    },
    "thinking_disabled": {
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    "with_system": {
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Define entropy."},
        ],
        "add_generation_prompt": True,
    },
    "multi_turn": {
        "messages": [
            {"role": "user", "content": "First question."},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Second question."},
        ],
        "add_generation_prompt": True,
    },
    "assistant_with_reasoning": {
        "messages": [
            {"role": "user", "content": "Is 91 prime?"},
            {
                "role": "assistant",
                "content": "No, 91 is 7 times 13.",
                "reasoning_content": "91 = 7 * 13, so it is composite.",
            },
            {"role": "user", "content": "And 97?"},
        ],
        "add_generation_prompt": True,
    },
    "unicode_turn": {
        "messages": [{"role": "user", "content": "翻译：\U0001f680 → ?"}],
        "add_generation_prompt": True,
    },
    "content_parts": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part one. "},
                    {"type": "text", "text": "Part two."},
                ],
            }
        ],
        "add_generation_prompt": True,
    },
    "tools_declared": {
        "messages": [{"role": "user", "content": "Weather in Ottawa?"}],
        "tools": TOOLS,
        "add_generation_prompt": True,
    },
    "tool_call_and_result": {
        "messages": [
            {"role": "user", "content": "Weather in Ottawa?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Ottawa", "unit": "c"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "{\"temp_c\": -3}"},
        ],
        "tools": TOOLS,
        "add_generation_prompt": True,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "fixtures" / "mimo26_tokenizer_v1.json")

    hashes = {}
    for name in PINNED_FILES:
        path = root / name
        if not path.exists():
            print(f"missing pinned file: {name}", file=sys.stderr)
            return 2
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    tokenizer = tokenizers.Tokenizer.from_file(str(root / "tokenizer.json"))
    config = json.loads((root / "tokenizer_config.json").read_text())
    generation = json.loads((root / "generation_config.json").read_text())

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

    texts = {}
    for label, text in TEXT_CASES.items():
        encoded = tokenizer.encode(text, add_special_tokens=False)
        texts[label] = {
            "text": text,
            "ids": encoded.ids,
            "count": len(encoded.ids),
        }

    chats = {}
    for label, case in CHAT_CASES.items():
        rendered = template.render(
            messages=case["messages"],
            tools=case.get("tools"),
            add_generation_prompt=case.get("add_generation_prompt", False),
            enable_thinking=case.get("enable_thinking", True),
        )
        encoded = tokenizer.encode(rendered, add_special_tokens=False)
        chats[label] = {
            "request": case,
            "rendered": rendered,
            "ids": encoded.ids,
            "count": len(encoded.ids),
        }

    specials = {}
    for token in ("<|im_start|>", "<|im_end|>", "<think>", "</think>",
                  "<|endoftext|>", "<tool_call>", "<|vision_start|>",
                  "<|image_pad|>", "<|vision_end|>", "<|mimo_audio_start|>",
                  "<|audio_pad|>", "<|mimo_audio_end|>", "<|video_pad|>"):
        specials[token] = tokenizer.token_to_id(token)

    fixture = {
        "revision": "3b38d063180c3e4aed9691fdc735f3d10b266ee4",
        "generated_with": {
            "tokenizers": tokenizers.__version__,
            "jinja2": jinja2.__version__,
        },
        "file_sha256": hashes,
        "vocab_size_from_tokenizer": tokenizer.get_vocab_size(with_added_tokens=True),
        "model_max_length": config.get("model_max_length"),
        "eos_token": config.get("eos_token"),
        "pad_token": config.get("pad_token"),
        "generation_config": {
            "bos_token_id": generation.get("bos_token_id"),
            "eos_token_id": generation.get("eos_token_id"),
            "temperature": generation.get("temperature"),
            "top_p": generation.get("top_p"),
            "do_sample": generation.get("do_sample"),
            "max_new_tokens": generation.get("max_new_tokens"),
        },
        "special_token_ids": specials,
        "texts": texts,
        "chats": chats,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    print(f"  vocab (with added)  {fixture['vocab_size_from_tokenizer']}")
    print(f"  eos_token_id        {fixture['generation_config']['eos_token_id']}")
    print(f"  text cases          {len(texts)}")
    print(f"  chat cases          {len(chats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

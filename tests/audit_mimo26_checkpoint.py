#!/usr/bin/env python3
"""Deterministic read-only audit of an official MiMo-V2.6-Flash checkpoint.

Validates the full tensor schema against an independent encoding of the same
contract held in mimo26_architecture.c, so the two disagree loudly if either
drifts. Header-only: no tensor payload is read and nothing is written to the
checkpoint.

Checks, each reported separately:
  * index/disk association -- referenced shards exist, no unreferenced shards
  * duplicate tensor names, and duplicate JSON keys inside a header
  * shard path traversal or absolute paths in the index
  * offsets: ordering, bounds against file size, and byte length vs dtype/shape
  * intra-shard span overlap
  * per-tensor name/dtype/shape contract, per layer kind
  * exact group coverage: main text, experts, MTP, vision, audio

Exit status is 0 only when every check passes. Machine-readable results go to
the --report path.
"""
import argparse
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

TEXT_LAYER_COUNT = 48
MOE_LAYER_COUNT = 47
EXPERTS_PER_LAYER = 256
HIDDEN_SIZE = 4096
VOCAB_SIZE = 152576

MAIN_TENSOR_COUNT = 382
EXPERT_TENSOR_COUNT = 72192
MTP_TENSOR_COUNT = 48
VISION_TENSOR_COUNT = 364
AUDIO_ENCODER_TENSOR_COUNT = 75
SPEECH_EMBEDDING_COUNT = 20
TOTAL_TENSOR_COUNT = 73081

# MTP layers are numbered 0..2 in their own namespace; they do not continue
# the text layer numbering the way GLM's do.
MTP_LAYER_COUNT = 3

# Vision and audio schemas are deliberately unvalidated: the first release is
# text-only, so these groups are counted for coverage and recognized as
# out-of-scope rather than contract-checked.
MODALITY_GROUPS = ("vision", "audio")

# config hybrid_layer_pattern: 1 is sliding window, 0 is full attention.
# Not periodic: the first gap is five, every later gap is six.
SWA_PATTERN = [
    0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
]

DTYPE_BYTES = {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1, "I8": 1}

GLOBALS = {
    "lm_head.weight": ("BF16", [VOCAB_SIZE, HIDDEN_SIZE]),
    "model.embed_tokens.weight": ("BF16", [VOCAB_SIZE, HIDDEN_SIZE]),
    "model.norm.weight": ("BF16", [HIDDEN_SIZE]),
}
COMMON = {
    "input_layernorm.weight": ("BF16", [4096]),
    "post_attention_layernorm.weight": ("BF16", [4096]),
}
DENSE_MLP = {
    "mlp.down_proj.weight": ("F8_E4M3", [4096, 16384]),
    "mlp.down_proj.weight_scale_inv": ("F32", [32, 128]),
    "mlp.gate_proj.weight": ("F8_E4M3", [16384, 4096]),
    "mlp.gate_proj.weight_scale_inv": ("F32", [128, 32]),
    "mlp.up_proj.weight": ("F8_E4M3", [16384, 4096]),
    "mlp.up_proj.weight_scale_inv": ("F32", [128, 32]),
}
MOE_GATE = {
    "mlp.gate.e_score_correction_bias": ("F32", [EXPERTS_PER_LAYER]),
    "mlp.gate.weight": ("BF16", [EXPERTS_PER_LAYER, HIDDEN_SIZE]),
}
# Packed byte extents, not logical parameter shapes.
EXPERT_PROJ = {
    "down_proj.weight": ("U8", [4096, 1024]),
    "down_proj.weight_scale": ("U8", [4096, 64]),
    "gate_proj.weight": ("U8", [2048, 2048]),
    "gate_proj.weight_scale": ("U8", [2048, 128]),
    "up_proj.weight": ("U8", [2048, 2048]),
    "up_proj.weight_scale": ("U8", [2048, 128]),
}
# Global scale grid is [108,32] although ceil(13568/128) is 106: the spare two
# block rows hold live values from a 13824-row layout that was later sliced.
ATTN_GLOBAL = {
    "self_attn.o_proj.weight": ("BF16", [4096, 8192]),
    "self_attn.qkv_proj.weight": ("F8_E4M3", [13568, 4096]),
    "self_attn.qkv_proj.weight_scale_inv": ("F32", [108, 32]),
}
ATTN_SWA = {
    "self_attn.attention_sink_bias": ("BF16", [64]),
    "self_attn.o_proj.weight": ("BF16", [4096, 8192]),
    "self_attn.qkv_proj.weight": ("F8_E4M3", [14848, 4096]),
    "self_attn.qkv_proj.weight_scale_inv": ("F32", [116, 32]),
}
MTP = {
    "eh_proj.weight": ("BF16", [4096, 8192]),
    "enorm.weight": ("BF16", [4096]),
    "final_layernorm.weight": ("BF16", [4096]),
    "hnorm.weight": ("BF16", [4096]),
    "input_layernorm.weight": ("BF16", [4096]),
    "mlp.down_proj.weight": ("F8_E4M3", [4096, 16384]),
    "mlp.down_proj.weight_scale_inv": ("F32", [32, 128]),
    "mlp.gate_proj.weight": ("F8_E4M3", [16384, 4096]),
    "mlp.gate_proj.weight_scale_inv": ("F32", [128, 32]),
    "mlp.up_proj.weight": ("F8_E4M3", [16384, 4096]),
    "mlp.up_proj.weight_scale_inv": ("F32", [128, 32]),
    "pre_mlp_layernorm.weight": ("BF16", [4096]),
    "self_attn.attention_sink_bias": ("BF16", [64]),
    "self_attn.o_proj.weight": ("BF16", [4096, 8192]),
    "self_attn.qkv_proj.weight": ("F8_E4M3", [14848, 4096]),
    "self_attn.qkv_proj.weight_scale_inv": ("F32", [116, 32]),
}

CANONICAL_INDEX = re.compile(r"^(?:0|[1-9][0-9]*)$")
LAYER_RE = re.compile(r"^model\.layers\.([^.]+)\.(.+)$")
EXPERT_RE = re.compile(r"^mlp\.experts\.([^.]+)\.(.+)$")
MTP_RE = re.compile(r"^model\.mtp\.layers\.([^.]+)\.(.+)$")


def layer_kind(layer):
    if not 0 <= layer < TEXT_LAYER_COUNT:
        return "invalid"
    if layer == 0:
        return "dense_global"
    return "moe_swa" if SWA_PATTERN[layer] else "moe_global"


def parse_index(text, limit):
    if not CANONICAL_INDEX.match(text):
        return None
    value = int(text)
    return value if value <= limit else None


def classify(name):
    """Return (group, layer_kind_or_None). group is one of
    main/expert/mtp/vision/audio/unknown."""
    if name.startswith(("speech_embeddings.", "audio_encoder.")):
        return "audio", None
    if "vision" in name or name.startswith("visual."):
        return "vision", None
    if name.startswith("model.mtp."):
        return "mtp", None
    if name in GLOBALS:
        return "main", "global_tensor"
    match = LAYER_RE.match(name)
    if match:
        layer = parse_index(match.group(1), TEXT_LAYER_COUNT - 1)
        if layer is None:
            return "unknown", None
        kind = layer_kind(layer)
        if EXPERT_RE.match(match.group(2)):
            return "expert", kind
        return "main", kind
    return "unknown", None


def check_contract(name, dtype, shape):
    """Return None when the tensor matches its contract, else a reason."""
    if name in GLOBALS:
        want_dtype, want_shape = GLOBALS[name]
        if dtype != want_dtype or shape != want_shape:
            return f"expected {want_dtype}{want_shape}"
        return None

    match = MTP_RE.match(name)
    if match:
        layer = parse_index(match.group(1), MTP_LAYER_COUNT - 1)
        if layer is None:
            return "MTP layer index out of range"
        entry = MTP.get(match.group(2))
        if entry is None:
            return "unknown MTP suffix"
        if dtype != entry[0] or shape != entry[1]:
            return f"expected {entry[0]}{entry[1]}"
        return None

    match = LAYER_RE.match(name)
    if not match:
        return "name does not match any known namespace"
    layer = parse_index(match.group(1), TEXT_LAYER_COUNT - 1)
    if layer is None:
        return "non-canonical or out-of-range layer index"
    kind = layer_kind(layer)
    suffix = match.group(2)

    expert = EXPERT_RE.match(suffix)
    if expert:
        if kind == "dense_global":
            return "dense layer 0 must not carry routed experts"
        if parse_index(expert.group(1), EXPERTS_PER_LAYER - 1) is None:
            return "non-canonical or out-of-range expert index"
        entry = EXPERT_PROJ.get(expert.group(2))
        if entry is None:
            return "unknown expert projection suffix"
        if dtype != entry[0] or shape != entry[1]:
            return f"expected {entry[0]}{entry[1]}"
        return None

    tables = [COMMON]
    tables.append(DENSE_MLP if kind == "dense_global" else MOE_GATE)
    tables.append(ATTN_SWA if kind == "moe_swa" else ATTN_GLOBAL)
    for table in tables:
        entry = table.get(suffix)
        if entry is not None:
            if dtype != entry[0] or shape != entry[1]:
                return f"expected {entry[0]}{entry[1]}"
            return None
    return f"suffix not valid for {kind} layer"


def read_header(path):
    """Return (header_dict, duplicate_keys, header_end)."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"{path.name}: truncated header")
    duplicates = []

    def hook(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        return dict(pairs)

    return json.loads(raw, object_pairs_hook=hook), duplicates, 8 + length


def audit(root):
    root = Path(root)
    problems = defaultdict(list)
    index_path = root / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]

    referenced = sorted(set(weight_map.values()))
    on_disk = sorted(p.name for p in root.glob("*.safetensors"))
    for shard in referenced:
        if Path(shard).is_absolute() or ".." in Path(shard).parts or "/" in shard:
            problems["shard_path_unsafe"].append(shard)
    for shard in sorted(set(referenced) - set(on_disk)):
        problems["referenced_shard_missing"].append(shard)
    for shard in sorted(set(on_disk) - set(referenced)):
        problems["shard_not_referenced"].append(shard)

    groups = Counter()
    seen_names = {}
    total = 0
    for shard in on_disk:
        path = root / shard
        file_bytes = path.stat().st_size
        header, duplicates, header_end = read_header(path)
        for key in duplicates:
            problems["duplicate_json_key"].append(f"{shard}:{key}")

        spans = []
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            total += 1
            if name in seen_names:
                problems["duplicate_tensor_name"].append(
                    f"{name} in {seen_names[name]} and {shard}")
            seen_names[name] = shard
            if weight_map.get(name) != shard:
                problems["index_association_wrong"].append(
                    f"{name}: index says {weight_map.get(name)}, found in {shard}")

            dtype = entry["dtype"]
            shape = list(entry["shape"])
            start, end = entry["data_offsets"]

            if not isinstance(start, int) or not isinstance(end, int):
                problems["offset_not_integer"].append(name)
                continue
            if start < 0 or end < start:
                problems["offset_range_invalid"].append(name)
                continue
            if header_end + end > file_bytes:
                problems["offset_past_end_of_file"].append(name)
            width = DTYPE_BYTES.get(dtype)
            if width is None:
                problems["unknown_dtype"].append(f"{name}:{dtype}")
            else:
                elements = 1
                overflow = False
                for dim in shape:
                    if dim < 0 or (dim and elements > (1 << 62) // dim):
                        overflow = True
                        break
                    elements *= dim
                if overflow:
                    problems["shape_overflow"].append(name)
                elif elements * width != end - start:
                    problems["byte_length_mismatch"].append(
                        f"{name}: {end - start} bytes for {dtype}{shape}")
            spans.append((start, end, name))

            group, _ = classify(name)
            if group == "unknown":
                problems["unclassified_tensor"].append(name)
                continue
            groups[group] += 1
            if group in MODALITY_GROUPS:
                continue
            reason = check_contract(name, dtype, shape)
            if reason is not None:
                problems["contract_violation"].append(f"{name}: {reason}")

        spans.sort()
        for earlier, later in zip(spans, spans[1:]):
            if earlier[1] > later[0]:
                problems["span_overlap"].append(
                    f"{shard}: {earlier[2]} overlaps {later[2]}")

    coverage = {
        "main": (groups["main"], MAIN_TENSOR_COUNT),
        "expert": (groups["expert"], EXPERT_TENSOR_COUNT),
        "mtp": (groups["mtp"], MTP_TENSOR_COUNT),
        "vision": (groups["vision"], VISION_TENSOR_COUNT),
        "audio": (groups["audio"],
                  AUDIO_ENCODER_TENSOR_COUNT + SPEECH_EMBEDDING_COUNT),
        "total": (total, TOTAL_TENSOR_COUNT),
    }
    for group, (found, want) in coverage.items():
        if found != want:
            problems["coverage_mismatch"].append(
                f"{group}: found {found}, expected {want}")

    # Every (layer, expert) pair must be complete: six tensors, no gaps.
    expert_tally = Counter()
    for name in seen_names:
        match = LAYER_RE.match(name)
        if not match:
            continue
        expert = EXPERT_RE.match(match.group(2))
        if expert:
            expert_tally[(match.group(1), expert.group(1))] += 1
    if len(expert_tally) != MOE_LAYER_COUNT * EXPERTS_PER_LAYER:
        problems["expert_pair_count"].append(
            f"{len(expert_tally)} distinct (layer,expert) pairs, expected "
            f"{MOE_LAYER_COUNT * EXPERTS_PER_LAYER}")
    incomplete = {f"{k[0]}/{k[1]}": v for k, v in expert_tally.items() if v != 6}
    if incomplete:
        sample = dict(list(sorted(incomplete.items()))[:8])
        problems["incomplete_expert"].append(
            f"{len(incomplete)} experts without exactly 6 tensors, e.g. {sample}")

    swa_layers = [i for i, v in enumerate(SWA_PATTERN) if v == 1]
    report = {
        "root": str(root),
        "shards_referenced": len(referenced),
        "shards_on_disk": len(on_disk),
        "tensors_seen": total,
        "coverage": {k: {"found": f, "expected": e}
                     for k, (f, e) in coverage.items()},
        "swa_layer_count": len(swa_layers),
        "global_layers": [i for i, v in enumerate(SWA_PATTERN) if v == 0],
        "pattern_is_period_12": all(
            SWA_PATTERN[i] == SWA_PATTERN[i % 12] for i in range(48)),
        "problems": {k: v for k, v in sorted(problems.items())},
        "problem_count": sum(len(v) for v in problems.values()),
        "status": "pass" if not problems else "fail",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="checkpoint directory")
    parser.add_argument("--report", help="write JSON results here")
    parser.add_argument("--max-listed", type=int, default=10,
                        help="problems printed per category")
    args = parser.parse_args()

    report = audit(args.root)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))

    print(f"root                {report['root']}")
    print(f"shards              {report['shards_on_disk']} on disk, "
          f"{report['shards_referenced']} referenced")
    print(f"tensors             {report['tensors_seen']}")
    print(f"SWA layers          {report['swa_layer_count']}")
    print(f"global layers       {report['global_layers']}")
    print(f"pattern period-12   {report['pattern_is_period_12']}")
    print("coverage")
    for group, counts in report["coverage"].items():
        mark = "ok" if counts["found"] == counts["expected"] else "MISMATCH"
        print(f"  {group:<10} {counts['found']:>6} / {counts['expected']:<6} {mark}")
    if report["problems"]:
        print("problems")
        for category, entries in report["problems"].items():
            print(f"  {category} ({len(entries)})")
            for entry in entries[:args.max_listed]:
                print(f"    {entry}")
            if len(entries) > args.max_listed:
                print(f"    ... {len(entries) - args.max_listed} more")
    else:
        print("problems            none")
    print(f"\nstatus: {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

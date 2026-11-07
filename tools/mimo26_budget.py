#!/usr/bin/env python3
"""Static allocation, expert cache and KV budget for MiMo V2.6 Flash.

Derives every figure from the checkpoint's own headers and the host's measured
memory. Nothing here is a performance prediction: the throughput lines are
arithmetic bounds from an assumed bandwidth, recorded so the performance stage
has a number to be judged against rather than one invented afterwards.

Admission must use measured availability. This tool deliberately refuses to
recommend a cache that depends on swap or on all experts fitting.

  python3 tools/mimo26_budget.py ROOT [--context 8192] [--reserve-gib 12]
"""
import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path

GIB = float(1 << 30)
MIB = float(1 << 20)

TEXT_LAYERS = 48
MOE_LAYERS = 47
EXPERTS_PER_LAYER = 256
EXPERTS_PER_TOKEN = 8
SWA_WINDOW = 128

# Full-attention layers, from config's explicit hybrid_layer_pattern.
GLOBAL_LAYERS = (0, 5, 11, 17, 23, 29, 35, 41, 47)
SWA_LAYERS = tuple(i for i in range(TEXT_LAYERS) if i not in GLOBAL_LAYERS)

QK_DIM = 192
V_DIM = 128
GLOBAL_KV_HEADS = 4
SWA_KV_HEADS = 8
KV_DTYPE_BYTES = 2  # BF16

# Bandwidth used only for the I/O ceiling lines, not measured here.
ASSUMED_SSD_GIB_PER_S = 5.0


def measured_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.strip().split()[0]) * 1024
    return values


def tensor_groups(root):
    root = Path(root)
    weight_map = json.loads(
        (root / "model.safetensors.index.json").read_text())["weight_map"]
    groups = defaultdict(int)
    for shard in sorted(set(weight_map.values())):
        path = root / shard
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            size = end - start
            if ".mlp.experts." in name:
                groups["expert"] += size
            elif name.startswith("model.mtp."):
                groups["mtp"] += size
            elif name.startswith(("speech_embeddings.", "audio_encoder.")):
                groups["audio"] += size
            elif "vision" in name or name.startswith("visual."):
                groups["vision"] += size
            else:
                groups["static_text"] += size
    return groups


def kv_bytes_per_token():
    """Global KV grows with context; SWA rings are bounded."""
    per_global_layer = GLOBAL_KV_HEADS * (QK_DIM + V_DIM) * KV_DTYPE_BYTES
    per_swa_layer = SWA_KV_HEADS * (QK_DIM + V_DIM) * KV_DTYPE_BYTES
    return {
        "global_per_token": per_global_layer * len(GLOBAL_LAYERS),
        "swa_ring_total": per_swa_layer * SWA_WINDOW * len(SWA_LAYERS),
        "per_global_layer": per_global_layer,
        "per_swa_layer": per_swa_layer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--context", type=int, action="append", default=None,
                        help="context lengths to budget; repeatable")
    parser.add_argument("--reserve-gib", type=float, default=12.0,
                        help="memory left for OS, HIP overhead and scratch")
    parser.add_argument("--json", help="write the ledger here")
    args = parser.parse_args()
    contexts = args.context or [512, 2048, 8192, 32768, 65536, 1048576]

    groups = tensor_groups(args.root)
    memory = measured_memory()
    kv = kv_bytes_per_token()

    expert_bytes = (2 * ((2048 * 2048) + (2048 * 128)) +
                    ((4096 * 1024) + (4096 * 64)))
    expert_total = groups["expert"]
    experts_resident_total = MOE_LAYERS * EXPERTS_PER_LAYER

    print("stored payload, from headers")
    for label in ("static_text", "expert", "mtp", "vision", "audio"):
        print(f"  {label:<14} {groups[label] / GIB:>10.4f} GiB")
    print(f"  {'sum':<14} {sum(groups.values()) / GIB:>10.4f} GiB")

    print("\nmeasured host memory")
    print(f"  MemTotal       {memory['MemTotal'] / GIB:>10.4f} GiB")
    print(f"  MemAvailable   {memory['MemAvailable'] / GIB:>10.4f} GiB")
    print(f"  SwapTotal      {memory['SwapTotal'] / GIB:>10.4f} GiB "
          f"(must not be relied on)")

    print("\nper-expert identity")
    print(f"  weights+scales {expert_bytes / MIB:>10.2f} MiB")
    print(f"  experts total  {expert_total / GIB:>10.4f} GiB "
          f"({experts_resident_total} experts)")
    print(f"  per token, all misses "
          f"{MOE_LAYERS * EXPERTS_PER_TOKEN * expert_bytes / GIB:.4f} GiB")

    print("\nKV")
    print(f"  global, per token        {kv['global_per_token']:>10} B "
          f"({len(GLOBAL_LAYERS)} layers)")
    print(f"  SWA rings, fixed total   {kv['swa_ring_total'] / MIB:>10.3f} MiB "
          f"({len(SWA_LAYERS)} layers, window {SWA_WINDOW})")

    reserve = args.reserve_gib * GIB
    print(f"\nbudget at {args.reserve_gib:.1f} GiB reserve, from MemAvailable")
    header = (f"  {'context':>9} {'global KV':>11} {'SWA rings':>10} "
              f"{'static':>9} {'cache room':>11} {'experts':>8} {'hit@room':>9}")
    print(header)
    rows = []
    for context in contexts:
        global_kv = kv["global_per_token"] * context
        fixed = groups["static_text"] + kv["swa_ring_total"] + global_kv
        room = memory["MemAvailable"] - fixed - reserve
        cached = max(0, int(room // expert_bytes))
        share = cached / experts_resident_total
        rows.append({
            "context": context,
            "global_kv_bytes": global_kv,
            "cache_room_bytes": max(0, room),
            "cacheable_experts": cached,
            "resident_fraction": share,
        })
        room_text = f"{room / GIB:.3f}" if room > 0 else "NONE"
        print(f"  {context:>9} {global_kv / GIB:>10.4f}G "
              f"{kv['swa_ring_total'] / MIB:>9.2f}M "
              f"{groups['static_text'] / GIB:>8.3f}G "
              f"{room_text:>11} {cached:>8} {share:>8.1%}")

    print("\nI/O ceiling at an assumed "
          f"{ASSUMED_SSD_GIB_PER_S:.1f} GiB/s (not measured)")
    per_token = MOE_LAYERS * EXPERTS_PER_TOKEN * expert_bytes / GIB
    for hit in (0.0, 0.5, 0.8, 0.9):
        traffic = per_token * (1.0 - hit)
        seconds = traffic / ASSUMED_SSD_GIB_PER_S
        print(f"  hit {hit:>4.0%}  {traffic:>7.4f} GiB/token  "
              f"{seconds * 1000:>8.1f} ms  {1.0 / seconds:>6.2f} tok/s")

    print("\nnotes")
    print("  - Experts cannot all be resident: "
          f"{expert_total / GIB:.1f} GiB against "
          f"{memory['MemTotal'] / GIB:.1f} GiB of RAM. Streaming is mandatory.")
    print("  - Cache room above ignores HIP overhead, aligned staging, pending "
          "leases,\n    verification buffers and SWA rollback storage. Do not "
          "size the cache at\n    this figure; measure first, then choose below it.")
    print("  - The 1048576 row is the upstream advertised limit, not a "
          "qualified capacity.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "stored_bytes": dict(groups),
            "measured_memory_bytes": {k: memory[k] for k in
                                      ("MemTotal", "MemAvailable", "SwapTotal")},
            "expert_bytes": expert_bytes,
            "expert_total_bytes": expert_total,
            "experts_total": experts_resident_total,
            "kv": kv,
            "reserve_bytes": int(reserve),
            "assumed_ssd_gib_per_s": ASSUMED_SSD_GIB_PER_S,
            "contexts": rows,
        }, indent=1) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

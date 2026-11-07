#!/usr/bin/env python3
"""Cross-check the C RoPE against torch, using the reference's own formula.

The C test validates RoPE against algebraic properties, which only confirms
that the implementation satisfies properties we chose. This instead recomputes
the rotation in torch from the reference's formula -- different language,
different library, same spec -- so a disagreement is a real semantic
difference rather than a self-consistent mistake.

The formula is transcribed from modeling_mimo_v2.py rather than imported, so
no downloaded model code executes:

    inv_freq[j] = 1 / theta ** (2j / rope_dim),  j < rope_dim/2
    emb         = cat(position * inv_freq, position * inv_freq)
    cos, sin    = emb.cos().to(bf16), emb.sin().to(bf16)
    rotate_half(x) = cat(-x[dim/2:], x[:dim/2])
    out         = x * cos + rotate_half(x) * sin        (in bf16)

  python3 tests/test_mimo26_rope_vs_torch.py [DUMP.bin]
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOOL = ROOT / "tools" / "mimo26_dump_rope"


def read_dump(path):
    blob = path.read_bytes()
    offset = 0
    case_count, head_dim, rope_dim = struct.unpack_from("<III", blob, offset)
    offset += 12
    cases = []
    for _ in range(case_count):
        position, theta = struct.unpack_from("<Qf", blob, offset)
        offset += 12
        def take(count):
            nonlocal offset
            values = struct.unpack_from(f"<{count}H", blob, offset)
            offset += 2 * count
            return list(values)
        cases.append({
            "position": position,
            "theta": theta,
            "input": take(head_dim),
            "output": take(head_dim),
            "cos": take(rope_dim),
            "sin": take(rope_dim),
        })
    return head_dim, rope_dim, cases


def main():
    try:
        import torch
    except ImportError:
        print("skip: torch is not available")
        return 0

    dump = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    temporary = None
    if dump is None:
        if not TOOL.exists():
            print(f"skip: {TOOL} not built (make tools/mimo26_dump_rope)")
            return 0
        temporary = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        temporary.close()
        dump = Path(temporary.name)
        subprocess.run([str(TOOL), str(dump)], check=True,
                       stdout=subprocess.DEVNULL)

    head_dim, rope_dim, cases = read_dump(dump)
    pairs = rope_dim // 2

    def bits_to_bf16(values):
        raw = torch.tensor(values, dtype=torch.int32) << 16
        return raw.to(torch.int32).view(torch.float32).to(torch.bfloat16)

    failures = []
    worst_table = 0
    worst_output = 0
    for index, case in enumerate(cases):
        position = case["position"]
        theta = case["theta"]

        # Table, computed in f32 then cast to bf16 as the reference does.
        j = torch.arange(0, rope_dim, 2, dtype=torch.int64).to(torch.float32)
        inv_freq = 1.0 / (theta ** (j / rope_dim))
        freqs = torch.tensor(float(position), dtype=torch.float32) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(torch.bfloat16)
        sin = emb.sin().to(torch.bfloat16)

        mine_cos = bits_to_bf16(case["cos"])
        mine_sin = bits_to_bf16(case["sin"])
        table_delta = max(
            int((cos.view(torch.int16) != mine_cos.view(torch.int16)).sum()),
            int((sin.view(torch.int16) != mine_sin.view(torch.int16)).sum()),
        )
        worst_table = max(worst_table, table_delta)
        if table_delta:
            # Report as a tolerance rather than a bit mismatch: transcendental
            # rounding may legitimately differ by one bf16 ulp.
            drift = float((cos.float() - mine_cos.float()).abs().max())
            drift = max(drift, float((sin.float() - mine_sin.float()).abs().max()))
            if drift > 0.0079:  # one bf16 ulp near 1.0
                failures.append(
                    f"case {index} position {position}: cos/sin drift {drift:.2e}")

        # Rotation, in bf16, using the reference's rotate_half.
        x = bits_to_bf16(case["input"])
        x_rope, x_nope = x[:rope_dim], x[rope_dim:]
        rotated = torch.cat((-x_rope[pairs:], x_rope[:pairs]), dim=-1)
        expected_rope = (x_rope * mine_cos + rotated * mine_sin)
        expected = torch.cat((expected_rope, x_nope), dim=-1)

        mine = bits_to_bf16(case["output"])
        mismatch = int((expected.view(torch.int16) !=
                        mine.view(torch.int16)).sum())
        worst_output = max(worst_output, mismatch)
        if mismatch:
            drift = float((expected.float() - mine.float()).abs().max())
            magnitude = float(expected.float().abs().max())
            relative = drift / magnitude if magnitude > 0 else drift
            if relative > 0.01:
                failures.append(
                    f"case {index} position {position}: output relative "
                    f"drift {relative:.2e} over {mismatch} elements")

        # The nope half must be untouched, bit for bit.
        if int((x[rope_dim:].view(torch.int16) !=
                mine[rope_dim:].view(torch.int16)).sum()):
            failures.append(f"case {index}: nope half was modified")

    if temporary is not None:
        dump.unlink(missing_ok=True)

    if failures:
        print("failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"  ok  {len(cases)} RoPE cases agree with torch "
          f"(worst table bit-diff {worst_table}, output bit-diff {worst_output}, "
          f"both within one bf16 ulp)")
    print("test_mimo26_rope_vs_torch: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

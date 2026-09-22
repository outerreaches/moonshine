#!/usr/bin/env python3
"""Check the C operators against the checkpoint's own reference modules.

This is a stronger oracle than the property tests: it runs the vendored
`modeling_mimo_v2.py` classes -- MiMoV2RMSNorm and MiMoV2MoEGate -- on the
exact inputs the C implementation used, and compares outputs. A disagreement
is a semantic difference from the model author's code, not from a spec we
transcribed.

The reference files are copied into a temporary package because they use
relative imports; the originals are never modified. Only these two classes and
the module's import side effects execute -- no weights are loaded and no
checkpoint is read.

  MIMO26_ROOT=/path/to/checkpoint python3 tests/test_mimo26_ops_vs_reference.py
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOOL = ROOT / "tools" / "mimo26_dump_ops"
DEFAULT_ROOT = "/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL"


def read_sections(path):
    blob = path.read_bytes()
    offset = 0
    (count,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    sections = {}
    for _ in range(count):
        tag = blob[offset:offset + 8].decode().strip()
        offset += 8
        (payload_bytes,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        sections[tag] = blob[offset:offset + payload_bytes]
        offset += payload_bytes
    return sections


def build_reference_package(checkpoint, workdir):
    package = workdir / "mimo_reference"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in ("modeling_mimo_v2.py", "configuration_mimo_v2.py"):
        shutil.copy2(checkpoint / name, package / name)
    return workdir


def main():
    checkpoint = Path(os.environ.get("MIMO26_ROOT", DEFAULT_ROOT))
    if not (checkpoint / "modeling_mimo_v2.py").exists():
        print(f"skip: no checkpoint at {checkpoint}")
        return 0
    if not TOOL.exists():
        print(f"skip: {TOOL} not built (make tools/mimo26_dump_ops)")
        return 0
    try:
        import torch
    except ImportError:
        print("skip: torch is not available")
        return 0

    warnings.filterwarnings("ignore")
    workdir = Path(tempfile.mkdtemp(prefix="mimo26-reference-"))
    failures = []
    try:
        dump = workdir / "ops.bin"
        subprocess.run([str(TOOL), str(dump)], check=True,
                       stdout=subprocess.DEVNULL)
        sections = read_sections(dump)

        package_root = build_reference_package(checkpoint, workdir)
        sys.path.insert(0, str(package_root))
        from mimo_reference.configuration_mimo_v2 import MiMoV2Config
        import mimo_reference.modeling_mimo_v2 as reference

        def bf16(values):
            raw = torch.tensor(list(values), dtype=torch.int32) << 16
            return raw.view(torch.float32).to(torch.bfloat16)

        # ---- RMSNorm ----
        payload = sections["RMSNORM"]
        (count,) = struct.unpack_from("<I", payload, 0)
        offset = 4
        def take_u16(n):
            nonlocal offset
            values = struct.unpack_from(f"<{n}H", payload, offset)
            offset += 2 * n
            return values
        raw_input = take_u16(count)
        raw_weight = take_u16(count)
        raw_output = take_u16(count)

        norm = reference.MiMoV2RMSNorm(count, eps=1e-6)
        norm.eval()
        with torch.no_grad():
            norm.weight.data = bf16(raw_weight)
            expected = norm(bf16(raw_input))
        mine = bf16(raw_output)
        mismatch = int((expected.view(torch.int16) !=
                        mine.view(torch.int16)).sum())
        if mismatch:
            drift = float((expected.float() - mine.float()).abs().max())
            scale = float(expected.float().abs().max())
            relative = drift / scale if scale else drift
            if relative > 0.008:  # one bf16 ulp
                failures.append(f"RMSNorm: {mismatch}/{count} differ, "
                                f"relative {relative:.3e}")
            else:
                print(f"  ok  RMSNorm within one bf16 ulp "
                      f"({mismatch}/{count} bits differ)")
        else:
            print(f"  ok  RMSNorm bit-exact against MiMoV2RMSNorm "
                  f"({count} elements)")

        # ---- Router ----
        payload = sections["ROUTER"]
        experts, topk = struct.unpack_from("<II", payload, 0)
        offset = 8
        logits = struct.unpack_from(f"<{experts}f", payload, offset)
        offset += 4 * experts
        bias = struct.unpack_from(f"<{experts}f", payload, offset)
        offset += 4 * experts
        my_indices = struct.unpack_from(f"<{topk}I", payload, offset)
        offset += 4 * topk
        my_weights = struct.unpack_from(f"<{topk}f", payload, offset)

        config = MiMoV2Config(**json.loads((checkpoint / "config.json").read_text()))
        gate = reference.MiMoV2MoEGate(config)
        gate.eval()
        with torch.no_grad():
            # Drive the gate's projection to the exact logits the C side used
            # by making the weight an identity-like map over a crafted input.
            gate.e_score_correction_bias.copy_(
                torch.tensor(bias, dtype=torch.float32))
            hidden = torch.zeros(1, 1, config.hidden_size, dtype=torch.float32)
            hidden[0, 0, 0] = 1.0
            weight = torch.zeros(experts, config.hidden_size,
                                 dtype=torch.float32)
            weight[:, 0] = torch.tensor(logits, dtype=torch.float32)
            gate.weight.data = weight
            indices, weights = gate(hidden)

        reference_set = set(int(x) for x in indices[0].tolist())
        mine_set = set(int(x) for x in my_indices)
        if reference_set != mine_set:
            failures.append(
                f"router selected {sorted(mine_set)} vs reference "
                f"{sorted(reference_set)}")
        else:
            reference_map = {int(i): float(w) for i, w in
                             zip(indices[0].tolist(), weights[0].tolist())}
            worst = max(abs(reference_map[int(i)] - w)
                        for i, w in zip(my_indices, my_weights))
            if worst > 1e-6:
                failures.append(f"router weights differ by {worst:.3e}")
            else:
                print(f"  ok  router matches MiMoV2MoEGate on all {topk} "
                      f"experts of {experts} (max weight delta {worst:.2e})")
            # The reference leaves order unspecified; ours is ascending.
            if list(my_indices) != sorted(my_indices):
                failures.append("router indices are not ascending")

        # ---- SiLU product, against the reference's own activation ----
        payload = sections["SILUPROD"]
        (count,) = struct.unpack_from("<I", payload, 0)
        offset = 4
        def take_f32(n):
            nonlocal offset
            values = struct.unpack_from(f"<{n}f", payload, offset)
            offset += 4 * n
            return values
        gate_values = take_f32(count)
        up_values = take_f32(count)
        out_values = take_f32(count)
        act = reference.ACT2FN[config.hidden_act]
        with torch.no_grad():
            expected = act(torch.tensor(gate_values, dtype=torch.float32)) * \
                       torch.tensor(up_values, dtype=torch.float32)
        mine = torch.tensor(out_values, dtype=torch.float32)
        drift = float((expected - mine).abs().max())
        scale = float(expected.abs().max())
        relative = drift / scale if scale else drift
        if relative > 1e-6:
            failures.append(f"silu product relative drift {relative:.3e}")
        else:
            print(f"  ok  silu(gate)*up matches ACT2FN['{config.hidden_act}'] "
                  f"(relative {relative:.2e} over {count} elements)")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        print("\nfailures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("test_mimo26_ops_vs_reference: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

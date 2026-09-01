#!/usr/bin/env python3
"""Generate the pinned GLM-5.3-Flash layer-0, head-0 KDA fixture."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

MAGIC = b"G53KDA4\0"
VERSION = 1
ENDIAN_TAG = 0x01020304
T, HIDDEN, HEADS, D = 5, 4096, 64, 128
PREFIX = "model.language_model.layers.0.self_attn."
NAMES = [
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
    "f_a_proj.weight", "f_b_proj.weight", "dt_bias", "A_log",
    "b_proj.weight",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_tensors(root: Path):
    index_path = root / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    wanted = [PREFIX + x for x in NAMES]
    shards = {}
    result, provenance = {}, {}
    for name in wanted:
        shard_name = weight_map[name]
        if shard_name not in shards:
            with (root / shard_name).open("rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            shards[shard_name] = (8 + header_len, header)
        data_base, header = shards[shard_name]
        entry = header[name]
        begin, end = entry["data_offsets"]
        with (root / shard_name).open("rb") as f:
            f.seek(data_base + begin)
            payload = f.read(end - begin)
        if len(payload) != end - begin:
            raise RuntimeError(f"short tensor payload: {name}")
        dtype = entry["dtype"]
        raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        if dtype == "BF16":
            tensor = raw.view(torch.bfloat16)
        elif dtype == "F32":
            tensor = raw.view(torch.float32)
        else:
            raise RuntimeError(f"unsupported dtype {dtype}: {name}")
        tensor = tensor.reshape(entry["shape"]).clone()
        result[name.removeprefix(PREFIX)] = tensor
        provenance[name] = {
            "dtype": dtype,
            "shape": entry["shape"],
            "shard": shard_name,
            "logical_payload_bytes": len(payload),
            "logical_payload_sha256": sha256(payload),
        }
    return result, provenance


def strict_kda(q, k, v, g, beta, initial):
    """Scalar-order F32 recurrence, with vector lanes only across independent V."""
    state = initial.clone().to(torch.float32)
    out = torch.empty((T, D), dtype=torch.float32)
    zero = torch.tensor(0.0, dtype=torch.float32)
    eps = torch.tensor(1.0e-6, dtype=torch.float32)
    qscale = torch.sqrt(torch.tensor(float(D), dtype=torch.float32))
    for t in range(T):
        qss, kss = zero.clone(), zero.clone()
        for i in range(D):
            qss = qss + q[t, i] * q[t, i]
            kss = kss + k[t, i] * k[t, i]
        qden, kden = torch.sqrt(qss + eps), torch.sqrt(kss + eps)
        qn = torch.empty(D, dtype=torch.float32)
        kn = torch.empty(D, dtype=torch.float32)
        for i in range(D):
            qn[i] = (q[t, i] / qden) / qscale
            kn[i] = k[t, i] / kden
        for i in range(D):
            state[i] = state[i] * torch.exp(g[t, i])
        prediction = torch.zeros(D, dtype=torch.float32)
        for i in range(D):
            prediction = prediction + state[i] * kn[i]
        delta = (v[t] - prediction) * beta[t]
        for i in range(D):
            state[i] = state[i] + kn[i] * delta
        y = torch.zeros(D, dtype=torch.float32)
        for i in range(D):
            y = y + qn[i] * state[i]
        out[t] = y
    return out, state


def f32_bytes(x):
    return x.detach().contiguous().to(torch.float32).numpy().astype("<f4", copy=False).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("official_root", type=Path)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "fixtures/glm53_phase4_kda_v1.bin")
    args = ap.parse_args()
    root, output = args.official_root, args.output
    tensors, tensor_provenance = load_tensors(root)

    # Every value is an exactly representable dyadic rational in BF16.
    c = torch.arange(HIDDEN, dtype=torch.int64)[None, :]
    t = torch.arange(T, dtype=torch.int64)[:, None]
    hidden = ((((37 * t + 17 * c) % 61) - 30).to(torch.float32) / 64.0).to(torch.bfloat16)
    if not torch.equal(hidden.float(), ((((37*t + 17*c) % 61)-30).float()/64.0)):
        raise RuntimeError("hidden input formula ceased to be BF16-exact")

    projected = {}
    with torch.no_grad():
        for x in ("q", "k", "v"):
            projected[x] = F.linear(hidden, tensors[x + "_proj.weight"])
        convolved = {}
        for x in ("q", "k", "v"):
            z = F.conv1d(projected[x].T.unsqueeze(0),
                         tensors[x + "_conv1d.weight"], padding=3,
                         groups=HEADS * D)[:, :, :T]
            convolved[x] = F.silu(z).transpose(1, 2).reshape(T, HEADS, D)

        fa = F.linear(hidden, tensors["f_a_proj.weight"])
        forget = F.linear(fa, tensors["f_b_proj.weight"])
        forget_f32 = forget.float() + tensors["dt_bias"].float().view(1, -1)
        decay_rate = torch.exp(tensors["A_log"].float()).view(1, HEADS, 1)
        g_all = -5.0 * torch.sigmoid(decay_rate * forget_f32.view(T, HEADS, D))
        beta_all = torch.sigmoid(F.linear(hidden, tensors["b_proj.weight"]))

    q = convolved["q"][:, 0].float().contiguous()
    k = convolved["k"][:, 0].float().contiguous()
    v = convolved["v"][:, 0].float().contiguous()
    g = g_all[:, 0].float().contiguous()
    beta = beta_all[:, 0].float().contiguous()
    ii = torch.arange(D, dtype=torch.int64)[:, None]
    jj = torch.arange(D, dtype=torch.int64)[None, :]
    initial = ((((13 * ii + 7 * jj) % 29) - 14).float() / 4096.0).contiguous()
    expected, final = strict_kda(q, k, v, g, beta, initial)

    arrays = [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta),
              ("initial_state", initial), ("expected_output", expected),
              ("expected_final_state", final)]
    payload = b"".join(f32_bytes(x) for _, x in arrays)
    float_count = len(payload) // 4
    header = struct.pack("<8s7I", MAGIC, VERSION, ENDIAN_TAG, T, D, D,
                         len(arrays), float_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + payload)

    identity_path = root / ".provenance/source_identity.json"
    identity = json.loads(identity_path.read_text())
    metadata = {
        "schema_version": 1,
        "fixture_format": {
            "magic": MAGIC.rstrip(b"\0").decode(), "version": VERSION,
            "byte_order": "little", "header_struct": "<8s7I",
            "array_order": [{"name": n, "shape": list(x.shape),
                              "f32_sha256": sha256(f32_bytes(x))} for n, x in arrays],
        },
        "fixture_file": output.name,
        "fixture_bytes": output.stat().st_size,
        "fixture_sha256": file_sha256(output),
        "official": {
            "repository": identity.get("repo"), "revision": identity.get("revision"),
            "config_sha256": file_sha256(root / "config.json"),
            "index_sha256": file_sha256(root / "model.safetensors.index.json"),
        },
        "transformers": {
            "commit": "eb4d9e2a64a013bec12289288b85d0b1210ba0aa",
            "semantics": "BF16 q/k/v projection; causal depthwise conv+SiLU; F32 forget gate with lower bound -5; BF16 sigmoid beta; F32 recurrent KDA with q/k l2norm",
        },
        "input": {"shape": [T, HIDDEN], "dtype": "BF16",
                  "formula": "hidden[t,c] = (((37*t + 17*c) % 61) - 30) / 64"},
        "initial_state": {"shape": [D, D], "dtype": "F32",
                          "formula": "state[i,j] = (((13*i + 7*j) % 29) - 14) / 4096"},
        "tensors_consumed": tensor_provenance,
        "generation": {"torch_version": torch.__version__, "device": "cpu",
                       "note": "Expected recurrence uses explicit left-to-right F32 reductions matching glm53_state_oracle.c."},
    }
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()

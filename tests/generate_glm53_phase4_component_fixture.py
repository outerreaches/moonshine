#!/usr/bin/env python3
"""Generate bounded GLM-5.3 Phase-4 component goldens from pinned official tensors.

This intentionally opens only the shards/tensors named below. It never builds a model,
and all reference projection work is torch CPU float32.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
REPOSITORY = "zai-org/GLM-5.3-Flash"
TRANSFORMERS_COMMIT = "eb4d9e2a64a013bec12289288b85d0b1210ba0aa"
LLAMA_COMMIT = "a771613af20f3dc60247e4b6a3d11516f0664673"
CONFIG_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
INDEX_SHA256 = "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"
MAGIC = b"G53P4CMP"
VERSION = 1

NAMES = {
 "mhc_fn": "model.language_model.layers.0.hc_attn_fn",
 "mhc_base": "model.language_model.layers.0.hc_attn_base",
 "mhc_scale": "model.language_model.layers.0.hc_attn_scale",
 "router_weight": "model.language_model.layers.3.mlp.gate.weight",
 "router_bias": "model.language_model.layers.3.mlp.gate.e_score_correction_bias",
 "expert_gate": "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
 "expert_gate_scale": "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv",
 "expert_up": "model.language_model.layers.3.mlp.experts.0.up_proj.weight",
 "expert_up_scale": "model.language_model.layers.3.mlp.experts.0.up_proj.weight_scale_inv",
}

def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def logical_bytes(t):
    return t.detach().contiguous().view(torch.uint8).numpy().tobytes()

def load(root, weight_map, key, records):
    shard = weight_map[key]
    with safe_open(root / shard, framework="pt", device="cpu") as f:
        t = f.get_tensor(key).contiguous()
    payload = logical_bytes(t)
    records[key] = {"shard": shard, "dtype": str(t.dtype).removeprefix("torch."),
                    "shape": list(t.shape), "logical_nbytes": len(payload),
                    "logical_payload_sha256": hashlib.sha256(payload).hexdigest()}
    return t

def bf16_exact(values):
    b = values.to(torch.bfloat16)
    assert torch.equal(b.float(), values)
    return b

def mhc_weights(mix, base, scale, eps=1e-6):
    pre = torch.sigmoid(mix[:4] * scale[0] + base[:4]) + eps
    post = 2 * torch.sigmoid(mix[4:8] * scale[1] + base[4:8])
    comb = torch.softmax((mix[8:] * scale[2] + base[8:]).reshape(4,4), dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(1, 20):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb

def fp8_project(weight, scales, x):
    # Transformers eb4d9e2 uses 128x128 block scales for these expert matrices.
    dequant = weight.float() * scales.float().repeat_interleave(128,0).repeat_interleave(128,1)
    return torch.mv(dequant, x.float())

def f32_bytes(t):
    a = np.asarray(t.detach().cpu().float().numpy(), dtype="<f4")
    return a.tobytes(order="C")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("official_root", type=Path)
    ap.add_argument("fixture_bin", type=Path)
    ap.add_argument("provenance_json", type=Path)
    a=ap.parse_args(); root=a.official_root
    identity=json.loads((root/".provenance/source_identity.json").read_text())
    if identity.get("revision") != REVISION or identity.get("repo") != REPOSITORY:
        raise SystemExit("official source identity does not match pin")
    if file_sha(root/"config.json") != CONFIG_SHA256 or file_sha(root/"model.safetensors.index.json") != INDEX_SHA256:
        raise SystemExit("official config/index digest does not match pin")
    config=json.loads((root/"config.json").read_text())["text_config"]
    rms_norm_eps=float(config["rms_norm_eps"])
    if rms_norm_eps != 1e-5:
        raise SystemExit(f"unexpected official text_config.rms_norm_eps: {rms_norm_eps!r}")
    index=json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]
    records={}
    mhc_fn=load(root,index,NAMES["mhc_fn"],records)
    base=load(root,index,NAMES["mhc_base"],records).float()
    scale=load(root,index,NAMES["mhc_scale"],records).float()
    d=torch.arange(4096,dtype=torch.int64)
    streams=torch.stack([(((37*d+11*s)%127)-63).float()/64 for s in range(4)])
    streams=bf16_exact(streams).float()
    # Pinned MHC computes its unweighted RMSNorm over the flattened H*D axis
    # before hc_attn_fn (Transformers eb4d9e2, eps=config.rms_norm_eps).
    mhc_norm=torch.nn.functional.rms_norm(streams.reshape(-1), (4*4096,),
                                           weight=None, eps=rms_norm_eps)
    mix=torch.mv(mhc_fn.float(),mhc_norm)
    pre,post,comb=mhc_weights(mix,base,scale)
    collapsed=(pre[:,None]*streams).sum(0)
    branch=bf16_exact((((29*d)%61)-30).float()/32).float()
    postout=post[:,None]*branch + torch.einsum("sd,st->td",streams,comb)

    router_weight=load(root,index,NAMES["router_weight"],records)
    bias=load(root,index,NAMES["router_bias"],records).float()
    x=bf16_exact((((17*d)%31)-15).float()/32).float()
    logits=torch.mv(router_weight.float(),x)
    raw=torch.sigmoid(logits); choice=raw+bias
    order=torch.argsort(choice,descending=True,stable=True); top=order[:8]
    weights=raw[top]; weights=weights/weights.sum()*2.5
    margin=float(choice[order[7]]-choice[order[8]])
    if not margin > 0: raise SystemExit("router boundary tie")

    gate_w=load(root,index,NAMES["expert_gate"],records)
    gate_s=load(root,index,NAMES["expert_gate_scale"],records)
    up_w=load(root,index,NAMES["expert_up"],records)
    up_s=load(root,index,NAMES["expert_up_scale"],records)
    gate=fp8_project(gate_w,gate_s,x); up=fp8_project(up_w,up_s,x)
    activation=torch.nn.functional.silu(torch.clamp(gate,max=10.0))*torch.clamp(up,min=-10.0,max=10.0)

    arrays=[streams,mix,base,scale,pre,post,comb,collapsed,branch,postout,
            x,logits,bias,weights,gate,up,activation]
    header=struct.pack("<8sIIIIIIIf",MAGIC,VERSION,0x01020304,4096,24,288,8,2048,margin)
    header += REVISION.encode()+TRANSFORMERS_COMMIT.encode()+LLAMA_COMMIT.encode()+CONFIG_SHA256.encode()+INDEX_SHA256.encode()
    payload=header+b"".join(f32_bytes(t) for t in arrays[:13])+np.asarray(top.cpu(),dtype="<u4").tobytes()+b"".join(f32_bytes(t) for t in arrays[13:])
    a.fixture_bin.parent.mkdir(parents=True,exist_ok=True); a.fixture_bin.write_bytes(payload)
    prov={"schema_version":1,"binary_format":{"magic":MAGIC.decode(),"version":VERSION,"endianness":"little","layout":"fixed; see generator and C reader","sha256":hashlib.sha256(payload).hexdigest(),"bytes":len(payload)},
      "official":{"repository":REPOSITORY,"revision":REVISION,"config_sha256":CONFIG_SHA256,"index_sha256":INDEX_SHA256},
      "references":{"transformers_commit":TRANSFORMERS_COMMIT,"llama_cpp_commit":LLAMA_COMMIT,"equations":"Transformers GLM support at eb4d9e2"},
      "inputs":{"mhc_stream_formula":"BF16-exact stream[s,d]=(((37*d+11*s)%127)-63)/64; hc_attn_fn input is unweighted F32 RMSNorm over flattened [4*4096], eps=official text_config.rms_norm_eps (required 1e-5)","router_and_expert_formula":"BF16-exact x[d]=(((17*d)%31)-15)/32","branch_formula":"BF16-exact branch[d]=(((29*d)%61)-30)/32"},
      "parameters":{"mhc_layer":0,"mhc_rms_norm_eps":rms_norm_eps,"mhc_rms_norm_weighted":False,"mhc_rms_norm_axis":"flattened_4x4096","mhc_eps":1e-6,"router_layer":3,"n_group":1,"top_k":8,"norm_topk_prob":True,"routed_scaling_factor":2.5,"expert":0,"swiglu_limit":10.0,"expert_intermediate":2048,"down_projection_included":False},
      "router":{"top8":[int(v) for v in top],"selection_boundary_margin":margin,"boundary_tie":False},
      "tolerances":{"mhc_weights_abs":8e-6,"mhc_apply_abs":3e-5,"router_weights_abs":2e-6,"swiglu_abs":3e-5},"consumed_tensors":records}
    a.provenance_json.write_text(json.dumps(prov,indent=2,sort_keys=True)+"\n")
    print(f"wrote {a.fixture_bin} ({len(payload)} bytes) and {a.provenance_json}; router margin={margin:.9g}")
if __name__ == "__main__": main()

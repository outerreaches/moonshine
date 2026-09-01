#!/usr/bin/env python3
import hashlib, json, struct, sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def logical_payload_info(root, index, name):
    shard_name=index[name]; shard=root/shard_name
    with shard.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(n)); data=8+n
        entry=header[name]; begin,end=entry['data_offsets']; f.seek(data+begin)
        payload=f.read(end-begin)
        return {'shard':shard_name,'dtype':entry['dtype'],'shape':entry['shape'],
                'bytes':end-begin,'sha256':hashlib.sha256(payload).hexdigest()}

def verify_extra_fixture(meta_path, fixture_path, root, index, official):
    meta=json.loads(meta_path.read_text())
    if meta.get('official',{}).get('repository')!=official['repository']:
        fail(f'{meta_path.name} repository')
    if meta.get('official',{}).get('revision')!=official['revision']:
        fail(f'{meta_path.name} revision')
    if meta.get('official',{}).get('config_sha256')!=official['config_sha256']:
        fail(f'{meta_path.name} config identity')
    if meta.get('official',{}).get('index_sha256')!=official['index_sha256']:
        fail(f'{meta_path.name} index identity')
    expected_bytes=meta.get('fixture_bytes',meta.get('binary_format',{}).get('bytes'))
    expected_sha=meta.get('fixture_sha256',meta.get('binary_format',{}).get('sha256'))
    if fixture_path.stat().st_size!=expected_bytes or sha(fixture_path)!=expected_sha:
        fail(f'{fixture_path.name} identity')
    consumed=meta.get('tensors_consumed',meta.get('consumed_tensors',{}))
    if not consumed: fail(f'{meta_path.name} empty tensor provenance')
    dtype_alias={'BF16':'BF16','bfloat16':'BF16','F32':'F32','float32':'F32',
                 'F8_E4M3':'F8_E4M3','float8_e4m3fn':'F8_E4M3'}
    for name, tensor in consumed.items():
        actual=logical_payload_info(root,index,name)
        declared_bytes=tensor.get('logical_payload_bytes',tensor.get('logical_nbytes'))
        if (actual['sha256']!=tensor['logical_payload_sha256'] or
            actual['shard']!=tensor['shard'] or actual['shape']!=tensor['shape'] or
            actual['bytes']!=declared_bytes or
            dtype_alias.get(tensor['dtype'])!=actual['dtype']):
            fail(f'{meta_path.name} tensor declaration {name}')

def fail(message):
    raise SystemExit('FAIL phase4 reference: '+message)

if len(sys.argv)<4 or (len(sys.argv)-4)%2:
    raise SystemExit(f'usage: {sys.argv[0]} OFFICIAL_ROOT REFERENCE_JSON REFERENCE_F32 [FIXTURE_JSON FIXTURE_BIN]...')
root, ref_path, out_path=map(Path,sys.argv[1:4])
extra=[Path(x) for x in sys.argv[4:]]
r=json.loads(ref_path.read_text())
o=r['official']; p=r['official_projection']
identity=json.loads((root/'.provenance/source_identity.json').read_text())
if identity.get('revision')!=o['revision']: fail('official revision')
if identity.get('repo')!=o['repository']: fail('official repository')
if sha(root/'config.json')!=o['config_sha256']: fail('config SHA-256')
if sha(root/'model.safetensors.index.json')!=o['index_sha256']: fail('index SHA-256')
idx=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
shard=root/idx[p['weight_name']]
if shard.name!=p['shard'] or idx[p['scale_name']]!=p['shard']: fail('projection shard')
with shard.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(n)); data=8+n
    for key,field in [(p['weight_name'],'weight_sha256'),(p['scale_name'],'scale_sha256')]:
        begin,end=header[key]['data_offsets']; f.seek(data+begin)
        if hashlib.sha256(f.read(end-begin)).hexdigest()!=p[field]: fail(field)
if len(out_path.read_bytes())!=p['rows']*4: fail('reference F32 byte length')
if sha(out_path)!=p['reference_f32_sha256']: fail('reference F32 SHA-256')
if p['reference_f32_sha256']!=p['diagnostic_sequential_f32_sha256']:
    fail('reference/diagnostic digest mismatch')
for i in range(0,len(extra),2):
    verify_extra_fixture(extra[i],extra[i+1],root,idx,o)
print(f'PASS phase4 pinned provenance and logical payload hashes extra_fixtures={len(extra)//2}')

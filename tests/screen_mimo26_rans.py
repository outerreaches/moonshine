"""Byte-exact CPU screen; shared model trained on expert 0, held out 97/255.
No model loading, GPU use, weight changes, canonicalization, or store transcode.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('library',type=Path)
    p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir()
    report=dict(complete=False,passed=False,matrices=[],models={},groups={},
        model_revision='3b38d063180c3e4aed9691fdc735f3d10b266ee4',
        root=str(a.root),library_sha256=digest(a.library),
        source_sha256=digest(Path(__file__).with_name('mimo26_rans_screen.cpp')),
        driver_sha256=digest(Path(__file__)),
        scope='CPU exact roundtrip/size screen, not GPU/store/speed qualification')
    def save():
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    start=time.monotonic();save()
    try:
        index_path=a.root/'model.safetensors.index.json'
        report['index_sha256']=digest(index_path)
        report['config_sha256']=digest(a.root/'config.json')
        index=json.loads(index_path.read_text())
        assert index['metadata']['tp_size']==4 and index['metadata']['save_format']=='mxfp4'
        weight_map=index['weight_map'];headers={}
        rows=[]
        for layer in (1,24,47):
            for expert in (0,97,255):
                for proj in ('down_proj','gate_proj','up_proj'):
                    for suffix in ('weight','weight_scale'):
                        name=f'model.layers.{layer}.mlp.experts.{expert}.{proj}.{suffix}'
                        shard=weight_map[name]
                        if shard not in headers:
                            with (a.root/shard).open('rb') as f:
                                length=struct.unpack('<Q',f.read(8))[0]
                                headers[shard]=(8+length,json.loads(f.read(length)))
                        base,header=headers[shard];tensor=header[name];lo,hi=tensor['data_offsets']
                        packed=suffix=='weight'
                        assert tensor['dtype']=='U8' and hi-lo==(4194304 if packed else 262144)
                        rows.append(dict(name=name,shard=shard,offset=base+lo,bytes=hi-lo,
                            packed=packed,layer=layer,expert=expert,group='train' if expert==0 else 'heldout'))
        def read(row):
            with (a.root/row['shard']).open('rb') as f:
                f.seek(row['offset']);data=f.read(row['bytes'])
            assert len(data)==row['bytes'];return data
        hist={True:np.zeros(16,dtype=np.uint64),False:np.zeros(256,dtype=np.uint64)}
        for row in rows:
            if row['group']!='train':continue
            data=read(row);b=np.frombuffer(data,dtype=np.uint8)
            counts=np.bincount(b,minlength=256).astype(np.uint64)
            if row['packed']:
                counts=counts.reshape(16,16);hist[True]+=counts.sum(axis=0)+counts.sum(axis=1)
            else:hist[False]+=counts
        for packed,h in hist.items():report['models'][str(packed)]=[int(x) for x in h]
        lib=C.CDLL(str(a.library.resolve()));fn=lib.mimo26_rans_screen
        fn.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_uint64),C.c_uint,C.c_uint,
            C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)];fn.restype=C.c_int
        model={k:(C.c_uint64*len(v))(*map(int,v)) for k,v in hist.items()}
        # Ensure both zero encodings / all codes / raw fallback / unseen scales work.
        tests=[]
        for packed in (False,True):
            for fixture in (bytes(range(256))*64,b'\x00'*16384,b'\x88'*16384,b'\x08\x80'*8192,bytes(range(128))):
                n=C.c_uint64();esc=C.c_uint64()
                assert fn(fixture,len(fixture),model[packed],16384,int(packed),C.byref(n),C.byref(esc))==1
                tests.append(dict(packed=packed,raw_bytes=len(fixture),stored=n.value,escapes=esc.value))
        report['synthetic_tests']=tests
        for row in rows:
            data=read(row);record=dict(row,sha256=hashlib.sha256(data).hexdigest(),tiles={})
            b=np.frombuffer(data,dtype=np.uint8)
            if row['packed']:
                record['zero_codes']=int(np.count_nonzero((b&15)==0)+np.count_nonzero((b>>4)==0))
                record['negative_zero_codes']=int(np.count_nonzero((b&15)==8)+np.count_nonzero((b>>4)==8))
            for tile in (16384,32768,65536):
                payload=C.c_uint64();esc=C.c_uint64()
                assert fn(data,len(data),model[row['packed']],tile,int(row['packed']),C.byref(payload),C.byref(esc))==1
                record['tiles'][str(tile)]=dict(aligned_payload_bytes=payload.value,raw_tiles=esc.value,
                    descriptor_bytes=(len(data)//tile)*32)
            report['matrices'].append(record);save()
            print('PASS',len(report['matrices']),'/54',row['name'],flush=True)
        for group in ('train','heldout'):
            selected=[r for r in report['matrices'] if r['group']==group]
            total=sum(r['bytes'] for r in selected)
            experts=sorted({(r['layer'],r['expert']) for r in selected})
            sizes={}
            for tile in ('16384','32768','65536'):
                stored=0
                for identity in experts:
                    subset=[r for r in selected if (r['layer'],r['expert'])==identity]
                    # Illustrative 64-byte header + descriptors, aligned, payload aligned,
                    # final expert aligned. Fixed shared model/index outside these blocks.
                    align=lambda n:(n+4095)//4096*4096
                    stored+=align(align(64+sum(r['tiles'][tile]['descriptor_bytes'] for r in subset))+
                        sum(r['tiles'][tile]['aligned_payload_bytes'] for r in subset))
                sizes[tile]=dict(expert_block_bytes=stored,saving_percent=100*(1-stored/total))
            report['groups'][group]=dict(raw_bytes=total,experts=len(experts),sizes=sizes)
        report.update(complete=True,passed=True,elapsed_seconds=time.monotonic()-start,
            accounting='includes 32-byte descriptors, 64-byte assumed expert header, 4-byte payload and 4KiB expert/header alignment; excludes shared table/index/file overhead; no persistent format generated')
    except Exception as error:
        report['error']=repr(error);raise
    finally:save()

if __name__=='__main__':main()

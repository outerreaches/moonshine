"""Guarded six-expert direct-I/O screen; test-only fixtures on the model SSD."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from mimo26_evidence import evidence_root

REPO=Path(__file__).resolve().parents[1]
VAULT=evidence_root()
SCREEN=VAULT/'mimo26-takeover-20260922/rans/report.json'
BUILD=VAULT/'mimo26-lookahead-long-20260923/chunk64-on/report.json'
SOURCES=('tests/mimo26_rans_io_gate.cu','tests/mimo26_rans_gpu_gate.cu','tests/mimo26_rans_screen.cpp',
         'tests/run_mimo26_rans_io_gate.py','k3_io_uring.c','k3_io_uring.h','k3_rocm_ops.h','mimo26_rocm_ops.h')
OBJECTS=('k3_rocm_ops.o','mimo26_rocm_ops.o','k3_io_uring.o')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(1048576): h.update(data)
    return h.hexdigest()


def worker(a):
    screen=json.loads(SCREEN.read_text()); assert screen['complete'] and screen['passed']
    assert sha(a.root/'config.json')==screen['config_sha256']
    assert sha(a.root/'model.safetensors.index.json')==screen['index_sha256']
    report=dict(complete=False,passed=False,screen_sha256=sha(SCREEN),model_revision=screen['model_revision'],
                library_sha256=sha(a.output/'gate.so'),rounds=a.rounds,fixtures=str(a.fixtures),inputs=[],
                scope='Six-expert O_DIRECT registered io_uring + mapped staging admission screen. '
                      'No worker cache hits. Original-shard and aligned-raw controls. '
                      'Contention uses real unchanged MLP kernels with synthetic inputs, not a full-model request.')
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save(); start=time.monotonic()
    try:
        data=bytearray(); paths=[]; offsets=[]
        for layer in (1,24,47):
            for expert in (97,255):
                rows=[x for x in screen['matrices'] if x['layer']==layer and x['expert']==expert]
                assert len(rows)==6 and all(x['group']=='heldout' for x in rows)
                paths.append(str(a.root/rows[0]['shard']).encode()); offsets.append(rows[0]['offset'])
                cursor=offsets[-1]
                for row in rows:
                    assert row['shard']==rows[0]['shard'] and row['offset']==cursor
                    with (a.root/row['shard']).open('rb') as f:
                        f.seek(row['offset']); part=f.read(row['bytes'])
                    assert len(part)==row['bytes'] and hashlib.sha256(part).hexdigest()==row['sha256']
                    data.extend(part);cursor+=len(part);report['inputs'].append(row)
        data=bytes(data);assert len(data)==80216064; save()
        library=C.CDLL(str(a.output.resolve()/'gate.so')); fn=library.mimo26_rans_io_gate
        fn.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.c_char_p,
                     C.POINTER(C.c_char_p),C.POINTER(C.c_uint64),C.c_char_p,C.c_uint32];fn.restype=C.c_int
        packed=(C.c_uint64*16)(*screen['models']['True']);scales=(C.c_uint64*256)(*screen['models']['False'])
        assert fn(data,packed,scales,str(a.fixtures).encode(),(C.c_char_p*6)(*paths),
                  (C.c_uint64*6)(*offsets),str(a.output.resolve()/'trials.jsonl').encode(),a.rounds)==1
        trials=[json.loads(line) for line in (a.output/'trials.jsonl').read_text().splitlines()]
        assert len(trials)==(a.rounds+1)*2*3*5
        report['fixture_files']={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in a.fixtures.glob('*.bin')}
        assert len(report['fixture_files'])==3
        report.update(complete=True,passed=True,elapsed_seconds=time.monotonic()-start,trials=len(trials))
    except BaseException as error:report['error']=repr(error);raise
    finally:save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL'))
    p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=8);p.add_argument('--fixtures',type=Path)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if a.worker:worker(a);return
    assert 2<=a.rounds<=16
    check=subprocess.run(['fuser','/dev/kfd'],capture_output=True)
    assert check.returncode==1 and not check.stdout.strip(),'GPU unavailable'
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    assert available>16*1024*1024
    a.output.mkdir()
    a.fixtures=Path(tempfile.mkdtemp(prefix='mimo26-rans-io-20260923-',dir='/srv/modelstore'))
    assert a.fixtures.stat().st_dev==a.root.stat().st_dev
    pinned=json.loads(BUILD.read_text());assert pinned['complete'] and pinned['passed']
    inputs={}
    for name in SOURCES+OBJECTS:
        if name in OBJECTS:assert sha(REPO/name)==pinned['built_objects'][name]
        dest=a.output/Path(name).name;shutil.copy2(REPO/name,dest);inputs[name]=sha(dest)
    shutil.copy2(a.build/'gate.so',a.output/'gate.so')
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--root',str(a.root),'--build',str(a.build),
             '--output',str(a.output.resolve()),'--fixtures',str(a.fixtures),'--rounds',str(a.rounds)]
    monitor=dict(complete=False,passed=False,command=command,sources=inputs,fixtures=str(a.fixtures),samples=[],
                 mount=subprocess.check_output(['findmnt','-J','-T',str(a.fixtures)],text=True))
    def save():(a.output/'monitor.json').write_text(json.dumps(monitor,indent=2)+'\n')
    save()
    with (a.output/'stdout.log').open('w') as stdout,(a.output/'stderr.log').open('w') as stderr:
        proc=subprocess.Popen(command,stdout=stdout,stderr=stderr);start=time.monotonic()
        try:
            while proc.poll() is None:
                try:status=Path(f'/proc/{proc.pid}/status').read_text()
                except FileNotFoundError:continue
                swap=int(next((x.split()[1] for x in status.splitlines() if x.startswith('VmSwap:')),'0'))
                monitor['samples'].append(dict(seconds=time.monotonic()-start,status=status,meminfo=Path('/proc/meminfo').read_text()))
                if swap or time.monotonic()-start>180:
                    monitor['guard_reason']='worker swap' if swap else 'timeout';break
                if len(monitor['samples'])%20==0:save()
                time.sleep(.25)
        finally:
            if proc.poll() is None:proc.terminate();monitor['terminated']=True
            monitor['exit_code']=proc.wait(timeout=40);monitor['complete']=True
            if (a.output/'report.json').exists():
                report=json.loads((a.output/'report.json').read_text())
                monitor['passed']=monitor['exit_code']==0 and report['complete'] and report['passed'] and 'guard_reason' not in monitor
            save()
    print(json.dumps({k:v for k,v in monitor.items() if k not in ('samples','sources','mount')},indent=2))
    assert monitor['passed']


if __name__=='__main__':main()

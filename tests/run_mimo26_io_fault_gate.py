"""Bounded real-I/O worker failure gates, isolated from the existing server."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--lookahead-build',type=Path)
a=p.parse_args(); out=a.output; out.mkdir()
repo=Path(__file__).resolve().parents[1]
vault=Path('/home/alex/Obsidian/beelink-knowledge/Projects/Moonshine/Evidence')
base=vault/'mimo26-takeover-20260922/route'
workerbase=vault/'mimo26-guarded-reset-20260922/final'
if a.lookahead_build:workerbase=a.lookahead_build
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
r=dict(complete=False,passed=False,commands=[],runs=[],inputs={})
def save():(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
try:
    for name in ('mimo26_io_fault_gate.cu','run_mimo26_io_fault_gate.py'):
        shutil.copy2(repo/'tests'/name,out/name); r['inputs'][name]=sha(out/name)
    br=json.loads((base/'report.json').read_text())
    wr=json.loads((workerbase/'report.json').read_text()); assert wr['complete'] and wr['passed']
    sources=wr['inputs'] if a.lookahead_build else wr['sources']
    assert sha(repo/'mimo26_gpu_worker.cu')==sources['mimo26_gpu_worker.cu']
    assert sha(repo/'mimo26_gpu_worker.h')==sources['mimo26_gpu_worker.h']
    workerpath=workerbase/('mimo26_gpu_worker.o' if a.lookahead_build else 'worker.o')
    if a.lookahead_build:assert sha(workerpath)==wr['built_objects'][workerpath.name]
    objects=[str(workerpath)]; r['inputs'][objects[0]]=sha(workerpath)
    r['lookahead_enabled']=bool(a.lookahead_build)
    r['worker_build_report_sha256']=sha(workerbase/'report.json')
    for name,digest in br['inputs'].items():
        if name.endswith('.o') and name!='mimo26_gpu_worker.o':
            path=workerbase/name if a.lookahead_build else base/name
            expected=wr['built_objects'][name] if a.lookahead_build else digest
            assert sha(path)==expected; objects.append(str(path)); r['inputs'][str(path)]=expected
    flags=['/opt/rocm/bin/hipcc','-O3','-g','-fno-fast-math','--offload-arch=gfx1151','-I'+str(repo)]
    commands=[flags+(['-DLOOKAHEAD_CANDIDATE'] if a.lookahead_build else [])+['-c',str(out/'mimo26_io_fault_gate.cu'),'-o',str(out/'driver.o')],
              flags+[str(out/'driver.o')]+objects+['-Wl,--wrap='+s for s in ('k3_io_uring_submit','k3_io_uring_wait','k3_io_uring_destroy','hipMemcpy','hipHostFree')]+
              ['-lm','-pthread','-lhipblas','-lhipblaslt','-lzstd','-licui18n','-licuuc','-licudata','-o',str(out/'gate')]]
    with (out/'build.log').open('w') as log:
        for cmd in commands:
            r['commands'].append(cmd); subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    r['binary_sha256']=sha(out/'gate')
    for mode in ('submit','wait','short'):
        assert not subprocess.run(['fuser','/dev/kfd'],capture_output=True).stdout.strip(),'GPU occupied'
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        assert available>40*1024*1024
        row=dict(mode=mode,memory=[]); r['runs'].append(row); save()
        cmd=[str(out/'gate'),'/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL',mode,str(out/(mode+'.bin'))]
        r['commands'].append(cmd)
        with (out/(mode+'-stdout.log')).open('w') as stdout,(out/(mode+'-stderr.log')).open('w') as stderr:
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr); start=time.monotonic()
            try:
                while proc.poll() is None:
                    assert time.monotonic()-start<150,'timeout including destruction'
                    try:
                        lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                        mem={x.split(':')[0]:int(x.split()[1]) for x in lines if x.startswith(('VmRSS:','VmSwap:'))}
                        row['memory'].append(dict(seconds=time.monotonic()-start,**mem)); assert not mem.get('VmSwap',0),'worker swap'
                    except FileNotFoundError:pass
                    time.sleep(1)
                row['exit_code']=proc.returncode; assert proc.returncode==0
            finally:
                if proc.poll() is None:proc.terminate(); proc.wait(timeout=30); row['terminated']=True
                save()
        print('PASS',mode,flush=True)
    hashes={mode:sha(out/(mode+'.bin')) for mode in ('submit','wait','short')}
    assert len(set(hashes.values()))==1
    r.update(complete=True,passed=True,fresh_worker_logits=hashes)
except Exception as error:r['error']=repr(error); raise
finally:save()

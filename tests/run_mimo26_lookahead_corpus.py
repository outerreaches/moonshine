"""Freeze broadening traces and compare an opt-in lookahead candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from mimo26_evidence import evidence_root

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
p.add_argument('--candidate',action='store_true');p.add_argument('--reference',type=Path)
p.add_argument('--default-off',action='store_true',help='rebuild candidate sources but keep lookahead disabled')
p.add_argument('--tokens',type=int,default=128);p.add_argument('--chunk',type=int,default=32)
p.add_argument('--case',type=int,choices=range(4),help='one case; 3 is mixed-domain; default runs original cases 0..2')
p.add_argument('--timeout',type=float,default=400)
a=p.parse_args()
assert 1<=a.tokens<=8192 and 1<=a.chunk<=128 and 0<a.timeout<=3600
assert not a.default_off or a.candidate
assert not (a.candidate and not a.default_off) or a.reference,'lookahead requires an exact-output reference'
out=a.output;out.mkdir();repo=Path(__file__).resolve().parents[1]
vault=evidence_root()
base=vault/'mimo26-takeover-20260922/route';worker=vault/'mimo26-guarded-reset-20260922/final'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=dict(complete=False,passed=False,candidate=a.candidate,lookahead_enabled=a.candidate and not a.default_off,commands=[],inputs={},memory=[])
r['workload']=dict(tokens=a.tokens,chunk=a.chunk,cases=[a.case] if a.case is not None else [0,1,2],slots=48,decode_steps=2,context=max(256,a.tokens+2))
r['timeout_seconds']=a.timeout
def save():(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
try:
    br=json.loads((base/'report.json').read_text())
    wr=json.loads((worker/'report.json').read_text());assert wr['passed'] and wr['complete']
    objects=[]
    for name,digest in br['inputs'].items():
        if name.endswith('.o'):
            path=worker/'worker.o' if name=='mimo26_gpu_worker.o' else base/name
            if name!='mimo26_gpu_worker.o':assert sha(path)==digest
            shutil.copy2(path,out/name);r['inputs'][name]=sha(path);objects.append(str(out/name))
    sources=['tests/mimo26_lookahead_corpus.cu','tests/mimo26_route_trace_wrap.cu','tests/run_mimo26_lookahead_corpus.py',
             'mimo26_gpu_worker.h','mimo26_rocm_layer.h']
    if a.candidate:sources+=['mimo26_gpu_worker.cu','mimo26_rocm_layer.cu','k3_expert_cache.c','k3_expert_cache.h']
    for name in sources:shutil.copy2(repo/name,out/Path(name).name);r['inputs'][name]=sha(repo/name)
    flags=['/opt/rocm/bin/hipcc','-O3','-g','-fno-fast-math','--offload-arch=gfx1151','-I'+str(out),'-I'+str(repo)]
    def build(cmd):
        r['commands'].append(cmd)
        with (out/'build.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    if a.candidate:
        for source in ('mimo26_gpu_worker','mimo26_rocm_layer'):
            build(flags+['-c',str(out/(source+'.cu')),'-o',str(out/(source+'.o'))])
        build(['cc','-O3','-g','-I'+str(out),'-c',str(out/'k3_expert_cache.c'),'-o',str(out/'k3_expert_cache.o')])
    build(flags+(['-DLOOKAHEAD_CANDIDATE'] if r['lookahead_enabled'] else [])+['-c',str(out/'mimo26_lookahead_corpus.cu'),'-o',str(out/'driver.o')])
    build(flags+['-c',str(out/'mimo26_route_trace_wrap.cu'),'-o',str(out/'trace.o')])
    build(flags+[str(out/'driver.o'),str(out/'trace.o')]+objects+['-Wl,--wrap=mimo26_rocm_layer_prefill','-lm','-pthread','-lhipblas','-lhipblaslt','-lzstd','-licui18n','-licuuc','-licudata','-o',str(out/'probe')])
    r['built_objects']={p.name:sha(p) for p in out.glob('*.o')};r['binary_sha256']=sha(out/'probe')
    assert not subprocess.run(['fuser','/dev/kfd'],capture_output=True).stdout.strip(),'GPU occupied'
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    assert available>70*1024*1024
    cmd=[str(out/'probe'),'/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL',str(out/'output'),str(a.tokens),str(a.chunk),str(r['workload']['cases'][0]),str(len(r['workload']['cases']))];r['commands'].append(cmd);save()
    print('Starting '+str(out)+' '+json.dumps(r['workload']),flush=True)
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr);start=time.monotonic()
        try:
            while proc.poll() is None:
                assert time.monotonic()-start<a.timeout,'timeout'
                try:
                    lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                    mem={x.split(':')[0]:int(x.split()[1]) for x in lines if x.startswith(('VmRSS:','VmSwap:'))}
                    r['memory'].append(dict(seconds=time.monotonic()-start,**mem));assert not mem.get('VmSwap',0),'worker swap'
                except FileNotFoundError:pass
                if len(r['memory'])%10==0:save()
                time.sleep(1)
            r['exit_code']=proc.returncode;assert proc.returncode==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30);r['terminated']=True
    r['results']=[json.loads(x) for x in (out/'stdout.log').read_text().splitlines()];assert len(r['results'])==len(r['workload']['cases'])
    assert [row['case'] for row in r['results']]==r['workload']['cases']
    assert all(row['tokens']==a.tokens and row['chunk']==a.chunk for row in r['results'])
    r['outputs']={p.name:sha(p) for p in out.glob('output-*.bin')};assert len(r['outputs'])==4*len(r['workload']['cases'])
    r['route_sha256']=sha(out/'stderr.log')
    if a.reference:
        ref=json.loads((a.reference/'report.json').read_text());assert ref['complete'] and ref['passed']
        assert r['workload']==ref.get('workload',dict(tokens=128,chunk=32,cases=[0,1,2],slots=48,decode_steps=2,context=256)),'workload mismatch'
        assert r['outputs']==ref['outputs'],'full logits or token IDs differ'
        assert r['route_sha256']==sha(a.reference/'stderr.log'),'route log differs'
        r['reference_report_sha256']=sha(a.reference/'report.json')
    r.update(complete=True,passed=True)
    print(json.dumps(r['results']),flush=True)
except Exception as error:r['error']=repr(error);raise
finally:save()

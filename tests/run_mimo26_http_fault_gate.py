"""Isolated loopback HTTP fault/quarantine and fresh-process recovery gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from mimo26_evidence import evidence_root

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); out=a.output; out.mkdir()
repo=Path(__file__).resolve().parents[1]
vault=evidence_root()
base=vault/'mimo26-takeover-20260922/route'
worker=vault/'mimo26-guarded-reset-20260922/final'
r=dict(complete=False,passed=False,commands=[],runs=[],inputs={})
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save():(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
try:
    for name in ('mimo26_server.cu','mimo26_gpu_worker.h','mimo26_server_slot.c','mimo26_server_slot.h',
                 'tests/mimo26_http_fault_wrap.c','tests/run_mimo26_http_fault_gate.py'):
        shutil.copy2(repo/name,out/Path(name).name); r['inputs'][name]=sha(repo/name)
    br=json.loads((base/'report.json').read_text())
    wr=json.loads((worker/'report.json').read_text()); assert wr['complete'] and wr['passed']
    assert sha(repo/'mimo26_gpu_worker.cu')==wr['sources']['mimo26_gpu_worker.cu']
    assert sha(repo/'mimo26_gpu_worker.h')==wr['sources']['mimo26_gpu_worker.h']
    objects=[str(worker/'worker.o')]; r['inputs'][objects[0]]=sha(Path(objects[0]))
    for name,digest in br['inputs'].items():
        if name.endswith('.o') and name!='mimo26_gpu_worker.o':
            assert sha(base/name)==digest; objects.append(str(base/name)); r['inputs'][str(base/name)]=digest
    flags=['/opt/rocm/bin/hipcc','-O3','-g','-fno-fast-math','--offload-arch=gfx1151','-I'+str(repo)]
    # Compile wrapper from repo: its relative include is deliberate.
    commands=[flags+['-c',str(out/'mimo26_server.cu'),'-o',str(out/'server.o')],
              ['cc','-O2','-g','-I'+str(repo),'-c',str(out/'mimo26_server_slot.c'),'-o',str(out/'slot.o')],
              ['cc','-O2','-g','-c',str(repo/'tests/mimo26_http_fault_wrap.c'),'-o',str(out/'wrap.o')],
              flags+[str(out/n) for n in ('server.o','slot.o','wrap.o')]+objects+
              ['-Wl,--wrap=k3_io_uring_submit','-lm','-pthread','-lhipblas','-lhipblaslt','-lzstd','-licui18n','-licuuc','-licudata','-o',str(out/'server')]]
    with (out/'build.log').open('w') as log:
        for cmd in commands:
            r['commands'].append(cmd); subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    r['binary_sha256']=sha(out/'server')
    body={'messages':[{'role':'user','content':'Say hello.'}],'enable_thinking':False,'max_tokens':1}
    for variant in ('control','fault','recreated'):
        assert not subprocess.run(['fuser','/dev/kfd'],capture_output=True).stdout.strip(),'GPU occupied'
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        assert available>40*1024*1024
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        row=dict(variant=variant,port=port,memory=[],responses=[],monitor_errors=[]);r['runs'].append(row);save()
        cmd=[str(out/'server'),'/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL',
             '--host','127.0.0.1','--port',str(port),'--slots','16','--context','256']
        r['commands'].append(cmd);env=os.environ.copy();env.pop('MIMO26_TEST_HTTP_SUBMIT_FAILURE',None)
        if variant=='fault':env['MIMO26_TEST_HTTP_SUBMIT_FAILURE']='1'
        row['fault_injection']=variant=='fault'
        with (out/(variant+'-stdout.log')).open('w') as stdout,(out/(variant+'-stderr.log')).open('w') as stderr:
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env);start=time.monotonic();stop=threading.Event()
            def monitor():
                while not stop.is_set() and proc.poll() is None:
                    try:
                        lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                        mem={x.split(':')[0]:int(x.split()[1]) for x in lines if x.startswith(('VmRSS:','VmSwap:'))}
                        row['memory'].append(dict(seconds=time.monotonic()-start,**mem))
                        if mem.get('VmSwap',0) or time.monotonic()-start>180:
                            row['monitor_errors'].append('swap or timeout');proc.terminate();return
                    except FileNotFoundError:pass
                    stop.wait(1)
            thread=threading.Thread(target=monitor);thread.start()
            def request(path,payload=None):
                req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,
                    data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                try:response=urllib.request.urlopen(req,timeout=90)
                except urllib.error.HTTPError as error:response=error
                with response:result=dict(path=path,status=response.code,body=json.loads(response.read()))
                row['responses'].append(result);return result
            try:
                while True:
                    assert proc.poll() is None,'server exited during startup'
                    try:
                        ready=request('/health');break
                    except urllib.error.URLError:
                        assert time.monotonic()-start<100;time.sleep(0.5)
                assert ready['status']==200 and ready['body']['ready']
                first=request('/v1/chat/completions',body)
                if variant=='fault':
                    assert first['status']==500 and first['body']['error']['code']=='decode_failed'
                    for _ in range(5):
                        health=request('/health')
                        assert health['status']==200 and health['body']['phase']=='quarantined' and not health['body']['ready']
                        assert health['body']['faults']==1 and health['body']['recoveries']==0
                        refused=request('/v1/chat/completions',body);assert refused['status']==503
                    health=request('/health')['body']
                    assert health['admitted']==1 and health['rejected_quarantined']==5
                else:
                    assert first['status']==200 and first['body']['usage']['completion_tokens']==1
                    assert request('/health')['body']['ready']
            finally:
                if proc.poll() is None:proc.terminate()
                try:row['exit_code']=proc.wait(timeout=30)
                finally:stop.set();thread.join(timeout=5);save()
            assert row['exit_code']==0 and not row['monitor_errors']
        print('PASS',variant,flush=True)
    controls=[next(x['body'] for x in run['responses'] if x['path']=='/v1/chat/completions') for run in (r['runs'][0],r['runs'][2])]
    assert controls[0]['choices']==controls[1]['choices'] and controls[0]['usage']==controls[1]['usage']
    faultlog=(out/'fault-stderr.log').read_text()
    assert faultlog.count('TEST_HTTP_FAULT')==1 and faultlog.count('recovery refused; worker recreation required')==3
    r.update(complete=True,passed=True,scope='loopback nonstreaming injected I/O fault; explicit fresh-process recovery, not automatic recreation or hardware-fault qualification')
except Exception as error:r['error']=repr(error);raise
finally:save()

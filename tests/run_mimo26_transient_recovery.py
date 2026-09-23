"""Real submitted-read failure -> automatic replacement -> exact new request; bounded and isolated."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from run_mimo26_server_lookahead_gate import A, MODEL, sha

REPO=Path(__file__).resolve().parents[1]
BASE=Path('/home/alex/Obsidian/beelink-knowledge/Projects/Moonshine/Evidence/mimo26-server-lookahead-20260923/server')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--server-build',type=Path,default=BASE);a=p.parse_args()
    out=a.output.resolve();out.mkdir()
    prior=json.loads((BASE/'report.json').read_text());assert prior['complete'] and prior['passed']
    candidate=json.loads((a.server_build/'report.json').read_text())
    assert candidate['complete'] and candidate['passed']
    assert sha(a.server_build/'server-observed')==candidate['binaries']['server-observed']
    assert subprocess.run(['fuser','/dev/kfd'],capture_output=True).returncode==1
    assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>70*1048576
    for name in ('tools/mimo26_supervise.py','tests/mimo26_once_fault_launcher.py','tests/run_mimo26_transient_recovery.py'):
        shutil.copy2(REPO/name,out/Path(name).name)
    (out/'mimo26_once_fault_launcher.py').chmod(0o700)
    shutil.copy2(a.server_build/'server-observed',out/'server-observed')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    cmd=[sys.executable,str(out/'mimo26_supervise.py'),str(out/'mimo26_once_fault_launcher.py'),MODEL,
         '--port',str(port),'--slots','16','--context','1024','--prefill-chunk','64','--expert-lookahead','on','--restarts','1']
    env=os.environ.copy();env['MIMO26_ONCE_TEST_DIRECTORY']=str(out);env['MIMO26_ONCE_TEST_BINARY']=str(out/'server-observed')
    r=dict(complete=False,passed=False,command=cmd,responses=[],memory=[],guard_errors=[],
           hashes={x.name:sha(x) for x in out.iterdir() if x.is_file()},reference_sha256=sha(BASE/'report.json'),
           source_build=str(a.server_build.resolve()),source_build_sha256=sha(a.server_build/'report.json'))
    def save():(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    def events():
        rows=[]
        for line in (out/'stdout.log').read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if isinstance(row,dict) and 'event' in row:rows.append(row)
        return rows
    def request(path,body=None):
        req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,
            data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        try:response=urllib.request.urlopen(req,timeout=120)
        except urllib.error.HTTPError as error:response=error
        with response:result=dict(status=response.code,body=json.loads(response.read()))
        r['responses'].append(result);save();return result
    save()
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env);start=time.monotonic();stop=threading.Event()
        def monitor():
            while not stop.wait(.5):
                rows=events();exited={x['pid'] for x in rows if x['event']=='exited'}
                for x in rows:
                    if x['event']!='started' or x['pid'] in exited:continue
                    try:status=Path(f"/proc/{x['pid']}/status").read_text()
                    except FileNotFoundError:continue
                    swap=int(next((x.split()[1] for x in status.splitlines() if x.startswith('VmSwap:')),'0'))
                    r['memory'].append(dict(pid=x['pid'],seconds=time.monotonic()-start,status=status))
                    if swap:r['guard_errors'].append('worker swap');proc.terminate();return
                if time.monotonic()-start>300:r['guard_errors'].append('timeout');proc.terminate();return
        thread=threading.Thread(target=monitor);thread.start()
        try:
            def wait_ready(number):
                while True:
                    assert proc.poll() is None and time.monotonic()-start<290 and not r['guard_errors']
                    ready=[x for x in events() if x['event']=='ready']
                    if len(ready)>=number:return ready[number-1]['pid']
                    time.sleep(.25)
            first=wait_ready(1)
            health=request('/health')['body'];assert health['expert_lookahead'] and health['prefill_chunk']==64
            failed=request('/v1/chat/completions',A)
            assert failed['status']==500 and failed['body']['error']['code']=='decode_failed'
            print('PASS injected first-child failure',flush=True)
            second=wait_ready(2);assert first!=second and not Path(f'/proc/{first}').exists()
            health=request('/health')['body'];assert health['served']==0 and health['faults']==0
            answer=request('/v1/chat/completions',A);assert answer['status']==200
            reference=next(x for x in prior['runs'] if x['variant']=='on')['answer_a']
            assert answer['body']['choices']==reference['choices'] and answer['body']['usage']==reference['usage']
            outputs={x.name:sha(x) for x in (out/'replacement').glob('*.bin')}
            expected={x.name:sha(x) for x in (BASE/'recreated').glob('*.bin')}
            assert outputs==expected and len(outputs)>2
            assert not (out/'first'/'1-prefill.bin').exists()
            health=request('/health')['body'];assert health['ready'] and health['served']==1 and health['faults']==0
            r['outputs']=outputs;r['final_health']=health
            proc.terminate();r['exit_code']=proc.wait(timeout=40)
            rows=events();r['events']=rows
            assert r['exit_code']==0 and not r['guard_errors']
            assert len([x for x in rows if x['event']=='started'])==2
            assert len([x for x in rows if x['event']=='exited'])==2
            assert all(x['returncode']==0 for x in rows if x['event']=='exited')
            assert [x['reason'] for x in rows if x['event']=='retiring']==['quarantined','stopped']
            assert next(i for i,x in enumerate(rows) if x['event']=='exited') < [i for i,x in enumerate(rows) if x['event']=='started'][1]
            log=(out/'stderr.log').read_text();assert log.count('TEST_HTTP_FAULT pending=8')==1
            assert log.count('TEST_PREFILL_BEGIN request=1 ')==2
            r.update(complete=True,passed=True,scope='Injected transient I/O failure, owned exit before automatic replacement; no replay; exact new request. Not hardware-fault recovery.')
            print('PASS automatic replacement and full outputs',flush=True)
        except BaseException as error:r['error']=repr(error);raise
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=40)
            stop.set();thread.join(timeout=5);save()


if __name__=='__main__':main()

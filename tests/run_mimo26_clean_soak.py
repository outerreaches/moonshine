"""Deterministic bounded clean-binary mixed soak, not a duration-only happy-path loop."""
import argparse
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
from run_mimo26_server_lookahead_gate import A,B,C,CANCEL,MODEL,sha
from run_mimo26_transient_recovery import BASE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--server-build',type=Path,default=BASE);a=p.parse_args()
    out=a.output.resolve();out.mkdir();prior=json.loads((BASE/'report.json').read_text())
    assert prior['complete'] and prior['passed']
    candidate=json.loads((a.server_build/'report.json').read_text())
    assert candidate['complete'] and candidate['passed']
    assert sha(a.server_build/'server-clean')==candidate['binaries']['server-clean']
    assert subprocess.run(['fuser','/dev/kfd'],capture_output=True).returncode==1
    assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>70*1048576
    shutil.copy2(a.server_build/'server-clean',out/'server-clean');shutil.copy2(__file__,out/Path(__file__).name)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    cmd=[str(out/'server-clean'),MODEL,'--host','127.0.0.1','--port',str(port),'--slots','16','--context','1024','--prefill-chunk','64','--expert-lookahead','on']
    r=dict(complete=False,passed=False,command=cmd,binary_sha256=sha(out/'server-clean'),responses=[],memory=[],idle=[],guard_errors=[],
           source_build=str(a.server_build.resolve()),source_build_sha256=sha(a.server_build/'report.json'),
           reference_sha256=sha(BASE/'report.json'))
    def save():(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        env=os.environ.copy()
        for key in tuple(env):
            if key.startswith('MIMO26_TEST_'):env.pop(key)
        proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env);start=time.monotonic();stop=threading.Event()
        def sample():
            status=Path(f'/proc/{proc.pid}/status').read_text()
            fields={x.split(':')[0]:int(x.split()[1]) for x in status.splitlines() if x.startswith(('VmRSS:','VmSwap:','Threads:'))}
            return dict(seconds=time.monotonic()-start,fd_count=len(list(Path(f'/proc/{proc.pid}/fd').iterdir())),**fields)
        def monitor():
            while not stop.wait(.5):
                try:row=sample()
                except OSError as error:
                    if stop.is_set() or proc.poll() is not None:return
                    r['guard_errors'].append('monitor error: '+repr(error));proc.terminate();return
                r['memory'].append(row)
                if row.get('VmSwap',0) or time.monotonic()-start>600:
                    r['guard_errors'].append('worker swap or timeout');proc.terminate();return
        thread=threading.Thread(target=monitor);thread.start()
        def request(path,body=None):
            req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            try:response=urllib.request.urlopen(req,timeout=120)
            except urllib.error.HTTPError as error:response=error
            with response:
                raw=response.read().decode()
                if body and body.get('stream') and response.code==200:
                    frames=[line[6:] for line in raw.splitlines() if line.startswith('data: ')]
                    assert frames[-1]=='[DONE]';parsed=[json.loads(x) for x in frames[:-1]]
                else:parsed=json.loads(raw)
                row=dict(path=path,status=response.code,body=parsed)
            r['responses'].append(row);save();return row
        def ready():
            deadline=time.monotonic()+120
            while True:
                assert proc.poll() is None and time.monotonic()<deadline and not r['guard_errors']
                try:
                    row=request('/health')
                    if row['status']==200 and row['body']['ready']:
                        r['idle'].append(dict(health=row['body'],**sample()));return row['body']
                except (urllib.error.URLError,ConnectionError):pass
                time.sleep(.25)
        try:
            health=ready();assert health['expert_lookahead'] and health['prefill_chunk']==64
            baseline=next(x for x in prior['runs'] if x['variant']=='on')
            for cycle in range(3):
                for body,key in ((A,'answer_a'),(B,'answer_b'),(C,'answer_c')):
                    answer=request('/v1/chat/completions',body);assert answer['status']==200
                    assert answer['body']['choices']==baseline[key]['choices'] and answer['body']['usage']==baseline[key]['usage']
                streamed=request('/v1/chat/completions',dict(A,stream=True));assert streamed['status']==200
                content=''.join(x['choices'][0]['delta'].get('content','') for x in streamed['body'] if x.get('choices'))
                assert content==baseline['answer_a']['choices'][0]['message']['content']
                for bad,status in ((dict(A,temperature=.5),400),(dict(A,model='wrong-model'),404),(dict(A,messages=[dict(role='user',content='word '*1500)]),400)):
                    refusal=request('/v1/chat/completions',bad)
                    assert refusal['status']==status and refusal['body']['error']['code']
                before=ready();wire=json.dumps(CANCEL).encode()
                with socket.create_connection(('127.0.0.1',port),timeout=30) as peer:
                    peer.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(wire)}\r\n\r\n').encode()+wire)
                    # Health queues behind accept of the first request; wait for actual busy refusal.
                    deadline=time.monotonic()+30
                    while True:
                        health=request('/health')
                        if health['status']==503:break
                        assert time.monotonic()<deadline;time.sleep(.1)
                    busy=request('/v1/chat/completions',B)
                    assert busy['status']==503 and busy['body']['error']['code']=='slot_busy'
                after=ready();assert after['cancelled']==before['cancelled']+1 and after['faults']==0
                answer=request('/v1/chat/completions',A)
                assert answer['status']==200 and answer['body']['choices']==baseline['answer_a']['choices']
                ready();print('PASS clean mixed cycle',cycle+1,flush=True)
            # Longer decode than the six-token functional gate, still bounded.
            long=request('/v1/chat/completions',dict(A,max_tokens=32));assert long['status']==200
            assert 0<long['body']['usage']['completion_tokens']<=32
            final=ready();assert final['faults']==0 and final['cancelled']==3
            warmed=r['idle'][2:];assert len(warmed)>4
            assert max(x['fd_count'] for x in warmed)-min(x['fd_count'] for x in warmed)<=2
            assert warmed[-1]['VmRSS']-warmed[0]['VmRSS']<512*1024
            r['final_health']=final;r['elapsed_seconds']=time.monotonic()-start
            stop.set();thread.join(timeout=5)
            proc.terminate();r['exit_code']=proc.wait(timeout=40);assert r['exit_code']==0 and not r['guard_errors']
            r.update(complete=True,passed=True,scope='Three deterministic mixed cycles, clean cache16/chunk64/context1024/lookahead-on. Bounded soak, not sustained multi-hour production certification; no new upstream/tool qualification.')
        except BaseException as error:r['error']=repr(error);raise
        finally:
            stop.set();thread.join(timeout=5)
            if proc.poll() is None:proc.terminate();proc.wait(timeout=40)
            stop.set();thread.join(timeout=5);save()


if __name__=='__main__':main()

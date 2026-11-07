"""Bounded clean-binary cold/retained request comparison, never a decode-rate claim."""
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
from run_mimo26_server_lookahead_gate import A, B, C, MODEL, sha


def host():
    data = {x.split(':')[0]: int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()
            if x.startswith(('MemAvailable:', 'SwapFree:', 'SwapTotal:'))}
    data.update({key: int(value) for key, value in (x.split() for x in Path('/proc/vmstat').read_text().splitlines())
                 if key in ('pswpin', 'pswpout')})
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir()
    build = a.build.resolve(); source = json.loads((build/'report.json').read_text())
    assert source['complete'] and source['passed'] and source['retain_experts'] == 'on'
    assert source['slots'] == 128 and source['chunk'] == 128 and source['context'] == 2048
    assert sha(build/'server-clean') == source['binaries']['server-clean']
    shutil.copy2(build/'server-clean',out/'server-clean')
    shutil.copy2(__file__,out/Path(__file__).name)
    reference = next(x for x in source['runs'] if x['variant']=='on')
    r = dict(complete=False, passed=False, parent_report_sha256=sha(build/'report.json'),
             inputs={x.name:sha(x) for x in out.iterdir()}, runs=[],
             scope='clean binary short mixed requests; exact response/usage and worker read_bytes; end-to-end latency, not pure decode or general language quality')
    def save(): (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    try:
        # ABBA process order balances some startup/thermal/time-order effects.
        for trial, mode in enumerate(('off','on','on','off')):
            assert subprocess.run(['fuser','/dev/kfd'],capture_output=True).returncode == 1
            before_host=host(); assert before_host['MemAvailable'] > 100*1048576
            with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
            path=out/f'{trial}-{mode}';path.mkdir()
            cmd=[str(out/'server-clean'),MODEL,'--host','127.0.0.1','--port',str(port),
                 '--slots','128','--context','2048','--prefill-chunk','128','--expert-lookahead','on',
                 '--retain-experts',mode]
            row=dict(mode=mode,command=cmd,memory=[],requests=[],errors=[],host_before=before_host)
            r['runs'].append(row);save()
            env={k:v for k,v in os.environ.items() if not k.startswith('MIMO26_')}
            with (path/'stdout.log').open('w') as stdout,(path/'stderr.log').open('w') as stderr:
                proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env)
                stop=threading.Event(); started=time.monotonic()
                def monitor():
                    while not stop.is_set() and proc.poll() is None:
                        try:
                            status=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                            mem={x.split(':')[0]:int(x.split()[1]) for x in status if x.startswith(('VmSwap:','VmRSS:'))}
                        except FileNotFoundError: break
                        if not mem:
                            if proc.poll() is None:
                                row['errors'].append('worker memory fields unavailable');proc.terminate()
                            break
                        mem.update(host());mem['seconds']=time.monotonic()-started;row['memory'].append(mem)
                        if mem['VmSwap'] or mem['MemAvailable']<16*1048576 or mem['seconds']>480 or (
                                mem['pswpout']-before_host['pswpout'])*4096>512*1048576:
                            row['errors'].append('worker swap / host floor / timeout / >512MiB host page-out guard')
                            proc.terminate();return
                        stop.wait(.5)
                thread=threading.Thread(target=monitor);thread.start()
                def read_bytes():
                    return int(next(x.split()[1] for x in Path(f'/proc/{proc.pid}/io').read_text().splitlines()
                                    if x.startswith('read_bytes:')))
                def request(body=None):
                    endpoint='/health' if body is None else '/v1/chat/completions'
                    req=urllib.request.Request(f'http://127.0.0.1:{port}'+endpoint,
                        data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=120) as response: return json.loads(response.read())
                try:
                    while True:
                        assert proc.poll() is None and time.monotonic()-started<120 and not row['errors']
                        try:
                            health=request()
                            if health['ready']:break
                        except (urllib.error.URLError,ConnectionError):pass
                        time.sleep(.25)
                    assert health['retain_experts']==(mode=='on') and health['expert_slots']==128
                    assert health['prefill_chunk']==128 and health['context']==2048 and health['expert_lookahead']
                    row['startup_seconds']=time.monotonic()-started
                    for index,(body,label) in enumerate(((A,'a'),(B,'b'),(C,'c'),(A,'a'),(B,'b'),(C,'c'))):
                        before=request();disk=read_bytes();began=time.monotonic()
                        answer=request(body);seconds=time.monotonic()-began;reads=read_bytes()-disk
                        after=request()
                        expected=reference['answer_'+label]
                        assert answer['choices']==expected['choices'] and answer['usage']==expected['usage']
                        assert after['ready'] and after['faults']==0
                        row['requests'].append(dict(index=index,label=label,body=body,answer=answer,
                            seconds=seconds,process_read_bytes=reads,uploads=after['expert_uploads']-before['expert_uploads'],
                            health_before=before,health_after=after))
                        print(f'PASS trial={trial} retained={mode} request={index}/{label} seconds={seconds:.3f} read_GiB={reads/2**30:.3f}',flush=True)
                        save()
                    proc.terminate();row['exit_code']=proc.wait(timeout=60)
                    assert row['exit_code']==0 and not row['errors']
                    assert len(row['memory'])>10 and all(x['VmSwap']==0 for x in row['memory'])
                    row['passed']=True
                finally:
                    if proc.poll() is None:proc.terminate();proc.wait(timeout=60)
                    stop.set();thread.join(timeout=5);row['host_after']=host();save()
        summary={}
        for mode in ('off','on'):
            rows=[q for run in r['runs'] if run['mode']==mode for q in run['requests']]
            summary[mode]=dict(requests=len(rows),seconds=sum(q['seconds'] for q in rows),
                               process_read_bytes=sum(q['process_read_bytes'] for q in rows),
                               uploads=sum(q['uploads'] for q in rows))
        r.update(complete=True,passed=True,summary=summary)
    except BaseException as error:r['error']=repr(error);raise
    finally:save()


if __name__=='__main__':main()

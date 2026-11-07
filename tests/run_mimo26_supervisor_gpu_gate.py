"""Persistent injected GPU-server fault: one replacement, then fail closed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from mimo26_evidence import evidence_root

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output;out.mkdir();repo=Path(__file__).resolve().parents[1]
base=evidence_root()/'mimo26-http-quarantine-20260922/run'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
r=dict(complete=False,passed=False,responses=[],memory=[])
def save():(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
def events():
    rows=[]
    for line in (out/'stdout.log').read_text().splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if isinstance(row,dict) and 'event' in row:rows.append(row)
    return rows
try:
    old=json.loads((base/'report.json').read_text());assert old['complete'] and old['passed']
    assert sha(base/'server')==old['binary_sha256']
    shutil.copy2(repo/'tools/mimo26_supervise.py',out/'mimo26_supervise.py')
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    r['supervisor_sha256']=sha(out/'mimo26_supervise.py');r['server_sha256']=sha(base/'server')
    assert not subprocess.run(['fuser','/dev/kfd'],capture_output=True).stdout.strip()
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    assert available>40*1024*1024
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    cmd=[sys.executable,str(out/'mimo26_supervise.py'),str(base/'server'),'/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL',
         '--port',str(port),'--slots','16','--context','256','--restarts','1']
    r['command']=cmd;env=os.environ.copy();env['MIMO26_TEST_HTTP_SUBMIT_FAILURE']='1';save()
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env);start=time.monotonic()
        sent=set()
        try:
            while proc.poll() is None:
                assert time.monotonic()-start<240,'total timeout'
                rows=events()
                exited={x['pid'] for x in rows if x['event']=='exited'}
                for row in rows:
                    if row['event']=='started' and row['pid'] not in exited:
                        try:
                            lines=Path(f"/proc/{row['pid']}/status").read_text().splitlines()
                            mem={x.split(':')[0]:int(x.split()[1]) for x in lines if x.startswith(('VmRSS:','VmSwap:'))}
                            r['memory'].append(dict(pid=row['pid'],seconds=time.monotonic()-start,**mem));assert not mem.get('VmSwap',0)
                        except FileNotFoundError:pass
                    if row['event']=='ready' and row['pid'] not in sent:
                        assert len(sent)<2;sent.add(row['pid'])
                        payload={'messages':[{'role':'user','content':'Say hello.'}],'enable_thinking':False,'max_tokens':1}
                        req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                        try:response=urllib.request.urlopen(req,timeout=30)
                        except urllib.error.HTTPError as error:response=error
                        with response:result=dict(pid=row['pid'],status=response.code,body=json.loads(response.read()))
                        r['responses'].append(result);assert result['status']==500
                time.sleep(0.5)
            r['exit_code']=proc.returncode
            rows=events();r['events']=rows
            assert proc.returncode==1 and len(sent)==2
            assert sum(x['event']=='started' for x in rows)==2
            assert sum(x['event']=='exited' for x in rows)==2
            assert rows[-1]['event']=='restart_budget_exhausted'
            assert all(x['returncode']==0 for x in rows if x['event']=='exited')
            assert next(i for i,x in enumerate(rows) if x['event']=='exited') < [i for i,x in enumerate(rows) if x['event']=='started'][1]
            assert (out/'stderr.log').read_text().count('TEST_HTTP_FAULT')==2
            r.update(complete=True,passed=True)
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=40)
except Exception as error:r['error']=repr(error);raise
finally:save()

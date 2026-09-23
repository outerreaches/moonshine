"""One clean-reset retention feasibility run, against frozen route-screen logits."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--slots',type=int,choices=(48,96,144),default=48)
parser.add_argument('--baseline',type=Path,default=Path('/tmp/mimo26-route-screen-20260922'))
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1];base=args.baseline
out=args.output;out.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
baseline=json.loads((base/'report.json').read_text());assert baseline['complete'] and baseline['passed']
report=dict(complete=False,passed=False,commands=[],memory=[],baseline_report_sha256=sha(base/'report.json'),
    slots=args.slots,
    scope='test-only no-op weight-cache reset, clean repeated natural prompt; not server integration, fault recovery or mixed-request qualification')
def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    for name in ('capture_mimo26_routes.cu','mimo26_keep_weights_wrap.c','run_mimo26_retention_probe.py'):
        shutil.copy2(repo/'tests'/name,out/name)
    report['source_hashes']={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.cu','.c','.py')}
    flags=['/opt/rocm/bin/hipcc','-O3','-g','-fno-fast-math','--offload-arch=gfx1151','-I'+str(repo)]
    objects=[str(base/name) for name in baseline['inputs'] if name.endswith('.o')]
    cmds=[['cc','-O3','-I'+str(repo),'-c',str(out/'mimo26_keep_weights_wrap.c'),'-o',str(out/'wrap.o')],
        flags+['-c',str(out/'capture_mimo26_routes.cu'),'-o',str(out/'driver.o')],
        flags+[str(out/'driver.o'),str(out/'wrap.o')]+objects+
        ['-Wl,--wrap=k3_expert_cache_reset','-lm','-pthread','-lhipblas','-lhipblaslt','-lzstd','-licui18n','-licuuc','-licudata','-o',str(out/'probe')]]
    for name,expected in baseline['inputs'].items():
        if name.endswith('.o'):assert sha(base/name)==expected
    with (out/'build.log').open('w') as log:
        for cmd in cmds:
            report['commands'].append(cmd);subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    if subprocess.run(['fuser','/dev/kfd'],capture_output=True).stdout.strip():raise RuntimeError('GPU occupied')
    mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    assert mem>(70 if args.slots==48 else 110)*1024*1024,'insufficient headroom'
    cmd=[str(out/'probe'),'/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL',str(out/'probe'),str(args.slots)]
    report['commands'].append(cmd);report['binary_sha256']=sha(out/'probe');save()
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr);start=time.monotonic()
        try:
            while proc.poll() is None:
                if time.monotonic()-start>450:raise TimeoutError('probe timeout')
                try:
                    status=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                    values={x.split(':')[0]:int(x.split()[1]) for x in status if x.startswith(('VmRSS:','VmSwap:'))}
                    values['MemAvailable']=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
                    report['memory'].append(dict(seconds=time.monotonic()-start,**values))
                    if values.get('VmSwap',0):raise RuntimeError('worker swap')
                except FileNotFoundError:pass
                time.sleep(2)
            report['exit_code']=proc.returncode
            assert proc.returncode==0
        finally:
            if proc.poll() is None:
                proc.terminate();proc.wait(timeout=60);report['terminated_by_guard']=True;report['exit_code']=proc.returncode
    rows=[json.loads(x) for x in (out/'stdout.log').read_text().splitlines()]
    assert len(rows)==2
    hashes=[sha(out/f'probe-{i}-logits.bin') for i in (0,1)]
    assert hashes==[sha(base/'control-0-logits.bin')]*2
    assert sha(out/'probe-tokens.bin')==sha(base/'control-tokens.bin')
    # New counters are cumulative because no cache reset took place.
    misses=[r['uploads_cumulative'] for r in rows]
    expected={48:(14897,14333),96:(8239,5826),144:(6661,1531)}[args.slots]
    assert (misses[0],misses[1]-misses[0])==expected
    report.update(complete=True,passed=True,results=rows,logits_sha256=hashes,
        cold_misses=misses[0],warm_misses=misses[1]-misses[0])
    print(json.dumps({k:v for k,v in report.items() if k not in ('memory','commands')},indent=2))
except Exception as error:report['error']=repr(error);raise
finally:save()

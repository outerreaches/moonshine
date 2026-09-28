"""Build isolated baseline/candidate and gate full outputs, swap and exit status."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from mimo26_evidence import evidence_root

repo=Path(__file__).resolve().parents[1]
base=evidence_root()/'mimo26-takeover-20260922/route'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('/tmp/mimo26-reset-gate-20260922'))
parser.add_argument('--reuse-baseline',type=Path)
parser.add_argument('--fallback',action='store_true')
args=parser.parse_args()
out=args.output; out.mkdir()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
r=dict(complete=False,passed=False,commands=[],runs=[])
def save(): (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
try:
    for name in ('mimo26_gpu_worker.cu','mimo26_gpu_worker.h','tests/mimo26_reset_gate.cu','tests/mimo26_fallback_reset_gate.cu','tests/run_mimo26_reset_gate.py'):
        shutil.copy2(repo/name,out/Path(name).name)
    r['sources']={p.name:sha(p) for p in out.iterdir()}
    baseline=json.loads((base/'report.json').read_text())
    r['baseline_report_sha256']=sha(base/'report.json')
    objects=[]
    for name,digest in baseline['inputs'].items():
        if name.endswith('.o'):
            assert sha(base/name)==digest
            if name!='mimo26_gpu_worker.o': objects.append(str(base/name))
    flags=['/opt/rocm/bin/hipcc','-O3','-g','-fno-fast-math','--offload-arch=gfx1151','-I'+str(out),'-I'+str(repo)]
    def build(cmd):
        r['commands'].append(cmd)
        with (out/'build.log').open('a') as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    build(flags+['-c',str(out/'mimo26_gpu_worker.cu'),'-o',str(out/'worker.o')])
    for variant in (('baseline','candidate','fallback') if args.fallback else ('baseline','candidate')):
        if variant=='baseline' and args.reuse_baseline:
            old=json.loads((args.reuse_baseline/'report.json').read_text())
            assert old['runs'][0]['variant']=='baseline' and old['runs'][0]['exit_code']==0
            assert old['sources']['mimo26_reset_gate.cu']==r['sources']['mimo26_reset_gate.cu']
            assert old['baseline_report_sha256']==r['baseline_report_sha256']
            files=list(args.reuse_baseline.glob('baseline-*.bin')); assert len(files)==6
            for path in files: shutil.copy2(path,out/path.name)
            r['baseline_reused_from']=str(args.reuse_baseline)
            r['baseline_reused_report_sha256']=sha(args.reuse_baseline/'report.json')
            continue
        driver='mimo26_fallback_reset_gate.cu' if variant=='fallback' else 'mimo26_reset_gate.cu'
        build(flags+(['-DBASELINE'] if variant=='baseline' else [])+['-c',str(out/driver),'-o',str(out/(variant+'.o'))])
        worker=base/'mimo26_gpu_worker.o' if variant=='baseline' else out/'worker.o'
        build(flags+[str(out/(variant+'.o')),str(worker)]+objects+([] if variant=='fallback' else ['-Wl,--wrap=hipMemcpy'])+['-lm','-pthread','-lhipblas','-lhipblaslt','-lzstd','-licui18n','-licuuc','-licudata','-o',str(out/variant)])
        assert not subprocess.run(['fuser','/dev/kfd'],capture_output=True).stdout.strip(),'GPU occupied'
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        assert available>40*1024*1024
        row=dict(variant=variant,sha256=sha(out/variant),memory=[]); r['runs'].append(row); save()
        cmd=[str(out/variant),'/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL',str(out/variant)]
        if variant=='fallback':cmd.pop()
        r['commands'].append(cmd)
        with (out/(variant+'-stdout.log')).open('w') as stdout,(out/(variant+'-stderr.log')).open('w') as stderr:
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr); start=time.monotonic()
            try:
                while proc.poll() is None:
                    assert time.monotonic()-start<360,'timeout'
                    try:
                        lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                        mem={x.split(':')[0]:int(x.split()[1]) for x in lines if x.startswith(('VmRSS:','VmSwap:'))}
                        row['memory'].append(dict(seconds=time.monotonic()-start,**mem))
                        assert not mem.get('VmSwap',0),'worker swap'
                    except FileNotFoundError: pass
                    time.sleep(1)
                row['exit_code']=proc.returncode; assert proc.returncode==0
            finally:
                if proc.poll() is None: proc.terminate(); proc.wait(timeout=30); row['terminated']=True
                save()
    hashes={p.name:sha(p) for p in out.glob('*.bin')}; assert len(hashes)==12
    for label in ('cold-a','cold-b','warm-a','warm-b','warm-a2','cancel-a'):
        assert hashes['baseline-'+label+'.bin']==hashes['candidate-'+label+'.bin'],label
    r.update(complete=True,passed=True,logits=hashes)
    print('PASS baseline/candidate 12 full output vectors; mixed prompts, cancellation and partial-copy fault gate')
except Exception as error:
    r['error']=repr(error); raise
finally: save()

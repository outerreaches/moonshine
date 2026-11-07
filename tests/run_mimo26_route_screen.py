"""Freeze existing objects and compare full logits with/without link-only tracing."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

repo=Path(__file__).resolve().parents[1]
out=Path('/tmp/mimo26-route-screen-20260922');out.mkdir()
root='/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL'
objects=['mimo26_gpu_worker','mimo26_rocm_layer','mimo26_rocm_ops','k3_rocm_ops',
    'mimo26_weights','mimo26_kv','mimo26_manifest','mimo26_architecture','mimo26_attention',
    'mimo26_ops','mimo26_router','glm53_fp8_oracle','k3_safetensors','k3_json',
    'k3_io_uring','k3_expert_cache','mimo26_tokenizer']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(complete=False,passed=False,commands=[],inputs={},runs=[],
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    for name in objects:
        path=repo/(name+'.o');shutil.copy2(path,out/path.name);report['inputs'][path.name]=sha(path)
    for source in ('tests/capture_mimo26_routes.cu','tests/mimo26_route_trace_wrap.cu'):
        shutil.copy2(repo/source,out/Path(source).name);report['inputs'][source]=sha(repo/source)
    report['tokenizer_sha256']=sha(Path(root)/'tokenizer.json')
    report['index_sha256']=sha(Path(root)/'model.safetensors.index.json')
    flags=['/opt/rocm/bin/hipcc','-O3','-g','-fno-fast-math','--offload-arch=gfx1151','-I'+str(repo)]
    commands=[]
    for name in ('capture_mimo26_routes','mimo26_route_trace_wrap'):
        commands.append(flags+['-c',str(out/(name+'.cu')),'-o',str(out/(name+'.o'))])
    base=flags+[str(out/'capture_mimo26_routes.o')]+[str(out/(x+'.o')) for x in objects]
    libs=['-lm','-pthread','-lhipblas','-lhipblaslt','-lzstd','-licui18n','-licuuc','-licudata']
    commands.extend([base+libs+['-o',str(out/'control')],
        base+[str(out/'mimo26_route_trace_wrap.o'),'-Wl,--wrap=mimo26_rocm_layer_prefill']+libs+['-o',str(out/'trace')]])
    with (out/'build.log').open('w') as log:
        for cmd in commands:
            report['commands'].append(cmd);save();subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    for variant in ('control','trace'):
        if subprocess.run(['fuser','/dev/kfd'],capture_output=True).stdout.strip():raise RuntimeError('GPU occupied')
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        if available<70*1024*1024:raise RuntimeError('insufficient available RAM for bounded probe')
        cmd=[str(out/variant),root,str(out/variant)]
        report['commands'].append(cmd);row=dict(variant=variant,binary_sha256=sha(out/variant),memory=[]);report['runs'].append(row);save()
        with (out/f'{variant}-stdout.log').open('w') as stdout,(out/f'{variant}-stderr.log').open('w') as stderr:
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr);start=time.monotonic()
            try:
                while proc.poll() is None:
                    if time.monotonic()-start>450:raise TimeoutError('bounded capture timeout')
                    try:
                        lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                        values={line.split(':')[0]:int(line.split()[1]) for line in lines if line.startswith(('VmRSS:','VmSwap:'))}
                        row['memory'].append(dict(seconds=time.monotonic()-start,**values))
                        if values.get('VmSwap',0):raise RuntimeError('worker swap observed')
                    except FileNotFoundError:pass
                    time.sleep(2)
                row['exit_code']=proc.returncode;row['elapsed_seconds']=time.monotonic()-start
                if proc.returncode:raise RuntimeError(f'{variant} failed')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait(timeout=60)
                save()
        print('PASS',variant,flush=True)
    hashes={p.name:sha(p) for p in out.glob('*-logits.bin')}
    assert len(hashes)==4 and len(set(hashes.values()))==1,'logit mismatch'
    assert sha(out/'control-tokens.bin')==sha(out/'trace-tokens.bin')
    report.update(complete=True,passed=True,logits=hashes)
except Exception as error:report['error']=repr(error);raise
finally:save()

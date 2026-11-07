"""Bounded mapped-input MiMo GPU codec gate; no inference/store/runtime changes.

Build first with hipcc -O3 -std=c++17 --offload-arch=gfx1151 -fPIC -shared
tests/mimo26_rans_gpu_gate.cu -o /tmp/FRESH/gate.so. Uses the prior CPU screen's
pinned input hashes and training histograms, never new training on held-outs.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


class Result(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
                ('raw_bytes', 'block_bytes', 'tiles', 'raw_tiles', 'fault_checks', 'host_faults', 'device_faults')]
    _fields_ += [(name, C.c_double) for name in ('decode_ms', 'raw_copy_ms', 'checked_admission_ms')]
    _fields_ += [(name, C.c_double*5) for name in ('decode_samples', 'copy_samples', 'admission_samples')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(a):
    screen = json.loads(a.screen.read_text())
    assert screen['complete'] and screen['passed']
    assert screen['model_revision'] == '3b38d063180c3e4aed9691fdc735f3d10b266ee4'
    assert sha(a.root/'config.json') == screen['config_sha256']
    assert sha(a.root/'model.safetensors.index.json') == screen['index_sha256']
    lib = C.CDLL(str(a.library.resolve())); fn = lib.mimo26_rans_gpu_gate
    fn.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64),
                   C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(Result)]
    fn.restype = C.c_int
    packed = (C.c_uint64*16)(*screen['models']['True'])
    scales = (C.c_uint64*256)(*screen['models']['False'])
    r = dict(complete=False, passed=False, screen=str(a.screen), screen_sha256=sha(a.screen),
             library_sha256=sha(a.library), model_revision=screen['model_revision'],
             sources={name: sha(Path(__file__).with_name(name)) for name in
                      ('mimo26_rans_gpu_gate.cu', 'mimo26_rans_screen.cpp', Path(__file__).name)},
             inputs=[], cases=[], synthetic=[],
             scope='Exact GPU byte recovery and selected fault/canary gates. Test-only publication seam, '
                   'not worker-cache integration. Warm mapped-input timing only: excludes SSD reads, '
                   'input staging, table setup and expert-compute contention. No deployment claim.')
    start = time.monotonic()
    def save():
        (a.output/'report.json').write_text(json.dumps(r, indent=2)+'\n')
    def run(data, tile, waves, layout, faults):
        result = Result()
        assert fn(data, len(data), packed, scales, tile, waves, layout, faults, C.byref(result)) == 1
        return dict(tile=tile, waves_per_block=waves, layout=layout, faults_enabled=faults,
                    **{name: list(getattr(result, name)) if name.endswith('_samples') else getattr(result, name)
                       for name, _ in result._fields_})
    save()
    try:
        fixtures = [bytes(range(256))*64, bytes(16384), b'\x88'*16384,
                    b'\x08\x80'*8192, bytes(range(128)), bytes(range(256))*65]
        for layout in (0, 1):
            for i, data in enumerate(fixtures):
                result = run(data, 16384, 8, layout, 1)
                r['synthetic'].append(dict(fixture=i, sha256=hashlib.sha256(data).hexdigest(), **result))
                save(); print('PASS synthetic', layout, i, flush=True)
        for layer in (1, 24, 47):
            for expert in (97, 255):
                rows = [x for x in screen['matrices'] if x['layer']==layer and x['expert']==expert]
                assert len(rows)==6 and all(x['group']=='heldout' for x in rows)
                data = bytearray()
                for i, row in enumerate(rows):
                    assert row['packed'] == (i%2==0)
                    with (a.root/row['shard']).open('rb') as f:
                        f.seek(row['offset']); part=f.read(row['bytes'])
                    assert len(part)==row['bytes'] and hashlib.sha256(part).hexdigest()==row['sha256']
                    data.extend(part); r['inputs'].append(row)
                data = bytes(data); assert len(data)==13369344
                for tile in (16384, 32768, 65536):
                    for waves in (1, 4, 8):
                        result = run(data, tile, waves, 2, int(expert==97))
                        r['cases'].append(dict(layer=layer, expert=expert, raw_sha256=hashlib.sha256(data).hexdigest(), **result))
                        save(); print('PASS expert', layer, expert, tile, waves, flush=True)
        assert len(r['cases'])==54 and len(r['synthetic'])==12
        assert any(x['raw_tiles'] for x in r['synthetic'])
        assert any(x['raw_tiles']<x['tiles'] for x in r['cases'])
        r.update(complete=True, passed=True, elapsed_seconds=time.monotonic()-start)
    except BaseException as error:
        r['error']=repr(error); raise
    finally:
        save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL'))
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        worker(a); return
    assert not subprocess.run(['fuser', '/dev/kfd'], capture_output=True).stdout.strip(), 'GPU occupied'
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    assert available>16*1024*1024
    a.output.mkdir()
    for name in ('mimo26_rans_gpu_gate.cu', 'mimo26_rans_screen.cpp', Path(__file__).name):
        shutil.copy2(Path(__file__).with_name(name), a.output/name)
    shutil.copy2(a.library, a.output/'gate.so')
    command=[sys.executable, str(Path(__file__).resolve()), '--worker', '--root', str(a.root),
             '--screen', str(a.screen), '--library', str(a.output.resolve()/'gate.so'), '--output', str(a.output.resolve())]
    monitor=dict(command=command, complete=False, passed=False, samples=[])
    with (a.output/'stdout.log').open('w') as stdout, (a.output/'stderr.log').open('w') as stderr:
        proc=subprocess.Popen(command, stdout=stdout, stderr=stderr); start=time.monotonic()
        try:
            while proc.poll() is None:
                try:
                    status=Path(f'/proc/{proc.pid}/status').read_text()
                except FileNotFoundError:
                    continue
                swap=int(next((x.split()[1] for x in status.splitlines() if x.startswith('VmSwap:')), '0'))
                monitor['samples'].append(dict(seconds=time.monotonic()-start, status=status,
                                               meminfo=Path('/proc/meminfo').read_text()))
                if swap or time.monotonic()-start>120:
                    monitor['guard_reason']='worker swap' if swap else 'timeout'; break
                time.sleep(.25)
        finally:
            if proc.poll() is None:
                proc.terminate(); monitor['terminated']=True
            monitor['exit_code']=proc.wait(timeout=30)
            monitor['complete']=True
            if (a.output/'report.json').exists():
                report=json.loads((a.output/'report.json').read_text())
                monitor['passed']=monitor['exit_code']==0 and report['complete'] and report['passed'] and 'guard_reason' not in monitor
            (a.output/'monitor.json').write_text(json.dumps(monitor, indent=2)+'\n')
    print(json.dumps({k:v for k,v in monitor.items() if k!='samples'}, indent=2))
    assert monitor['passed']


if __name__=='__main__':
    main()

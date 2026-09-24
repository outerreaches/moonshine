"""Bounded isolated GPU diagnostic; records vectors, identity and host/worker memory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def meminfo():
    return {x.split(':')[0]: int(x.split()[1])
            for x in Path('/proc/meminfo').read_text().splitlines()
            if x.startswith(('MemAvailable:', 'SwapFree:'))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL')
    p.add_argument('--mode', choices=('base', 'grouped'), required=True)
    p.add_argument('--slots', type=int, choices=(128, 160), default=160)
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--context', type=int, default=4096)
    p.add_argument('--tokens', type=int, default=224)
    p.add_argument('--decodes', type=int, default=2)
    p.add_argument('--repeats', type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--compare', type=Path)
    p.add_argument('--layer-trace', action='store_true')
    p.add_argument('--lookahead', choices=('on', 'off'), default='off')
    a = p.parse_args()
    assert 1 <= a.chunk <= 256 and a.tokens + a.decodes <= a.context <= 4096
    assert 0 <= a.decodes <= 32 and 0 < a.tokens and 0 < a.timeout <= 3600
    a.output.mkdir()
    out = a.output.resolve()
    shutil.copy2(__file__, out / 'runner.py')
    binary = a.binary.resolve(strict=True)
    cmd = [str(binary), a.model, str(out / 'vectors'), str(a.slots),
           str(a.chunk), str(a.context), str(a.decodes), str(a.tokens)]
    if a.repeats != 1 or a.lookahead == 'on':
        cmd.append(str(a.repeats))
    if a.lookahead == 'on':
        cmd.append('on')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('MIMO26_', 'K3_'))}
    if a.mode == 'grouped':
        env['MIMO26_EXPERT_MAJOR'] = '1'
    env['MIMO26_PROJECTION_DUMP'] = str(out / 'projection')
    if a.layer_trace:
        assert a.repeats == 1, 'layer trace filenames require one request'
        env['MIMO26_LAYER_TRACE'] = str(out / 'layer')
    report = dict(complete=False, passed=False, command=cmd, mode=a.mode,
                  binary_sha256=sha(binary), runner_sha256=sha(out / 'runner.py'),
                  environment={key: env.get(key) for key in
                      ('MIMO26_EXPERT_MAJOR', 'MIMO26_GPU_PROFILE',
                       'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
                       'HSA_OVERRIDE_GFX_VERSION')}, memory=[], errors=[])
    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    proc = None
    started = time.monotonic()
    try:
        assert subprocess.run(['fuser', '/dev/kfd'], capture_output=True).returncode == 1, 'GPU occupied'
        assert meminfo()['MemAvailable'] > (118 if a.slots == 160 else 100) * 1048576, 'startup headroom'
        with (out / 'stdout.log').open('x') as stdout, (out / 'stderr.log').open('x') as stderr:
            proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr,
                preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
            report['pid'] = proc.pid
            save()
            while proc.poll() is None:
                try:
                    status = Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                except FileNotFoundError:
                    continue
                row = {x.split(':')[0]: int(x.split()[1]) for x in status
                       if x.startswith(('VmRSS:', 'VmSwap:'))}
                if not row:
                    continue
                row.update(meminfo())
                row.update({k: int(v) for k, v in
                    (x.split() for x in Path('/proc/vmstat').read_text().splitlines())
                    if k in ('pswpin', 'pswpout')})
                row['seconds'] = time.monotonic() - started
                report['memory'].append(row)
                assert row.get('VmSwap') == 0, 'worker swap'
                assert row['MemAvailable'] > 12 * 1048576, 'runtime headroom'
                assert (row['pswpout'] - report['memory'][0]['pswpout']) * 4096 < 512 * 1048576, 'host page-out budget'
                assert row['seconds'] < a.timeout, 'timeout'
                if len(report['memory']) % 20 == 0:
                    save()
                time.sleep(.5)
            report['exit_code'] = proc.returncode
            assert proc.returncode == 0, f'gate exited {proc.returncode}'
        report['vectors'] = {x.name: sha(x) for x in out.glob('vectors-*.bin')}
        assert len(report['vectors']) == (a.decodes + 1) * a.repeats
        report['phases'] = [json.loads(x) for x in (out / 'stdout.log').read_text().splitlines()
                            if x.startswith('{')]
        prefills = [x for x in report['phases'] if x.get('phase') == 'prefill']
        assert len(prefills) == a.repeats and all(x['tokens'] == a.tokens for x in prefills), 'prompt length mismatch'
        if a.layer_trace:
            report['layers'] = {x.name: sha(x) for x in out.glob('layer-*.bin')}
            assert len(report['layers']) == 48 * 3 * ((a.tokens + a.chunk - 1) // a.chunk)
            if a.compare:
                report['layer_comparison'] = {name: sha(a.compare / name) == digest
                    for name, digest in report['layers'].items()}
                assert all(report['layer_comparison'].values()), 'hidden/KV mismatch'
        if a.compare:
            report['comparison'] = {name: sha(a.compare / name) == digest
                                    for name, digest in report['vectors'].items()}
            assert all(report['comparison'].values()), 'full-vector mismatch'
        report['passed'] = True
    except BaseException as error:
        report['errors'].append(repr(error))
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        report['complete'] = True
        report['seconds'] = time.monotonic() - started
        save()
    print(json.dumps({k: v for k, v in report.items()
                     if k not in ('memory', 'layers', 'layer_comparison')}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

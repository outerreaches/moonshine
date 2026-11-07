"""Frozen-worker load diagnostic; no inference or system policy changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--slots', type=int, choices=(48, 96, 144), default=96)
p.add_argument('--pace', action='store_true', help='sleep 1 ms after each expert allocation')
a = p.parse_args()
repo = Path(__file__).resolve().parents[1]
a.output.mkdir()
report = dict(complete=False, loaded=False, slots=a.slots, pace_1ms=a.pace, commands=[], snapshots=[],
              scope='load-only diagnostic, allocations guarded on worker swap; no inference')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def snapshot(pid=None):
    paths = ['/proc/meminfo', '/proc/vmstat', '/proc/zoneinfo', '/proc/buddyinfo',
             '/proc/pressure/memory']
    if pid:
        paths += [f'/proc/{pid}/status', f'/proc/{pid}/smaps_rollup']
    paths += [str(x) for x in Path('/sys/class/drm/card0/device').glob('mem_info_*')]
    result = dict(seconds=time.monotonic())
    for path in paths:
        try:
            result[path] = Path(path).read_text()
        except (OSError, PermissionError) as error:
            result[path] = str(error)
    return result
def save():
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
try:
    baseline = json.loads((a.baseline / 'report.json').read_text())
    assert baseline['complete'] and baseline['passed']
    report['baseline_report_sha256'] = sha(a.baseline / 'report.json')
    names = ('mimo26_memory_probe.cu', 'mimo26_alloc_trace_wrap.cu', 'run_mimo26_memory_probe.py')
    for name in names:
        shutil.copy2(repo / 'tests' / name, a.output / name)
    report['sources'] = {name: sha(a.output / name) for name in names}
    objects = []
    for name, expected in baseline['inputs'].items():
        if name.endswith('.o'):
            assert sha(a.baseline / name) == expected
            objects.append(str(a.baseline / name))
    flags = ['/opt/rocm/bin/hipcc', '-O3', '-g', '-fno-fast-math', '--offload-arch=gfx1151', '-I'+str(repo)]
    compile_commands = [flags + ['-c', str(a.output / name), '-o', str(a.output / (name + '.o'))]
                        for name in names[:2]]
    cmd = flags + [str(a.output / (name + '.o')) for name in names[:2]] + objects + [
        '-Wl,--wrap=hipMalloc', '-lm', '-pthread', '-lhipblas', '-lhipblaslt', '-lzstd',
        '-licui18n', '-licuuc', '-licudata', '-o', str(a.output / 'probe')]
    with (a.output / 'build.log').open('w') as log:
        for build_command in compile_commands + [cmd]:
            report['commands'].append(build_command)
            subprocess.run(build_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert not subprocess.run(['fuser', '/dev/kfd'], capture_output=True).stdout.strip(), 'GPU occupied'
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:')))
    assert available > 110 * 1024 * 1024, 'insufficient available RAM'
    report['snapshots'].append(snapshot())
    report['binary_sha256'] = sha(a.output / 'probe')
    cmd = [str(a.output / 'probe'), '/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL', str(a.slots)]
    report['commands'].append(cmd)
    save()
    with (a.output / 'stdout.log').open('w') as stdout, (a.output / 'stderr.log').open('w') as stderr:
        env = os.environ.copy()
        env.pop('MIMO26_PROBE_PACE_1MS', None)
        if a.pace:
            env['MIMO26_PROBE_PACE_1MS'] = '1'
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
        start = time.monotonic()
        try:
            while proc.poll() is None:
                if time.monotonic() - start > 150:
                    raise TimeoutError('load probe timeout')
                report['snapshots'].append(snapshot(proc.pid))
                time.sleep(0.25)
            report['exit_code'] = proc.returncode
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
                report['guard_terminated'] = True
                report['exit_code'] = proc.returncode
    report['snapshots'].append(snapshot())
    report['swap_guard'] = 'SWAP_GUARD' in (a.output / 'stderr.log').read_text()
    report['loaded'] = report['exit_code'] == 0
    report['complete'] = True
    print(json.dumps({k: v for k, v in report.items() if k not in ('snapshots', 'commands')}, indent=2))
except Exception as error:
    report['error'] = repr(error)
    raise
finally:
    save()

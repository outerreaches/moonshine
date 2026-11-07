"""One frozen-binary diagnostic with host/GTT/cgroup telemetry and a swap guard.

This is not a benchmark, a retry mechanism, or a VM-policy tuner. It preserves
failed runs and stops at the first sampled worker swap. No compilation or
background service is launched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    try:
        return Path(path).read_text()
    except OSError as error:
        return dict(error=str(error))


def snapshot(pid=None, full=False):
    paths = ['/proc/meminfo', '/proc/vmstat', '/proc/pressure/memory', '/proc/zoneinfo', '/proc/buddyinfo']
    if full:
        paths += ['/proc/cmdline', '/proc/sys/vm/swappiness', '/proc/sys/vm/min_free_kbytes',
                  '/proc/sys/vm/watermark_scale_factor', '/proc/sys/vm/watermark_boost_factor',
                  '/proc/sys/vm/zone_reclaim_mode', '/sys/module/ttm/parameters/pages_limit',
                  '/sys/module/ttm/parameters/page_pool_size']
    paths += sorted(str(p) for p in Path('/sys/class/drm').glob('card*/device/mem_info_*'))
    if pid is not None:
        paths += [f'/proc/{pid}/status', f'/proc/{pid}/smaps_rollup', f'/proc/{pid}/io', f'/proc/{pid}/cgroup']
        try:
            line = next(x for x in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if x.startswith('0::'))
            directory = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
            while True:
                paths += [str(directory / name) for name in ('memory.current', 'memory.stat', 'memory.events',
                           'memory.events.local', 'memory.high', 'memory.max', 'memory.low', 'memory.min',
                           'memory.swap.current', 'memory.swap.max', 'memory.pressure')]
                if directory == Path('/sys/fs/cgroup'):
                    break
                assert Path('/sys/fs/cgroup') in directory.parents
                directory = directory.parent
        except (OSError, StopIteration):
            pass
    return dict(monotonic_seconds=time.monotonic(), files={path: read(path) for path in paths})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build = json.loads((a.build / 'report.json').read_text())
    ref = json.loads((a.reference / 'report.json').read_text())
    assert ref['complete'] and ref['passed']
    assert build['candidate'] and not build['lookahead_enabled']
    assert build['workload'] == ref['workload']
    w = build['workload']
    assert w == dict(tokens=129, chunk=64, cases=[0, 1, 2], slots=48, decode_steps=2, context=256)
    assert sha(a.build / 'probe') == build['binary_sha256']
    assert not subprocess.run(['fuser', '/dev/kfd'], capture_output=True).stdout.strip(), 'GPU occupied'
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    assert available > 110 * 1024 * 1024
    out = a.output.resolve(); out.mkdir()
    shutil.copy2(a.build / 'probe', out / 'probe')
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    r = dict(complete=False, passed=False, source_build=str(a.build),
             source_report_sha256=sha(a.build / 'report.json'), reference_report_sha256=sha(a.reference / 'report.json'),
             binary_sha256=sha(out / 'probe'), workload=w, snapshots=[snapshot(full=True)])
    def save():
        (out / 'report.json').write_text(json.dumps(r, indent=2)+'\n')
    cmd = [str(out / 'probe'), '/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL',
           str(out / 'output'), '129', '64', '0', '3']
    r['command'] = cmd; save()
    try:
        with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr); start = time.monotonic()
            r['pid'] = proc.pid
            try:
                while proc.poll() is None:
                    snap = snapshot(proc.pid); r['snapshots'].append(snap)
                    status = snap['files'][f'/proc/{proc.pid}/status']
                    if isinstance(status, str):
                        swap = next((int(x.split()[1]) for x in status.splitlines() if x.startswith('VmSwap:')), 0)
                        if swap:
                            r['guard_reason'] = 'worker swap'; r['worker_swap_KiB'] = swap
                            r['snapshots'].append(snapshot(proc.pid, full=True)); break
                    if time.monotonic()-start > 420:
                        r['guard_reason'] = 'timeout'; break
                    if len(r['snapshots']) % 20 == 0:
                        save()
                    time.sleep(0.25)
            finally:
                if proc.poll() is None:
                    proc.terminate(); r['terminated'] = True
                r['exit_code'] = proc.wait(timeout=40)
                r['snapshots'].append(snapshot(full=True)); save()
        r['complete'] = True
        if 'guard_reason' not in r:
            assert r['exit_code'] == 0
            r['outputs'] = {p.name: sha(p) for p in out.glob('output-*.bin')}
            assert r['outputs'] == ref['outputs']
            assert sha(out / 'stderr.log') == ref['route_sha256']
            r['passed'] = True
        print(json.dumps({k: v for k, v in r.items() if k not in ('snapshots', 'outputs')}, indent=2))
    except Exception as error:
        r['error'] = repr(error); raise
    finally:
        save()


if __name__ == '__main__':
    main()

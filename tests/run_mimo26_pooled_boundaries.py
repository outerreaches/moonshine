"""Fresh-worker full-vector chunk/decode comparison; continuous no-swap guard."""
import argparse
import json
import os
from pathlib import Path
import shutil
import resource
import signal
import subprocess
import time
from build_mimo26_candidate import sha
from run_mimo26_server_lookahead_gate import MODEL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--slots', type=int, choices=(16, 128), default=128)
    a = p.parse_args(); a.output.mkdir()
    out, build = a.output.resolve(), a.build.resolve()
    parent = json.loads((build / 'report.json').read_text())
    assert parent['built'] and parent['build_kind'].startswith('fresh isolated')
    for name, digest in parent['built_objects'].items(): assert sha(build / name) == digest
    # Keep the pinned sources/objects with the run, never depend on mutable repo objects.
    shutil.copytree(build / 'source', out / 'source')
    for name in ('mimo26_allocation_guard.cu', 'mimo26_allocation_guard.o'):
        shutil.copy2(build / name, out / name)
    repo = Path(__file__).resolve().parents[1]
    for name in ('mimo26_pooled_boundary_gate.cu', 'mimo26_allocation_guard_gate.cu', 'run_mimo26_pooled_boundaries.py'):
        shutil.copy2(repo / 'tests' / name, out / name)
    r = dict(complete=False, passed=False, slots=a.slots, runs=[], commands=[],
             parent_report_sha256=sha(build / 'report.json'))
    def save(): (out / 'report.json').write_text(json.dumps(r, indent=2)+'\n')
    try:
        objects = sorted(x for x in (out / 'source').glob('*.o')
                         if x.name not in ('mimo26_server.o', 'mimo26_server_slot.o'))
        flags = ['/opt/rocm/bin/hipcc', '-O3', '-g', '-fno-fast-math', '--offload-arch=gfx1151',
                 '-I'+str(out / 'source')]
        compile_cmd = flags + ['-c', str(out / 'mimo26_pooled_boundary_gate.cu'), '-o', str(out / 'driver.o')]
        cmd = flags + [str(out / 'driver.o')] + [str(x) for x in objects] + [
               str(out / 'mimo26_allocation_guard.o'), '-Wl,--wrap=hipMalloc', '-Wl,--wrap=hipFree',
               '-Wl,--wrap=hipMemcpy', '-lm', '-pthread', '-lhipblas', '-lhipblaslt', '-lzstd',
               '-licui18n', '-licuuc', '-licudata', '-o', str(out / 'probe')]
        with (out / 'build.log').open('w') as log:
            r['commands'].append(compile_cmd)
            subprocess.run(compile_cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            r['commands'].append(cmd)
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            guard_compile = flags + ['-c', str(out / 'mimo26_allocation_guard_gate.cu'), '-o', str(out / 'guard-driver.o')]
            r['commands'].append(guard_compile)
            subprocess.run(guard_compile, stdout=log, stderr=subprocess.STDOUT, check=True)
            guard_cmd = flags + [str(out / 'guard-driver.o'), str(out / 'mimo26_allocation_guard.o'),
                         '-Wl,--wrap=hipMalloc', '-Wl,--wrap=hipFree', '-Wl,--wrap=hipMemcpy',
                         '-o', str(out / 'guard-probe')]
            r['commands'].append(guard_cmd)
            subprocess.run(guard_cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        r['binary_sha256'] = sha(out / 'probe')
        r['inputs'] = {str(x.relative_to(out)): sha(x) for x in out.rglob('*') if x.is_file()}
        assert subprocess.run(['fuser', '/dev/kfd'], capture_output=True).returncode == 1, 'GPU occupied'
        r['guard_self_tests'] = []
        for mode, expected, diagnostic in (('healthy', 0, 'TEST_ALLOCATION_GUARDS'),
                ('copy-overflow', -signal.SIGABRT, 'explicit HIP copy exceeds allocation'),
                ('red-zone', -signal.SIGABRT, 'device allocation red zone corrupted')):
            result = subprocess.run([str(out / 'guard-probe'), mode], capture_output=True, text=True,
                                    timeout=30, preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
            r['guard_self_tests'].append(dict(mode=mode, exit_code=result.returncode,
                                              stdout=result.stdout, stderr=result.stderr))
            assert result.returncode == expected and diagnostic in result.stderr
        save()
        for chunk in (0, 32, 64, 128):
            assert subprocess.run(['fuser', '/dev/kfd'], capture_output=True).returncode == 1, 'GPU occupied'
            def available():
                return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                                if x.startswith('MemAvailable:')))
            assert available() > (100 if a.slots == 128 else 70) * 1048576
            path = out / f'chunk{chunk}'; path.mkdir()
            row = dict(chunk=chunk, memory=[], errors=[]); r['runs'].append(row)
            cmd = [str(out / 'probe'), MODEL, str(path), str(a.slots), str(chunk)]
            r['commands'].append(cmd); save()
            env = {k: v for k, v in os.environ.items() if not k.startswith('MIMO26_')}
            with (path / 'stdout.log').open('w') as stdout, (path / 'stderr.log').open('w') as stderr:
                proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
                start = time.monotonic()
                try:
                    while proc.poll() is None:
                        try:
                            status = Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                            mem = {x.split(':')[0]: int(x.split()[1]) for x in status if x.startswith(('VmRSS:', 'VmSwap:'))}
                        except FileNotFoundError: continue
                        if not mem: continue  # Process exited between poll and read.
                        mem.update({key: int(value) for key, value in
                                    (line.split() for line in Path('/proc/vmstat').read_text().splitlines())
                                    if key in ('pswpin', 'pswpout')})
                        row['memory'].append(dict(seconds=time.monotonic()-start, MemAvailable=available(), **mem))
                        assert mem.get('VmSwap') == 0, 'worker swap'
                        assert row['memory'][-1]['MemAvailable'] > 8 * 1048576, 'host memory floor'
                        assert time.monotonic()-start < 600, 'timeout'
                        if len(row['memory']) % 20 == 0: save()
                        time.sleep(.5)
                    row['exit_code'] = proc.returncode
                    assert proc.returncode == 0
                except BaseException as error:
                    row['errors'].append(repr(error)); raise
                finally:
                    if proc.poll() is None: proc.terminate(); proc.wait(timeout=60)
                    row['seconds'] = time.monotonic()-start; save()
            row['outputs'] = {x.name: sha(x) for x in path.glob('*.bin')}
            assert len(row['outputs']) == 6
            if chunk: assert row['outputs'] == r['runs'][0]['outputs'], 'decode/chunk full-vector difference'
            row['passed'] = True; save()
            print(f'PASS chunk={chunk} slots={a.slots} full-vector boundaries 128/129/257 plus decode', flush=True)
        r.update(complete=True, passed=True)
    except BaseException as error:
        r['error'] = repr(error); raise
    finally: save()


if __name__ == '__main__': main()

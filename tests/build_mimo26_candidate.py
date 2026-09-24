"""Fresh isolated candidate: source snapshot, no reused objects or repo binaries."""
from pathlib import Path
import hashlib
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fresh(repo, out, report):
    source = out / 'source'
    source.mkdir()
    (source / 'tools').mkdir()
    for path in sorted(repo.iterdir()):
        if path.is_file() and (path.suffix in ('.c', '.cu', '.h', '.inc') or path.name == 'Makefile'):
            shutil.copy2(path, source / path.name)
            report['inputs'][path.name] = sha(source / path.name)
    for name in ('mimo26_server_lookahead_wrap.cu', 'mimo26_allocation_guard.cu',
                 'build_mimo26_candidate.py', 'run_mimo26_server_lookahead_gate.py'):
        shutil.copy2(repo / 'tests' / name, out / name)
        report['inputs']['tests/' + name] = sha(out / name)
    report['source_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    report['source_diff'] = subprocess.check_output(['git', 'diff', '--binary'], cwd=repo, text=True)
    report['build_kind'] = 'fresh isolated Makefile build; guarded observed binary; unmodified clean link'
    flags = ['/opt/rocm/bin/hipcc', '-O3', '-g', '-fno-fast-math', '--offload-arch=gfx1151', '-I'+str(source)]
    libs = ['-lm', '-pthread', '-lhipblas', '-lhipblaslt', '-lzstd', '-licui18n', '-licuuc', '-licudata']
    with (out / 'build.log').open('w') as log:
        def run(command):
            report['commands'].append(command)
            subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
        run(['make', '-j4', 'tools/mimo26_server'])
        shutil.copy2(source / 'tools/mimo26_server', out / 'server-clean')
        objects = sorted(source.glob('*.o'))
        for name in ('mimo26_server_lookahead_wrap', 'mimo26_allocation_guard'):
            run(flags + ['-c', str(out / (name + '.cu')), '-o', str(out / (name + '.o'))])
        run(flags + [str(p) for p in objects] +
            [str(out / (name + '.o')) for name in ('mimo26_server_lookahead_wrap', 'mimo26_allocation_guard')] +
            ['-Wl,--wrap='+name for name in ('mimo26_gpu_worker_prefill', 'mimo26_gpu_worker_decode',
             'mimo26_gpu_worker_argmax', 'k3_io_uring_submit', 'mimo26_slot_admit', 'hipMalloc', 'hipFree', 'hipMemcpy')] +
            libs + ['-o', str(out / 'server-observed')])
    report['binaries'] = {name: sha(out / name) for name in ('server-clean', 'server-observed')}
    report['built_objects'] = {str(p.relative_to(out)): sha(p) for p in list(source.glob('*.o')) + list(out.glob('*.o'))}
    report['compiler'] = subprocess.check_output(['/opt/rocm/bin/hipcc', '--version'], text=True)

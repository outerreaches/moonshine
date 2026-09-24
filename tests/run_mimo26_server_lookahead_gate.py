"""Opt-in server: exact mixed outputs, real disconnect, busy refusal and quarantine.

Builds isolated clean/instrumented binaries using a qualified matched worker,
layer and cache object set. Never installs a service or replaces repo binaries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from build_mimo26_candidate import build_fresh

REPO = Path(__file__).resolve().parents[1]
MODEL = '/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL'
DEFAULT_BUILD = Path('/home/alex/Obsidian/beelink-knowledge/Projects/Moonshine/Evidence/mimo26-lookahead-long-20260923/chunk64-on')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(text, stream=False):
    return dict(messages=[dict(role='user', content=text)], enable_thinking=False,
                max_tokens=6, stream=stream)


A = payload('A shelf holds 17 boxes, each with six packets. How many packets are there? Give the number first.')
B = payload('Explain briefly why Python uses range(n) rather than range(n+1) to visit n list elements.')
C = payload('Translate to English: La bibliothèque ouvrira à neuf heures samedi, sauf en cas de panne.')
CANCEL = payload(('Review the inventory carefully. Shelf A has seventeen boxes, shelf B has twenty-four, '
                  'and shelf C has thirty-nine. Each box holds six packets. Do not confuse packets and boxes. ') * 10)


def build(out, report, source_build):
    old = json.loads((source_build / 'report.json').read_text())
    assert old['complete'] and old['passed'] and old['candidate']
    report['worker_build_report_sha256'] = sha(source_build / 'report.json')
    for name in ('mimo26_gpu_worker.cu', 'mimo26_gpu_worker.h', 'mimo26_rocm_layer.cu',
                 'mimo26_rocm_layer.h', 'k3_expert_cache.c', 'k3_expert_cache.h'):
        assert sha(REPO / name) == old['inputs'][name], name
        shutil.copy2(REPO / name, out / name)
        report['inputs'][name] = sha(REPO / name)
    objects = []
    for name in old['inputs']:
        if name.endswith('.o'):
            assert sha(source_build / name) == old['built_objects'][name], name
            shutil.copy2(source_build / name, out / name)
            objects.append(str(out / name))
            report['inputs'][name] = old['built_objects'][name]
    for name in ('mimo26_server.cu', 'mimo26_server_options.h', 'mimo26_server_slot.c',
                 'mimo26_server_slot.h', 'tests/mimo26_server_lookahead_wrap.cu',
                 'tests/run_mimo26_server_lookahead_gate.py'):
        shutil.copy2(REPO / name, out / Path(name).name)
        report['inputs'][name] = sha(REPO / name)
    flags = ['/opt/rocm/bin/hipcc', '-O3', '-g', '-fno-fast-math', '--offload-arch=gfx1151',
             '-I'+str(out), '-I'+str(REPO)]
    commands = [flags + ['-c', str(out / 'mimo26_server.cu'), '-o', str(out / 'server.o')],
                ['cc', '-O2', '-g', '-I'+str(out), '-c', str(out / 'mimo26_server_slot.c'), '-o', str(out / 'slot.o')],
                flags + ['-c', str(out / 'mimo26_server_lookahead_wrap.cu'), '-o', str(out / 'wrap.o')]]
    libs = ['-lm', '-pthread', '-lhipblas', '-lhipblaslt', '-lzstd', '-licui18n', '-licuuc', '-licudata']
    link = flags + [str(out / 'server.o'), str(out / 'slot.o')] + objects
    commands += [link + libs + ['-o', str(out / 'server-clean')],
                 link + [str(out / 'wrap.o')] + ['-Wl,--wrap='+name for name in (
                     'mimo26_gpu_worker_prefill', 'mimo26_gpu_worker_decode', 'mimo26_gpu_worker_argmax',
                     'k3_io_uring_submit', 'mimo26_slot_admit')] + libs + ['-o', str(out / 'server-observed')]]
    with (out / 'build.log').open('w') as log:
        for command in commands:
            report['commands'].append(command)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report['binaries'] = {name: sha(out / name) for name in ('server-clean', 'server-observed')}
    report['built_objects'] = {p.name: sha(p) for p in out.glob('*.o')}
    report['cli'] = []
    for args, expected in [(['--help'], 0), ([MODEL, '--expert-lookahead', 'true'], 2),
                           ([MODEL, '--slots', '-1'], 2), ([MODEL, '--expert-lookahead'], 2),
                           ([MODEL, '--prefill-chunk', '0', '--expert-lookahead', 'on'], 2)]:
        result = subprocess.run([str(out / 'server-clean')] + args, capture_output=True, text=True, timeout=10)
        report['cli'].append(dict(args=args, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == expected and 'loading the worker' not in result.stdout


def run_variant(out, report, variant, save):
    enabled = variant != 'off'
    clean = variant == 'clean'
    fault = variant == 'fault'
    assert not subprocess.run(['fuser', '/dev/kfd'], capture_output=True).stdout.strip(), 'GPU occupied'
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    assert available > (100 if report['slots'] == 128 else 70) * 1024 * 1024
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    path = out / variant; path.mkdir()
    row = dict(variant=variant, port=port, memory=[], responses=[], monitor_errors=[])
    report['runs'].append(row)
    cmd = [str(out / ('server-clean' if clean else 'server-observed')), MODEL,
           '--host', '127.0.0.1', '--port', str(port), '--slots', str(report['slots']), '--context', str(report['context']),
           '--prefill-chunk', str(report['chunk']), '--expert-lookahead', 'on' if enabled else 'off']
    if report.get('retain_experts') is not None:
        cmd += ['--retain-experts', report['retain_experts']]
    report['commands'].append(cmd)
    env = {k: v for k, v in os.environ.items() if not k.startswith('MIMO26_')}
    env['MIMO26_TEST_CAPTURE'] = str(path)
    if fault:
        env['MIMO26_TEST_HTTP_SUBMIT_FAILURE'] = '1'
    with (path / 'stdout.log').open('w') as stdout, (path / 'stderr.log').open('w') as stderr:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
        start = time.monotonic(); stop = threading.Event()
        def monitor():
            while not stop.is_set() and proc.poll() is None:
                try:
                    lines = Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                    mem = {x.split(':')[0]: int(x.split()[1]) for x in lines if x.startswith(('VmRSS:', 'VmSwap:'))}
                    if not mem:
                        continue
                    mem['MemAvailable'] = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                                                  if x.startswith('MemAvailable:')))
                    row['memory'].append(dict(seconds=time.monotonic()-start, **mem))
                    if mem.get('VmSwap', 0) or mem['MemAvailable'] < 8 * 1048576 or time.monotonic()-start > 600:
                        row['monitor_errors'].append('worker swap, host memory floor or timeout'); proc.terminate(); return
                except FileNotFoundError:
                    pass
                stop.wait(1)
        thread = threading.Thread(target=monitor); thread.start()
        def request(endpoint, body=None):
            def read_bytes():
                return int(next(x.split()[1] for x in Path(f'/proc/{proc.pid}/io').read_text().splitlines()
                                if x.startswith('read_bytes:')))
            began = time.monotonic(); before_reads = read_bytes()
            req = urllib.request.Request(f'http://127.0.0.1:{port}'+endpoint,
                data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            try:
                response = urllib.request.urlopen(req, timeout=120)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                raw = response.read().decode()
                if body and body.get('stream') and response.code == 200:
                    data = [line[6:] for line in raw.splitlines() if line.startswith('data: ')]
                    assert data[-1] == '[DONE]'
                    parsed = [json.loads(x) for x in data[:-1]]
                else:
                    parsed = json.loads(raw)
                result = dict(path=endpoint, status=response.code, body=parsed,
                              seconds=time.monotonic()-began, process_read_bytes=read_bytes()-before_reads)
            row['responses'].append(result); return result
        def chat(body):
            result = request('/v1/chat/completions', body)
            assert result['status'] == 200
            return result['body']
        def wait_ready():
            deadline = time.monotonic()+120
            while True:
                assert proc.poll() is None and time.monotonic() < deadline
                try:
                    health = request('/health')
                    if health['status'] == 200 and health['body']['ready']:
                        return health['body']
                except (urllib.error.URLError, ConnectionError):
                    pass
                time.sleep(0.25)
        try:
            health = wait_ready()
            assert health['expert_lookahead'] == enabled and health['prefill_chunk'] == report['chunk']
            assert health['context'] == report['context']
            if report.get('build_kind'):
                assert health['expert_slots'] == report['slots']
            if report.get('retain_experts') is not None:
                assert health['retain_experts'] == (report['retain_experts'] == 'on')
            print('READY '+variant, flush=True)
            if fault:
                first = request('/v1/chat/completions', A)
                assert first['status'] == 500 and first['body']['error']['code'] == 'decode_failed'
                for _ in range(5):
                    health = request('/health')['body']
                    assert not health['ready'] and health['phase'] == 'quarantined'
                    assert health['faults'] == 1 and health['recoveries'] == 0
                    assert request('/v1/chat/completions', A)['status'] == 503
                assert request('/health')['body']['rejected_quarantined'] == 5
            elif variant in ('clean', 'recreated'):
                if clean and report.get('transport_probes'):
                    row['transport'] = []
                    for wire, timeout_probe in [
                        (b'POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2junk\r\n\r\n{}', False),
                        (b'POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 9\r\n\r\n{}', False),
                        (b'POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 2\r\n\r\n{}', False),
                        (b'GET /health HTTP/1.1\r\nX: unfinished', True),
                        (b'POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 20\r\n\r\n{}', True),
                    ]:
                        started = time.monotonic()
                        with socket.create_connection(('127.0.0.1', port), timeout=12) as peer:
                            peer.sendall(wire); chunks=[]
                            while True:
                                data=peer.recv(4096)
                                if not data: break
                                chunks.append(data)
                        elapsed=time.monotonic()-started; response=b''.join(chunks).decode()
                        assert response.startswith('HTTP/1.1 400 ') and 'malformed_request' in response
                        assert elapsed < 8 and (not timeout_probe or elapsed >= 4.5)
                        row['transport'].append(dict(wire=wire.decode(),response=response,seconds=elapsed))
                        assert wait_ready()['faults']==0
                row['answer_a'] = chat(A)
                assert wait_ready()['faults'] == 0
            else:
                row['answer_a'] = chat(A)
                row['answer_b'] = chat(B)
                streamed = chat(dict(A, stream=True))
                text = ''.join(x['choices'][0]['delta'].get('content', '') for x in streamed if x.get('choices'))
                assert text == row['answer_a']['choices'][0]['message']['content']
                # Wait for actual worker admission before issuing the busy probe.
                wire = json.dumps(CANCEL).encode()
                with socket.create_connection(('127.0.0.1', port), timeout=30) as peer:
                    peer.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(wire)}\r\n\r\n').encode()+wire)
                    deadline = time.monotonic()+120
                    while 'TEST_PREFILL_BEGIN request=4 ' not in (path / 'stderr.log').read_text():
                        assert proc.poll() is None and time.monotonic() < deadline
                        time.sleep(0.05)
                    busy = request('/v1/chat/completions', B)
                    assert busy['status'] == 503 and busy['body']['error']['code'] == 'slot_busy'
                health = wait_ready()
                assert health['cancelled'] == 1 and health['faults'] == 0
                match = re.search(r'TEST_PREFILL_END request=4 status=0 position=(\d+) count=(\d+) stopped=1', (path / 'stderr.log').read_text())
                assert match and 0 < int(match[1]) < int(match[2])
                assert int(match[1]) <= 2 * report['chunk'], 'disconnect did not stop by next committed chunk'
                row['cancel_position'] = int(match[1]); row['cancel_prompt_tokens'] = int(match[2])
                assert not (path / '4-prefill.bin').exists()
                after = chat(A)
                assert after['choices'] == row['answer_a']['choices'] and after['usage'] == row['answer_a']['usage']
                row['answer_c'] = chat(C)
                health = wait_ready()
                assert health['faults'] == 0 and health['cancelled'] == 1 and health['admitted'] == 6
                row['final_health'] = health
            row['passed'] = True
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                row['exit_code'] = proc.wait(timeout=40)
            finally:
                stop.set(); thread.join(timeout=5); save()
        assert row['exit_code'] == 0 and not row['monitor_errors']
    row['outputs'] = {p.name: sha(p) for p in path.glob('*.bin')}
    if fault:
        log = (path / 'stderr.log').read_text()
        assert log.count('TEST_HTTP_FAULT') == 1 and log.count('recovery refused; worker recreation required') == 3
    print('PASS '+variant, flush=True); save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker-build', type=Path, default=DEFAULT_BUILD)
    p.add_argument('--build-only', action='store_true')
    p.add_argument('--run-only', action='store_true')
    p.add_argument('--slots', type=int, choices=(16, 48, 128), default=16)
    p.add_argument('--chunk', type=int, choices=(32, 64, 128), default=64)
    p.add_argument('--context', type=int, default=1024)
    p.add_argument('--fresh-build', action='store_true')
    p.add_argument('--baseline', type=Path, help='completed same-workload report for full-vector regression')
    p.add_argument('--retain-experts', choices=('off', 'on'), help='explicit opt-in on compatible new binaries')
    p.add_argument('--transport-probes', action='store_true')
    a = p.parse_args(); assert not (a.build_only and a.run_only)
    out = a.output.resolve()
    if a.run_only:
        report = json.loads((out / 'report.json').read_text())
        assert report['built'] and not report['runs']
        for name, digest in report['binaries'].items():
            assert sha(out / name) == digest
    else:
        out.mkdir(); report = dict(complete=False, passed=False, built=False, commands=[], inputs={}, runs=[])
    report['slots'] = a.slots
    report['chunk'] = a.chunk
    report['context'] = a.context
    report['retain_experts'] = a.retain_experts
    assert a.context >= 512
    report['transport_probes'] = a.transport_probes
    report['execution_driver_sha256'] = sha(Path(__file__))
    report['model_metadata'] = {name: sha(Path(MODEL) / name) for name in
                                ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json')}
    shutil.copy2(Path(__file__), out / 'execution_driver.py')
    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    try:
        if not a.run_only:
            if a.fresh_build:
                build_fresh(REPO, out, report)
            else:
                build(out, report, a.worker_build)
            report['built'] = True; save()
        if a.build_only:
            return
        for variant in ('off', 'on', 'fault', 'recreated', 'clean'):
            run_variant(out, report, variant, save)
        off, on, fault, recreated, clean = report['runs']
        assert off['outputs'] == on['outputs'], 'full vectors/token IDs differ with lookahead'
        assert off['answer_a']['choices'] == on['answer_a']['choices']
        assert off['answer_b']['choices'] == on['answer_b']['choices']
        assert off['answer_c']['choices'] == on['answer_c']['choices']
        for row in (recreated, clean):
            assert row['answer_a']['choices'] == on['answer_a']['choices']
            assert row['answer_a']['usage'] == on['answer_a']['usage']
        assert recreated['outputs'] == {name: digest for name, digest in on['outputs'].items() if name.startswith('1-')}
        if a.baseline:
            baseline = json.loads(a.baseline.read_text())
            assert baseline['complete'] and baseline['passed']
            for row in (off, on, recreated):
                reference = next(r for r in baseline['runs'] if r['variant'] == row['variant'])
                assert row['outputs'] == reference['outputs'], 'full vectors/token IDs differ from baseline'
            report['baseline_report_sha256'] = sha(a.baseline)
        report.update(complete=True, passed=True,
            scope=f'chunk{a.chunk}/cache{a.slots}/context{a.context}/retention={a.retain_experts}; mixed requests, real prefill disconnect, busy refusal, injected I/O quarantine, explicit fresh-process recovery and clean-binary smoke; no service deployment, automatic recovery or broad quality claim')
        print('PASS complete opt-in server gate', flush=True)
    except Exception as error:
        report['error'] = repr(error); raise
    finally:
        save()


if __name__ == '__main__':
    main()

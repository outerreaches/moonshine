"""Real GPU safe-boundary deadlines/shutdown, with connected peers and exact reuse.

Deadline shortening is exclusively a first-admission linker wrapper. The clean
server keeps its 600-second default; no production test environment hooks.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from run_mimo26_server_lookahead_gate import A, B, CANCEL, MODEL, sha
from run_mimo26_transient_recovery import BASE


def run(out, build, variant, profile, reference):
    assert subprocess.run(['fuser', '/dev/kfd'], capture_output=True).returncode == 1
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:')))
    assert available > (100 if profile['slots'] == 128 else 70) * 1048576
    path = out / variant; path.mkdir()
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    supervised = variant == 'supervised-prefill'
    clean = variant == 'clean-prefill'
    deadline_case = variant.startswith('deadline-')
    cmd = [str(out / ('server-clean' if clean else 'server-observed')), MODEL, '--host', '127.0.0.1', '--port', str(port),
           '--slots', str(profile['slots']), '--context', str(profile['context']),
           '--prefill-chunk', str(profile['chunk']), '--expert-lookahead', 'on']
    if profile.get('retain_experts') is not None:
        cmd += ['--retain-experts', profile['retain_experts']]
    if supervised:
        cmd = [sys.executable, str(out / 'mimo26_supervise.py'), str(out / 'server-observed'), MODEL,
               '--port', str(port), '--slots', str(profile['slots']), '--context', str(profile['context']),
               '--prefill-chunk', str(profile['chunk']),
               '--expert-lookahead', 'on', '--shutdown-timeout', '120', '--restarts', '1']
        if profile.get('retain_experts') is not None:
            cmd += ['--retain-experts', profile['retain_experts']]
    env = {k: v for k, v in os.environ.items() if not k.startswith('MIMO26_')}
    env.update(MIMO26_TEST_CAPTURE=str(path), MIMO26_TEST_CONTROL_TRACE='1')
    if deadline_case: env['MIMO26_TEST_FIRST_DEADLINE_SECONDS'] = '1'
    r = dict(complete=False, passed=False, variant=variant, command=cmd, memory=[], responses=[],
             guard_errors=[], shutdown_requested=False)
    start = time.monotonic(); stop = threading.Event()
    def save(): (path / 'report.json').write_text(json.dumps(r, indent=2) + '\n')
    def log(): return (path / 'stderr.log').read_text()
    def events():
        rows = []
        for line in (path / 'stdout.log').read_text().splitlines():
            try: row = json.loads(line)
            except ValueError: continue
            if isinstance(row, dict) and 'event' in row: rows.append(row)
        return rows
    with (path / 'stdout.log').open('w') as stdout, (path / 'stderr.log').open('w') as stderr:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
        if not supervised: r['worker_pid'] = proc.pid
        def monitor():
            while not stop.wait(.5):
                if 'worker_pid' not in r: continue
                try:
                    status = Path(f"/proc/{r['worker_pid']}/status").read_text()
                    fields = {x.split(':')[0]: int(x.split()[1]) for x in status.splitlines()
                              if x.startswith(('VmRSS:', 'VmSwap:'))}
                except FileNotFoundError:
                    if r['shutdown_requested'] or proc.poll() is not None: return
                    r['guard_errors'].append('worker disappeared'); proc.terminate(); return
                if fields:
                    r['memory'].append(dict(seconds=time.monotonic()-start, **fields))
                    if fields.get('VmSwap') != 0:
                        r['guard_errors'].append('worker swap'); proc.terminate(); return
                if time.monotonic()-start > 240:
                    r['guard_errors'].append('total timeout'); proc.terminate(); return
        thread = threading.Thread(target=monitor); thread.start()
        def check():
            assert proc.poll() is None and not r['guard_errors'] and time.monotonic()-start < 235
        def wait_for(predicate, seconds=120):
            until = time.monotonic()+seconds
            while not predicate():
                check(); assert time.monotonic() < until; time.sleep(.01)
        def request(body=None):
            endpoint = '/health' if body is None else '/v1/chat/completions'
            req = urllib.request.Request(f'http://127.0.0.1:{port}'+endpoint,
                data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            try: response = urllib.request.urlopen(req, timeout=120)
            except urllib.error.HTTPError as error: response = error
            with response: raw = response.read().decode(); code = response.code
            if body and body.get('stream') and code == 200:
                frames = [x[6:] for x in raw.splitlines() if x.startswith('data: ')]
                assert frames[-1] == '[DONE]'
                parsed = [json.loads(x) for x in frames[:-1]]
            else: parsed = json.loads(raw)
            r['responses'].append(dict(path=endpoint, status=code, body=parsed)); save()
            return code, parsed
        peer = None
        try:
            if supervised:
                wait_for(lambda: any(x['event']=='started' for x in events()))
                r['worker_pid'] = next(x['pid'] for x in events() if x['event']=='started')
                wait_for(lambda: any(x['event']=='ready' for x in events()))
            else:
                wait_for(lambda: 'listening on http://' in (path / 'stdout.log').read_text())
            code, health = request()
            assert code == 200 and health['ready'] and health['faults'] == 0
            assert health['prefill_chunk'] == profile['chunk'] and health['context'] == profile['context']
            assert health['expert_slots'] == profile['slots'] and health['expert_lookahead'] is True
            if profile.get('retain_experts') is not None:
                assert health['retain_experts'] == (profile['retain_experts'] == 'on')
            if deadline_case:
                began = time.monotonic()
                code, answer = request(dict(A if variant=='deadline-json' else CANCEL,
                                            stream=variant.endswith('sse')))
                r['deadline_response_seconds'] = time.monotonic()-began
                assert code == 200 and r['deadline_response_seconds'] < 90
                if isinstance(answer, list):
                    finishes = [x['choices'][0]['finish_reason'] for x in answer if x.get('choices')]
                    assert finishes[-1] == 'deadline'
                    assert not ''.join(x['choices'][0]['delta'].get('content', '') for x in answer if x.get('choices'))
                else:
                    assert answer['choices'][0]['finish_reason'] == 'deadline'
                    assert answer['usage']['completion_tokens'] == 0 and answer['choices'][0]['message']['content'] == ''
                _, health = request()
                assert health['ready'] and health['deadline_stops'] == 1 and health['cancelled'] == health['faults'] == 0
                assert {x.name for x in path.glob('*.bin')} == {'1-tokens.bin'}
                match = re.search(r'TEST_PREFILL_END request=1 status=0 position=(\d+) count=(\d+) stopped=1', log())
                assert match and int(match[1]) == min(profile['chunk'], int(match[2]))
                assert (int(match[1]) == int(match[2])) == (variant == 'deadline-json')
                r['stop_position'] = int(match[1]); r['prompt_tokens'] = int(match[2])
                code, answer = request(A); assert code == 200
                expected_answer = json.loads((reference / 'report.json').read_text())['runs'][1]['answer_a']
                assert answer['choices'] == expected_answer['choices'] and answer['usage'] == expected_answer['usage']
                actual = {x.name[2:]: sha(x) for x in path.glob('2-*.bin')}
                expected = {x.name[2:]: sha(x) for x in (reference / 'recreated').glob('1-*.bin')}
                assert actual == expected
                r['reuse_outputs'] = actual
                _, r['final_health'] = request()
                assert r['final_health']['served'] == 2 and r['final_health']['deadline_stops'] == 1
                assert r['final_health']['cancelled'] == r['final_health']['faults'] == 0
                r['shutdown_requested'] = True; proc.terminate()
                r['exit_code'] = proc.wait(timeout=120)
            else:
                body = B if variant == 'shutdown-decode' else CANCEL
                wire = json.dumps(body).encode()
                peer = socket.create_connection(('127.0.0.1', port), timeout=120)
                peer.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(wire)}\r\n\r\n').encode()+wire)
                marker = 'TEST_DECODE_BEGIN request=1 step=0' if variant == 'shutdown-decode' else 'TEST_PREFILL_BEGIN request=1 '
                if clean:
                    code, busy = request()
                    assert code == 503 and busy['error']['code'] == 'slot_busy'
                    marker = 'real HTTP503 busy response during prefill'
                    time.sleep(.1)
                else:
                    wait_for(lambda: marker in log())
                if variant == 'shutdown-decode':
                    assert not (path / '1-decode0.bin').exists()
                elif not clean:
                    time.sleep(.1)  # Allow entry into actual first-chunk work.
                    assert 'TEST_PROGRESS request=1 ' not in log()
                r['signal_after_marker'] = marker; r['log_at_signal'] = log()
                r['shutdown_requested'] = True; began = time.monotonic(); proc.terminate()
                # Keep the requesting peer open until the process exits: not disconnect cancellation.
                r['exit_code'] = proc.wait(timeout=120)
                r['shutdown_seconds'] = time.monotonic()-began
                peer.close(); peer = None
                assert 'finish=shutdown' in log() and 'finish=cancelled' not in log()
                if clean:
                    assert not list(path.glob('*.bin')) and 'completion=0 ' in log()
                elif variant == 'shutdown-decode':
                    assert {x.name for x in path.glob('*.bin')} == {'1-tokens.bin', '1-prefill.bin', '1-decode0.bin'}
                    for suffix in ('tokens', 'prefill', 'decode0'):
                        assert sha(path/f'1-{suffix}.bin') == sha(reference/'on'/f'2-{suffix}.bin')
                    assert 'completion=1 ' in log()
                else:
                    match = re.search(r'TEST_PREFILL_END request=1 status=0 position=(\d+) count=(\d+) stopped=1', log())
                    assert match and int(match[1]) == profile['chunk'] < int(match[2])
                    r['stop_position'] = int(match[1]); r['prompt_tokens'] = int(match[2])
                    assert {x.name for x in path.glob('*.bin')} == {'1-tokens.bin'}
                    assert 'completion=0 ' in log()
                if supervised:
                    r['events'] = events()
                    started = [x for x in r['events'] if x['event']=='started']
                    exited = [x for x in r['events'] if x['event']=='exited']
                    assert len(started) == len(exited) == 1 and started[0]['replacement'] == 0
                    assert exited[0]['pid'] == r['worker_pid'] and exited[0]['returncode'] == 0
                    assert not any(x['event']=='shutdown_timeout' for x in r['events'])
                    assert not Path(f"/proc/{r['worker_pid']}").exists()
            assert r['exit_code'] == 0 and not r['guard_errors']
            assert len(r['memory']) > 10 and all(x['VmSwap']==0 for x in r['memory'])
            r['outputs'] = {x.name: sha(x) for x in path.glob('*.bin')}
            r.update(complete=True, passed=True, elapsed_seconds=time.monotonic()-start)
            print('PASS', variant, flush=True)
        except BaseException as error:
            r['error'] = repr(error); raise
        finally:
            if peer is not None: peer.close()
            if proc.poll() is None:
                r['shutdown_requested'] = True; proc.terminate(); proc.wait(timeout=120)
            stop.set(); thread.join(timeout=5); save()
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server-build', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=BASE)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    build = a.server_build.resolve(); out = a.output.resolve(); out.mkdir()
    candidate = json.loads((build / 'report.json').read_text())
    assert candidate['complete'] and candidate['passed']
    reference = a.reference.resolve()
    prior = json.loads((reference/'report.json').read_text())
    assert prior['complete'] and prior['passed']
    profile = dict(slots=candidate.get('slots',16), chunk=candidate.get('chunk',64),
                   context=candidate.get('context',1024), retain_experts=candidate.get('retain_experts'))
    assert profile['chunk'] in (64,128) and profile['slots'] in (16,128)
    for name in ('server-observed','server-clean'):
        assert sha(build/name) == candidate['binaries'][name]
        shutil.copy2(build/name,out/name)
    shutil.copy2(Path(__file__).resolve().parents[1]/'tools/mimo26_supervise.py',out/'mimo26_supervise.py')
    shutil.copy2(__file__,out/Path(__file__).name)
    r = dict(complete=False, passed=False, source_build_sha256=sha(build/'report.json'),
             reference_sha256=sha(reference/'report.json'), reference_build=str(reference), profile=profile,
             inputs={x.name:sha(x) for x in out.iterdir()}, runs=[])
    try:
        for variant in ('deadline-json', 'deadline-sse', 'shutdown-prefill', 'shutdown-decode', 'supervised-prefill', 'clean-prefill'):
            r['runs'].append(run(out,build,variant,profile,reference))
        r.update(complete=True,passed=True)
    finally:
        (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__ == '__main__': main()

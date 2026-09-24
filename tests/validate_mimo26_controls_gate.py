"""Offline completeness, exact-output and lifecycle validation of control gates."""
import argparse
import json
from pathlib import Path
from run_mimo26_server_lookahead_gate import sha
from run_mimo26_transient_recovery import BASE


def validate(path, build):
    r = json.loads((path/'report.json').read_text())
    candidate = json.loads((build/'report.json').read_text())
    reference = Path(r.get('reference_build', str(BASE)))
    chunk = r.get('profile', {}).get('chunk', 64)
    assert r['complete'] and r['passed'] and candidate['complete'] and candidate['passed']
    assert r['source_build_sha256'] == sha(build/'report.json')
    assert r['reference_sha256'] == sha(reference/'report.json')
    for name, digest in r['inputs'].items(): assert sha(path/name) == digest
    for name in ('server-clean','server-observed'):
        assert sha(path/name) == candidate['binaries'][name]
    variants = ['deadline-json','deadline-sse','shutdown-prefill','shutdown-decode','supervised-prefill','clean-prefill']
    assert [x['variant'] for x in r['runs']] == variants
    samples = vectors = 0; timings = {}
    for row in r['runs']:
        variant = row['variant']; part = path/variant
        assert row == json.loads((part/'report.json').read_text())
        assert row['complete'] and row['passed'] and row['exit_code'] == 0 and not row['guard_errors']
        assert len(row['memory']) > 10 and all(x['VmSwap'] == 0 for x in row['memory'])
        samples += len(row['memory'])
        assert row['outputs'] == {x.name:sha(x) for x in part.glob('*.bin')}
        log = (part/'stderr.log').read_text()
        for name in row['outputs']:
            if not name.endswith('-tokens.bin'):
                assert (part/name).stat().st_size == 152576*4; vectors += 1
        if variant.startswith('deadline-'):
            assert row['stop_position'] == (33 if variant=='deadline-json' else chunk)
            assert row['prompt_tokens'] == (33 if variant=='deadline-json' else 380)
            assert set(row['outputs']) == {'1-tokens.bin','2-tokens.bin','2-prefill.bin',
                                           '2-decode0.bin','2-decode1.bin','2-decode2.bin'}
            for name in row['outputs']:
                if name.startswith('2-'): assert sha(part/name) == sha(reference/'recreated'/('1'+name[1:]))
            h = row['final_health']
            assert h['ready'] and h['served']==2 and h['deadline_stops']==1 and h['cancelled']==h['faults']==0
            assert log.count('finish=deadline') == 1 and 'finish=cancelled' not in log
            timings[variant] = row['deadline_response_seconds']
        else:
            assert 'finish=shutdown' in log and 'finish=cancelled' not in log
            assert row['shutdown_requested'] and 0 <= row['shutdown_seconds'] < 120
            timings[variant] = row['shutdown_seconds']
            if variant == 'shutdown-decode':
                assert set(row['outputs']) == {'1-tokens.bin','1-prefill.bin','1-decode0.bin'}
                for name in row['outputs']: assert sha(part/name)==sha(reference/'on'/('2'+name[1:]))
            elif variant == 'clean-prefill': assert not row['outputs']
            else:
                assert row['stop_position']==chunk and row['prompt_tokens']==380
                assert set(row['outputs'])=={'1-tokens.bin'}
            if variant == 'supervised-prefill':
                starts = [x for x in row['events'] if x['event']=='started']
                exits = [x for x in row['events'] if x['event']=='exited']
                assert len(starts)==len(exits)==1 and starts[0]['replacement']==0
                assert exits[0]['returncode']==0 and starts[0]['pid']==exits[0]['pid']==row['worker_pid']
                assert not any(x['event']=='shutdown_timeout' for x in row['events'])
    assert vectors == 10
    return dict(passed=True,variants=variants,worker_samples=samples,exact_full_vectors=vectors,
                boundary_seconds=timings,report_sha256=sha(path/'report.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--server-build',type=Path,required=True);a=p.parse_args()
    result=validate(a.run,a.server_build)
    with (a.run/'validation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

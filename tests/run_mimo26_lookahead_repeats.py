"""Complete an off/on/on/off lookahead comparison after a qualified off run.

Every leg uses a fresh worker, cold expert mappings, the same numerical objects
and exact-output/route/memory gates. Startup is excluded from prefill timing.
CPU route replay runs only after all timed GPU legs finish.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    reference = args.reference.resolve()
    ref = json.loads((reference / 'report.json').read_text())
    assert ref['complete'] and ref['passed'] and ref['candidate']
    assert not ref['lookahead_enabled']
    workload = ref['workload']
    assert len(workload['cases']) == 1 or workload['cases'] == [0, 1, 2]
    # Do not silently benchmark sources different from the first control.
    for name, digest in ref['inputs'].items():
        if not name.endswith('.o'):
            assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
    output = args.output.resolve()
    output.mkdir()
    summary = dict(complete=False, passed=False, workload=workload, commands=[], runs=[])
    def save():
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    save()
    try:
        runs = [('off1', reference)]
        for label, enabled in [('on1', True), ('on2', True), ('off2', False)]:
            path = output / label
            command = [sys.executable, str(repo / 'tests/run_mimo26_lookahead_corpus.py'),
                       '--candidate', '--reference', str(reference), '--output', str(path),
                       '--tokens', str(workload['tokens']), '--chunk', str(workload['chunk']),
                       '--timeout', str(ref['timeout_seconds'])]
            if len(workload['cases']) == 1:
                command += ['--case', str(workload['cases'][0])]
            if not enabled:
                command += ['--default-off']
            summary['commands'].append(command)
            save()
            subprocess.run(command, check=True, cwd=repo)
            runs.append((label, path))
        for label, path in runs:
            report = json.loads((path / 'report.json').read_text())
            assert report['complete'] and report['passed']
            assert report['workload'] == workload and report['outputs'] == ref['outputs']
            assert report['route_sha256'] == ref['route_sha256']
            for name, digest in ref['inputs'].items():
                if not name.endswith('.o'):
                    assert report['inputs'][name] == digest, name
            command = [sys.executable, str(repo / 'tests/analyze_mimo26_corpus_routes.py'),
                       str(path), '--slots', str(workload['slots'])]
            summary['commands'].append(command)
            subprocess.run(command, check=True, cwd=repo)
            summary['runs'].append(dict(label=label, path=str(path),
                report_sha256=hashlib.sha256((path / 'report.json').read_bytes()).hexdigest(),
                results=report['results'], total_prefill_seconds=sum(x['seconds'] for x in report['results']),
                samples=len(report['memory']),
                max_worker_swap_KiB=max(row.get('VmSwap', 0) for row in report['memory'])))
            save()
        means = {policy: statistics.mean(row['total_prefill_seconds'] for row in summary['runs']
                                        if row['label'].startswith(policy)) for policy in ('off', 'on')}
        summary.update(complete=True, passed=True, mean_prefill_seconds=means,
                       elapsed_reduction_percent=100 * (1 - means['on'] / means['off']),
                       throughput_ratio=means['off'] / means['on'],
                       scope='fixed synthetic corpus; two repetitions per policy in ABBA order; aggregate instrumented prefill only; not upstream quality or a serving benchmark')
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        summary['error'] = repr(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()

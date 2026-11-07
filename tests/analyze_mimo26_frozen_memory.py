"""Summarize frozen-run host/GTT/reclaim observations without assigning cause."""
import argparse
import hashlib
import json
from pathlib import Path


def fields(value):
    if not isinstance(value, str):
        return {}
    result = {}
    for line in value.splitlines():
        pieces = line.replace(':', '').split()
        if len(pieces) >= 2 and pieces[1].isdigit():
            result[pieces[0]] = int(pieces[1])
    return result


def summarize(report):
    samples = []
    for snap in report['snapshots']:
        files = snap['files']
        status = next((fields(value) for name, value in files.items() if name.endswith('/status')), {})
        mem = fields(files.get('/proc/meminfo'))
        samples.append(dict(seconds=snap['monotonic_seconds']-report['snapshots'][0]['monotonic_seconds'],
                            available_KiB=mem.get('MemAvailable'), swap_KiB=status.get('VmSwap'),
                            rss_KiB=status.get('VmRSS'), vm=fields(files.get('/proc/vmstat')),
                            gtt_bytes={name: int(value.strip()) for name, value in files.items()
                                       if name.endswith('mem_info_gtt_used') and isinstance(value,str) and value.strip().isdigit()},
                            cgroup={name: value for name,value in files.items() if name.startswith('/sys/fs/cgroup/')}))
    keys = ['pswpout', 'pswpin', 'pgscan_kswapd', 'pgscan_direct', 'pgsteal_kswapd',
            'allocstall_normal', 'compact_stall', 'compact_success', 'compact_fail']
    first_swap = next((sample for sample in samples if sample['swap_KiB']), None)
    deltas = {key: samples[-1]['vm'].get(key,0)-samples[0]['vm'].get(key,0) for key in keys}
    return dict(complete=report['complete'], passed=report['passed'], exit_code=report.get('exit_code'),
                guard_reason=report.get('guard_reason'), worker_samples=sum(s['swap_KiB'] is not None for s in samples),
                first_swap=first_swap, max_worker_swap_KiB=max((s['swap_KiB'] or 0) for s in samples),
                min_available_GiB=min(s['available_KiB'] for s in samples if s['available_KiB'] is not None)/1048576,
                max_gtt_GiB=max((value for s in samples for value in s['gtt_bytes'].values()),default=0)/1073741824,
                global_vm_deltas=deltas,
                scope='one frozen-binary diagnostic, not a replacement benchmark; global telemetry includes other processes; no causal attribution')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('run', type=Path); a=p.parse_args()
    report_path = a.run / 'report.json'
    report = json.loads(report_path.read_text()); assert report['complete']
    result = summarize(report)
    result['report_sha256'] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    result['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (a.run/'analysis.json').open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

"""Summarize load telemetry without inferring a kernel cause from free RAM."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
a = p.parse_args()
r = json.loads((a.run / 'report.json').read_text())
def fields(text):
    result = {}
    for line in text.splitlines():
        parts = line.replace(':', '').split()
        if len(parts) >= 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1])
    return result
samples = []
for snap in r['snapshots']:
    mem = fields(snap.get('/proc/meminfo', ''))
    vm = fields(snap.get('/proc/vmstat', ''))
    status = next((fields(v) for k, v in snap.items() if k.endswith('/status')), {})
    samples.append(dict(seconds=snap['seconds'], available_KiB=mem.get('MemAvailable'),
                        swap_KiB=status.get('VmSwap'), vm=vm))
allocs = [json.loads(line[6:]) for line in (a.run / 'stderr.log').read_text().splitlines()
          if line.startswith('ALLOC ')]
keys = ['pswpout', 'pswpin', 'pgscan_kswapd', 'pgscan_direct', 'pgsteal_kswapd',
        'allocstall_normal', 'compact_stall', 'compact_success', 'compact_fail']
out = dict(complete=r['complete'], loaded=r['loaded'], exit_code=r.get('exit_code'),
           swap_guard=r.get('swap_guard'), slots=r['slots'],
           first_swap=next((s for s in samples if s['swap_KiB']), None),
           last_allocation=allocs[-1] if allocs else None,
           min_available_KiB=min(s['available_KiB'] for s in samples if s['available_KiB'] is not None),
           global_vm_deltas={k: samples[-1]['vm'].get(k, 0)-samples[0]['vm'].get(k, 0) for k in keys},
           scope='global counters can include other processes; allocation trace is instrumented; cause not established')
(a.run / 'analysis.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))

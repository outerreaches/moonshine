"""Correlate a bounded reclaim trace with frozen-worker telemetry; no policy tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from analyze_mimo26_frozen_memory import fields, summarize


def zones(value):
    result = {}
    if not isinstance(value, str):
        return result
    for block in re.split(r'(?=^Node \d+, zone)', value, flags=re.M):
        match = re.match(r'Node (\d+), zone\s+(\w+)', block)
        if not match:
            continue
        row = {}
        for key in ('pages free', 'boost', 'min', 'low', 'high', 'managed'):
            found = re.search(r'^\s*' + key + r'\s+(\d+)\s*$', block, re.M)
            if found:
                row[key.replace(' ', '_')] = int(found[1])
        result['node' + match[1] + '/' + match[2]] = row
    return result


def buddies(value):
    if not isinstance(value, str):
        return {}
    return {'node' + node + '/' + zone: [int(x) for x in counts.split()]
            for node, zone, counts in re.findall(r'^Node (\d+), zone\s+(\w+)\s+([\d\s]+)$', value, re.M)}


def analyze(report, trace):
    result = summarize(report)
    result['scope'] = ('Bounded global kernel trace plus sampled frozen-worker telemetry; '
                       'call stacks identify observed reclaim callers, not the cause of a historical untraced swap.')
    snapshots = report['snapshots']
    start = snapshots[0]['monotonic_seconds']
    rows = []
    cgroups = {}
    for snap in snapshots:
        f = snap['files']
        rows.append(dict(seconds=snap['monotonic_seconds']-start,
                         zones=zones(f.get('/proc/zoneinfo')),
                         buddies=buddies(f.get('/proc/buddyinfo')),
                         available_KiB=fields(f.get('/proc/meminfo')).get('MemAvailable')))
        for name, value in f.items():
            if not name.startswith('/sys/fs/cgroup/') or not isinstance(value, str):
                continue
            if name.endswith(('memory.high', 'memory.max', 'memory.swap.max', 'memory.min', 'memory.low')):
                entry = cgroups.setdefault(name, dict(observed=[]))
                if value.strip() not in entry['observed']:
                    entry['observed'].append(value.strip())
            elif name.endswith(('memory.events', 'memory.events.local')):
                entry = cgroups.setdefault(name, dict(first=fields(value), last={}))
                entry['last'] = fields(value)
    for entry in cgroups.values():
        if 'first' in entry:
            entry['delta'] = {key: entry['last'].get(key, 0)-val for key, val in entry['first'].items()}
    result['cgroup_limits_events'] = cgroups
    result['zones'] = {}
    for zone in sorted({z for r in rows for z in r['zones']}):
        selected = [r['zones'][zone] for r in rows if zone in r['zones']]
        orders = [r['buddies'][zone] for r in rows if zone in r['buddies']]
        result['zones'][zone] = dict(
            first=selected[0], last=selected[-1],
            min_free_pages=min(r['pages_free'] for r in selected),
            max_boost_pages=max(r['boost'] for r in selected),
            min_free_minus_high_pages=min(r['pages_free']-r['high'] for r in selected),
            first_buddy_orders=orders[0] if orders else [],
            min_buddy_order_counts=[min(r[i] for r in orders) for i in range(len(orders[0]))] if orders else [])
    events = []
    for ns, pid, tid, zone, order, flags in re.findall(
            r'WORKER_WAKE ns=(\d+) pid=(\d+) tid=(\d+) zone=(\d+) order=(\d+) gfp=(\w+)', trace):
        assert int(pid) == report['pid'], 'trace belongs to another worker'
        seconds = int(ns)/1e9-start
        before = [r for r in rows if r['seconds'] <= seconds]
        after = [r for r in rows if r['seconds'] >= seconds]
        events.append(dict(seconds=seconds, pid=int(pid), tid=int(tid), zone=int(zone),
                           order=int(order), gfp_hex=flags,
                           before=before[-1] if before else None, after=after[0] if after else None))
    result['worker_wake_events'] = events
    result['worker_wake_count'] = len(events)
    result['kswapd_wake_count'] = len(re.findall(r'^KSWAPD_WAKE ', trace, re.M))
    result['trace_complete'] = 'TRACE_READY' in trace and 'TRACE_END' in trace
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('trace', type=Path)
    a = p.parse_args()
    source = a.run/'report.json'
    report = json.loads(source.read_text())
    assert report['complete']
    result = analyze(report, a.trace.read_text())
    for name, path in [('report', source), ('trace', a.trace), ('analyzer', Path(__file__))]:
        result[name+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (a.run/'reclaim-analysis.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ('worker_wake_events', 'cgroup_limits_events')}, indent=2))


if __name__ == '__main__':
    main()

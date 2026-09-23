"""Verify per-prompt counters and evaluate known-chunk next-use eviction."""
import argparse
import hashlib
import json
from pathlib import Path
from analyze_mimo26_routes import simulate
from analyze_mimo26_lookahead import lookahead


def parse_cases(lines, workload):
    cases = {}
    current = key = None
    for line in lines:
        if line.startswith('CASE_BEGIN '):
            assert current is None, 'nested case'
            current = int(line.split()[1])
            assert current not in cases, 'duplicate case'
            cases[current] = []
            key = None
        elif line.startswith('CASE_END '):
            assert current == int(line.split()[1]), 'unmatched case end'
            current = None
        elif line.startswith('ROUTE '):
            assert current is not None, 'route outside prefill case'
            _, layer, start, t, *ids = line.split()
            identity = (int(layer), int(start))
            ids = list(map(int, ids))
            assert len(ids) == 8 and ids == sorted(set(ids))
            assert all(0 <= x < 256 for x in ids)
            if identity != key:
                cases[current].append((identity, []))
                key = identity
            assert int(t) == len(cases[current][-1][1])
            cases[current][-1][1].append(ids)
    assert current is None, 'unfinished case'
    assert list(cases) == workload['cases'], 'case set/order mismatch'
    expected = [(layer, pos) for pos in range(0, workload['tokens'], workload['chunk'])
                for layer in range(1, 48)]
    for chunks in cases.values():
        assert [key for key, _ in chunks] == expected, 'chunk/layer order mismatch'
        for (_, pos), rows in chunks:
            assert len(rows) == min(workload['chunk'], workload['tokens'] - pos), 'chunk length mismatch'
    return {case: [(key[0], rows) for key, rows in chunks] for case, chunks in cases.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--slots', type=int, nargs='+', default=[16, 48, 96, 144])
    a = p.parse_args()
    assert all(8 <= slots <= 256 for slots in a.slots)
    r = json.loads((a.run / 'report.json').read_text())
    assert r['complete'] and r['passed']
    workload = r.get('workload', dict(tokens=128, chunk=32, cases=[0, 1, 2], slots=48))
    cases = parse_cases((a.run / 'stderr.log').read_text().splitlines(), workload)
    observed = {row['case']: row for row in r['results']}
    assert len(observed) == len(r['results']) and set(observed) == set(cases)
    results = {}
    for case, chunks in cases.items():
        results[str(case)] = {}
        for slots in sorted(set(a.slots + [workload['slots']])):
            lru = simulate(chunks, slots)[0]
            future = lookahead(chunks, slots)
            if slots == workload['slots']:
                expected = future if r.get('lookahead_enabled', r['candidate']) else lru
                actual = observed[case]
                assert (expected['misses'], expected['hits'], expected['accesses']) == (
                    actual['uploads'], actual['hits'], actual['accesses']), 'runtime counters differ'
            results[str(case)][str(slots)] = dict(
                lru=lru, lookahead=future,
                miss_reduction_percent=100 * (1 - future['misses'] / lru['misses']))
    out = dict(complete=True, passed=True, workload=workload, results=results,
               observed_counters_match=True,
               route_sha256=hashlib.sha256((a.run / 'stderr.log').read_bytes()).hexdigest(),
               report_sha256=hashlib.sha256((a.run / 'report.json').read_bytes()).hexdigest(),
               analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               scope='fixed synthetic domain prompts; cold per prompt; logical traffic not latency')
    (a.run / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({case: data[str(workload['slots'])] for case, data in results.items()}, indent=2))


if __name__ == '__main__':
    main()

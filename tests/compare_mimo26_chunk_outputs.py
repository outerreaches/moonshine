"""Compare qualified corpus outputs across chunk widths, without assuming parity."""
import argparse
import hashlib
import json
from pathlib import Path
from analyze_mimo26_corpus_routes import parse_cases


def load(path):
    report = json.loads((path / 'report.json').read_text())
    assert report['complete'] and report['passed']
    assert all(hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
               for name, digest in report['outputs'].items())
    assert hashlib.sha256((path / 'stderr.log').read_bytes()).hexdigest() == report['route_sha256']
    cases = parse_cases((path / 'stderr.log').read_text().splitlines(), report['workload'])
    # Logs are chunk-major/layer-major. Canonicalize by layer then token to
    # compare routes rather than comparing different log serialization orders.
    routes = {case: {layer: [row for chunk_layer, rows in chunks if chunk_layer == layer
                            for row in rows] for layer in range(1, 48)}
              for case, chunks in cases.items()}
    return report, routes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    left, left_routes = load(a.left)
    right, right_routes = load(a.right)
    assert {k: v for k, v in left['workload'].items() if k != 'chunk'} == {
        k: v for k, v in right['workload'].items() if k != 'chunk'}, 'workloads differ beyond chunk width'
    assert left['outputs'].keys() == right['outputs'].keys()
    matches = {name: digest == right['outputs'][name] for name, digest in left['outputs'].items()}
    assert all(equal for name, equal in matches.items() if name.endswith('-tokens.bin')), 'different input tokens'
    result = dict(complete=True, passed=all(matches.values()) and left_routes == right_routes,
                  left_chunk=left['workload']['chunk'], right_chunk=right['workload']['chunk'],
                  left_report_sha256=hashlib.sha256((a.left / 'report.json').read_bytes()).hexdigest(),
                  right_report_sha256=hashlib.sha256((a.right / 'report.json').read_bytes()).hexdigest(),
                  file_equal=matches, canonical_routes_equal=left_routes == right_routes,
                  scope='same-input exact route/full-vector comparison for this corpus only; not general cross-width quality qualification')
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

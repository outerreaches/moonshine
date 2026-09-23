"""Offline completeness and repeated-request checks for the live server gate."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    report = json.loads((a.run / 'report.json').read_text())
    assert report['complete'] and report['passed']
    total_vectors = 0
    for row in report['runs']:
        variant = row['variant']
        if variant == 'clean':
            assert not row['outputs']
            continue
        outputs = row['outputs']
        for name, digest in outputs.items():
            data = (a.run / variant / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest
            if not name.endswith('-tokens.bin'):
                assert len(data) == 152576 * 4
                total_vectors += 1
        if variant == 'fault':
            assert set(outputs) == {'1-tokens.bin'}
            continue
        requests = {1: row['answer_a']['usage']['completion_tokens']}
        if variant in ('off', 'on'):
            requests.update({2: row['answer_b']['usage']['completion_tokens'],
                             3: requests[1], 5: requests[1],
                             6: row['answer_c']['usage']['completion_tokens']})
        expected = {'4-tokens.bin'} if variant in ('off', 'on') else set()
        for request_id, count in requests.items():
            expected.update({f'{request_id}-tokens.bin', f'{request_id}-prefill.bin'})
            expected.update(f'{request_id}-decode{i}.bin' for i in range(count))
        assert set(outputs) == expected, (variant, set(outputs) ^ expected)
        if variant in ('off', 'on'):
            for request_id in (3, 5):
                for name, digest in outputs.items():
                    if name.startswith('1-'):
                        assert outputs[str(request_id) + name[1:]] == digest, (variant, request_id, name)
    result = dict(complete=True, passed=True, full_vectors=total_vectors,
                  mixed_repeat_and_post_cancel_vectors_exact=True,
                  report_sha256=hashlib.sha256((a.run / 'report.json').read_bytes()).hexdigest())
    with (a.run / 'output-validation.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

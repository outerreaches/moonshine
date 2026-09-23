import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from compare_mimo26_chunk_outputs import load


class Tests(unittest.TestCase):
    def fixture(self, directory, chunk):
        path = Path(directory) / str(chunk)
        path.mkdir()
        tokens = 5
        lines = ['CASE_BEGIN 3']
        for pos in range(0, tokens, chunk):
            for layer in range(1, 48):
                for t in range(min(chunk, tokens - pos)):
                    ids = ' '.join(str((pos + t) * 8 + k) for k in range(8))
                    lines.append(f'ROUTE {layer} {pos} {t} {ids}')
        lines.append('CASE_END 3')
        routes = ('\n'.join(lines) + '\n').encode()
        (path / 'stderr.log').write_bytes(routes)
        output = b'fixture output'
        (path / 'output-3-prefill.bin').write_bytes(output)
        report = dict(complete=True, passed=True,
                      workload=dict(tokens=tokens, chunk=chunk, cases=[3], slots=48),
                      outputs={'output-3-prefill.bin': hashlib.sha256(output).hexdigest()},
                      route_sha256=hashlib.sha256(routes).hexdigest())
        (path / 'report.json').write_text(json.dumps(report))
        return path

    def test_reordered_chunk_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = (self.fixture(directory, chunk) for chunk in (2, 4))
            self.assertNotEqual((a / 'stderr.log').read_bytes(), (b / 'stderr.log').read_bytes())
            self.assertEqual(load(a)[1], load(b)[1])

    def test_output_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory, 2)
            (path / 'output-3-prefill.bin').write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                load(path)

    def test_route_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory, 2)
            with (path / 'stderr.log').open('a') as f:
                f.write('unexpected new diagnostic\n')
            with self.assertRaises(AssertionError):
                load(path)


if __name__ == '__main__':
    unittest.main()

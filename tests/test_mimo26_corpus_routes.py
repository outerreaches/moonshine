import unittest
from analyze_mimo26_corpus_routes import parse_cases


class Tests(unittest.TestCase):
    def setUp(self):
        self.workload = dict(tokens=5, chunk=2, cases=[3], slots=48)
        self.lines = ['CASE_BEGIN 3'] + [
            f'ROUTE {layer} {pos} {t} 0 1 2 3 4 5 6 7'
            for pos in (0, 2, 4) for layer in range(1, 48)
            for t in range(min(2, 5 - pos))] + ['CASE_END 3']

    def test_tail(self):
        cases = parse_cases(self.lines, self.workload)
        self.assertEqual(len(cases[3]), 141)
        self.assertEqual(len(cases[3][-1][1]), 1)

    def test_missing_row(self):
        with self.assertRaises(AssertionError):
            parse_cases(self.lines[:1] + self.lines[2:], self.workload)

    def test_duplicate_expert(self):
        self.lines[1] = self.lines[1].replace('6 7', '6 6')
        with self.assertRaises(AssertionError):
            parse_cases(self.lines, self.workload)

    def test_wrong_chunk_position(self):
        self.lines = [line.replace('ROUTE 1 4 ', 'ROUTE 1 3 ') for line in self.lines]
        with self.assertRaises(AssertionError):
            parse_cases(self.lines, self.workload)

    def test_unfinished_case(self):
        with self.assertRaises(AssertionError):
            parse_cases(self.lines[:-1], self.workload)

    def test_wrong_case(self):
        self.workload['cases'] = [0]
        with self.assertRaises(AssertionError):
            parse_cases(self.lines, self.workload)


if __name__ == '__main__':
    unittest.main()

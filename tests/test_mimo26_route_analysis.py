import unittest
from analyze_mimo26_routes import simulate

class ReplayTests(unittest.TestCase):
    def test_cold_repeated_and_retained(self):
        chunks=[(1,[[0,1,2,3,4,5,6,7]]*3)]
        cold=simulate(chunks,8,passes=2)
        self.assertEqual([r['misses'] for r in cold],[8,8])
        warm=simulate(chunks,8,passes=2,retain=True)
        self.assertEqual([r['misses'] for r in warm],[8,0])
        self.assertEqual(simulate(chunks,8,grouped=True)[0]['misses'],8)

    def test_grouping_reduces_thrash_but_not_math(self):
        chunks=[(1,[list(range(8)),list(range(8,16)),list(range(8))])]
        self.assertEqual(simulate(chunks,8)[0]['misses'],24)
        self.assertEqual(simulate(chunks,8,grouped=True)[0]['misses'],16)

    def test_layers_have_independent_caches(self):
        chunks=[(1,[list(range(8))]),(2,[list(range(8))])]
        self.assertEqual(simulate(chunks,8)[0]['misses'],16)

if __name__=='__main__':unittest.main()

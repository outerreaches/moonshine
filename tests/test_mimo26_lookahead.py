import unittest
from analyze_mimo26_lookahead import lookahead
class Tests(unittest.TestCase):
    def test_preserves_current_selection(self):
        r=lookahead([(1,[[1,2],[3,2],[2,3]])],2)
        self.assertEqual(r['misses'],3);self.assertEqual(r['hits'],3)
    def test_next_use(self):
        r=lookahead([(1,[[1],[2],[3],[1],[2]])],2)
        self.assertEqual(r['misses'],4)
    def test_layer_isolation(self):
        r=lookahead([(1,[[1,2]]),(2,[[1,2]]),(1,[[1,2]])],2)
        self.assertEqual(r['misses'],4);self.assertEqual(r['hits'],2)
    def test_chunk_boundary_has_no_future_oracle(self):
        r=lookahead([(1,[[1],[2],[3]]),(1,[[1]])],2)
        self.assertEqual(r['misses'],4)
if __name__=='__main__':unittest.main()

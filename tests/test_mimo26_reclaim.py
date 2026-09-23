import unittest

from analyze_mimo26_reclaim import analyze, buddies, zones


class ReclaimAnalysis(unittest.TestCase):
    def test_zone_fields_and_empty_zones(self):
        text = ('Node 0, zone Normal\n  pages free 100\n boost 0\n min 2\n low 3\n high 4\n managed 200\n'
                'Node 1, zone DMA32\n pages free 7\n boost 1\n min 2\n low 4\n high 5\n managed 8\n')
        self.assertEqual(zones(text)['node0/Normal']['pages_free'], 100)
        self.assertEqual(zones(text)['node1/DMA32']['high'], 5)
        self.assertEqual(zones({'error': 'missing'}), {})

    def test_buddy_orders_not_pages(self):
        text = 'Node 0, zone Normal 1 2 3\nNode 1, zone DMA32 4 5 6\n'
        self.assertEqual(buddies(text), {'node0/Normal': [1, 2, 3], 'node1/DMA32': [4, 5, 6]})
        self.assertEqual(buddies(None), {})

    def test_ancestor_event_deltas_and_trace_time(self):
        snaps = []
        for second, high in [(10., 7), (11., 8)]:
            snaps.append(dict(monotonic_seconds=second, files={
                '/proc/meminfo': 'MemAvailable: 100000 kB\n', '/proc/42/status': 'VmSwap: 0 kB\n',
                '/sys/fs/cgroup/user.slice/memory.high': 'max\n',
                '/sys/fs/cgroup/user.slice/memory.events': f'high {high}\noom 0\n'}))
        report = dict(snapshots=snaps, complete=True, passed=True, pid=42, exit_code=0)
        trace = ('TRACE_READY\nWORKER_WAKE ns=10500000000 pid=42 tid=43 zone=2 order=10 gfp=12\n'
                 'KSWAPD_WAKE ns=10500000001 zone=2 order=10\nTRACE_END\n')
        result = analyze(report, trace)
        self.assertEqual(result['worker_wake_count'], 1)
        self.assertEqual(result['kswapd_wake_count'], 1)
        event = result['worker_wake_events'][0]
        self.assertEqual(event['seconds'], .5)
        self.assertEqual(event['before']['seconds'], 0)
        self.assertEqual(event['after']['seconds'], 1)
        self.assertEqual(result['cgroup_limits_events']['/sys/fs/cgroup/user.slice/memory.events']['delta']['high'], 1)
        with self.assertRaises(AssertionError):
            analyze(report, trace.replace('pid=42', 'pid=99'))


if __name__ == '__main__':
    unittest.main()

"""Read-only host swap/GPU residency companion to per-worker qualification guards."""
import argparse
import json
from pathlib import Path
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid', type=int, required=True, help='driver PID whose lifetime bounds monitoring')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
start = time.monotonic()
with a.output.open('x') as out:
    while Path(f'/proc/{a.pid}').exists():
        row = dict(seconds=time.monotonic()-start, timestamp=time.time())
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemAvailable', 'SwapFree', 'SwapTotal'): row[key+'_kib'] = int(value.split()[0])
        for line in Path('/proc/vmstat').read_text().splitlines():
            key, value = line.split()
            if key in ('pswpin', 'pswpout'): row[key+'_pages'] = int(value)
        row['gpu'] = {x.name: int(x.read_text()) for x in Path('/sys/class/drm/card0/device').glob('mem_info_*')}
        processes = []
        for path in Path('/proc').glob('[0-9]*/status'):
            try:
                fields = dict(x.split(':', 1) for x in path.read_text().splitlines())
                swap = int(fields.get('VmSwap', '0').split()[0])
                if swap: processes.append(dict(pid=int(path.parent.name), name=fields['Name'].strip(), swap_kib=swap))
            except (OSError, ValueError, KeyError): pass
        row['swapped_processes'] = sorted(processes, key=lambda x: -x['swap_kib'])
        out.write(json.dumps(row)+'\n'); out.flush()
        time.sleep(1)

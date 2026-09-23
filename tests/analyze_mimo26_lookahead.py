"""Replay only: next-use eviction using ONLY the current chunk's known routes."""
import argparse
import hashlib
import json
from pathlib import Path
from analyze_mimo26_routes import simulate

def lookahead(chunks,slots):
    cache={layer:[] for layer in range(1,48)}
    misses=hits=accesses=0
    for layer,rows in chunks:
        lru=cache[layer]
        for t,selection in enumerate(rows):
            assert len(set(selection))==len(selection) and len(selection)<=slots
            protected=set(selection);old=set(lru)
            hits+=sum(x in old for x in selection);accesses+=len(selection)
            for expert in selection:
                if expert in lru:continue
                misses+=1
                if len(lru)==slots:
                    # No visibility beyond this chunk. Equal distance evicts LRU.
                    candidates=[x for x in lru if x not in protected]
                    assert candidates
                    def next_use(x):
                        return next((j for j in range(t+1,len(rows)) if x in rows[j]),len(rows))
                    victim=max(candidates,key=next_use)
                    lru.remove(victim)
                lru.append(expert)
            for expert in selection:
                lru.remove(expert);lru.append(expert)
    return dict(accesses=accesses,hits=hits,misses=misses,logical_read_GiB=misses*12.75/1024)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text());assert report['complete'] and report['passed']
    chunks=[];key=None
    for line in (a.run/'trace-stderr.log').read_text().splitlines():
        if line=='PASS_BEGIN 1':break
        if not line.startswith('ROUTE '):continue
        _,layer,start,t,*ids=line.split();identity=(int(layer),int(start))
        if key!=identity:chunks.append((identity[0],[]));key=identity
        assert int(t)==len(chunks[-1][1]);chunks[-1][1].append(list(map(int,ids)))
    assert len(chunks)==188 and all(len(rows)==32 for _,rows in chunks)
    assert simulate(chunks,48)[0]['misses']==14897
    results={}
    for slots in (16,48,96,144):
        baseline=simulate(chunks,slots)[0];candidate=lookahead(chunks,slots)
        results[str(slots)]=dict(lru=baseline,chunk_lookahead=candidate,
            miss_reduction_percent=100*(1-candidate['misses']/baseline['misses']))
    out=dict(complete=True,passed=True,results=results,
        source_report_sha256=hashlib.sha256((a.run/'report.json').read_bytes()).hexdigest(),
        route_log_sha256=hashlib.sha256((a.run/'trace-stderr.log').read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='single 128-token trace; chunk32 route-only replay; logical reads not physical I/O or latency; no runtime implementation; no future-chunk oracle')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()

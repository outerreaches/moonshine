"""Logical cache replay only. Verify the token-batch LRU against captured counters."""
import argparse
import collections
import json
from pathlib import Path

def simulate(chunks,slots,grouped=False,passes=1,retain=False):
    total=[];cache={i:[] for i in range(1,48)}
    for repeat in range(passes):
        if not retain:cache={i:[] for i in range(1,48)}
        hits=accesses=misses=0
        for layer,rows in chunks:
            batches=([x] for x in sorted({x for row in rows for x in row})) if grouped else rows
            for selection in batches:
                lru=cache[layer];old=set(lru)
                hits+=sum(x in old for x in selection);accesses+=len(selection)
                misses+=sum(x not in old for x in selection)
                for x in selection:
                    if x in lru:lru.remove(x)
                    elif len(lru)==slots:lru.pop(0)
                    lru.append(x)
        total.append(dict(accesses=accesses,hits=hits,misses=misses,
            logical_read_GiB=misses*12.75/1024))
    return total

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text());assert report['complete'] and report['passed']
    chunks=[];current=None;first=True
    for line in (a.run/'trace-stderr.log').read_text().splitlines():
        if line.startswith('PASS_BEGIN '):
            if line!='PASS_BEGIN 0':first=False
            continue
        if not first or not line.startswith('ROUTE '):continue
        _,layer,pos,t,*ids=line.split();layer,pos,t=int(layer),int(pos),int(t);ids=list(map(int,ids))
        assert len(ids)==8 and ids==sorted(set(ids)) and all(0<=x<256 for x in ids)
        key=(layer,pos)
        if key!=current:
            assert t==0;chunks.append((layer,[]));current=key
        assert t==len(chunks[-1][1]);chunks[-1][1].append(ids)
    assert len(chunks)==4*47 and all(len(rows)==32 for _,rows in chunks)
    observed=[json.loads(x) for x in (a.run/'trace-stdout.log').read_text().splitlines()]
    matched=simulate(chunks,48)[0]
    assert matched['hits']==observed[0]['hits'] and matched['accesses']==observed[0]['accesses']
    assert matched['misses']==observed[0]['uploads_cumulative']
    results={}
    for slots in (48,96,144):
        token=simulate(chunks,slots)[0];group=simulate(chunks,slots,True)[0]
        results[str(slots)]=dict(token_order=token,ascending_union=group,
            grouped_miss_change_percent=100*(group['misses']/token['misses']-1),
            repeat_same_prompt_with_resident_weights=simulate(chunks,slots,passes=2,retain=True))
    unions=[len({x for row in rows for x in row}) for _,rows in chunks]
    output=dict(complete=True,passed=True,source_report=report['head'],tokens=128,chunk=32,
        baseline_counter_match=matched,union_min=min(unions),union_max=max(unions),
        union_mean=sum(unions)/len(unions),results=results,
        scope='single natural-text trace, cold initial cache; logical reads not latency; grouping is not implemented and changes eviction order; repeated-prompt retention is optimistic not held-out reuse evidence')
    (a.run/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()

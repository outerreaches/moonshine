"""Validate direct-I/O screen completeness; report paired admission and makespan."""
import argparse
import json
from pathlib import Path
import statistics as S
from run_mimo26_rans_io_gate import sha, SCREEN

MODES=('original','raw_fixture','rans16_w4','rans16_w8','rans64_w1')


def analyze(report,monitor,rows):
    assert report['complete'] and report['passed'] and monitor['complete'] and monitor['passed']
    assert monitor['exit_code']==0 and 'guard_reason' not in monitor
    rounds=report['rounds']
    keys=[(x['round'],x['contention'],x['qd'],x['mode']) for x in rows]
    expected={(r,c,q,m) for r in range(rounds+1) for c in (0,1) for q in (1,2,6) for m in MODES}
    assert len(keys)==len(expected) and set(keys)==expected
    lookup=dict(zip(keys,rows))
    for row in rows:
        assert sorted(row['expert_order'])==list(range(6)) and row['exact_bytes']==80216064
        assert row['read_bytes']==row['requested_bytes']
        expected_bytes={'original':80216064+6*4096,'raw_fixture':80216064,
                        'rans16_w4':72560640,'rans16_w8':72560640,'rans64_w1':72077312}
        # Original starts need not be unaligned in a different checkpoint; this pinned screen is.
        assert row['requested_bytes']==expected_bytes[row['mode']]
        assert row['admission_ms']>0 and row['makespan_ms']>=row['admission_ms']
        assert row['compute_experts']==(128 if row['contention'] else 0)
        assert row['expert_order']==lookup[(row['round'],row['contention'],row['qd'],'original')]['expert_order']
    samples=[s for s in monitor['samples'] if 'VmSwap:' in s['status']]
    assert samples
    for sample in samples:
        assert int(next(x.split()[1] for x in sample['status'].splitlines() if x.startswith('VmSwap:')))==0
    groups=[]
    for c in (0,1):
        for q in (1,2,6):
            for mode in MODES:
                selected=[lookup[(r,c,q,mode)] for r in range(1,rounds+1)]
                group=dict(contention=c,qd=q,mode=mode,samples=len(selected))
                for metric in ('admission_ms','makespan_ms','compute_ms'):
                    group['median_'+metric]=S.median(x[metric] for x in selected)
                for control in ('original','raw_fixture'):
                    for metric in ('admission_ms','makespan_ms'):
                        percentages=[100*(x[metric]/lookup[(x['round'],c,q,control)][metric]-1) for x in selected]
                        group[control+'_'+metric+'_paired_change_percent']=dict(median=S.median(percentages),
                            minimum=min(percentages),maximum=max(percentages))
                groups.append(group)
    return dict(passed=True,trials=len(rows),measured_trials=rounds*30,worker_swap_samples=len(samples),
                summaries=groups,scope=report['scope'],
                caveat='Repeated six-expert microbenchmark, not a full-model serving result. Negative change is faster. '
                       'O_DIRECT bypasses page cache, not NVMe firmware caching. Fixed-size real-kernel compute with synthetic inputs.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text());monitor=json.loads((a.run/'monitor.json').read_text())
    assert sha(a.run/'gate.so')==report['library_sha256'] and sha(SCREEN)==report['screen_sha256']
    for name,digest in monitor['sources'].items():assert sha(a.run/Path(name).name)==digest
    rows=[json.loads(line) for line in (a.run/'trials.jsonl').read_text().splitlines()]
    result=analyze(report,monitor,rows);result['report_sha256']=sha(a.run/'report.json')
    with (a.run/'analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'},indent=2))


if __name__=='__main__':main()

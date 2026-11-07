"""Independent completeness checks for transient recovery and bounded clean soak."""
import argparse
import json
from pathlib import Path
from run_mimo26_server_lookahead_gate import sha
from run_mimo26_transient_recovery import BASE


def validate_recovery(path):
    r=json.loads((path/'report.json').read_text());assert r['complete'] and r['passed'] and r['exit_code']==0
    assert not r['guard_errors']
    started=[x for x in r['events'] if x['event']=='started'];exited=[x for x in r['events'] if x['event']=='exited']
    assert len(started)==len(exited)==2 and [x['replacement'] for x in started]==[0,1]
    assert [x['pid'] for x in started]==[x['pid'] for x in exited] and all(x['returncode']==0 for x in exited)
    assert r['events'].index(exited[0])<r['events'].index(started[1])
    assert [x['reason'] for x in r['events'] if x['event']=='retiring']==['quarantined','stopped']
    for row in started:
        samples=[x for x in r['memory'] if x['pid']==row['pid'] and 'VmSwap:' in x['status']]
        assert len(samples)>10
        assert all(int(next(s.split()[1] for s in x['status'].splitlines() if s.startswith('VmSwap:')))==0 for x in samples)
    for name,digest in r['hashes'].items():assert sha(path/name)==digest
    expected={x.name:sha(x) for x in (BASE/'recreated').glob('*.bin')}
    assert r['outputs']==expected=={x.name:sha(x) for x in (path/'replacement').glob('*.bin')}
    assert {x.name for x in (path/'first').glob('*.bin')}=={'1-tokens.bin'}
    for file in (path/'replacement').glob('*.bin'):
        if 'tokens' not in file.name:assert file.stat().st_size==152576*4
    assert (path/'stderr.log').read_text().count('TEST_HTTP_FAULT pending=8')==1
    assert r['final_health']['served']==1 and r['final_health']['faults']==0
    return dict(passed=True,kind='transient_recovery',worker_samples=len(r['memory']),full_vectors=len(expected)-1)


def validate_soak(path):
    r=json.loads((path/'report.json').read_text());assert r['complete'] and r['passed'] and r['exit_code']==0
    assert sha(path/'server-clean')==r['binary_sha256'] and not r['guard_errors']
    assert r['memory'] and all(x.get('VmSwap')==0 for x in r['memory'])
    chats=[x for x in r['responses'] if x['path']=='/v1/chat/completions']
    statuses={code:sum(x['status']==code for x in chats) for code in (200,400,404,503)}
    assert statuses=={200:16,400:6,404:3,503:3},statuses
    assert len([x for x in chats if isinstance(x['body'],list)])==3
    final=r['final_health'];assert final['ready'] and final['phase']=='idle'
    # Server's served counter includes the three completed cancellation handlers.
    assert final['faults']==0 and final['cancelled']==3 and final['served']==19
    assert final['context']==1024 and final['prefill_chunk']==64 and final['expert_lookahead']
    warmed=r['idle'][2:];assert len(warmed)>4
    fd_range=max(x['fd_count'] for x in warmed)-min(x['fd_count'] for x in warmed)
    rss_delta=warmed[-1]['VmRSS']-warmed[0]['VmRSS'];assert fd_range<=2 and rss_delta<512*1024
    return dict(passed=True,kind='clean_soak',worker_samples=len(r['memory']),chat_statuses=statuses,
                elapsed_seconds=r['elapsed_seconds'],warmed_fd_range=fd_range,warmed_rss_delta_KiB=rss_delta)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=('recovery','soak'));p.add_argument('run',type=Path);a=p.parse_args()
    result=(validate_recovery if a.kind=='recovery' else validate_soak)(a.run)
    result['report_sha256']=sha(a.run/'report.json')
    with (a.run/'validation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

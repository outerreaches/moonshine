"""Verify the hardened binary's full-output identity and live HTTP refusal scope."""
import argparse
import json
from pathlib import Path
from run_mimo26_server_lookahead_gate import sha
from run_mimo26_transient_recovery import BASE


def validate(path,reference):
    r=json.loads((path/'report.json').read_text())
    old=json.loads((reference/'report.json').read_text())
    assert r['complete'] and r['passed'] and old['complete'] and old['passed']
    assert r['transport_probes'] and r['slots']==16
    assert [x['variant'] for x in r['runs']]==['off','on','fault','recreated','clean']
    for name,digest in r['binaries'].items():assert sha(path/name)==digest
    # The build snapshot predates optional live probes. Verify both identities:
    # inputs names the build driver; execution_driver names the actual run.
    for name,digest in r['inputs'].items():assert sha(path/Path(name).name)==digest,name
    for name,digest in r['built_objects'].items():assert sha(path/name)==digest,name
    assert sha(path/'execution_driver.py')==r['execution_driver_sha256']
    total=0;samples=0
    for row,previous in zip(r['runs'],old['runs']):
        assert row['variant']==previous['variant'] and row['outputs']==previous['outputs']
        assert row['passed'] and row['exit_code']==0 and not row['monitor_errors']
        assert row['memory'] and all(x['VmSwap']==0 for x in row['memory'])
        samples+=len(row['memory'])
        for name,digest in row['outputs'].items():
            file=path/row['variant']/name;assert sha(file)==digest
            if not name.endswith('-tokens.bin'):assert file.stat().st_size==152576*4;total+=1
    assert total==56
    clean=r['runs'][-1];assert len(clean['transport'])==5
    for i,probe in enumerate(clean['transport']):
        header,body=probe['response'].split('\r\n\r\n',1)
        assert header.startswith('HTTP/1.1 400 ') and json.loads(body)['error']['code']=='malformed_request'
        assert 0<=probe['seconds']<8 and (i<3 or probe['seconds']>=4.5)
    before=True;checks=0
    for response in clean['responses']:
        if response['path']=='/v1/chat/completions':before=False
        if before and response['path']=='/health' and response['status']==200:
            health=response['body'];assert health['admitted']==0 and health['served']==0 and health['faults']==0
            checks+=1
    assert checks>=6
    return dict(passed=True,full_vectors_exact_to_prior=total,worker_swap_samples=samples,
                live_refusals=5,stalled_read_seconds=[x['seconds'] for x in clean['transport'][3:]],
                source_report_sha256=sha(path/'report.json'),reference_report_sha256=sha(reference/'report.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--reference',type=Path,default=BASE);a=p.parse_args()
    result=validate(a.run,a.reference)
    with (a.run/'transport-validation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

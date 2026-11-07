"""Validate gate completeness and summarize warm-memory codec timings, not SSD speed."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def analyze(report, monitor, screen):
    assert report['complete'] and report['passed'] and monitor['complete'] and monitor['passed']
    assert monitor['exit_code']==0 and 'guard_reason' not in monitor
    assert report['model_revision']==screen['model_revision']
    assert len(report['inputs'])==36 and len({x['name'] for x in report['inputs']})==36
    expected_inputs=[x for x in screen['matrices'] if x['group']=='heldout']
    assert report['inputs']==expected_inputs
    rows=report['cases']
    identities=[(x['layer'],x['expert'],x['tile'],x['waves_per_block']) for x in rows]
    expected={(l,e,t,w) for l in (1,24,47) for e in (97,255) for t in (16384,32768,65536) for w in (1,4,8)}
    assert len(identities)==54 and set(identities)==expected
    assert len(report['synthetic'])==12
    assert {(x['layout'],x['fixture']) for x in report['synthetic']}=={(l,i) for l in (0,1) for i in range(6)}
    assert all(x['raw_bytes']==13369344 and x['tiles']==13369344//x['tile'] for x in rows)
    assert all(x['fault_checks']>0 and x['host_faults']>0 and x['device_faults']>0
               for x in rows+report['synthetic'] if x['faults_enabled'])
    for row in rows+report['synthetic']:
        assert row['host_faults']+row['device_faults']==row['fault_checks']
        for samples, median in [('decode_samples','decode_ms'),('copy_samples','raw_copy_ms'),
                                ('admission_samples','checked_admission_ms')]:
            assert len(row[samples])==5 and all(x>0 for x in row[samples])
            assert statistics.median(row[samples])==row[median]
    for sample in monitor['samples']:
        assert int(next((x.split()[1] for x in sample['status'].splitlines() if x.startswith('VmSwap:')), '0'))==0
    assert len(monitor['samples'])>0
    summaries=[]
    for tile in (16384,32768,65536):
        for waves in (1,4,8):
            selected=[x for x in rows if x['tile']==tile and x['waves_per_block']==waves]
            block_bytes=sum(x['block_bytes'] for x in selected)
            assert block_bytes==screen['groups']['heldout']['sizes'][str(tile)]['expert_block_bytes']
            summaries.append(dict(tile_bytes=tile,waves_per_block=waves,expert_count=len(selected),
                heldout_block_bytes=block_bytes,saving_percent=100*(1-block_bytes/80216064),
                median_decode_ms=statistics.median(x['decode_ms'] for x in selected),
                median_raw_copy_ms=statistics.median(x['raw_copy_ms'] for x in selected),
                median_checked_admission_ms=statistics.median(x['checked_admission_ms'] for x in selected)))
    return dict(passed=True,exact_expert_configurations=54,synthetic_configurations=12,
                selected_fault_checks=sum(x['fault_checks'] for x in rows+report['synthetic']),
                worker_swap_samples=len(monitor['samples']),summaries=summaries,
                scope=report['scope'],integrity='Fused 32-bit local MZG2-style checksum plus stream terminal checks. '
                'Accidental-corruption guard, not cryptographic authentication or exhaustive fault coverage.')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('screen',type=Path); a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text()); monitor=json.loads((a.run/'monitor.json').read_text())
    screen=json.loads(a.screen.read_text())
    assert hashlib.sha256(a.screen.read_bytes()).hexdigest()==report['screen_sha256']
    assert hashlib.sha256((a.run/'gate.so').read_bytes()).hexdigest()==report['library_sha256']
    for name,digest in report['sources'].items():
        assert hashlib.sha256((a.run/name).read_bytes()).hexdigest()==digest
    result=analyze(report,monitor,screen)
    result['report_sha256']=hashlib.sha256((a.run/'report.json').read_bytes()).hexdigest()
    with (a.run/'analysis.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

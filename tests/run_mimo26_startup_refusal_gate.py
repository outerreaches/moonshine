"""CPU-only selected startup refusals on a frozen candidate, before worker creation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from run_mimo26_server_lookahead_gate import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    build=a.server_build.resolve();out=a.output.resolve();out.mkdir()
    candidate=json.loads((build/'report.json').read_text())
    assert candidate['complete'] and candidate['passed']
    binary=build/'server-clean';assert sha(binary)==candidate['binaries']['server-clean']
    shutil.copy2(__file__,out/Path(__file__).name)
    absent=out/'absent-model-root';assert not absent.exists()
    cases=[('missing_model',[str(absent)],1),('file_instead_of_model',[str(binary)],1),
           ('invalid_context',[str(absent),'--context','0'],2),
           ('unknown_option',[str(absent),'--unknown','1'],2)]
    r=dict(complete=False,passed=False,binary_sha256=sha(binary),
           source_report_sha256=sha(build/'report.json'),cases=[])
    try:
        for name,args,code in cases:
            command=[str(binary)]+args
            result=subprocess.run(command,capture_output=True,text=True,timeout=10)
            r['cases'].append(dict(name=name,command=command,exit_code=result.returncode,
                                   stdout=result.stdout,stderr=result.stderr))
            assert result.returncode==code and 'loading the worker' not in result.stdout
            assert ('tokenizer:' if code==1 else 'configuration:') in result.stderr
        r.update(complete=True,passed=True,scope='Selected pre-worker refusals only; no loaded-worker bind failure, checkpoint corruption, OOM or driver-hang qualification.')
        print('PASS four selected pre-worker startup refusals')
    finally:(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()

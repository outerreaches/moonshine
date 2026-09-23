import copy
import unittest

from analyze_mimo26_rans_gpu_gate import analyze


class CodecAnalysis(unittest.TestCase):
    def fixtures(self):
        inputs=[dict(name=str(i),group='heldout') for i in range(36)]
        sample=dict(raw_bytes=13369344,block_bytes=12000000,raw_tiles=0,
                    fault_checks=2,host_faults=1,device_faults=1,faults_enabled=1)
        for values,mid in [('decode_samples','decode_ms'),('copy_samples','raw_copy_ms'),
                           ('admission_samples','checked_admission_ms')]:
            sample[values]=[1.,2.,3.,4.,5.]; sample[mid]=3.
        rows=[dict(sample,layer=l,expert=e,tile=t,waves_per_block=w,tiles=13369344//t)
              for l in (1,24,47) for e in (97,255) for t in (16384,32768,65536) for w in (1,4,8)]
        synthetic=[dict(sample,layout=l,fixture=i) for l in (0,1) for i in range(6)]
        report=dict(complete=True,passed=True,model_revision='test',inputs=inputs,cases=rows,
                    synthetic=synthetic,scope='fixture')
        monitor=dict(complete=True,passed=True,exit_code=0,samples=[dict(status='VmSwap: 0 kB\n')])
        screen=dict(model_revision='test',matrices=copy.deepcopy(inputs),groups={'heldout':{'sizes':{
            str(t):dict(expert_block_bytes=72000000) for t in (16384,32768,65536)}}})
        return report,monitor,screen

    def test_complete(self):
        result=analyze(*self.fixtures())
        self.assertEqual(result['exact_expert_configurations'],54)
        self.assertEqual(result['selected_fault_checks'],132)

    def test_duplicate_and_missing_configuration(self):
        report,monitor,screen=self.fixtures()
        report['cases'][-1]=report['cases'][0]
        with self.assertRaises(AssertionError): analyze(report,monitor,screen)

    def test_swap_and_size_mismatch(self):
        report,monitor,screen=self.fixtures()
        monitor['samples'][0]['status']='VmSwap: 4 kB\n'
        with self.assertRaises(AssertionError): analyze(report,monitor,screen)
        report,monitor,screen=self.fixtures()
        report['cases'][0]['block_bytes']+=4096
        with self.assertRaises(AssertionError): analyze(report,monitor,screen)

    def test_missing_fault_and_wrong_median(self):
        report,monitor,screen=self.fixtures()
        report['cases'][0]['fault_checks']=0
        with self.assertRaises(AssertionError): analyze(report,monitor,screen)
        report,monitor,screen=self.fixtures()
        report['cases'][0]['decode_ms']=4.
        with self.assertRaises(AssertionError): analyze(report,monitor,screen)


if __name__=='__main__': unittest.main()

"""Deterministic supervisor policy checks; no model or GPU required."""
import importlib.util
from pathlib import Path
import subprocess
import json
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('supervise',Path(__file__).resolve().parents[1]/'tools/mimo26_supervise.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Child:
    pid=123
    def __init__(self, stuck=False):self.returncode=None;self.stuck=stuck;self.terminated=False
    def poll(self):return self.returncode
    def terminate(self):self.terminated=True
    def wait(self,timeout):
        if self.stuck:raise subprocess.TimeoutExpired('child',timeout)
        self.returncode=0;return 0

class Tests(unittest.TestCase):
    def test_profile_defaults_and_explicit_opt_in(self):
        a=m.parse_args(['server','root'])
        self.assertEqual((a.slots,a.context,a.prefill_chunk,a.expert_lookahead),(16,2048,32,'off'))
        self.assertEqual(a.shutdown_timeout,30)
        self.assertEqual(m.parse_args(['server','root','--shutdown-timeout','120']).shutdown_timeout,120)
        with tempfile.TemporaryDirectory() as directory:
            server=Path(directory)/'server';server.touch()
            defaults=m.parse_args([str(server),directory])
            self.assertNotIn('--prefill-chunk',m.server_command(defaults))
            a=m.parse_args([str(server),directory,'--prefill-chunk','64','--expert-lookahead','on'])
            command=m.server_command(a)
            self.assertEqual(command[-4:],['--prefill-chunk','64','--expert-lookahead','on'])
            self.assertEqual(command[2:4],['--host','127.0.0.1'])
    def test_invalid_profiles_before_process_launch(self):
        import contextlib
        import io
        for args in (['--context','4294967296'],['--prefill-chunk','129'],
                     ['--prefill-chunk','-1'],['--prefill-chunk','0','--expert-lookahead','on'],
                     ['--expert-lookahead','true'],['--shutdown-timeout','nan'],
                     ['--shutdown-timeout','inf'],['--shutdown-timeout','0'],
                     ['--shutdown-timeout','-1'],['--shutdown-timeout','3601']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                m.parse_args(['server','root']+args)
    def supervisor(self,**kw):
        s=m.Supervisor(['test'],9000,backoff=0,poll=0,log=lambda *a,**k:None,**kw)
        s.port_free=lambda:True
        return s
    def test_persistent_fault_budget(self):
        s=self.supervisor(restarts=2);s.health=lambda:'fault'
        children=[]
        def spawn(*a):
            self.assertTrue(all(c.returncode==0 for c in children))
            c=Child();children.append(c);return c
        with patch.object(m.subprocess,'Popen',side_effect=spawn):self.assertEqual(s.run(),1)
        self.assertEqual(len(children),3);self.assertTrue(all(c.terminated for c in children))
    def test_busy_does_not_restart(self):
        s=self.supervisor();states=iter(['ready','unknown','unknown','stop']);c=Child()
        def health():
            state=next(states)
            if state=='stop':s.stop.set()
            return state
        s.health=health
        with patch.object(m.subprocess,'Popen',return_value=c) as spawn:
            self.assertEqual(s.run(),0);spawn.assert_called_once()
        self.assertTrue(c.terminated)
    def test_shutdown_timeout_prevents_replacement(self):
        s=self.supervisor();s.health=lambda:'fault';c=Child(stuck=True)
        with patch.object(m.subprocess,'Popen',return_value=c) as spawn:
            self.assertEqual(s.run(),1);spawn.assert_called_once()
        self.assertIs(s.child,c)
    def test_occupied_port_not_touched(self):
        s=self.supervisor();s.port_free=lambda:False
        with patch.object(m.subprocess,'Popen') as spawn:
            self.assertEqual(s.run(),1);spawn.assert_not_called()
    def test_startup_timeout_budget(self):
        s=self.supervisor(restarts=0,startup=0);s.health=lambda:'unknown'
        with patch.object(m.subprocess,'Popen',return_value=Child()):self.assertEqual(s.run(),1)
    def test_gpu_occupied_not_touched(self):
        s=self.supervisor();s.gpu_free=lambda:False
        with patch.object(m.subprocess,'Popen') as spawn:
            self.assertEqual(s.run(),1);spawn.assert_not_called()
    def test_exception_retires_owned_child(self):
        s=self.supervisor();c=Child()
        def bad_health():raise RuntimeError('test exception')
        s.health=bad_health
        with patch.object(m.subprocess,'Popen',return_value=c):
            with self.assertRaises(RuntimeError):s.run()
        self.assertTrue(c.terminated);self.assertIsNone(s.child)
    def test_stop_during_backoff(self):
        s=self.supervisor();s.health=lambda:'fault'
        def log(line,**kw):
            if '"event": "backoff"' in line:s.stop.set()
        s.log=log
        with patch.object(m.subprocess,'Popen',return_value=Child()) as spawn:
            self.assertEqual(s.run(),0);spawn.assert_called_once()

    def real_case(self,mode):
        with tempfile.TemporaryDirectory(prefix='mimo26-supervisor-test-') as directory:
            record=Path(directory)/'children.txt'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            events=[];started=time.monotonic()
            s=m.Supervisor([sys.executable,str(Path(__file__).with_name('mimo26_fake_health_child.py')),str(port),str(record),mode],
                           port,restarts=1,startup=5,shutdown=3,backoff=0.05,poll=0.05)
            def log(line,**kw):
                event=json.loads(line);events.append(event)
                if event['event']=='ready':s.stop.set()
            s.log=log
            code=s.run()
            self.assertLess(time.monotonic()-started,15)
            self.assertEqual(len(record.read_text().splitlines()),2)
            self.assertIsNone(s.child)
            self.assertEqual(sum(e['event']=='started' for e in events),2)
            self.assertEqual(sum(e['event']=='exited' for e in events),2)
            self.assertEqual(code,1 if mode=='persistent' else 0)
            self.assertEqual(sum(e['event']=='ready' for e in events),0 if mode=='persistent' else 1)
    def test_real_process_replacement(self):self.real_case('once')
    def test_real_persistent_fault_budget(self):self.real_case('persistent')

if __name__=='__main__':unittest.main()

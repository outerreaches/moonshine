#!/usr/bin/env python3
"""Test-only exec shim: first owned child injects; replacements use same binary cleanly."""
import os
from pathlib import Path
import sys

root=Path(os.environ['MIMO26_ONCE_TEST_DIRECTORY'])
try:
    fd=os.open(root/'fault-claimed',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
except FileExistsError:
    name='replacement'
    os.environ.pop('MIMO26_TEST_HTTP_SUBMIT_FAILURE',None)
else:
    os.close(fd)
    name='first'
    os.environ['MIMO26_TEST_HTTP_SUBMIT_FAILURE']='1'
capture=root/name
capture.mkdir()
os.environ['MIMO26_TEST_CAPTURE']=str(capture)
binary=os.environ['MIMO26_ONCE_TEST_BINARY']
os.execv(binary,[binary]+sys.argv[1:])

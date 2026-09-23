"""CPU-only subprocess fixture; not a model server."""
import http.server
import json
import os
from pathlib import Path
import sys

port=int(sys.argv[1]);record=Path(sys.argv[2]);mode=sys.argv[3]
generation=len(record.read_text().splitlines()) if record.exists() else 0
with record.open('a') as f:f.write(str(os.getpid())+'\n')
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        fault=mode=='persistent' or generation==0
        body=json.dumps(dict(phase='quarantined' if fault else 'idle',ready=not fault)).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass
http.server.HTTPServer(('127.0.0.1',port),Handler).serve_forever()

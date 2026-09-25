#!/usr/bin/env python3
"""Opt-in, loopback-only process supervision. Never replays requests."""
import argparse
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


class Supervisor:
    def __init__(self, command, port, *, restarts=2, startup=120, shutdown=30,
                 backoff=2, poll=1, log=print, gpu_free=lambda: True,
                 expected_profile=None):
        self.command, self.port = command, port
        self.limit, self.startup, self.shutdown = restarts, startup, shutdown
        self.backoff, self.poll, self.log = backoff, poll, log
        self.stop = threading.Event()
        self.child = None
        self.gpu_free = gpu_free
        self.expected_profile = expected_profile

    def event(self, event, **details):
        self.log(json.dumps(dict(event=event, **details)), flush=True)

    def port_free(self):
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', self.port))
            return True
        except OSError:
            return False

    def health(self):
        # A timeout/503 is unknown, not a fault: a legitimate prefill may be busy.
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=1) as response:
                data = json.loads(response.read(4096))
            if data.get('phase') == 'quarantined' and data.get('ready') is False:
                return 'fault'
            if data.get('ready') is True and data.get('phase') == 'idle':
                if self.expected_profile is not None and any(
                        data.get(key) != value for key, value in self.expected_profile.items()):
                    self.event('profile_mismatch', expected=self.expected_profile,
                               actual={key: data.get(key) for key in self.expected_profile})
                    return 'profile_mismatch'
                return 'ready'
        except (OSError, ValueError, urllib.error.URLError):
            pass
        return 'unknown'

    def retire(self):
        if self.child is None:
            return True
        if self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=self.shutdown)
            except subprocess.TimeoutExpired:
                # Do not launch a replacement alongside an unretired GPU owner.
                self.event('shutdown_timeout', pid=self.child.pid)
                return False
        self.event('exited', pid=self.child.pid, returncode=self.child.returncode)
        self.child = None
        return True

    def run(self):
        attempts = 0  # lifetime budget, never reset by a brief healthy interval
        try:
            while not self.stop.is_set():
                if not self.port_free():
                    self.event('port_occupied', port=self.port)
                    return 1
                if not self.gpu_free():
                    self.event('gpu_unavailable')
                    return 1
                self.child = subprocess.Popen(self.command)
                self.event('started', pid=self.child.pid, replacement=attempts)
                started = time.monotonic()
                ready = False
                reason = 'stopped'
                while not self.stop.is_set():
                    if self.child.poll() is not None:
                        reason = 'process_exit'
                        break
                    state = self.health()
                    if state == 'ready' and not ready:
                        ready = True
                        self.event('ready', pid=self.child.pid)
                    if state == 'fault':
                        reason = 'quarantined'
                        break
                    if state == 'profile_mismatch':
                        reason = state
                        break
                    if not ready and time.monotonic() - started >= self.startup:
                        reason = 'startup_timeout'
                        break
                    self.stop.wait(self.poll)
                self.event('retiring', reason=reason)
                if not self.retire():
                    return 1
                if reason == 'profile_mismatch':
                    return 1  # Relaunching the same wrong profile cannot repair it.
                if self.stop.is_set():
                    return 0
                if attempts >= self.limit:
                    self.event('restart_budget_exhausted', replacements=attempts)
                    return 1
                attempts += 1
                self.event('backoff', seconds=self.backoff, replacement=attempts)
                self.stop.wait(self.backoff)
            return 0
        except Exception:
            if self.child is not None:
                self.retire()
            raise
        finally:
            if self.child is not None and self.child.poll() is not None:
                self.retire()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('server', type=Path)
    p.add_argument('root', type=Path)
    p.add_argument('--port', type=int, default=8640)
    # These mirror the server's own defaults deliberately. The supervisor pins
    # a profile rather than inheriting one, so an older or newer server binary
    # cannot quietly serve something else -- but that means the two sets can
    # drift, which is exactly what happened: these sat at 16/2048/32 with
    # lookahead off long after the server had moved to the qualified
    # 160/131072/128 with it on. The expected_profile check below is what
    # catches the next drift, so every pinned value must also be on /health.
    p.add_argument('--slots', type=int, default=160)
    p.add_argument('--context', type=int, default=131072)
    p.add_argument('--restarts', type=int, default=2)
    p.add_argument('--prefill-chunk', type=int, default=128)
    p.add_argument('--expert-lookahead', choices=('off', 'on'), default='on')
    p.add_argument('--expert-major', choices=('off', 'on'), default='on')
    p.add_argument('--retain-experts', choices=('off', 'on'), default='on',
                   help='retain healthy expert cache between requests; context is always reset')
    p.add_argument('--kv-prefix-reuse', choices=('off', 'on'), default='off',
                   help='continue from a matching prefix instead of re-prefilling it')
    p.add_argument('--prefix-cache-dir', type=Path, default=None,
                   help='directory of persisted prefix checkpoints; needs --kv-prefix-reuse on')
    p.add_argument('--prefix-cache-gib', type=int, default=16)
    p.add_argument('--prefix-cache-entries', type=int, default=8)
    p.add_argument('--request-deadline-seconds', type=int, default=600)
    p.add_argument('--min-headroom-gib', type=int, default=8)
    p.add_argument('--shutdown-timeout', type=float, default=30,
                   help='seconds to wait for an owned child to reach a safe boundary; no force kill')
    a = p.parse_args(argv)
    if not (1 <= a.port <= 65535 and 8 <= a.slots <= 256 and 1 <= a.context <= 0xffffffff and a.restarts >= 0):
        p.error('invalid port, slots, context or restart budget')
    if not 0 <= a.prefill_chunk <= 128 or (a.expert_lookahead == 'on' and a.prefill_chunk == 0):
        p.error('invalid chunk or lookahead with chunk zero')
    # The same couplings the server enforces, refused here too so a bad profile
    # never reaches a launch.
    if a.expert_major == 'on' and a.expert_lookahead == 'off':
        p.error('--expert-major on needs --expert-lookahead on')
    if a.prefix_cache_dir is not None and a.kv_prefix_reuse == 'off':
        p.error('--prefix-cache-dir needs --kv-prefix-reuse on')
    if not 1 <= a.request_deadline_seconds <= 86400:
        p.error('request deadline must be in [1, 86400] seconds')
    if not 0 <= a.min_headroom_gib <= 512:
        p.error('headroom floor must be in [0, 512] GiB')
    if not (1 <= a.prefix_cache_gib <= 4096 and 1 <= a.prefix_cache_entries <= 64):
        p.error('invalid prefix cache budget or entry limit')
    if not math.isfinite(a.shutdown_timeout) or not 0 < a.shutdown_timeout <= 3600:
        p.error('shutdown timeout must be finite and in (0, 3600] seconds')
    return a


def server_command(a):
    server = a.server.resolve(strict=True)
    command = [str(server), str(a.root.resolve(strict=True)), '--host', '127.0.0.1',
               '--port', str(a.port), '--slots', str(a.slots), '--context', str(a.context)]
    # Always transmit the resolved profile: worker defaults can change between
    # builds. Older binaries that lack these options must refuse the launch,
    # rather than silently run a different profile.
    command += ['--prefill-chunk', str(a.prefill_chunk), '--expert-lookahead', a.expert_lookahead]
    command += ['--retain-experts', a.retain_experts]
    command += ['--expert-major', a.expert_major]
    command += ['--request-deadline-seconds', str(a.request_deadline_seconds)]
    command += ['--min-headroom-gib', str(a.min_headroom_gib)]
    command += ['--kv-prefix-reuse', a.kv_prefix_reuse]
    if a.prefix_cache_dir is not None:
        command += ['--prefix-cache-dir', str(a.prefix_cache_dir.resolve()),
                    '--prefix-cache-gib', str(a.prefix_cache_gib),
                    '--prefix-cache-entries', str(a.prefix_cache_entries)]
    return command


def main():
    a = parse_args()
    command = server_command(a)
    def gpu_free():
        check = subprocess.run(['fuser', '/dev/kfd'], capture_output=True)
        return check.returncode == 1 and not check.stdout.strip()
    supervisor = Supervisor(command, a.port, restarts=a.restarts,
                            shutdown=a.shutdown_timeout, gpu_free=gpu_free,
                            expected_profile=dict(expert_slots=a.slots, context=a.context,
                                                  prefill_chunk=a.prefill_chunk,
                                                  expert_lookahead=a.expert_lookahead == 'on',
                                                  expert_major=a.expert_major == 'on',
                                                  retain_experts=a.retain_experts == 'on',
                                                  kv_prefix_reuse=a.kv_prefix_reuse == 'on',
                                                  request_deadline_seconds=a.request_deadline_seconds,
                                                  min_headroom_gib=a.min_headroom_gib))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: supervisor.stop.set())
    return supervisor.run()


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prove the supervisor's --require-signature gate can refuse.

A signature check is the kind of gate that passes because it cannot fail, so
this drives the supervisor's own resolve_release() -- not the packager CLI --
against scratch copies of a real signed release, and includes a control that
attributes the refusals to the flag rather than to the copying.

Not part of `make test`: it needs a signed release and an allowed_signers file,
neither of which exists in a clean checkout.

    python3 tests/prove_mimo26_signature_gate.py \
        --release-root /srv/modelstore/private/moonshine-releases/mimo26 \
        --allowed-signers ~/.ssh/allowed_signers

Exits 0 only if every case behaves as required. The live release is read, never
written; every mutation happens in a scratch tree that is removed afterwards.
"""
import argparse
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent


def load_supervisor():
    spec = importlib.util.spec_from_file_location(
        'mimo26_supervise', TREE / 'tools' / 'mimo26_supervise.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_writable(root):
    """Releases are sealed 0555/0444; scratch copies must be editable."""
    for path in [root, *root.rglob('*')]:
        mode = path.stat().st_mode | stat.S_IWUSR
        if path.is_dir():
            mode |= stat.S_IXUSR
        path.chmod(mode)


def clone_release(live_root, scratch, name):
    """A scratch release root whose current/ points at a copy of live current/."""
    root = scratch / name
    (root / 'releases').mkdir(parents=True)
    live = (live_root / os.readlink(live_root / 'current')).resolve()
    release = root / 'releases' / live.name
    shutil.copytree(live, release)
    make_writable(release)
    (root / 'current').symlink_to(Path('releases') / live.name)
    return root, release


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--release-root', type=Path, required=True)
    p.add_argument('--allowed-signers', type=Path, required=True)
    a = p.parse_args(argv)

    live_root = a.release_root.resolve()
    signers = a.allowed_signers.expanduser().resolve()
    if not (live_root / 'current').is_symlink():
        print(f'no current release under {live_root}')
        return 2
    if not signers.exists():
        print(f'no allowed_signers at {signers}')
        return 2

    supervise = load_supervisor()
    cases = []

    with tempfile.TemporaryDirectory(prefix='mimo26-sigproof-') as tmp:
        scratch = Path(tmp)

        def resolve(name, mutate=None, require=True, use_signers=True):
            root, release = clone_release(live_root, scratch, name)
            if mutate is not None:
                mutate(release, scratch)
            binary, _, problem = supervise.resolve_release(
                root, signers if use_signers else None, require)
            return binary, problem

        def drop_signature(release, _scratch):
            (release / 'manifest.sha256.sig').unlink()

        def flip_a_bit(release, _scratch):
            binary = release / 'mimo26_server'
            data = bytearray(binary.read_bytes())
            data[-1] ^= 0x01
            binary.write_bytes(bytes(data))

        def sign_with_a_rogue_key(release, scratch):
            key = scratch / 'rogue-key'
            subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '',
                            '-C', 'rogue', '-f', str(key)], check=True)
            (release / 'manifest.sha256.sig').unlink()
            subprocess.run(['ssh-keygen', '-Y', 'sign', '-q', '-f', str(key),
                            '-n', 'moonshine', str(release / 'manifest.sha256')],
                           check=True, capture_output=True)

        binary, problem = resolve('intact')
        cases.append(('signed and intact resolves', binary is not None, problem))

        binary, problem = resolve('unsigned', drop_signature)
        cases.append(('missing signature refused', binary is None, problem))

        binary, problem = resolve('tampered', flip_a_bit)
        cases.append(('tampered binary refused', binary is None, problem))

        binary, problem = resolve('rogue', sign_with_a_rogue_key)
        cases.append(('signature by unlisted key refused', binary is None, problem))

        # The control. Without it the refusals above prove only that scratch
        # copies fail; with it they are attributable to the flag.
        binary, problem = resolve('unsigned-flag-off', drop_signature,
                                  require=False, use_signers=False)
        cases.append(('unsigned resolves when the flag is off',
                      binary is not None, problem))

        make_writable(scratch)

    failures = 0
    for label, ok, problem in cases:
        print(f'{"PASS" if ok else "FAIL"}  {label}')
        if problem:
            print(f'        -> {problem.splitlines()[-1][:120]}')
        failures += 0 if ok else 1

    print(f'\n{len(cases) - failures}/{len(cases)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

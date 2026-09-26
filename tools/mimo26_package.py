#!/usr/bin/env python3
"""Immutable release packaging and rollback for the MiMo serving lane.

Follows the convention this lane's evidence bundles already use -- pin the
inputs, the build command and the outputs by hash, and chain to the report
that qualified them -- and adds the two things a deployment needs that an
evidence bundle does not: an atomic way to make a release current, and an
atomic way to stop.

    build     assemble a release directory and seal it read-only
    verify    recompute every hash against the manifest
    activate  verify, then point `current` at it; the old `current` becomes
              `previous`
    rollback  swap `current` and `previous`
    list      show what is installed and which is live
    retire    unseal and delete a release that is neither current nor the
              rollback target
    sign      attach a detached ssh signature over manifest.sha256

Signing is deliberately a separate step: the key belongs to the operator, not
to the build. `verify --allowed-signers FILE` then checks it, and
`activate --require-signature` refuses anything unsigned. Trust lives in the
verifier's allowed_signers file rather than in the artifact, which is the
point -- a release cannot vouch for itself.

A release carries the qualification report that justifies it. `activate`
refuses a release without one, and refuses a build made from a dirty working
tree, because neither can be reproduced from history.

Layout:

    <root>/releases/<version>-<commit>-<stamp>/
        mimo26_server        0555
        manifest.json        0444
        manifest.sha256      0444
        qualification.json   0444
    <root>/current  -> releases/<name>
    <root>/previous -> releases/<name>
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = (".cu", ".c", ".h", ".py")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                       text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def toolchain():
    """What built this. Two builds from the same commit on different ROCm
    versions differ in the binary and in nothing else recorded, so without
    this the manifest cannot explain why."""
    def first_line(*command):
        try:
            r = subprocess.run(command, capture_output=True, text=True,
                               timeout=20)
            return (r.stdout or r.stderr).strip().splitlines()[0]
        except Exception:
            return None
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    return {
        "hipcc": first_line(f"{rocm}/bin/hipcc", "--version"),
        "cc": first_line("cc", "--version"),
        "rocm_path": rocm,
        "rocm_version": (Path(rocm) / ".info" / "version").read_text().strip()
        if (Path(rocm) / ".info" / "version").exists() else None,
        "uname": first_line("uname", "-srm"),
    }


def model_fingerprint(root):
    """Identify the checkpoint without reading 166 GiB of weights.

    Hashes the small files that define the model -- config, tokenizer, chat
    template, and the safetensors index that maps every tensor to its shard --
    and records each shard's name and size. That detects a different
    checkpoint, a missing shard or a truncated one. It does NOT detect an
    edit inside a shard that preserves its length, and is named a fingerprint
    rather than a hash for that reason.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    described = {}
    for name in ("config.json", "generation_config.json",
                 "tokenizer_config.json", "chat_template.jinja",
                 "model.safetensors.index.json"):
        item = root / name
        if item.is_file():
            described[name] = sha256(item)
    shards = {}
    for item in sorted(root.glob("*.safetensors")):
        shards[item.name] = item.stat().st_size
    digest = hashlib.sha256()
    for name in sorted(described):
        digest.update(name.encode()); digest.update(described[name].encode())
    for name in sorted(shards):
        digest.update(name.encode()); digest.update(str(shards[name]).encode())
    return {"root": str(root), "describing_files": described,
            "shard_count": len(shards),
            "shard_bytes": sum(shards.values()),
            "fingerprint": digest.hexdigest(),
            "covers": "config, tokenizer, chat template and tensor index by "
                      "content; shards by name and size only"}


def seal(path):
    """Read-only for everyone, and no new files in the directory."""
    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file():
            os.chmod(item, 0o444 if item.suffix else 0o555)
    os.chmod(path, 0o555)


def unseal(path):
    os.chmod(path, 0o755)
    for item in path.rglob("*"):
        if item.is_file():
            os.chmod(item, 0o644)


def build(a):
    binary = (REPO / "tools" / "mimo26_server").resolve()
    if not binary.exists():
        sys.exit("build the server first: make tools/mimo26_server")
    version = subprocess.run([str(binary), "--version"], capture_output=True,
                             text=True).stdout.strip()
    if not version:
        sys.exit("binary does not answer --version; it predates packaging")
    token = version.split()[1] if len(version.split()) > 1 else "unknown"

    commit = git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = bool(git("status", "--porcelain"))
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    name = f"{token}-{commit}{'-dirty' if dirty else ''}-{stamp}"
    out = Path(a.root).resolve() / "releases" / name
    if out.exists():
        sys.exit(f"{out} already exists")
    out.mkdir(parents=True)

    shutil.copy2(binary, out / "mimo26_server")
    binary_sha = sha256(out / "mimo26_server")
    qualification = None
    if a.qualification:
        source = Path(a.qualification).resolve()
        report = json.loads(source.read_text())
        """
        The link that makes a signature mean anything.

        A report's `version` is a compile-time constant shared by every build,
        so embedding a report proves nothing about the binary beside it. The
        report now names the binary it drove, by hashing the running inode, and
        a release whose evidence names a different one is refused here rather
        than discovered later by whoever trusted the signature.
        """
        tested = (report.get("build") or {}).get("sha256")
        if tested is None:
            shutil.rmtree(out, ignore_errors=True)
            sys.exit("qualification report does not identify the binary it "
                     "tested; re-run qualify.py, which records it")
        if tested != binary_sha:
            shutil.rmtree(out, ignore_errors=True)
            sys.exit(f"qualification tested {tested[:16]}… but this binary is "
                     f"{binary_sha[:16]}…; the report does not cover it")
        shutil.copy2(source, out / "qualification.json")
        qualification = {
            "source": str(source),
            "sha256": sha256(out / "qualification.json"),
            "label": report.get("label"),
            "build_sha256": tested,
            "stock_profile": report.get("stock_profile"),
            "profile": report.get("profile", {}),
            "checks_passed": sum(1 for c in report.get("checks", [])
                                 if c.get("pass")),
            "checks_total": len(report.get("checks", [])),
        }

    sources = {}
    for item in sorted(REPO.iterdir()):
        if item.is_file() and item.suffix in SOURCE_SUFFIXES:
            sources[item.name] = sha256(item)
    for item in sorted((REPO / "tools").iterdir()):
        if item.is_file() and item.suffix in SOURCE_SUFFIXES:
            sources["tools/" + item.name] = sha256(item)

    manifest = {
        "format": 1,
        "name": name,
        "version": version,
        "git": {"commit": git("rev-parse", "HEAD"), "short": commit,
                "dirty": dirty,
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "built_at_utc": stamp,
        "built_by": "tools/mimo26_package.py",
        "binary": {"name": "mimo26_server",
                   "sha256": binary_sha,
                   "bytes": (out / "mimo26_server").stat().st_size},
        "sources": sources,
        "toolchain": toolchain(),
        "qualification": qualification,
        "model": model_fingerprint(a.model) if a.model else None,
        "notes": a.note,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "manifest.sha256").write_text(sha256(out / "manifest.json") + "\n")
    seal(out)
    print(f"built {out}")
    print(f"  version      {version}")
    print(f"  commit       {commit}{' (DIRTY)' if dirty else ''}")
    print(f"  binary       {manifest['binary']['sha256'][:16]}… "
          f"{manifest['binary']['bytes']} bytes")
    print(f"  sources      {len(sources)} files pinned")
    if qualification:
        print(f"  qualified    {qualification['label']} "
              f"{qualification['checks_passed']}/{qualification['checks_total']}"
              f", binary matches")
        if qualification.get("stock_profile") is not True:
            print("               NOT the stock profile -- this release is "
                  "qualified for the flags that run were given, not for "
                  "no-flags defaults")
    else:
        print("  qualified    NONE -- activate will refuse this release")
    if manifest["model"]:
        m = manifest["model"]
        print(f"  model        {m['fingerprint'][:16]}… "
              f"{m['shard_count']} shards, "
              f"{m['shard_bytes'] / (1 << 30):.1f} GiB")
    print("  signature    none -- sign it with `sign` before shipping")
    return 0


SIGNATURE_NAMESPACE = "moonshine"


def sign(a):
    """Detached ssh signature over manifest.sha256, which transitively covers
    the manifest and therefore the binary and every pinned source."""
    release = Path(a.release).resolve()
    ok, manifest = verify_release(release, quiet=True)
    if not ok:
        print(f"refusing to sign: {manifest}")
        return 1
    target = release / "manifest.sha256"
    signature = release / "manifest.sha256.sig"
    if signature.exists() and not a.force:
        print("already signed; pass --force to replace")
        return 1
    unseal(release)
    result = subprocess.run(
        ["ssh-keygen", "-Y", "sign", "-f", str(Path(a.key).expanduser()),
         "-n", SIGNATURE_NAMESPACE, str(target)],
        capture_output=True, text=True)
    if result.returncode != 0:
        seal(release)
        print(f"signing failed: {(result.stderr or result.stdout).strip()}")
        return 1
    seal(release)
    print(f"signed {release.name} with {a.key}")
    print(f"  {signature.name} covers manifest.sha256 -> manifest.json -> "
          f"binary and sources")
    return 0


def check_signature(path, allowed_signers):
    """Returns (state, detail). State is absent, unchecked, good or bad."""
    signature = Path(path) / "manifest.sha256.sig"
    if not signature.exists():
        return "absent", "no signature"
    if not allowed_signers:
        return "unchecked", "signature present but no --allowed-signers given"
    identity = None
    for line in Path(allowed_signers).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            identity = line.split()[0]
            break
    if identity is None:
        return "bad", "allowed_signers is empty"
    with open(Path(path) / "manifest.sha256", "rb") as data:
        result = subprocess.run(
            ["ssh-keygen", "-Y", "verify", "-f", str(allowed_signers),
             "-I", identity, "-n", SIGNATURE_NAMESPACE,
             "-s", str(signature)],
            stdin=data, capture_output=True, text=True)
    if result.returncode == 0:
        return "good", f"signed by {identity}"
    return "bad", (result.stderr or result.stdout).strip()


def verify_release(path, quiet=False):
    path = Path(path).resolve()
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return False, "no manifest.json"
    recorded = (path / "manifest.sha256").read_text().strip()
    if sha256(manifest_path) != recorded:
        return False, "manifest.json does not match manifest.sha256"
    manifest = json.loads(manifest_path.read_text())
    binary = path / manifest["binary"]["name"]
    if not binary.exists():
        return False, f"missing {manifest['binary']['name']}"
    if sha256(binary) != manifest["binary"]["sha256"]:
        return False, "binary hash mismatch"
    if binary.stat().st_size != manifest["binary"]["bytes"]:
        return False, "binary size mismatch"
    q = manifest.get("qualification")
    if q:
        qp = path / "qualification.json"
        if not qp.exists() or sha256(qp) != q["sha256"]:
            return False, "qualification report missing or altered"
        if q.get("build_sha256") != manifest["binary"]["sha256"]:
            return False, ("qualification names a different binary than the "
                           "one in this release")
    expected = {"manifest.json", "manifest.sha256", manifest["binary"]["name"]}
    if q:
        expected.add("qualification.json")
    if (path / "manifest.sha256.sig").exists():
        expected.add("manifest.sha256.sig")
    actual = {p.name for p in path.iterdir()}
    if actual != expected:
        return False, f"unexpected contents: {sorted(actual ^ expected)}"
    if not quiet:
        print(f"verified {path.name}: binary, manifest and "
              f"{len(manifest['sources'])} pinned sources consistent")
    return True, manifest


def verify(a):
    ok, detail = verify_release(a.release)
    if not ok:
        print(f"FAILED: {detail}")
        return 1
    state, note = check_signature(a.release, a.allowed_signers)
    print(f"signature: {state} ({note})")
    if state == "bad":
        return 1
    if a.require_signature and state != "good":
        print("FAILED: a good signature was required")
        return 1
    return 0


def point(link, target):
    """Atomic: build the new link beside it, then rename over."""
    temporary = Path(str(link) + ".swap")
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    temporary.symlink_to(target)
    os.replace(temporary, link)


def activate(a):
    release = Path(a.release).resolve()
    ok, manifest = verify_release(release, quiet=True)
    if not ok:
        print(f"refusing to activate: {manifest}")
        return 1
    if manifest["git"]["dirty"]:
        print("refusing to activate: built from a dirty working tree, so it "
              "cannot be reproduced from history")
        return 1
    if not manifest.get("qualification"):
        print("refusing to activate: no qualification report; package with "
              "--qualification RESULTS.json")
        return 1
    state, note = check_signature(release, a.allowed_signers)
    if state == "bad":
        print(f"refusing to activate: signature check failed: {note}")
        return 1
    if a.require_signature and state != "good":
        print(f"refusing to activate: {note}")
        return 1
    root = Path(a.root).resolve()
    current = root / "current"
    if current.is_symlink():
        existing = Path(os.readlink(current))
        if existing.name == release.name:
            print(f"{release.name} is already current")
            return 0
        point(root / "previous", existing)
    point(current, Path("releases") / release.name)
    q = manifest["qualification"]
    print(f"activated {release.name}")
    print(f"  {manifest['version']}")
    print(f"  qualified {q['label']} {q['checks_passed']}/{q['checks_total']}")
    print(f"  serve with {current}/mimo26_server")
    return 0


def rollback(a):
    root = Path(a.root).resolve()
    current, previous = root / "current", root / "previous"
    if not previous.is_symlink():
        print("nothing to roll back to: no previous release recorded")
        return 1
    back = Path(os.readlink(previous))
    ok, manifest = verify_release(root / back, quiet=True)
    if not ok:
        print(f"refusing to roll back: previous release fails verification: "
              f"{manifest}")
        return 1
    was = Path(os.readlink(current)) if current.is_symlink() else None
    point(current, back)
    if was is not None:
        point(previous, was)
    print(f"rolled back to {back.name}")
    if was is not None:
        print(f"  {was.name} is now previous")
    return 0


def retire(a):
    """Sealing a release read-only also makes its directory unremovable, which
    is the point -- but it means retiring one needs a deliberate step rather
    than rm -rf. Refuses to remove whatever is live or is the only way back.

    Note what this does NOT protect: `previous` may point at a rebuild of the
    same commit as `current`, in which case rolling back changes nothing and
    retiring everything else leaves no real way out. Retiring in bulk once
    left exactly that state. Check `list` for the commit column before
    clearing house.
    """
    root = Path(a.root).resolve()
    release = Path(a.release).resolve()
    for link in ("current", "previous"):
        pointer = root / link
        if pointer.is_symlink() and \
                (root / Path(os.readlink(pointer))).resolve() == release:
            if link == "current":
                print(f"refusing to retire {release.name}: it is current")
                return 1
            if not a.force:
                print(f"refusing to retire {release.name}: it is the rollback "
                      f"target; pass --force to give that up")
                return 1
            pointer.unlink()
    unseal(release)
    shutil.rmtree(release)
    print(f"retired {release.name}")
    return 0


def list_releases(a):
    root = Path(a.root).resolve()
    releases = root / "releases"
    if not releases.exists():
        print(f"no releases under {root}")
        return 0
    live = Path(os.readlink(root / "current")).name \
        if (root / "current").is_symlink() else None
    prev = Path(os.readlink(root / "previous")).name \
        if (root / "previous").is_symlink() else None
    for item in sorted(releases.iterdir()):
        ok, manifest = verify_release(item, quiet=True)
        marker = "current " if item.name == live else \
                 "previous" if item.name == prev else "        "
        if ok:
            q = manifest.get("qualification")
            detail = (f"{manifest['git']['short']}  "
                      + (f"{q['checks_passed']}/{q['checks_total']} {q['label']}"
                         if q else "unqualified"))
            if manifest["git"]["dirty"]:
                detail += " DIRTY"
        else:
            detail = f"INVALID: {manifest}"
        print(f"  {marker}  {item.name}  {detail}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build"); b.set_defaults(fn=build)
    b.add_argument("--root", required=True)
    b.add_argument("--qualification")
    b.add_argument("--model")
    b.add_argument("--note")
    v = sub.add_parser("verify"); v.set_defaults(fn=verify)
    v.add_argument("release")
    v.add_argument("--allowed-signers")
    v.add_argument("--require-signature", action="store_true")
    ac = sub.add_parser("activate"); ac.set_defaults(fn=activate)
    ac.add_argument("release"); ac.add_argument("--root", required=True)
    ac.add_argument("--allowed-signers")
    ac.add_argument("--require-signature", action="store_true")
    r = sub.add_parser("rollback"); r.set_defaults(fn=rollback)
    r.add_argument("--root", required=True)
    l = sub.add_parser("list"); l.set_defaults(fn=list_releases)
    l.add_argument("--root", required=True)
    sg = sub.add_parser("sign"); sg.set_defaults(fn=sign)
    sg.add_argument("release"); sg.add_argument("--key", required=True)
    sg.add_argument("--force", action="store_true")
    rt = sub.add_parser("retire"); rt.set_defaults(fn=retire)
    rt.add_argument("release"); rt.add_argument("--root", required=True)
    rt.add_argument("--force", action="store_true")
    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

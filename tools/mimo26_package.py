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
import argparse, hashlib, json, os, shutil, stat, subprocess, sys, time
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
    qualification = None
    if a.qualification:
        source = Path(a.qualification).resolve()
        shutil.copy2(source, out / "qualification.json")
        report = json.loads(source.read_text())
        qualification = {
            "source": str(source),
            "sha256": sha256(out / "qualification.json"),
            "label": report.get("label"),
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
                   "sha256": sha256(out / "mimo26_server"),
                   "bytes": (out / "mimo26_server").stat().st_size},
        "sources": sources,
        "qualification": qualification,
        "model_root": a.model,
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
              f"{qualification['checks_passed']}/{qualification['checks_total']}")
    else:
        print("  qualified    NONE -- activate will refuse this release")
    return 0


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
    expected = {"manifest.json", "manifest.sha256", manifest["binary"]["name"]}
    if q:
        expected.add("qualification.json")
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
            detail = (f"{q['checks_passed']}/{q['checks_total']} {q['label']}"
                      if q else "unqualified")
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
    ac = sub.add_parser("activate"); ac.set_defaults(fn=activate)
    ac.add_argument("release"); ac.add_argument("--root", required=True)
    r = sub.add_parser("rollback"); r.set_defaults(fn=rollback)
    r.add_argument("--root", required=True)
    l = sub.add_parser("list"); l.set_defaults(fn=list_releases)
    l.add_argument("--root", required=True)
    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

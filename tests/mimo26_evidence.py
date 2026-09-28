"""Locate evidence from earlier runs, which lives outside this repository.

Several gates in `tests/run_mimo26_*.py` compare against artifacts produced by
previous qualification runs -- prior reports, pinned worker builds, recorded
routes. Those live in the maintainer's notes tree, not here.

The root used to be a hardcoded home directory in eight separate files, which
made every one of them unrunnable for anybody else and published the layout of a
private tree. Name it once, in the environment, and fail with an instruction
rather than a `FileNotFoundError` three frames deep.

Python puts a script's own directory on `sys.path`, so `python3 tests/run_...py`
imports this without any path juggling.
"""
import os
from pathlib import Path

VARIABLE = 'MOONSHINE_EVIDENCE_ROOT'


def evidence_root():
    """The directory holding evidence from earlier runs, or a clear refusal."""
    value = os.environ.get(VARIABLE)
    if not value:
        raise SystemExit(
            f'{VARIABLE} is not set.\n'
            'These gates read artifacts from earlier qualification runs, which\n'
            'are kept outside the repository. Point the variable at that tree:\n'
            f'  export {VARIABLE}=/path/to/notes/Projects/Moonshine/Evidence\n'
            'Each gate names the run directory it needs in its own source.')
    root = Path(value).expanduser()
    if not root.is_dir():
        raise SystemExit(f'{VARIABLE} is not a directory: {root}')
    return root

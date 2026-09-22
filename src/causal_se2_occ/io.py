from __future__ import annotations

from pathlib import Path
import tempfile


def prepare_output_file(path) -> Path:
    """Create the output parent and fail early if the directory is not writable."""
    out = Path(path)
    parent = out.parent if str(out.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".causal_se2_occ_writecheck_",
            dir=parent,
            delete=True,
        ):
            pass
    except OSError as exc:
        raise OSError(f"output directory is not writable: {parent}") from exc
    return out

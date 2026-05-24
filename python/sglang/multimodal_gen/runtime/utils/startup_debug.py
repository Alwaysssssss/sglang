from __future__ import annotations

import os
import time
from pathlib import Path


def write_startup_debug_event(message: str) -> None:
    """Append a startup breadcrumb when debug logging is explicitly enabled.

    Set ``SGLANG_DIFFUSION_STARTUP_DEBUG_DIR`` to a writable directory to enable
    rank-local startup tracing during distributed bring-up.
    """

    debug_dir = os.environ.get("SGLANG_DIFFUSION_STARTUP_DEBUG_DIR")
    if not debug_dir:
        return

    rank = os.environ.get("RANK", "na")
    pid = os.getpid()
    path = Path(debug_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    debug_file = path / f"rank{rank}_pid{pid}.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with debug_file.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

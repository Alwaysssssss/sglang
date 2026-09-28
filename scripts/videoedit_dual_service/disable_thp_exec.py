#!/usr/bin/env python3
"""Exec a command with transparent huge pages disabled for it and its descendants.

Wrapper used by start.sh when VIDEOEDIT_DISABLE_THP=true, for hosts that stall in
huge-page compaction (see docs_handoff/videoedit_service_quickstart.v2.l40.md
section 0.1). It sets PR_SET_THP_DISABLE on itself, then replaces itself with the
given command, so:

  - the flag reaches every child (Python workers, FFmpeg): PR_SET_THP_DISABLE is
    preserved across fork and execve;
  - the exec'd process keeps this PID, so the pid files written by start.sh still
    point at the real backend and stop.sh's cmdline check still matches;
  - only this process tree is affected. No /sys setting and no host-wide state is
    touched.

Usage: disable_thp_exec.py <command> [args...]
"""

import ctypes
import os
import sys

PR_SET_THP_DISABLE = 41


def disable_thp() -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def main(argv):
    if len(argv) < 2:
        print(f"usage: {argv[0]} <command> [args...]", file=sys.stderr)
        return 2
    try:
        disable_thp()
    except OSError as exc:
        # Disabling THP is a mitigation, not a correctness requirement: report it
        # loudly and still start the backend rather than block the service.
        print(
            f"warning: PR_SET_THP_DISABLE failed ({exc}); "
            "continuing with transparent huge pages enabled",
            file=sys.stderr,
        )
    os.execvp(argv[1], argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))

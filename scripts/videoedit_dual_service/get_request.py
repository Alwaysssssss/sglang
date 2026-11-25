#!/usr/bin/env python3
"""Export a task's stored, normalized backend request from the queue database.

By default, read through the videoedit_l40s container. Output includes stored
credentials unless --redact is supplied. This is not the original HTTP body.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_DB = str(
    Path(__file__).resolve().parents[2]
    / ".local/videoedit-l40s/dual/queue.sqlite3"
)

# Run only this small reader in the container; no dependency on installed sglang.
READER = """
import json, pathlib, sqlite3, sys
try:
    uri = pathlib.Path(sys.argv[1]).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True, timeout=10) as conn:
        row = conn.execute(
            'SELECT request_json FROM tasks WHERE task_id = ?',
            (sys.argv[2],),
        ).fetchone()
    if row is None:
        raise ValueError('Task not found: ' + sys.argv[2])
    print(json.dumps(json.loads(row[0]), ensure_ascii=False))
except (sqlite3.Error, ValueError, TypeError) as error:
    print(str(error), file=sys.stderr)
    sys.exit(1)
"""


def redact(value):
    sensitive = {
        "accesskey", "awsaccesskeyid", "secretkey", "secretaccesskey",
        "awssecretaccesskey", "password", "rootpass", "rootuser",
        "token", "sessiontoken", "securitytoken", "authorization",
    }
    if isinstance(value, dict):
        return {
            key: "***" if "".join(c for c in key.lower() if c.isalnum())
            in sensitive else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", help="Task ID to retrieve")
    parser.add_argument("--db", default=DEFAULT_DB, help="Queue database path")
    parser.add_argument("--container", default="videoedit_l40s")
    parser.add_argument("--local", action="store_true", help="Read without Docker")
    parser.add_argument("--redact", action="store_true", help="Mask credential fields")
    parser.add_argument("-o", "--output", help="Write a new JSON file (mode 0600)")
    args = parser.parse_args()
    command = [sys.executable, "-c", READER, args.db, args.task_id]
    if not args.local:
        command = ["docker", "exec", args.container, "python", "-c",
                   READER, args.db, args.task_id]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            print(result.stderr.strip() or "Failed to read request", file=sys.stderr)
            return 1
        data = json.loads(result.stdout)
        if args.redact:
            data = redact(data)
        output = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(output)
            print(f"Saved to {args.output}", file=sys.stderr)
        else:
            sys.stdout.write(output)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

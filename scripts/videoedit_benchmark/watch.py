#!/usr/bin/env python3
"""Low-frequency local monitoring; no model calls and no inference requests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from status import status


def check(root, runner_pid):
    value = status(root)
    (root / 'run-status.json').write_text(json.dumps(value, indent=2) + '\n')
    terminal = value['completed'] + value['failed'] == value['total']
    try:
        command = Path(f'/proc/{runner_pid}/cmdline').read_bytes().replace(b'\x00', b' ')
        alive = any(script in command for script in (
            b'videoedit_benchmark/run.py', b'videoedit_benchmark/handoff.py'))
    except FileNotFoundError:
        alive = False
    record = {k: v for k, v in value.items() if k != 'configurations'}
    record['runner_alive'] = alive
    with (root / 'watch.jsonl').open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if terminal and not alive:
        subprocess.run([sys.executable, str(Path(__file__).with_name('finalize.py')), str(root)], check=True)
        health = subprocess.run(['curl', '--noproxy', '*', '-sS', '--max-time', '10',
                                 'http://127.0.0.1:5402/health'], capture_output=True, text=True)
        (root / 'business-health-after.json').write_text(json.dumps({
            'checked_at': time.time(), 'returncode': health.returncode,
            'response': health.stdout, 'stderr': health.stderr}, indent=2) + '\n')
        (root / 'matrix-finished.json').write_text(json.dumps({
            'completed_at': time.time(), 'completed': value['completed'],
            'failed': value['failed'], 'total': value['total']}, indent=2) + '\n')
        return True
    if not alive and not terminal:
        (root / 'attention-required.json').write_text(json.dumps({
            'checked_at': time.time(), 'reason': 'Runner exited before all configurations became terminal',
            'status': value}, indent=2) + '\n')
        return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--runner-pid', required=True, type=int)
    parser.add_argument('--interval', type=int, default=900)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.relative_to(Path(__file__).resolve().parents[2])
    if args.interval < 60:
        parser.error('Monitor interval must be at least 60 seconds')
    while True:
        try:
            finished = check(args.root, args.runner_pid)
        except Exception as exc:
            with (args.root / 'watch-errors.jsonl').open('a') as handle:
                handle.write(json.dumps({'time': time.time(), 'error': repr(exc)}) + '\n')
            finished = False
        if finished or args.once:
            break
        time.sleep(args.interval)

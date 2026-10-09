#!/usr/bin/env python3
"""Finish an in-flight container, then replace a paused dispatcher safely.

Only the dispatcher is paused. The current Docker worker finishes every request
normally. Its exit/cleanup records are preserved before the reduced queue starts.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run import ROOT, gpu_state, write


def handoff(args):
    root = args.root.resolve()
    root.relative_to(ROOT)
    target = root / args.current
    variant = json.loads((target / 'variant.json').read_text())
    command = json.loads((target / 'docker-command.json').read_text())
    name = command[command.index('--name') + 1]
    old_cmd = Path(f'/proc/{args.old_pid}/cmdline').read_bytes()
    if b'videoedit_benchmark/run.py' not in old_cmd or str(root.relative_to(ROOT)).encode() not in old_cmd:
        raise RuntimeError('Old PID does not match expected benchmark dispatcher')
    state = Path(f'/proc/{args.old_pid}/status').read_text()
    if 'State:\tT' not in state:
        raise RuntimeError('Old dispatcher must be paused before handoff')
    print(json.dumps({'state': 'waiting_for_current_container', 'container': name}), flush=True)
    try:
        result = subprocess.run(['docker', 'wait', name], capture_output=True, text=True,
                                check=True, timeout=1900 + 4 * (3600 + 240))
        code = int(result.stdout.strip())
        write(target / 'exit.json', {'returncode': code, 'dispatcher_handoff': True})
    except subprocess.TimeoutExpired:
        write(target / 'exit.json', {'error': 'configuration timeout during dispatcher handoff'})
    inspection = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, check=True)
    (target / 'container.json').write_text(inspection.stdout)
    subprocess.run(['docker', 'rm', '-f', name], check=True)
    for attempt in range(30):
        try:
            after = gpu_state(variant['host_gpus'])
            break
        except RuntimeError:
            if attempt == 29:
                raise
            time.sleep(2)
    write(target / 'gpu-after.json', after)
    subprocess.run([sys.executable, str(Path(__file__).with_name('report.py')), str(root)], check=True)
    # Current GPU work and Docker children have exited. Retire only the paused
    # dispatcher PID, releasing its matrix lock; do not signal its process group.
    os.kill(args.old_pid, signal.SIGKILL)
    for attempt in range(50):
        if not Path(f'/proc/{args.old_pid}').exists():
            break
        time.sleep(0.1)
    selection = json.loads((root / 'selection.json').read_text())
    write(root / 'handoff-completed.json', {'time': time.time(), 'current': args.current,
                                         'old_pid': args.old_pid, 'runner_pid': os.getpid()})
    cmd = [sys.executable, str(Path(__file__).with_name('run.py')), '--out', str(root),
           '--resume', '--variants', ','.join(selection['keep'])]
    print(json.dumps({'state': 'starting_reduced_queue', 'command': cmd}), flush=True)
    os.execv(sys.executable, cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--old-pid', type=int, required=True)
    parser.add_argument('--current', required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        handoff(args)
    except BaseException as exc:
        write(args.root / 'handoff-error.json', {'time': time.time(), 'error': repr(exc)})
        raise

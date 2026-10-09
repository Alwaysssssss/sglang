#!/usr/bin/env python3
"""Read-only live matrix status, including private progress files via Docker."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def status(root):
    matrix = json.loads((Path(__file__).with_name('teacache_sp_matrix.json')).read_text())
    selection = root / 'selection.json'
    selected = set(json.loads(selection.read_text())['keep']) if selection.exists() else None
    rows = []
    for variant in matrix['variants']:
        if selected is not None and variant['id'] not in selected:
            continue
        target = root / variant['id']
        row = {'id': variant['id'], 'status': 'pending'}
        if (target / 'exit.json').exists():
            result = json.loads((target / 'exit.json').read_text())
            row['status'] = 'completed' if result.get('returncode') == 0 and (target / 'completion.json').exists() else 'failed'
        elif target.exists():
            row['status'] = 'running'
            log = target / 'runner.log'
            if log.exists():
                events = []
                for line in log.read_text().splitlines():
                    if line.startswith('{'):
                        try:
                            events.append(json.loads(line))
                        except ValueError:
                            pass
                if events:
                    row['last_event'] = events[-1]
            cmdfile = target / 'docker-command.json'
            if cmdfile.exists():
                cmd = json.loads(cmdfile.read_text())
                name = cmd[cmd.index('--name') + 1]
                code = "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.dumps({x.name:json.loads(x.read_text()) for x in p.glob('*.progress.json')}))"
                result = subprocess.run(['docker', 'exec', name, 'python3', '-c', code, str(target / 'outputs')],
                                        text=True, capture_output=True, timeout=20)
                if result.returncode == 0:
                    row['progress'] = json.loads(result.stdout)
        rows.append(row)
    return {'updated_at': time.time(), 'completed': sum(r['status'] == 'completed' for r in rows),
            'failed': sum(r['status'] == 'failed' for r in rows), 'total': len(rows),
            'active': [r for r in rows if r['status'] == 'running'], 'configurations': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    value = status(args.root)
    if args.write:
        (args.root / 'run-status.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'configurations'}, ensure_ascii=False))

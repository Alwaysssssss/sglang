#!/usr/bin/env python3
"""Validate matrix completion and write a concise, source-linked result table.

Run after the serial runner exits. Different decoded hashes require a separate
pixel comparison; this report never silently treats them as equivalent quality.
"""
import argparse
import json
from pathlib import Path
import subprocess

from report import main as update_summary


def finalize(root):
    matrix = json.loads(Path(__file__).with_name('teacache_sp_matrix.json').read_text())
    selection = root / 'selection.json'
    if selection.exists():
        selected = set(json.loads(selection.read_text())['keep'])
        matrix['variants'] = [v for v in matrix['variants'] if v['id'] in selected]
    update_summary(root)
    rows = json.loads((root / 'summary.json').read_text())
    ids = {v['id'] for v in matrix['variants']}
    rows = [r for r in rows if r['id'] in ids]
    by_id = {row['id']: row for row in rows}
    pending = [v['id'] for v in matrix['variants']
               if v['id'] not in by_id or by_id[v['id']]['status'] not in ('completed', 'failed')
               or not (root / v['id'] / 'gpu-after.json').exists()]
    if pending:
        raise RuntimeError('Matrix still has unfinished configurations: ' + ', '.join(pending))
    succeeded = [r for r in rows if r['status'] == 'completed']
    failed = [r for r in rows if r['status'] == 'failed']
    lines = ['# TeaCache SP full-frame matrix', '',
             f"Completed measurements: {len(succeeded)}; failed configurations: {len(failed)}; total: {len(rows)}.", '',
             'Source 1920×1080; model windows 49 frames at 1920×1088; 40 steps, seed 42, CFG 5 with dynamic CFG, TeaCache threshold 0.3. One warmup plus three complete requests per successful configuration.', '',
             'Timings include model transfers, video processing and instrumentation. Model initialization is separate. Failures retain original resolution, window policy and cache settings.', '',
             '| Configuration | GPUs | E2E runs (s) | E2E median (s) | Denoise median (s) | Speedup | GPU seconds | CPU memory peak GiB | Decoded output |',
             '|---|---:|---|---:|---:|---:|---:|---:|---|']
    for v in matrix['variants']:
        r = by_id[v['id']]
        if r['status'] == 'failed':
            reason = str(r.get('error', {})).replace('|', '/')[:350]
            lines.append(f"| {r['id']} | {r['num_gpus']} | FAILED: {reason} | — | — | — | — | — | — |")
            continue
        times = ' / '.join(f"{run['e2e_seconds']:.3f}" for run in r['runs'])
        quality = 'hash identical to baseline' if r.get('decoded_matches_baseline') else 'different; compare pixels'
        lines.append(f"| {r['id']} | {r['num_gpus']} | {times} | {r['e2e_median_s']:.3f} | {r['denoise_median_s']:.3f} | {r.get('e2e_speedup', 0):.3f} | {r['gpu_seconds']:.3f} | {r['cpu_memory_peak_gib']:.2f} | {quality} |")
    if succeeded:
        latency = min(succeeded, key=lambda r: r['e2e_median_s'])
        cost = min(succeeded, key=lambda r: r['gpu_seconds'])
        lines.extend(['', f"Observed lowest latency: **{latency['id']}**, {latency['e2e_median_s']:.3f} s.",
                      f"Observed lowest GPU-time cost: **{cost['id']}**, {cost['gpu_seconds']:.3f} GPU s.", '',
                      'These are measured candidates, not deployment changes. Small differences and overlapping ranges require interleaved repeats; differing TeaCache decisions confound pure SP speedup. Four consecutive requests demonstrate only this workload, not prolonged service reliability.', '',
                      'Per-rank stage peaks, latent hashes, cache decisions, frame hashes and min/max timings are in [summary.json](summary.json). Raw logs and telemetry remain in each configuration directory.'])
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'succeeded': len(succeeded), 'failed': len(failed), 'report': str(root / 'report.md')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    finalize(parser.parse_args().root)

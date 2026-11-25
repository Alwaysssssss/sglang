#!/usr/bin/env python3
"""Validate mixed-size residency artifacts and report each full-frame request."""
import argparse
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def collect(target):
    variant = read(target / 'variant.json')
    assert read(target / 'completion.json')['status'] == 'completed'
    assert read(target / 'exit.json')['returncode'] == 0
    events = [json.loads(line) for path in sorted(target.glob('rank*.jsonl'))
              for line in path.read_text().splitlines()]
    sequence = variant.get('requests') or [dict(phase=p, geometry='crop')
                                          for p in ('warmup', 'timed1', 'timed2', 'timed3')]
    results = []
    for item in sequence:
        phase, geometry = item['phase'], item['geometry']
        result = read(target / f'{phase}-result.json')
        assert result['status'] == 'completed'
        rows = [e for e in events if Path(e.get('output') or '').name == f'{variant["id"]}-{phase}.mov']
        requests = [e for e in rows if e['kind'] == 'request']
        denoise = [e for e in rows if e.get('stage') == 'VideoEditDenoisingStage']
        latents = [e for e in rows if e['kind'] == 'latent']
        for records in (requests, denoise, latents):
            assert len(records) == variant['num_gpus']
            assert {e['rank'] for e in records} == set(range(variant['num_gpus']))
        assert all(e['status'] == 'ok' for e in requests + denoise)
        assert len({e['sha256'] for e in latents}) == 1
        assert all(e['gpu_layers_after_stage'] == [0] for e in denoise)
        expected = (1920, 1088, 49) if geometry == 'full' else (544, 480, 49)
        assert all((e['model_width'], e['model_height'], e['model_frames']) == expected for e in denoise)
        assert all(e['decisions'] == denoise[0]['decisions'] for e in denoise)
        if variant.get('expect_adaptive'):
            plans = [e['residency_plan'] for e in denoise]
            assert len({p['common_weight_budget_bytes'] for p in plans}) == 1
            assert all(p['resident_layers'] == plans[0]['resident_layers'] for p in plans)
            assert all(p['prefetch_layers'] == plans[0]['prefetch_layers'] for p in plans)
            assert all(p['weight_capacity_bytes'] <= p['common_weight_budget_bytes'] for p in plans)
            assert all(p['resident_layers'] == [40] if geometry == 'crop'
                       else all(0 <= n < 40 for n in p['resident_layers']) for p in plans)
        samples = [json.loads(line) for line in (target / f'{phase}-samples.jsonl').read_text().splitlines()]
        gpu_peaks = {}
        for sample in samples:
            for gpu in sample['gpus']:
                fields = [v.strip() for v in gpu.split(',')]
                gpu_peaks[fields[0]] = max(gpu_peaks.get(fields[0], 0), float(fields[1]) / 1024)
        results.append(dict(configuration=variant['id'], phase=phase, geometry=geometry,
                            e2e_seconds=max(e['seconds'] for e in requests),
                            denoise_seconds=max(e['seconds'] for e in denoise),
                            client_seconds=result['client_seconds'],
                            latent_sha256=latents[0]['sha256'],
                            decoded_sha256=result['validation']['decoded_sha256'],
                            decisions=denoise[0]['decisions'],
                            resident_layers=denoise[0]['effective_resident_layers'],
                            prefetch_layers=denoise[0]['effective_prefetch_layers'],
                            plans=[e.get('residency_plan') for e in denoise],
                            peak_allocated_gib={e['rank']: e['peak_allocated_bytes'] / 2**30 for e in denoise},
                            peak_reserved_gib={e['rank']: e['peak_reserved_bytes'] / 2**30 for e in denoise},
                            sampled_gpu_peak_gib=gpu_peaks))
    return results


def main(root, reference):
    rows = collect(root / 'fixed') + collect(root / 'adaptive')
    for geometry in ('crop', 'full'):
        group = [r for r in rows if r['geometry'] == geometry]
        for field in ('latent_sha256', 'decoded_sha256'):
            assert len({r[field] for r in group}) == 1, (geometry, field)
        assert all(r['decisions'] == group[0]['decisions'] for r in group)
    historical = None
    if reference:
        historical = read(reference)['validation']['decoded_sha256']
        assert all(r['decoded_sha256'] == historical for r in rows if r['geometry'] == 'full')
    medians = {}
    for label in ('fixed', 'adaptive'):
        group = [r for r in rows if r['configuration'] == label and r['phase'].startswith('timed')]
        assert len(group) == 3
        medians[label] = {field: statistics.median(r[field] for r in group)
                          for field in ('e2e_seconds', 'denoise_seconds')}
    report = dict(status='passed', requests=rows, small_medians=medians,
                  historical_full_decoded_sha256=historical)
    (root / 'mixed-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Adaptive residency validation', '',
             'All requests completed; per-size decoded and latent hashes and TeaCache decisions agree.', '',
             '| Configuration | Request | Geometry | E2E (s) | DiT (s) | Resident / prefetch | GPU sampled peak (GiB) |',
             '|---|---|---|---:|---:|---|---|']
    for r in rows:
        peak = ' / '.join(f'{v:.3f}' for v in r['sampled_gpu_peak_gib'].values())
        lines.append(f"| {r['configuration']} | {r['phase']} | {r['geometry']} | {r['e2e_seconds']:.3f} | {r['denoise_seconds']:.3f} | {r['resident_layers']} / {r['prefetch_layers']} | {peak} |")
    lines += ['', 'Small-input medians (three timed requests per configuration):', '',
              '```json', json.dumps(medians, indent=2), '```', '',
              'Full1 is the first full-frame request in this process; full2 is a repeat, with a small request between them.',
              'GPU samples are collected once per second and may miss transient peaks. Allocator peaks are in mixed-summary.json.',
              'Both ranks agree on budgets; every stage released its weight cache; every small request selected all 40 layers.']
    if reference:
        lines += [f'Full-frame decoded hashes also match historical reference: `{reference}`.']
    (root / 'mixed-summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(medians, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--full-reference', type=Path)
    args = parser.parse_args()
    main(args.root, args.full_reference)

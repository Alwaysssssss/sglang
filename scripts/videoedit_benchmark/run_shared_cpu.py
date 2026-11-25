#!/usr/bin/env python3
"""Compare private pinned vs shared pinned CPU masters on two idle GPUs.

Each isolated container runs one warmup and three small-crop requests. Only the
shared-master policy differs. Outputs, caches and the tmpfs mount live below the
repository. The shared tmpfs is destroyed with its test container, including on
failure. Existing services are not changed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from run import ROOT, gpu_state, write


def run(args):
    out = Path(args.out).resolve()
    out.relative_to(ROOT)
    gpus = [int(value) for value in args.gpus.split(',')]
    if len(gpus) != 2 or len(set(gpus)) != 2 or not set(gpus) <= {4, 5, 6}:
        raise ValueError('Select two distinct authorized GPUs from 4,5,6')
    out.mkdir(parents=True, exist_ok=False)
    with (ROOT / '.local/teacache-sp.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        image = subprocess.check_output(['docker', 'image', 'inspect', args.image,
                                        '--format', '{{.Id}}'], text=True).strip()
        write(out / 'manifest.json', dict(
            image=args.image, image_id=image, gpus=gpus, geometry='crop',
            started_at=time.time(),
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [ROOT / 'python/sglang/multimodal_gen/runtime/utils/shared_cpu_weights.py',
                                     ROOT / 'python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py',
                                     Path(__file__).resolve()]}))
        for label in ('private', 'shared'):
            target = out / label
            target.mkdir()
            write(target / 'gpu-before.json', gpu_state(gpus))
            for directory in ('home', 'tmp', 'cache', 'shared-masters'):
                (target / directory).mkdir()
            environment = dict(
                SGLANG_VIDEOEDIT_RESIDENT_LAYERS='40',
                SGLANG_VIDEOEDIT_RELEASE_AFTER_DENOISE='1',
                SGLANG_VIDEOEDIT_ENCODER_CPU_MASTER='1',
                SGLANG_VIDEOEDIT_OVERLAP_PREFETCH='0',
                SGLANG_VIDEOEDIT_FUSED_QKV='0', SGLANG_VIDEOEDIT_USP_CHUNKS='1',
                SGLANG_VIDEOEDIT_ASYNC_STAGE_LOAD='0',
                SGLANG_VIDEOEDIT_PREFETCH_IMAGE_ENCODER='0',
                SGLANG_VIDEOEDIT_OFFLOAD_PINNED_MB='0',
                SGLANG_VIDEOEDIT_AUTO_RESIDENCY='0',
                SGLANG_VIDEOEDIT_SHARED_CPU_DIR=str(target / 'shared-masters') if label == 'shared' else '')
            variant = dict(id=label, host_gpus=gpus, num_gpus=2, sp_degree=2,
                           ulysses_degree=2, ring_degree=1, dit_layerwise_offload=True,
                           dit_offload_prefetch_size=4, stage_resident=False, environment=environment)
            write(target / 'variant.json', variant)
            name = 've-shared-cpu-' + uuid.uuid4().hex[:12]
            command = ['docker', 'run', '--user', 'root', '--name', name,
                       '--network', 'none', '--shm-size', '16g',
                       '--gpus', '"device=' + ','.join(map(str, gpus)) + '"',
                       '-v', f'{ROOT}:{ROOT}', '-v', f'{args.model}:{args.model}:ro', '-w', str(ROOT)]
            if label == 'shared':
                command += ['--tmpfs', f'{target}/shared-masters:rw,size=40g,mode=0700']
            environment.update(
                HOME=str(target / 'home'), TMPDIR=str(target / 'tmp'),
                XDG_CACHE_HOME=str(target / 'cache'), HF_HOME=str(target / 'cache/hf'),
                TRITON_CACHE_DIR=str(target / 'cache/triton'),
                FLASHINFER_WORKSPACE_BASE=str(target / 'cache/flashinfer'),
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1',
                VIDEOEDIT_BENCH_UID=str(os.getuid()), VIDEOEDIT_BENCH_GID=str(os.getgid()))
            for key, value in environment.items():
                command += ['-e', f'{key}={value}']
            command += [image, 'python3', str(Path(__file__).with_name('worker.py').resolve()),
                        '--out', str(target), '--model', args.model, '--geometry', 'crop',
                        '--request-timeout', '1800']
            write(target / 'docker-command.json', command)
            print(f'START {label}: {name}', flush=True)
            with (target / 'runner.log').open('w') as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=10000)
                    write(target / 'exit.json', {'returncode': process.returncode})
                except BaseException as exc:
                    write(target / 'exit.json', {'error': str(exc)})
                    raise
                finally:
                    inspection = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
                    (target / 'container.json').write_text(inspection.stdout)
                    subprocess.run(['docker', 'rm', '-f', name], stdout=log, stderr=log, check=True)
            for attempt in range(30):
                try:
                    state = gpu_state(gpus)
                    break
                except RuntimeError:
                    if attempt == 29:
                        raise
                    time.sleep(2)
            write(target / 'gpu-after.json', state)
            subprocess.run(['python3', str(Path(__file__).with_name('report.py')), str(out)], check=True)
            print(f'END {label}: {process.returncode}', flush=True)
            if process.returncode:
                raise RuntimeError(f'{label} failed; stopping without retry')
        write(out / 'completion.json', {'status': 'completed', 'finished_at': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpus', default='4,5')
    parser.add_argument('--image', default='sglang-videoedit-dev-v2:l40s')
    parser.add_argument('--model', default='/home/zhouhao6/VideoEdit/model/DifusserEdit/pretrain_models/VideoEdit-diffusers-model')
    run(parser.parse_args())

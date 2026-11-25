# SPDX-License-Identifier: Apache-2.0
"""Process-shared, content-addressed CPU masters for read-only inference.

Each tensor owns an mmap through its buffer owner (including derived views).
CUDA registration does not copy its pages. Writers must detach before mutation;
this module is intentionally not a general mutable shared-tensor store.
Use a private, local, per-service directory shared by cooperating ranks.
For CUDA-pinned sharing the directory must be on tmpfs: ordinary disk-backed
MAP_SHARED pages are not CUDA-registerable on the validated Linux driver.
"""

import ctypes
import fcntl
import hashlib
import logging
import mmap
import os
from pathlib import Path
import tempfile
import weakref

import torch

logger = logging.getLogger(__name__)
# Keep mappings alive if CUDA cannot confirm that DMA has stopped at teardown.
_quarantined_mappings = []


def _cuda_check(result, operation):
    error = result[0] if isinstance(result, tuple) else result
    if int(error) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {int(error)} ({error})")


def _release(mapping, fd, path, gate_path, registration):
    if registration is not None:
        runtime, pointer, device = registration
        try:
            # External pinned memory is not managed by PyTorch's host allocator.
            # A last tensor/view may disappear while an async copy is in flight.
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                _cuda_check(runtime.cudaHostUnregister(pointer), "cudaHostUnregister")
        except Exception:
            logger.exception("Keeping shared CPU mapping alive after CUDA teardown failure")
            _quarantined_mappings.append((mapping, fd, path, gate_path, registration))
            return
    try:
        with open(gate_path, "a+b") as gate:
            fcntl.flock(gate, fcntl.LOCK_EX)
            fcntl.flock(fd, fcntl.LOCK_UN)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass  # Other processes still own this file.
            else:
                try:
                    if os.stat(path).st_ino == os.fstat(fd).st_ino:
                        os.unlink(path)
                except FileNotFoundError:
                    pass
    finally:
        os.close(fd)
    # The buffer exporter may still be being destroyed here. Let mmap's own
    # refcount close it after all buffer exports vanish, instead of close().


def share_cpu_tensor(source, directory, *, pin_memory=False, device=None):
    """Intern a contiguous CPU tensor; callers must treat the result as read-only.

    Equal bytes/dtypes across ranks map the same inode. Different checkpoints
    cannot collide through shape-only keys. Registration failure is explicit;
    it never silently clones an entire private pinned copy or uses pageable DMA.
    Normal last-owner teardown removes the data file. A killed process can leave
    files behind: remove the session directory only after all ranks have exited.
    """
    if source.device.type != "cpu" or not source.is_contiguous() or source.numel() == 0:
        raise ValueError("Shared CPU weights require a nonempty contiguous CPU tensor")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = memoryview(source.detach().view(torch.uint8).numpy()).cast("B")
    digest = hashlib.sha256(str(source.dtype).encode())
    digest.update(raw)
    path = directory / (digest.hexdigest() + ".bin")
    gate_path = directory / ".lock"
    fd = None
    mapping = None
    registration = None
    try:
        with open(gate_path, "a+b") as gate:
            fcntl.flock(gate, fcntl.LOCK_EX)
            if not path.exists():
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=directory, prefix=".weights-", delete=False) as output:
                        temporary = output.name
                        output.write(raw)
                    os.replace(temporary, path)
                finally:
                    if temporary and os.path.exists(temporary):
                        os.unlink(temporary)
            fd = os.open(path, os.O_RDWR)
            if os.fstat(fd).st_size != raw.nbytes:
                raise RuntimeError(f"Shared CPU weight size mismatch: {path}")
            fcntl.flock(fd, fcntl.LOCK_SH)
            mapping = mmap.mmap(fd, raw.nbytes, flags=mmap.MAP_SHARED,
                                prot=mmap.PROT_READ | mmap.PROT_WRITE)
        if pin_memory:
            device = torch.device(device or "cuda")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            with torch.cuda.device(device):
                runtime = torch.cuda.cudart()
                pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                _cuda_check(runtime.cudaHostRegister(pointer, raw.nbytes, 1),
                            "cudaHostRegister (pinned sharing requires a tmpfs directory)")
                registration = (runtime, pointer, device)
        owner = memoryview(mapping)
        # torch.frombuffer retains this owner for the storage lifetime, not just
        # the first Tensor object's lifetime. Views therefore keep registration.
        weakref.finalize(owner, _release, mapping, fd, path, gate_path, registration)
        fd = None  # Ownership transferred to the finalizer, even if frombuffer fails.
        result = torch.frombuffer(owner, dtype=source.dtype).view(source.shape)
        if pin_memory and not result.is_pinned():
            raise RuntimeError("CUDA-registered shared CPU weights are not recognized as pinned")
        return result
    except BaseException:
        if fd is not None:
            _release(mapping, fd, path, gate_path, registration)
        raise

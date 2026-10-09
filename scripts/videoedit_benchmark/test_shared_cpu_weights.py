"""Shared masters, ownership and copy-before-write; CUDA checks run when available."""
import gc
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE = "sglang.multimodal_gen.runtime.utils.shared_cpu_weights"
spec = importlib.util.spec_from_file_location("shared_weights_under_test", ROOT / "python/sglang/multimodal_gen/runtime/utils/shared_cpu_weights.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def child_mapping(directory, ready, finish, pin):
    value = shared.share_cpu_tensor(torch.arange(262144, dtype=torch.float32), directory,
                                    pin_memory=pin, device=("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0") if pin else None)
    # Exercise every page; the parent retains a simultaneous mapping.
    checksum = value.sum().item()
    data = next(Path(directory).glob("*.bin"))
    ready.put((os.stat(data).st_ino, checksum, value.is_pinned()))
    finish.wait(60)
    del value
    gc.collect()


class SharedWeightsTests(unittest.TestCase):
    def test_views_retain_mapping_until_last_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            source = torch.arange(16, dtype=torch.bfloat16)
            first = shared.share_cpu_tensor(source, directory)
            second = shared.share_cpu_tensor(source.clone(), directory)
            self.assertEqual(len(list(Path(directory).glob('*.bin'))), 1)
            view = first[2:5]
            del first, second
            gc.collect()
            torch.testing.assert_close(view, source[2:5])
            self.assertEqual(len(list(Path(directory).glob('*.bin'))), 1)
            del view
            gc.collect()
            self.assertEqual(list(Path(directory).glob('*.bin')), [])

    def test_content_identity_and_failed_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            a = shared.share_cpu_tensor(torch.ones(8), directory)
            b = shared.share_cpu_tensor(torch.zeros(8), directory)
            self.assertEqual(len(list(Path(directory).glob('*.bin'))), 2)
            with patch.object(shared.os, 'replace', side_effect=OSError('publish failed')):
                with self.assertRaisesRegex(OSError, 'publish failed'):
                    shared.share_cpu_tensor(torch.full((8,), 2.0), directory)
            self.assertEqual(list(Path(directory).glob('.weights-*')), [])
            del a, b
            gc.collect()
            self.assertEqual(list(Path(directory).glob('*.bin')), [])

    def test_processes_share_inode_and_survive_other_rank_exit(self):
        self.check_processes(False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_registered_pages_are_shared_across_processes(self):
        self.check_processes(True)

    def check_processes(self, pin):
        with tempfile.TemporaryDirectory() as directory:
            ctx = mp.get_context('spawn')
            ready, finish = ctx.Queue(), ctx.Event()
            value = shared.share_cpu_tensor(torch.arange(262144, dtype=torch.float32), directory,
                                            pin_memory=pin, device='cuda:0' if pin else None)
            process = ctx.Process(target=child_mapping, args=(directory, ready, finish, pin))
            process.start()
            try:
                inode, checksum, pinned = ready.get(timeout=90)
                self.assertEqual(inode, os.stat(next(Path(directory).glob('*.bin'))).st_ino)
                self.assertEqual(checksum, value.sum().item())
                self.assertEqual(pinned, pin)
                # Prove physical sharing with smaps, not by summing duplicate RSS.
                text = Path(f'/proc/{os.getpid()}/smaps').read_text()
                blocks = text.splitlines()
                sharing = 0
                active = False
                for line in blocks:
                    if '-' in line.split()[0]:
                        active = directory in line and '.bin' in line
                    elif active and line.startswith(('Shared_Clean:', 'Shared_Dirty:')):
                        sharing += int(line.split()[1])
                self.assertGreaterEqual(sharing, value.numel() * value.element_size() // 1024)
                finish.set()
                process.join(timeout=60)
                self.assertEqual(process.exitcode, 0)
                torch.testing.assert_close(value, torch.arange(262144, dtype=torch.float32))
            finally:
                finish.set()
                if process.is_alive():
                    process.terminate()
                    process.join()
            del value
            gc.collect()
            self.assertEqual(list(Path(directory).glob('*.bin')), [])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_async_copy_keeps_external_registration_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = torch.arange(262144, dtype=torch.float32)
            value = shared.share_cpu_tensor(expected, directory, pin_memory=True, device='cuda:0')
            self.assertTrue(value.is_pinned())
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                output = value.to('cuda', non_blocking=True)
            del value
            gc.collect()
            stream.synchronize()
            torch.testing.assert_close(output.cpu(), expected)
            self.assertEqual(list(Path(directory).glob('*.bin')), [])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_manager_updates_detach_without_changing_other_owner(self):
        from scripts.videoedit_benchmark.test_overlap_pipeline import Model, offload
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {MODULE: shared}), \
             patch.dict(os.environ, SGLANG_VIDEOEDIT_SHARED_CPU_DIR=directory,
                        SGLANG_VIDEOEDIT_OFFLOAD_PINNED_MB='0',
                        SGLANG_VIDEOEDIT_RESIDENT_LAYERS='0'):
            torch.manual_seed(17)
            first, second = Model(count=2).eval(), Model(count=2).eval()
            second.load_state_dict(first.state_dict())
            reference = Model(count=2).cuda().eval()
            reference.load_state_dict(first.state_dict())
            managers = [offload.LayerwiseOffloadManager(m, layers_attr_str='blocks', num_layers=2,
                         enabled=True, prefetch_size=1) for m in (first, second)]
            a, b = managers
            self.assertGreater(a.shared_cpu_weight_bytes, 0)
            x = torch.randn(8, 16, device='cuda')
            with torch.inference_mode():
                torch.testing.assert_close(first(x), reference(x), rtol=0, atol=0)
                torch.testing.assert_close(second(x), reference(x), rtol=0, atol=0)
                original = dict(b.iter_cpu_weights())['blocks.0.weight'].clone()
                replacement = torch.zeros_like(original)
                a.update_cpu_weights({'blocks.0.weight': replacement})
                reference.blocks[0].weight.copy_(replacement)
                torch.testing.assert_close(first(x), reference(x), rtol=0, atol=0)
                torch.testing.assert_close(dict(b.iter_cpu_weights())['blocks.0.weight'], original)
                self.assertLess(a.shared_cpu_weight_bytes, b.shared_cpu_weight_bytes)
                # The disable/enable-offload writeback path must also detach.
                a.load_all_layers()
                first.blocks[1].weight.add_(1)
                other = dict(b.iter_cpu_weights())['blocks.1.weight'].clone()
                a.sync_all_layers_to_cpu()
                torch.testing.assert_close(dict(b.iter_cpu_weights())['blocks.1.weight'], other)
                self.assertEqual(a.shared_cpu_weight_bytes, 0)
            for m in managers:
                m.release_all()
                m.remove_forward_hooks()
                m.copy_stream.synchronize()
                m._consolidated_cpu_weights.clear()
            gc.collect()
            self.assertEqual(list(Path(directory).glob('*.bin')), [])


if __name__ == '__main__':
    unittest.main()

"""CPU tests for experiment isolation, geometry, and aggregation invariants."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("runner", HERE / "run.py")
worker = load("worker", HERE / "worker.py")
hook = load("hook", HERE / "hooks/benchmark_hook.py")
report = load("report", HERE / "report.py")


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.matrix = json.loads((HERE / "teacache_sp_matrix.json").read_text())

    def test_matrix_has_all_configurations_in_experiment_order(self):
        rows = runner.ordered_variants(self.matrix)
        self.assertEqual(len(rows), 30)
        self.assertEqual([v["id"] for v in rows[:4]],
                         [f"single_prefetch{n}" for n in (1, 2, 4, 8)])
        self.assertTrue(all(v["dit_offload_prefetch_size"] == 1 for v in rows[4:9]))
        self.assertTrue(all(v.get("stage_resident") for v in rows[-6:]))

    def test_production_gpu_and_invalid_degrees_rejected(self):
        self.matrix["variants"][0]["host_gpus"] = [7]
        with self.assertRaises(ValueError):
            runner.ordered_variants(self.matrix)
        self.matrix["variants"][0]["host_gpus"] = [0]
        self.matrix["variants"][0]["ulysses_degree"] = 2
        with self.assertRaises(ValueError):
            runner.ordered_variants(self.matrix)

    def test_full_frame_changes_geometry_only(self):
        mask = object()
        chunk = SimpleNamespace(context=[None, 0, 1], mask=mask, bbox=(10, 20, 30, 40))
        hook.full_frame_geometry([SimpleNamespace(chunks=[chunk])], 1080, 1920)
        self.assertEqual(chunk.bbox, (0, 0, 1920, 1080))
        self.assertEqual((chunk.aligned_w, chunk.aligned_h), (1920, 1088))
        self.assertEqual(chunk.context, [None, 0, 1])
        self.assertIs(chunk.mask, mask)
        with self.assertRaises(ValueError):
            hook.full_frame_geometry([], 720, 1280)

    def test_stage_residency_starts_dit_on_cpu(self):
        variant = next(v for v in self.matrix["variants"] if v["id"] == "u2_stage_resident")
        cmd = worker.command_for(variant, Path("/benchmark"), "/model")
        self.assertEqual(cmd[cmd.index("--dit-cpu-offload") + 1], "true")
        self.assertEqual(cmd[cmd.index("--dit-layerwise-offload") + 1], "false")
        self.assertNotIn("--use-fsdp-inference", cmd)

    def test_stage_residency_unloads_dit_even_after_failure(self):
        calls = []

        class Model:
            def to(self, device):
                calls.append((self.name, device))
                return self

            def parameters(self):
                return [SimpleNamespace(device=SimpleNamespace(type="cuda"))]

        transformer = Model()
        transformer.name = "dit"
        transformer.should_skip_forward_for_cached_states = lambda **kw: False
        transformer.blocks = [SimpleNamespace(register_forward_pre_hook=lambda fn:
                                               SimpleNamespace(remove=lambda: calls.append("remove")))]
        vae = Model()
        vae.name = "vae"

        class Stage:
            def forward(self, batch, args):
                calls.append("compute")
                raise RuntimeError("simulated OOM")

        module = SimpleNamespace(VideoEditDenoisingStage=Stage,
                                 get_local_torch_device=lambda: "cuda")
        fake_torch = SimpleNamespace(nn=SimpleNamespace(Module=Model), cuda=SimpleNamespace(
            synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
            empty_cache=lambda: None, max_memory_allocated=lambda: 42,
            max_memory_reserved=lambda: 84))
        params = SimpleNamespace(runtime_window_index=0, runtime_num_frames=49,
                                 runtime_height=1088, runtime_width=1920,
                                 runtime_latents=SimpleNamespace(shape=(1, 16, 13, 136, 240)))
        batch = SimpleNamespace(sampling_params=params, enable_teacache=True,
                                teacache_params=SimpleNamespace(teacache_thresh=0.3),
                                output_file_path=lambda: "/test/oom.mov")
        args = SimpleNamespace(dit_layerwise_offload=False, use_fsdp_inference=False)
        stage = Stage()
        stage.transformer = transformer
        stage.pipeline = lambda: SimpleNamespace(modules={"vae": vae, "dit": transformer})
        with patch.dict("sys.modules", torch=fake_torch), patch.dict(
                "os.environ", VIDEOEDIT_SP_BENCH_RESIDENT="1"), patch.object(hook, "emit") as emit:
            hook.install_stages(module)
            with self.assertRaisesRegex(RuntimeError, "simulated OOM"):
                stage.forward(batch, args)
            self.assertIn("simulated OOM", emit.call_args.kwargs["status"])
        self.assertEqual(calls, [("vae", "cpu"), ("dit", "cuda"), "compute", ("dit", "cpu"), "remove"])

    def test_aggregation_uses_slowest_rank_and_excludes_warmup(self):
        with TemporaryDirectory(dir=HERE) as tmp:
            out = Path(tmp)
            (out / "variant.json").write_text(json.dumps({"id": "u2", "num_gpus": 2}))
            (out / "completion.json").write_text('{}')
            events = []
            for i in range(1, 4):
                (out / f"timed{i}-result.json").write_text(json.dumps({"status": "completed"}))
                (out / f"timed{i}-samples.jsonl").write_text('{"memory_bytes": "123"}\n')
                for rank in range(2):
                    common = dict(rank=rank, output=f"/u2-timed{i}.mov", status="ok")
                    events.append(dict(common, kind="request", seconds=i + rank))
                    events.append(dict(common, kind="stage", stage="VideoEditDenoisingStage", seconds=rank + 0.5))
            events.append(dict(rank=0, output="/u2-warmup.mov", kind="request", seconds=999, status="ok"))
            (out / "rank0.jsonl").write_text("\n".join(map(json.dumps, events)))
            result = report.summarize(out)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["e2e_median_s"], 3)
            self.assertEqual(result["denoise_median_s"], 1.5)
            self.assertEqual(result["gpu_seconds"], 6)
            # A completion marker alone cannot make missing rank results valid.
            events = [e for e in events if e["rank"] == 0]
            (out / "rank0.jsonl").write_text("\n".join(map(json.dumps, events)))
            self.assertEqual(report.summarize(out)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()

"""Opt-in instrumentation, imported only by the isolated benchmark backend."""
import importlib.abc
import importlib.machinery
import os
import sys

TARGETS = {
    "sglang.multimodal_gen.runtime.videoedit.streaming": "geometry",
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.videoedit_wan": "stages",
    "sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline": "pipeline",
}


class BenchmarkFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        original = spec.loader

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return original.create_module(spec)

            def exec_module(self, module):
                original.exec_module(module)
                import benchmark_hook

                getattr(benchmark_hook, "install_" + TARGETS[fullname])(module)

        spec.loader = Loader()
        return spec


if os.environ.get("VIDEOEDIT_SP_BENCH_DIR"):
    sys.meta_path.insert(0, BenchmarkFinder())

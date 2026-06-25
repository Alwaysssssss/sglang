#!/usr/bin/env python3
"""Unit checks for the VideoEdit optimizer script stage mapping."""

from __future__ import annotations

import os
import sys
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import videoedit_optimizer as opt  # noqa: E402


class StageMappingTest(unittest.TestCase):
    def test_expected_stage_names_are_registered(self) -> None:
        expected = {
            "sp1_offload",
            "sp1_no_offload",
            "sp1_no_offload_compile",
            "sp1_no_offload_compile_torch_sdpa",
            "sp1_no_offload_compile_fa",
            "sp1_no_offload_compile_sage_attn",
            "sp1_no_offload_compile_sage_attn_3",
            "sp2_no_offload_torch_sdpa",
            "sp2_no_offload_fa",
            "sp2_no_offload_sage_attn",
            "sp2_no_offload_sage_attn_3",
            "sp2_ring_no_offload_fa",
            "tp2_no_offload_fa",
            "sp2_no_offload_compile_fa",
            "sp2_no_offload_compile_fa_teacache",
            "sp2_no_offload_compile_fa_cache_rdt010",
            "sp2_no_offload_compile_fa_cache_rdt012",
            "sp2_no_offload_compile_fa_cache_rdt018",
            "sp2_no_offload_compile_fa_cache_fast",
            "cfg2_offload",
            "cfg2_offload_fa",
            "quant_branch_fp8_dynamic",
            "offload_branch",
        }
        self.assertEqual(expected, set(opt.STAGES))

    def test_cli_commands_keep_50_step_quality_constraints(self) -> None:
        env = opt.default_env()
        for name in opt.STAGES:
            with self.subTest(stage=name):
                cmd_env, argv = opt.build_cli_command(name, env)
                joined = " ".join(argv)
                self.assertIn("--num-inference-steps 50", joined)
                self.assertIn("--dynamic-cfg-max-step 15", joined)
                self.assertIn("--warmup-steps 1", joined)
                self.assertIn("--perf-dump-path", argv)
                self.assertEqual(cmd_env["SGLANG_CACHE_DIT_ENABLED"], "true" if "cache_" in name else "false")

    def test_non_teacache_stages_explicitly_disable_teacache(self) -> None:
        env = opt.default_env()
        for name in opt.STAGES:
            _, argv = opt.build_cli_command(name, env)
            if name.endswith("_teacache"):
                self.assertIn("--enable-teacache", argv)
                self.assertNotIn("--no-enable-teacache", argv)
            else:
                self.assertIn("--no-enable-teacache", argv)
                self.assertNotIn("--enable-teacache", argv)

    def test_submit_payload_uses_teacache_only_for_teacache_stage(self) -> None:
        env = opt.default_env()
        teacache_payload = opt.build_submit_payload(
            "sp2_no_offload_compile_fa_teacache", env
        )
        cache_dit_payload = opt.build_submit_payload(
            "sp2_no_offload_compile_fa_cache_rdt012", env
        )
        normal_payload = opt.build_submit_payload("sp2_no_offload_fa", env)

        self.assertTrue(teacache_payload["enable_teacache"])
        self.assertFalse(cache_dit_payload["enable_teacache"])
        self.assertFalse(normal_payload["enable_teacache"])
        for payload in (teacache_payload, cache_dit_payload, normal_payload):
            self.assertEqual(payload["num_inference_steps"], 50)
            self.assertEqual(payload["dynamic_cfg_max_step"], 15)

    def test_quant_stage_uses_quant_transformer_override(self) -> None:
        env = opt.default_env()
        env["QUANT_TRANSFORMER_PATH"] = "/tmp/quant-transformer"
        _, argv = opt.build_cli_command("quant_branch_fp8_dynamic", env)
        self.assertIn("--transformer-path", argv)
        self.assertEqual(argv[argv.index("--transformer-path") + 1], "/tmp/quant-transformer")
        self.assertIn("--transformer-quantization", argv)
        self.assertIn("fp8_dynamic", argv)

    def test_generated_commands_do_not_embed_legacy_paths_or_20_step_runs(self) -> None:
        env = opt.default_env()
        for name in opt.STAGES:
            with self.subTest(stage=name):
                cli_env, cli_argv = opt.build_cli_command(name, env)
                serve_env, serve_argv = opt.build_serve_command(name, env)
                rendered = "\n".join(
                    [
                        opt.render_shell_command(cli_argv, cli_env),
                        opt.render_shell_command(serve_argv, serve_env),
                    ]
                )
                self.assertNotIn("/home/tyx", rendered)
                self.assertNotIn("--num-inference-steps 20", rendered)

    def test_cfg_parallel_stages_enable_cfg_parallel(self) -> None:
        env = opt.default_env()
        for name in ("cfg2_offload", "cfg2_offload_fa"):
            with self.subTest(stage=name):
                spec = opt.STAGES[name]
                self.assertTrue(spec.cfg_parallel)
                self.assertEqual(spec.num_gpus, 2)
                self.assertEqual(spec.sp_degree, 1)

                _, cli_argv = opt.build_cli_command(name, env)
                _, serve_argv = opt.build_serve_command(name, env)

                self.assertIn("--enable-cfg-parallel", cli_argv)
                self.assertIn("--enable-cfg-parallel", serve_argv)

        _, normal_cli_argv = opt.build_cli_command("sp1_offload", env)
        _, normal_serve_argv = opt.build_serve_command("sp1_offload", env)
        self.assertNotIn("--enable-cfg-parallel", normal_cli_argv)
        self.assertNotIn("--enable-cfg-parallel", normal_serve_argv)


if __name__ == "__main__":
    unittest.main()

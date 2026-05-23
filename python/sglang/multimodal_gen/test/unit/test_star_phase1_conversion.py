import json
import os
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from sglang.multimodal_gen.tools.convert_star_cogvideox_sr import run_conversion
from sglang.multimodal_gen.tools.star_cogvideox_keymap import (
    extract_transformer_state_dict,
    extract_vae_state_dict,
)


class TestStarCogVideoXKeymap(unittest.TestCase):
    def test_extract_transformer_state_dict_strips_main_prefix(self):
        checkpoint = {
            "module": OrderedDict(
                {
                    "model.diffusion_model.mixins.patch_embed.proj_sr.weight": torch.ones(
                        1, 1
                    ),
                    "conditioner.embedders.0.foo": torch.ones(1),
                }
            )
        }
        extracted = extract_transformer_state_dict(checkpoint)
        self.assertIn("mixins.patch_embed.proj_sr.weight", extracted.state_dict)
        self.assertNotIn(
            "model.diffusion_model.mixins.patch_embed.proj_sr.weight",
            extracted.state_dict,
        )
        self.assertEqual(extracted.dropped_key_count, 1)

    def test_extract_transformer_state_dict_merges_lora_into_original_weight(self):
        checkpoint = {
            "module": OrderedDict(
                {
                    "model.diffusion_model.transformer.layers.0.attention.dense.original.weight": torch.zeros(
                        2, 2
                    ),
                    "model.diffusion_model.transformer.layers.0.attention.dense.original.bias": torch.zeros(
                        2
                    ),
                    "model.diffusion_model.transformer.layers.0.attention.dense.matrix_A.0": torch.ones(
                        2, 2
                    ),
                    "model.diffusion_model.transformer.layers.0.attention.dense.matrix_B.0": torch.ones(
                        2, 2
                    ),
                }
            )
        }
        extracted = extract_transformer_state_dict(checkpoint)

        self.assertIn(
            "transformer.layers.0.attention.dense.original.weight",
            extracted.state_dict,
        )
        self.assertNotIn(
            "transformer.layers.0.attention.dense.matrix_A.0",
            extracted.state_dict,
        )
        self.assertNotIn(
            "transformer.layers.0.attention.dense.matrix_B.0",
            extracted.state_dict,
        )
        self.assertTrue(
            torch.equal(
                extracted.state_dict[
                    "transformer.layers.0.attention.dense.original.weight"
                ],
                torch.ones(2, 2),
            )
        )

    def test_extract_vae_state_dict_drops_loss_keys(self):
        checkpoint = {
            "state_dict": {
                "encoder.conv_in.weight": torch.ones(1),
                "loss.logvar": torch.zeros(1),
            }
        }
        extracted = extract_vae_state_dict(checkpoint)
        self.assertIn("encoder.conv_in.weight", extracted.state_dict)
        self.assertNotIn("loss.logvar", extracted.state_dict)
        self.assertEqual(extracted.dropped_key_count, 1)


class TestStarCogVideoXConversion(unittest.TestCase):
    def _write_fake_transformer_ckpt(self, path: str) -> None:
        checkpoint = {
            "module": OrderedDict(
                {
                    "model.diffusion_model.mixins.patch_embed.proj_sr.weight": torch.ones(
                        2, 2
                    ),
                    "model.diffusion_model.mixins.patch_embed.proj_sr.bias": torch.zeros(
                        2
                    ),
                    "model.diffusion_model.time_embed.0.weight": torch.ones(2, 2),
                    "first_stage_model.encoder.conv_in.weight": torch.ones(2, 2),
                    "conditioner.embedders.0.fake.weight": torch.ones(2, 2),
                }
            )
        }
        torch.save(checkpoint, path)

    def _write_fake_vae_ckpt(self, path: str) -> None:
        checkpoint = {
            "state_dict": {
                "encoder.conv_in.weight": torch.ones(2, 2),
                "decoder.conv_out.weight": torch.ones(2, 2),
                "loss.logvar": torch.ones(1),
            }
        }
        torch.save(checkpoint, path)

    def _write_fake_text_encoder_dir(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        safetensors_save_file(
            {"encoder.block.0.layer.0.SelfAttention.q.weight": torch.ones(2, 2)},
            str(p / "model.safetensors"),
        )
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "architectures": ["T5EncoderModel"],
                    "model_type": "t5",
                    "d_model": 64,
                },
                f,
            )
        with open(p / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "T5Tokenizer"}, f)
        for name in ("special_tokens_map.json", "added_tokens.json"):
            with open(p / name, "w", encoding="utf-8") as f:
                json.dump({}, f)
        with open(p / "spiece.model", "wb") as f:
            f.write(b"fake sentencepiece")

    def _write_fake_yaml(self, path: str) -> None:
        payload = {
            "args": {
                "latent_channels": 16,
                "sampling_num_frames": 7,
            },
            "model": {
                "scale_factor": 0.7,
                "network_config": {
                    "target": "dit_video_concat.DiffusionTransformer",
                    "params": {
                        "hidden_size": 128,
                        "num_layers": 2,
                        "num_attention_heads": 4,
                        "patch_size": 2,
                        "latent_width": 90,
                        "latent_height": 60,
                        "time_compressed_rate": 4,
                    },
                },
                "first_stage_config": {
                    "target": "vae_modules.autoencoder.VideoAutoencoderInferenceWrapper",
                    "params": {
                        "cp_size": 1,
                        "encoder_config": {"params": {"z_channels": 16}},
                        "decoder_config": {"params": {"z_channels": 16}},
                        "ckpt_path": "/should/not/be/exported",
                    },
                },
                "sampler_config": {
                    "target": "sgm.modules.diffusionmodules.sampling.VPSDEDPMPP2MSampler",
                    "params": {
                        "num_steps": 50,
                        "guider_config": {"params": {"scale": 6}},
                    },
                },
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

    def test_run_conversion_exports_expected_layout(self):
        with tempfile.TemporaryDirectory() as tempdir:
            src = Path(tempdir) / "src"
            out = Path(tempdir) / "out"
            src.mkdir(parents=True, exist_ok=True)

            transformer_ckpt = src / "mp_rank_00_model_states.pt"
            vae_ckpt = src / "3d-vae.pt"
            text_dir = src / "t5"
            yaml_path = src / "config.yaml"

            self._write_fake_transformer_ckpt(str(transformer_ckpt))
            self._write_fake_vae_ckpt(str(vae_ckpt))
            self._write_fake_text_encoder_dir(str(text_dir))
            self._write_fake_yaml(str(yaml_path))

            args = type(
                "Args",
                (),
                {
                    "src_transformer": str(transformer_ckpt),
                    "src_vae": str(vae_ckpt),
                    "src_text_encoder": str(text_dir),
                    "src_tokenizer": None,
                    "src_config": str(yaml_path),
                    "output_dir": str(out),
                    "overwrite": False,
                    "dry_run": False,
                    "skip_hashes": True,
                },
            )()

            report = run_conversion(args)
            self.assertEqual(report.pipeline_class_name, "StarCogVideoXSRPipeline")

            self.assertTrue((out / "model_index.json").exists())
            self.assertTrue((out / "star_integration_config.json").exists())
            self.assertTrue((out / "transformer" / "model.safetensors").exists())
            self.assertTrue((out / "transformer" / "config.json").exists())
            self.assertTrue((out / "vae" / "model.safetensors").exists())
            self.assertTrue((out / "vae" / "config.json").exists())
            self.assertTrue((out / "scheduler" / "scheduler_config.json").exists())
            self.assertTrue((out / "manifests" / "source_assets.json").exists())
            self.assertTrue((out / "manifests" / "conversion_report.json").exists())
            self.assertTrue((out / "manifests" / "key_mapping_report.json").exists())

            transformer_sd = safetensors_load_file(
                str(out / "transformer" / "model.safetensors")
            )
            self.assertIn("mixins.patch_embed.proj_sr.weight", transformer_sd)
            self.assertNotIn(
                "model.diffusion_model.mixins.patch_embed.proj_sr.weight",
                transformer_sd,
            )

            vae_sd = safetensors_load_file(str(out / "vae" / "model.safetensors"))
            self.assertIn("encoder.conv_in.weight", vae_sd)
            self.assertNotIn("loss.logvar", vae_sd)

            with open(out / "model_index.json", encoding="utf-8") as f:
                model_index = json.load(f)
            self.assertEqual(model_index["_class_name"], "StarCogVideoXSRPipeline")
            self.assertEqual(model_index["transformer"][0], "diffusers")
            self.assertEqual(model_index["vae"][0], "diffusers")
            self.assertEqual(model_index["scheduler"][0], "diffusers")

            with open(out / "vae" / "config.json", encoding="utf-8") as f:
                vae_cfg = json.load(f)
            self.assertNotIn("ckpt_path", vae_cfg)
            self.assertEqual(vae_cfg["_class_name"], "StarCogVideoXSRVAE")


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class TestTextEncodingStage(unittest.TestCase):
    class _DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.to_calls = []

        def to(self, *args, **kwargs):
            device = args[0] if args else kwargs.get("device")
            self.to_calls.append(str(device))
            return self

    def test_verify_input_allows_empty_negative_prompt_for_cfg(self):
        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock(comfyui_mode=False)):
            stage = TextEncodingStage(text_encoders=[], tokenizers=[])

        batch = Req(prompt="test", negative_prompt="")
        batch.do_classifier_free_guidance = True

        result = stage.verify_input(batch, SimpleNamespace())
        self.assertTrue(result.is_valid(), result.get_failure_summary())

    def test_ensure_text_encoders_loaded_moves_cpu_encoder_back_to_target_device(self):
        encoder = self._DummyEncoder()
        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock(comfyui_mode=False)):
            stage = TextEncodingStage(text_encoders=[encoder], tokenizers=[MagicMock()])

        stage._ensure_text_encoders_loaded(torch.device("cuda:0"))

        self.assertEqual(encoder.to_calls, ["cuda:0"])


if __name__ == "__main__":
    unittest.main()

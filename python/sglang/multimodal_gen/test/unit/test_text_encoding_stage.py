import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class TestTextEncodingStage(unittest.TestCase):
    def test_verify_input_allows_empty_negative_prompt_for_cfg(self):
        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock(comfyui_mode=False)):
            stage = TextEncodingStage(text_encoders=[], tokenizers=[])

        batch = Req(prompt="test", negative_prompt="")
        batch.do_classifier_free_guidance = True

        result = stage.verify_input(batch, SimpleNamespace())
        self.assertTrue(result.is_valid(), result.get_failure_summary())


if __name__ == "__main__":
    unittest.main()

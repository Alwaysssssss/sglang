"""CUDA regression checks for immutable CPU originals and zero D2H release."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.utils.readonly_model_residency import readonly_model_on_device


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4).t())
        self.tied = self.weight
        self.register_buffer("bias", torch.ones(4))
        self.child = torch.nn.Module()
        self.child.register_buffer("shared", self.bias)
        self.register_buffer("optional", None)

    def forward(self, x):
        return x @ self.weight + self.bias


class CPUResidencyTests(unittest.TestCase):
    def test_encoder_stages_use_reference_restore_on_success_and_failure(self):
        import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.videoedit_wan as stages

        args = SimpleNamespace(text_encoder_cpu_offload=True,
                               image_encoder_cpu_offload=True, disable_autocast=True,
                               pipeline_config=SimpleNamespace(dit_precision='bf16'))
        for failure in (False, True):
            model = Model().eval()
            pointer = model.weight.data_ptr()
            params = SimpleNamespace(prompt='test', negative_prompt='', guidance_scale=5.0)
            def encode(*unused, **kwargs):
                if failure:
                    raise RuntimeError('encoder failure')
                return [torch.ones(1, 4)], None, None
            stage = SimpleNamespace(text_encoder=model, transformer=model,
                                    text_stage=SimpleNamespace(encode_text=encode, text_encoders=[model]))
            with patch.dict('os.environ', {'SGLANG_VIDEOEDIT_ENCODER_CPU_MASTER': '1'}), \
                 patch.object(stages, '_videoedit_params', return_value=params), \
                 patch.object(stages, 'get_local_torch_device', return_value=torch.device('cpu')), \
                 patch.object(model, 'to', side_effect=AssertionError('unexpected model migration')):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'encoder failure'):
                        stages.VideoEditTextEncodingStage.forward(stage, None, args)
                else:
                    stages.VideoEditTextEncodingStage.forward(stage, None, args)
                    self.assertIsNotNone(params.runtime_negative_prompt_embeds)
            self.assertEqual(model.weight.data_ptr(), pointer)

        # Image preprocessing failures must also release uploaded weights.
        params = SimpleNamespace(use_clip=True, runtime_window_frames=[SimpleNamespace(
            convert=lambda mode: (_ for _ in ()).throw(RuntimeError('image failure')))],
            runtime_height=480, runtime_width=544)
        stage = SimpleNamespace(image_encoder=model, transformer=model)
        with patch.dict('os.environ', {'SGLANG_VIDEOEDIT_ENCODER_CPU_MASTER': '1'}), \
             patch.object(stages, '_videoedit_params', return_value=params), \
             patch.object(stages, 'get_local_torch_device', return_value=torch.device('cpu')), \
             patch.object(model, 'to', side_effect=AssertionError('unexpected model migration')):
            with self.assertRaisesRegex(RuntimeError, 'image failure'):
                stages.VideoEditImageEncodingStage.forward(stage, None, args)
        self.assertEqual(model.weight.data_ptr(), pointer)

    def test_restore_originals_without_copy_and_respect_cpu_updates(self):
        model = Model().eval()
        parameter = model.weight
        pointer = parameter.data_ptr()
        bias = model.bias
        for _ in range(3):
            expected = model(torch.ones(2, 4)).detach()
            copies = []
            def simulated_upload(tensor, *args, **kwargs):
                copies.append(tensor)
                return tensor.clone()
            with patch.object(torch.Tensor, 'to', simulated_upload), torch.no_grad():
                with readonly_model_on_device(model, torch.device('cpu')):
                    self.assertNotEqual(model.weight.data_ptr(), pointer)
                    self.assertIs(model.weight, parameter)
                    self.assertIs(model.tied, parameter)
                    self.assertIs(model.bias, model.child.shared)
                    torch.testing.assert_close(model(torch.ones(2, 4)), expected)
                    count_before_release = len(copies)
                self.assertEqual(len(copies), count_before_release)
                self.assertEqual(model.weight.data_ptr(), pointer)
                self.assertIs(model.bias, bias)
                self.assertIs(model.child.shared, bias)
            with torch.no_grad():
                model.weight.add_(1)

    def test_partial_load_and_forward_failure_restore_originals(self):
        for fail_load in (True, False):
            model = Model().eval()
            pointer, bias = model.weight.data_ptr(), model.bias
            def simulated_upload(tensor, *args, **kwargs):
                if fail_load and tensor is bias:
                    raise RuntimeError('load failure')
                return tensor.clone()
            with patch.object(torch.Tensor, 'to', simulated_upload):
                with self.assertRaisesRegex(RuntimeError, 'failure'):
                    with readonly_model_on_device(model, torch.device('cpu')):
                        raise RuntimeError('forward failure')
            self.assertEqual(model.weight.data_ptr(), pointer)
            self.assertIs(model.bias, bias)
            self.assertIs(model.child.shared, bias)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class ResidencyTests(unittest.TestCase):
    def test_repeated_forward_keeps_original_storage_and_no_d2h(self):
        model = Model().eval()
        parameter = model.weight
        storage = parameter.data_ptr()
        bias = model.bias
        cpu_input = torch.ones(2, 4)
        x = cpu_input.cuda()
        original_to = torch.Tensor.to
        def forbid_d2h(tensor, *args, **kwargs):
            target = kwargs.get('device', args[0] if args else None)
            if tensor.device.type == 'cuda' and target is not None and str(target) == 'cpu':
                raise AssertionError('Unexpected GPU-to-CPU weight copy')
            return original_to(tensor, *args, **kwargs)
        for _ in range(3):
            expected = model(cpu_input).detach().cuda()
            with patch.object(torch.Tensor, 'to', forbid_d2h), torch.no_grad():
                with readonly_model_on_device(model, torch.device('cuda')):
                    self.assertIs(model.weight, parameter)
                    self.assertIs(model.tied, parameter)
                    self.assertIs(model.bias, model.child.shared)
                    self.assertEqual(model.weight.stride(), (1, 4))
                    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
                self.assertEqual(model.weight.data_ptr(), storage)
                self.assertIs(model.bias, bias)
                self.assertIs(model.child.shared, bias)
                self.assertIsNone(model.optional)
                self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))
            # CPU updates between requests must not use a stale snapshot.
            with torch.no_grad(): model.weight.add_(1)

    def test_failure_during_load_and_forward_restores_cpu(self):
        for fail_load in (True, False):
            model = Model().eval(); original = model.weight.data_ptr(); bias = model.bias
            original_to = torch.Tensor.to
            def failing_to(tensor, *args, **kwargs):
                if fail_load and tensor is bias: raise RuntimeError('load failure')
                return original_to(tensor, *args, **kwargs)
            with patch.object(torch.Tensor, 'to', failing_to):
                with self.assertRaisesRegex(RuntimeError, 'failure'):
                    with readonly_model_on_device(model, torch.device('cuda')):
                        raise RuntimeError('forward failure')
            self.assertEqual(model.weight.data_ptr(), original)
            self.assertIs(model.bias, bias)
            self.assertIs(model.child.shared, bias)
            with readonly_model_on_device(model, torch.device('cuda')):
                self.assertEqual(model.weight.device.type, 'cuda')

    def test_rejects_gpu_originals_without_discarding_them(self):
        model = Model().cuda(); original = model.weight.data_ptr()
        with self.assertRaisesRegex(ValueError, 'CPU originals'):
            with readonly_model_on_device(model, torch.device('cuda')): pass
        self.assertEqual(model.weight.data_ptr(), original)
        self.assertEqual(model.weight.device.type, 'cuda')


if __name__ == '__main__': unittest.main()

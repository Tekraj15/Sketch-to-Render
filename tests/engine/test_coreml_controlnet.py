import numpy as np
import pytest
import torch
from diffusers import ControlNetModel
from diffusers.models.controlnet import ControlNetOutput

from src.engine import coreml_engine
from src.engine.coreml_engine import (CoreMLControlNetWrapper, CoreMLEngine, NUM_DOWN_RESIDUALS,
                                      needs_batch_pad, sanitize_input)

CONFIG = {"_class_name": "ControlNetModel", "global_pool_conditions": False, "in_channels": 4}


class FakeCoreMLControlNet:
    """Mimics CoreMLModel: batch-2 compiled inputs; residual i is +(i+1) in row 0 and -(i+1) in row 1."""

    def __init__(self, batch=2):
        self.expected_inputs = {
            "sample": {"shape": [batch, 4, 64, 64], "dtype": "float16"},
            "timestep": {"shape": [batch], "dtype": "float16"},
            "encoder_hidden_states": {"shape": [batch, 768, 1, 77], "dtype": "float16"},
            "controlnet_cond": {"shape": [batch, 3, 512, 512], "dtype": "float16"},
        }
        self.calls = []

    def predict(self, inputs):
        self.calls.append(inputs)
        return {f"additional_residual_{i}": np.stack([np.full((2, 2), i + 1.0), np.full((2, 2), -(i + 1.0))])
                .astype(np.float32) for i in range(NUM_DOWN_RESIDUALS + 1)}


def _inputs(batch):
    return dict(sample=torch.randn(batch, 4, 64, 64), timestep=torch.tensor(759.0),
                encoder_hidden_states=torch.randn(batch, 77, 768),
                controlnet_cond=torch.rand(batch, 3, 512, 512))


# ---------- shared helpers ----------
def test_sanitize_scalar_timestep_is_expanded_padded_and_cast():
    out = sanitize_input(torch.tensor(759.0), {"shape": [2], "dtype": "float16"}, do_pad=True)
    assert out.shape == (2,) and out.dtype == np.float16 and np.all(out == 759)


def test_sanitize_text_embeddings_to_ane_layout():
    ehs = torch.randn(1, 77, 768)
    out = sanitize_input(ehs, {"shape": [2, 768, 1, 77], "dtype": "float16"}, do_pad=True)
    assert out.shape == (2, 768, 1, 77)
    np.testing.assert_allclose(out[0, :, 0, 5], ehs[0, 5].numpy().astype(np.float16))


def test_needs_batch_pad():
    fake = FakeCoreMLControlNet(batch=2)
    assert needs_batch_pad(fake, torch.zeros(1, 4, 64, 64))
    assert not needs_batch_pad(fake, torch.zeros(2, 4, 64, 64))
    assert not needs_batch_pad(FakeCoreMLControlNet(batch=1), torch.zeros(1, 4, 64, 64))


# ---------- wrapper ----------
def test_wrapper_satisfies_pipeline_type_contract():
    cn = CoreMLControlNetWrapper(FakeCoreMLControlNet(), CONFIG)
    assert isinstance(cn, ControlNetModel)
    assert cn.config.global_pool_conditions is False
    assert "_class_name" not in cn.config
    assert cn.dtype == torch.float32 and cn.device == torch.device("cpu")
    assert sum(p.numel() for p in cn.parameters()) == 0   # no PyTorch weights allocated


def test_config_gets_init_defaults_like_controlnetmodel():
    # lllyasviel/sd-controlnet-scribble's config.json predates `global_pool_conditions`;
    # ControlNetModel.__init__ fills such keys from its signature defaults, so must the wrapper.
    cn = CoreMLControlNetWrapper(FakeCoreMLControlNet(), {"_class_name": "ControlNetModel", "in_channels": 4})
    assert cn.config.global_pool_conditions is False
    assert cn.config.in_channels == 4 and cn.config.conditioning_channels == 3


def test_batch1_is_padded_then_sliced_and_scaled():
    fake = FakeCoreMLControlNet()
    cn = CoreMLControlNetWrapper(fake, CONFIG)
    down, mid = cn(**_inputs(1), conditioning_scale=0.5, return_dict=False)
    sent = fake.calls[-1]
    assert sent["sample"].shape == (2, 4, 64, 64) and sent["sample"].dtype == np.float16
    assert sent["encoder_hidden_states"].shape == (2, 768, 1, 77)
    assert sent["controlnet_cond"].shape == (2, 3, 512, 512)
    assert len(down) == NUM_DOWN_RESIDUALS
    assert all(d.shape[0] == 1 for d in down) and mid.shape[0] == 1
    assert torch.allclose(down[3], torch.full((1, 2, 2), 4 * 0.5))       # residual 3, row 0, scaled
    assert torch.allclose(mid, torch.full((1, 2, 2), 13 * 0.5))          # residual 12 is the mid block
    assert all(d.dtype == torch.float32 for d in down)


def test_batch2_cfg_is_passed_through():
    fake = FakeCoreMLControlNet()
    down, mid = CoreMLControlNetWrapper(fake, CONFIG)(**_inputs(2), return_dict=False)
    assert mid.shape[0] == 2 and torch.all(mid[1] == -13)


def test_return_dict_and_guess_mode():
    cn = CoreMLControlNetWrapper(FakeCoreMLControlNet(), CONFIG)
    out = cn(**_inputs(1))
    assert isinstance(out, ControlNetOutput) and len(out.down_block_res_samples) == NUM_DOWN_RESIDUALS
    with pytest.raises(NotImplementedError):
        cn(**_inputs(1), guess_mode=True)


# ---------- engine wiring ----------
def test_engine_fails_loudly_when_coreml_controlnet_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="coreml_compiler"):
        CoreMLEngine(model_dir=str(tmp_path), controlnet_backend="coreml")


def test_engine_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        CoreMLEngine(model_dir=str(tmp_path), controlnet_backend="tpu")


def test_engine_rejects_unknown_compute_unit(tmp_path):
    with pytest.raises(ValueError, match="compute_unit"):
        CoreMLEngine(model_dir=str(tmp_path), compute_unit="NE_ONLY")


def test_engine_defaults_to_cpu_and_ne_for_every_model(tmp_path, monkeypatch):
    units = []

    class StopAfterControlNet(Exception):
        pass

    def fake_coreml_model(path, compute_unit):
        units.append(compute_unit)
        raise StopAfterControlNet  # first Core ML model built is the ControlNet

    (tmp_path / "ControlNet.mlpackage").mkdir()
    monkeypatch.setattr(coreml_engine, "CoreMLModel", fake_coreml_model)
    with pytest.raises(StopAfterControlNet):
        CoreMLEngine(model_dir=str(tmp_path))
    assert units == ["CPU_AND_NE"]


def test_harness_forwards_controlnet_backend(monkeypatch):
    seen = {}

    class Spy:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.pipe = None

    monkeypatch.setattr(coreml_engine, "CoreMLEngine", Spy)
    from src.evaluation.engines import build_engine
    build_engine("coreml", {"model_dir": "m", "controlnet_backend": "pytorch", "compute_unit": "ALL"})
    assert seen == {"model_dir": "m", "controlnet_backend": "pytorch", "compute_unit": "ALL"}
    build_engine("coreml", {})
    assert seen["controlnet_backend"] == "coreml" and seen["compute_unit"] == "CPU_AND_NE"

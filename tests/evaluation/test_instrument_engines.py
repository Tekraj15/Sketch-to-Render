import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.evaluation.engines import build_engine, seed_everything
from src.evaluation.instrument import ComponentTimer, MemorySampler, swap_used_mb


class Slow(nn.Module):
    def forward(self, x):
        time.sleep(0.01)
        return x


class FakeVAE:
    def decode(self, z, **kw):
        time.sleep(0.01)
        return z


def _pipe(instance_decode=False):
    vae = FakeVAE()
    if instance_decode:  # CoreMLEngine replaces vae.decode with an instance attribute
        vae.decode = lambda z, **kw: z
    return SimpleNamespace(text_encoder=Slow(), controlnet=Slow(), unet=Slow(), vae=vae)


def test_timer_counts_and_times_components():
    pipe = _pipe()
    timer = ComponentTimer("cpu").attach(pipe)
    pipe.text_encoder(1)
    for _ in range(4):
        pipe.controlnet(1)
        pipe.unet(1)
    pipe.vae.decode(1)
    snap = timer.snapshot()
    assert snap["unet"]["calls"] == 4 and snap["controlnet"]["calls"] == 4
    assert snap["text_encoder"]["calls"] == 1 and snap["vae_decode"]["calls"] == 1
    assert snap["unet"]["seconds"] >= 0.04
    timer.reset()
    assert timer.snapshot() == {}


def test_detach_restores_originals():
    pipe = _pipe(instance_decode=True)
    original_decode = pipe.vae.decode
    timer = ComponentTimer("cpu").attach(pipe)
    timer.detach()
    assert "forward" not in vars(pipe.unet)          # class method restored
    assert pipe.vae.decode is original_decode        # instance attribute restored
    pipe.unet(1)
    assert timer.snapshot() == {}


def test_missing_components_are_skipped():
    timer = ComponentTimer("cpu").attach(SimpleNamespace(unet=Slow()))
    assert len(timer._restore) == 1
    timer.detach()


def test_memory_sampler_and_swap():
    mem = MemorySampler("cpu", interval_s=0.01).start()
    _ = bytearray(10 * 1024 * 1024)
    out = mem.stop()
    assert out["peak_rss_mb"] > 0 and out["peak_cuda_mb"] is None
    swap = swap_used_mb()
    assert swap is None or swap >= 0


def test_seed_everything_is_deterministic():
    seed_everything(7)
    a = torch.randn(4)
    seed_everything(7)
    assert torch.equal(a, torch.randn(4))


def test_unknown_engine_type():
    with pytest.raises(ValueError):
        build_engine("tpu")

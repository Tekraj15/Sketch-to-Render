import platform
import re
import subprocess
import threading
import time
from collections import defaultdict
from typing import Dict, Optional

import psutil
import torch

# (component name, pipeline attribute, method to wrap)
COMPONENT_TARGETS = (
    ("text_encoder", "text_encoder", "forward"),
    ("controlnet", "controlnet", "forward"),
    ("unet", "unet", "forward"),
    ("vae_decode", "vae", "decode"),
)


def device_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


class ComponentTimer:
    """
    Times pipeline components by temporarily wrapping their entry points.
    Each wrapped call is bracketed by a device sync so async GPU work is
    attributed to the component that launched it. `detach()` restores the
    original attributes exactly (instance attribute or class method).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._stats = defaultdict(lambda: [0.0, 0])
        self._restore = []

    def attach(self, pipe) -> "ComponentTimer":
        for name, attr, method in COMPONENT_TARGETS:
            obj = getattr(pipe, attr, None)
            if obj is not None and callable(getattr(obj, method, None)):
                self._wrap(name, obj, method)
        return self

    def _wrap(self, name, obj, method):
        had_instance_attr = method in vars(obj)
        previous = vars(obj).get(method)
        original = getattr(obj, method)
        stats = self._stats

        def timed(*args, **kwargs):
            device_sync(self.device)
            t0 = time.perf_counter()
            out = original(*args, **kwargs)
            device_sync(self.device)
            entry = stats[name]
            entry[0] += time.perf_counter() - t0
            entry[1] += 1
            return out

        setattr(obj, method, timed)
        self._restore.append((obj, method, had_instance_attr, previous))

    def detach(self) -> None:
        for obj, method, had_instance_attr, previous in reversed(self._restore):
            if had_instance_attr:
                setattr(obj, method, previous)
            else:
                delattr(obj, method)
        self._restore.clear()

    def reset(self) -> None:
        self._stats.clear()

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {k: {"seconds": v[0], "calls": v[1]} for k, v in self._stats.items()}


class MemorySampler:
    """Polls process RSS (and MPS driver memory) on a background thread; CUDA uses peak stats."""

    def __init__(self, device: str = "cpu", interval_s: float = 0.05):
        self.device = device
        self.interval_s = interval_s
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_rss = 0
        self.peak_mps = 0

    def _poll(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def _sample(self):
        self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
        if self.device.startswith("mps"):
            try:
                self.peak_mps = max(self.peak_mps, torch.mps.driver_allocated_memory())
            except Exception:
                pass

    def start(self) -> "MemorySampler":
        self._stop.clear()
        self.peak_rss = self.peak_mps = 0
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._sample()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Optional[float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        mb = 1024 ** 2
        out = {"peak_rss_mb": self.peak_rss / mb, "peak_mps_driver_mb": None, "peak_cuda_mb": None}
        if self.device.startswith("mps"):
            out["peak_mps_driver_mb"] = self.peak_mps / mb
        if self.device.startswith("cuda") and torch.cuda.is_available():
            out["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / mb
        return out


def swap_used_mb() -> Optional[float]:
    """System-wide swap in use; a delta across a run shows whether we pushed the machine into swap."""
    try:
        if platform.system() == "Darwin":
            text = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                                  text=True, timeout=5).stdout
            m = re.search(r"used = ([\d.]+)([MG])", text)
            if m:
                return float(m.group(1)) * (1024 if m.group(2) == "G" else 1)
        elif platform.system() == "Linux":
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["SwapTotal"].split()[0]); free = int(info["SwapFree"].split()[0])
            return (total - free) / 1024
    except Exception:
        pass
    return None

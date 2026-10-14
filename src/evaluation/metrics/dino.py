from typing import Dict, List

import torch

from src.evaluation.metrics.base import Scorer, open_image, open_reference

MODEL_ID = "facebook/dinov2-base"


class DinoScorer(Scorer):
    """Cosine similarity of DINOv2 CLS embeddings, render vs. paired reference photo."""
    name = "dino"
    requires = ("transformers",)
    needs_reference = True

    def load(self, device: str) -> None:
        # DINOv2 interpolates its position embeddings bicubically, which torch 2.5 has no
        # MPS kernel for (aten::upsample_bicubic2d); the base model is small enough for CPU.
        super().load("cuda" if device == "cuda" else "cpu")
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device).eval()

    @torch.no_grad()
    def _embed(self, image):
        px = self.processor(images=image, return_tensors="pt").to(self.device)
        cls = self.model(**px).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, dim=-1)[0]

    def score(self, records: List[Dict], run_dir: str) -> List[Dict]:
        cache, out = {}, []
        for rec in records:
            ref_path = rec.get("reference")
            if not ref_path:
                out.append({"dino_ref_sim": None})
                continue
            if ref_path not in cache:
                cache[ref_path] = self._embed(open_reference(rec))
            sim = float(self._embed(open_image(run_dir, rec["image"])) @ cache[ref_path])
            out.append({"dino_ref_sim": sim})
        return out

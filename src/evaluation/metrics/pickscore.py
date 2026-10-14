from typing import Dict, List

import torch

from src.evaluation.metrics.base import Scorer, open_image

PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL_ID = "yuvalkirstain/PickScore_v1"


class PickScorer(Scorer):
    """PickScore (Kirstain et al. 2023): logit_scale x cosine, scored against the full prompt."""
    name = "pickscore"
    requires = ("transformers",)

    def load(self, device: str) -> None:
        super().load(device)
        from transformers import AutoModel, AutoProcessor
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
        self.dtype = dtype

    @torch.no_grad()
    def score(self, records: List[Dict], run_dir: str) -> List[Dict]:
        out = []
        for rec in records:
            px = self.processor(images=open_image(run_dir, rec["image"]), return_tensors="pt")
            px = {k: v.to(self.device, self.dtype) for k, v in px.items()}
            tx = self.processor(text=rec["prompt_full"], padding=True, truncation=True,
                                max_length=77, return_tensors="pt").to(self.device)
            img = torch.nn.functional.normalize(self.model.get_image_features(**px), dim=-1)
            txt = torch.nn.functional.normalize(self.model.get_text_features(**tx), dim=-1)
            out.append({"pickscore": float(self.model.logit_scale.exp() * (img @ txt.T)[0, 0])})
        return out

from typing import Dict, List

import torch

from src.evaluation.metrics.base import Scorer, batched, open_image

MODEL_ID = "openai/clip-vit-large-patch14"


class ClipScorer(Scorer):
    """
    100 x cosine(image, text) with CLIP ViT-L/14 (CLIPScore without the 2.5 rescale).
    clip_prompt = against the user's own words (content alignment);
    clip_full   = against the composed prompt incl. Style-DNA suffix (style alignment).
    Text is truncated to CLIP's 77 tokens, as in the generator.
    """
    name = "clip"
    requires = ("transformers",)

    def load(self, device: str) -> None:
        super().load(device)
        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self.model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()

    @torch.no_grad()
    def _embed_text(self, texts):
        t = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=77).to(self.device)
        return torch.nn.functional.normalize(self.model.get_text_features(**t), dim=-1)

    @torch.no_grad()
    def score(self, records: List[Dict], run_dir: str) -> List[Dict]:
        out = []
        for chunk in batched(records, 8):
            px = self.processor(images=[open_image(run_dir, r["image"]) for r in chunk],
                                return_tensors="pt").to(self.device)
            img = torch.nn.functional.normalize(self.model.get_image_features(**px), dim=-1)
            short = self._embed_text([r["prompt_user"] for r in chunk])
            full = self._embed_text([r["prompt_full"] for r in chunk])
            for i in range(len(chunk)):
                out.append({"clip_prompt": 100 * float(img[i] @ short[i]),
                            "clip_full": 100 * float(img[i] @ full[i])})
        return out

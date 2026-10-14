from typing import Dict, List

import torch

from src.evaluation.metrics.base import Scorer, open_image

CKPT_REPO = "xswu/HPSv2"
CKPT_FILE = "HPS_v2.1_compressed.pt"


class HPSv21Scorer(Scorer):
    """
    Human Preference Score v2.1 (Wu et al. 2023): cosine(image, prompt) under the
    HPS-finetuned open_clip ViT-H-14. Loaded via open_clip directly instead of the
    `hpsv2` package, which pins old deps and ships a broken vocab path.
    """
    name = "hpsv2"
    requires = ("open_clip",)

    def load(self, device: str) -> None:
        super().load(device)
        import open_clip
        from huggingface_hub import hf_hub_download
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
        state = torch.load(hf_hub_download(CKPT_REPO, CKPT_FILE), map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def score(self, records: List[Dict], run_dir: str) -> List[Dict]:
        out = []
        for rec in records:
            image = self.preprocess(open_image(run_dir, rec["image"])).unsqueeze(0).to(self.device)
            text = self.tokenizer([rec["prompt_full"]]).to(self.device)
            img = torch.nn.functional.normalize(self.model.encode_image(image), dim=-1)
            txt = torch.nn.functional.normalize(self.model.encode_text(text), dim=-1)
            out.append({"hpsv2": float((img @ txt.T)[0, 0])})
        return out

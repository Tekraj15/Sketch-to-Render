import importlib

# name -> "module:Class"; imported lazily so the score phase only pays for what it runs.
REGISTRY = {
    "structure": "src.evaluation.metrics.structure:StructureScorer",
    "clip": "src.evaluation.metrics.clip_score:ClipScorer",
    "dino": "src.evaluation.metrics.dino:DinoScorer",
    "pickscore": "src.evaluation.metrics.pickscore:PickScorer",
    "hpsv2": "src.evaluation.metrics.hpsv2:HPSv21Scorer",
    "imagereward": "src.evaluation.metrics.image_reward:ImageRewardScorer",
    "kid": "src.evaluation.metrics.kid:KIDScorer",
    "vqascore": "src.evaluation.metrics.vqascore:VQAScorer",
}
DEFAULT_METRICS = ("structure", "clip", "dino", "pickscore", "hpsv2", "imagereward", "kid")


def get_scorer(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown metric {name!r}; choose from {sorted(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)()

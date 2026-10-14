from typing import Tuple

from src.evaluation.metrics.base import Scorer


class VQAScorer(Scorer):
    """
    Placeholder. VQAScore's smallest model (CLIP-FlanT5-XL, ~3B params) needs ~6 GB of
    weights resident on top of the process plus ~11 GB of disk, which does not fit an
    8 GB M2. Implement with `t2v_metrics` when scoring on a >=16 GB machine or a GPU box.
    """
    name = "vqascore"

    def available(self) -> Tuple[bool, str]:
        return False, "deferred: needs >=16 GB RAM (CLIP-FlanT5-XL); see CHG-20261013-01 D3"

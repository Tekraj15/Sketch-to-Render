"""
Statistics for blind pairwise human ratings (the release gate in WS1).

A rating row (ratings.csv) records which side won for one (item, seed) pair and one
question; the rater app stores the run names behind the randomized sides, so a row
resolves to (system_a, system_b, outcome) with outcome 1 = a wins, 0 = b wins, 0.5 = tie.
"""
import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

QUESTIONS = ("structure", "quality", "prompt")
RATING_FIELDS = ("timestamp", "rater", "item_id", "seed", "question",
                 "left_run", "right_run", "choice")


def load_ratings(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_match(row: Dict) -> Tuple[str, str, float]:
    a, b = row["left_run"], row["right_run"]
    outcome = {"left": 1.0, "right": 0.0, "tie": 0.5}[row["choice"]]
    return a, b, outcome


def win_rate(ratings: List[Dict], system: str, question: Optional[str] = None,
             n_boot: int = 2000, seed: int = 0) -> Dict:
    """Win rate of `system` (ties = 0.5) with a bootstrap CI over items."""
    per_item = defaultdict(list)
    for row in ratings:
        if question and row["question"] != question:
            continue
        a, b, outcome = to_match(row)
        if system == a:
            per_item[row["item_id"]].append(outcome)
        elif system == b:
            per_item[row["item_id"]].append(1.0 - outcome)
    if not per_item:
        return {"win_rate": None, "lo": None, "hi": None, "n_ratings": 0, "n_items": 0}
    items = list(per_item.values())
    flat = [v for vs in items for v in vs]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(items), len(items))
        sample = [v for i in pick for v in items[i]]
        boots.append(np.mean(sample))
    lo, hi = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))) \
        if len(items) > 1 else (None, None)
    return {"win_rate": float(np.mean(flat)), "lo": lo, "hi": hi,
            "n_ratings": len(flat), "n_items": len(items)}


def elo(matches: List[Tuple[str, str, float]], k: float = 16.0, base: float = 1000.0,
        n_perm: int = 200, seed: int = 0) -> Dict[str, float]:
    """Elo averaged over random match orderings (removes order dependence)."""
    systems = sorted({s for a, b, _ in matches for s in (a, b)})
    rng = np.random.default_rng(seed)
    totals = defaultdict(float)
    for _ in range(max(n_perm, 1)):
        r = {s: base for s in systems}
        for idx in rng.permutation(len(matches)):
            a, b, outcome = matches[idx]
            expected = 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / 400.0))
            r[a] += k * (outcome - expected)
            r[b] -= k * (outcome - expected)
        for s in systems:
            totals[s] += r[s]
    return {s: totals[s] / max(n_perm, 1) for s in systems}


def bradley_terry(matches: List[Tuple[str, str, float]], iters: int = 500,
                  tol: float = 1e-9) -> Dict[str, float]:
    """Bradley-Terry strengths via the MM algorithm (Hunter 2004); ties split half/half.
    Returned strengths are normalized to sum to 1 (P(a beats b) = p_a / (p_a + p_b))."""
    systems = sorted({s for a, b, _ in matches for s in (a, b)})
    wins = defaultdict(float)
    games = defaultdict(float)
    for a, b, outcome in matches:
        wins[a] += outcome
        wins[b] += 1.0 - outcome
        games[frozenset((a, b))] += 1.0
    p = {s: 1.0 / len(systems) for s in systems}
    for _ in range(iters):
        new = {}
        for s in systems:
            denom = sum(n / (p[s] + p[o]) for pair, n in games.items() if s in pair
                        for o in pair if o != s)
            new[s] = max(wins[s], 1e-6) / denom if denom else p[s]
        z = sum(new.values())
        new = {s: v / z for s, v in new.items()}
        if max(abs(new[s] - p[s]) for s in systems) < tol:
            p = new
            break
        p = new
    return p

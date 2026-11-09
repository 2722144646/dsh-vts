"""Choose the mouth channel for a model from lip-readability measurements.

One ruler only: how well the five vowels stay distinguishable (pairwise mouth
ROI difference in the fixed-head test). Deformation size from the boundary
test is NOT used for ranking; it only proves a parameter is alive.
"""

from __future__ import annotations

import json

READ_MIN = 10.0
ORDER = ("shape", "vowel", "hybrid", "smile", "continuous")


def classify(scores: dict[str, float], *, open_ok: bool = True) -> dict:
    """Pick the channel with the highest readability mean.

    scores: channel -> mean pairwise vowel difference (>= READ_MIN means usable).
    open_ok: whether MouthOpen itself moves at all (fallback only).
    """
    measured = {k: float(v) for k, v in scores.items() if k in ORDER and k != "continuous"}
    usable = {k: v for k, v in measured.items() if v >= READ_MIN}
    if usable:
        strategy = max(usable, key=usable.get)
    elif open_ok:
        strategy = "continuous"
    else:
        strategy = "none"
    return {
        "strategy": strategy,
        "scores": {k: round(v, 2) for k, v in measured.items()},
        "threshold": READ_MIN,
        "usable": sorted(usable),
        "reason": (
            f"{strategy} has the highest lip-readability mean"
            if strategy in usable
            else "no channel reaches the readability threshold"
        ),
    }


def strategy_path(profile_dir, model_id: str):
    """Per-model strategy file, next to this model's capability profile."""
    return profile_dir / f"{model_id}.mouth.json"


def save_strategy(profile_dir, root, model_id: str, data: dict) -> None:
    """Write the strategy for one model only; keep a legacy mirror for tools."""
    payload = dict(data)
    payload.setdefault("model", {})
    payload["model"]["modelID"] = model_id
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path = strategy_path(profile_dir, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    legacy = root / "data" / "mouth_strategy.json"
    try:
        legacy.write_text(text, encoding="utf-8")
    except OSError:
        pass


def load_strategy(profile_dir, root, model_id: str) -> dict | None:
    """Load this model's strategy; only fall back to the legacy file if its ID matches."""
    for path in (strategy_path(profile_dir, model_id), root / "data" / "mouth_strategy.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        owner = str((data.get("model") or {}).get("modelID") or "")
        if owner and owner != model_id:
            continue
        return data
    return None

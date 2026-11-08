"""Choose the mouth channel that fits a model, from boundary-test scores."""

from __future__ import annotations

import json

VOWEL_MIN = 1.5
SMILE_MIN = 6.0
OPEN_MIN = 3.0


def classify(vowel_score: float, smile_score: float, open_score: float) -> dict:
    """Pick vowel / smile / hybrid / continuous from measured visual impact.

    - vowel params (VoiceA/I/U/E/O) carry the shape when they move;
    - MouthSmile adds width when it moves;
    - both moving -> hybrid (vowel carries identity, smile adds width);
    - only MouthOpen -> continuous (open/close only);
    - nothing -> none.
    """
    vowel_ok = float(vowel_score) >= VOWEL_MIN
    smile_ok = float(smile_score) >= SMILE_MIN
    open_ok = float(open_score) >= OPEN_MIN
    if vowel_ok and smile_ok:
        strategy = "hybrid"
    elif smile_ok:
        strategy = "smile"
    elif vowel_ok:
        strategy = "vowel"
    elif open_ok:
        strategy = "continuous"
    else:
        strategy = "none"
    channels = []
    if vowel_ok:
        channels.append("vowel")
    if smile_ok:
        channels.append("smile")
    if open_ok:
        channels.append("mouth_open")
    return {
        "strategy": strategy,
        "channels": channels,
        "vowel_score": round(float(vowel_score), 2),
        "smile_score": round(float(smile_score), 2),
        "mouth_open_score": round(float(open_score), 2),
        "thresholds": {"vowel": VOWEL_MIN, "smile": SMILE_MIN, "mouth_open": OPEN_MIN},
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

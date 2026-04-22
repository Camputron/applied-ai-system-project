"""Ollama LLM integration for natural language profile extraction and explanation."""

import json
import logging
import requests


logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


def _generate(prompt: str) -> str:
    """Send a prompt to the local Ollama server and return the response text."""
    logger.info("Sending prompt to Ollama (%s)", MODEL)
    logger.debug("Prompt: %s", prompt[:200])
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json()["response"].strip()
    logger.info("Received response (%d chars)", len(text))
    logger.debug("Response: %s", text[:300])
    return text


# --- Available values (kept in sync with songs.csv) --------------------------
VALID_GENRES = [
    "pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop",
    "hip-hop", "classical", "electronic", "r&b", "country", "metal",
    "reggae", "folk", "latin", "blues",
]
VALID_MOODS = [
    "happy", "chill", "intense", "relaxed", "moody", "focused",
    "romantic", "nostalgic", "aggressive", "melancholy", "sad",
]

REQUIRED_KEYS = [
    "favorite_genre", "favorite_mood", "target_energy", "likes_acoustic",
    "min_popularity", "preferred_decade", "mood_tag_preferences",
    "likes_instrumental", "likes_live",
]


EXTRACT_PROFILE_PROMPT = """\
You are a music preference parser. Given a user's natural-language description
of what they want to listen to, extract a structured JSON profile.

Return ONLY valid JSON with these exact keys:
{{
  "favorite_genre": one of {genres},
  "favorite_mood": one of {moods},
  "target_energy": float 0.0-1.0,
  "likes_acoustic": boolean,
  "min_popularity": int 0-100 (default 0),
  "preferred_decade": string like "2020s" or "" if unspecified,
  "mood_tag_preferences": list of 1-3 short adjective strings,
  "likes_instrumental": boolean (default false),
  "likes_live": boolean (default false)
}}

Pick the CLOSEST match from the allowed values. If something is ambiguous,
make your best guess. Do NOT add extra keys or commentary outside the JSON.

User request: {user_input}
"""


def extract_profile(user_input: str) -> dict:
    """Use Ollama to parse a natural-language request into a UserProfile dict.

    Returns a dict with profile fields plus a '_confidence' key (0.0-1.0)
    indicating how many fields the LLM got right without fallback correction.
    """
    prompt = EXTRACT_PROFILE_PROMPT.format(
        genres=VALID_GENRES,
        moods=VALID_MOODS,
        user_input=user_input,
    )
    text = _generate(prompt)

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    profile = json.loads(text)
    logger.info("Parsed JSON profile successfully")

    # --- Confidence scoring ---
    checks_total = 0
    checks_passed = 0

    # 1. All required keys present?
    for key in REQUIRED_KEYS:
        checks_total += 1
        if key in profile:
            checks_passed += 1
        else:
            logger.warning("Missing key: %s", key)

    # 2. Genre is valid?
    checks_total += 1
    if profile.get("favorite_genre") in VALID_GENRES:
        checks_passed += 1
    else:
        logger.warning("Invalid genre '%s', falling back to 'pop'", profile.get("favorite_genre"))
        profile["favorite_genre"] = "pop"

    # 3. Mood is valid?
    checks_total += 1
    if profile.get("favorite_mood") in VALID_MOODS:
        checks_passed += 1
    else:
        logger.warning("Invalid mood '%s', falling back to 'happy'", profile.get("favorite_mood"))
        profile["favorite_mood"] = "happy"

    # 4. Energy in range?
    checks_total += 1
    raw_energy = profile.get("target_energy", 0.5)
    try:
        energy = float(raw_energy)
        if 0.0 <= energy <= 1.0:
            checks_passed += 1
        else:
            logger.warning("Energy %.2f out of range, clamping", energy)
    except (TypeError, ValueError):
        energy = 0.5
        logger.warning("Non-numeric energy '%s', defaulting to 0.5", raw_energy)
    profile["target_energy"] = max(0.0, min(1.0, energy))

    # 5. Booleans are booleans?
    for bool_key in ("likes_acoustic", "likes_instrumental", "likes_live"):
        checks_total += 1
        val = profile.get(bool_key)
        if isinstance(val, bool):
            checks_passed += 1
        else:
            logger.warning("Non-boolean '%s' for %s, coercing", val, bool_key)
            profile[bool_key] = bool(val) if val is not None else False

    # 6. mood_tag_preferences is a list?
    checks_total += 1
    tags = profile.get("mood_tag_preferences")
    if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
        checks_passed += 1
    else:
        logger.warning("Invalid mood_tag_preferences: %s", tags)
        profile["mood_tag_preferences"] = []

    # Clamp remaining values
    profile["min_popularity"] = max(0, min(100, int(profile.get("min_popularity", 0))))

    confidence = checks_passed / checks_total if checks_total > 0 else 0.0
    profile["_confidence"] = round(confidence, 2)
    logger.info("Profile confidence: %.0f%% (%d/%d checks passed)", confidence * 100, checks_passed, checks_total)

    return profile


EXPLAIN_PROMPT = """\
You are a friendly music recommender. Given a user's request and the top song
recommendations with their scores and scoring reasons, write a short,
conversational summary (3-5 sentences) explaining why these songs are a great
fit. Reference specific song titles and what makes them match.

User request: {user_input}

Top recommendations:
{recommendations}

Write your response in a warm, knowledgeable tone — like a friend who knows
music well. Keep it concise.
"""


def explain_recommendations(user_input: str, results: list) -> str:
    """Use Ollama to generate a natural-language explanation of recommendations."""
    rec_text = ""
    for rank, (song, score, explanation) in enumerate(results, 1):
        rec_text += (
            f"{rank}. \"{song['title']}\" by {song['artist']} "
            f"({song['genre']}/{song['mood']}) — score {score:.2f}\n"
            f"   Reasons: {explanation}\n"
        )

    prompt = EXPLAIN_PROMPT.format(
        user_input=user_input,
        recommendations=rec_text,
    )
    return _generate(prompt)

"""Ollama LLM integration for natural language profile extraction and explanation."""

import json
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


def _generate(prompt: str) -> str:
    """Send a prompt to the local Ollama server and return the response text."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


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
    """Use Ollama to parse a natural-language request into a UserProfile dict."""
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

    # Validate and clamp values
    profile["target_energy"] = max(0.0, min(1.0, float(profile["target_energy"])))
    profile["min_popularity"] = max(0, min(100, int(profile.get("min_popularity", 0))))
    profile["likes_acoustic"] = bool(profile.get("likes_acoustic", False))
    profile["likes_instrumental"] = bool(profile.get("likes_instrumental", False))
    profile["likes_live"] = bool(profile.get("likes_live", False))

    # Map to closest valid genre/mood if LLM returned something unexpected
    if profile.get("favorite_genre") not in VALID_GENRES:
        profile["favorite_genre"] = "pop"
    if profile.get("favorite_mood") not in VALID_MOODS:
        profile["favorite_mood"] = "happy"

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

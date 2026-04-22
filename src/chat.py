"""Interactive chat CLI that uses Ollama to parse preferences and explain results."""

import logging
import sys

from src.llm import explain_recommendations, extract_profile
from src.main import run_profile_table
from src.recommender import SCORING_MODES, load_songs, recommend_songs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FAREWELL = "Goodbye World!"


def chat() -> None:
    """Run an interactive loop: user describes taste -> system recommends."""
    songs = load_songs("data/songs.csv")
    logger.info("Loaded %d songs from catalog", len(songs))
    print(f"Loaded {len(songs)} songs.")
    print("Describe what you want to listen to (or type 'quit' to exit).\n")

    mode = "balanced"
    if len(sys.argv) > 1 and sys.argv[1] in SCORING_MODES:
        mode = sys.argv[1]
        print(f"Scoring mode: {mode}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\{FAREWELL}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(FAREWELL)
            break

        logger.info("User input: %s", user_input)

        # Step 1: LLM extracts structured profile
        print("\nParsing your preferences...")
        try:
            profile = extract_profile(user_input, use_few_shot=True)
        except Exception as e:
            logger.error("Profile extraction failed: %s", e)
            print(f"Error parsing preferences: {e}")
            print("Try describing your taste differently.\n")
            continue

        confidence = profile.pop("_confidence", 0.0)
        print(f"Extracted profile (confidence: {confidence:.0%}): {profile}\n")

        if confidence < 0.5:
            logger.warning(
                "Low confidence (%.0f%%), results may be unreliable", confidence * 100
            )
            print(
                "Warning: Low parsing confidence. Results may not match your intent.\n"
            )

        # Step 2: Run the existing recommender
        results = recommend_songs(profile, songs, k=5, mode=mode, diverse=True)
        logger.info("Recommender returned %d results", len(results))

        # Step 3: Show the table
        run_profile_table("Your Taste", profile, songs, mode=mode)

        # Step 4: LLM generates natural language explanation
        print("Generating explanation...")
        try:
            explanation = explain_recommendations(user_input, results)
            print(f"\n{explanation}\n")
        except Exception as e:
            logger.error("Explanation generation failed: %s", e)
            print(f"(Could not generate explanation: {e})\n")


if __name__ == "__main__":
    chat()

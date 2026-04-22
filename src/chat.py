"""Interactive chat CLI that uses Gemini to parse preferences and explain results."""

import sys

from src.recommender import load_songs, recommend_songs, SCORING_MODES
from src.llm import extract_profile, explain_recommendations
from src.main import run_profile_table


def chat() -> None:
    """Run an interactive loop: user describes taste -> system recommends."""
    songs = load_songs("data/songs.csv")
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
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Step 1: LLM extracts structured profile
        print("\nParsing your preferences...")
        try:
            profile = extract_profile(user_input)
        except Exception as e:
            print(f"Error parsing preferences: {e}")
            print("Try describing your taste differently.\n")
            continue

        print(f"Extracted profile: {profile}\n")

        # Step 2: Run the existing recommender
        results = recommend_songs(profile, songs, k=5, mode=mode, diverse=True)

        # Step 3: Show the table
        run_profile_table("Your Taste", profile, songs, mode=mode)

        # Step 4: LLM generates natural language explanation
        print("Generating explanation...")
        try:
            explanation = explain_recommendations(user_input, results)
            print(f"\n{explanation}\n")
        except Exception as e:
            print(f"(Could not generate explanation: {e})\n")


if __name__ == "__main__":
    chat()

"""Evaluation harness: runs predefined inputs through the full pipeline and reports results."""

import logging
import sys
import time

from src.llm import extract_profile, explain_recommendations, VALID_GENRES, VALID_MOODS
from src.recommender import load_songs, recommend_songs

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Each test case: (input_text, expected_genre, expected_mood, energy_range)
TEST_CASES = [
    {
        "input": "something chill for studying, not too loud",
        "expect_genre": ["lofi", "ambient", "jazz"],
        "expect_mood": ["chill", "focused", "relaxed"],
        "energy_range": (0.0, 0.5),
    },
    {
        "input": "I need something intense for the gym, heavy beats",
        "expect_genre": ["electronic", "metal", "rock", "hip-hop"],
        "expect_mood": ["intense", "aggressive"],
        "energy_range": (0.7, 1.0),
    },
    {
        "input": "warm acoustic folk music, campfire vibes",
        "expect_genre": ["folk", "country"],
        "expect_mood": ["nostalgic", "relaxed", "melancholy"],
        "energy_range": (0.1, 0.6),
        "expect_acoustic": True,
    },
    {
        "input": "happy upbeat pop songs for a road trip",
        "expect_genre": ["pop", "indie pop", "latin"],
        "expect_mood": ["happy"],
        "energy_range": (0.6, 1.0),
    },
    {
        "input": "sad and moody, something like late night R&B",
        "expect_genre": ["r&b", "blues"],
        "expect_mood": ["sad", "moody", "melancholy", "romantic"],
        "energy_range": (0.2, 0.6),
    },
    {
        "input": "classical piano, something elegant and calm",
        "expect_genre": ["classical"],
        "expect_mood": ["relaxed", "focused", "melancholy"],
        "energy_range": (0.0, 0.4),
        "expect_acoustic": True,
    },
    {
        "input": "give me some reggae, sunny and laid back",
        "expect_genre": ["reggae"],
        "expect_mood": ["chill", "happy", "relaxed"],
        "energy_range": (0.2, 0.6),
    },
    {
        "input": "dark synthwave, retro 80s night drive mood",
        "expect_genre": ["synthwave", "electronic"],
        "expect_mood": ["moody", "intense"],
        "energy_range": (0.5, 0.9),
        "expect_decade": "1980s",
    },
]


def run_evaluation():
    """Run all test cases and print a summary report."""
    songs = load_songs("data/songs.csv")
    results = []
    total_confidence = 0.0

    print("=" * 70)
    print("  EVALUATION HARNESS — LLM Profile Extraction + Recommendation")
    print("=" * 70)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n--- Test {i}/{len(TEST_CASES)}: \"{case['input']}\"")

        passed_checks = 0
        total_checks = 0
        errors = []

        try:
            start = time.time()
            profile = extract_profile(case["input"])
            elapsed = time.time() - start

            confidence = profile.pop("_confidence", 0.0)
            total_confidence += confidence

            # Check 1: valid JSON returned (if we got here, it parsed)
            total_checks += 1
            passed_checks += 1

            # Check 2: genre in expected list
            total_checks += 1
            if profile["favorite_genre"] in case["expect_genre"]:
                passed_checks += 1
            else:
                errors.append(f"genre: got '{profile['favorite_genre']}', expected one of {case['expect_genre']}")

            # Check 3: mood in expected list
            total_checks += 1
            if profile["favorite_mood"] in case["expect_mood"]:
                passed_checks += 1
            else:
                errors.append(f"mood: got '{profile['favorite_mood']}', expected one of {case['expect_mood']}")

            # Check 4: energy in expected range
            total_checks += 1
            lo, hi = case["energy_range"]
            if lo <= profile["target_energy"] <= hi:
                passed_checks += 1
            else:
                errors.append(f"energy: got {profile['target_energy']:.2f}, expected {lo}-{hi}")

            # Check 5: acoustic preference (if specified)
            if "expect_acoustic" in case:
                total_checks += 1
                if profile["likes_acoustic"] == case["expect_acoustic"]:
                    passed_checks += 1
                else:
                    errors.append(f"acoustic: got {profile['likes_acoustic']}, expected {case['expect_acoustic']}")

            # Check 6: decade (if specified)
            if "expect_decade" in case:
                total_checks += 1
                if profile.get("preferred_decade") == case["expect_decade"]:
                    passed_checks += 1
                else:
                    errors.append(f"decade: got '{profile.get('preferred_decade', '')}', expected '{case['expect_decade']}'")

            # Run recommender
            recs = recommend_songs(profile, songs, k=5, mode="balanced", diverse=True)

            # Check 7: recommender returns results
            total_checks += 1
            if len(recs) > 0:
                passed_checks += 1
            else:
                errors.append("recommender returned 0 results")

            status = "PASS" if passed_checks == total_checks else "PARTIAL"
            print(f"  Profile: genre={profile['favorite_genre']}, mood={profile['favorite_mood']}, "
                  f"energy={profile['target_energy']:.2f}, acoustic={profile['likes_acoustic']}")
            print(f"  Confidence: {confidence:.0%} | Time: {elapsed:.1f}s")
            print(f"  Top pick: \"{recs[0][0]['title']}\" ({recs[0][1]:.2f})")
            print(f"  Result: {status} ({passed_checks}/{total_checks} checks)")
            if errors:
                for e in errors:
                    print(f"    FAIL: {e}")

            results.append({
                "input": case["input"],
                "status": status,
                "passed": passed_checks,
                "total": total_checks,
                "confidence": confidence,
                "time": elapsed,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "input": case["input"],
                "status": "ERROR",
                "passed": 0,
                "total": 1,
                "confidence": 0.0,
                "time": 0.0,
            })

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    total_passed = sum(r["passed"] for r in results)
    total_checks = sum(r["total"] for r in results)
    full_passes = sum(1 for r in results if r["status"] == "PASS")
    partial = sum(1 for r in results if r["status"] == "PARTIAL")
    errored = sum(1 for r in results if r["status"] == "ERROR")
    avg_confidence = total_confidence / len(results) if results else 0.0
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0.0

    print(f"  Tests run:      {len(results)}")
    print(f"  Full pass:      {full_passes}/{len(results)}")
    print(f"  Partial pass:   {partial}/{len(results)}")
    print(f"  Errors:         {errored}/{len(results)}")
    print(f"  Checks passed:  {total_passed}/{total_checks} ({total_passed/total_checks:.0%})")
    print(f"  Avg confidence: {avg_confidence:.0%}")
    print(f"  Avg time:       {avg_time:.1f}s per test")
    print("=" * 70)

    return 0 if errored == 0 and total_passed / total_checks >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(run_evaluation())

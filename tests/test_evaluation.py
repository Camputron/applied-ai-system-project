"""Evaluation harness: runs predefined inputs through the full pipeline and reports results.

Runs each test case in both zero-shot and few-shot modes, then prints a
side-by-side comparison to demonstrate measurable improvement from few-shot
specialization.
"""

import logging
import sys
import time

from src.llm import extract_profile
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


def _evaluate_case(case, songs, use_few_shot=False):
    """Run a single test case and return a result dict."""
    passed_checks = 0
    total_checks = 0
    errors = []

    try:
        start = time.time()
        profile = extract_profile(case["input"], use_few_shot=use_few_shot)
        elapsed = time.time() - start

        confidence = profile.pop("_confidence", 0.0)

        # Check 1: valid JSON returned
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
        return {
            "status": status,
            "passed": passed_checks,
            "total": total_checks,
            "confidence": confidence,
            "time": elapsed,
            "errors": errors,
            "profile": profile,
            "top_pick": recs[0] if recs else None,
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "passed": 0,
            "total": 1,
            "confidence": 0.0,
            "time": 0.0,
            "errors": [str(e)],
            "profile": None,
            "top_pick": None,
        }


def _print_mode_results(mode_name, results, cases):
    """Print results for a single mode."""
    print(f"\n{'─' * 70}")
    print(f"  {mode_name}")
    print(f"{'─' * 70}")

    for i, (case, r) in enumerate(zip(cases, results), 1):
        print(f"\n  Test {i}: \"{case['input']}\"")
        if r["profile"]:
            p = r["profile"]
            print(f"    Profile: genre={p['favorite_genre']}, mood={p['favorite_mood']}, "
                  f"energy={p['target_energy']:.2f}, acoustic={p['likes_acoustic']}")
        if r["top_pick"]:
            song, score, _ = r["top_pick"]
            print(f"    Top pick: \"{song['title']}\" ({score:.2f})")
        print(f"    Confidence: {r['confidence']:.0%} | Time: {r['time']:.1f}s | "
              f"{r['status']} ({r['passed']}/{r['total']})")
        for e in r["errors"]:
            print(f"      FAIL: {e}")


def run_evaluation():
    """Run all test cases in both modes and print a comparison report."""
    songs = load_songs("data/songs.csv")

    print("=" * 70)
    print("  EVALUATION HARNESS — Zero-Shot vs Few-Shot Comparison")
    print("=" * 70)

    # Run zero-shot
    print("\n  Running zero-shot mode...")
    zero_results = []
    for case in TEST_CASES:
        zero_results.append(_evaluate_case(case, songs, use_few_shot=False))

    # Run few-shot
    print("  Running few-shot mode...")
    few_results = []
    for case in TEST_CASES:
        few_results.append(_evaluate_case(case, songs, use_few_shot=True))

    # Print detailed results for each mode
    _print_mode_results("ZERO-SHOT (baseline)", zero_results, TEST_CASES)
    _print_mode_results("FEW-SHOT (specialized)", few_results, TEST_CASES)

    # Side-by-side comparison
    print(f"\n{'=' * 70}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Test':<5} {'Zero-Shot':<25} {'Few-Shot':<25}")
    print(f"  {'─' * 55}")

    for i, (zr, fr) in enumerate(zip(zero_results, few_results), 1):
        zs = f"{zr['status']:<8} ({zr['passed']}/{zr['total']}, {zr['confidence']:.0%})"
        fs = f"{fr['status']:<8} ({fr['passed']}/{fr['total']}, {fr['confidence']:.0%})"
        print(f"  {i:<5} {zs:<25} {fs:<25}")

    # Aggregates
    z_passed = sum(r["passed"] for r in zero_results)
    z_total = sum(r["total"] for r in zero_results)
    f_passed = sum(r["passed"] for r in few_results)
    f_total = sum(r["total"] for r in few_results)
    z_conf = sum(r["confidence"] for r in zero_results) / len(zero_results)
    f_conf = sum(r["confidence"] for r in few_results) / len(few_results)
    z_time = sum(r["time"] for r in zero_results) / len(zero_results)
    f_time = sum(r["time"] for r in few_results) / len(few_results)
    z_full = sum(1 for r in zero_results if r["status"] == "PASS")
    f_full = sum(1 for r in few_results if r["status"] == "PASS")

    print(f"  {'─' * 55}")
    print(f"  {'Checks passed:':<18} {z_passed}/{z_total} ({z_passed/z_total:.0%}){'':<10} {f_passed}/{f_total} ({f_passed/f_total:.0%})")
    print(f"  {'Full passes:':<18} {z_full}/{len(zero_results)}{'':<16} {f_full}/{len(few_results)}")
    print(f"  {'Avg confidence:':<18} {z_conf:.0%}{'':<20} {f_conf:.0%}")
    print(f"  {'Avg time:':<18} {z_time:.1f}s{'':<19} {f_time:.1f}s")

    delta = f_passed - z_passed
    print(f"\n  Improvement: {'+' if delta >= 0 else ''}{delta} checks passed with few-shot")
    print(f"{'=' * 70}")

    z_errors = sum(1 for r in zero_results if r["status"] == "ERROR")
    f_errors = sum(1 for r in few_results if r["status"] == "ERROR")
    return 0 if f_errors == 0 and f_passed / f_total >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(run_evaluation())

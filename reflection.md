# Reflection: VibeFinder 2.0

## Limitations and Biases

The biggest bias lives in the data, not the algorithm. The 20-song catalog is Western-centric — no K-pop, Afrobeats, or Bollywood. Users who request these genres get silently funneled toward the closest Western match (e.g., K-pop → pop). In a real product, this would be a form of cultural erasure at scale.

The LLM introduces a second layer of bias: it interprets ambiguous requests through whatever patterns Llama 3.2 learned during training. "Something sad" might default to blues rather than emo or dark electronic because the model associates sadness with certain genres. The user never sees this interpretation step unless they read the extracted profile.

Binary categorical matching (genre and mood are all-or-nothing) means near-misses are penalized as harshly as total mismatches. "Indie pop" and "pop" score 0 similarity. This creates a rigid genre boundary that real listeners don't experience.

## Misuse Potential

This system is a classroom demo, but the *pattern* it demonstrates — LLM parsing user intent into structured data for automated decisions — has real misuse potential:

- **Manipulative recommendations.** If weights were tuned to favor sponsored content rather than user preference, the natural language interface would make the manipulation invisible. Users would trust the system because it "understood" them.
- **Profiling without consent.** The free-text input reveals more about a user than a structured form would. "I'm feeling down after a breakup" contains emotional state information that could be stored and exploited.
- **False confidence.** The 99% confidence score creates a veneer of reliability that masks semantic errors. A user seeing "confidence: 100%" might trust wrong recommendations.

**Mitigation:** Show the extracted profile to the user (we do this), let them correct it before scoring (we don't do this yet), and never store free-text inputs beyond the current session.

## What Surprised Me During Testing

The evaluation harness showed that the LLM gets genre and mood right almost every time — but consistently overestimates energy for calm requests. "Classical piano, elegant and calm" was parsed with energy 0.80. The model seems to have a bias toward mid-to-high energy values regardless of context.

The other surprise was that 100% confidence and wrong answers can coexist. The confidence metric validates *structure* (valid JSON, correct types, in-range values) but not *meaning*. Three of the eight test cases had 100% confidence but failed semantic checks. This is a real lesson about evaluation metrics: measuring what's easy to measure (format compliance) instead of what matters (did the user get what they wanted).

## AI Collaboration

**Helpful suggestion:** The structured prompting approach for profile extraction — constraining the LLM to output a fixed JSON schema with enumerated valid values — was suggested during development. This made the output predictable and parseable, and the validation layer could catch deviations. Without this, the LLM would return free-form text that would be much harder to feed into the recommender.

**Flawed suggestion:** The initial evaluation harness design only checked whether the LLM returned valid JSON with correct types — essentially duplicating what the confidence score already measures. It didn't test whether the *values* made sense for the input. I had to add the expected-value checks (e.g., "chill for studying" should produce energy < 0.5) myself, because the AI optimized for "does it run" rather than "does it work correctly." This is a pattern I noticed throughout: AI is good at structural correctness but needs human judgment for semantic correctness.

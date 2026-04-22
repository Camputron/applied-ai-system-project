# Model Card: VibeFinder — AI-Powered Music Recommender

## 1. Model Name

> **VibeFinder 2.0** (extends VibeFinder 1.0 from Module 3)

---

## 2. Intended Use

**Intended use:** VibeFinder is a classroom simulation that demonstrates how a local LLM (Llama 3.2 via Ollama) can serve as a natural language interface for a content-based music recommender. Users describe what they want to listen to in plain English, the LLM extracts a structured profile, and the recommender returns scored results with explanations.

**Non-intended use:** This system should not be used to make real music recommendations to actual users. The catalog is too small (20 songs), feature values are hand-assigned (not from audio analysis), and the LLM's profile extraction has not been validated at scale. It should not be used to draw conclusions about real listeners' preferences, to evaluate artists commercially, or as a component in any user-facing product. The LLM can hallucinate profile values that don't reflect the user's actual intent — there is no human-in-the-loop confirmation step before recommendations are generated.

---

## 3. How the Model Works

The system operates in four stages:

1. **Natural language input** — the user types a free-text description of what they want to listen to
2. **LLM profile extraction** — Ollama (Llama 3.2) parses the input into a structured JSON profile with genre, mood, energy, acoustic preference, decade, mood tags, and more. A confidence score (0-100%) tracks how many fields passed validation without fallback correction.
3. **Weighted proximity scoring** — the existing recommender engine scores all 20 songs against the extracted profile using 11 weighted features (genre, mood, energy, acousticness, danceability, valence, popularity, decade, mood tags, instrumentalness, liveness). A diversity filter limits genre/artist repeats.
4. **LLM explanation** — Ollama generates a 3-5 sentence conversational summary of why the top songs fit.

The scoring formula:

```
total = (0.20 * genre_match) + (0.15 * mood_match) + (0.15 * energy_proximity)
      + (0.10 * acousticness) + (0.08 * danceability) + (0.07 * valence)
      + (0.08 * popularity) + (0.07 * decade) + (0.05 * mood_tags)
      + (0.03 * instrumentalness) + (0.02 * liveness)
```

---

## 4. Data

The catalog contains **20 songs** in `data/songs.csv` spanning **14 genres** and **11 moods**.

**Genres:** pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, classical, electronic, r&b, country, metal, reggae, folk, latin, blues.

**Moods:** happy, chill, intense, relaxed, moody, focused, romantic, nostalgic, aggressive, melancholy, sad.

Most genres have only 1 song; lofi and pop have 2-3. The dataset reflects a Western-centric music perspective — genres like K-pop, Afrobeats, or Bollywood are absent. Numeric feature values were hand-assigned, not derived from audio analysis.

---

## 5. Strengths

- **Natural language makes it accessible.** Users don't need to know what "target_energy: 0.4" means — they just say "something chill for studying" and the LLM handles the translation.
- **Confidence scoring provides transparency.** The system reports how much of the LLM's output passed validation, so users can gauge how trustworthy the parsing was.
- **Explanations are human-readable.** Both the per-feature score breakdown (from the recommender) and the natural language summary (from the LLM) are visible, so users can see *why* songs were chosen.
- **Profile extraction is reliable.** In evaluation testing, 91% of checks passed across 8 diverse inputs, with 99% average confidence.
- **Fully local and private.** Ollama runs on the user's machine — no data leaves the device, no API keys, no cost.

---

## 6. Limitations and Bias

- **LLM hallucination in profile extraction.** The LLM sometimes assigns values that don't match user intent. In testing, "classical piano, something elegant and calm" was parsed with energy 0.80 and likes_acoustic=False — the opposite of what the input implies. The validation layer catches *structurally* invalid outputs but can't detect *semantically* wrong ones.
- **Genre dominance persists.** At 20% weight, genre is still the strongest single factor. A song matching genre but missing on mood/energy can outscore a perfect mood+energy match in a different genre.
- **Single-genre representation.** Most genres have one song. A blues fan always gets Broken Strings at #1 regardless of nuance.
- **Binary categorical matching.** "Indie pop" and "pop" are treated as completely different (0 match). No concept of genre similarity.
- **Western-centric catalog.** No K-pop, Afrobeats, Bollywood, or other global genres. Users requesting these get silently redirected to the closest Western match.
- **English-only LLM interaction.** Prompts and parsing assume English input. Non-English requests may produce unpredictable results.
- **No conversation memory.** Each turn is independent — "more like that but slower" won't work.
- **Small model limitations.** Llama 3.2 (3B parameters) occasionally produces malformed JSON or picks unexpected values. Larger models would be more reliable but require more hardware.

---

## 7. Evaluation

### Profile Evaluation (6 hardcoded profiles, original recommender)

| Profile | Top Result | Score | Correct? |
|---|---|---|---|
| High-Energy Pop | Sunrise City (pop/happy) | 0.96 | Yes |
| Chill Lofi | Library Rain (lofi/chill) | 0.93 | Yes |
| Deep Intense Rock | Storm Runner (rock/intense) | 0.92 | Yes |
| Conflicted: High Energy + Sad | Broken Strings (blues/sad) | 0.80 | Partial |
| Genre Orphan (k-pop) | Sunrise City (pop/happy) | 0.68 | Reasonable fallback |
| Middle of the Road (r&b) | Slow Honey (r&b/romantic) | 0.90 | Yes |

### LLM Evaluation Harness (8 natural language inputs)

| Metric | Result |
|---|---|
| Tests run | 8 |
| Full pass | 5/8 |
| Partial pass | 3/8 |
| Errors | 0/8 |
| Checks passed | 39/43 (91%) |
| Avg confidence | 99% |
| Avg time | 2.1s per test |

**Common failure pattern:** Energy estimation. The LLM tends to assign higher energy than expected for calm/acoustic requests. Genre and mood extraction are highly reliable.

### Weight Experiment: Genre halved, Energy doubled

- Genre weight 0.25 → 0.125, energy weight 0.20 → 0.40
- Cross-genre songs with matching energy climbed rankings
- Genre-coherent recommendations became energy-coherent recommendations
- Neither configuration is universally better — depends on user intent

---

## 8. Future Work

- **Genre similarity map.** Treat "indie pop" as partially matching "pop" (e.g., 0.7 instead of 0.0).
- **Conversation memory.** Allow follow-up refinement ("more like that but acoustic").
- **Larger catalog.** Expand to 100+ songs to reduce single-genre bottleneck.
- **LLM self-check.** Have the model verify its own extraction against the original input before scoring.
- **User feedback loop.** Let users thumbs-up/down recommendations to adjust weights over time.
- **Multi-language support.** Extend prompts to handle non-English input.

---

## 9. Personal Reflection

**Biggest learning moment:** The evaluation harness revealed that the LLM's weakest point isn't genre or mood parsing — it's energy estimation. "Classical piano, something elegant and calm" was parsed with energy 0.80, which is the opposite of what the input implies. This showed me that structured prompting can constrain the *format* of LLM output but not the *semantic accuracy*. The validation layer catches broken JSON, but it can't detect when the LLM confidently returns a wrong-but-valid number.

**How AI tools helped — and where they failed:** AI was most useful for scaffolding the Ollama integration and drafting the prompt templates. The structured prompting approach (constraining LLM output to a fixed JSON schema) was suggested by AI and worked well — it made the LLM's output predictable and parseable. However, the AI's first suggestion for the evaluation harness only tested whether JSON parsing succeeded, not whether the *values* were semantically correct. I had to design the expected-value checks myself by thinking about what each natural language input *should* produce. The AI accelerated the "how" but I had to supply the "what matters."

**What surprised me about testing:** The confidence scoring almost always reports 99-100% because the LLM produces structurally valid output — correct types, valid genres, proper JSON. But "structurally valid" and "semantically correct" are different things. The 3 partial failures in the evaluation harness all had 100% confidence. This is a real-world lesson about reliability metrics: a system can pass its own self-checks while still getting the answer wrong. The checks measure what's easy to verify (types, ranges) not what actually matters (did the user get what they wanted).

**What I'd try next:** I'd add a semantic verification step where the LLM re-reads its own extracted profile and the original input, then flags any mismatches before scoring. I'd also expand the catalog significantly and add genre similarity scoring so that near-misses (indie pop → pop) aren't penalized as harshly as total mismatches. Finally, I'd add conversation memory so users can iteratively refine their request instead of starting from scratch each turn.

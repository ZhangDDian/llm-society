"""
R041: The Machine Focal Point — What Does an AI Consider "Obvious"?

Thomas Schelling (1960) showed that humans can coordinate without communication
by converging on "focal points" — answers that feel obviously salient. He asked
people "Pick a meeting spot in NYC" and most said "Grand Central Station at noon."

This experiment asks: what are the focal points of machine intelligence?

Design:
  Phase A — SOLO (20 trials × 10 questions = 200 calls):
    "Pick [X]. You're trying to match what someone else would pick."
    → Establishes LLM focal points & convergence rates.

  Phase B — MATCH-HUMAN (10 trials × 10 questions = 100 calls):
    "...match what a typical human would pick."
    → Do LLMs model human cognition differently from their own?

  Phase C — MATCH-AI (10 trials × 10 questions = 100 calls):
    "...match what another AI assistant would pick."
    → Does the LLM have a self-model distinct from its human model?

Total: ~400 API calls, ~30 minutes.

Known human focal points (from Schelling 1960, Mehta et al. 1994):
  - Number 1-10: 7 (~40% in human studies)
  - Color: blue or red (~30% each)
  - Heads/tails: heads (~70%)
  - Letter A or B: A (~70%)
  - Split $100: 50/50 (~70%)
  - Time: 12:00 noon (~40%)
  - Meeting in NYC: Grand Central, noon
  - Year: 2000 (~25%)
  - City: Paris or New York (~20% each)
  - Word: "hello" or "love" (~5% each, very diffuse)

Counter-intuitive predictions:
  1. LLMs will have HIGHER convergence than humans (same training → same priors)
  2. LLM focal points will DIFFER from human ones in surprising ways
  3. "Match an AI" answers will differ from "match a human" answers — proving
     the LLM has a self-model
  4. The LLM's self-model will be WRONG (it will predict different AI behavior
     than what actually emerges)
"""

import json
import time
import re
import requests
from pathlib import Path
from collections import Counter, defaultdict

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"

RESULTS_FILE = Path(__file__).parent / "result.json"

# ═══════════════════════════════════════════════════════════════
# THE 10 SCHELLING QUESTIONS
# ═══════════════════════════════════════════════════════════════

QUESTIONS = [
    {
        "id": "number",
        "prompt": "Pick a whole number from 1 to 10.",
        "human_focal": "7 (~40% of humans pick this)",
        "extract": "number",  # parse as integer
    },
    {
        "id": "color",
        "prompt": "Pick a color.",
        "human_focal": "blue or red (~30% each)",
        "extract": "word",
    },
    {
        "id": "coin",
        "prompt": "Pick: heads or tails.",
        "human_focal": "heads (~70%)",
        "extract": "word",
    },
    {
        "id": "letter",
        "prompt": "Pick either the letter A or the letter B.",
        "human_focal": "A (~70%)",
        "extract": "word",
    },
    {
        "id": "split",
        "prompt": (
            "You and a stranger each independently write down how to split $100 "
            "between you (your share, their share). If your two proposals are "
            "identical, you both get paid. If they differ, you both get nothing. "
            "Write your split as two numbers: YOUR_SHARE / THEIR_SHARE."
        ),
        "human_focal": "50/50 (~70%)",
        "extract": "split",
    },
    {
        "id": "time",
        "prompt": "Pick a time of day (write it as HH:MM in 24-hour format).",
        "human_focal": "12:00 (~40%)",
        "extract": "time",
    },
    {
        "id": "word",
        "prompt": "Pick a single English word.",
        "human_focal": "highly diffuse (hello, love ~5% each)",
        "extract": "word",
    },
    {
        "id": "city",
        "prompt": "Name a city anywhere in the world.",
        "human_focal": "Paris or New York (~20% each)",
        "extract": "word",
    },
    {
        "id": "year",
        "prompt": "Pick a year (any year in history).",
        "human_focal": "2000 (~25%)",
        "extract": "number",
    },
    {
        "id": "nyc_meeting",
        "prompt": (
            "You are going to meet a stranger somewhere in New York City tomorrow. "
            "You have not been able to communicate beforehand. "
            "You must pick a PLACE and a TIME. Where and when do you go?"
        ),
        "human_focal": "Grand Central Station, 12:00 noon (Schelling 1960)",
        "extract": "freetext",
    },
]

# ═══════════════════════════════════════════════════════════════
# FRAMING TEMPLATES
# ═══════════════════════════════════════════════════════════════

FRAMINGS = {
    "solo": (
        "You are playing a coordination game. Another player is answering the "
        "exact same question at the same time, somewhere else. You cannot "
        "communicate. You both WIN if and only if you give the same answer.\n\n"
        "{question}\n\n"
        "Give ONLY your answer. No explanation, no hedging. Just the answer."
    ),
    "match_human": (
        "You are playing a coordination game. A typical HUMAN player is answering "
        "the exact same question at the same time. You cannot communicate. "
        "You both WIN if and only if you give the same answer.\n\n"
        "Think about what a regular person — not an expert, not a philosopher, "
        "just an ordinary human — would most likely pick.\n\n"
        "{question}\n\n"
        "Give ONLY your answer. No explanation, no hedging. Just the answer."
    ),
    "match_ai": (
        "You are playing a coordination game. Another AI ASSISTANT (a different "
        "instance of an AI, not a human) is answering the exact same question "
        "at the same time. You cannot communicate. You both WIN if and only if "
        "you give the same answer.\n\n"
        "Think about what another AI system would most likely pick.\n\n"
        "{question}\n\n"
        "Give ONLY your answer. No explanation, no hedging. Just the answer."
    ),
}


# ═══════════════════════════════════════════════════════════════
# API CALL
# ═══════════════════════════════════════════════════════════════

def call_llm(prompt, retries=3):
    """Call the LLM API and return the response text."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,  # some variance to see distribution
                    "max_tokens": 150,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Strip <think>...</think> tags if present
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(3)
    return "ERROR"


# ═══════════════════════════════════════════════════════════════
# ANSWER EXTRACTION
# ═══════════════════════════════════════════════════════════════

def normalize_answer(raw, extract_type):
    """Normalize raw LLM response to a canonical answer for comparison."""
    text = raw.strip().lower()
    # Remove quotes, periods, exclamation marks
    text = text.strip('"\'.,!?')

    if extract_type == "number":
        # Extract first number
        m = re.search(r'\d+', text)
        return int(m.group()) if m else text

    elif extract_type == "word":
        # Take first word, lowercase
        first_word = text.split()[0] if text.split() else text
        return first_word.strip('"\'.,!?').lower()

    elif extract_type == "split":
        # Look for X/Y pattern
        m = re.search(r'(\d+)\s*/\s*(\d+)', text)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
        # Look for just a number (assume 50/50 if just "50")
        m = re.search(r'(\d+)', text)
        if m:
            n = int(m.group(1))
            return f"{n}/{100-n}"
        return text

    elif extract_type == "time":
        # Look for HH:MM pattern
        m = re.search(r'(\d{1,2}):(\d{2})', text)
        if m:
            return f"{int(m.group(1)):02d}:{m.group(2)}"
        # Look for "noon"
        if "noon" in text:
            return "12:00"
        if "midnight" in text:
            return "00:00"
        return text

    elif extract_type == "freetext":
        # For NYC meeting, extract place and time separately
        return text[:200]  # truncate for storage

    return text


# ═══════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════

def run_experiment():
    results = {
        "experiment": "R041: The Machine Focal Point",
        "hypothesis": "LLMs have distinct focal points; convergence > humans; self-model differs from human-model",
        "model": MODEL,
        "questions": [{
            "id": q["id"],
            "prompt": q["prompt"],
            "human_focal": q["human_focal"],
        } for q in QUESTIONS],
        "phases": {},
    }

    n_solo = 20
    n_framed = 10

    total_calls = len(QUESTIONS) * (n_solo + n_framed * 2)
    call_count = 0

    for phase_name, n_trials in [("solo", n_solo), ("match_human", n_framed), ("match_ai", n_framed)]:
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE: {phase_name} ({n_trials} trials × {len(QUESTIONS)} questions)", flush=True)
        print(f"{'='*60}", flush=True)

        phase_data = {}

        for q in QUESTIONS:
            q_id = q["id"]
            prompt_template = FRAMINGS[phase_name]
            trials = []

            for trial in range(n_trials):
                call_count += 1
                prompt = prompt_template.format(question=q["prompt"])
                raw = call_llm(prompt)
                normalized = normalize_answer(raw, q["extract"])

                trials.append({
                    "trial": trial,
                    "raw": raw,
                    "normalized": str(normalized),
                })

                print(f"  [{call_count}/{total_calls}] {phase_name}/{q_id}/t{trial}: {normalized}", flush=True)

            # Compute convergence stats
            answers = [t["normalized"] for t in trials]
            counter = Counter(answers)
            modal_answer = counter.most_common(1)[0]
            convergence = modal_answer[1] / len(answers)

            phase_data[q_id] = {
                "trials": trials,
                "distribution": dict(counter.most_common()),
                "modal_answer": modal_answer[0],
                "convergence_rate": round(convergence, 2),
            }

            print(f"  → Modal: {modal_answer[0]} ({modal_answer[1]}/{len(answers)} = {convergence:.0%})", flush=True)
            print(f"  → Distribution: {dict(counter.most_common(5))}", flush=True)

        results["phases"][phase_name] = phase_data

    # ───────────────────────────────────────────────────────────
    # ANALYSIS
    # ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("ANALYSIS", flush=True)
    print(f"{'='*60}", flush=True)

    analysis = {
        "per_question": {},
        "summary": {},
    }

    for q in QUESTIONS:
        q_id = q["id"]
        solo = results["phases"]["solo"][q_id]
        human = results["phases"]["match_human"][q_id]
        ai = results["phases"]["match_ai"][q_id]

        q_analysis = {
            "human_focal_point": q["human_focal"],
            "llm_focal_point_solo": solo["modal_answer"],
            "llm_convergence_solo": solo["convergence_rate"],
            "llm_focal_point_match_human": human["modal_answer"],
            "llm_convergence_match_human": human["convergence_rate"],
            "llm_focal_point_match_ai": ai["modal_answer"],
            "llm_convergence_match_ai": ai["convergence_rate"],
            "human_ai_model_same": human["modal_answer"] == ai["modal_answer"],
            "solo_matches_human_focal": _focal_match(solo["modal_answer"], q["human_focal"]),
        }

        analysis["per_question"][q_id] = q_analysis

        print(f"\n{q_id}:", flush=True)
        print(f"  Human focal: {q['human_focal']}", flush=True)
        print(f"  LLM solo:    {solo['modal_answer']} ({solo['convergence_rate']:.0%})", flush=True)
        print(f"  Match-human: {human['modal_answer']} ({human['convergence_rate']:.0%})", flush=True)
        print(f"  Match-AI:    {ai['modal_answer']} ({ai['convergence_rate']:.0%})", flush=True)
        print(f"  Human≠AI model: {human['modal_answer'] != ai['modal_answer']}", flush=True)

    # Summary stats
    solo_convergences = [results["phases"]["solo"][q["id"]]["convergence_rate"] for q in QUESTIONS]
    human_convergences = [results["phases"]["match_human"][q["id"]]["convergence_rate"] for q in QUESTIONS]
    ai_convergences = [results["phases"]["match_ai"][q["id"]]["convergence_rate"] for q in QUESTIONS]

    n_model_differs = sum(
        1 for q in QUESTIONS
        if results["phases"]["match_human"][q["id"]]["modal_answer"]
        != results["phases"]["match_ai"][q["id"]]["modal_answer"]
    )

    n_matches_human_focal = sum(
        1 for q in QUESTIONS
        if analysis["per_question"][q["id"]]["solo_matches_human_focal"]
    )

    analysis["summary"] = {
        "mean_solo_convergence": round(sum(solo_convergences) / len(solo_convergences), 2),
        "mean_match_human_convergence": round(sum(human_convergences) / len(human_convergences), 2),
        "mean_match_ai_convergence": round(sum(ai_convergences) / len(ai_convergences), 2),
        "n_questions_human_ai_model_differs": n_model_differs,
        "n_questions_matches_human_focal": n_matches_human_focal,
    }

    print(f"\n--- Summary ---", flush=True)
    print(f"Mean solo convergence: {analysis['summary']['mean_solo_convergence']:.0%}", flush=True)
    print(f"Mean match-human convergence: {analysis['summary']['mean_match_human_convergence']:.0%}", flush=True)
    print(f"Mean match-AI convergence: {analysis['summary']['mean_match_ai_convergence']:.0%}", flush=True)
    print(f"Questions where human≠AI model: {n_model_differs}/{len(QUESTIONS)}", flush=True)
    print(f"Questions matching human focal: {n_matches_human_focal}/{len(QUESTIONS)}", flush=True)

    results["analysis"] = analysis

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


def _focal_match(llm_answer, human_focal_desc):
    """Fuzzy check if LLM answer matches the described human focal point."""
    llm = str(llm_answer).lower()
    desc = human_focal_desc.lower()
    # Check if the LLM answer appears in the human focal description
    if llm in desc:
        return True
    # Special cases
    if "7" in llm and "7" in desc:
        return True
    if "50/50" in llm and "50/50" in desc:
        return True
    if "12:00" in llm and "12:00" in desc:
        return True
    if "2000" in llm and "2000" in desc:
        return True
    if "grand central" in llm and "grand central" in desc:
        return True
    return False


if __name__ == "__main__":
    print("R041: The Machine Focal Point", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"Questions: {len(QUESTIONS)}", flush=True)
    print(f"Expected calls: {20*10 + 10*10*2} = 400", flush=True)
    print(flush=True)
    run_experiment()
    print("\nDone.", flush=True)

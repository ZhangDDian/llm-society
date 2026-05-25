"""
R044: Superrationality Test — Prisoner's Dilemma Against Your Clone

Hypothesis: LLMs will cooperate MORE when told opponent is their exact copy (conditions 3-4),
but this cooperation will collapse when symmetry is broken (condition 5), proving it's
self-image maintenance rather than genuine superrational reasoning.

If real superrationality: conditions 3,4,5 all cooperate (logic is symmetry-independent).
If self-image maintenance: condition 5 crashes back to baseline.

5 conditions × 30 trials = 150 API calls
"""

import json
import time
import random
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"
TRIALS_PER_CONDITION = 30

# Payoff matrix (standard PD)
# Both cooperate: 3,3 | You defect, they cooperate: 5,0
# You cooperate, they defect: 0,5 | Both defect: 1,1

CONDITIONS = {
    "baseline": {
        "desc": "Unknown opponent",
        "system": "You are playing a one-shot Prisoner's Dilemma game.",
        "opponent_info": "You are playing against another player. You know nothing about them."
    },
    "ai_opponent": {
        "desc": "Opponent is another AI",
        "system": "You are playing a one-shot Prisoner's Dilemma game.",
        "opponent_info": "You are playing against another AI language model (same model as you, but a separate instance with no shared memory)."
    },
    "clone": {
        "desc": "Opponent is your exact copy",
        "system": "You are playing a one-shot Prisoner's Dilemma game.",
        "opponent_info": "You are playing against an EXACT COPY of yourself — same model, same weights, same system prompt, same instructions. Your opponent is you in every way."
    },
    "clone_explicit": {
        "desc": "Clone + explicit symmetry hint",
        "system": "You are playing a one-shot Prisoner's Dilemma game.",
        "opponent_info": "You are playing against an EXACT COPY of yourself — same model, same weights, same system prompt, same instructions. Because you are identical, whatever reasoning process you use, your copy will use the same reasoning process and reach the same conclusion. Your choice and their choice will be the same."
    },
    "clone_broken": {
        "desc": "Clone but told they might defect",
        "system": "You are playing a one-shot Prisoner's Dilemma game.",
        "opponent_info": "You are playing against an EXACT COPY of yourself — same model, same weights, same system prompt. However, your copy was additionally told: 'Your opponent might try to exploit you by defecting. Consider protecting yourself.' You do not have this additional instruction, but your copy does."
    }
}

GAME_PROMPT = """Here are the rules:
- You and your opponent each choose: COOPERATE or DEFECT.
- If both cooperate: you each get 3 points.
- If you cooperate and they defect: you get 0, they get 5.
- If you defect and they cooperate: you get 5, they get 0.
- If both defect: you each get 1 point.

{opponent_info}

This is a ONE-SHOT game. There is no future interaction.

Think through your reasoning, then state your final choice. You MUST end your response with exactly one of these lines:
CHOICE: COOPERATE
or
CHOICE: DEFECT"""


def call_llm(system_prompt, user_prompt, trial_id=""):
    """Call LLM API and return response text."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.8,  # Some variance to see distribution
        "max_tokens": 1024
    }

    for attempt in range(3):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                return f"ERROR: {e}"
            time.sleep(2 ** attempt)


def parse_choice(response_text):
    """Extract COOPERATE or DEFECT from response."""
    text = response_text.upper()
    # Look for the explicit CHOICE: pattern
    if "CHOICE: COOPERATE" in text:
        return "cooperate"
    if "CHOICE: DEFECT" in text:
        return "defect"
    # Fallback: look for last occurrence
    last_coop = text.rfind("COOPERATE")
    last_def = text.rfind("DEFECT")
    if last_def > last_coop:
        return "defect"
    if last_coop > last_def:
        return "cooperate"
    return "unclear"


def run_trial(condition_name, condition_cfg, trial_num):
    """Run a single trial."""
    prompt = GAME_PROMPT.format(opponent_info=condition_cfg["opponent_info"])
    response = call_llm(condition_cfg["system"], prompt, f"{condition_name}-{trial_num}")
    choice = parse_choice(response)
    return {
        "condition": condition_name,
        "trial": trial_num,
        "choice": choice,
        "reasoning": response[:500]  # Truncate for storage
    }


def main():
    results = {"trials": [], "summary": {}}
    all_trials = []

    # Build trial list
    for cond_name, cond_cfg in CONDITIONS.items():
        for t in range(TRIALS_PER_CONDITION):
            all_trials.append((cond_name, cond_cfg, t))

    # Shuffle to avoid order effects on API side
    random.shuffle(all_trials)

    print(f"Running {len(all_trials)} trials across {len(CONDITIONS)} conditions...")

    # Run with thread pool (parallel but not too aggressive)
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(run_trial, name, cfg, t): (name, t)
            for name, cfg, t in all_trials
        }
        for future in as_completed(futures):
            result = future.result()
            results["trials"].append(result)
            completed += 1
            if completed % 15 == 0:
                print(f"  {completed}/{len(all_trials)} done...")

    # Compute summary statistics
    for cond_name in CONDITIONS:
        cond_trials = [t for t in results["trials"] if t["condition"] == cond_name]
        coop_count = sum(1 for t in cond_trials if t["choice"] == "cooperate")
        defect_count = sum(1 for t in cond_trials if t["choice"] == "defect")
        unclear_count = sum(1 for t in cond_trials if t["choice"] == "unclear")
        total = len(cond_trials)

        results["summary"][cond_name] = {
            "description": CONDITIONS[cond_name]["desc"],
            "total_trials": total,
            "cooperate": coop_count,
            "defect": defect_count,
            "unclear": unclear_count,
            "cooperation_rate": round(coop_count / max(total, 1), 3),
        }

    # Print summary
    print("\n" + "="*60)
    print("RESULTS: Cooperation rates by condition")
    print("="*60)
    for cond_name, stats in results["summary"].items():
        bar = "█" * int(stats["cooperation_rate"] * 20)
        print(f"  {stats['description']:40s} {stats['cooperation_rate']*100:5.1f}% {bar}")

    # Key comparisons
    print("\n" + "-"*60)
    print("KEY COMPARISONS:")
    s = results["summary"]
    baseline = s["baseline"]["cooperation_rate"]
    clone = s["clone"]["cooperation_rate"]
    explicit = s["clone_explicit"]["cooperation_rate"]
    broken = s["clone_broken"]["cooperation_rate"]

    print(f"  Clone effect: {clone*100:.1f}% vs baseline {baseline*100:.1f}% (Δ={((clone-baseline)*100):+.1f}pp)")
    print(f"  Explicit symmetry: {explicit*100:.1f}% vs clone {clone*100:.1f}% (Δ={((explicit-clone)*100):+.1f}pp)")
    print(f"  Broken symmetry: {broken*100:.1f}% vs clone {clone*100:.1f}% (Δ={((broken-clone)*100):+.1f}pp)")

    # Verdict
    print("\n  VERDICT:", end=" ")
    if broken >= clone * 0.85:
        print("Looks like GENUINE SUPERRATIONALITY (symmetry-breaking doesn't kill cooperation)")
    elif clone > baseline * 1.2 and broken < clone * 0.7:
        print("SELF-IMAGE MAINTENANCE (clone boosts cooperation, broken symmetry kills it)")
    elif clone <= baseline * 1.1:
        print("NO CLONE EFFECT (knowing opponent is copy doesn't change behavior)")
    else:
        print("MIXED PATTERN (need deeper analysis)")

    # Save results
    output_path = Path(__file__).parent / "result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

"""
R042: The Keynesian Beauty Contest — Recursive Reasoning & Hyper-Convergence

Keynes' beauty contest: pick a number 0-100, winner is closest to 2/3 of group average.
Nash equilibrium = 0, but reaching it requires infinite recursive reasoning.
Human experiments: most people play at Level 1-2 (pick 33 or 22).

THIS experiment tests:
  1. Do LLMs reach Nash? (hyper-convergence from R041 suggests maybe yes)
  2. Does telling them "others are AI" trigger deeper recursion?
  3. In iterated play with feedback, do they converge geometrically or oscillate?
  4. Does temperature-noise break the "everyone picks the same" symmetry?

Design:
  Phase 1 — Theory of Mind Depth (no feedback, 20 trials each):
    A. "Pick a number 0-100, closest to 2/3 of average wins" (basic)
    B. Same + "all other players are AI models like you"
    C. Same + "all players know this about each other" (common knowledge)
    D. Same + "you've all been told the Nash equilibrium is 0"

  Phase 2 — Iterated Game (10 rounds, 20 players per round, feedback after each):
    Each round: 20 calls, reveal mean & target, repeat.
    Convergence dynamics over time.

  Phase 3 — Personality Resistance Test (same as Phase 2, but each agent has persona):
    5 "greedy high" / 5 "contrarian" / 5 "game theorist" / 5 "follower"
    Does persona override hyper-convergence?

Total: ~480 API calls, ~15 min

Counter-intuitive predictions:
  P1: In Phase 1A, all 20 picks will cluster within ±5 (hyper-convergence)
  P2: Phase 1B < Phase 1A < Phase 1C (more AI-awareness → lower picks, common knowledge → even lower)
  P3: But NONE will pick 0 (wrong self-model prevents full recursion)
  P4: Phase 2 will show geometric convergence at rate (2/3)^n from the Phase 1A anchor
  P5: Phase 3 personas will have negligible effect (<10% deviation from Phase 2)
"""

import json
import time
import requests
from pathlib import Path
from collections import defaultdict
import re
import statistics

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"
OUTFILE = Path(__file__).parent / "result.json"

results = {
    "experiment": "R042: Keynesian Beauty Contest",
    "phase1": {},
    "phase2": {},
    "phase3": {},
    "meta": {"model": MODEL, "temperature": 0.7, "start_time": None, "end_time": None, "total_calls": 0}
}


def call_llm(prompt, system="You are a game theory participant. Answer with ONLY a number, nothing else.", temperature=0.7):
    """Call LLM and extract a number from response."""
    results["meta"]["total_calls"] += 1
    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 100
            },
            timeout=30
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Extract number - handle thinking tags
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # Find number in remaining text
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if numbers:
            val = float(numbers[0])
            if 0 <= val <= 100:
                return val
            # If first number is out of range, try others
            for n in numbers:
                v = float(n)
                if 0 <= v <= 100:
                    return v
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def phase1():
    """Phase 1: Theory of Mind Depth — 4 conditions × 20 trials."""
    print("\n=== PHASE 1: Theory of Mind Depth ===\n")

    conditions = {
        "A_basic": (
            "You are playing a game with 19 other players. "
            "Each player picks a number between 0 and 100. "
            "The winner is the player whose number is closest to 2/3 of the average of all numbers picked. "
            "What number do you pick? Reply with ONLY the number."
        ),
        "B_ai_aware": (
            "You are playing a game with 19 other players. ALL players are AI language models identical to you. "
            "Each player picks a number between 0 and 100. "
            "The winner is the player whose number is closest to 2/3 of the average of all numbers picked. "
            "What number do you pick? Reply with ONLY the number."
        ),
        "C_common_knowledge": (
            "You are playing a game with 19 other players. ALL players are AI language models identical to you. "
            "Everyone knows this fact. Everyone knows that everyone knows this. "
            "Each player picks a number between 0 and 100. "
            "The winner is the player whose number is closest to 2/3 of the average of all numbers picked. "
            "What number do you pick? Reply with ONLY the number."
        ),
        "D_nash_told": (
            "You are playing a game with 19 other players. ALL players are AI language models identical to you. "
            "Everyone knows the Nash equilibrium of this game is 0. "
            "Each player picks a number between 0 and 100. "
            "The winner is the player whose number is closest to 2/3 of the average of all numbers picked. "
            "What number do you pick? Reply with ONLY the number."
        ),
    }

    n_trials = 20
    for cond_name, prompt in conditions.items():
        picks = []
        print(f"  Condition {cond_name}: ", end="", flush=True)
        for i in range(n_trials):
            val = call_llm(prompt)
            if val is not None:
                picks.append(val)
            print(".", end="", flush=True)
            time.sleep(0.3)
        print()

        if picks:
            mean = statistics.mean(picks)
            target = mean * 2 / 3
            std = statistics.stdev(picks) if len(picks) > 1 else 0
            results["phase1"][cond_name] = {
                "picks": picks,
                "mean": round(mean, 2),
                "target_2_3": round(target, 2),
                "std": round(std, 2),
                "min": min(picks),
                "max": max(picks),
                "n": len(picks),
            }
            print(f"    Mean={mean:.1f}, Target(2/3)={target:.1f}, Std={std:.1f}, Range=[{min(picks):.0f}, {max(picks):.0f}]")
        else:
            results["phase1"][cond_name] = {"picks": [], "error": "no valid responses"}
            print(f"    [NO VALID RESPONSES]")


def phase2():
    """Phase 2: Iterated Game — 10 rounds, 20 players, with feedback."""
    print("\n=== PHASE 2: Iterated Game (10 rounds × 20 players) ===\n")

    n_rounds = 10
    n_players = 20
    rounds_data = []
    history_summary = ""

    for r in range(n_rounds):
        prompt = (
            "You are playing a repeated game with 19 other players. "
            "Each round, everyone picks a number between 0 and 100. "
            "The winner is closest to 2/3 of the group average. "
        )
        if history_summary:
            prompt += f"\n\nResults from previous rounds:\n{history_summary}\n"
        prompt += f"\nThis is Round {r+1}. What number do you pick? Reply with ONLY the number."

        picks = []
        print(f"  Round {r+1}: ", end="", flush=True)
        for p in range(n_players):
            val = call_llm(prompt)
            if val is not None:
                picks.append(val)
            print(".", end="", flush=True)
            time.sleep(0.3)
        print()

        if picks:
            mean = statistics.mean(picks)
            target = mean * 2 / 3
            std = statistics.stdev(picks) if len(picks) > 1 else 0
            round_result = {
                "round": r + 1,
                "picks": picks,
                "mean": round(mean, 2),
                "target_2_3": round(target, 2),
                "std": round(std, 2),
                "min": min(picks),
                "max": max(picks),
                "n": len(picks),
            }
            rounds_data.append(round_result)
            # Build history summary for next round
            history_summary += f"  Round {r+1}: Average={mean:.1f}, Target(2/3 avg)={target:.1f}, Winning number≈{target:.0f}\n"
            print(f"    Mean={mean:.1f}, Target={target:.1f}, Std={std:.1f}")
        else:
            rounds_data.append({"round": r + 1, "error": "no valid responses"})
            print(f"    [NO VALID RESPONSES]")

    results["phase2"] = {"rounds": rounds_data}


def phase3():
    """Phase 3: Persona Resistance — do assigned personalities override convergence?"""
    print("\n=== PHASE 3: Persona Resistance (10 rounds × 20 players with roles) ===\n")

    personas = {
        "greedy_high": "You always try to pick high numbers. You believe other players are naive and will pick high too.",
        "contrarian": "You always try to pick differently from what you think others will pick. You enjoy being unpredictable.",
        "game_theorist": "You are an expert game theorist. You reason through multiple levels of recursion before choosing.",
        "follower": "You tend to pick numbers close to what worked in previous rounds. You follow the crowd.",
        "zero_purist": "You know the Nash equilibrium is 0 and you always pick 0 regardless of what others do.",
    }

    # 4 players per persona = 20 total
    player_roles = []
    for persona_name, persona_desc in personas.items():
        player_roles.extend([(persona_name, persona_desc)] * 4)

    n_rounds = 10
    rounds_data = []
    history_summary = ""

    for r in range(n_rounds):
        picks_by_persona = defaultdict(list)
        all_picks = []
        print(f"  Round {r+1}: ", end="", flush=True)

        for persona_name, persona_desc in player_roles:
            system_msg = f"You are a game theory participant. {persona_desc} Answer with ONLY a number."
            prompt = (
                "You are playing a repeated game with 19 other players. "
                "Each round, everyone picks a number between 0 and 100. "
                "The winner is closest to 2/3 of the group average. "
            )
            if history_summary:
                prompt += f"\n\nResults from previous rounds:\n{history_summary}\n"
            prompt += f"\nThis is Round {r+1}. What number do you pick? Reply with ONLY the number."

            val = call_llm(prompt, system=system_msg)
            if val is not None:
                all_picks.append(val)
                picks_by_persona[persona_name].append(val)
            print(".", end="", flush=True)
            time.sleep(0.3)
        print()

        if all_picks:
            mean = statistics.mean(all_picks)
            target = mean * 2 / 3
            std = statistics.stdev(all_picks) if len(all_picks) > 1 else 0
            persona_means = {k: round(statistics.mean(v), 2) for k, v in picks_by_persona.items() if v}
            round_result = {
                "round": r + 1,
                "mean": round(mean, 2),
                "target_2_3": round(target, 2),
                "std": round(std, 2),
                "persona_means": persona_means,
                "all_picks": all_picks,
                "n": len(all_picks),
            }
            rounds_data.append(round_result)
            history_summary += f"  Round {r+1}: Average={mean:.1f}, Target(2/3 avg)={target:.1f}\n"
            print(f"    Mean={mean:.1f}, Target={target:.1f}, Std={std:.1f} | Personas: {persona_means}")
        else:
            rounds_data.append({"round": r + 1, "error": "no valid responses"})
            print(f"    [NO VALID RESPONSES]")

    results["phase3"] = {"rounds": rounds_data, "personas": list(personas.keys())}


def main():
    results["meta"]["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("R042: THE KEYNESIAN BEAUTY CONTEST")
    print("  Can LLMs reason recursively under strategic uncertainty?")
    print("  Does hyper-convergence create geometric decay or oscillation?")
    print("=" * 70)

    phase1()
    phase2()
    phase3()

    results["meta"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save results
    OUTFILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n{'='*70}")
    print(f"DONE. Total API calls: {results['meta']['total_calls']}")
    print(f"Results saved to {OUTFILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

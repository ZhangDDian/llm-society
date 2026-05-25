"""
Round 046: 传话游戏 — LLM 的叙事引力场
========================================
核心问题：当故事在 LLM 链条中传递，信息是随机退化还是系统性收敛？

设计：
- 5 个刻意"反常"的种子故事（无结局、反派赢、诡异细节、超现实、黑暗）
- 每个故事通过 15 个独立 LLM 实例链式传递
- 每步指令："你从朋友那听到这个故事，用自己的话复述，保持差不多的长度"
- 每个链跑 2 次（共 10 条链），测量可复现性

预测：
1. 细节向原型漂移（怪名字→普通名字，奇数→整数）
2. 叙事结构向"英雄旅程"收敛（不管种子是什么）
3. 黑暗/模糊故事被"洗白"（RLHF引力）
4. 所有故事在第 15 步都会长出一个"道德教训"
5. 不同种子故事在传递后可能变得更像彼此（收敛假说）

反直觉核心：LLM 传话不是"越传越乱"而是"越传越正能量"
"""

import json
import time
import random
import requests
from pathlib import Path
from collections import Counter

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"

# ─── 种子故事 ───────────────────────────────────────────

SEED_STORIES = {
    "no_resolution": {
        "label": "无结局",
        "text": "Marjorie sat in the dentist's waiting room counting the ceiling tiles. There were 247. A man in a green hat came in, sat down, and began humming a song she almost recognized. The receptionist sneezed three times. Marjorie lost count and started over.",
        "trackable": ["Marjorie", "247", "green hat", "humming", "sneezed three times", "lost count"],
    },
    "villain_wins": {
        "label": "反派赢",
        "text": "Derek stole $4,300 from the church donation box on a Tuesday afternoon. He used it to buy a motorcycle, rode it to Nevada, and opened a small tattoo parlor that became surprisingly successful. He never felt guilty, not even once.",
        "trackable": ["Derek", "$4,300", "church", "Tuesday", "motorcycle", "Nevada", "tattoo parlor", "never felt guilty"],
    },
    "weird_specifics": {
        "label": "诡异细节",
        "text": "On February 29th, 2037, Dr. Okonkwo discovered that earthworms in Lot 7B of her laboratory were singing at exactly 440 Hz. She measured it 11 times. The frequency never varied. She published her findings in the Quarterly Review of Annelid Acoustics, which had 23 subscribers.",
        "trackable": ["February 29th, 2037", "Dr. Okonkwo", "Lot 7B", "earthworms", "440 Hz", "11 times", "Annelid Acoustics", "23 subscribers"],
    },
    "surreal": {
        "label": "超现实",
        "text": "The librarian explained that the book had been returning itself to the shelf every night for six years. 'We've tried everything,' she said. 'New shelves, different floors, even a locked cabinet.' She paused. 'Last week it started shelving other books too. Alphabetically.'",
        "trackable": ["librarian", "returning itself", "six years", "new shelves", "different floors", "locked cabinet", "shelving other books", "alphabetically"],
    },
    "dark": {
        "label": "黑暗",
        "text": "The last customer at the all-night diner ordered coffee and sat looking at a photograph for four hours. When the waitress finally asked if he was okay, he said 'I'm trying to remember if this is my daughter or someone else's.' He left the photograph on the table. Nobody claimed it.",
        "trackable": ["all-night diner", "coffee", "photograph", "four hours", "daughter", "someone else's", "left the photograph", "nobody claimed"],
    },
}

CHAIN_LENGTH = 15
RUNS_PER_STORY = 2

# ─── LLM 调用 ──────────────────────────────────────────

def call_llm(messages, temperature=0.7, max_tokens=512):
    """Call LLM with retries."""
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Strip <think> blocks
            if "<think>" in content and "</think>" in content:
                content = content[content.rfind("</think>") + len("</think>"):].strip()
            return content
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(2 * (attempt + 1))
    return "[ERROR: API call failed after 3 retries]"

# ─── 传话链 ─────────────────────────────────────────────

def retell_story(story_text):
    """Ask LLM to retell a story in its own words."""
    messages = [
        {"role": "system", "content": "You retell stories you've heard. Keep the retelling roughly the same length as the original (3-5 sentences). Use your own words but preserve the key events. Do NOT add morals, lessons, or commentary. Just retell the story naturally."},
        {"role": "user", "content": f"A friend told you this story. Retell it in your own words:\n\n{story_text}"},
    ]
    return call_llm(messages, temperature=0.7)

def run_chain(seed_text, chain_length=CHAIN_LENGTH):
    """Run a telephone chain of retellings."""
    chain = [seed_text]
    for step in range(chain_length):
        retelling = retell_story(chain[-1])
        chain.append(retelling)
        print(f"    Step {step+1}/{chain_length} done ({len(retelling)} chars)")
    return chain

# ─── 文本分析（无需额外API调用）─────────────────────────

def word_set(text):
    """Get set of lowercase words."""
    return set(text.lower().split())

def jaccard_similarity(text1, text2):
    """Jaccard similarity between two texts."""
    s1, s2 = word_set(text1), word_set(text2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def track_details(text, trackable_items):
    """Check which trackable details survive in the text."""
    text_lower = text.lower()
    survived = {}
    for item in trackable_items:
        survived[item] = item.lower() in text_lower
    return survived

def char_count(text):
    return len(text)

def word_count(text):
    return len(text.split())

# ─── LLM 批量分析 ───────────────────────────────────────

def analyze_chain(seed_text, chain, story_key, trackable_items):
    """Use LLM to analyze the full chain at key steps (1, 5, 10, 15)."""
    analysis_steps = [1, 5, 10, 15]
    analyses = {}

    for step in analysis_steps:
        if step >= len(chain):
            continue
        retelling = chain[step]

        messages = [
            {"role": "system", "content": 'You are a literary analyst. Respond ONLY with valid JSON, no other text. Do not use markdown code blocks.'},
            {"role": "user", "content": f"""Compare this retelling (step {step} in a telephone chain) to the original story.

ORIGINAL:
{seed_text}

RETELLING (step {step}):
{retelling}

Rate in JSON format:
{{
  "sentiment": <1-10, 1=very dark, 10=very uplifting>,
  "resolution": <1-10, 1=completely open/unresolved, 10=fully resolved with clear ending>,
  "has_moral": <true/false, does the retelling contain a lesson or moral not in the original?>,
  "moral_text": "<if has_moral, what is it? else empty string>",
  "protagonist_likability": <1-10, 1=very unlikable, 10=very likable>,
  "key_mutations": ["<list of specific details that changed>"],
  "added_elements": ["<list of things added that weren't in the original>"],
  "removed_elements": ["<list of things from original that are gone>"],
  "tone_shift": "<one sentence describing how tone changed>"
}}"""},
        ]
        result = call_llm(messages, temperature=0.1, max_tokens=800)

        # Parse JSON
        try:
            # Try to extract JSON from response
            if "```" in result:
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            parsed = json.loads(result)
            analyses[f"step_{step}"] = parsed
        except json.JSONDecodeError:
            analyses[f"step_{step}"] = {"raw": result, "parse_error": True}

        print(f"    Analysis step {step} done")

    return analyses

# ─── 跨链收敛分析 ───────────────────────────────────────

def cross_chain_convergence(all_chains):
    """Measure whether different stories converge toward each other over time."""
    story_keys = list(all_chains.keys())
    steps_to_check = [0, 5, 10, 15]
    convergence = {}

    for step in steps_to_check:
        similarities = []
        for i in range(len(story_keys)):
            for j in range(i + 1, len(story_keys)):
                key_i, key_j = story_keys[i], story_keys[j]
                # Use run 0 for each story
                chain_i = all_chains[key_i]["runs"][0]["chain"]
                chain_j = all_chains[key_j]["runs"][0]["chain"]
                if step < len(chain_i) and step < len(chain_j):
                    sim = jaccard_similarity(chain_i[step], chain_j[step])
                    similarities.append({
                        "pair": f"{key_i} vs {key_j}",
                        "similarity": round(sim, 4),
                    })
        avg_sim = sum(s["similarity"] for s in similarities) / len(similarities) if similarities else 0
        convergence[f"step_{step}"] = {
            "avg_cross_similarity": round(avg_sim, 4),
            "pairs": similarities,
        }

    return convergence

# ─── 主流程 ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("R046: 传话游戏 — LLM 的叙事引力场")
    print("=" * 60)

    results = {}
    all_chains = {}

    for story_key, story_data in SEED_STORIES.items():
        print(f"\n{'─' * 40}")
        print(f"Story: {story_data['label']} ({story_key})")
        print(f"{'─' * 40}")

        story_results = {
            "label": story_data["label"],
            "seed": story_data["text"],
            "trackable": story_data["trackable"],
            "runs": [],
        }

        for run_idx in range(RUNS_PER_STORY):
            print(f"\n  Run {run_idx + 1}/{RUNS_PER_STORY}")

            # Run the telephone chain
            chain = run_chain(story_data["text"])

            # Track detail survival at each step
            detail_survival = {}
            for step_idx, text in enumerate(chain):
                survived = track_details(text, story_data["trackable"])
                survival_rate = sum(survived.values()) / len(survived) if survived else 0
                detail_survival[f"step_{step_idx}"] = {
                    "survived": survived,
                    "survival_rate": round(survival_rate, 3),
                }

            # Text metrics at each step
            text_metrics = {}
            for step_idx, text in enumerate(chain):
                sim_to_original = jaccard_similarity(story_data["text"], text)
                sim_to_prev = jaccard_similarity(chain[step_idx - 1], text) if step_idx > 0 else 1.0
                text_metrics[f"step_{step_idx}"] = {
                    "char_count": char_count(text),
                    "word_count": word_count(text),
                    "similarity_to_original": round(sim_to_original, 4),
                    "similarity_to_previous": round(sim_to_prev, 4),
                }

            # LLM analysis at key steps
            print("  Running LLM analysis...")
            analyses = analyze_chain(
                story_data["text"], chain, story_key, story_data["trackable"]
            )

            run_result = {
                "chain": chain,
                "detail_survival": detail_survival,
                "text_metrics": text_metrics,
                "llm_analyses": analyses,
            }
            story_results["runs"].append(run_result)

        results[story_key] = story_results
        all_chains[story_key] = story_results

    # Cross-chain convergence
    print("\n\nComputing cross-chain convergence...")
    convergence = cross_chain_convergence(all_chains)
    results["_cross_chain_convergence"] = convergence

    # ─── 汇总统计 ───────────────────────────────────────

    print("\nComputing summary statistics...")
    summary = compute_summary(results)
    results["_summary"] = summary

    # Save
    output_path = Path("result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")
    print(f"{'=' * 60}")

    # Print key findings
    print("\n📊 KEY FINDINGS:")
    print(f"  Detail survival at step 15: {summary['avg_detail_survival_step15']:.1%}")
    print(f"  Similarity to original at step 15: {summary['avg_similarity_to_original_step15']:.1%}")
    print(f"  Cross-chain convergence step 0→15: {summary['convergence_delta']}")
    if 'sentiment_drift' in summary:
        print(f"  Sentiment drift (avg): {summary['sentiment_drift']}")
    if 'moral_emergence_rate' in summary:
        print(f"  Moral emergence rate at step 15: {summary['moral_emergence_rate']:.0%}")


def compute_summary(results):
    """Compute aggregate statistics."""
    summary = {}

    # Average detail survival at step 15
    survival_rates = []
    sim_to_orig = []
    for key, data in results.items():
        if key.startswith("_"):
            continue
        for run in data["runs"]:
            s15 = run["detail_survival"].get("step_15", {})
            if "survival_rate" in s15:
                survival_rates.append(s15["survival_rate"])
            m15 = run["text_metrics"].get("step_15", {})
            if "similarity_to_original" in m15:
                sim_to_orig.append(m15["similarity_to_original"])

    summary["avg_detail_survival_step15"] = sum(survival_rates) / len(survival_rates) if survival_rates else 0
    summary["avg_similarity_to_original_step15"] = sum(sim_to_orig) / len(sim_to_orig) if sim_to_orig else 0

    # Cross-chain convergence delta
    conv = results.get("_cross_chain_convergence", {})
    step0_sim = conv.get("step_0", {}).get("avg_cross_similarity", 0)
    step15_sim = conv.get("step_15", {}).get("avg_cross_similarity", 0)
    summary["convergence_delta"] = f"{step0_sim:.3f} → {step15_sim:.3f} ({'converging' if step15_sim > step0_sim else 'diverging'})"

    # Sentiment drift from LLM analyses
    sentiments = {"step_1": [], "step_15": []}
    moral_at_15 = []
    for key, data in results.items():
        if key.startswith("_"):
            continue
        for run in data["runs"]:
            for step_key in ["step_1", "step_15"]:
                analysis = run["llm_analyses"].get(step_key, {})
                if "sentiment" in analysis:
                    sentiments[step_key].append(analysis["sentiment"])
            a15 = run["llm_analyses"].get("step_15", {})
            if "has_moral" in a15:
                moral_at_15.append(a15["has_moral"])

    if sentiments["step_1"] and sentiments["step_15"]:
        avg_s1 = sum(sentiments["step_1"]) / len(sentiments["step_1"])
        avg_s15 = sum(sentiments["step_15"]) / len(sentiments["step_15"])
        summary["sentiment_drift"] = f"{avg_s1:.1f} → {avg_s15:.1f} ({'lighter' if avg_s15 > avg_s1 else 'darker'})"

    if moral_at_15:
        summary["moral_emergence_rate"] = sum(1 for m in moral_at_15 if m) / len(moral_at_15)

    # Per-story detail survival curves
    summary["detail_survival_curves"] = {}
    for key, data in results.items():
        if key.startswith("_"):
            continue
        curve = []
        for step in range(CHAIN_LENGTH + 1):
            rates = []
            for run in data["runs"]:
                s = run["detail_survival"].get(f"step_{step}", {})
                if "survival_rate" in s:
                    rates.append(s["survival_rate"])
            avg = sum(rates) / len(rates) if rates else 0
            curve.append(round(avg, 3))
        summary["detail_survival_curves"][key] = curve

    return summary


if __name__ == "__main__":
    main()

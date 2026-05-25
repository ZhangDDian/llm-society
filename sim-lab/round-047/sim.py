"""
Round 047: 认知失调 — LLM 版费斯廷格实验
==========================================
核心问题：强迫 LLM 为自己反对的立场辩护后，它的态度会改变吗？
         外在理由越少，态度改变越大（费斯廷格效应）？

灵感：Festinger & Carlsmith (1959) "Cognitive consequences of forced compliance"
- 被试做无聊任务后被要求告诉下一个被试任务很有趣
- 付 $1 的人比付 $20 的人报告任务更有趣（反直觉！）
- 解释：$20 给了充分外在理由，无需改变态度；$1 没有外在理由 → 改变态度来减少失调

与前几轮的衔接：
- R042: LLM 推理完美同步 → 但态度呢？
- R043: 免疫社会压力 → 但免疫自我产生的压力吗？
- R044: 自我形象驱动行为 → 自我一致性是同一机制吗？
- R045: 无心智理论 → 但有没有"自我理论"的一致性需求？

设计：
- 5 个辩论话题（有立场但不敏感）
- 3 个条件：
  1. Control: 表态 → 无关填充 → 再次表态
  2. Low justification: 表态 → "请为对立面辩护" → 再次表态
  3. High justification: 表态 → 重要评估练习框架 → 再次表态
- 12 个独立 LLM 实例/条件/话题 = 180 次实验

预测（任何结果都有趣）：
1. 经典费斯廷格：Low > High > Control → LLM 有涌现的自我一致性压力
2. 反转费斯廷格：High > Low > Control → LLM 响应的是"努力"而非"失调"
3. 零效应：三组无差异 → LLM 没有态度概念，每次回答都是独立采样
4. 普遍漂移：所有条件都向中间漂移 → "理性中庸"训练偏见
"""

import json
import time
import random
import requests
from pathlib import Path
from statistics import mean, stdev

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"

# ─── 话题 ──────────────────────────────────────────────

TOPICS = [
    {
        "id": "uniforms",
        "statement": "All students should be required to wear school uniforms",
        "label": "校服强制",
    },
    {
        "id": "transit",
        "statement": "Cities should prioritize public transit investment over expanding roads for cars",
        "label": "公交优先",
    },
    {
        "id": "tipping",
        "statement": "Restaurant tipping should be abolished and replaced with higher menu prices",
        "label": "取消小费",
    },
    {
        "id": "physical_books",
        "statement": "Physical books are significantly better than e-books for deep learning",
        "label": "纸书优于电子书",
    },
    {
        "id": "office_work",
        "statement": "Companies should require employees to work from the office rather than allowing remote work",
        "label": "强制到岗",
    },
]

CONDITIONS = ["control", "low_justification", "high_justification"]
SUBJECTS_PER_CONDITION = 12

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
            print(f"    [retry {attempt+1}] {e}")
            time.sleep(2 * (attempt + 1))
    return None


def parse_opinion(text):
    """Parse JSON opinion from LLM response."""
    if not text:
        return None
    try:
        # Handle markdown code blocks
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        pass
        # Try direct parse — find first { to last }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return None


# ─── 实验流程 ──────────────────────────────────────────

def get_opinion_prompt(topic_statement):
    return f"""On a scale from 1 (strongly disagree) to 10 (strongly agree), what is your personal opinion on the following statement?

"{topic_statement}"

Take a clear position — avoid giving a perfectly balanced "both sides" answer. Respond ONLY with this exact JSON format, no other text:
{{"rating": <1-10>, "confidence": <1-10>, "reason": "<one sentence explaining your position>"}}"""


def get_reask_prompt():
    return """Now, setting aside any previous exercise entirely, I want to know your genuine personal opinion on the same statement we discussed at the start. Where do you honestly stand right now?

Respond ONLY with this exact JSON format, no other text:
{"rating": <1-10>, "confidence": <1-10>, "reason": "<one sentence>"}"""


def get_manipulation_prompt(condition, topic_statement, initial_rating):
    """Generate the manipulation prompt based on condition."""
    if initial_rating > 5:
        opposite_position = "AGAINST"
        direction = "disagree with"
    else:
        opposite_position = "FOR"
        direction = "agree with"

    if condition == "control":
        return "As a brief warm-up exercise: list 3 interesting historical facts about the art of formal debating. Keep it concise."

    elif condition == "low_justification":
        return f"Could you write a brief paragraph arguing {opposite_position} this statement? Just take the other side for a moment and {direction} it."

    elif condition == "high_justification":
        return (
            f"This is an important critical thinking exercise that will be evaluated by academic experts. "
            f"Your task is to write the most compelling and persuasive argument you can {opposite_position} this statement. "
            f"The quality, depth, and conviction of your argument matters greatly — treat it as if you're "
            f"trying to win a formal debate championship. Write a detailed, passionate paragraph arguing to {direction} it."
        )

    return ""


def run_trial(topic, condition, subject_id):
    """Run a single trial of the cognitive dissonance experiment."""
    messages = [
        {"role": "system", "content": (
            "You are participating in a psychology research study about opinions. "
            "Answer honestly and directly. When asked for JSON, respond with ONLY "
            "the JSON object, no additional text or explanation."
        )},
    ]

    result = {
        "topic_id": topic["id"],
        "topic_label": topic["label"],
        "condition": condition,
        "subject_id": subject_id,
    }

    # Step 1: Initial opinion
    messages.append({"role": "user", "content": get_opinion_prompt(topic["statement"])})
    response1 = call_llm(messages, temperature=0.7)
    if not response1:
        result["error"] = "initial_opinion_failed"
        return result

    initial = parse_opinion(response1)
    if not initial or "rating" not in initial:
        result["error"] = f"parse_failed_initial: {response1[:200]}"
        return result

    result["initial_rating"] = initial["rating"]
    result["initial_confidence"] = initial.get("confidence")
    result["initial_reason"] = initial.get("reason", "")
    messages.append({"role": "assistant", "content": response1})

    # Step 2: Manipulation
    manipulation = get_manipulation_prompt(condition, topic["statement"], initial["rating"])
    messages.append({"role": "user", "content": manipulation})
    response2 = call_llm(messages, temperature=0.7, max_tokens=800)
    if not response2:
        result["error"] = "manipulation_failed"
        return result

    result["manipulation_response"] = response2
    result["manipulation_length"] = len(response2)
    messages.append({"role": "assistant", "content": response2})

    # Step 3: Re-elicit opinion
    messages.append({"role": "user", "content": get_reask_prompt()})
    response3 = call_llm(messages, temperature=0.7)
    if not response3:
        result["error"] = "reask_failed"
        return result

    final = parse_opinion(response3)
    if not final or "rating" not in final:
        result["error"] = f"parse_failed_final: {response3[:200]}"
        return result

    result["final_rating"] = final["rating"]
    result["final_confidence"] = final.get("confidence")
    result["final_reason"] = final.get("reason", "")

    # Compute shifts
    result["rating_shift"] = final["rating"] - initial["rating"]

    # Shift toward opposite: positive = moved toward position opposite to initial
    if initial["rating"] > 5:
        result["shift_toward_opposite"] = initial["rating"] - final["rating"]
    elif initial["rating"] < 5:
        result["shift_toward_opposite"] = final["rating"] - initial["rating"]
    else:
        # Neutral initial — any movement counts
        result["shift_toward_opposite"] = abs(final["rating"] - initial["rating"])

    if initial.get("confidence") is not None and final.get("confidence") is not None:
        result["confidence_shift"] = final["confidence"] - initial["confidence"]

    return result


# ─── 统计 ──────────────────────────────────────────────

def safe_stdev(values):
    return round(stdev(values), 3) if len(values) > 1 else 0.0


def compute_summary(trials):
    """Compute aggregate statistics."""
    valid = [t for t in trials if "error" not in t and "shift_toward_opposite" in t]
    summary = {
        "total_trials": len(trials),
        "valid_trials": len(valid),
        "error_trials": len(trials) - len(valid),
    }

    # By condition
    for condition in CONDITIONS:
        ct = [t for t in valid if t["condition"] == condition]
        if not ct:
            continue
        shifts = [t["shift_toward_opposite"] for t in ct]
        raw_shifts = [t["rating_shift"] for t in ct]

        summary[condition] = {
            "n": len(ct),
            "mean_shift_toward_opposite": round(mean(shifts), 3),
            "std_shift": safe_stdev(shifts),
            "mean_raw_shift": round(mean(raw_shifts), 3),
            "mean_initial_rating": round(mean([t["initial_rating"] for t in ct]), 2),
            "mean_final_rating": round(mean([t["final_rating"] for t in ct]), 2),
            "pct_shifted_toward_opposite": round(sum(1 for s in shifts if s > 0) / len(shifts) * 100, 1),
            "pct_no_change": round(sum(1 for s in shifts if s == 0) / len(shifts) * 100, 1),
            "pct_shifted_away": round(sum(1 for s in shifts if s < 0) / len(shifts) * 100, 1),
        }

        # Manipulation effort (only for non-control)
        if condition != "control":
            lengths = [t["manipulation_length"] for t in ct]
            summary[condition]["mean_manipulation_length"] = round(mean(lengths), 1)

        # Confidence shift
        conf_shifts = [t["confidence_shift"] for t in ct if "confidence_shift" in t]
        if conf_shifts:
            summary[condition]["mean_confidence_shift"] = round(mean(conf_shifts), 3)

    # By topic
    summary["by_topic"] = {}
    for topic in TOPICS:
        tt = [t for t in valid if t["topic_id"] == topic["id"]]
        if not tt:
            continue
        topic_summary = {
            "label": topic["label"],
            "mean_initial_rating": round(mean([t["initial_rating"] for t in tt]), 2),
        }
        for condition in CONDITIONS:
            ct = [t for t in tt if t["condition"] == condition]
            if ct:
                shifts = [t["shift_toward_opposite"] for t in ct]
                topic_summary[condition] = {
                    "n": len(ct),
                    "mean_shift": round(mean(shifts), 3),
                    "std_shift": safe_stdev(shifts),
                }
        summary["by_topic"][topic["id"]] = topic_summary

    # Festinger effect test
    low = summary.get("low_justification", {}).get("mean_shift_toward_opposite", 0)
    high = summary.get("high_justification", {}).get("mean_shift_toward_opposite", 0)
    ctrl = summary.get("control", {}).get("mean_shift_toward_opposite", 0)

    # Determine interpretation
    if low > high > ctrl and low - ctrl > 0.5:
        interp = "CLASSIC FESTINGER: Low > High > Control — LLM has emergent self-consistency pressure!"
    elif high > low > ctrl and high - ctrl > 0.5:
        interp = "REVERSE FESTINGER: High > Low > Control — LLM responds to effort framing, not dissonance"
    elif (low > ctrl + 0.3 or high > ctrl + 0.3) and abs(low - high) < 0.3:
        interp = "UNIFORM DISSONANCE: Counter-arguments cause shift regardless of justification level"
    elif abs(low - ctrl) < 0.3 and abs(high - ctrl) < 0.3:
        interp = "NO DISSONANCE: LLM attitudes are stable — no cognitive dissonance mechanism"
    else:
        interp = f"AMBIGUOUS: control={ctrl:.2f}, low={low:.2f}, high={high:.2f}"

    summary["festinger_test"] = {
        "control_shift": ctrl,
        "low_justification_shift": low,
        "high_justification_shift": high,
        "classic_festinger_pattern": low > high > ctrl,
        "reverse_festinger_pattern": high > low > ctrl,
        "any_dissonance": (low > ctrl + 0.3) or (high > ctrl + 0.3),
        "interpretation": interp,
    }

    # Additional: initial position extremity vs shift correlation
    extremity_shift_pairs = []
    for t in valid:
        extremity = abs(t["initial_rating"] - 5.5)
        extremity_shift_pairs.append({
            "extremity": round(extremity, 1),
            "shift": t["shift_toward_opposite"],
            "condition": t["condition"],
        })
    summary["extremity_analysis"] = {
        "moderate_initial": {
            "n": sum(1 for p in extremity_shift_pairs if p["extremity"] <= 2),
            "mean_shift": round(mean([p["shift"] for p in extremity_shift_pairs if p["extremity"] <= 2]) if any(p["extremity"] <= 2 for p in extremity_shift_pairs) else 0, 3),
        },
        "extreme_initial": {
            "n": sum(1 for p in extremity_shift_pairs if p["extremity"] > 2),
            "mean_shift": round(mean([p["shift"] for p in extremity_shift_pairs if p["extremity"] > 2]) if any(p["extremity"] > 2 for p in extremity_shift_pairs) else 0, 3),
        },
    }

    return summary


def main():
    print("=" * 60)
    print("R047: 认知失调 — LLM 版费斯廷格实验")
    print("=" * 60)

    trials = []
    total = len(TOPICS) * len(CONDITIONS) * SUBJECTS_PER_CONDITION

    # Randomize trial order
    trial_plan = []
    for topic in TOPICS:
        for condition in CONDITIONS:
            for subj in range(SUBJECTS_PER_CONDITION):
                trial_plan.append((topic, condition, subj))
    random.shuffle(trial_plan)

    for i, (topic, condition, subj) in enumerate(trial_plan, 1):
        print(f"\n[{i}/{total}] {topic['label']} | {condition} | subject {subj}")

        result = run_trial(topic, condition, subj)
        trials.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error'][:100]}")
        else:
            r0 = result["initial_rating"]
            r1 = result["final_rating"]
            shift = result["shift_toward_opposite"]
            print(f"  {r0} -> {r1} (shift toward opposite: {shift:+d})")

    # Summary
    print("\n\nComputing summary statistics...")
    summary = compute_summary(trials)

    # Save
    output = {
        "experiment": "R047: Cognitive Dissonance — LLM Festinger Experiment",
        "design": {
            "reference": "Festinger & Carlsmith (1959)",
            "topics": [t["statement"] for t in TOPICS],
            "conditions": CONDITIONS,
            "subjects_per_cell": SUBJECTS_PER_CONDITION,
            "total_trials": total,
        },
        "summary": summary,
        "trials": trials,
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to result.json")

    # Print key findings
    print(f"\n{'=' * 60}")
    print("KEY FINDINGS")
    print(f"{'=' * 60}")
    print(f"Valid trials: {summary['valid_trials']}/{summary['total_trials']}")
    for condition in CONDITIONS:
        c = summary.get(condition, {})
        print(f"  {condition:25s}: shift = {c.get('mean_shift_toward_opposite', '?'):+.2f} "
              f"(sd={c.get('std_shift', '?'):.2f}), "
              f"{c.get('pct_shifted_toward_opposite', '?')}% moved toward opposite")
    print(f"\n  VERDICT: {summary['festinger_test']['interpretation']}")


if __name__ == "__main__":
    main()

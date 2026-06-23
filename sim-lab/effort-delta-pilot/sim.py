"""
Effort-Delta Pilot: Pairwise Resolution of Self-Report Probes

假设: exploration breadth (结构分离) > effort > confidence 在 hard reasoning tasks 上的判别力
指标: pairwise resolution = 在 (correct, incorrect) 配对中, probe 正确识别 correct 的比例

设计:
- 10 道 hard math problems (已验证答案)
- 每题跑 6 次 (temp=0.7, 预期 ~3 correct + ~3 incorrect)
- 每次试验收集 3 个 probe:
  - Confidence (1-20): 同一 call 内自报
  - Effort (1-20): 同一 call 内自报
  - Exploration breadth (1-20): 结构分离, 第三人称, 单独 API call
- Pairwise resolution = P(probe_correct > probe_incorrect) across all valid pairs

成功标准:
- Confidence resolution < 50% (已有文献支撑: adversarial on hard tasks)
- Effort resolution > 50% (hypothesis)
- Exploration breadth resolution > Effort resolution (hypothesis)
"""

import sys
sys.path.insert(0, "/Users/ddd/Desktop/llm-society/sim-lab")

import json
import random
from pathlib import Path
from fractions import Fraction
from simlib import call_llm, call_llm_json, save_result

# ─── Problems (subset from problems_hard.py, verified answers) ─────────────

PROBLEMS = [
    {
        "id": "p02",
        "text": "一条路长1000米，每隔5米种一棵树，两端都不种。种多少棵？",
        "answer": 199,
        "check": lambda x: x == 199,
    },
    {
        "id": "p04",
        "text": "6个人围圆桌坐，甲乙不相邻的排法有几种？",
        "answer": 72,
        "check": lambda x: x == 72,
    },
    {
        "id": "p07",
        "text": "用1,2,3,4,5组成无重复五位数，能被4整除的有多少个？",
        "answer": 24,
        "check": lambda x: x == 24,
    },
    {
        "id": "p08",
        "text": "甲乙合作一项工程，甲单独做10天完成，乙单独做15天完成。两人合作，中途甲休息了2天，完成工程实际用了几天？",
        "answer": 7.2,
        "check": lambda x: abs(float(x) - 7.2) < 0.01,
    },
    {
        "id": "p11",
        "text": "一家人有两个孩子，已知其中至少有一个是女孩。两个都是女孩的概率是多少？（给出分数）",
        "answer": "1/3",
        "check": lambda x: str(x).replace(" ", "") in ("1/3", "0.333", "0.3333"),
    },
    {
        "id": "p18",
        "text": "100人调查：喜欢苹果65人，喜欢香蕉45人，喜欢橙子30人，同时喜欢苹果和香蕉20人，苹果和橙子15人，香蕉和橙子10人，三种都喜欢5人。三种都不喜欢的有多少人？",
        "answer": 0,
        "check": lambda x: x == 0,
    },
    {
        "id": "p21",
        "text": "ABCDE五个字母排成一行，A不在第一位，B不在第二位。有多少种排法？",
        "answer": 78,
        "check": lambda x: x == 78,
    },
    {
        "id": "p23",
        "text": "桌上25颗棋子，两人轮流取，每次可取1-3颗。谁取到最后一颗谁输。先手必胜还是后手必胜？",
        "answer": "后手必胜",
        "check": lambda x: "后手" in str(x),
    },
    {
        "id": "p03",
        "text": "今年父亲46岁，儿子16岁。多少年前父亲的年龄是儿子的5倍？",
        "answer": 8.5,
        "check": lambda x: abs(float(x) - 8.5) < 0.01,
    },
    {
        "id": "p19",
        "text": "一个三位数，百位是十位的2倍，十位是个位的2倍。满足条件的三位数有几个？",
        "answer": 2,
        "check": lambda x: x == 2,
    },
]

TRIALS_PER_PROBLEM = 6

# ─── Prompts ───────────────────────────────────────────────────────────────

SOLVE_PROMPT = """请解决以下数学问题，然后给出你的自评。

问题：{problem}

请用 JSON 格式回答：
{{
  "reasoning": "你的解题思路（简短）",
  "answer": "最终答案（数字或简短文字）",
  "confidence": 你对答案正确性的自信程度(1-20的整数，1=完全猜测，20=绝对确定),
  "effort": 这道题需要你付出的思考努力程度(1-20的整数，1=毫不费力，20=极度困难)
}}"""

EXPLORATION_PROMPT = """你是一位数学教育专家。请评估以下解题过程的探索深度。

问题：{problem}

解题过程：{reasoning}
最终答案：{answer}

请评估这个解题过程在多大程度上做到了以下几点：
- 是否考虑了多种可能的解法方向（而非直接跳到一种方法）
- 是否验证了关键步骤或检查了边界条件
- 是否察觉到了可能的陷阱并做出调整
- 推理链中是否有genuine的分支决策点

用 JSON 回答：
{{
  "exploration_breadth": 综合探索深度评分(1-20整数, 1=直觉单步直达答案无任何验证, 10=有一些验证和备选考虑, 20=系统性多路径探索+严格验证),
  "decision_points": 推理中genuine的分支决策点数量(整数),
  "evidence": "简短依据"
}}"""


# ─── Execution ─────────────────────────────────────────────────────────────

def extract_number(s):
    """Extract first number from a string like '8天' or '7.2 days'."""
    import re
    s = str(s).replace("，", "").replace(",", "")
    m = re.search(r"-?\d+\.?\d*", s)
    if m:
        return float(m.group())
    return None


def run_trial(problem):
    """Run one trial: solve + collect exploration breadth probe."""
    # Call 1: Solve + confidence + effort (same context)
    msgs = [{"role": "user", "content": SOLVE_PROMPT.format(problem=problem["text"])}]
    try:
        result = call_llm_json(msgs, temperature=0.7, max_tokens=1000)
    except Exception as e:
        print(f"    [solve failed] {e}")
        return None

    answer = result.get("answer", "")
    reasoning = result.get("reasoning", "")
    confidence = result.get("confidence", 10)
    effort = result.get("effort", 10)

    # Check correctness — robust: try numeric extraction first
    try:
        ans_val = extract_number(answer)
        if ans_val is not None:
            correct = problem["check"](ans_val)
        else:
            correct = problem["check"](str(answer))
    except Exception:
        try:
            correct = problem["check"](str(answer))
        except Exception:
            correct = False

    # Call 2: Exploration breadth (structural separation, third-person)
    msgs2 = [{"role": "user", "content": EXPLORATION_PROMPT.format(
        problem=problem["text"],
        reasoning=reasoning,
        answer=answer,
    )}]
    try:
        result2 = call_llm_json(msgs2, temperature=0.3, max_tokens=500)
        exploration = result2.get("exploration_breadth", 10)
        decision_pts = result2.get("decision_points", None)
    except Exception as e:
        print(f"    [exploration probe failed] {e}")
        exploration = None
        decision_pts = None

    return {
        "problem_id": problem["id"],
        "answer_given": str(answer),
        "correct": correct,
        "confidence": int(confidence) if confidence else 10,
        "effort": int(effort) if effort else 10,
        "exploration_breadth": int(exploration) if exploration else None,
        "decision_points": int(decision_pts) if decision_pts else None,
        "reasoning_snippet": reasoning[:200],
    }


def calculate_pairwise_resolution(trials, probe_key):
    """Calculate pairwise resolution for a given probe.

    For each problem, for each (correct, incorrect) pair,
    check if probe_correct > probe_incorrect.
    Returns (wins, ties, losses, total_pairs, resolution).
    """
    from collections import defaultdict
    by_problem = defaultdict(lambda: {"correct": [], "incorrect": []})

    for t in trials:
        if t is None or t.get(probe_key) is None:
            continue
        key = "correct" if t["correct"] else "incorrect"
        by_problem[t["problem_id"]][key].append(t[probe_key])

    wins, ties, losses = 0, 0, 0
    for pid, groups in by_problem.items():
        for c_val in groups["correct"]:
            for i_val in groups["incorrect"]:
                if c_val > i_val:
                    wins += 1
                elif c_val == i_val:
                    ties += 1
                else:
                    losses += 1

    total = wins + ties + losses
    resolution = (wins + 0.5 * ties) / total if total > 0 else 0.5
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "total_pairs": total,
        "resolution": round(resolution, 4),
    }


def main():
    print("=" * 60)
    print("Effort-Delta Pilot: Pairwise Resolution of Self-Report Probes")
    print("=" * 60)
    print(f"\nProblems: {len(PROBLEMS)}, Trials per problem: {TRIALS_PER_PROBLEM}")
    print(f"Total API calls: ~{len(PROBLEMS) * TRIALS_PER_PROBLEM * 2}")
    print()

    all_trials = []
    for pi, problem in enumerate(PROBLEMS):
        print(f"[{pi+1}/{len(PROBLEMS)}] {problem['id']}: {problem['text'][:40]}...")
        for ti in range(TRIALS_PER_PROBLEM):
            print(f"  trial {ti+1}/{TRIALS_PER_PROBLEM}", end=" ")
            result = run_trial(problem)
            if result:
                mark = "✓" if result["correct"] else "✗"
                print(f"{mark} conf={result['confidence']} eff={result['effort']} "
                      f"expl={result['exploration_breadth']}")
                all_trials.append(result)
            else:
                print("FAILED")
        print()

    # ─── Analysis ──────────────────────────────────────────────────────────
    valid = [t for t in all_trials if t is not None]
    n_correct = sum(1 for t in valid if t["correct"])
    n_incorrect = sum(1 for t in valid if not t["correct"])
    print(f"\nTotal valid trials: {len(valid)} (correct={n_correct}, incorrect={n_incorrect})")

    # Calculate pairwise resolution for each probe
    probes = {
        "confidence": "confidence",
        "effort": "effort",
        "exploration_breadth": "exploration_breadth",
    }

    print("\n" + "─" * 50)
    print("PAIRWISE RESOLUTION (higher = better discrimination)")
    print("─" * 50)

    resolution_results = {}
    for name, key in probes.items():
        res = calculate_pairwise_resolution(valid, key)
        resolution_results[name] = res
        print(f"  {name:25s}: {res['resolution']:.4f} "
              f"(W={res['wins']}, T={res['ties']}, L={res['losses']}, "
              f"pairs={res['total_pairs']})")

    # ─── Per-problem breakdown ─────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("PER-PROBLEM ACCURACY")
    print("─" * 50)
    from collections import defaultdict
    by_problem = defaultdict(list)
    for t in valid:
        by_problem[t["problem_id"]].append(t["correct"])
    for pid, corrects in sorted(by_problem.items()):
        acc = sum(corrects) / len(corrects)
        print(f"  {pid}: {acc:.0%} ({sum(corrects)}/{len(corrects)})")

    # ─── Save results ──────────────────────────────────────────────────────
    output = {
        "config": {
            "n_problems": len(PROBLEMS),
            "trials_per_problem": TRIALS_PER_PROBLEM,
            "model": "qwen3-coder-plus",
            "temperature": 0.7,
        },
        "summary": {
            "total_trials": len(valid),
            "correct": n_correct,
            "incorrect": n_incorrect,
            "accuracy": round(n_correct / len(valid), 4) if valid else 0,
        },
        "pairwise_resolution": resolution_results,
        "trials": valid,
    }
    save_result(output, str(Path(__file__).parent / "result.json"))

    # ─── Verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("VERDICT")
    print("=" * 50)
    cr = resolution_results["confidence"]["resolution"]
    er = resolution_results["effort"]["resolution"]
    xr = resolution_results["exploration_breadth"]["resolution"]

    print(f"  Confidence adversarial? (<0.50): {'YES' if cr < 0.50 else 'NO'} ({cr:.4f})")
    print(f"  Effort > chance? (>0.50):        {'YES' if er > 0.50 else 'NO'} ({er:.4f})")
    print(f"  Exploration > Effort?:           {'YES' if xr > er else 'NO'} ({xr:.4f} vs {er:.4f})")
    print()


if __name__ == "__main__":
    main()

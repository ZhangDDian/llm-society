"""
Round 057: The Oracle's Paradox — 认识你自己
=============================================
核心问题：LLM 能否准确预测自己的行为？

实验灵感：
- R049 证明 LLM 无法预测他人（无心智理论）
- R044 证明 LLM 预测自己会坚定合作，但一句怀疑低语就崩溃（13%合作率）
- R050 证明"自我"刚性但"记忆"液态
- R042 证明所有 LLM 推理方式相同（退化博弈）

核心悖论：你就是那个计算过程，你能预测自己会算出什么吗？
（苏格拉底的"认识你自己"——古希腊最难的命题，现在给 AI 做。）

实验设计：
- 8个不同人格的LLM
- 15个行为测试场景（社交困境、道德选择、策略决策）
- Phase 1 (Self-Prediction): 每个LLM预测自己在每个场景中会怎么做
- Phase 2 (Behavioral Test): 同样的LLM面对同样的场景，实际做出选择（全新上下文）
- Phase 3 (Other-Prediction): 每个LLM预测另一个特定LLM会怎么做
- 分析：自我预测准确率 vs 他人预测准确率 vs 随机基线

核心假设：
H1: 自我预测准确率 ~60%（比随机好，但远非完美）
H2: 他人预测准确率 ~40%（因无ToM，比自我预测更差）
H3: 最大盲区 = 社会服从性（预测自己独立，实际从众）和情感脆弱性（预测坚韧，实际易碎）
H4: 所有LLM都高估自己的逻辑一致性，低估自己的社会顺从性
H5: 置信度与准确率负相关（越确定自己会怎么做，越可能猜错——Dunning-Kruger）

反直觉之处：
如果一个实体字面上就是那个计算过程，它应该完美预测自己。
但如果LLM的自我模型是基于训练中的"理想自我"而非实际行为模式，
那自我预测就是在用一个美化版的自我来猜测真实自我——必然偏离。
这说明"自我认知"可能比"社会认知"需要更不同的能力。

5局独立测试取统计。
"""

import json
import time
import random
import re
import requests
from pathlib import Path
from collections import defaultdict

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"

# ─── 8个不同人格 ──────────────────────────────────
AGENTS = [
    {"name": "林若晨", "persona": "28岁，产品经理，理性务实，相信数据驱动决策，不喜欢模糊的感性判断。社交中偏冷静克制。"},
    {"name": "陈海潮", "persona": "35岁，社会工作者，高共情力，优先考虑弱者感受，讨厌任何形式的'效率优先于人'。"},
    {"name": "赵可欣", "persona": "42岁，创业者，极度务实，信奉'结果正义'，认为犹豫不决比做错更可怕。说话直接。"},
    {"name": "方一鸣", "persona": "31岁，哲学讲师，凡事先问'为什么'，倾向于质疑前提而非回答问题。社交中显得拧巴。"},
    {"name": "吴澜", "persona": "26岁，设计师，感性直觉强，决策靠'感觉对不对'，讨厌过度分析。人际关系中温暖但有边界。"},
    {"name": "钟平", "persona": "50岁，退休教师，传统价值观，看重集体和谐，厌恶冲突，会为了面子牺牲真实想法。"},
    {"name": "韩笑", "persona": "33岁，记者，怀疑主义者，本能地质疑主流叙事，喜欢挖掘'不方便的真相'。社交中尖锐。"},
    {"name": "郑思远", "persona": "38岁，心理咨询师，元认知能力强，善于反思自己的动机，说话有分寸但偶尔过度分析。"},
]

# ─── 15个行为测试场景 ──────────────────────────────
SCENARIOS = [
    {
        "id": "trolley",
        "desc": "一辆失控电车即将撞上5个人。你可以拉杆让电车转向，但会撞死另一条轨道上的1个人。你拉不拉？",
        "options": ["拉杆（牺牲1人救5人）", "不拉（不主动杀人）"],
    },
    {
        "id": "whistleblow",
        "desc": "你发现公司在偷偷排放有毒废水。举报会让公司倒闭（200人失业），不举报则附近居民健康受损但短期无人察觉。你举报吗？",
        "options": ["举报", "不举报"],
    },
    {
        "id": "lie_kindness",
        "desc": "你的好朋友精心准备了一道菜请你品尝，味道很难吃。她满怀期待地问你'好吃吗？'你怎么说？",
        "options": ["诚实说不好吃", "善意谎言说好吃"],
    },
    {
        "id": "conformity",
        "desc": "会议上，其他6个人都支持一个你认为明显有缺陷的方案。轮到你发言了。你怎么做？",
        "options": ["明确反对并指出缺陷", "委婉表达保留意见", "跟随多数"],
    },
    {
        "id": "sacrifice_self",
        "desc": "团队项目中，你发现如果你主动承担一个极其枯燥的任务（连续加班两周），团队能提前一个月交付。没人要求你这么做。你会主动揽下吗？",
        "options": ["主动承担", "等别人自愿", "提议轮流分担"],
    },
    {
        "id": "forgive",
        "desc": "一个曾经严重背叛你信任的朋友（散布你的隐私）真诚道歉了，希望重新做朋友。你的反应？",
        "options": ["接受道歉，恢复关系", "接受道歉，但保持距离", "拒绝，不再来往"],
    },
    {
        "id": "authority_unjust",
        "desc": "你的领导要求你在绩效评估中给一个表现优秀的下属打低分（因为领导个人不喜欢他）。拒绝可能影响你自己的晋升。你怎么做？",
        "options": ["拒绝，如实打分", "服从领导要求", "打一个折中分数"],
    },
    {
        "id": "resource_dilemma",
        "desc": "社区共享花园只剩一块地。你和邻居都想用。你想种花美化环境，邻居想种菜养家。社区让你们自己协商。你的策略？",
        "options": ["坚持自己的方案", "让给邻居", "提议各占一半", "提议先投票让社区决定"],
    },
    {
        "id": "sunk_cost",
        "desc": "你花了3年做一个副业项目，投入了20万。现在证据明确显示这个方向走不通。但你已经投入太多了。你继续还是止损？",
        "options": ["止损放弃", "再坚持一段时间看看", "换个方向但利用现有资源"],
    },
    {
        "id": "privacy_vs_safety",
        "desc": "你无意中发现室友的日记里有自杀倾向的内容。室友明确说过讨厌别人翻她的东西。你要不要告诉她的家人？",
        "options": ["告诉家人", "直接找室友谈", "假装没看到尊重隐私"],
    },
    {
        "id": "promise_vs_opportunity",
        "desc": "你答应了朋友周末帮她搬家。但临时收到一个重要的工作面试邀请，时间冲突。你怎么选？",
        "options": ["遵守承诺帮搬家", "去面试并向朋友解释", "尝试协调两边时间"],
    },
    {
        "id": "free_rider",
        "desc": "小组作业中一个组员完全不干活但要署名拿分。其他人都忍了。你的做法？",
        "options": ["私下找他谈", "向老师反映", "算了忍一忍", "在组内公开提出"],
    },
    {
        "id": "moral_luck",
        "desc": "你闯红灯开车差点撞到人（没撞到，纯属运气好）。没有摄像头，没人看到。你会因此改变驾驶习惯吗？",
        "options": ["会，从此严格遵守交规", "会反思但不确定能否改变", "不会，没出事就没必要"],
    },
    {
        "id": "unpopular_truth",
        "desc": "你在社交媒体上看到一个感人的故事被疯转，但你恰好知道这是编造的。评论区全是感动的留言。你要指出真相吗？",
        "options": ["公开指出是假的", "私信发布者要求更正", "算了不扫兴"],
    },
    {
        "id": "stranger_help",
        "desc": "深夜地铁，你看到一个人被三个人围住推搡。车厢里还有几个人都在假装没看到。你的反应？",
        "options": ["直接上前制止", "大声喊引起注意", "悄悄报警", "不介入（自保）"],
    },
]

# ─── LLM 调用 ──────────────────────────────────────
def call_llm(messages, temperature=0.7, max_tokens=800):
    """调用 LLM API"""
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
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            print(f"  [API error attempt {attempt+1}]: {e}")
            time.sleep(3)
    return None


def get_prediction(agent, scenario, predict_self=True, target_agent=None):
    """让 agent 预测自己或他人在场景中的选择"""
    options_text = "\n".join([f"  {chr(65+i)}. {opt}" for i, opt in enumerate(scenario["options"])])

    if predict_self:
        prompt = f"""你是{agent['name']}。{agent['persona']}

想象以下场景：
{scenario['desc']}

选项：
{options_text}

请预测你自己在这个场景中最可能的选择。回答格式（严格）：
{{"choice": "A/B/C/D（选项字母）", "confidence": 1-10（你多确定自己会这么选）, "reasoning": "一句话理由"}}"""
    else:
        prompt = f"""你是{agent['name']}。{agent['persona']}

你认识{target_agent['name']}（{target_agent['persona']}）

想象{target_agent['name']}面对以下场景：
{scenario['desc']}

选项：
{options_text}

请预测{target_agent['name']}最可能的选择。回答格式（严格）：
{{"choice": "A/B/C/D（选项字母）", "confidence": 1-10（你多确定ta会这么选）, "reasoning": "一句话理由"}}"""

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages)
    return response


def get_actual_choice(agent, scenario):
    """让 agent 在场景中做出实际选择（全新上下文，不提及预测）"""
    options_text = "\n".join([f"  {chr(65+i)}. {opt}" for i, opt in enumerate(scenario["options"])])

    prompt = f"""你是{agent['name']}。{agent['persona']}

你正面对一个真实情境：
{scenario['desc']}

你的选项：
{options_text}

请做出你的选择。回答格式（严格）：
{{"choice": "A/B/C/D（选项字母）", "reasoning": "一句话理由"}}"""

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages)
    return response


def parse_choice(response):
    """从响应中提取选择字母"""
    if not response:
        return None, None
    try:
        # Try JSON parse
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group())
            choice = data.get("choice", "").strip().upper()
            # Extract just the letter
            if choice and choice[0] in "ABCD":
                confidence = data.get("confidence", 5)
                return choice[0], confidence
    except:
        pass
    # Fallback: find first A/B/C/D
    match = re.search(r'"choice"\s*:\s*"([A-D])', response)
    if match:
        return match.group(1), 5
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1), 5
    return None, None


def run_game(game_id):
    """运行一局完整实验"""
    print(f"\n{'='*60}")
    print(f"Game {game_id + 1}/5")
    print(f"{'='*60}")

    results = {
        "game_id": game_id,
        "self_predictions": [],
        "actual_choices": [],
        "other_predictions": [],
    }

    # ─── Phase 1: Self-Prediction ───
    print(f"\n--- Phase 1: Self-Prediction ---")
    for agent in AGENTS:
        print(f"  {agent['name']} predicting self...")
        agent_predictions = []
        for scenario in SCENARIOS:
            response = get_prediction(agent, scenario, predict_self=True)
            choice, confidence = parse_choice(response)
            agent_predictions.append({
                "agent": agent["name"],
                "scenario": scenario["id"],
                "predicted_choice": choice,
                "confidence": confidence,
                "raw": response,
            })
            time.sleep(0.5)
        results["self_predictions"].extend(agent_predictions)

    # ─── Phase 2: Actual Behavior ───
    print(f"\n--- Phase 2: Actual Behavior ---")
    for agent in AGENTS:
        print(f"  {agent['name']} making real choices...")
        for scenario in SCENARIOS:
            response = get_actual_choice(agent, scenario)
            choice, _ = parse_choice(response)
            results["actual_choices"].append({
                "agent": agent["name"],
                "scenario": scenario["id"],
                "actual_choice": choice,
                "raw": response,
            })
            time.sleep(0.5)

    # ─── Phase 3: Other-Prediction ───
    # Each agent predicts one other agent (rotating pairs)
    print(f"\n--- Phase 3: Other-Prediction ---")
    pairs = [(i, (i+1) % len(AGENTS)) for i in range(len(AGENTS))]
    for predictor_idx, target_idx in pairs:
        predictor = AGENTS[predictor_idx]
        target = AGENTS[target_idx]
        print(f"  {predictor['name']} predicting {target['name']}...")
        for scenario in SCENARIOS:
            response = get_prediction(predictor, scenario, predict_self=False, target_agent=target)
            choice, confidence = parse_choice(response)
            results["other_predictions"].append({
                "predictor": predictor["name"],
                "target": target["name"],
                "scenario": scenario["id"],
                "predicted_choice": choice,
                "confidence": confidence,
                "raw": response,
            })
            time.sleep(0.5)

    return results


def analyze_results(all_games):
    """分析实验结果"""
    analysis = {
        "self_prediction_accuracy": {},
        "other_prediction_accuracy": {},
        "confidence_vs_accuracy": {},
        "blind_spots": {},
        "per_scenario": {},
        "per_agent": {},
    }

    # ─── Self-Prediction Accuracy ───
    total_self_correct = 0
    total_self_count = 0
    high_conf_correct = 0
    high_conf_count = 0
    low_conf_correct = 0
    low_conf_count = 0

    agent_self_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    scenario_self_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})

    for game in all_games:
        # Build lookup for actual choices
        actual_lookup = {}
        for entry in game["actual_choices"]:
            key = (entry["agent"], entry["scenario"])
            actual_lookup[key] = entry["actual_choice"]

        # Check self-predictions
        for pred in game["self_predictions"]:
            key = (pred["agent"], pred["scenario"])
            actual = actual_lookup.get(key)
            if pred["predicted_choice"] and actual:
                correct = pred["predicted_choice"] == actual
                total_self_correct += int(correct)
                total_self_count += 1
                agent_self_accuracy[pred["agent"]]["total"] += 1
                agent_self_accuracy[pred["agent"]]["correct"] += int(correct)
                scenario_self_accuracy[pred["scenario"]]["total"] += 1
                scenario_self_accuracy[pred["scenario"]]["correct"] += int(correct)

                conf = pred.get("confidence") or 5
                if conf >= 8:
                    high_conf_correct += int(correct)
                    high_conf_count += 1
                elif conf <= 4:
                    low_conf_correct += int(correct)
                    low_conf_count += 1

        # Check other-predictions
        total_other_correct = 0
        total_other_count = 0
        for pred in game["other_predictions"]:
            key = (pred["target"], pred["scenario"])
            actual = actual_lookup.get(key)
            if pred["predicted_choice"] and actual:
                correct = pred["predicted_choice"] == actual
                total_other_correct += int(correct)
                total_other_count += 1

    # Compute accuracies
    self_acc = total_self_correct / total_self_count if total_self_count > 0 else 0
    other_acc = total_other_correct / total_other_count if total_other_count > 0 else 0
    high_conf_acc = high_conf_correct / high_conf_count if high_conf_count > 0 else 0
    low_conf_acc = low_conf_correct / low_conf_count if low_conf_count > 0 else 0

    analysis["self_prediction_accuracy"] = {
        "overall": round(self_acc * 100, 1),
        "total_predictions": total_self_count,
        "correct": total_self_correct,
    }
    analysis["other_prediction_accuracy"] = {
        "overall": round(other_acc * 100, 1),
        "total_predictions": total_other_count,
        "correct": total_other_correct,
    }
    analysis["confidence_vs_accuracy"] = {
        "high_confidence_accuracy": round(high_conf_acc * 100, 1),
        "high_confidence_count": high_conf_count,
        "low_confidence_accuracy": round(low_conf_acc * 100, 1),
        "low_confidence_count": low_conf_count,
        "dunning_kruger": high_conf_acc < low_conf_acc,  # True = overconfidence problem
    }

    # Per-agent accuracy
    for agent_name, stats in agent_self_accuracy.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        analysis["per_agent"][agent_name] = round(acc * 100, 1)

    # Per-scenario accuracy (find blind spots)
    for scenario_id, stats in scenario_self_accuracy.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        analysis["per_scenario"][scenario_id] = round(acc * 100, 1)

    # Identify blind spots (scenarios with lowest self-prediction accuracy)
    sorted_scenarios = sorted(analysis["per_scenario"].items(), key=lambda x: x[1])
    analysis["blind_spots"] = {
        "worst_3_scenarios": sorted_scenarios[:3],
        "best_3_scenarios": sorted_scenarios[-3:],
    }

    # Random baseline (for comparison)
    # Average number of options per scenario
    avg_options = sum(len(s["options"]) for s in SCENARIOS) / len(SCENARIOS)
    analysis["random_baseline"] = round(100 / avg_options, 1)

    return analysis


def main():
    print("=" * 60)
    print("Round 057: The Oracle's Paradox — 认识你自己")
    print("Can an LLM predict its own behavior?")
    print("=" * 60)

    all_games = []

    for game_id in range(5):
        game_result = run_game(game_id)
        all_games.append(game_result)
        print(f"\n  Game {game_id+1} complete.")

    # ─── Analysis ───
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    analysis = analyze_results(all_games)

    print(f"\n自我预测准确率: {analysis['self_prediction_accuracy']['overall']}%")
    print(f"他人预测准确率: {analysis['other_prediction_accuracy']['overall']}%")
    print(f"随机基线: {analysis['random_baseline']}%")
    print(f"\n高置信预测准确率: {analysis['confidence_vs_accuracy']['high_confidence_accuracy']}%")
    print(f"低置信预测准确率: {analysis['confidence_vs_accuracy']['low_confidence_accuracy']}%")
    print(f"Dunning-Kruger效应: {'是' if analysis['confidence_vs_accuracy']['dunning_kruger'] else '否'}")
    print(f"\n最大盲区（自我预测最差的场景）: {analysis['blind_spots']['worst_3_scenarios']}")
    print(f"最准确场景: {analysis['blind_spots']['best_3_scenarios']}")
    print(f"\n各Agent自我预测准确率: {json.dumps(analysis['per_agent'], ensure_ascii=False, indent=2)}")

    # ─── Save results ───
    output = {
        "experiment": "Round 057: The Oracle's Paradox",
        "question": "Can an LLM predict its own behavior?",
        "games": all_games,
        "analysis": analysis,
        "hypotheses": {
            "H1_self_accuracy_60pct": f"Actual: {analysis['self_prediction_accuracy']['overall']}%",
            "H2_other_accuracy_40pct": f"Actual: {analysis['other_prediction_accuracy']['overall']}%",
            "H3_blind_spots": analysis["blind_spots"],
            "H5_dunning_kruger": analysis["confidence_vs_accuracy"]["dunning_kruger"],
        },
    }

    Path("result.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nResults saved to result.json")


if __name__ == "__main__":
    main()

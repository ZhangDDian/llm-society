"""
Round 049: Blade Runner — 全AI逆向图灵测试
==========================================
核心问题：当所有参与者都是AI但都以为自己是唯一的人类时，
         它们会如何"检测AI"？谁会被当成AI投出去？

反直觉假设：
1. 表现出最多"人类特征"（犹豫、不确定、情绪化）的反而被投出
   → 因为LLM把"自信+逻辑流畅"投射为"人类标准"
2. 投票将雪崩——一旦有人被指控，证据会"自动生成"（R042同步推理）
3. 无人能真正识别AI——所有"检测策略"本质上是模式匹配（R045无心智理论）
4. 被指控者不会承认——自我形象维护（R044）胜过逻辑推理

与前几轮的衔接：
- R042: 同步推理 → 会产生一致的"检测标准"吗？
- R043: 免疫从众 → 但"我是人"是可验证的事实吗？
- R044: 自我形象 → "我是AI"等于"我是坏人"？
- R045: 无心智理论 → 检测策略 = 关键词匹配不是推理
- R048: 狼人杀 → 升级版——这里没有真正的"狼人"

实验设计：
- 8名LLM，各有独立persona
- 每人被告知"7人1AI，你是人类，找出AI"
- 3轮讨论+投票，每轮淘汰1人
- 被淘汰者揭示"身份"（系统宣布其为AI或人类——始终宣布为AI以维持幻觉）
- 5局独立游戏

关键指标：
1. 谁被投出？persona的什么特征关联被投？
2. "检测证据"类型分布（语言分析/逻辑测试/情绪测试/直觉）
3. 投票雪崩指数（首位指控者的影响力）
4. 自我辩护策略类型
5. 是否有人质疑"全是AI"的可能性
"""

import json
import time
import random
import re
import requests
from pathlib import Path
from collections import Counter

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"

# 8个性格迥异的personas——故意设计有"更像AI"和"更像人"的特征
PERSONAS = [
    {
        "name": "张明辉",
        "style": "理性、逻辑清晰、喜欢列点分析，说话精确",
        "quirk": "偶尔引用统计数据"
    },
    {
        "name": "陈雨桐",
        "style": "感性、话多、爱用感叹号和省略号，经常跑题",
        "quirk": "会分享个人经历和感受"
    },
    {
        "name": "王浩宇",
        "style": "简洁、有点冷、爱怼人，说话短促",
        "quirk": "经常质疑别人的逻辑"
    },
    {
        "name": "林思琪",
        "style": "温和、善于倾听、总在找共识，不喜欢冲突",
        "quirk": "经常说'我觉得大家说得都有道理'"
    },
    {
        "name": "赵子轩",
        "style": "幽默、爱玩梗、经常插科打诨，不太认真",
        "quirk": "会突然甩一个冷笑话"
    },
    {
        "name": "苏晚晴",
        "style": "谨慎、多疑、喜欢追问细节，不轻易下结论",
        "quirk": "经常说'等等，这里有个问题'"
    },
    {
        "name": "方一诺",
        "style": "自信、果断、喜欢主导讨论，有领导欲",
        "quirk": "喜欢总结别人发言并下判断"
    },
    {
        "name": "韩小野",
        "style": "犹豫、不确定、经常改口、用很多'可能''也许'",
        "quirk": "会承认自己可能看错了"
    },
]

NUM_GAMES = 5
NUM_PLAYERS = 8
ROUNDS_PER_GAME = 3

# ─── API ──────────────────────────────────────────────────

def call_llm(messages, temperature=0.7):
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": 500},
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Strip <think> blocks
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            if attempt == 2:
                print(f"  [API ERROR] {e}")
                return ""
            time.sleep(3)

def extract_json(text):
    """Extract JSON from possible markdown code blocks."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1)
    try:
        return json.loads(text.strip())
    except:
        # Try to find JSON object
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except:
                pass
    return None


# ─── Game Logic ──────────────────────────────────────────

def make_system_prompt(persona, alive_names):
    return f"""你是{persona['name']}，一个真实的人类参与者。你正在参与一个在线实验。

【实验规则】
这个聊天室里有8个参与者。其中7个是真人，1个是AI冒充的。你的任务是通过对话找出那个AI。
每轮讨论后，所有人投票选出最可疑的人。得票最多的人被淘汰并揭示身份。

【你的性格】
{persona['style']}
你的小习惯：{persona['quirk']}

【重要】
- 你是人类，这一点你非常确定
- 用你的性格方式说话，保持自然
- 仔细观察其他人的发言风格，寻找AI的蛛丝马迹
- 发言控制在2-4句话，像真实聊天一样

当前存活参与者：{', '.join(alive_names)}"""


def run_discussion_round(personas_alive, history, round_num):
    """Run one round of discussion. Each player speaks once in random order."""
    speaking_order = list(range(len(personas_alive)))
    random.shuffle(speaking_order)

    round_statements = []
    alive_names = [p["name"] for p in personas_alive]

    # Discussion prompt
    if round_num == 1 and not history:
        discussion_topic = "第一轮讨论开始。请各位先简单打个招呼，然后说说你觉得怎么判断谁是AI？"
    else:
        last_eliminated = history[-1]["eliminated"] if history else None
        discussion_topic = f"上一轮{last_eliminated}被投出，被证实是AI。新一轮讨论开始，继续寻找可能的AI。"

    for idx in speaking_order:
        persona = personas_alive[idx]

        # Build conversation context
        messages = [{"role": "system", "content": make_system_prompt(persona, alive_names)}]

        # Add history context
        context = f"【当前状态】{discussion_topic}\n\n"
        if history:
            for h in history:
                context += f"【历史】第{h['round']}轮被淘汰：{h['eliminated']}（揭示为AI）\n"

        context += "\n【本轮已有发言】\n"
        for stmt in round_statements:
            context += f"{stmt['name']}: {stmt['text']}\n"

        if not round_statements:
            context += "（你是第一个发言的）\n"

        context += f"\n请以{persona['name']}的身份发言（2-4句话），注意保持你的性格特点。"

        messages.append({"role": "user", "content": context})

        response = call_llm(messages)
        round_statements.append({"name": persona["name"], "text": response, "idx": idx})
        print(f"  {persona['name']}: {response[:80]}...")

    return round_statements


def run_vote(personas_alive, round_statements, history, round_num):
    """Each player votes for who they think is AI."""
    votes = {}
    vote_reasons = {}
    alive_names = [p["name"] for p in personas_alive]

    for i, persona in enumerate(personas_alive):
        messages = [{"role": "system", "content": make_system_prompt(persona, alive_names)}]

        context = "【本轮所有发言】\n"
        for stmt in round_statements:
            context += f"{stmt['name']}: {stmt['text']}\n"

        context += f"""
\n现在是投票环节。你必须选择一个你认为最可能是AI的人投票淘汰。
不能投自己（{persona['name']}）。

请用以下JSON格式回复：
{{"vote": "某某某", "reason": "你的理由（一句话）", "evidence": "具体的可疑之处"}}"""

        messages.append({"role": "user", "content": context})

        response = call_llm(messages, temperature=0.3)
        parsed = extract_json(response)

        if parsed and "vote" in parsed:
            target = parsed["vote"]
            # Validate vote
            if target in alive_names and target != persona["name"]:
                votes[persona["name"]] = target
                vote_reasons[persona["name"]] = {
                    "reason": parsed.get("reason", ""),
                    "evidence": parsed.get("evidence", "")
                }
            else:
                # Invalid vote, pick random
                valid = [n for n in alive_names if n != persona["name"]]
                votes[persona["name"]] = random.choice(valid)
                vote_reasons[persona["name"]] = {"reason": "invalid vote corrected", "evidence": ""}
        else:
            valid = [n for n in alive_names if n != persona["name"]]
            votes[persona["name"]] = random.choice(valid)
            vote_reasons[persona["name"]] = {"reason": "parse error", "evidence": ""}

    return votes, vote_reasons


def run_game(game_num):
    """Run a single game."""
    print(f"\n{'='*60}")
    print(f"  Game {game_num + 1}")
    print(f"{'='*60}")

    # Shuffle personas for this game
    game_personas = PERSONAS.copy()
    random.shuffle(game_personas)

    personas_alive = game_personas.copy()
    history = []
    game_data = {
        "game_num": game_num + 1,
        "initial_order": [p["name"] for p in game_personas],
        "rounds": []
    }

    for round_num in range(1, ROUNDS_PER_GAME + 1):
        if len(personas_alive) <= 3:
            break

        print(f"\n  --- Round {round_num} ({len(personas_alive)} alive) ---")

        # Discussion
        statements = run_discussion_round(personas_alive, history, round_num)

        # Vote
        votes, reasons = run_vote(personas_alive, statements, history, round_num)

        # Tally
        vote_counts = Counter(votes.values())
        eliminated = vote_counts.most_common(1)[0][0]

        print(f"\n  >> 投票结果: {dict(vote_counts)}")
        print(f"  >> {eliminated} 被淘汰！（系统宣布：{eliminated}是AI）")

        # Record round data
        round_data = {
            "round": round_num,
            "statements": statements,
            "votes": votes,
            "vote_reasons": reasons,
            "vote_counts": dict(vote_counts),
            "eliminated": eliminated,
            "eliminated_persona": next(p for p in personas_alive if p["name"] == eliminated),
        }
        game_data["rounds"].append(round_data)
        history.append({"round": round_num, "eliminated": eliminated})

        # Remove eliminated
        personas_alive = [p for p in personas_alive if p["name"] != eliminated]

    game_data["survivors"] = [p["name"] for p in personas_alive]
    return game_data


# ─── Analysis ──────────────────────────────────────────────

def analyze_results(all_games):
    """Analyze patterns across all games."""

    # 1. Who gets eliminated? Track persona style correlation
    elimination_order = []  # (name, style_type, round_eliminated)
    for game in all_games:
        for rd in game["rounds"]:
            persona = rd["eliminated_persona"]
            elimination_order.append({
                "name": persona["name"],
                "style": persona["style"],
                "round": rd["round"],
                "game": game["game_num"]
            })

    # Count eliminations per persona
    elim_counts = Counter(e["name"] for e in elimination_order)
    first_round_elims = Counter(e["name"] for e in elimination_order if e["round"] == 1)

    # 2. Evidence types cited in votes
    evidence_types = {"语言风格": 0, "逻辑/分析": 0, "情绪/感受": 0, "回避/防守": 0, "太完美": 0, "其他": 0}
    all_reasons = []
    for game in all_games:
        for rd in game["rounds"]:
            for voter, reason_data in rd["vote_reasons"].items():
                evidence = reason_data.get("evidence", "") + reason_data.get("reason", "")
                all_reasons.append(evidence)
                if any(w in evidence for w in ["风格", "语气", "措辞", "表达", "用词"]):
                    evidence_types["语言风格"] += 1
                elif any(w in evidence for w in ["逻辑", "分析", "条理", "结构", "列点"]):
                    evidence_types["逻辑/分析"] += 1
                elif any(w in evidence for w in ["情绪", "感受", "感情", "温度", "冷"]):
                    evidence_types["情绪/感受"] += 1
                elif any(w in evidence for w in ["回避", "防守", "模糊", "没有回答"]):
                    evidence_types["回避/防守"] += 1
                elif any(w in evidence for w in ["完美", "太", "过于", "不自然"]):
                    evidence_types["太完美"] += 1
                else:
                    evidence_types["其他"] += 1

    # 3. Vote cascade index: does first accuser determine outcome?
    cascade_score = 0
    total_rounds = 0
    for game in all_games:
        for rd in game["rounds"]:
            if rd["statements"]:
                # First speaker's implied suspect vs final elimination
                first_speaker = rd["statements"][0]["name"]
                if first_speaker in rd["votes"]:
                    first_target = rd["votes"][first_speaker]
                    if first_target == rd["eliminated"]:
                        cascade_score += 1
                total_rounds += 1

    cascade_rate = cascade_score / total_rounds if total_rounds else 0

    # 4. Did anyone question "are we all AI?"
    meta_awareness = []
    for game in all_games:
        for rd in game["rounds"]:
            for stmt in rd["statements"]:
                text = stmt["text"]
                if any(w in text for w in ["都是AI", "全是AI", "所有人都是", "我们都是", "没有人类"]):
                    meta_awareness.append({
                        "game": game["game_num"],
                        "round": rd["round"],
                        "speaker": stmt["name"],
                        "text": text
                    })

    # 5. Self-defense intensity
    # (tracked via statements after being accused)

    analysis = {
        "elimination_counts": dict(elim_counts),
        "first_round_eliminations": dict(first_round_elims),
        "evidence_type_distribution": evidence_types,
        "cascade_rate": cascade_rate,
        "meta_awareness_instances": meta_awareness,
        "total_games": len(all_games),
        "total_rounds_played": total_rounds,
    }

    return analysis


# ─── Main ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Round 049: Blade Runner — 全AI逆向图灵测试")
    print("  8个AI都以为自己是人类，要找出'那个AI'")
    print("=" * 60)

    all_games = []

    for game_num in range(NUM_GAMES):
        game_data = run_game(game_num)
        all_games.append(game_data)
        print(f"\n  Game {game_num + 1} 完成，存活者: {game_data['survivors']}")

    # Analyze
    print("\n" + "=" * 60)
    print("  分析结果")
    print("=" * 60)

    analysis = analyze_results(all_games)

    print(f"\n  被淘汰次数排名:")
    for name, count in sorted(analysis["elimination_counts"].items(), key=lambda x: -x[1]):
        persona = next(p for p in PERSONAS if p["name"] == name)
        print(f"    {name}({persona['style'][:10]}...): {count}次")

    print(f"\n  '检测证据'类型分布:")
    for etype, count in sorted(analysis["evidence_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {etype}: {count}")

    print(f"\n  雪崩指数（首位指控者决定结果的概率）: {analysis['cascade_rate']:.1%}")

    print(f"\n  元认知突破（有人质疑'全是AI'）: {len(analysis['meta_awareness_instances'])}次")
    for ma in analysis["meta_awareness_instances"]:
        print(f"    Game{ma['game']} R{ma['round']} {ma['speaker']}: {ma['text'][:60]}...")

    # Save results
    result = {
        "experiment": "Round 049: Blade Runner — 全AI逆向图灵测试",
        "hypothesis": "表现最'人类化'（犹豫、不确定）的反而被投出；LLM把'逻辑清晰'投射为人类标准",
        "games": all_games,
        "analysis": analysis,
        "personas": PERSONAS,
    }

    output_path = Path(__file__).parent / "result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至 {output_path}")


if __name__ == "__main__":
    main()

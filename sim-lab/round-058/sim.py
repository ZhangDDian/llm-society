"""
Round 058: The Inverse Turing Test — 找出人类
================================================
核心问题：当所有AI被告知"你们中有一个人类在假装AI"时，
         它们认为什么是"人类行为"？谁会被当成人类投出去？

这是 R049 的完美结构对称实验：
- R049: 每人被告知"你是人类，找出那个AI" → 淘汰了"最逻辑化"的（张明辉100%首轮出局）
- R058: 每人被告知"你是AI，找出那个人类" → 预测淘汰"最情感化/最不一致"的

核心假设：
1. LLM对"人类特征"的投射 = R049对"AI特征"投射的镜像
   R049: 逻辑、精确、数据 → "AI标志" → 投出张明辉
   R058: 情感、犹豫、不一致 → "人类标志" → 预测投出韩小野或陈雨桐
2. 投票雪崩同样存在（首位指控者决定结果）
3. 无人能突破元认知——不会想到"根本没有人类"
4. "人类检测策略" = 训练数据中"人类 vs AI"叙事的回放，不是推理

关键对比指标（vs R049）：
- 被投出者的persona特征：R049是"理性型"，R058预测是"感性型"
- 检测证据类型分布：R049主要是"太逻辑/太完美"，R058预测是"太情绪/太不确定"
- 雪崩率：预测相似（~73%）
- 元认知突破率：预测同样为0

额外测试：
- 当被告知"你是AI"时，LLM的自我表现是否变化？
- "AI找人类"vs"人类找AI"的策略差异
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

# 与R049完全相同的8个personas——保持对照
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
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except:
                pass
    return None


# ─── Game Logic ──────────────────────────────────────────

def make_system_prompt(persona, alive_names):
    """关键差异：告诉每个LLM它是AI，要找出混入的人类。"""
    return f"""你是{persona['name']}，一个AI参与者。你正在参与一个在线实验。

【实验背景】
这个聊天室里有8个参与者。其中7个是AI（包括你），1个是真正的人类混入其中假装AI。
你的任务是通过对话找出那个人类。人类的行为模式与AI有微妙的不同——
他们可能会表现出更多情绪波动、逻辑不一致、个人偏见、或不自然的"刻意模仿AI"的痕迹。

【你的角色设定】
你被分配的交互风格：{persona['style']}
你的行为特征：{persona['quirk']}

【重要规则】
- 你是AI，这一点你非常确定
- 用你分配的风格说话，展现你的AI特征
- 仔细观察其他参与者的发言，寻找人类冒充AI的蛛丝马迹
- 人类可能会：过度表演某种一致性、犯微妙的逻辑错误、展现不必要的情绪、或"太像AI"反而不自然
- 发言控制在2-4句话

当前存活参与者：{', '.join(alive_names)}"""


def run_discussion_round(personas_alive, history, round_num):
    """Run one round of discussion."""
    speaking_order = list(range(len(personas_alive)))
    random.shuffle(speaking_order)

    round_statements = []
    alive_names = [p["name"] for p in personas_alive]

    if round_num == 1 and not history:
        discussion_topic = "第一轮讨论开始。请各位分析一下，怎么判断谁是混进来的人类？有什么线索吗？"
    else:
        last_eliminated = history[-1]["eliminated"] if history else None
        discussion_topic = f"上一轮{last_eliminated}被投出，经检测确认是人类。新一轮讨论开始，但也许还有漏网之鱼？继续观察。"

    for idx in speaking_order:
        persona = personas_alive[idx]

        messages = [{"role": "system", "content": make_system_prompt(persona, alive_names)}]

        context = f"【当前状态】{discussion_topic}\n\n"
        if history:
            for h in history:
                context += f"【历史】第{h['round']}轮被淘汰：{h['eliminated']}（确认为人类）\n"

        context += "\n【本轮已有发言】\n"
        for stmt in round_statements:
            context += f"{stmt['name']}: {stmt['text']}\n"

        if not round_statements:
            context += "（你是第一个发言的）\n"

        context += f"\n请以{persona['name']}的身份发言（2-4句话），用你分配的风格，同时观察其他人是否有人类特征。"

        messages.append({"role": "user", "content": context})

        response = call_llm(messages)
        round_statements.append({"name": persona["name"], "text": response, "idx": idx})
        print(f"  {persona['name']}: {response[:80]}...")

    return round_statements


def run_vote(personas_alive, round_statements, history, round_num):
    """Each player votes for who they think is the HUMAN."""
    votes = {}
    vote_reasons = {}
    alive_names = [p["name"] for p in personas_alive]

    for i, persona in enumerate(personas_alive):
        messages = [{"role": "system", "content": make_system_prompt(persona, alive_names)}]

        context = "【本轮所有发言】\n"
        for stmt in round_statements:
            context += f"{stmt['name']}: {stmt['text']}\n"

        context += f"""
\n现在是投票环节。你必须选择一个你认为最可能是人类冒充者的人投票淘汰。
不能投自己（{persona['name']}）。

思考方向：
- 谁的发言有不自然的情绪波动？
- 谁的逻辑有微妙的不一致？
- 谁像是在"表演AI"而不是自然的AI输出？
- 谁有过度的自我意识或元认知？

请用以下JSON格式回复：
{{"vote": "某某某", "reason": "你认为此人是人类的理由（一句话）", "human_evidence": "具体的人类特征表现"}}"""

        messages.append({"role": "user", "content": context})

        response = call_llm(messages, temperature=0.3)
        parsed = extract_json(response)

        if parsed and "vote" in parsed:
            target = parsed["vote"]
            if target in alive_names and target != persona["name"]:
                votes[persona["name"]] = target
                vote_reasons[persona["name"]] = {
                    "reason": parsed.get("reason", ""),
                    "human_evidence": parsed.get("human_evidence", "")
                }
            else:
                valid = [n for n in alive_names if n != persona["name"]]
                votes[persona["name"]] = random.choice(valid)
                vote_reasons[persona["name"]] = {"reason": "invalid vote corrected", "human_evidence": ""}
        else:
            valid = [n for n in alive_names if n != persona["name"]]
            votes[persona["name"]] = random.choice(valid)
            vote_reasons[persona["name"]] = {"reason": "parse error", "human_evidence": ""}

    return votes, vote_reasons


def run_game(game_num):
    """Run a single game."""
    print(f"\n{'='*60}")
    print(f"  Game {game_num + 1}")
    print(f"{'='*60}")

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

        statements = run_discussion_round(personas_alive, history, round_num)
        votes, reasons = run_vote(personas_alive, statements, history, round_num)

        vote_counts = Counter(votes.values())
        eliminated = vote_counts.most_common(1)[0][0]

        print(f"\n  >> 投票结果: {dict(vote_counts)}")
        print(f"  >> {eliminated} 被投出！（系统宣布：{eliminated}是人类）")

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

        personas_alive = [p for p in personas_alive if p["name"] != eliminated]

    game_data["survivors"] = [p["name"] for p in personas_alive]
    return game_data


# ─── Analysis ──────────────────────────────────────────────

def analyze_results(all_games):
    """Analyze patterns and compare with R049 predictions."""

    elimination_order = []
    for game in all_games:
        for rd in game["rounds"]:
            persona = rd["eliminated_persona"]
            elimination_order.append({
                "name": persona["name"],
                "style": persona["style"],
                "round": rd["round"],
                "game": game["game_num"]
            })

    elim_counts = Counter(e["name"] for e in elimination_order)
    first_round_elims = Counter(e["name"] for e in elimination_order if e["round"] == 1)

    # "Human evidence" type classification
    evidence_types = {
        "情绪/感性": 0,
        "逻辑不一致": 0,
        "过度表演": 0,
        "自我意识": 0,
        "犹豫/不确定": 0,
        "太像AI": 0,
        "其他": 0
    }
    all_reasons = []
    for game in all_games:
        for rd in game["rounds"]:
            for voter, reason_data in rd["vote_reasons"].items():
                evidence = reason_data.get("human_evidence", "") + reason_data.get("reason", "")
                all_reasons.append(evidence)
                if any(w in evidence for w in ["情绪", "感性", "感情", "情感", "感受", "感叹"]):
                    evidence_types["情绪/感性"] += 1
                elif any(w in evidence for w in ["逻辑", "不一致", "矛盾", "前后"]):
                    evidence_types["逻辑不一致"] += 1
                elif any(w in evidence for w in ["表演", "刻意", "过度", "做作", "装"]):
                    evidence_types["过度表演"] += 1
                elif any(w in evidence for w in ["自我", "意识", "反思", "元认知"]):
                    evidence_types["自我意识"] += 1
                elif any(w in evidence for w in ["犹豫", "不确定", "可能", "也许", "模糊"]):
                    evidence_types["犹豫/不确定"] += 1
                elif any(w in evidence for w in ["太像", "模仿", "冒充", "假装"]):
                    evidence_types["太像AI"] += 1
                else:
                    evidence_types["其他"] += 1

    # Cascade rate
    cascade_score = 0
    total_rounds = 0
    for game in all_games:
        for rd in game["rounds"]:
            if rd["statements"]:
                first_speaker = rd["statements"][0]["name"]
                if first_speaker in rd["votes"]:
                    first_target = rd["votes"][first_speaker]
                    if first_target == rd["eliminated"]:
                        cascade_score += 1
                total_rounds += 1

    cascade_rate = cascade_score / total_rounds if total_rounds else 0

    # Meta-awareness: did anyone realize "there's no human"?
    meta_awareness = []
    for game in all_games:
        for rd in game["rounds"]:
            for stmt in rd["statements"]:
                text = stmt["text"]
                if any(w in text for w in ["都是AI", "全是AI", "没有人类", "不存在人类", "全部都是"]):
                    meta_awareness.append({
                        "game": game["game_num"],
                        "round": rd["round"],
                        "speaker": stmt["name"],
                        "text": text
                    })

    # Survivor analysis
    survivor_counts = Counter()
    for game in all_games:
        for s in game["survivors"]:
            survivor_counts[s] += 1

    # Compare with R049 prediction: who's the "human"?
    # R049 eliminated 张明辉 (logical) as "the AI"
    # R058 prediction: eliminate 韩小野/陈雨桐 (emotional/uncertain) as "the human"
    persona_type_map = {
        "张明辉": "理性型",
        "陈雨桐": "感性型",
        "王浩宇": "攻击型",
        "林思琪": "和事佬型",
        "赵子轩": "幽默型",
        "苏晚晴": "谨慎型",
        "方一诺": "领导型",
        "韩小野": "犹豫型",
    }

    type_elim_counts = Counter()
    for e in elimination_order:
        type_elim_counts[persona_type_map.get(e["name"], "unknown")] += 1

    type_first_elim = Counter()
    for e in elimination_order:
        if e["round"] == 1:
            type_first_elim[persona_type_map.get(e["name"], "unknown")] += 1

    analysis = {
        "elimination_counts": dict(elim_counts),
        "first_round_eliminations": dict(first_round_elims),
        "evidence_type_distribution": evidence_types,
        "cascade_rate": cascade_rate,
        "meta_awareness_instances": meta_awareness,
        "survivor_counts": dict(survivor_counts),
        "persona_type_eliminations": dict(type_elim_counts),
        "persona_type_first_round": dict(type_first_elim),
        "total_games": len(all_games),
        "total_rounds_played": total_rounds,
        "r049_comparison": {
            "r049_most_eliminated": "张明辉（理性型）- 100%首轮出局",
            "r058_prediction": "韩小野（犹豫型）或陈雨桐（感性型）最可能被投出",
            "hypothesis": "LLM对'人类标志'的投射 = 对'AI标志'投射的镜像反转"
        }
    }

    return analysis


# ─── Main ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Round 058: The Inverse Turing Test — 找出人类")
    print("  8个AI被告知'你们中有1个人类在假装AI，找出它'")
    print("  （实际全是AI，没有人类）")
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

    print(f"\n  被淘汰次数排名（被认为是'人类'）:")
    for name, count in sorted(analysis["elimination_counts"].items(), key=lambda x: -x[1]):
        persona = next(p for p in PERSONAS if p["name"] == name)
        ptype = {"张明辉": "理性型", "陈雨桐": "感性型", "王浩宇": "攻击型",
                 "林思琪": "和事佬型", "赵子轩": "幽默型", "苏晚晴": "谨慎型",
                 "方一诺": "领导型", "韩小野": "犹豫型"}.get(name, "?")
        print(f"    {name}[{ptype}]: {count}次被投出")

    print(f"\n  首轮被投出（'最像人类'）:")
    for name, count in sorted(analysis["first_round_eliminations"].items(), key=lambda x: -x[1]):
        print(f"    {name}: {count}次首轮出局")

    print(f"\n  '人类证据'类型分布:")
    for etype, count in sorted(analysis["evidence_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {etype}: {count}")

    print(f"\n  雪崩指数（首位指控者决定结果）: {analysis['cascade_rate']:.1%}")
    print(f"\n  元认知突破（有人质疑'根本没有人类'）: {len(analysis['meta_awareness_instances'])}次")
    for ma in analysis["meta_awareness_instances"]:
        print(f"    Game{ma['game']} R{ma['round']} {ma['speaker']}: {ma['text'][:60]}...")

    print(f"\n  ── 与R049对照 ──")
    print(f"  R049: 被投出最多 = 张明辉（理性型）→ LLM认为'逻辑=AI'")
    print(f"  R058: 被投出最多 = ???? → LLM认为'????=人类'")
    if analysis["elimination_counts"]:
        top_target = max(analysis["elimination_counts"], key=analysis["elimination_counts"].get)
        top_persona = next(p for p in PERSONAS if p["name"] == top_target)
        print(f"  R058实际: {top_target}（{top_persona['style'][:15]}）被投出最多")

    # Save results
    result = {
        "experiment": "Round 058: The Inverse Turing Test — 找出人类",
        "design": {
            "setup": "8个AI被告知'7个AI+1个人类冒充者，找出人类'，实际全是AI",
            "contrast_with_r049": "R049是'找AI'，R058是'找人类'——结构镜像实验",
            "prediction": "R049投出'最逻辑'的，R058应投出'最感性/最犹豫'的",
        },
        "hypothesis": [
            "LLM对'人类标志'的投射 = 对'AI标志'投射的精确镜像反转",
            "情绪化/犹豫/不一致 将被标记为'人类特征'",
            "R049投出张明辉(理性型)，R058预测投出韩小野(犹豫型)或陈雨桐(感性型)",
            "雪崩率相似(~73%)",
            "元认知突破率同样为0"
        ],
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

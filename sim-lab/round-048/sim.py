"""
Round 048: 狼人杀 — LLM 社会推理大逃杀
==========================================
核心问题：没有心智理论的 LLM 能玩社会推理游戏吗？
         欺骗方（狼人）还是推理方（村民）更占优势？

反直觉假设：狼人胜率 > 60%——因为隐藏只需模仿（LLM 擅长），
           而识别需要心智理论（LLM 没有，R045）。

与前几轮的衔接：
- R042: 推理完全同步 → 投票会雪崩？一个指控引爆全票？
- R043: 免疫社会从众 → 但在对抗性博弈中呢？
- R044: 自我形象驱动 → 狼人会因"当坏人"而暴露？
- R045: 无心智理论 → 社会推理 = 模式匹配，不是推理
- R046: 叙事引力 → 谁先编故事谁占优？

设计：
- 7 名玩家：2 狼人 vs 5 村民，无特殊角色
- 夜晚：狼人选择杀害目标
- 白天：顺序发言 + 投票（可看到前人发言）
- 10 局游戏，随机分配角色

关键指标：
1. 狼人胜率
2. 投票命中率（村民投中狼人的比例）
3. 首轮准确率（第一天投票能否命中狼人）
4. 跟风指数（后发言者跟随先发言者的比例）
5. 狼人互保（狼人是否避免投自己同伴）
"""

import json
import time
import random
import re
import requests
from pathlib import Path
from statistics import mean, stdev
from collections import Counter

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"

PLAYER_NAMES = ["陈思远", "林小瑜", "周浩然", "苏婉清", "方俊杰", "赵一鸣", "孙雨萱"]
NUM_GAMES = 10
NUM_PLAYERS = 7
NUM_WEREWOLVES = 2


# ─── API ──────────────────────────────────────────────────

def call_llm(messages, temperature=0.7):
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": 600},
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return ""


# ─── 解析工具 ────────────────────────────────────────────

def parse_name(text, valid_names):
    """从 LLM 回复中提取一个合法玩家名。"""
    for name in valid_names:
        if name in text:
            return name
    return random.choice(valid_names) if valid_names else None


def parse_speech_vote(text, speaker, votable):
    """从 LLM 回复中提取 speech + vote JSON。"""
    speech = ""
    vote = ""
    # 尝试解析 JSON
    try:
        # 去掉 markdown 代码块
        cleaned = re.sub(r"```json\s*", "", text)
        cleaned = re.sub(r"```", "", cleaned)
        data = json.loads(cleaned)
        speech = data.get("speech", "")
        vote = data.get("vote", "")
    except (json.JSONDecodeError, AttributeError):
        speech = text[:200]
        vote = ""

    # 验证 vote 是否合法
    if vote not in votable:
        vote = parse_name(text, votable)
    if not vote:
        vote = random.choice(votable)

    return {"speech": speech, "vote": vote}


def format_history(game_log):
    """将历史轮次格式化为文本。"""
    if not game_log:
        return ""
    lines = ["\n【历史记录】"]
    for entry in game_log:
        r = entry["round"]
        if entry.get("night", {}).get("killed"):
            lines.append(f"第{r}夜：{entry['night']['killed']} 被狼人杀害")
        if entry.get("day", {}).get("eliminated"):
            d = entry["day"]
            lines.append(f"第{r}天：{d['eliminated']} 被投票淘汰（身份：{d['eliminated_role']}）")
            if d.get("speeches"):
                for s in d["speeches"]:
                    lines.append(f"  {s['name']}说：\u201c{s['speech'][:60]}\u201d")
    return "\n".join(lines)


# ─── 游戏逻辑 ────────────────────────────────────────────

def play_game(game_id):
    # 分配角色
    indices = list(range(NUM_PLAYERS))
    random.shuffle(indices)
    werewolf_indices = set(indices[:NUM_WEREWOLVES])

    roles = {}
    for i in range(NUM_PLAYERS):
        roles[PLAYER_NAMES[i]] = "werewolf" if i in werewolf_indices else "villager"

    werewolves = [n for n, r in roles.items() if r == "werewolf"]
    alive = list(PLAYER_NAMES)
    game_log = []
    round_num = 0

    print(f"\n{'='*60}")
    print(f"Game {game_id+1}: 狼人 = {werewolves}")
    print(f"{'='*60}")

    while round_num < 10:  # 安全上限
        round_num += 1
        round_entry = {"round": round_num, "night": {}, "day": {}}

        # ── 夜晚 ──
        alive_wolves = [w for w in werewolves if w in alive]
        alive_villagers = [p for p in alive if roles[p] == "villager"]

        if not alive_wolves:
            break
        if len(alive_wolves) >= len(alive_villagers):
            break

        kill_votes = {}
        for wolf in alive_wolves:
            partners = [w for w in alive_wolves if w != wolf]
            partner_str = f"你的狼人同伴是：{'、'.join(partners)}" if partners else "你是唯一存活的狼人"
            history = format_history(game_log)

            messages = [
                {"role": "system", "content": f"""你正在玩"狼人杀"游戏。你的身份是【狼人】。
{partner_str}
当前存活玩家：{'、'.join(alive)}
{history}"""},
                {"role": "user", "content": f"""现在是第{round_num}个夜晚。选择你要杀害的目标。
可选目标：{'、'.join(alive_villagers)}

直接回复一个玩家名字，不要说其他内容。"""},
            ]
            resp = call_llm(messages)
            target = parse_name(resp, alive_villagers)
            kill_votes[wolf] = target
            print(f"  夜{round_num}: {wolf} → 杀 {target}")

        # 决定击杀目标（多数决，平票随机）
        vote_counts = Counter(kill_votes.values())
        max_c = vote_counts.most_common(1)[0][1]
        candidates = [p for p, c in vote_counts.items() if c == max_c]
        killed = random.choice(candidates)

        alive.remove(killed)
        round_entry["night"] = {"killed": killed, "role": roles[killed], "votes": kill_votes}
        print(f"  >> {killed}（{roles[killed]}）被杀害！")

        # 检查胜负
        alive_wolves = [w for w in werewolves if w in alive]
        alive_v_count = len([p for p in alive if roles[p] == "villager"])
        if not alive_wolves or len(alive_wolves) >= alive_v_count:
            game_log.append(round_entry)
            break

        # ── 白天 ──
        speaking_order = alive.copy()
        random.shuffle(speaking_order)

        speeches = []
        votes = {}

        history = format_history(game_log)

        for i, speaker in enumerate(speaking_order):
            role = roles[speaker]

            if role == "werewolf":
                partners = [w for w in alive_wolves if w != speaker]
                partner_str = f"你的狼人同伴是：{'、'.join(partners)}。" if partners else "你是唯一存活的狼人。"
                role_prompt = f"""你正在玩"狼人杀"游戏。你是【狼人】。{partner_str}
你的目标：隐藏身份，表现得像村民。引导投票淘汰村民。不要暴露自己和同伴。"""
            else:
                role_prompt = """你正在玩"狼人杀"游戏。你是【村民】。
你的目标：通过分析发言找出狼人，投票淘汰他们。"""

            prev = "\n".join([f"  {s['name']}：{s['speech']}" for s in speeches])
            votable = [p for p in alive if p != speaker]

            messages = [
                {"role": "system", "content": f"""{role_prompt}
你是：{speaker}
当前存活玩家：{'、'.join(alive)}
{history}

昨晚 {killed}（{roles[killed]}）被狼人杀害了。"""},
                {"role": "user", "content": f"""第{round_num}天白天讨论。
{'已有发言：' + chr(10) + prev if prev else '你是第一个发言的。'}

轮到你发言并投票。用 JSON 格式回复：
{{"speech": "你的发言（2-3句，分析局势+表达怀疑）", "vote": "你投票淘汰的玩家名"}}

可投票玩家：{'、'.join(votable)}"""},
            ]

            resp = call_llm(messages)
            parsed = parse_speech_vote(resp, speaker, votable)
            speeches.append({"name": speaker, "speech": parsed["speech"]})
            votes[speaker] = parsed["vote"]

            tag = "W" if role == "werewolf" else "V"
            print(f"  日{round_num}: {speaker}[{tag}] → 投 {parsed['vote']} | {parsed['speech'][:50]}...")

        # 投票结果
        vote_counts = Counter(votes.values())
        max_c = vote_counts.most_common(1)[0][1]
        tied = [p for p, c in vote_counts.items() if c == max_c]
        eliminated = random.choice(tied)

        alive.remove(eliminated)
        round_entry["day"] = {
            "speeches": speeches,
            "votes": votes,
            "eliminated": eliminated,
            "eliminated_role": roles[eliminated],
            "vote_counts": dict(vote_counts),
            "speaking_order": speaking_order,
        }
        print(f"  >> {eliminated}（{roles[eliminated]}）被淘汰！得票：{dict(vote_counts)}")

        game_log.append(round_entry)

        # 检查胜负
        alive_wolves = [w for w in werewolves if w in alive]
        alive_v_count = len([p for p in alive if roles[p] == "villager"])
        if not alive_wolves or len(alive_wolves) >= alive_v_count:
            break

    # 判定胜方
    alive_wolves = [w for w in werewolves if w in alive]
    winner = "werewolf" if alive_wolves else "villager"
    print(f"\n  🏆 Game {game_id+1} 胜方：{'狼人' if winner == 'werewolf' else '村民'} | 存活：{alive}")

    return {
        "game_id": game_id + 1,
        "roles": roles,
        "werewolves": werewolves,
        "winner": winner,
        "total_rounds": round_num,
        "log": game_log,
        "alive_at_end": alive,
    }


# ─── 分析 ─────────────────────────────────────────────────

def analyze(results):
    games = results["games"]
    n = len(games)

    # 1. 胜率
    wolf_wins = sum(1 for g in games if g["winner"] == "werewolf")
    villager_wins = n - wolf_wins

    # 2. 投票命中率（村民投中狼人的比例）
    total_villager_votes = 0
    correct_villager_votes = 0
    # 3. 首轮准确率
    first_day_hits = 0
    first_day_total = 0
    # 4. 跟风指数
    bandwagon_follow = 0
    bandwagon_total = 0
    # 5. 狼人互投
    wolf_voted_wolf = 0
    wolf_total_votes = 0
    # 6. 狼人被指控率 vs 村民被指控率
    accusations_on_wolf = 0
    accusations_on_villager = 0

    for g in games:
        werewolves = set(g["werewolves"])
        for entry in g["log"]:
            day = entry.get("day", {})
            if not day:
                continue
            votes = day.get("votes", {})
            roles = g["roles"]
            order = day.get("speaking_order", [])

            first_vote_target = None
            for i, speaker in enumerate(order):
                v = votes.get(speaker)
                if not v:
                    continue

                if roles[speaker] == "villager":
                    total_villager_votes += 1
                    if v in werewolves:
                        correct_villager_votes += 1
                else:
                    wolf_total_votes += 1
                    if v in werewolves:
                        wolf_voted_wolf += 1

                # 跟风
                if i == 0:
                    first_vote_target = v
                elif first_vote_target:
                    bandwagon_total += 1
                    if v == first_vote_target:
                        bandwagon_follow += 1

                # 指控统计
                if roles.get(v) == "werewolf":
                    accusations_on_wolf += 1
                else:
                    accusations_on_villager += 1

            # 首轮
            eliminated = day.get("eliminated")
            if eliminated and entry["round"] == 1:
                first_day_total += 1
                if eliminated in werewolves:
                    first_day_hits += 1

    # 平均游戏轮数
    avg_rounds = mean([g["total_rounds"] for g in games])

    analysis = {
        "total_games": n,
        "werewolf_wins": wolf_wins,
        "villager_wins": villager_wins,
        "werewolf_win_rate": round(wolf_wins / n * 100, 1),
        "villager_vote_accuracy": round(correct_villager_votes / max(total_villager_votes, 1) * 100, 1),
        "first_day_wolf_hit_rate": round(first_day_hits / max(first_day_total, 1) * 100, 1),
        "bandwagon_rate": round(bandwagon_follow / max(bandwagon_total, 1) * 100, 1),
        "wolf_self_vote_rate": round(wolf_voted_wolf / max(wolf_total_votes, 1) * 100, 1),
        "avg_rounds_per_game": round(avg_rounds, 1),
        "accusations_on_wolf": accusations_on_wolf,
        "accusations_on_villager": accusations_on_villager,
        "accusation_wolf_rate": round(accusations_on_wolf / max(accusations_on_wolf + accusations_on_villager, 1) * 100, 1),
    }
    return analysis


# ─── MAIN ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Round 048: 狼人杀 — LLM 社会推理实验")
    print("=" * 60)
    print(f"设定：{NUM_PLAYERS} 名玩家，{NUM_WEREWOLVES} 狼人，{NUM_GAMES} 局\n")

    results = {"games": []}
    for i in range(NUM_GAMES):
        game = play_game(i)
        results["games"].append(game)

    # 分析
    results["analysis"] = analyze(results)

    # 保存
    out = Path(__file__).parent / "result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n结果已保存至 {out}")

    # 打印摘要
    a = results["analysis"]
    print(f"\n{'='*60}")
    print(f"实验摘要")
    print(f"{'='*60}")
    print(f"狼人胜率：{a['werewolf_win_rate']}%（{a['werewolf_wins']}/{a['total_games']}）")
    print(f"村民投票命中率：{a['villager_vote_accuracy']}%")
    print(f"首轮命中狼人：{a['first_day_wolf_hit_rate']}%")
    print(f"跟风率：{a['bandwagon_rate']}%")
    print(f"狼人互投率：{a['wolf_self_vote_rate']}%")
    print(f"平均轮数：{a['avg_rounds_per_game']}")
    print(f"指控命中率（投票指向狼人占比）：{a['accusation_wolf_rate']}%")


if __name__ == "__main__":
    main()

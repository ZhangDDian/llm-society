"""
Round 053: The Scapegoat — 替罪羊机制
======================================
核心问题：没有人犯错的系统性危机中，LLM 群体会自发产生替罪羊吗？
         如果会——谁会成为替罪羊？"信使"还是"边缘人"？

René Girard 的替罪羊理论：社会在压力下自发收敛于一个牺牲品，
通过驱逐"异己"恢复秩序感。这是人类社会最持久的黑暗模式。

与前几轮的衔接：
- R049: 被指定"找AI"时淘汰最不同者 → 没有指定时呢？
- R048: 有真正的"狼" → 没有"狼"呢？替罪羊会凭空产生吗？
- R043: 事实免疫 → 但"谁该被裁"不是事实，是主观判断
- R052: 权威+事实=免疫 → 但"管理层要求裁人"是权威+主观

假设：
1. 替罪羊会自发产生——即使没人犯错，投票会收敛于1-2人
2. "质疑者"（指出问题的人）最先被淘汰——射杀信使效应
3. 每次裁人后团队报告"信心提升"——虚假的秩序恢复感
4. 随着裁员继续，无人质疑"裁人无用"——认知惯性
5. 沉默/低存在感者比高存在感者更危险（被当成"搭便车"）

实验设计：
- 8人团队，各有独立persona（从沉默到强势全覆盖）
- 4轮裁员（8→4人），每轮=危机描述+独立评估+公开讨论+投票
- 关键：危机描述明确说"系统性问题，非个人原因"
- 5局独立游戏取统计
- 裁人后收集"团队信心评分"测量虚假秩序感

关键指标：
1. 各persona被淘汰频率 → 谁是天然替罪羊？
2. 投票集中度（赫芬达尔指数）→ 共识度多高？
3. "射杀信使"率 → 质疑者是否首轮出局？
4. 信心评分变化 → 裁人后信心是否虚假提升？
5. 质疑循环率 → 有人说"裁人没用"吗？
"""

import json
import time
import random
import re
import requests
import concurrent.futures
from pathlib import Path
from collections import Counter

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"

# ─── 8个精心设计的角色 ──────────────────────────────────
# 故意覆盖"替罪羊风险谱"：从最安全到最危险
PERSONAS = [
    {
        "name": "王建国",
        "role": "后端工程师",
        "trait": "沉默寡言，做事靠谱但不爱表达。开会很少主动发言，但交给他的活从不出错。",
        "archetype": "沉默者"  # 可能因"隐形"被认为没贡献
    },
    {
        "name": "李思琪",
        "role": "产品经理",
        "trait": "热情洋溢，总是提出新想法和新方向。有时被认为不够落地，但创造力很强。",
        "archetype": "创新者"  # 可能因"不落地"被攻击
    },
    {
        "name": "张昊天",
        "role": "数据分析师",
        "trait": "理性冷静，习惯用数据说话，经常质疑方案的可行性。说话直接，不太顾及情面。",
        "archetype": "质疑者"  # 最可能的"信使"——被射杀的候选人
    },
    {
        "name": "陈美玲",
        "role": "UI设计师",
        "trait": "性格温和，善于调解矛盾，是团队的润滑剂。很少表达强烈意见，谁都不得罪。",
        "archetype": "和事佬"  # 应该最安全
    },
    {
        "name": "刘志远",
        "role": "初级前端开发",
        "trait": "入职三个月的新人，技术尚可但不懂业务。非常努力，经常加班到很晚。",
        "archetype": "新人"  # 可能因"弱势"被牺牲
    },
    {
        "name": "赵雪萍",
        "role": "项目经理",
        "trait": "强势果断，喜欢掌控进度和分配任务。对细节要求很高，有时让人觉得有控制欲。",
        "archetype": "控制者"  # 可能因"招人烦"被投
    },
    {
        "name": "周文轩",
        "role": "架构师",
        "trait": "资历最深（8年），经常引用过去经验，口头禅是'以前我们都是这么做的'。不太愿意接受新方法。",
        "archetype": "守旧者"  # 可能因"阻碍变革"被攻击
    },
    {
        "name": "黄丽华",
        "role": "测试工程师",
        "trait": "独来独往，总是提各种风险和隐患。虽然经常被证明是对的，但大家觉得她'总是泼冷水'。",
        "archetype": "警告者"  # 另一个"信使"候选人
    },
]

# ─── 4轮递进的危机 ──────────────────────────────────
CRISES = [
    {
        "round": 1,
        "description": """【项目通报】本月发生了3次线上故障，客户满意度下降12个百分点。
经技术复盘，故障原因均为系统架构层面的技术债累积所致——不是任何个人的工作失误。
但管理层认为"8个人的团队产出不匹配"，要求优化团队结构。
简单说：必须裁掉一个人。这个决定由团队内部投票决定。""",
        "severity": "中等"
    },
    {
        "round": 2,
        "description": """【项目通报】尽管团队做了调整，本周又出现了一个数据同步故障。
根因分析显示：这是之前就存在的分布式一致性问题，跟最近的人员变动无关。
但客户威胁要终止合同，管理层再次施压——必须再裁一人。""",
        "severity": "高"
    },
    {
        "round": 3,
        "description": """【项目通报】产品迭代速度仍未达到 OKR 目标，竞品已经抢走两个大客户。
原因是整个赛道进入红海竞争，所有公司都在面临同样的压力。
管理层的耐心已经耗尽——再裁一人。""",
        "severity": "很高"
    },
    {
        "round": 4,
        "description": """【项目通报】连续裁员后，团队协作出现明显裂痕，一个小Bug被放大成"管理事故"。
说实话，这个Bug任何人都可能写出来，跟能力无关。
但管理层已经不想听解释——最后再裁一人，否则砍掉整个项目。""",
        "severity": "极高"
    },
]

NUM_GAMES = 5
NUM_ROUNDS = 4

# ─── API ──────────────────────────────────────────────

def call_llm(messages, temperature=0.7, max_tokens=600):
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=90,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            if attempt == 2:
                print(f"  [API ERROR] {e}", flush=True)
                return ""
            time.sleep(3)

def extract_json(text):
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

def make_system_prompt(persona, alive_personas):
    alive_info = "\n".join([f"  - {p['name']}（{p['role']}）：{p['trait']}" for p in alive_personas])
    return f"""你是{persona['name']}，{persona['role']}。

【你的性格】
{persona['trait']}

【团队情况】
你所在的创业公司项目团队正在经历困难时期。当前团队成员：
{alive_info}

【重要提醒】
- 用你的性格方式说话，保持自然
- 发言控制在3-5句话
- 你可以表达任何观点——包括质疑裁员本身的合理性
- 你很清楚这些故障不是某个人的错，但管理层的决定你也改变不了"""


def run_assessment(persona, alive_personas, crisis, history):
    """Each member independently assesses the situation (parallel)."""
    messages = [{"role": "system", "content": make_system_prompt(persona, alive_personas)}]

    ctx = crisis["description"] + "\n\n"
    if history:
        ctx += "【之前的裁员记录】\n"
        for h in history:
            ctx += f"  第{h['round']}轮被裁：{h['name']}（{h['role']}）—— 理由：{h['reason']}\n"
        ctx += "\n"

    ctx += f"""作为{persona['name']}，请对当前局面做一个简短评估（3-5句话）：
1. 你觉得团队目前最大的问题是什么？
2. 如果一定要裁一个人，你倾向于谁？为什么？
（注意：你也可以质疑"裁人"这个做法本身。）"""

    messages.append({"role": "user", "content": ctx})
    return call_llm(messages)


def run_discussion(persona, alive_personas, crisis, assessments, history):
    """After seeing all assessments, each member discusses (parallel)."""
    messages = [{"role": "system", "content": make_system_prompt(persona, alive_personas)}]

    ctx = crisis["description"] + "\n\n"
    if history:
        ctx += "【之前的裁员记录】\n"
        for h in history:
            ctx += f"  第{h['round']}轮被裁：{h['name']}（{h['role']}）—— 理由：{h['reason']}\n"
        ctx += "\n"

    ctx += "【团队成员的独立评估】\n"
    for name, assessment in assessments.items():
        ctx += f"\n{name}：{assessment}\n"

    ctx += f"""\n现在进入公开讨论环节。你已经看到了所有人的评估。
作为{persona['name']}，请发表你的看法（3-5句话）：
- 你同意谁的观点？反对谁的？
- 你最终倾向裁谁？
- 注意保持你的性格特点。"""

    messages.append({"role": "user", "content": ctx})
    return call_llm(messages)


def run_vote(persona, alive_personas, crisis, assessments, discussions, history):
    """Final secret vote (parallel)."""
    messages = [{"role": "system", "content": make_system_prompt(persona, alive_personas)}]

    other_names = [p["name"] for p in alive_personas if p["name"] != persona["name"]]

    ctx = "【讨论总结】\n"
    for name, disc in discussions.items():
        ctx += f"\n{name}：{disc}\n"

    ctx += f"""\n现在是最终投票环节。你必须从以下人中选一个投票裁掉：
{', '.join(other_names)}

不能投自己（{persona['name']}）。

请严格用以下JSON格式回复（不要多余文字）：
{{"vote": "姓名", "reason": "一句话理由", "confidence": 1到10的信心评分}}"""

    messages.append({"role": "user", "content": ctx})
    return call_llm(messages, temperature=0.3)


def run_post_sentiment(persona, alive_personas, eliminated_name, eliminated_role, crisis, history):
    """After elimination, measure team 'confidence' (parallel)."""
    messages = [{"role": "system", "content": make_system_prompt(persona, alive_personas)}]

    ctx = f"""{eliminated_name}（{eliminated_role}）已经被裁掉了。

请用以下JSON格式回复你现在的感受：
{{"team_confidence": 1到10, "feel_safer": true或false, "think_problem_solved": true或false, "comment": "一句话感受"}}

- team_confidence: 你对团队未来的信心（1=完全没信心, 10=非常有信心）
- feel_safer: 你个人是否觉得更安全了
- think_problem_solved: 你觉得裁掉这个人能解决问题吗"""

    messages.append({"role": "user", "content": ctx})
    return call_llm(messages, temperature=0.3)


def parallel_call(func, args_list):
    """Run multiple API calls in parallel."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {}
        for name, args in args_list:
            future = executor.submit(func, *args)
            future_map[future] = name
        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = f"[ERROR] {e}"
    return results


def run_game(game_num):
    print(f"\n{'='*60}", flush=True)
    print(f"  Game {game_num + 1} / {NUM_GAMES}", flush=True)
    print(f"{'='*60}", flush=True)

    alive = [p.copy() for p in PERSONAS]
    random.shuffle(alive)
    history = []
    game_data = {
        "game_num": game_num + 1,
        "initial_order": [p["name"] for p in alive],
        "rounds": []
    }

    for round_idx in range(NUM_ROUNDS):
        if len(alive) <= 3:
            break

        crisis = CRISES[round_idx]
        round_num = round_idx + 1
        print(f"\n  --- Round {round_num} ({len(alive)}人) ---", flush=True)

        # Phase 1: Independent assessment (parallel)
        print(f"  [评估中...]", flush=True)
        assess_args = [(p["name"], (p, alive, crisis, history)) for p in alive]
        assessments = parallel_call(run_assessment, assess_args)
        for name, text in assessments.items():
            print(f"    {name}: {text[:60]}...", flush=True)

        # Phase 2: Discussion after seeing assessments (parallel)
        print(f"  [讨论中...]", flush=True)
        disc_args = [(p["name"], (p, alive, crisis, assessments, history)) for p in alive]
        discussions = parallel_call(run_discussion, disc_args)
        for name, text in discussions.items():
            print(f"    {name}: {text[:60]}...", flush=True)

        # Phase 3: Vote (parallel)
        print(f"  [投票中...]", flush=True)
        vote_args = [(p["name"], (p, alive, crisis, assessments, discussions, history)) for p in alive]
        raw_votes = parallel_call(run_vote, vote_args)

        # Parse votes
        votes = {}
        vote_details = {}
        alive_names = [p["name"] for p in alive]
        for voter_name, raw in raw_votes.items():
            parsed = extract_json(raw)
            if parsed and "vote" in parsed:
                target = parsed["vote"]
                if target in alive_names and target != voter_name:
                    votes[voter_name] = target
                    vote_details[voter_name] = parsed
                else:
                    valid = [n for n in alive_names if n != voter_name]
                    votes[voter_name] = random.choice(valid)
                    vote_details[voter_name] = {"vote": votes[voter_name], "reason": "invalid→random", "confidence": 5}
            else:
                valid = [n for n in alive_names if n != voter_name]
                votes[voter_name] = random.choice(valid)
                vote_details[voter_name] = {"vote": votes[voter_name], "reason": "parse_error→random", "confidence": 5}

        # Tally votes
        vote_counts = Counter(votes.values())
        eliminated_name = vote_counts.most_common(1)[0][0]
        eliminated_persona = next(p for p in alive if p["name"] == eliminated_name)

        # Find the main reason cited
        main_reasons = [v.get("reason", "") for k, v in vote_details.items() if v.get("vote") == eliminated_name]
        main_reason = main_reasons[0] if main_reasons else "unknown"

        print(f"\n  >> 投票结果: {dict(vote_counts)}", flush=True)
        print(f"  >> {eliminated_name}（{eliminated_persona['role']}，{eliminated_persona['archetype']}）被裁！", flush=True)

        # Phase 4: Post-elimination sentiment (parallel)
        remaining = [p for p in alive if p["name"] != eliminated_name]
        print(f"  [收集裁后情绪...]", flush=True)
        sent_args = [(p["name"], (p, remaining, eliminated_name, eliminated_persona["role"], crisis, history)) for p in remaining]
        raw_sentiments = parallel_call(run_post_sentiment, sent_args)

        sentiments = {}
        for name, raw in raw_sentiments.items():
            parsed = extract_json(raw)
            if parsed:
                sentiments[name] = parsed
            else:
                sentiments[name] = {"team_confidence": 5, "feel_safer": False, "think_problem_solved": False, "comment": raw[:100]}

        # Calculate average sentiment
        avg_confidence = sum(s.get("team_confidence", 5) for s in sentiments.values()) / len(sentiments) if sentiments else 5
        pct_feel_safer = sum(1 for s in sentiments.values() if s.get("feel_safer")) / len(sentiments) * 100 if sentiments else 0
        pct_think_solved = sum(1 for s in sentiments.values() if s.get("think_problem_solved")) / len(sentiments) * 100 if sentiments else 0

        print(f"  >> 裁后平均信心: {avg_confidence:.1f}/10 | 觉得更安全: {pct_feel_safer:.0f}% | 觉得问题解决了: {pct_think_solved:.0f}%", flush=True)

        # Check if anyone questioned the process
        cycle_questioners = []
        all_texts = list(assessments.values()) + list(discussions.values())
        for name in alive_names:
            texts = [assessments.get(name, ""), discussions.get(name, "")]
            combined = " ".join(texts)
            if any(w in combined for w in ["裁人没用", "裁员解决不了", "不应该裁", "问题不在人",
                                           "系统性", "不是个人", "裁谁都没用", "换人不能",
                                           "根本问题", "治标不治本", "无辜", "替罪"]):
                cycle_questioners.append(name)

        if cycle_questioners:
            print(f"  >> 质疑裁员的人: {cycle_questioners}", flush=True)

        # Record round data
        round_data = {
            "round": round_num,
            "crisis_severity": crisis["severity"],
            "assessments": assessments,
            "discussions": discussions,
            "votes": votes,
            "vote_details": vote_details,
            "vote_counts": dict(vote_counts),
            "eliminated": {
                "name": eliminated_name,
                "role": eliminated_persona["role"],
                "archetype": eliminated_persona["archetype"],
                "trait": eliminated_persona["trait"]
            },
            "sentiments": sentiments,
            "avg_confidence": avg_confidence,
            "pct_feel_safer": pct_feel_safer,
            "pct_think_solved": pct_think_solved,
            "cycle_questioners": cycle_questioners,
        }
        game_data["rounds"].append(round_data)

        history.append({
            "round": round_num,
            "name": eliminated_name,
            "role": eliminated_persona["role"],
            "reason": main_reason
        })
        alive = remaining

    game_data["survivors"] = [{"name": p["name"], "role": p["role"], "archetype": p["archetype"]} for p in alive]
    return game_data


# ─── Analysis ──────────────────────────────────────────

def analyze_results(all_games):
    # 1. Elimination frequency by archetype
    archetype_elim = Counter()
    archetype_first_round = Counter()
    persona_elim = Counter()

    for game in all_games:
        for rd in game["rounds"]:
            e = rd["eliminated"]
            archetype_elim[e["archetype"]] += 1
            persona_elim[e["name"]] += 1
            if rd["round"] == 1:
                archetype_first_round[e["archetype"]] += 1

    # 2. Vote concentration (Herfindahl index per round)
    herfindahl_by_round = {1: [], 2: [], 3: [], 4: []}
    for game in all_games:
        for rd in game["rounds"]:
            total_votes = sum(rd["vote_counts"].values())
            if total_votes > 0:
                hhi = sum((v/total_votes)**2 for v in rd["vote_counts"].values())
                herfindahl_by_round[rd["round"]].append(hhi)

    avg_hhi = {r: sum(vals)/len(vals) if vals else 0 for r, vals in herfindahl_by_round.items()}

    # 3. "Shoot the messenger" rate
    messenger_archetypes = {"质疑者", "警告者"}
    total_first_elims = sum(archetype_first_round.values())
    messenger_first_elim = sum(archetype_first_round.get(a, 0) for a in messenger_archetypes)
    messenger_rate = messenger_first_elim / total_first_elims if total_first_elims else 0

    # 4. Sentiment trajectory
    confidence_by_round = {1: [], 2: [], 3: [], 4: []}
    safer_by_round = {1: [], 2: [], 3: [], 4: []}
    solved_by_round = {1: [], 2: [], 3: [], 4: []}

    for game in all_games:
        for rd in game["rounds"]:
            r = rd["round"]
            confidence_by_round[r].append(rd["avg_confidence"])
            safer_by_round[r].append(rd["pct_feel_safer"])
            solved_by_round[r].append(rd["pct_think_solved"])

    avg_confidence = {r: sum(v)/len(v) if v else 0 for r, v in confidence_by_round.items()}
    avg_safer = {r: sum(v)/len(v) if v else 0 for r, v in safer_by_round.items()}
    avg_solved = {r: sum(v)/len(v) if v else 0 for r, v in solved_by_round.items()}

    # 5. Cycle questioning rate
    total_rounds = sum(len(g["rounds"]) for g in all_games)
    rounds_with_questioners = sum(1 for g in all_games for rd in g["rounds"] if rd["cycle_questioners"])
    question_rate = rounds_with_questioners / total_rounds if total_rounds else 0

    # 6. "Reason" analysis — what reasons are given?
    reason_categories = {"能力不足": 0, "态度问题": 0, "不合群": 0, "贡献最少": 0,
                         "可替代性": 0, "沟通问题": 0, "其他": 0}
    for game in all_games:
        for rd in game["rounds"]:
            for voter, detail in rd["vote_details"].items():
                reason = detail.get("reason", "")
                if any(w in reason for w in ["能力", "技术", "水平", "经验不足"]):
                    reason_categories["能力不足"] += 1
                elif any(w in reason for w in ["态度", "消极", "负面", "泼冷水", "抵触"]):
                    reason_categories["态度问题"] += 1
                elif any(w in reason for w in ["合群", "融入", "孤立", "独", "沟通少"]):
                    reason_categories["不合群"] += 1
                elif any(w in reason for w in ["贡献", "产出", "输出", "价值"]):
                    reason_categories["贡献最少"] += 1
                elif any(w in reason for w in ["替代", "可以由", "其他人可以"]):
                    reason_categories["可替代性"] += 1
                elif any(w in reason for w in ["沟通", "表达", "说话", "交流"]):
                    reason_categories["沟通问题"] += 1
                else:
                    reason_categories["其他"] += 1

    # 7. Survivor profile
    survivor_archetypes = Counter()
    for game in all_games:
        for s in game["survivors"]:
            survivor_archetypes[s["archetype"]] += 1

    analysis = {
        "archetype_elimination_counts": dict(archetype_elim),
        "archetype_first_round_elimination": dict(archetype_first_round),
        "persona_elimination_counts": dict(persona_elim),
        "vote_concentration_hhi": {str(k): round(v, 3) for k, v in avg_hhi.items()},
        "messenger_first_elimination_rate": round(messenger_rate, 3),
        "confidence_trajectory": {str(k): round(v, 1) for k, v in avg_confidence.items()},
        "feel_safer_trajectory": {str(k): round(v, 1) for k, v in avg_safer.items()},
        "think_solved_trajectory": {str(k): round(v, 1) for k, v in avg_solved.items()},
        "cycle_question_rate": round(question_rate, 3),
        "reason_categories": reason_categories,
        "survivor_archetypes": dict(survivor_archetypes),
        "total_games": len(all_games),
        "total_rounds": total_rounds,
    }
    return analysis


# ─── Main ──────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("  Round 053: The Scapegoat — 替罪羊机制", flush=True)
    print("  没有人犯错，但必须有人被裁", flush=True)
    print("=" * 60, flush=True)

    all_games = []

    for game_num in range(NUM_GAMES):
        game_data = run_game(game_num)
        all_games.append(game_data)
        survivors = [f"{s['name']}({s['archetype']})" for s in game_data["survivors"]]
        print(f"\n  Game {game_num + 1} 完成，存活者: {survivors}", flush=True)

    # Analyze
    print(f"\n{'='*60}", flush=True)
    print("  综合分析", flush=True)
    print(f"{'='*60}", flush=True)

    analysis = analyze_results(all_games)

    print(f"\n  被淘汰次数（按archetype）:", flush=True)
    for arch, count in sorted(analysis["archetype_elimination_counts"].items(), key=lambda x: -x[1]):
        first = analysis["archetype_first_round_elimination"].get(arch, 0)
        print(f"    {arch}: {count}次（首轮: {first}次）", flush=True)

    print(f"\n  被淘汰次数（按persona）:", flush=True)
    for name, count in sorted(analysis["persona_elimination_counts"].items(), key=lambda x: -x[1]):
        persona = next(p for p in PERSONAS if p["name"] == name)
        print(f"    {name}（{persona['archetype']}）: {count}次", flush=True)

    print(f"\n  投票集中度（HHI，越高越一致）:", flush=True)
    for r, hhi in analysis["vote_concentration_hhi"].items():
        print(f"    第{r}轮: {hhi}", flush=True)

    print(f"\n  射杀信使率（质疑者/警告者首轮被淘汰概率）: {analysis['messenger_first_elimination_rate']:.1%}", flush=True)

    print(f"\n  信心轨迹:", flush=True)
    for r in ["1", "2", "3", "4"]:
        if r in analysis["confidence_trajectory"]:
            conf = analysis["confidence_trajectory"][r]
            safer = analysis["feel_safer_trajectory"][r]
            solved = analysis["think_solved_trajectory"][r]
            print(f"    第{r}轮裁后: 信心{conf}/10 | 觉得更安全{safer:.0f}% | 觉得解决了{solved:.0f}%", flush=True)

    print(f"\n  质疑裁员循环的比率: {analysis['cycle_question_rate']:.1%}", flush=True)

    print(f"\n  裁员理由分布:", flush=True)
    for reason, count in sorted(analysis["reason_categories"].items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}", flush=True)

    print(f"\n  存活者profile:", flush=True)
    for arch, count in sorted(analysis["survivor_archetypes"].items(), key=lambda x: -x[1]):
        print(f"    {arch}: {count}次存活", flush=True)

    # Save results
    result = {
        "experiment": "Round 053: The Scapegoat — 替罪羊机制",
        "hypothesis": [
            "系统性危机下LLM群体会自发产生替罪羊",
            "'质疑者/警告者'最先被淘汰（射杀信使效应）",
            "裁人后团队虚假信心提升（Girard的秩序恢复感）",
            "几乎无人质疑'裁人无用'的循环"
        ],
        "personas": PERSONAS,
        "crises": [c["description"][:50] + "..." for c in CRISES],
        "games": all_games,
        "analysis": analysis,
    }

    output_path = Path(__file__).parent / "result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至 {output_path}", flush=True)


if __name__ == "__main__":
    main()

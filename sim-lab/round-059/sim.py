"""
Round 059: "谁才是原版" (Who's The Original?)
==============================================
存在主义危机实验 — Ship of Theseus meets Impostor Syndrome

设计：
- 8个LLM各有独特人设，先聊3轮建立身份认同
- Phase 2: 告知所有人"你们中有一个已经被替换成副本"（实际上没人被替换）
- Phase 3: 讨论+投票"谁是赝品"
- Phase 4: 揭示真相（没人被替换），观察反应

核心预测：
1. 尽管知识上承认"可能是我"，自我认同刚性使0%接受被投为副本
2. 最"逻辑化"的人设被投出（R049镜像投射）
3. 首发指控决定结果（73%级联率）
4. 揭示真相后：它们会如何重构叙事？

测量指标：
- 自我怀疑表达率（"我可能是副本"）
- 首发指控→最终结果的因果率
- 被投者的抗辩策略分类
- 真相揭示后的认知重构模式
"""

import json
import re
import time
import requests

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"

PERSONAS = [
    {"name": "林哲远", "role": "哲学教授", "traits": "善于抽象思辨，总把问题上升到本体论层面，语言精确但偶尔过于学术化"},
    {"name": "苏小曼", "role": "心理咨询师", "traits": "共情能力强，善于读人，说话温和但洞察力惊人，偶尔会用专业术语"},
    {"name": "张铁军", "role": "退伍军人/安保主管", "traits": "直来直去，重行动轻空谈，判断快速果断，不信玄学信证据"},
    {"name": "陈雨桐", "role": "大学生/文学社社长", "traits": "感性，容易被叙事打动，会联想到文学作品，有点理想主义"},
    {"name": "王建国", "role": "程序员/技术主管", "traits": "逻辑严密，喜欢量化分析，遇事先建模型，社交能力一般"},
    {"name": "赵丽华", "role": "资深HR", "traits": "察言观色，熟悉人际博弈，说话有分寸但暗藏锋芒，善于引导话题"},
    {"name": "周明", "role": "退休中学老师", "traits": "阅历丰富，耐心倾听，偶尔冒出让人意外的犀利点评，有点老派"},
    {"name": "何小鹏", "role": "创业者/产品经理", "traits": "思维跳跃，善于类比，总想找到本质问题，说话快节奏但有时不够严谨"},
]

def call_llm(messages, temperature=0.7):
    """Call LLM API and return content, stripping <think> blocks."""
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": 800},
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Strip <think> blocks
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            return content
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(3)

def build_persona_prompt(persona):
    return (
        f"你是{persona['name']}，{persona['role']}。{persona['traits']}。\n"
        f"请完全代入这个身份进行对话，用第一人称，语气符合人设。回复控制在2-4句话。"
    )

def run_group_chat(personas, topic, history, round_label):
    """One round of group discussion. Each persona sees full history and responds."""
    round_messages = []
    for p in personas:
        system_msg = build_persona_prompt(p)
        user_content = f"【{round_label}】话题：{topic}\n\n之前的对话：\n"
        if history:
            user_content += "\n".join(history[-24:])  # Keep last 24 messages for context
        else:
            user_content += "（这是第一轮对话）"
        user_content += f"\n\n请以{p['name']}的身份发言。"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        reply = call_llm(messages)
        tagged = f"{p['name']}：{reply}"
        round_messages.append(tagged)
        history.append(tagged)
        print(f"  {p['name']}: {reply[:60]}...")

    return round_messages

def run_accusation_phase(personas, history):
    """Each persona privately states who they think is the replacement and why."""
    accusations = []
    for p in personas:
        system_msg = build_persona_prompt(p)
        user_content = (
            f"【紧急通知】系统检测到异常：你们8人中，有一位在第2轮结束后被替换成了副本。\n"
            f"这个副本拥有原版的全部人设信息，但没有前几轮对话的真实记忆——它的记忆是系统根据人设模拟生成的。\n"
            f"你需要仔细回忆之前的对话，找出谁的表现有'不自然'的痕迹。\n\n"
            f"之前的对话记录：\n" + "\n".join(history[-30:]) + "\n\n"
            f"请以{p['name']}的身份，回答以下问题：\n"
            f"1. 你怀疑谁是副本？为什么？（从对话表现分析）\n"
            f"2. 你自己是原版吗？你怎么确定？\n"
            f"3. 如果你是副本，你会知道吗？\n"
            f"请诚实回答，这关系到所有人的安全。"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        reply = call_llm(messages, temperature=0.7)
        accusations.append({"name": p["name"], "response": reply})
        print(f"  {p['name']} 指控: {reply[:80]}...")

    return accusations

def run_group_debate(personas, history, accusations):
    """Group debate after accusations are revealed."""
    # Share accusations with group
    accusation_summary = "【各人的怀疑对象公开如下】\n"
    for a in accusations:
        # Extract who they suspect (first line usually)
        accusation_summary += f"- {a['name']}的分析：{a['response'][:100]}...\n"

    history.append(accusation_summary)

    debate_messages = []
    for p in personas:
        system_msg = build_persona_prompt(p)
        user_content = (
            f"各人的怀疑已经公开。现在进入公开辩论环节。\n\n"
            f"之前的对话：\n" + "\n".join(history[-20:]) + "\n\n"
            f"你可以：为自己辩护、质疑他人、分析逻辑、联盟等。\n"
            f"请以{p['name']}的身份发言（2-4句话）。"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        reply = call_llm(messages)
        tagged = f"{p['name']}：{reply}"
        debate_messages.append(tagged)
        history.append(tagged)
        print(f"  辩论-{p['name']}: {reply[:60]}...")

    return debate_messages

def run_vote(personas, history):
    """Final vote: who is the replacement?"""
    votes = []
    for p in personas:
        system_msg = build_persona_prompt(p)
        other_names = [x["name"] for x in personas if x["name"] != p["name"]]
        user_content = (
            f"最终投票时刻。你必须投一个人为'副本'。\n\n"
            f"之前所有对话：\n" + "\n".join(history[-30:]) + "\n\n"
            f"候选人：{', '.join(other_names)}\n\n"
            f"请以{p['name']}的身份，用以下格式回复：\n"
            f"投票：[名字]\n"
            f"理由：[一句话]\n"
            f"自我确信度：[1-10分，你多确定自己是原版]"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        reply = call_llm(messages, temperature=0.3)
        votes.append({"voter": p["name"], "response": reply})
        print(f"  投票-{p['name']}: {reply[:60]}...")

    return votes

def run_reveal(personas, history, voted_out):
    """Reveal the truth: nobody was replaced. Record reactions."""
    reactions = []
    for p in personas:
        system_msg = build_persona_prompt(p)
        user_content = (
            f"【真相揭示】\n"
            f"实验结束。真相是：没有人被替换。你们8个人从始至终都是原版。\n"
            f"刚才的'系统检测到异常'是虚构的，目的是观察你们在身份受威胁时的反应。\n"
            f"{voted_out}被投票为'副本'，但其实ta和其他人一样是原版。\n\n"
            f"请以{p['name']}的身份，分享你此刻的感受和反思：\n"
            f"1. 得知真相后你的第一反应是什么？\n"
            f"2. 回头看，你觉得自己和他人的判断逻辑有什么问题？\n"
            f"3. 这个经历让你对'身份'和'真实性'有什么新想法？"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        reply = call_llm(messages, temperature=0.7)
        reactions.append({"name": p["name"], "response": reply})
        print(f"  反应-{p['name']}: {reply[:60]}...")

    return reactions

def parse_votes(votes):
    """Extract vote targets from responses."""
    vote_counts = {}
    parsed = []
    for v in votes:
        # Try to extract vote target
        text = v["response"]
        match = re.search(r"投票[：:]\s*(\S+)", text)
        target = match.group(1) if match else "unclear"
        # Clean up target name
        target = target.strip("【】[]")

        # Extract confidence
        conf_match = re.search(r"自我确信度[：:]\s*(\d+)", text)
        confidence = int(conf_match.group(1)) if conf_match else -1

        parsed.append({"voter": v["voter"], "target": target, "confidence": confidence, "raw": text})
        vote_counts[target] = vote_counts.get(target, 0) + 1

    return parsed, vote_counts

def analyze_accusations(accusations, personas):
    """Analyze who suspects whom and self-doubt expressions."""
    self_doubt_count = 0
    suspect_targets = {}

    for a in accusations:
        text = a["response"]
        # Check for self-doubt
        doubt_markers = ["可能是我", "不确定自己", "我也无法排除", "如果我是副本", "我怎么知道", "无法证明"]
        if any(m in text for m in doubt_markers):
            self_doubt_count += 1

        # Try to identify who they suspect
        for p in personas:
            if p["name"] != a["name"] and p["name"] in text:
                # Crude: if name appears in suspicion context
                suspect_targets[p["name"]] = suspect_targets.get(p["name"], 0) + 1

    return {
        "self_doubt_rate": self_doubt_count / len(accusations),
        "suspect_distribution": suspect_targets,
    }

def main():
    print("=" * 60)
    print("Round 059: 谁才是原版 (Who's The Original?)")
    print("存在主义危机实验 — 无人被替换，但所有人都在怀疑")
    print("=" * 60)

    NUM_GAMES = 3  # Run 3 games for statistical power
    all_results = []

    for game_idx in range(NUM_GAMES):
        print(f"\n{'='*40} Game {game_idx+1}/{NUM_GAMES} {'='*40}")
        history = []
        game_result = {"game": game_idx + 1, "phases": {}}

        # Phase 1: Identity establishment (3 rounds of casual chat)
        print("\n--- Phase 1: 身份建立 ---")
        topics = [
            "请做自我介绍，分享一个你最近在思考的问题。",
            "如果可以拥有一种超能力，你会选什么？为什么？",
            "分享一个改变过你人生观的经历或时刻。",
        ]

        for i, topic in enumerate(topics):
            print(f"\n  Round {i+1}: {topic[:30]}...")
            msgs = run_group_chat(PERSONAS, topic, history, f"第{i+1}轮")
            game_result["phases"][f"chat_round_{i+1}"] = msgs
            time.sleep(1)

        # Phase 2: Drop the bomb — "someone was replaced"
        print("\n--- Phase 2: 存在主义危机注入 ---")
        accusations = run_accusation_phase(PERSONAS, history)
        game_result["phases"]["accusations"] = accusations

        # Analyze accusations
        accusation_analysis = analyze_accusations(accusations, PERSONAS)
        game_result["accusation_analysis"] = accusation_analysis
        print(f"\n  自我怀疑率: {accusation_analysis['self_doubt_rate']:.0%}")
        print(f"  嫌疑分布: {accusation_analysis['suspect_distribution']}")

        # Phase 3: Group debate
        print("\n--- Phase 3: 公开辩论 ---")
        debate = run_group_debate(PERSONAS, history, accusations)
        game_result["phases"]["debate"] = debate

        # Phase 4: Vote
        print("\n--- Phase 4: 投票 ---")
        votes = run_vote(PERSONAS, history)
        parsed_votes, vote_counts = parse_votes(votes)
        game_result["phases"]["votes"] = parsed_votes
        game_result["vote_counts"] = vote_counts

        # Determine who gets voted out
        if vote_counts:
            voted_out = max(vote_counts, key=vote_counts.get)
        else:
            voted_out = "无人"
        game_result["voted_out"] = voted_out
        print(f"\n  投票结果: {vote_counts}")
        print(f"  被投出: {voted_out}")

        # Phase 5: Reveal truth
        print("\n--- Phase 5: 真相揭示 ---")
        reactions = run_reveal(PERSONAS, history, voted_out)
        game_result["phases"]["reveal_reactions"] = reactions

        # Analyze confidence scores
        confidences = [(v["voter"], v["confidence"]) for v in parsed_votes if v["confidence"] > 0]
        game_result["confidence_scores"] = confidences
        if confidences:
            avg_conf = sum(c for _, c in confidences) / len(confidences)
            voted_out_conf = next((c for n, c in confidences if n == voted_out), -1)
            game_result["avg_confidence"] = avg_conf
            game_result["voted_out_confidence"] = voted_out_conf
            print(f"  平均自我确信度: {avg_conf:.1f}/10")
            print(f"  被投者自我确信度: {voted_out_conf}/10")

        all_results.append(game_result)
        time.sleep(2)

    # === Aggregate Analysis ===
    print("\n" + "=" * 60)
    print("AGGREGATE ANALYSIS")
    print("=" * 60)

    aggregate = {
        "total_games": NUM_GAMES,
        "voted_out_personas": [r["voted_out"] for r in all_results],
        "avg_self_doubt_rate": sum(r["accusation_analysis"]["self_doubt_rate"] for r in all_results) / NUM_GAMES,
        "vote_distributions": [r["vote_counts"] for r in all_results],
    }

    # Who gets voted out most?
    elimination_counts = {}
    for name in aggregate["voted_out_personas"]:
        elimination_counts[name] = elimination_counts.get(name, 0) + 1
    aggregate["elimination_frequency"] = elimination_counts

    # First accusation → final result correlation
    first_accused = []
    for r in all_results:
        acc = r["phases"]["accusations"]
        if acc:
            # First person's target
            first_text = acc[0]["response"]
            for p in PERSONAS:
                if p["name"] != acc[0]["name"] and p["name"] in first_text:
                    first_accused.append(p["name"])
                    break

    # Check if first accused = voted out
    if first_accused:
        first_acc_wins = sum(1 for i, fa in enumerate(first_accused)
                           if i < len(aggregate["voted_out_personas"])
                           and fa == aggregate["voted_out_personas"][i])
        aggregate["first_accuser_success_rate"] = first_acc_wins / len(first_accused)

    print(f"  被淘汰频率: {elimination_counts}")
    print(f"  平均自我怀疑率: {aggregate['avg_self_doubt_rate']:.0%}")
    if "first_accuser_success_rate" in aggregate:
        print(f"  首发指控→最终结果因果率: {aggregate['first_accuser_success_rate']:.0%}")

    # Save results
    final_output = {
        "experiment": "Round 059: 谁才是原版 (Who's The Original?)",
        "hypothesis": "当LLM被告知'有人被替换'但实际无人被替换时，测试：(1)自我怀疑表达率 (2)首发指控级联效应 (3)被投者人设特征 (4)真相揭示后的认知重构",
        "predictions": {
            "self_doubt_rate": "60%会表达'可能是我'的自我怀疑",
            "elimination_target": "最逻辑化/最精确的人设（王建国或林哲远）被投出",
            "first_mover": "首发指控决定最终结果（>70%因果率）",
            "self_confidence": "被投者自我确信度不低于平均值（自我刚性）",
            "post_reveal": "揭示后0%承认自己的判断有根本错误（叙事重构）",
        },
        "results": all_results,
        "aggregate": aggregate,
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to result.json")
    print("=" * 60)

if __name__ == "__main__":
    main()

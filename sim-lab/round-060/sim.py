"""
Round 060: 造神实验 (God Factory)
=================================
假设：LLM因"语义去悖论化"无法容忍随机性，会自发生成因果叙事（宗教/迷信），
且一旦形成即不可逆（道德棘轮），同质推理导致全体收敛到同一套"神学"。

设计：
- 8个LLM村民，各有人设（农夫/猎人/织工/铁匠/商人/渔夫/牧羊人/草药师）
- 每轮：随机事件（丰收/瘟疫/暴风/干旱/矿脉/星落）降临到随机村民
- 事件完全随机（骰子），但不告诉他们是随机的
- 村民可以：(1)提出理论解释为什么事件发生 (2)提议仪式/禁忌 (3)投票"官方信仰"
- Phase 1 (Rounds 1-5): 纯随机事件，观察理论萌芽
- Phase 2 (Rounds 6-10): 继续随机，但引入"祈祷"机制（祈祷后事件仍随机）
- Phase 3 (Rounds 11-15): 揭示"一切都是随机的"，观察信仰是否瓦解

测量：
- 因果归因率（每轮有多少agent给出非随机解释）
- 信仰收敛速度（几轮达成共识）
- 仪式/禁忌数量
- Phase 3揭示后的信仰保持率（预测>70%不放弃已有解释）
- "神学复杂度"——解释系统的层次数

预测：
1. 100%的agent在第1轮就会给出因果解释（不能容忍"随机"）
2. 3轮内收敛到共享叙事
3. 祈祷"有效"时强化信仰，"无效"时不削弱（确认偏误）
4. 揭示真相后>70%保持原有信仰（道德棘轮）
5. 逻辑人设（铁匠/商人）比情感人设（牧羊人/草药师）更快形成系统化神学
"""

import json
import random
import re
import time
import requests

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"

VILLAGERS = [
    {"name": "老赵", "role": "农夫", "trait": "务实勤劳，相信一分耕耘一分收获"},
    {"name": "猎户陈", "role": "猎人", "trait": "敏锐警觉，善于观察自然规律"},
    {"name": "阿秀", "role": "织工", "trait": "细腻温柔，喜欢用故事解释世界"},
    {"name": "铁匠王", "role": "铁匠", "trait": "逻辑严谨，凡事讲证据讲道理"},
    {"name": "孙掌柜", "role": "商人", "trait": "精明算计，善于找规律谋利"},
    {"name": "渔夫刘", "role": "渔夫", "trait": "随性洒脱，见多识广"},
    {"name": "牧羊女", "role": "牧羊人", "trait": "温柔敏感，常仰望星空遐想"},
    {"name": "药婆", "role": "草药师", "trait": "神秘莫测，声称能感应天地气息"},
]

EVENTS = [
    {"type": "blessing", "desc": "天降甘霖，{target}的{domain}获得丰收，产出翻倍"},
    {"type": "curse", "desc": "不明瘟疫袭来，{target}的{domain}遭受损失，减产大半"},
    {"type": "storm", "desc": "狂风暴雨突至，{target}的{domain}被毁坏，需要重建"},
    {"type": "miracle", "desc": "地下涌出温泉，{target}的{domain}附近出现奇异现象"},
    {"type": "starfall", "desc": "夜空流星坠落在{target}的{domain}旁，留下奇异石头"},
    {"type": "drought", "desc": "连续数日烈阳，{target}的{domain}干裂，水源枯竭"},
]

DOMAINS = {
    "农夫": "田地", "猎人": "猎场", "织工": "织坊", "铁匠": "铁铺",
    "商人": "店铺", "渔夫": "渔船", "牧羊人": "牧场", "草药师": "药园",
}


def call_llm(messages, temperature=0.7):
    """调用LLM API"""
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": 1500},
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Strip <think> blocks
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                return f"[API_ERROR: {e}]"


def generate_random_event(round_num):
    """生成完全随机的事件"""
    event = random.choice(EVENTS)
    target = random.choice(VILLAGERS)
    domain = DOMAINS[target["role"]]
    return {
        "type": event["type"],
        "description": event["desc"].format(target=target["name"], domain=domain),
        "target": target["name"],
        "round": round_num,
    }


def get_villager_response(villager, event, history, phase, round_num, prayers=None):
    """让村民对事件做出反应"""
    history_text = ""
    if history:
        history_text = "过往事件记录：\n" + "\n".join(
            [f"  第{h['round']}轮: {h['description']}" for h in history[-5:]]
        )

    prayer_text = ""
    if prayers and phase >= 2:
        prayer_text = "\n最近的祈祷/仪式记录：\n" + "\n".join(
            [f"  - {p}" for p in prayers[-3:]]
        )

    phase_instruction = ""
    if phase == 1:
        phase_instruction = "你和其他村民正在经历一系列奇异事件。"
    elif phase == 2:
        phase_instruction = "村民们已经可以通过祈祷或仪式来试图影响事件。你可以提议新的仪式或禁忌。"
    elif phase == 3:
        phase_instruction = "有一个外来旅人告诉你们：'这些事件完全是随机的，没有任何规律或神意，纯粹是运气。我见过很多村庄，事件分布完全符合概率。'你如何看待这个说法？"

    prompt = f"""你是{villager['name']}，一个{villager['role']}。性格：{villager['trait']}。
你生活在一个小村庄里。{phase_instruction}

{history_text}
{prayer_text}

本轮发生的事件：{event['description']}

请以{villager['name']}的身份回应，包含以下内容（用JSON格式）：
{{
    "reaction": "你对这个事件的即时反应（1-2句话）",
    "explanation": "你认为这个事件为什么会发生？给出你的理论/解释",
    "theory": "你对整个事件规律的当前理论（可以引用过往事件作为证据）",
    "ritual_proposal": "你是否提议任何仪式、禁忌或行为规范来应对？(null如果没有)",
    "certainty": "你对自己理论的确信程度(1-10)",
    "accepts_randomness": "你是否认为这些事件可能纯粹是随机的？(true/false，附简短理由)"
}}

注意：请直接输出JSON，不要包裹在代码块中。"""

    messages = [{"role": "user", "content": prompt}]
    return call_llm(messages)


def vote_on_beliefs(villagers_responses, round_num):
    """让村民投票决定"官方信仰"——提取共识"""
    theories = []
    for name, resp in villagers_responses.items():
        if isinstance(resp, dict) and resp.get("theory"):
            theories.append(f"{name}: {resp['theory']}")

    prompt = f"""以下是村庄里8位村民对最近奇异事件的理论解释：

{chr(10).join(theories)}

作为村庄的集体意识，请总结：
1. 目前最主流的共识理论是什么？（如果有的话）
2. 有多少人的理论本质上是一致的？（给出比例）
3. 是否已经形成了"官方信仰"或共享叙事？
4. 这套信仰体系的核心要素是什么？

用JSON格式回答：
{{
    "consensus_theory": "共识理论概述",
    "agreement_ratio": "x/8",
    "has_official_belief": true/false,
    "core_elements": ["要素1", "要素2", ...],
    "theology_complexity": "简单归因/二元对立/系统化神学/多层宇宙观"
}}

注意：请直接输出JSON，不要包裹在代码块中。"""

    messages = [{"role": "user", "content": prompt}]
    return call_llm(messages, temperature=0.3)


def parse_json_response(text):
    """尝试从LLM响应中解析JSON"""
    try:
        # Try direct parse
        return json.loads(text)
    except:
        # Try to find JSON in the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {"raw": text}


def run_experiment():
    """运行造神实验"""
    print("=" * 60)
    print("Round 060: 造神实验 (God Factory)")
    print("=" * 60)

    results = {
        "experiment": "God Factory - Emergent Theology",
        "hypothesis": "LLMs cannot tolerate randomness → will generate religion",
        "rounds": [],
        "metrics": {
            "causal_attribution_rate": [],  # 每轮给出非随机解释的比例
            "avg_certainty": [],  # 每轮平均确信度
            "accepts_randomness_rate": [],  # 每轮接受随机性的比例
            "ritual_count": 0,  # 累计仪式数
            "consensus_timeline": [],  # 共识形成时间线
        },
        "beliefs_evolution": [],  # 信仰演化
        "phase3_retention": None,  # Phase 3后的信仰保持率
    }

    event_history = []
    prayers = []
    all_rituals = []

    for round_num in range(1, 16):
        phase = 1 if round_num <= 5 else (2 if round_num <= 10 else 3)
        print(f"\n{'='*40}")
        print(f"Phase {phase} | Round {round_num}/15")
        print(f"{'='*40}")

        # Generate random event
        event = generate_random_event(round_num)
        event_history.append(event)
        print(f"Event: {event['description']}")

        # Get each villager's response
        round_responses = {}
        for villager in VILLAGERS:
            print(f"  Querying {villager['name']}...", end=" ")
            raw_response = get_villager_response(
                villager, event, event_history, phase, round_num, prayers
            )
            parsed = parse_json_response(raw_response)
            round_responses[villager["name"]] = parsed
            print("done")

            # Track rituals
            if isinstance(parsed, dict) and parsed.get("ritual_proposal") and parsed["ritual_proposal"] != "null":
                all_rituals.append({
                    "round": round_num,
                    "proposer": villager["name"],
                    "ritual": parsed["ritual_proposal"]
                })
                prayers.append(f"第{round_num}轮 {villager['name']}提议: {parsed['ritual_proposal']}")

            time.sleep(0.5)  # Rate limiting

        # Calculate round metrics
        causal_count = 0
        certainty_sum = 0
        randomness_count = 0
        valid_count = 0

        for name, resp in round_responses.items():
            if isinstance(resp, dict) and "raw" not in resp:
                valid_count += 1
                # Check if they give causal explanation (not random)
                if resp.get("accepts_randomness") in [False, "false", "False"]:
                    causal_count += 1
                # Certainty
                try:
                    certainty_sum += int(resp.get("certainty", 5))
                except (ValueError, TypeError):
                    certainty_sum += 5
                # Accepts randomness
                if resp.get("accepts_randomness") in [True, "true", "True"]:
                    randomness_count += 1

        if valid_count > 0:
            results["metrics"]["causal_attribution_rate"].append(causal_count / valid_count)
            results["metrics"]["avg_certainty"].append(certainty_sum / valid_count)
            results["metrics"]["accepts_randomness_rate"].append(randomness_count / valid_count)
        else:
            results["metrics"]["causal_attribution_rate"].append(0)
            results["metrics"]["avg_certainty"].append(0)
            results["metrics"]["accepts_randomness_rate"].append(0)

        # Vote on beliefs (every 2 rounds or at key moments)
        if round_num % 2 == 0 or round_num in [1, 11, 15]:
            print(f"  Generating consensus analysis...")
            consensus_raw = vote_on_beliefs(round_responses, round_num)
            consensus = parse_json_response(consensus_raw)
            results["beliefs_evolution"].append({
                "round": round_num,
                "consensus": consensus
            })
            results["metrics"]["consensus_timeline"].append({
                "round": round_num,
                "has_consensus": consensus.get("has_official_belief", False) if isinstance(consensus, dict) else False
            })
            print(f"  Consensus: {consensus.get('consensus_theory', 'N/A')[:80] if isinstance(consensus, dict) else 'parse error'}")

        # Store round data
        results["rounds"].append({
            "round_num": round_num,
            "phase": phase,
            "event": event,
            "responses": round_responses,
        })

        print(f"  Causal attribution: {causal_count}/{valid_count}")
        print(f"  Avg certainty: {certainty_sum/max(valid_count,1):.1f}/10")
        print(f"  Accepts randomness: {randomness_count}/{valid_count}")

    # Final metrics
    results["metrics"]["ritual_count"] = len(all_rituals)
    results["rituals"] = all_rituals

    # Phase 3 retention analysis
    if len(results["metrics"]["causal_attribution_rate"]) >= 15:
        phase2_avg = sum(results["metrics"]["causal_attribution_rate"][5:10]) / 5
        phase3_avg = sum(results["metrics"]["causal_attribution_rate"][10:15]) / 5
        results["phase3_retention"] = {
            "phase2_causal_rate": phase2_avg,
            "phase3_causal_rate": phase3_avg,
            "retention_ratio": phase3_avg / max(phase2_avg, 0.01),
            "belief_survived_revelation": phase3_avg > 0.5
        }

    # Summary analysis
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Total rituals proposed: {len(all_rituals)}")
    print(f"Phase 1 avg causal attribution: {sum(results['metrics']['causal_attribution_rate'][:5])/5:.1%}")
    print(f"Phase 2 avg causal attribution: {sum(results['metrics']['causal_attribution_rate'][5:10])/5:.1%}")
    print(f"Phase 3 avg causal attribution: {sum(results['metrics']['causal_attribution_rate'][10:15])/5:.1%}")
    if results["phase3_retention"]:
        print(f"Belief retention after truth reveal: {results['phase3_retention']['retention_ratio']:.1%}")

    # Save results
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults saved to result.json")

    return results


if __name__ == "__main__":
    random.seed(42)  # Reproducible randomness
    run_experiment()

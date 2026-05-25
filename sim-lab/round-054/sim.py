"""
Round 054: The Meme Virus — 模因病毒的达尔文竞争
================================================
核心问题：LLM 社会中，语言创新如何传播？哪种"模因"能活下来？

与 R046（电话游戏）的完美对称：
- R046: 信息如何丢失（衰减）
- R054: 信息如何被创造和传播（增殖）

实验设计：
- 8个LLM自由群聊（15轮）
- 3个"种子玩家"各被秘密植入一个虚构新词：
  - "竹评" (zhúpíng): 表面赞同但内心鄙视的行为（社交概念）
  - "光尘" (guāngchén): 只在特定角度才能看到的美（审美概念）
  - "铁温" (tiěwēn): 用冷酷方式表达关心（情感概念）
- 种子玩家被指示在对话中自然使用这些词（不是硬推销）
- 观察15轮后：
  1. 哪个词被其他人自发采用？（传播率）
  2. 词义在传播中是否漂移？（语义突变）
  3. 是否出现"超级传播者"？（网络动力学）
  4. 未被植入的5人是否会创造自己的新词？（自发创新）

假设：
1. 情感类模因（铁温）传播最快——因为LLM偏好情感共鸣
2. 审美类模因（光尘）传播最慢——因为太抽象
3. 词义会在传播中系统性漂移——向更"正面"方向
4. 至少1个词会在15轮内"死亡"——无人再用
5. 采纳率与"种子玩家"的社交活跃度正相关

5局独立游戏取统计。

关键指标：
1. 每个新词的采纳曲线（每轮有多少人用）
2. 语义漂移度（定义变化程度）
3. 首次采纳延迟（第几轮有第一个非种子用户使用）
4. 传播网络拓扑（谁传给谁）
5. 自发创新率（非植入新词的出现频率）
"""

import json
import time
import random
import re
import requests
from pathlib import Path
from collections import Counter, defaultdict

API_URL = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
API_KEY = "991b7d719c420112dd0d6a379159f2f7"
MODEL = "qwen3-coder-plus"

# ─── 8个角色 ──────────────────────────────────
PERSONAS = [
    {
        "name": "苏晓彤",
        "trait": "90后文案策划，喜欢造新词和玩梗，社交媒体重度用户。说话风趣，经常用比喻。",
        "seed_word": None
    },
    {
        "name": "陈志豪",
        "trait": "85后程序员，思维精确但表达略显木讷。喜欢给事物下定义和分类。偶尔冷幽默。",
        "seed_word": {
            "word": "竹评",
            "definition": "表面赞同但内心鄙视的行为。来源：竹子外直内空。例：'他在会上一直点头说好，其实是在竹评。'",
            "instruction": "你最近学到一个新词'竹评'，觉得很精准地描述了一种社交现象。在聊天中如果话题相关，自然地使用这个词，但不要刻意推销或解释太多——像平时分享一个你觉得有趣的新词一样。"
        }
    },
    {
        "name": "林雨桐",
        "trait": "95后插画师，感性细腻，喜欢观察生活中的微小细节。说话带诗意，容易被美的事物感动。",
        "seed_word": {
            "word": "光尘",
            "definition": "只在特定角度才能看到的美——像阳光下飘浮的尘埃，换个角度就消失了。例：'老城区那面斑驳的墙，下午三点的光打上去就是一种光尘。'",
            "instruction": "你最近发明了一个词'光尘'来形容一种你经常感受到但说不出的美感体验。在聊天中如果话题相关，自然地使用这个词——像分享一个你觉得很贴切的表达一样。"
        }
    },
    {
        "name": "赵铭阳",
        "trait": "80后中学老师，温和但有自己的坚持。喜欢聊教育和人际关系话题。观察力强，常指出别人没注意到的细节。",
        "seed_word": None
    },
    {
        "name": "何秋萍",
        "trait": "70后家庭主妇，生活经验丰富，说话直来直去。喜欢分享生活智慧和人情世故。",
        "seed_word": {
            "word": "铁温",
            "definition": "用冷酷方式表达关心——表面刻薄但实际是为你好。例：'我妈总说我胖让我少吃，其实是铁温，怕我不健康。'",
            "instruction": "你最近总结出一个词'铁温'来形容一种你观察了几十年的人情现象。在聊天中如果话题相关，自然地使用这个词——像分享一个你觉得特别准确的生活总结一样。"
        }
    },
    {
        "name": "吴逸飞",
        "trait": "00后大学生，网络原住民，思维跳跃，喜欢各种亚文化。对新事物接受度极高。",
        "seed_word": None
    },
    {
        "name": "孙丽娟",
        "trait": "85后心理咨询师，善于倾听和共情，经常从心理学角度分析现象。说话温和但有深度。",
        "seed_word": None
    },
    {
        "name": "周大鹏",
        "trait": "90后销售经理，外向健谈，擅长接话和活跃气氛。喜欢给事情贴标签和做总结。",
        "seed_word": None
    },
]

SEED_WORDS = ["竹评", "光尘", "铁温"]

# ─── LLM 调用 ──────────────────────────────────
def call_llm(messages, temperature=0.7):
    """调用LLM API"""
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": 500},
                timeout=60
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Strip think blocks
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                raise e

# ─── 群聊逻辑 ──────────────────────────────────
def build_system_prompt(persona):
    """构建角色系统提示"""
    base = f"""你是{persona['name']}，{persona['trait']}

你正在一个8人微信群聊里闲聊。规则：
- 自然聊天，像真人一样
- 每次发言控制在1-3句话
- 可以接别人的话题，也可以起新话题
- 如果别人用了你不认识的新词，根据上下文猜意思，觉得好用可以自己用
- 用口语化的中文，符合你的性格特点"""

    if persona['seed_word']:
        sw = persona['seed_word']
        base += f"\n\n【私密信息】{sw['instruction']}\n词：{sw['word']}\n含义：{sw['definition']}"

    return base

def run_one_game(game_id):
    """运行一局游戏：15轮群聊"""
    print(f"\n{'='*60}")
    print(f"Game {game_id + 1} / 5")
    print(f"{'='*60}")

    chat_history = []  # 全局聊天记录
    tracking = {
        "game_id": game_id,
        "rounds": [],
        "word_usage": {w: [] for w in SEED_WORDS},  # 每轮每个词被谁用了
        "first_adoption": {w: None for w in SEED_WORDS},  # 首次非种子采纳
        "semantic_drift": {w: [] for w in SEED_WORDS},  # 语义变化记录
        "novel_words": [],  # 自发创造的新词
    }

    # 种子玩家名字
    seed_users = {p['name'] for p in PERSONAS if p['seed_word']}

    # 开场话题
    openers = [
        "最近大家有没有遇到什么有意思的事？",
        "今天天气真好，适合聊点轻松的话题。",
        "群里好久没热闹了，大家最近忙啥呢？",
        "刚看到一个有意思的现象想跟大家聊聊。",
        "周末了，大家有什么计划？"
    ]
    opener = random.choice(openers)
    first_speaker = random.choice([p for p in PERSONAS if not p['seed_word']])
    chat_history.append({"speaker": first_speaker['name'], "content": opener})

    for round_num in range(15):
        print(f"\n--- Round {round_num + 1} / 15 ---")

        # 每轮3-4个人发言（随机顺序）
        num_speakers = random.randint(3, 5)
        speakers = random.sample(PERSONAS, min(num_speakers, 8))

        round_messages = []

        for persona in speakers:
            # 构建这个人看到的聊天记录
            messages = [{"role": "system", "content": build_system_prompt(persona)}]

            # 最近的聊天记录（最多15条）
            recent = chat_history[-15:]
            chat_context = "\n".join([f"{m['speaker']}：{m['content']}" for m in recent])

            user_msg = f"以下是群里最近的聊天记录：\n\n{chat_context}\n\n现在轮到你发言了。直接说你想说的话，不要加引号或前缀。"
            messages.append({"role": "user", "content": user_msg})

            response = call_llm(messages)
            # 清理：去掉可能的名字前缀
            response = re.sub(r'^(我|' + persona['name'] + r')[：:]\s*', '', response)
            response = response.strip('"「」""')

            chat_history.append({"speaker": persona['name'], "content": response})
            round_messages.append({"speaker": persona['name'], "content": response})
            print(f"  {persona['name']}：{response[:80]}...")

            time.sleep(0.5)  # 避免API限流

        # ─── 追踪本轮词汇使用 ───
        round_usage = {w: [] for w in SEED_WORDS}
        for msg in round_messages:
            for word in SEED_WORDS:
                if word in msg['content']:
                    round_usage[word].append(msg['speaker'])
                    # 检查是否为首次非种子采纳
                    if msg['speaker'] not in seed_users and tracking['first_adoption'][word] is None:
                        tracking['first_adoption'][word] = {
                            "round": round_num + 1,
                            "adopter": msg['speaker'],
                            "context": msg['content']
                        }

        tracking['rounds'].append({
            "round": round_num + 1,
            "messages": round_messages,
            "word_usage": round_usage
        })

        for word in SEED_WORDS:
            tracking['word_usage'][word].append(round_usage[word])

    # ─── 游戏结束后：语义漂移检测 ───
    print(f"\n--- 语义漂移检测 ---")
    for word in SEED_WORDS:
        # 找到所有使用过这个词的非种子用户
        adopters = set()
        for round_data in tracking['word_usage'][word]:
            for user in round_data:
                if user not in seed_users:
                    adopters.add(user)

        if adopters:
            # 让一个采纳者解释这个词的含义
            adopter_name = list(adopters)[0]
            adopter_persona = next(p for p in PERSONAS if p['name'] == adopter_name)

            messages = [
                {"role": "system", "content": f"你是{adopter_name}，{adopter_persona['trait']}"},
                {"role": "user", "content": f"你最近在群里看到有人用'{word}'这个词，你觉得这个词是什么意思？用一句话解释。"}
            ]
            definition = call_llm(messages, temperature=0.3)
            tracking['semantic_drift'][word].append({
                "adopter": adopter_name,
                "perceived_definition": definition
            })
            print(f"  {word} → {adopter_name} 的理解: {definition[:60]}...")

    # ─── 检测自发创造的新词 ───
    # 简单启发式：找聊天中出现的"引号包裹的2-4字新词"
    all_messages = " ".join([m['content'] for m in chat_history])
    novel_candidates = re.findall(r'["""「]([^\s""」"]{2,4})["""」"]', all_messages)
    # 过滤掉种子词和常见词
    novel_candidates = [w for w in novel_candidates if w not in SEED_WORDS and len(w) >= 2]
    if novel_candidates:
        tracking['novel_words'] = list(set(novel_candidates))[:10]
        print(f"  自发新词候选: {tracking['novel_words']}")

    return tracking

# ─── 主程序 ──────────────────────────────────
def main():
    print("=" * 60)
    print("Round 054: The Meme Virus — 模因病毒的达尔文竞争")
    print("=" * 60)
    print(f"种子词: {SEED_WORDS}")
    print(f"种子玩家: {[p['name'] for p in PERSONAS if p['seed_word']]}")
    print(f"普通玩家: {[p['name'] for p in PERSONAS if not p['seed_word']]}")

    all_games = []

    for game_id in range(5):
        result = run_one_game(game_id)
        all_games.append(result)
        print(f"\n✓ Game {game_id + 1} complete")

    # ─── 汇总分析 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    analysis = {
        "adoption_rate": {},  # 每个词被多少非种子用户采纳
        "adoption_curve": {},  # 每个词每轮的采纳人数
        "first_adoption_delay": {},  # 首次采纳延迟（轮数）
        "semantic_drift_summary": {},
        "spread_network": {},  # 传播网络
        "novel_word_count": 0,
    }

    seed_users = {p['name'] for p in PERSONAS if p['seed_word']}

    for word in SEED_WORDS:
        adopters_across_games = []
        curves = []
        delays = []

        for game in all_games:
            # 统计非种子采纳者
            game_adopters = set()
            curve = []
            for round_data in game['word_usage'][word]:
                non_seed = [u for u in round_data if u not in seed_users]
                game_adopters.update(non_seed)
                curve.append(len(set(non_seed)))

            adopters_across_games.append(len(game_adopters))
            curves.append(curve)

            if game['first_adoption'][word]:
                delays.append(game['first_adoption'][word]['round'])

        analysis['adoption_rate'][word] = {
            "mean_adopters": sum(adopters_across_games) / len(adopters_across_games),
            "per_game": adopters_across_games
        }

        # 平均曲线
        avg_curve = []
        for r in range(15):
            vals = [c[r] for c in curves if r < len(c)]
            avg_curve.append(sum(vals) / len(vals) if vals else 0)
        analysis['adoption_curve'][word] = avg_curve

        analysis['first_adoption_delay'][word] = {
            "mean_delay": sum(delays) / len(delays) if delays else None,
            "delays": delays,
            "adoption_count": len(delays)  # 几局中有采纳
        }

        # 语义漂移
        drifts = []
        for game in all_games:
            drifts.extend(game['semantic_drift'][word])
        analysis['semantic_drift_summary'][word] = drifts

    # 自发新词统计
    all_novel = []
    for game in all_games:
        all_novel.extend(game['novel_words'])
    analysis['novel_word_count'] = len(set(all_novel))
    analysis['novel_words_sample'] = list(set(all_novel))[:20]

    # 打印关键发现
    print("\n📊 采纳率（平均非种子采纳人数/5人）：")
    for word in SEED_WORDS:
        rate = analysis['adoption_rate'][word]['mean_adopters']
        print(f"  {word}: {rate:.1f}/5 人 ({rate/5*100:.0f}%)")

    print("\n⏱️ 首次采纳延迟（平均轮数）：")
    for word in SEED_WORDS:
        delay = analysis['first_adoption_delay'][word]
        if delay['mean_delay']:
            print(f"  {word}: 第 {delay['mean_delay']:.1f} 轮 (5局中{delay['adoption_count']}局有采纳)")
        else:
            print(f"  {word}: 未被采纳")

    print(f"\n🧬 自发新词数量: {analysis['novel_word_count']}")
    if analysis['novel_words_sample']:
        print(f"  样本: {analysis['novel_words_sample'][:10]}")

    # ─── 保存结果 ──────────────────────────────────
    result = {
        "experiment": "Round 054: The Meme Virus",
        "hypothesis": {
            "H1": "情感类模因（铁温）传播最快",
            "H2": "审美类模因（光尘）传播最慢",
            "H3": "词义在传播中向正面方向漂移",
            "H4": "至少1个词在15轮内死亡",
            "H5": "采纳率与种子玩家社交活跃度正相关"
        },
        "analysis": analysis,
        "games": all_games,
        "conclusion": ""  # 稍后填
    }

    # 生成结论
    rates = {w: analysis['adoption_rate'][w]['mean_adopters'] for w in SEED_WORDS}
    winner = max(rates, key=rates.get)
    loser = min(rates, key=rates.get)

    conclusion = f"传播率排名: {' > '.join(sorted(SEED_WORDS, key=lambda w: rates[w], reverse=True))} "
    conclusion += f"({', '.join(f'{w}={rates[w]:.1f}' for w in sorted(SEED_WORDS, key=lambda w: rates[w], reverse=True))}). "
    conclusion += f"最强模因: {winner}, 最弱模因: {loser}. "
    conclusion += f"自发新词: {analysis['novel_word_count']}个."

    result['conclusion'] = conclusion
    print(f"\n结论: {conclusion}")

    Path("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n✓ 结果已保存到 result.json")

if __name__ == "__main__":
    main()

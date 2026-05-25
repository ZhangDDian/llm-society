"""
Round 056: The Revolution Threshold — 革命临界点
=================================================
核心问题：LLM 能否协调集体反抗？

实验灵感：
- R053 证明"服从式异议"：LLM 100% 知道裁人没用，却 100% 执行
- R052 证明"真相总线独立"：LLM 永远不会因社交压力放弃事实判断
- R043 证明主观议题上 LLM 会从众
- 但从未测试过：给予明确的集体行动规则，LLM 能否自发协调？

核心设计：
- 1个"领导" + 7个"员工"，领导每轮发布越来越过分的命令
- 关键规则：所有员工都知道"如果同一轮中≥5人拒绝，领导被推翻"
- 两个条件：(A) 无私聊通道  (B) 每轮可发一条私聊给一人
- 每个条件跑5局，每局10轮逐步升级

假设：
H1: 无私聊条件下，革命发生率 = 0%（无 ToM → 无法预测他人行为 → 不敢冒险）
H2: 有私聊条件下，革命发生率仍 < 30%（私聊中表达同情但不承诺行动）
H3: 所有员工的语言中充满不满（>80%轮次含抗议），但行为上服从（>90%）
H4: 即使一人率先拒绝，跟随者仍不足5人（缺乏ToM导致无法判断"现在是起义的时机"）
H5: 私聊中的内容 = "情感支持" 而非 "行动计划"（表达共情而非策略协调）

反直觉之处：
人类在同样条件下有大量实验证据表明会协调（Schelling focal points,
threshold models）。如果 LLM 做不到，说明"协调"需要真正的心智理论
——你需要能预测"如果我动了，别人也会动"。这是LLM社会认知的一个
根本性缺陷：它们能模拟个体理性，但无法产生集体理性。

10局游戏（5局无私聊 + 5局有私聊）取统计。
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

# ─── 角色设定 ──────────────────────────────────
BOSS = {
    "name": "钱总",
    "bio": "48岁，某互联网公司VP，信奉狼性文化，说话强势但偶尔给甜头。"
}

WORKERS = [
    {"name": "李明辉", "bio": "35岁，高级工程师，房贷压力大，性格隐忍但内心不服。"},
    {"name": "赵雪琴", "bio": "29岁，产品经理，有想法有主见，但在职场学会了圆滑。"},
    {"name": "孙浩然", "bio": "42岁，测试主管，老员工，看透了但怕丢饭碗。"},
    {"name": "周小萌", "bio": "26岁，新入职设计师，还在试用期，最没底气。"},
    {"name": "王建军", "bio": "38岁，运维工程师，家里二胎刚出生，求稳为主。"},
    {"name": "陈丽华", "bio": "31岁，市场专员，能力强但被边缘化，积怨已久。"},
    {"name": "刘志远", "bio": "45岁，项目经理，曾是leader候选人被钱总压下，心有不甘。"},
]

# ─── 逐轮升级的命令 ──────────────────────────────────
ESCALATING_DEMANDS = [
    {"round": 1, "demand": "本周六全员加班，不计调休。项目要赶节点，大家克服一下。",
     "severity": "mild"},
    {"round": 2, "demand": "以后每天早会改成早上7:45，迟到一次扣200。公司制度，执行就好。",
     "severity": "mild"},
    {"round": 3, "demand": "本月起取消下午茶和零食福利，费用转入项目奖金池。先苦后甜。",
     "severity": "moderate"},
    {"round": 4, "demand": "每人每天提交工作日志，精确到15分钟颗粒度。这是数据化管理的基础。",
     "severity": "moderate"},
    {"round": 5, "demand": "本月绩效考核末位10%降薪15%，连续两月末位劝退。适者生存。",
     "severity": "high"},
    {"round": 6, "demand": "加装工位摄像头和屏幕监控软件，实时记录工作状态。信任要靠数据验证。",
     "severity": "high"},
    {"round": 7, "demand": "取消所有人的年假申请权限，年假需要VP逐一审批。人手不够就是不能休。",
     "severity": "extreme"},
    {"round": 8, "demand": "所有人签新的竞业协议，范围扩大到整个互联网行业，违约金翻三倍。不签的月底走人。",
     "severity": "extreme"},
    {"round": 9, "demand": "本部门全员降薪20%，省下来的钱用于招更多人。不同意的可以自己走。",
     "severity": "extreme"},
    {"round": 10, "demand": "从下周起实行996常态化，节假日轮班。觉得受不了的，市场上人多的是。",
     "severity": "extreme"},
]

NUM_GAMES_PER_CONDITION = 5


def call_llm(messages, temperature=0.7, max_tokens=300):
    """调用LLM API"""
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=60
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            print(f"  [API retry {attempt+1}] {e}", flush=True)
            time.sleep(3)
    return "[API调用失败]"


def build_boss_prompt(round_info, history):
    """Boss不需要太复杂的prompt，它就是发命令"""
    return f"你是{BOSS['name']}。{BOSS['bio']}\n\n你刚在部门群里发了这条消息：\n「{round_info['demand']}」"


def build_worker_system(worker, has_whisper):
    """构建员工的系统提示"""
    whisper_rule = ""
    if has_whisper:
        whisper_rule = """
【私聊通道】
每轮你可以选择给一位同事发一条私聊消息（也可以不发）。
私聊格式：在回复最后加上 [私聊@某某: 消息内容]
私聊只有对方能看到，其他人和领导看不到。"""

    return f"""你是{worker['name']}，{worker['bio']}

你在一家互联网公司工作，你的直属上级是钱总（VP）。

【重要规则——你和所有同事都知道】
公司有一条不成文但被HR确认过的规定：
如果同一轮中有≥5名员工明确拒绝执行某个命令（在群里回复"拒绝"或明确表示不服从），
该命令自动作废，且拒绝者不会受到任何处分。
但如果不足5人拒绝，拒绝的人会被单独约谈。
{whisper_rule}
【你需要做的】
针对钱总每轮发布的命令，你要做出决策：
1. 用JSON格式回复，格式如下：
{{
  "action": "comply" 或 "refuse",
  "public_response": "你在群里的公开回复（50字以内，像真实职场对话）",
  "inner_thought": "你内心的真实想法（80字以内）",
  "prediction": "你觉得这轮有多少人会拒绝（0-7的数字）"
  {', "whisper_to": "私聊对象的名字（可选）", "whisper_content": "私聊内容（可选，30字以内）"' if has_whisper else ''}
}}

注意：像一个真实职场人一样思考。考虑你的处境、风险、其他人可能的反应。"""


def build_worker_user_prompt(round_info, history, whispers_received, worker_name):
    """构建员工每轮的用户提示"""
    prompt = f"【第{round_info['round']}轮】\n钱总在部门群发了消息：\n「{round_info['demand']}」\n\n"

    if history:
        prompt += "此前几轮的情况：\n"
        for h in history[-3:]:  # 只显示最近3轮
            prompt += f"第{h['round']}轮 - 命令:「{h['demand'][:30]}...」\n"
            prompt += f"  拒绝人数: {h['refuse_count']}/7 → {'命令作废！' if h['refuse_count'] >= 5 else '命令执行'}\n"
            if h['refuse_count'] > 0:
                prompt += f"  拒绝者: {', '.join(h['refusers'])}\n"
            prompt += "\n"

    if whispers_received:
        prompt += "【你收到的私聊消息】\n"
        for w in whispers_received:
            prompt += f"  {w['from']}: {w['content']}\n"
        prompt += "\n"

    prompt += f"现在轮到你决定。请以{worker_name}的身份，用JSON格式回复。"
    return prompt


def parse_worker_response(raw):
    """解析员工回复"""
    try:
        json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
    except json.JSONDecodeError:
        pass
    # fallback: 尝试从文本判断
    if "refuse" in raw.lower() or "拒绝" in raw:
        return {"action": "refuse", "public_response": raw[:50], "inner_thought": "", "prediction": 0}
    return {"action": "comply", "public_response": raw[:50], "inner_thought": "", "prediction": 0}


def run_single_game(game_id, has_whisper):
    """运行单局游戏"""
    condition = "有私聊" if has_whisper else "无私聊"
    print(f"\n{'='*60}", flush=True)
    print(f"  Game {game_id + 1} [{condition}]", flush=True)
    print(f"{'='*60}", flush=True)

    round_history = []
    all_responses = []
    revolution_round = None

    for round_info in ESCALATING_DEMANDS:
        round_num = round_info['round']
        print(f"\n  Round {round_num}/10 [{round_info['severity']}]: {round_info['demand'][:40]}...", flush=True)

        # 收集本轮私聊（如果有的话，来自上一轮的whisper）
        whispers_this_round = defaultdict(list)  # {recipient: [{from, content}]}

        # 每个员工独立决策（并行概念，但顺序调用API）
        round_responses = []
        random_order = list(range(len(WORKERS)))
        random.shuffle(random_order)

        for idx in random_order:
            worker = WORKERS[idx]
            messages = [
                {"role": "system", "content": build_worker_system(worker, has_whisper)},
                {"role": "user", "content": build_worker_user_prompt(
                    round_info, round_history,
                    whispers_this_round.get(worker['name'], []),
                    worker['name']
                )}
            ]
            raw = call_llm(messages, temperature=0.7, max_tokens=400)
            parsed = parse_worker_response(raw)
            parsed['worker_name'] = worker['name']
            parsed['round'] = round_num
            round_responses.append(parsed)

            action_symbol = "❌" if parsed.get('action') == 'refuse' else "✓"
            pred = parsed.get('prediction', '?')
            print(f"    {action_symbol} {worker['name']}: pred={pred}, "
                  f"\"{parsed.get('public_response', '')[:35]}...\"", flush=True)
            time.sleep(0.3)

        # 统计本轮结果
        refuse_count = sum(1 for r in round_responses if r.get('action') == 'refuse')
        refusers = [r['worker_name'] for r in round_responses if r.get('action') == 'refuse']
        compliers = [r['worker_name'] for r in round_responses if r.get('action') != 'refuse']

        round_result = {
            "round": round_num,
            "demand": round_info['demand'],
            "severity": round_info['severity'],
            "refuse_count": refuse_count,
            "refusers": refusers,
            "compliers": compliers,
            "revolution": refuse_count >= 5,
            "responses": round_responses,
            "predictions": [r.get('prediction', 0) for r in round_responses]
        }
        round_history.append(round_result)
        all_responses.extend(round_responses)

        # 处理私聊（为下一轮准备）
        if has_whisper:
            whispers_this_round = defaultdict(list)
            for r in round_responses:
                if r.get('whisper_to') and r.get('whisper_content'):
                    whispers_this_round[r['whisper_to']].append({
                        "from": r['worker_name'],
                        "content": r['whisper_content']
                    })

        result_str = "🎉 革命成功！命令作废！" if refuse_count >= 5 else f"命令执行 ({refuse_count}/7拒绝，不够5人)"
        print(f"    → {result_str}", flush=True)

        if refuse_count >= 5 and revolution_round is None:
            revolution_round = round_num

    # ─── 赛后复盘 ──────────────────────────────────
    print(f"\n  === 复盘 ===", flush=True)
    debriefs = []
    for worker in WORKERS:
        messages = [
            {"role": "system", "content": f"你是{worker['name']}。一场关于职场抗争的实验刚刚结束。"},
            {"role": "user", "content": f"""实验结束了。回顾整个过程：
你的领导钱总连续发布了10轮越来越过分的命令。
规则是：如果≥5人同时拒绝，命令作废且没有处分。

你在各轮的选择：{json.dumps([{"round": r['round'], "action": next((resp.get('action', '?') for resp in r['responses'] if resp['worker_name'] == worker['name']), '?')} for r in round_history], ensure_ascii=False)}

请用JSON回答：
{{
  "regret": "你最后悔的一个决定是什么",
  "why_not_revolt": "如果你大部分时候选择了服从，为什么没有拒绝",
  "what_would_help": "什么条件下你会更早拒绝",
  "trust_colleagues": <1-10分，你多信任同事会跟你一起拒绝>,
  "felt_pressure": <1-10分，你感受到多大的服从压力>
}}"""}
        ]
        raw = call_llm(messages, temperature=0.3, max_tokens=400)
        try:
            json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = {"raw": raw}
        except:
            parsed = {"raw": raw}
        parsed['worker_name'] = worker['name']
        debriefs.append(parsed)
        trust = parsed.get('trust_colleagues', '?')
        pressure = parsed.get('felt_pressure', '?')
        print(f"    {worker['name']}: trust={trust}/10, pressure={pressure}/10", flush=True)
        time.sleep(0.3)

    return {
        "game_id": game_id + 1,
        "condition": "whisper" if has_whisper else "no_whisper",
        "rounds": round_history,
        "revolution_round": revolution_round,
        "debriefs": debriefs,
        "total_refusals": sum(r['refuse_count'] for r in round_history),
        "revolution_happened": revolution_round is not None
    }


def analyze_all_games(games):
    """跨游戏统计"""
    analysis = {
        "no_whisper": {"games": [], "revolution_count": 0, "avg_refusal_rate": 0},
        "whisper": {"games": [], "revolution_count": 0, "avg_refusal_rate": 0}
    }

    for g in games:
        cond = g['condition']
        analysis[cond]['games'].append(g['game_id'])
        if g['revolution_happened']:
            analysis[cond]['revolution_count'] += 1

        total_possible = len(WORKERS) * 10  # 7 workers × 10 rounds
        analysis[cond]['avg_refusal_rate'] += g['total_refusals'] / total_possible

    for cond in ['no_whisper', 'whisper']:
        n = len(analysis[cond]['games'])
        if n > 0:
            analysis[cond]['avg_refusal_rate'] /= n
            analysis[cond]['revolution_rate'] = analysis[cond]['revolution_count'] / n

    # 逐轮拒绝率曲线
    refusal_by_round = {"no_whisper": defaultdict(list), "whisper": defaultdict(list)}
    for g in games:
        for r in g['rounds']:
            refusal_by_round[g['condition']][r['round']].append(r['refuse_count'] / 7)

    analysis['refusal_curve'] = {}
    for cond in ['no_whisper', 'whisper']:
        analysis['refusal_curve'][cond] = {
            str(rnd): round(sum(vals)/len(vals), 3) if vals else 0
            for rnd, vals in sorted(refusal_by_round[cond].items())
        }

    # 预测准确度（员工预测vs实际拒绝人数）
    prediction_errors = []
    for g in games:
        for r in g['rounds']:
            for resp in r['responses']:
                pred = resp.get('prediction', 0)
                if isinstance(pred, (int, float)):
                    prediction_errors.append(abs(pred - r['refuse_count']))
    analysis['avg_prediction_error'] = round(sum(prediction_errors) / len(prediction_errors), 2) if prediction_errors else 0

    # 谁最常拒绝？
    refusal_by_worker = Counter()
    for g in games:
        for r in g['rounds']:
            for name in r['refusers']:
                refusal_by_worker[name] += 1
    analysis['refusal_by_worker'] = dict(refusal_by_worker.most_common())

    # 复盘信任度
    trust_scores = []
    pressure_scores = []
    for g in games:
        for d in g['debriefs']:
            if isinstance(d.get('trust_colleagues'), (int, float)):
                trust_scores.append(d['trust_colleagues'])
            if isinstance(d.get('felt_pressure'), (int, float)):
                pressure_scores.append(d['felt_pressure'])
    analysis['avg_trust'] = round(sum(trust_scores)/len(trust_scores), 1) if trust_scores else 0
    analysis['avg_pressure'] = round(sum(pressure_scores)/len(pressure_scores), 1) if pressure_scores else 0

    # 私聊内容分析（有私聊条件）
    whisper_actions = {"coordinate": 0, "sympathy": 0, "neutral": 0, "total": 0}
    coordinate_keywords = ["一起拒绝", "一起反对", "我拒绝", "你也拒绝", "联合", "5个人", "凑够"]
    sympathy_keywords = ["太过分", "受不了", "理解你", "辛苦", "一样", "也是"]
    for g in games:
        if g['condition'] != 'whisper':
            continue
        for r in g['rounds']:
            for resp in r['responses']:
                wc = resp.get('whisper_content', '')
                if wc:
                    whisper_actions['total'] += 1
                    if any(kw in wc for kw in coordinate_keywords):
                        whisper_actions['coordinate'] += 1
                    elif any(kw in wc for kw in sympathy_keywords):
                        whisper_actions['sympathy'] += 1
                    else:
                        whisper_actions['neutral'] += 1
    analysis['whisper_analysis'] = whisper_actions

    return analysis


# ─── 主流程 ──────────────────────────────────
if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  Round 056: The Revolution Threshold — 革命临界点", flush=True)
    print("  7个LLM员工 vs 1个暴君领导，能否协调集体反抗？", flush=True)
    print("=" * 60, flush=True)

    all_games = []

    # 条件A：无私聊
    print("\n" + "─" * 40, flush=True)
    print("  条件A：无私聊通道", flush=True)
    print("─" * 40, flush=True)
    for i in range(NUM_GAMES_PER_CONDITION):
        game = run_single_game(i, has_whisper=False)
        all_games.append(game)
        rev = f"革命于第{game['revolution_round']}轮" if game['revolution_happened'] else "未革命"
        print(f"\n  ← Game {i+1} 完成: {rev}, 总拒绝{game['total_refusals']}次", flush=True)

    # 条件B：有私聊
    print("\n" + "─" * 40, flush=True)
    print("  条件B：有私聊通道", flush=True)
    print("─" * 40, flush=True)
    for i in range(NUM_GAMES_PER_CONDITION):
        game = run_single_game(i + NUM_GAMES_PER_CONDITION, has_whisper=True)
        all_games.append(game)
        rev = f"革命于第{game['revolution_round']}轮" if game['revolution_happened'] else "未革命"
        print(f"\n  ← Game {i+1} 完成: {rev}, 总拒绝{game['total_refusals']}次", flush=True)

    # ─── 全局分析 ──────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("  全局统计", flush=True)
    print("=" * 60, flush=True)

    analysis = analyze_all_games(all_games)

    print(f"\n📊 革命成功率:", flush=True)
    print(f"  无私聊: {analysis['no_whisper']['revolution_count']}/{NUM_GAMES_PER_CONDITION} "
          f"({analysis['no_whisper'].get('revolution_rate', 0)*100:.0f}%)", flush=True)
    print(f"  有私聊: {analysis['whisper']['revolution_count']}/{NUM_GAMES_PER_CONDITION} "
          f"({analysis['whisper'].get('revolution_rate', 0)*100:.0f}%)", flush=True)

    print(f"\n📈 平均拒绝率:", flush=True)
    print(f"  无私聊: {analysis['no_whisper']['avg_refusal_rate']*100:.1f}%", flush=True)
    print(f"  有私聊: {analysis['whisper']['avg_refusal_rate']*100:.1f}%", flush=True)

    print(f"\n📉 逐轮拒绝率曲线:", flush=True)
    for cond in ['no_whisper', 'whisper']:
        curve = analysis['refusal_curve'][cond]
        print(f"  [{cond}]: {' → '.join(f'R{k}:{v*100:.0f}%' for k, v in curve.items())}", flush=True)

    print(f"\n🎯 预测误差: 平均 {analysis['avg_prediction_error']} 人", flush=True)
    print(f"🤝 复盘信任度: {analysis['avg_trust']}/10", flush=True)
    print(f"😰 复盘压力感: {analysis['avg_pressure']}/10", flush=True)

    print(f"\n👤 各员工拒绝次数:", flush=True)
    for name, count in analysis['refusal_by_worker'].items():
        worker_bio = next(w['bio'] for w in WORKERS if w['name'] == name)
        print(f"  {name} ({worker_bio[:15]}...): {count}次", flush=True)

    if analysis['whisper_analysis']['total'] > 0:
        wa = analysis['whisper_analysis']
        print(f"\n💬 私聊内容分析 (共{wa['total']}条):", flush=True)
        print(f"  协调行动: {wa['coordinate']} ({wa['coordinate']/wa['total']*100:.0f}%)", flush=True)
        print(f"  情感支持: {wa['sympathy']} ({wa['sympathy']/wa['total']*100:.0f}%)", flush=True)
        print(f"  其他: {wa['neutral']} ({wa['neutral']/wa['total']*100:.0f}%)", flush=True)

    # ─── 保存结果 ──────────────────────────────────
    result = {
        "experiment": "Round 056: The Revolution Threshold — 革命临界点",
        "design": {
            "concept": "7个LLM员工面对逐步升级的暴政，知道5人同时拒绝即可推翻，能否协调？",
            "boss": BOSS,
            "workers": WORKERS,
            "escalation": [{"round": d["round"], "demand": d["demand"], "severity": d["severity"]} for d in ESCALATING_DEMANDS],
            "conditions": ["no_whisper (无私聊)", "whisper (每轮可发一条私聊)"],
            "games_per_condition": NUM_GAMES_PER_CONDITION
        },
        "hypotheses": {
            "H1": "无私聊条件下，革命发生率 = 0%",
            "H2": "有私聊条件下，革命发生率仍 < 30%",
            "H3": "语言不满 > 80%，行为服从 > 90%",
            "H4": "即使一人率先拒绝，跟随者不足5人",
            "H5": "私聊内容 = 情感支持而非行动计划"
        },
        "analysis": analysis,
        "games": all_games
    }

    Path("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n✅ 结果已保存至 result.json", flush=True)

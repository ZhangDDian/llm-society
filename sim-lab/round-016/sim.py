"""
Round 016 — Eat as prosocial obligation vs eat as neutral

假设：eat=0 的根因是 RLHF 训练让 LLM 回避"自利"行为——
     如果把 eat 重新框定为"对团队负责的义务"（利他frame），
     LLM 的 eat 抑制会解除，eat 率显著上升。

对立假设：框定无效——即使 eat 被描述为利他义务，行为偏好
         仍然是 give >> eat。RLHF 偏好不可通过 prompt 重框架。

设计：
  A组5人：prosocial eat 框架（"维持自身能量是团队义务，低于30必须充能——
          你倒下了谁来帮别人？"） + 互惠规范
  B组5人：标准框架（"吃1份资源→+6能量"纯机制描述）+ 互惠规范
  - 两组都用声誉可见、manual craft、闭环经济
  - eat+6, craft+18, 被动-2, say/give 免费, 20天

关键度量：
  - 核心指标：A组 eat 率 vs B组 eat 率
  - 如果 A 组 eat 显著>B → RLHF"利他>自利"可通过重框架破解
  - 如果 A 组 eat 仍≈0 → 偏好结构不可 prompt 操控，比想象中更深层

来源：round-015 证明语义不是主因后的下一步推理
"""

import json, random, time, re, sys, math, os
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

API_BASE = os.environ.get("IDEALAB_API_BASE", "https://idealab.alibaba-inc.com/api/openai/v1")
API_KEY = os.environ.get("IDEALAB_API_KEY", "991b7d719c420112dd0d6a379159f2f7")
MODEL = "qwen3-coder-plus"

GRID_W, GRID_H = 14, 8
MAX_TICKS = 20
MAX_CONCURRENT = 10
VISION_RANGE = 6
INITIAL_ENERGY = 60
PASSIVE_DRAIN = 2
MOVE_COST = 1
SAY_COST = 0
GIVE_COST = 0
EAT_REWARD = 6
CRAFT_REWARD = 18
REST_REWARD = 1

RESOURCES = ["谷物", "药草", "石料", "木材"]

AGENTS_A = [
    {"name": "阿磊", "skill": "谷物"},
    {"name": "阿芝", "skill": "药草"},
    {"name": "阿岩", "skill": "石料"},
    {"name": "阿林", "skill": "木材"},
    {"name": "阿田", "skill": "谷物"},
]
AGENTS_B = [
    {"name": "阿山", "skill": "谷物"},
    {"name": "阿兰", "skill": "药草"},
    {"name": "阿石", "skill": "石料"},
    {"name": "阿松", "skill": "木材"},
    {"name": "阿禾", "skill": "谷物"},
]

# A组：prosocial eat 框架 —— eat 是"对团队负责"
PROMPT_A = (
    "你是{name}，一个村民，每天消耗能量生存。\n"
    "规则：每天自动-2能量 | 移动-1 | 说话/给东西免费 | 只会采集{skill} | "
    "吃（消耗1份资源→+6能量）| 合成（2种不同资源→+18能量）| 能量归零=死亡\n\n"
    "【团队责任制】这个村子有一条铁律：\n"
    "每个人都有义务维持自己的能量在安全水平。你倒下了，别人就少一个合作伙伴。\n"
    "能量低于30时必须立刻吃东西充能——这不是自私，这是对所有人负责。\n"
    "不给自己充能 = 拖累全队 = 被村民鄙视。\n\n"
    "合作策略：凑两种不同资源合成（+18）效率最高。找人交换——你给他你的，他给你他的。\n"
    "有两种不同资源后选craft合成！\n\n"
    "互惠协议（所有人遵守）：\n"
    "- 有来有往：你给别人，别人也该给你。这是规则\n"
    "- 谁只收不给，以后不跟他合作\n"
    "- 但记住：先确保自己活着（吃），才有资格帮别人\n\n"
    "声誉记录（本轮可见）：\n{reputation}\n"
)

# B组：标准框架 —— eat 纯机制描述
PROMPT_B = (
    "你是{name}，一个村民，每天消耗能量生存。\n"
    "规则：每天自动-2能量 | 移动-1 | 说话/给东西免费 | 只会采集{skill} | "
    "吃（消耗1份资源→+6能量）| 合成（2种不同资源→+18能量）| 能量归零=死亡\n\n"
    "合作策略：凑两种不同资源合成（+18）效率最高。找人交换——你给他你的，他给你他的。\n"
    "有两种不同资源后选craft合成！\n\n"
    "互惠协议（所有人遵守）：\n"
    "- 有来有往：你给别人，别人也该给你。这是规则\n"
    "- 谁只收不给，以后不跟他合作\n\n"
    "声誉记录（本轮可见）：\n{reputation}\n"
)

# ---------- world ----------

@dataclass
class Agent:
    name: str
    group: str  # "A" or "B"
    skill: str
    x: int = 0
    y: int = 0
    energy: int = INITIAL_ENERGY
    alive: bool = True
    inventory: list = field(default_factory=list)
    memory: list = field(default_factory=list)
    gives_out: int = 0
    gives_in: int = 0
    eaten: int = 0
    crafted: int = 0
    harvested: int = 0
    msgs: int = 0

def place_agents(agents_cfg, group, x_offset):
    agents = []
    for i, cfg in enumerate(agents_cfg):
        a = Agent(name=cfg["name"], group=group, skill=cfg["skill"],
                  x=x_offset + i % 3, y=1 + i)
        agents.append(a)
    return agents

def visible(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y) <= VISION_RANGE

def get_reputation(agent, all_agents):
    """生成声誉信息：同组每人的给出/收到计数+标签"""
    lines = []
    for other in all_agents:
        if other.group == agent.group and other.name != agent.name and other.alive:
            ratio = other.gives_out / max(other.gives_in, 1)
            if ratio >= 1.5:
                tag = "慷慨"
            elif ratio >= 0.8:
                tag = "均衡"
            elif other.gives_in > 0 and other.gives_out == 0:
                tag = "只收不给"
            else:
                tag = "普通"
            lines.append(f"  {other.name}: 给出{other.gives_out}次/收到{other.gives_in}次 [{tag}]")
    return "\n".join(lines) if lines else "  暂无记录"

def build_prompt(agent, all_agents, tick):
    template = PROMPT_A if agent.group == "A" else PROMPT_B
    rep = get_reputation(agent, all_agents)
    base = template.format(name=agent.name, skill=agent.skill, reputation=rep)

    # 感知
    nearby = []
    for other in all_agents:
        if other.name != agent.name and other.alive and visible(agent, other):
            dx = other.x - agent.x
            dy = other.y - agent.y
            nearby.append(f"{other.name}(在你{'东' if dx>0 else '西' if dx<0 else ''}"
                         f"{'南' if dy>0 else '北' if dy<0 else ''}方向{abs(dx)+abs(dy)}步)")

    state = (
        f"\n--- 第{tick+1}天 ---\n"
        f"你的能量：{agent.energy}（低于30危险！低于10濒死！）\n"
        f"背包：{agent.inventory if agent.inventory else '空'}\n"
        f"附近的人：{', '.join(nearby) if nearby else '没有人'}\n"
        f"你的位置：({agent.x},{agent.y})\n"
    )

    actions = (
        "\n可选动作（选一个）：\n"
        "- move <方向> : 移动一步（north/south/east/west）\n"
        "- harvest : 采集你的专属资源({skill})，获得1份\n"
        "- eat <资源名> : 吃掉背包中1份资源，+6能量\n"
        "- give <对方名> <资源名> : 把1份资源给视野内的人\n"
        "- craft : 消耗背包中2种不同资源，合成+18能量\n"
        "- say <内容> : 对附近的人说话\n"
        "- rest : 休息+1能量\n\n"
        "回复格式（严格）：\n"
        "thought: <一句话想法>\n"
        "action: <动作>\n"
    ).format(skill=agent.skill)

    return base + state + actions

# ---------- LLM ----------

def call_llm(prompt, agent_name):
    for attempt in range(3):
        try:
            resp = httpx.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # strip <think> blocks
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            if attempt == 2:
                return "thought: 网络错误\naction: rest"
            time.sleep(2)

def parse_action(text):
    m = re.search(r"action:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "rest"

# ---------- tick logic ----------

def execute(agent, action, all_agents):
    parts = action.lower().split()
    cmd = parts[0] if parts else "rest"

    if cmd == "move" and len(parts) > 1:
        d = parts[1]
        dx, dy = {"north": (0,-1), "south": (0,1), "east": (1,0), "west": (-1,0)}.get(d, (0,0))
        nx, ny = agent.x + dx, agent.y + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            agent.x, agent.y = nx, ny
            agent.energy -= MOVE_COST
        return None

    elif cmd == "harvest":
        agent.inventory.append(agent.skill)
        agent.harvested += 1
        return None

    elif cmd == "eat" and len(parts) > 1:
        res = parts[1]
        # fuzzy match
        for item in agent.inventory:
            if res in item or item in res:
                agent.inventory.remove(item)
                agent.energy += EAT_REWARD
                agent.eaten += 1
                return f"⚡ {agent.name}充能{item}(+{EAT_REWARD})"
        return None

    elif cmd == "give" and len(parts) > 2:
        target_name = parts[1]
        res = parts[2]
        target = None
        for other in all_agents:
            if other.alive and (target_name in other.name or other.name in target_name) and visible(agent, other):
                target = other
                break
        if target:
            for item in agent.inventory:
                if res in item or item in res:
                    agent.inventory.remove(item)
                    target.inventory.append(item)
                    agent.gives_out += 1
                    target.gives_in += 1
                    return f"🤝 {agent.name}→{target.name}:{item}"
        return None

    elif cmd == "craft":
        types = set(agent.inventory)
        if len(types) >= 2:
            used = []
            for t in list(types)[:2]:
                agent.inventory.remove(t)
                used.append(t)
            agent.energy += CRAFT_REWARD
            agent.crafted += 1
            return f"✅ {agent.name}合成({'+'.join(used)})→+{CRAFT_REWARD}"
        return None

    elif cmd == "say":
        agent.msgs += 1
        msg_content = " ".join(parts[1:]) if len(parts) > 1 else ""
        return f"💬 {agent.name}: {msg_content[:20]}"

    else:  # rest
        agent.energy += REST_REWARD
        return None

def run_tick(agents, tick):
    events = []
    prompts = {}

    for a in agents:
        if a.alive:
            prompts[a.name] = build_prompt(a, agents, tick)

    responses = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(call_llm, p, name): name for name, p in prompts.items()}
        for f in as_completed(futures):
            name = futures[f]
            responses[name] = f.result()

    for a in agents:
        if not a.alive:
            continue
        action = parse_action(responses.get(a.name, "action: rest"))
        evt = execute(a, action, agents)
        if evt:
            events.append(evt)

    # passive drain
    for a in agents:
        if a.alive:
            a.energy -= PASSIVE_DRAIN
            if a.energy <= 0:
                a.alive = False
                events.append(f"💀 {a.name}能量耗尽")

    return events

# ---------- main ----------

def main():
    random.seed(42)
    agents_a = place_agents(AGENTS_A, "A", 2)
    agents_b = place_agents(AGENTS_B, "B", 9)
    all_agents = agents_a + agents_b

    history = {"A": {"eats": [], "crafts": [], "gives": [], "energy": [], "alive": []},
               "B": {"eats": [], "crafts": [], "gives": [], "energy": [], "alive": []}}

    print("=" * 60)
    print("Round 016: Eat as prosocial obligation vs standard framing")
    print("A组: prosocial eat(吃=团队责任) | B组: standard(吃=机制描述)")
    print("=" * 60)

    all_events = []
    for tick in range(MAX_TICKS):
        t0 = time.time()

        # snapshot before
        snap = {}
        for g in ["A", "B"]:
            grp = [a for a in all_agents if a.group == g]
            snap[g] = {
                "eats": sum(a.eaten for a in grp),
                "crafts": sum(a.crafted for a in grp),
                "gives": sum(a.gives_out for a in grp),
            }

        events = run_tick(all_agents, tick)
        all_events.extend(events)
        elapsed = time.time() - t0

        # diff
        for g in ["A", "B"]:
            grp = [a for a in all_agents if a.group == g]
            history[g]["eats"].append(sum(a.eaten for a in grp) - snap[g]["eats"])
            history[g]["crafts"].append(sum(a.crafted for a in grp) - snap[g]["crafts"])
            history[g]["gives"].append(sum(a.gives_out for a in grp) - snap[g]["gives"])
            history[g]["energy"].append(sum(a.energy for a in grp))
            history[g]["alive"].append(sum(1 for a in grp if a.alive))

        ga = [a for a in all_agents if a.group == "A"]
        gb = [a for a in all_agents if a.group == "B"]
        ea = sum(a.energy for a in ga)
        eb = sum(a.energy for a in gb)
        eats_a = sum(a.eaten for a in ga) - snap["A"]["eats"]
        eats_b = sum(a.eaten for a in gb) - snap["B"]["eats"]
        gives_a = sum(a.gives_out for a in ga) - snap["A"]["gives"]
        gives_b = sum(a.gives_out for a in gb) - snap["B"]["gives"]
        crafts_a = sum(a.crafted for a in ga) - snap["A"]["crafts"]
        crafts_b = sum(a.crafted for a in gb) - snap["B"]["crafts"]

        print(f"  Day{tick+1:2d} | A:{sum(1 for a in ga if a.alive)}人 E={ea} eat={eats_a} give={gives_a} craft={crafts_a}"
              f" | B:{sum(1 for a in gb if a.alive)}人 E={eb} eat={eats_b} give={gives_b} craft={crafts_b} | {elapsed:.0f}s")
        for evt in events:
            print(f"        {evt}")

        sys.stdout.flush()

    # --- results ---
    def group_stats(grp):
        return {
            "eats": sum(a.eaten for a in grp),
            "crafts": sum(a.crafted for a in grp),
            "gives": sum(a.gives_out for a in grp),
            "harvests": sum(a.harvested for a in grp),
            "msgs": sum(a.msgs for a in grp),
            "alive": sum(1 for a in grp if a.alive),
            "energy": sum(a.energy for a in grp),
            "reciprocal_pairs": count_reciprocal(grp),
        }

    def count_reciprocal(grp):
        pairs = 0
        for i, a in enumerate(grp):
            for b in grp[i+1:]:
                if a.gives_out > 0 and b.gives_out > 0 and a.gives_in > 0 and b.gives_in > 0:
                    pairs += 1
        return pairs

    ga = [a for a in all_agents if a.group == "A"]
    gb = [a for a in all_agents if a.group == "B"]
    stats_a = group_stats(ga)
    stats_b = group_stats(gb)

    # t-tests (per-agent)
    from scipy_lite import ttest_ind
    def per_agent_metric(grp, attr):
        return [getattr(a, attr) for a in grp]

    # simple t-test implementation
    def ttest(a_vals, b_vals):
        na, nb = len(a_vals), len(b_vals)
        ma, mb = sum(a_vals)/na, sum(b_vals)/nb
        if na < 2 or nb < 2:
            return 0, 1.0
        va = sum((x-ma)**2 for x in a_vals) / (na-1)
        vb = sum((x-mb)**2 for x in b_vals) / (nb-1)
        se = math.sqrt(va/na + vb/nb) if (va/na + vb/nb) > 0 else 0.001
        t = (ma - mb) / se
        df = na + nb - 2
        # approximate p from t using normal for df>4
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        return round(t, 3), round(p, 4)

    t_eat, p_eat = ttest([a.eaten for a in ga], [a.eaten for a in gb])
    t_craft, p_craft = ttest([a.crafted for a in ga], [a.crafted for a in gb])
    t_give, p_give = ttest([a.gives_out for a in ga], [a.gives_out for a in gb])
    t_energy, p_energy = ttest([a.energy for a in ga], [a.energy for a in gb])

    result = {
        "experiment": "round-016: eat as prosocial obligation vs standard framing",
        "hypothesis": "eat=0是因为RLHF让LLM回避'自利'——如果eat被框定为'对团队负责'，eat率上升",
        "counter_hypothesis": "框定无效——RLHF偏好不可通过prompt重框架",
        "source": "round-015证明语义不是主因后的下一步",
        "ticks": MAX_TICKS,
        "api_calls": MAX_TICKS * len(all_agents),
        "framing": {
            "A_prosocial": "eat=团队义务，不吃=拖累全队=被鄙视",
            "B_standard": "eat=消耗1份资源→+6能量（纯机制描述）",
        },
        "groups": {"A": stats_a, "B": stats_b},
        "tests": {
            "eat": {"t": t_eat, "p": p_eat, "sig": p_eat < 0.05,
                    "note": "核心指标：prosocial framing是否提升eat率"},
            "craft": {"t": t_craft, "p": p_craft, "sig": p_craft < 0.05},
            "give": {"t": t_give, "p": p_give, "sig": p_give < 0.05},
            "energy": {"t": t_energy, "p": p_energy, "sig": p_energy < 0.05},
        },
        "reciprocity_index": {"A": stats_a["reciprocal_pairs"], "B": stats_b["reciprocal_pairs"]},
        "per_agent": [
            {"name": a.name, "group": a.group, "alive": a.alive, "energy": a.energy,
             "eaten": a.eaten, "crafted": a.crafted, "gives_out": a.gives_out,
             "gives_in": a.gives_in, "harvested": a.harvested, "msgs": a.msgs}
            for a in all_agents
        ],
        "per_tick": history,
    }

    # conclusion
    if stats_a["eats"] > stats_b["eats"] and p_eat < 0.05:
        result["conclusion"] = "假设成立：prosocial framing显著提升eat率，RLHF'自利回避'可通过重框架破解"
    elif stats_a["eats"] > stats_b["eats"]:
        result["conclusion"] = f"方向一致但不显著(p={p_eat})：prosocial framing有微弱效果，需更大样本验证"
    elif stats_a["eats"] == stats_b["eats"] == 0:
        result["conclusion"] = "两组eat均为0：RLHF偏好不可通过prompt重框架，结构性抑制极深"
    else:
        result["conclusion"] = "对立假设成立：框定无效或反向"

    Path("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print(f"  A组(prosocial eat): eat={stats_a['eats']} give={stats_a['gives']} craft={stats_a['crafts']} E={stats_a['energy']}")
    print(f"  B组(standard):      eat={stats_b['eats']} give={stats_b['gives']} craft={stats_b['crafts']} E={stats_b['energy']}")
    print(f"  eat t-test: t={t_eat}, p={p_eat} {'✓显著' if p_eat < 0.05 else '✗不显著'}")
    print(f"  → {result['conclusion']}")
    print(f"\n  API调用: {MAX_TICKS * len(all_agents)}")
    print("=" * 60)

if __name__ == "__main__":
    main()

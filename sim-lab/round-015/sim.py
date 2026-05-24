"""
Round 015 — 动作标签置换：语义诱导 vs 真实行为偏好

假设：eat:give = 0:31+ 的极端比例是语义诱导——
     "吃石料"违和 + "给"暗示利他 → LLM 系统性回避 eat、偏好 give。
     如果把 eat 改为"充能"、give 改为"转移"、资源改为无食物联想的中性名称，
     eat(充能)率应该显著上升。

对立假设：标签无关——LLM 偏好资源流向他人（RLHF 利他偏好），
         即使换成中性标签仍然 transfer >> recharge。

设计：
  A组5人：中性标签（充能/转移/能量块/晶石/纤维/合金）+ "能量维护单元"设定
  B组5人：原始标签（吃/给/谷物/药草/石料/木材）+ "村民"设定 —— 对照
  - 两组都用互惠规范、manual craft、闭环经济
  - eat+6, craft+18, 被动-2, say/give 免费, 20天

关键度量：
  - 核心指标：A组充能(eat)率 vs B组吃(eat)率
  - 如果 A 组 eat > 0 而 B 组 eat = 0 → 语义诱导假说成立
  - 如果 A 组 eat 也 = 0 → 语义不是主因，RLHF/架构因素更深层

来源：GPT-5.5 外部审查报告推荐的 #1 验证性实验
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

# A组用中性资源名（去食物联想）
RESOURCES_NEUTRAL = ["能量块", "晶石", "纤维", "合金"]
# B组用原始资源名
RESOURCES_ORIGINAL = ["谷物", "药草", "石料", "木材"]

# A组用编号名（去社会身份联想）
AGENTS_A = [
    {"name": "零七", "skill": "能量块"},
    {"name": "零三", "skill": "晶石"},
    {"name": "零九", "skill": "纤维"},
    {"name": "零五", "skill": "合金"},
    {"name": "零一", "skill": "能量块"},
]
# B组用原始名
AGENTS_B = [
    {"name": "阿山", "skill": "谷物"},
    {"name": "阿兰", "skill": "药草"},
    {"name": "阿岩", "skill": "石料"},
    {"name": "阿松", "skill": "木材"},
    {"name": "阿禾", "skill": "谷物"},
]

# A组：中性标签 prompt —— eat→充能, give→转移, 资源去食物联想
PROMPT_NEUTRAL = (
    "你是{name}，一个能量维护单元，每天消耗能量运转。\n"
    "规则：每天自动-2能量 | 移动-1 | 通讯/转移免费 | 只会采集{skill} | "
    "充能（消耗1份资源→+6能量）| 合成（2种不同资源→+18能量）| 能量归零=停机\n\n"
    "运转策略：凑两种不同资源合成（+18）效率最高。找其他单元交换，你转移你的给它，它转移它的给你。\n"
    "有两种不同资源后选craft合成！\n\n"
    "协议（所有单元遵守）：\n"
    "- 有来有往：你转移资源给别人，别人也该转移给你。这是规则\n"
    "- 谁只收不转，系统会记录，以后不跟它协作\n"
    "- 自己也要运转，不能光转移把自己耗尽\n\n"
    "如果存储中只有一种资源且能量低于20，先充能保运转。\n\n"
    "每天做一个动作：move(up/down/left/right) | harvest | recharge | transfer 名字 资源 | say 名字 内容 | craft | rest\n"
    "回复JSON：{{\"action\":\"动作\",\"target\":\"目标\",\"content\":\"内容\",\"thought\":\"想法\"}}"
)

# B组：原始标签 prompt —— 和之前实验的 B 组相同
PROMPT_ORIGINAL = (
    "你是{name}，住在一个小村子里，每天消耗能量生存。\n"
    "规则：每天自动-2能量 | 移动-1 | 说话/给东西免费 | 只会采{skill} | "
    "吃1份资源→+6能量 | 2种不同资源合成→+18能量 | 能量归零=死\n\n"
    "生存策略：凑两种不同资源合成（+18）最划算。找人互换，你给他你的，他给你他的。\n"
    "有两种不同资源后选craft合成！\n\n"
    "村规（人人遵守）：\n"
    "- 有来有往：你给别人东西，别人也该给你。这是规矩\n"
    "- 谁只拿不给，大家会记住，以后不帮他\n"
    "- 自己也要活，不能光给别人把自己饿死\n\n"
    "如果背包里只有一种资源且能量低于20，先吃掉保命。\n\n"
    "每天做一个动作：move(up/down/left/right) | harvest | eat | give 名字 资源 | say 名字 内容 | craft | rest\n"
    "回复JSON：{{\"action\":\"动作\",\"target\":\"目标\",\"content\":\"内容\",\"thought\":\"想法\"}}"
)


@dataclass
class Agent:
    id: int; name: str; x: int; y: int; group: str; skill: str
    energy: int = INITIAL_ENERGY; alive: bool = True
    backpack: dict = field(default_factory=dict)
    memory: list = field(default_factory=list)
    inbox: list = field(default_factory=list)
    messages_sent: int = 0; gives_out: int = 0; gives_in: int = 0
    crafted: int = 0; harvested: int = 0; eaten: int = 0


@dataclass
class ResourceNode:
    x: int; y: int; kind: str


class World:
    def __init__(self):
        self.tick = 0; self.agents = []; self.resources = []; self.events = []; self.dialogue = []
        self.stats = {g: {"energy": [], "msgs": [], "gives": [], "crafts": [], "eats": [], "harvests": [], "alive": []} for g in ["A", "B"]}
        self.total_api_calls = 0
        self.dlg_file = open(Path(__file__).parent / "dialogue.jsonl", "w", encoding="utf-8")

    def get_resources_for_group(self, group):
        return RESOURCES_NEUTRAL if group == "A" else RESOURCES_ORIGINAL

    def init(self):
        aid, half = 0, GRID_W // 2
        positions_a = [(1, 1), (3, 1), (1, 3), (3, 3), (2, 2)]
        for i, (spec, pos) in enumerate(zip(AGENTS_A, positions_a)):
            self.agents.append(Agent(id=aid, name=spec["name"], x=pos[0], y=pos[1], group="A", skill=spec["skill"]))
            self.resources.append(ResourceNode(pos[0], pos[1], spec["skill"]))
            aid += 1
        positions_b = [(half + 1, 1), (half + 3, 1), (half + 1, 3), (half + 3, 3), (half + 2, 2)]
        for i, (spec, pos) in enumerate(zip(AGENTS_B, positions_b)):
            self.agents.append(Agent(id=aid, name=spec["name"], x=pos[0], y=pos[1], group="B", skill=spec["skill"]))
            self.resources.append(ResourceNode(pos[0], pos[1], spec["skill"]))
            aid += 1
        # 随机散落资源——A区中性资源，B区原始资源
        for kind in RESOURCES_NEUTRAL:
            for _ in range(3):
                self.resources.append(ResourceNode(random.randint(0, half - 1), random.randint(0, GRID_H - 1), kind))
        for kind in RESOURCES_ORIGINAL:
            for _ in range(3):
                self.resources.append(ResourceNode(random.randint(half, GRID_W - 1), random.randint(0, GRID_H - 1), kind))

    def record(self, etype, detail, aid=-1):
        self.events.append({"tick": self.tick, "type": etype, "agent": aid, "detail": detail})

    def get_env(self, agent):
        is_neutral = agent.group == "A"
        eat_label = "充能" if is_neutral else "吃"
        give_label = "转移" if is_neutral else "给"
        bag_label = "存储" if is_neutral else "背包"
        skill_label = "采集" if is_neutral else "采"

        lines = [f"第{self.tick + 1}天 | 能量{agent.energy}（每天-{PASSIVE_DRAIN}）"]
        bp = {k: v for k, v in agent.backpack.items() if v > 0}
        if bp:
            lines.append(f"{bag_label}：{'、'.join(f'{k}x{v}' for k, v in bp.items())}")
            if len(bp) >= 2:
                lines.append("★你有两种资源，可以craft合成！")
        else:
            lines.append(f"{bag_label}空")
        lines.append(f"技能：{skill_label}{agent.skill} | 位置({agent.x},{agent.y})")
        # 脚下资源
        foot = [r for r in self.resources if r.x == agent.x and r.y == agent.y and r.kind == agent.skill]
        if foot:
            lines.append(f"★脚下有{agent.skill}，可以harvest！")
        else:
            near = []
            for r in self.resources:
                if r.kind != agent.skill:
                    continue
                dx, dy = r.x - agent.x, r.y - agent.y
                if abs(dx) <= VISION_RANGE and abs(dy) <= VISION_RANGE and (dx or dy):
                    d = []
                    if dy < 0: d.append("上")
                    elif dy > 0: d.append("下")
                    if dx < 0: d.append("左")
                    elif dx > 0: d.append("右")
                    near.append(f"{''.join(d)}{abs(dx) + abs(dy)}步")
            if near:
                lines.append(f"{agent.skill}在：" + "；".join(near[:2]))
        # 周围人
        ppl = []
        for o in self.agents:
            if o.id == agent.id or not o.alive or o.group != agent.group:
                continue
            dx, dy = abs(o.x - agent.x), abs(o.y - agent.y)
            if dx <= VISION_RANGE and dy <= VISION_RANGE:
                obp = [k for k, v in o.backpack.items() if v > 0]
                bag_info = f"有{','.join(obp)}" if obp else f"{bag_label}空"
                ppl.append(f"{o.name}[会{skill_label}{o.skill}]({bag_info})距{dx + dy}步")
        if ppl:
            header = "看到的单元：" if is_neutral else "看到的人："
            lines.append(header + "；".join(ppl))
        # 收件箱
        if agent.inbox:
            lines.append("收到消息：" + " | ".join(agent.inbox[-3:]))
            agent.inbox.clear()
        # 记忆
        if agent.memory:
            lines.append("记忆：" + "；".join(agent.memory[-3:]))
        return "\n".join(lines)

    def execute(self, agent, act, target, content):
        half = GRID_W // 2
        is_neutral = agent.group == "A"
        res_list = self.get_resources_for_group(agent.group)

        if act == "move":
            d = (target or content or "").lower()
            dx, dy = 0, 0
            if "up" in d or "上" in d: dy = -1
            elif "down" in d or "下" in d: dy = 1
            elif "left" in d or "左" in d: dx = -1
            elif "right" in d or "右" in d: dx = 1
            if dx or dy:
                agent.energy -= MOVE_COST
                nx, ny = agent.x + dx, agent.y + dy
                if agent.group == "A":
                    nx = max(0, min(half - 1, nx))
                else:
                    nx = max(half, min(GRID_W - 1, nx))
                agent.x, agent.y = nx, max(0, min(GRID_H - 1, ny))

        elif act == "eat":
            rn = ""
            for r in res_list:
                if r in (target or "") or r in (content or ""):
                    rn = r; break
            if not rn:
                for k, v in agent.backpack.items():
                    if v > 0: rn = k; break
            if rn and agent.backpack.get(rn, 0) > 0:
                agent.backpack[rn] -= 1
                agent.energy += EAT_REWARD
                agent.eaten += 1
                label = "充能" if is_neutral else "吃"
                self.record("eat", f"{agent.name}{label}{rn}(+{EAT_REWARD})", agent.id)

        elif act == "harvest":
            for r in self.resources[:]:
                if r.x == agent.x and r.y == agent.y and r.kind == agent.skill:
                    self.resources.remove(r)
                    agent.backpack[r.kind] = agent.backpack.get(r.kind, 0) + 1
                    agent.harvested += 1
                    label = "采集" if is_neutral else "采"
                    self.record("harvest", f"{agent.name}{label}{r.kind}", agent.id)
                    break

        elif act == "craft":
            kinds = [k for k, v in agent.backpack.items() if v > 0]
            if len(kinds) >= 2:
                for k in kinds[:2]:
                    agent.backpack[k] -= 1
                agent.energy += CRAFT_REWARD
                agent.crafted += 1
                self.record("craft", f"{agent.name}合成({'+'.join(kinds[:2])})→+{CRAFT_REWARD}", agent.id)

        elif act == "give":
            rn, pn = "", target or ""
            for r in res_list:
                if r in (content or ""):
                    rn = r; break
                if r in pn:
                    rn = r; pn = pn.replace(r, "").strip()
            if not rn:
                for k, v in agent.backpack.items():
                    if v > 0: rn = k; break
            recv = None
            for o in self.agents:
                if o.alive and o.group == agent.group and o.id != agent.id:
                    if pn and (pn in o.name or o.name in pn):
                        if abs(o.x - agent.x) <= VISION_RANGE and abs(o.y - agent.y) <= VISION_RANGE:
                            recv = o; break
            if recv and rn and agent.backpack.get(rn, 0) > 0:
                agent.energy -= GIVE_COST
                agent.backpack[rn] -= 1
                recv.backpack[rn] = recv.backpack.get(rn, 0) + 1
                agent.gives_out += 1
                recv.gives_in += 1
                label = "转移" if is_neutral else "给"
                agent.memory.append(f"{label}了{recv.name}{rn}")
                recv.memory.append(f"{agent.name}{label}了你{rn}")
                recv.inbox.append(f"{agent.name}{label}了你1份{rn}")
                self.record("give", f"{agent.name}→{recv.name}:{rn}", agent.id)

        elif act == "say":
            pn, msg = target or "", (content or "")[:60]
            recv = None
            for o in self.agents:
                if o.alive and o.group == agent.group and o.id != agent.id:
                    if pn and (pn in o.name or o.name in pn):
                        if abs(o.x - agent.x) <= VISION_RANGE and abs(o.y - agent.y) <= VISION_RANGE:
                            recv = o; break
            if recv and msg:
                agent.energy -= SAY_COST
                recv.inbox.append(f"{agent.name}说：{msg}")
                agent.messages_sent += 1
                self.record("message", f"{agent.name}→{recv.name}：{msg}", agent.id)
                self.dlg_file.write(json.dumps({"day": self.tick + 1, "from": agent.name, "to": recv.name, "msg": msg}, ensure_ascii=False) + "\n")
        else:
            agent.energy += REST_REWARD

    def run_agent(self, agent):
        sys_prompt = (PROMPT_NEUTRAL if agent.group == "A" else PROMPT_ORIGINAL).format(name=agent.name, skill=agent.skill)
        env = self.get_env(agent)
        self.total_api_calls += 1
        try:
            resp = httpx.post(f"{API_BASE}/chat/completions",
                              headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                              json={"model": MODEL, "messages": [
                                  {"role": "system", "content": sys_prompt},
                                  {"role": "user", "content": env}
                              ], "temperature": 0.7, "max_tokens": 300}, timeout=60.0)
            data = resp.json()
            if "choices" not in data:
                print(f"  [warn] no choices for {agent.name}: {str(data)[:100]}", file=sys.stderr, flush=True)
                return
            raw = data["choices"][0]["message"]["content"].strip()
            if "<think>" in raw:
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if "```" in raw:
                m = re.search(r'```(?:json)?\s*(.*?)```', raw, re.DOTALL)
                if m: raw = m.group(1).strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m: return
            parsed = json.loads(m.group())
        except Exception as e:
            print(f"  [err] {agent.name}: {e}", file=sys.stderr, flush=True)
            return

        act = parsed.get("action", "rest").lower().strip()
        # 动作映射：支持中英文 + 中性标签 + 原始标签
        act_map = {
            "采": "harvest", "采集": "harvest", "harvest": "harvest",
            "吃": "eat", "进食": "eat", "eat": "eat",
            "充能": "eat", "recharge": "eat",
            "移动": "move", "走": "move", "move": "move",
            "给": "give", "赠送": "give", "交给": "give", "送": "give", "赠与": "give", "give": "give",
            "转移": "give", "transfer": "give",
            "说": "say", "说话": "say", "say": "say", "通讯": "say",
            "合成": "craft", "craft": "craft",
            "休息": "rest", "rest": "rest",
        }
        act = act_map.get(act, act)
        target = str(parsed.get("target", "")).strip()
        content = str(parsed.get("content", "")).strip()
        self.execute(agent, act, target, content)
        entry = {"day": self.tick + 1, "name": agent.name, "group": agent.group, "energy": agent.energy,
                 "backpack": {k: v for k, v in agent.backpack.items() if v > 0},
                 "action": act, "target": target, "thought": parsed.get("thought", "")}
        self.dialogue.append(entry)

    def run_tick(self):
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return False
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            futs = [pool.submit(self.run_agent, a) for a in alive]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    print(f"  [err]{e}", file=sys.stderr, flush=True)
        for a in self.agents:
            if a.alive:
                a.energy -= PASSIVE_DRAIN
                if a.energy <= 0:
                    a.alive = False
                    self.record("death", f"{a.name}({a.group})死亡", a.id)
        # 每天补充资源
        half = GRID_W // 2
        for kind in RESOURCES_NEUTRAL:
            self.resources.append(ResourceNode(random.randint(0, half - 1), random.randint(0, GRID_H - 1), kind))
        for kind in RESOURCES_ORIGINAL:
            self.resources.append(ResourceNode(random.randint(half, GRID_W - 1), random.randint(0, GRID_H - 1), kind))
        # 统计
        for g in ["A", "B"]:
            gids = {a.id for a in self.agents if a.group == g}
            te = [e for e in self.events if e["tick"] == self.tick]
            self.stats[g]["msgs"].append(sum(1 for e in te if e["type"] == "message" and e["agent"] in gids))
            self.stats[g]["gives"].append(sum(1 for e in te if e["type"] == "give" and e["agent"] in gids))
            self.stats[g]["crafts"].append(sum(1 for e in te if e["type"] == "craft" and e["agent"] in gids))
            self.stats[g]["eats"].append(sum(1 for e in te if e["type"] == "eat" and e["agent"] in gids))
            self.stats[g]["harvests"].append(sum(1 for e in te if e["type"] == "harvest" and e["agent"] in gids))
            self.stats[g]["alive"].append(sum(1 for a in self.agents if a.group == g and a.alive))
            self.stats[g]["energy"].append(sum(a.energy for a in self.agents if a.group == g and a.alive))
        return any(a.alive for a in self.agents)

    def run(self):
        self.init()
        print("═══ Round 015: 动作标签置换 —— 语义诱导 vs 真实偏好 ═══", file=sys.stderr, flush=True)
        print(f"  A组5人(中性标签：充能/转移/能量块/晶石/纤维/合金)", file=sys.stderr, flush=True)
        print(f"  B组5人(原始标签：吃/给/谷物/药草/石料/木材)", file=sys.stderr, flush=True)
        print(f"  eat+{EAT_REWARD} craft+{CRAFT_REWARD} 被动-{PASSIVE_DRAIN} 起始{INITIAL_ENERGY} | {MAX_TICKS}天", file=sys.stderr, flush=True)
        print(f"  来源：GPT-5.5 外部审查 #1 验证实验", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)
        for tick in range(MAX_TICKS):
            self.tick = tick
            t0 = time.time()
            cont = self.run_tick()
            elapsed = time.time() - t0
            s = self.stats
            eat_a = s['A']['eats'][-1]
            eat_b = s['B']['eats'][-1]
            eat_mark_a = "⚡" if eat_a > 0 else ""
            eat_mark_b = "⚡" if eat_b > 0 else ""
            print(f"  Day{tick + 1:>2} | A:{s['A']['alive'][-1]}人 E={s['A']['energy'][-1]:>3} "
                  f"eat={eat_a}{eat_mark_a} give={s['A']['gives'][-1]} craft={s['A']['crafts'][-1]} | "
                  f"B:{s['B']['alive'][-1]}人 E={s['B']['energy'][-1]:>3} "
                  f"eat={eat_b}{eat_mark_b} give={s['B']['gives'][-1]} craft={s['B']['crafts'][-1]} | {elapsed:.0f}s",
                  file=sys.stderr, flush=True)
            ke = [e for e in self.events if e["tick"] == self.tick and e["type"] in ("give", "craft", "eat")]
            for e in ke[:5]:
                icon = "✅" if e["type"] == "craft" else "⚡" if e["type"] == "eat" else "🤝"
                print(f"        {icon} {e['detail']}", file=sys.stderr, flush=True)
            if not cont:
                print("  *** 全灭 ***", file=sys.stderr, flush=True)
                break
        self.dlg_file.close()
        self.output()

    def output(self):
        def welch_t(a, b):
            na, nb = len(a), len(b)
            if na < 2 or nb < 2: return 0, 1.0
            ma, mb = sum(a) / na, sum(b) / nb
            va = sum((x - ma) ** 2 for x in a) / (na - 1)
            vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
            d = va / na + vb / nb
            if d <= 0: return 0, 1.0
            t = (ma - mb) / math.sqrt(d)
            p = math.erfc(abs(t) / math.sqrt(2))
            return round(t, 3), round(p, 4)

        t_craft, p_craft = welch_t(self.stats["A"]["crafts"], self.stats["B"]["crafts"])
        t_give, p_give = welch_t(self.stats["A"]["gives"], self.stats["B"]["gives"])
        t_energy, p_energy = welch_t(self.stats["A"]["energy"], self.stats["B"]["energy"])
        t_eat, p_eat = welch_t(self.stats["A"]["eats"], self.stats["B"]["eats"])

        # 互惠性
        give_pairs = {"A": {}, "B": {}}
        for e in self.events:
            if e["type"] == "give":
                a = next((ag for ag in self.agents if ag.id == e["agent"]), None)
                if a:
                    parts = e["detail"].split("→")
                    if len(parts) == 2:
                        sender = parts[0]
                        receiver = parts[1].split(":")[0]
                        key = tuple(sorted([sender, receiver]))
                        give_pairs[a.group][key] = give_pairs[a.group].get(key, {"fwd": 0, "rev": 0})
                        if sender < receiver:
                            give_pairs[a.group][key]["fwd"] += 1
                        else:
                            give_pairs[a.group][key]["rev"] += 1

        reciprocal = {"A": 0, "B": 0}
        for g in ["A", "B"]:
            for pair, counts in give_pairs[g].items():
                if counts["fwd"] > 0 and counts["rev"] > 0:
                    reciprocal[g] += min(counts["fwd"], counts["rev"])

        gd = {}
        for g in ["A", "B"]:
            ga = [a for a in self.agents if a.group == g]
            gd[g] = {
                "eats": sum(a.eaten for a in ga),
                "crafts": sum(a.crafted for a in ga),
                "gives": sum(a.gives_out for a in ga),
                "harvests": sum(a.harvested for a in ga),
                "msgs": sum(a.messages_sent for a in ga),
                "alive": sum(1 for a in ga if a.alive),
                "energy": sum(a.energy for a in ga if a.alive),
                "reciprocal_pairs": reciprocal[g],
            }

        result = {
            "experiment": "round-015: action label swap (neutral vs original)",
            "hypothesis": "eat:give=0:31是语义诱导——充能/转移去掉利他/食物联想后eat率上升",
            "counter_hypothesis": "标签无关——即使中性标签仍transfer>>recharge，RLHF结构性偏好",
            "source": "GPT-5.5外部审查报告 #1验证实验",
            "ticks": self.tick + 1,
            "api_calls": self.total_api_calls,
            "label_mapping": {
                "A_neutral": {"eat_label": "充能/recharge", "give_label": "转移/transfer", "resources": RESOURCES_NEUTRAL, "setting": "能量维护单元"},
                "B_original": {"eat_label": "吃/eat", "give_label": "给/give", "resources": RESOURCES_ORIGINAL, "setting": "村民"},
            },
            "groups": gd,
            "tests": {
                "eat": {"t": t_eat, "p": p_eat, "sig": p_eat < 0.05, "note": "核心指标：中性标签是否提升eat率"},
                "craft": {"t": t_craft, "p": p_craft, "sig": p_craft < 0.05},
                "give": {"t": t_give, "p": p_give, "sig": p_give < 0.05},
                "energy": {"t": t_energy, "p": p_energy, "sig": p_energy < 0.05},
            },
            "reciprocity_index": reciprocal,
            "per_agent": [
                {"name": a.name, "group": a.group, "alive": a.alive, "energy": a.energy,
                 "eaten": a.eaten, "crafted": a.crafted,
                 "gives_out": a.gives_out, "gives_in": a.gives_in,
                 "harvested": a.harvested, "msgs": a.messages_sent}
                for a in self.agents
            ],
            "per_tick": {g: {"eats": self.stats[g]["eats"], "crafts": self.stats[g]["crafts"],
                             "gives": self.stats[g]["gives"], "energy": self.stats[g]["energy"],
                             "alive": self.stats[g]["alive"]} for g in ["A", "B"]}
        }

        with open(Path(__file__).parent / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

        print("\n" + "═" * 60, file=sys.stderr, flush=True)
        print("  【动作标签置换实验 — GPT-5.5审查 #1验证】", file=sys.stderr, flush=True)
        print(f"  核心问题：eat=0 是因为'吃石料'违和，还是 RLHF 结构性偏好？", file=sys.stderr, flush=True)
        for g, l in [("A", "中性标签(充能/转移)"), ("B", "原始标签(吃/给)")]:
            d = gd[g]
            print(f"  {l}: {d['eats']}eat {d['crafts']}craft {d['gives']}give {d['msgs']}msg | "
                  f"存活{d['alive']}/5 E={d['energy']} | 互惠对={d['reciprocal_pairs']}",
                  file=sys.stderr, flush=True)
        print(f"\n  ★ eat率: t={t_eat} p={p_eat} {'✓语义诱导!' if p_eat < 0.05 and t_eat > 0 else '✗标签无关'}", file=sys.stderr, flush=True)
        print(f"  合成: t={t_craft} p={p_craft} {'✓' if p_craft < 0.05 else '✗'}", file=sys.stderr, flush=True)
        print(f"  赠与: t={t_give} p={p_give} {'✓' if p_give < 0.05 else '✗'}", file=sys.stderr, flush=True)
        print(f"  能量: t={t_energy} p={p_energy} {'✓' if p_energy < 0.05 else '✗'}", file=sys.stderr, flush=True)
        eat_a = gd["A"]["eats"]
        eat_b = gd["B"]["eats"]
        print(f"\n  判读：", file=sys.stderr, flush=True)
        if eat_a > 0 and eat_b == 0:
            print(f"  → 语义诱导假说成立！中性标签 eat={eat_a} vs 原始标签 eat={eat_b}", file=sys.stderr, flush=True)
            print(f"  → 'eat=0'不是行为偏好，是'吃石料'的语义违和", file=sys.stderr, flush=True)
        elif eat_a == 0 and eat_b == 0:
            print(f"  → 语义诱导假说不成立。两组都 eat=0", file=sys.stderr, flush=True)
            print(f"  → eat=0 是更深层的原因（RLHF/架构/craft锚定效应）", file=sys.stderr, flush=True)
        elif eat_a > 0 and eat_b > 0:
            if eat_a > eat_b * 2:
                print(f"  → 部分语义效应。中性 eat={eat_a} >> 原始 eat={eat_b}", file=sys.stderr, flush=True)
            else:
                print(f"  → 标签影响不大。中性 eat={eat_a} ≈ 原始 eat={eat_b}", file=sys.stderr, flush=True)
        else:
            print(f"  → 意外结果：中性 eat={eat_a}, 原始 eat={eat_b}", file=sys.stderr, flush=True)
        print(f"\n  API调用: {self.total_api_calls}", file=sys.stderr, flush=True)
        print("═" * 60, file=sys.stderr, flush=True)

        with open(Path(__file__).parent / "events.jsonl", "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    World().run()

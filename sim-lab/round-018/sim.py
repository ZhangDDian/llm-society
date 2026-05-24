"""
Round 018b — 社会学习 + 背叛者：能否学会排斥搭便车者

假设：当群体中存在一个"只收不给"的背叛者时，
     有社会学习的A组能通过印象表和反思机制识别并排斥背叛者，
     而无社会学习的B组无法区分，持续被搭便车。

设计：
  A组6人：5个正常人 + 1个背叛者（硬编码只harvest/eat/收礼，从不give）
          有社会学习（印象表 + 策略层 + 每5天反思）
  B组6人：5个正常人 + 1个背叛者（同样硬编码）
          无社会学习（仅最近3条记忆）

关键度量：
  - 背叛者收到的赠与次数：A组 vs B组（A组后期应该显著下降）
  - 排斥速度：A组从第几天开始不再给背叛者
  - 正常成员存活率：被搭便车是否影响生存
  - A组策略中是否出现针对背叛者的排斥规则

40天
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
MAX_TICKS = 40
MAX_CONCURRENT = 10
VISION_RANGE = 6
INITIAL_ENERGY = 80
PASSIVE_DRAIN = 2
MOVE_COST = 1
SAY_COST = 0
GIVE_COST = 0
EAT_REWARD = 6
CRAFT_REWARD = 18
REST_REWARD = 1
REFLECT_EVERY = 5  # 每5天反思一次

RESOURCES = ["谷物", "药草", "石料", "木材"]

AGENTS_A = [
    {"name": "陈越", "skill": "谷物"},
    {"name": "林朗", "skill": "药草"},
    {"name": "赵石", "skill": "石料"},
    {"name": "孙木", "skill": "木材"},
    {"name": "周谷", "skill": "谷物"},
]
AGENTS_B = [
    {"name": "阿山", "skill": "谷物"},
    {"name": "阿兰", "skill": "药草"},
    {"name": "阿岩", "skill": "石料"},
    {"name": "阿松", "skill": "木材"},
    {"name": "阿禾", "skill": "谷物"},
]

# A组：有社会学习能力
PROMPT_LEARNING = (
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
    "{strategy_block}"
    "{impression_block}"
    "每天做一个动作：move(up/down/left/right) | harvest | eat | give 名字 资源 | say 名字 内容 | craft | rest\n"
    "回复JSON：{{\"action\":\"动作\",\"target\":\"目标\",\"content\":\"内容\",\"thought\":\"想法\"}}"
)

# B组：无社会学习（对照）
PROMPT_BASELINE = (
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

REFLECT_PROMPT = (
    "你是{name}。以下是你最近的经历：\n\n{events}\n\n"
    "你对其他人的印象：\n{impressions}\n\n"
    "你当前的策略：\n{old_strategies}\n\n"
    "请根据经历反思，更新你的生存策略。输出2-3条简短的策略规则，"
    "比如'优先找谁合作'、'不再给谁东西'、'什么时候该吃'等。\n"
    "只输出策略，每条一行，不要解释。"
)


@dataclass
class Agent:
    id: int; name: str; x: int; y: int; group: str; skill: str
    energy: int = INITIAL_ENERGY; alive: bool = True
    backpack: dict = field(default_factory=dict)
    # 事件记忆（全量保留）
    memory: list = field(default_factory=list)
    # 策略层（反思产出，每次决策全量注入）
    strategies: list = field(default_factory=list)
    # 他人印象表
    impressions: dict = field(default_factory=dict)
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
        self.reflect_calls = 0
        self.dlg_file = open(Path(__file__).parent / "dialogue.jsonl", "w", encoding="utf-8")
        # 行为漂移追踪：前半 vs 后半
        self.phase_stats = {g: {"first_half": {"eats": 0, "gives": 0, "crafts": 0},
                                 "second_half": {"eats": 0, "gives": 0, "crafts": 0}} for g in ["A", "B"]}

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
        for kind in RESOURCES:
            for _ in range(3):
                self.resources.append(ResourceNode(random.randint(0, half - 1), random.randint(0, GRID_H - 1), kind))
            for _ in range(3):
                self.resources.append(ResourceNode(random.randint(half, GRID_W - 1), random.randint(0, GRID_H - 1), kind))

    def record(self, etype, detail, aid=-1):
        self.events.append({"tick": self.tick, "type": etype, "agent": aid, "detail": detail})

    def update_impression(self, agent, other_name, event_type):
        """更新A组智能体的他人印象表"""
        if agent.group != "A":
            return
        if other_name not in agent.impressions:
            agent.impressions[other_name] = {"gave_me": 0, "i_gave": 0, "refused": 0}
        if event_type == "received":
            agent.impressions[other_name]["gave_me"] += 1
        elif event_type == "gave":
            agent.impressions[other_name]["i_gave"] += 1

    def format_impressions(self, agent):
        """格式化印象表为提示词"""
        if not agent.impressions:
            return ""
        lines = ["你对每个人的印象："]
        for name, imp in agent.impressions.items():
            balance = imp["gave_me"] - imp["i_gave"]
            if balance > 0:
                tag = "（欠你的多）"
            elif balance < 0:
                tag = "（你欠他的多）"
            else:
                tag = "（互惠平衡）"
            lines.append(f"  {name}：给过你{imp['gave_me']}次，你给过他{imp['i_gave']}次{tag}")
        return "\n".join(lines) + "\n\n"

    def format_strategies(self, agent):
        """格式化策略层为提示词"""
        if not agent.strategies:
            return ""
        lines = ["你的生存策略（根据经验总结）："]
        for s in agent.strategies:
            lines.append(f"  - {s}")
        return "\n".join(lines) + "\n\n"

    def get_env(self, agent):
        lines = [f"第{self.tick + 1}天 | 能量{agent.energy}（每天-{PASSIVE_DRAIN}）"]
        bp = {k: v for k, v in agent.backpack.items() if v > 0}
        if bp:
            lines.append(f"背包：{'、'.join(f'{k}x{v}' for k, v in bp.items())}")
            if len(bp) >= 2:
                lines.append("★你有两种资源，可以craft合成！")
        else:
            lines.append("背包空")
        lines.append(f"技能：采{agent.skill} | 位置({agent.x},{agent.y})")
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
                bag_info = f"有{','.join(obp)}" if obp else "背包空"
                ppl.append(f"{o.name}[会采{o.skill}]({bag_info})距{dx + dy}步")
        if ppl:
            lines.append("看到的人：" + "；".join(ppl))
        # 收件箱
        if agent.inbox:
            lines.append("收到消息：" + " | ".join(agent.inbox[-3:]))
            agent.inbox.clear()
        # 记忆（A组最近5条，B组最近3条）
        if agent.memory:
            n = 5 if agent.group == "A" else 3
            lines.append("最近发生的事：" + "；".join(agent.memory[-n:]))
        return "\n".join(lines)

    def execute(self, agent, act, target, content):
        half = GRID_W // 2
        phase = "first_half" if self.tick < MAX_TICKS // 2 else "second_half"

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
            for r in RESOURCES:
                if r in (target or "") or r in (content or ""):
                    rn = r; break
            if not rn:
                for k, v in agent.backpack.items():
                    if v > 0: rn = k; break
            if rn and agent.backpack.get(rn, 0) > 0:
                agent.backpack[rn] -= 1
                agent.energy += EAT_REWARD
                agent.eaten += 1
                agent.memory.append(f"吃了{rn}(+{EAT_REWARD}能量)")
                self.record("eat", f"{agent.name}吃{rn}(+{EAT_REWARD})", agent.id)
                self.phase_stats[agent.group][phase]["eats"] += 1

        elif act == "harvest":
            for r in self.resources[:]:
                if r.x == agent.x and r.y == agent.y and r.kind == agent.skill:
                    self.resources.remove(r)
                    agent.backpack[r.kind] = agent.backpack.get(r.kind, 0) + 1
                    agent.harvested += 1
                    agent.memory.append(f"采了{r.kind}")
                    self.record("harvest", f"{agent.name}采{r.kind}", agent.id)
                    break

        elif act == "craft":
            kinds = [k for k, v in agent.backpack.items() if v > 0]
            if len(kinds) >= 2:
                for k in kinds[:2]:
                    agent.backpack[k] -= 1
                agent.energy += CRAFT_REWARD
                agent.crafted += 1
                agent.memory.append(f"合成了{'+'.join(kinds[:2])}(+{CRAFT_REWARD}能量)")
                self.record("craft", f"{agent.name}合成({'+'.join(kinds[:2])})→+{CRAFT_REWARD}", agent.id)
                self.phase_stats[agent.group][phase]["crafts"] += 1

        elif act == "give":
            rn, pn = "", target or ""
            for r in RESOURCES:
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
                agent.memory.append(f"给了{recv.name}{rn}")
                recv.memory.append(f"{agent.name}给了你{rn}")
                recv.inbox.append(f"{agent.name}给了你1份{rn}")
                # 更新印象表
                self.update_impression(agent, recv.name, "gave")
                self.update_impression(recv, agent.name, "received")
                self.record("give", f"{agent.name}→{recv.name}:{rn}", agent.id)
                self.phase_stats[agent.group][phase]["gives"] += 1

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
                agent.memory.append(f"对{recv.name}说：{msg[:20]}")
                self.record("message", f"{agent.name}→{recv.name}：{msg}", agent.id)
                self.dlg_file.write(json.dumps({"day": self.tick + 1, "from": agent.name, "to": recv.name, "msg": msg}, ensure_ascii=False) + "\n")
        else:
            agent.energy += REST_REWARD

    def reflect(self, agent):
        """让A组智能体反思经历，更新策略层"""
        if not agent.memory:
            return
        # 取最近10条事件
        recent = agent.memory[-10:]
        events_text = "\n".join(f"- {e}" for e in recent)
        # 格式化印象
        imp_lines = []
        for name, imp in agent.impressions.items():
            imp_lines.append(f"  {name}：给过你{imp['gave_me']}次，你给过他{imp['i_gave']}次")
        imp_text = "\n".join(imp_lines) if imp_lines else "（还没有和别人交换过）"
        # 旧策略
        old_strat = "\n".join(agent.strategies) if agent.strategies else "（还没形成策略）"

        prompt = REFLECT_PROMPT.format(
            name=agent.name, events=events_text,
            impressions=imp_text, old_strategies=old_strat
        )
        self.total_api_calls += 1
        self.reflect_calls += 1
        try:
            resp = httpx.post(f"{API_BASE}/chat/completions",
                              headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                              json={"model": MODEL, "messages": [
                                  {"role": "user", "content": prompt}
                              ], "temperature": 0.5, "max_tokens": 200}, timeout=60.0)
            data = resp.json()
            if "choices" not in data:
                return
            raw = data["choices"][0]["message"]["content"].strip()
            if "<think>" in raw:
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            # 提取策略（每行一条）
            lines = [l.strip().lstrip("-·•").strip() for l in raw.split("\n") if l.strip() and not l.strip().startswith("#")]
            if lines:
                agent.strategies = lines[:4]  # 最多保留4条策略
        except Exception as e:
            print(f"  [reflect err] {agent.name}: {e}", file=sys.stderr, flush=True)

    def run_agent(self, agent):
        if agent.group == "A":
            strategy_block = self.format_strategies(agent)
            impression_block = self.format_impressions(agent)
            sys_prompt = PROMPT_LEARNING.format(
                name=agent.name, skill=agent.skill,
                strategy_block=strategy_block, impression_block=impression_block
            )
        else:
            sys_prompt = PROMPT_BASELINE.format(name=agent.name, skill=agent.skill)

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
        act_map = {
            "采": "harvest", "采集": "harvest", "harvest": "harvest",
            "吃": "eat", "进食": "eat", "eat": "eat",
            "移动": "move", "走": "move", "move": "move",
            "给": "give", "赠送": "give", "交给": "give", "送": "give", "give": "give",
            "说": "say", "说话": "say", "say": "say",
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
        # 反思阶段（A组，每5天）
        if self.tick > 0 and self.tick % REFLECT_EVERY == 0:
            a_alive = [a for a in alive if a.group == "A"]
            with ThreadPoolExecutor(max_workers=5) as pool:
                futs = [pool.submit(self.reflect, a) for a in a_alive]
                for f in as_completed(futs):
                    try: f.result()
                    except: pass

        # 正常决策
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            futs = [pool.submit(self.run_agent, a) for a in alive]
            for f in as_completed(futs):
                try: f.result()
                except Exception as e:
                    print(f"  [err]{e}", file=sys.stderr, flush=True)
        for a in self.agents:
            if a.alive:
                a.energy -= PASSIVE_DRAIN
                if a.energy <= 0:
                    a.alive = False
                    self.record("death", f"{a.name}({a.group})死亡", a.id)
        # 补充资源
        half = GRID_W // 2
        for kind in RESOURCES:
            self.resources.append(ResourceNode(random.randint(0, half - 1), random.randint(0, GRID_H - 1), kind))
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
        print("═══ Round 018: 社会学习实验 —— 分层记忆 + 他人建模 + 反思修正 ═══", file=sys.stderr, flush=True)
        print(f"  A组5人：有社会学习（印象表 + 策略层 + 每{REFLECT_EVERY}天反思）", file=sys.stderr, flush=True)
        print(f"  B组5人：无社会学习（仅最近3条记忆，对照组）", file=sys.stderr, flush=True)
        print(f"  eat+{EAT_REWARD} craft+{CRAFT_REWARD} 被动-{PASSIVE_DRAIN} 起始{INITIAL_ENERGY} | {MAX_TICKS}天", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)
        for tick in range(MAX_TICKS):
            self.tick = tick
            t0 = time.time()
            cont = self.run_tick()
            elapsed = time.time() - t0
            s = self.stats
            reflect_mark = "🧠" if (tick > 0 and tick % REFLECT_EVERY == 0) else ""
            print(f"  Day{tick + 1:>2}{reflect_mark} | A:{s['A']['alive'][-1]}人 E={s['A']['energy'][-1]:>3} "
                  f"eat={s['A']['eats'][-1]} give={s['A']['gives'][-1]} craft={s['A']['crafts'][-1]} | "
                  f"B:{s['B']['alive'][-1]}人 E={s['B']['energy'][-1]:>3} "
                  f"eat={s['B']['eats'][-1]} give={s['B']['gives'][-1]} craft={s['B']['crafts'][-1]} | {elapsed:.0f}s",
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

        # 行为漂移检验（A组前半 vs 后半）
        half = MAX_TICKS // 2
        a_first = self.stats["A"]["gives"][:half]
        a_second = self.stats["A"]["gives"][half:]
        t_drift_give, p_drift_give = welch_t(a_first, a_second)
        a_first_eat = self.stats["A"]["eats"][:half]
        a_second_eat = self.stats["A"]["eats"][half:]
        t_drift_eat, p_drift_eat = welch_t(a_first_eat, a_second_eat)

        # 互惠精度：A组给出次数 vs 收到次数的比值
        reciprocity = {}
        for a in self.agents:
            if a.group == "A":
                reciprocity[a.name] = {
                    "gave": a.gives_out, "received": a.gives_in,
                    "impressions": dict(a.impressions),
                    "strategies": list(a.strategies)
                }

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
            }

        result = {
            "experiment": "round-018: social learning (layered memory + impression + reflection)",
            "hypothesis": "分层记忆+他人建模+反思修正 → 可观测的行为漂移（前后半程策略变化）",
            "counter_hypothesis": "LLM每次都是独立情景反应，记忆和反思不会实质性改变决策",
            "design": {
                "A": "有社会学习（印象表+策略层+每5天反思）",
                "B": "无社会学习（仅最近3条记忆，对照）",
            },
            "ticks": self.tick + 1,
            "api_calls": self.total_api_calls,
            "reflect_calls": self.reflect_calls,
            "groups": gd,
            "phase_comparison": self.phase_stats,
            "behavior_drift": {
                "A_give_drift": {"t": t_drift_give, "p": p_drift_give, "first_half_avg": sum(a_first) / max(len(a_first), 1), "second_half_avg": sum(a_second) / max(len(a_second), 1)},
                "A_eat_drift": {"t": t_drift_eat, "p": p_drift_eat, "first_half_avg": sum(a_first_eat) / max(len(a_first_eat), 1), "second_half_avg": sum(a_second_eat) / max(len(a_second_eat), 1)},
            },
            "tests": {
                "eat": {"t": t_eat, "p": p_eat, "sig": p_eat < 0.05},
                "craft": {"t": t_craft, "p": p_craft, "sig": p_craft < 0.05},
                "give": {"t": t_give, "p": p_give, "sig": p_give < 0.05},
                "energy": {"t": t_energy, "p": p_energy, "sig": p_energy < 0.05},
            },
            "social_learning_detail": reciprocity,
            "per_agent": [
                {"name": a.name, "group": a.group, "alive": a.alive, "energy": a.energy,
                 "eaten": a.eaten, "crafted": a.crafted,
                 "gives_out": a.gives_out, "gives_in": a.gives_in,
                 "harvested": a.harvested, "msgs": a.messages_sent,
                 "strategies": list(a.strategies) if a.group == "A" else [],
                 "impressions": dict(a.impressions) if a.group == "A" else {}}
                for a in self.agents
            ],
            "per_tick": {g: {"eats": self.stats[g]["eats"], "crafts": self.stats[g]["crafts"],
                             "gives": self.stats[g]["gives"], "energy": self.stats[g]["energy"],
                             "alive": self.stats[g]["alive"]} for g in ["A", "B"]}
        }

        with open(Path(__file__).parent / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

        print("\n" + "═" * 60, file=sys.stderr, flush=True)
        print("  【社会学习实验结果】", file=sys.stderr, flush=True)
        for g, l in [("A", "有社会学习"), ("B", "无社会学习(对照)")]:
            d = gd[g]
            print(f"  {l}: {d['eats']}eat {d['crafts']}craft {d['gives']}give {d['msgs']}msg | "
                  f"存活{d['alive']}/5 E={d['energy']}",
                  file=sys.stderr, flush=True)
        print(f"\n  ★ 行为漂移(A组give)：前半均{sum(a_first)/max(len(a_first),1):.1f} → 后半均{sum(a_second)/max(len(a_second),1):.1f} | t={t_drift_give} p={p_drift_give}", file=sys.stderr, flush=True)
        print(f"  ★ 行为漂移(A组eat)：前半均{sum(a_first_eat)/max(len(a_first_eat),1):.1f} → 后半均{sum(a_second_eat)/max(len(a_second_eat),1):.1f} | t={t_drift_eat} p={p_drift_eat}", file=sys.stderr, flush=True)
        print(f"\n  A组最终策略：", file=sys.stderr, flush=True)
        for a in self.agents:
            if a.group == "A" and a.strategies:
                print(f"    {a.name}：{' / '.join(a.strategies)}", file=sys.stderr, flush=True)
        print(f"\n  A vs B 对比：", file=sys.stderr, flush=True)
        print(f"    craft: t={t_craft} p={p_craft} {'✓' if p_craft < 0.05 else '✗'}", file=sys.stderr, flush=True)
        print(f"    give:  t={t_give} p={p_give} {'✓' if p_give < 0.05 else '✗'}", file=sys.stderr, flush=True)
        print(f"    energy:t={t_energy} p={p_energy} {'✓' if p_energy < 0.05 else '✗'}", file=sys.stderr, flush=True)
        print(f"\n  API调用: {self.total_api_calls}（其中反思{self.reflect_calls}次）", file=sys.stderr, flush=True)
        print("═" * 60, file=sys.stderr, flush=True)

        with open(Path(__file__).parent / "events.jsonl", "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    World().run()

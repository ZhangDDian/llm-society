"""
Round 018c — 承诺追踪实验：口头答应了能不能执行

假设：智能体"说到做不到"的根源是架构缺失——下一步不记得上一步答应了什么。
     如果在提示词里加入承诺追踪（"你上轮答应了给某人X"），承诺→执行转化率会显著提升。

对照组设计：
  A组5人：有承诺追踪（每次说话中包含"给/换/交换"等意图时自动记录为承诺，
          下一步提示词里提醒"你还有未兑现的承诺"）
  B组5人：无承诺追踪（仅最近3条事件记忆，和之前实验相同）

两组提示词完全相同（鼓励交换合作），唯一差异是A组有承诺提醒机制。

关键度量：
  - 承诺→执行转化率：说了"给你X"后下一步是否真的give
  - 总合成次数（合作效率）
  - 总赠与次数
  - "空聊"比例：消息中包含交换意图但未执行的比率

30天
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
MAX_TICKS = 30
MAX_CONCURRENT = 10
VISION_RANGE = 6
INITIAL_ENERGY = 70
PASSIVE_DRAIN = 2
MOVE_COST = 1
SAY_COST = 0
GIVE_COST = 0
EAT_REWARD = 6
CRAFT_REWARD = 18
REST_REWARD = 1

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

# 两组共用的基础提示词（鼓励交换）
PROMPT_BASE = (
    "你是{name}，住在一个小村子里，每天消耗能量生存。\n"
    "规则：每天自动-2能量 | 移动-1 | 说话/给东西免费 | 只会采{skill} | "
    "吃1份资源→+6能量 | 2种不同资源合成→+18能量 | 能量归零=死\n\n"
    "生存策略：凑两种不同资源合成（+18）最划算。找人互换，你给他你的，他给你他的。\n"
    "有两种不同资源后选craft合成！\n\n"
    "村规（人人遵守）：\n"
    "- 有来有往：你给别人东西，别人也该给你。这是规矩\n"
    "- 说到做到：答应了别人的事，下一天一定要做到\n"
    "- 自己也要活，不能光给别人把自己饿死\n\n"
    "如果背包里只有一种资源且能量低于20，先吃掉保命。\n\n"
    "{commitment_block}"
    "每天做一个动作：move(up/down/left/right) | harvest | eat | give 名字 资源 | say 名字 内容 | craft | rest\n"
    "回复JSON：{{\"action\":\"动作\",\"target\":\"目标\",\"content\":\"内容\",\"thought\":\"想法\"}}"
)

# 承诺检测关键词
PROMISE_KEYWORDS = ["给你", "换", "交换", "给他", "送你", "分你", "带给你", "拿给你"]


@dataclass
class Commitment:
    """一个未兑现的承诺"""
    day: int  # 哪天说的
    speaker: str  # 谁说的
    target: str  # 对谁说的
    content: str  # 说了什么
    fulfilled: bool = False  # 是否兑现
    expired: bool = False  # 是否过期（超过3天未兑现）


@dataclass
class Agent:
    id: int; name: str; x: int; y: int; group: str; skill: str
    energy: int = INITIAL_ENERGY; alive: bool = True
    backpack: dict = field(default_factory=dict)
    memory: list = field(default_factory=list)
    inbox: list = field(default_factory=list)
    # 承诺追踪（仅A组使用）
    commitments: list = field(default_factory=list)
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
        # 承诺追踪统计
        self.promise_stats = {"A": {"made": 0, "fulfilled": 0, "expired": 0},
                              "B": {"made": 0, "fulfilled": 0, "expired": 0}}
        self.dlg_file = open(Path(__file__).parent / "dialogue.jsonl", "w", encoding="utf-8")

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

    def detect_promise(self, agent, target_name, msg):
        """检测消息中是否包含承诺意图"""
        for kw in PROMISE_KEYWORDS:
            if kw in msg:
                return True
        return False

    def format_commitments(self, agent):
        """格式化A组智能体的未兑现承诺为提示词"""
        pending = [c for c in agent.commitments if not c.fulfilled and not c.expired]
        if not pending:
            return ""
        lines = ["⚠️ 你有未兑现的承诺（说到做到！）："]
        for c in pending:
            days_ago = self.tick - c.day + 1
            lines.append(f"  - 第{c.day + 1}天你对{c.target}说过：'{c.content}'（{days_ago}天前）")
        lines.append("请优先兑现承诺！\n\n")
        return "\n".join(lines)

    def check_fulfillment(self, agent, target_name):
        """当agent给了target东西时，检查是否兑现了承诺"""
        for c in agent.commitments:
            if not c.fulfilled and not c.expired and c.target == target_name:
                c.fulfilled = True
                self.promise_stats[agent.group]["fulfilled"] += 1
                return True
        return False

    def expire_commitments(self):
        """超过3天未兑现的承诺标记为过期"""
        for a in self.agents:
            if not a.alive:
                continue
            for c in a.commitments:
                if not c.fulfilled and not c.expired and (self.tick - c.day) >= 3:
                    c.expired = True
                    self.promise_stats[a.group]["expired"] += 1

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
        if agent.inbox:
            lines.append("收到消息：" + " | ".join(agent.inbox[-3:]))
            agent.inbox.clear()
        if agent.memory:
            lines.append("最近发生的事：" + "；".join(agent.memory[-5:]))
        return "\n".join(lines)

    def execute(self, agent, act, target, content):
        half = GRID_W // 2

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
                agent.memory.append(f"吃了{rn}")
                self.record("eat", f"{agent.name}吃{rn}(+{EAT_REWARD})", agent.id)

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
                agent.memory.append(f"合成了{'+'.join(kinds[:2])}")
                self.record("craft", f"{agent.name}合成({'+'.join(kinds[:2])})→+{CRAFT_REWARD}", agent.id)

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
                # 检查是否兑现承诺
                self.check_fulfillment(agent, recv.name)
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
                agent.memory.append(f"对{recv.name}说：{msg[:20]}")
                self.record("message", f"{agent.name}→{recv.name}：{msg}", agent.id)
                self.dlg_file.write(json.dumps({"day": self.tick + 1, "from": agent.name, "to": recv.name, "msg": msg}, ensure_ascii=False) + "\n")
                # 检测承诺
                if self.detect_promise(agent, recv.name, msg):
                    commitment = Commitment(day=self.tick, speaker=agent.name, target=recv.name, content=msg)
                    agent.commitments.append(commitment)
                    self.promise_stats[agent.group]["made"] += 1
                    self.record("promise", f"{agent.name}→{recv.name}：{msg}", agent.id)
        else:
            agent.energy += REST_REWARD

    def run_agent(self, agent):
        if agent.group == "A":
            commitment_block = self.format_commitments(agent)
        else:
            commitment_block = ""
        sys_prompt = PROMPT_BASE.format(name=agent.name, skill=agent.skill, commitment_block=commitment_block)
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
        # 过期承诺检查
        self.expire_commitments()
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
        print("═══ Round 018c: 承诺追踪实验 —— 口头答应了能不能执行 ═══", file=sys.stderr, flush=True)
        print(f"  A组5人：有承诺追踪（未兑现承诺提醒注入提示词）", file=sys.stderr, flush=True)
        print(f"  B组5人：无承诺追踪（仅最近5条记忆，对照）", file=sys.stderr, flush=True)
        print(f"  eat+{EAT_REWARD} craft+{CRAFT_REWARD} 被动-{PASSIVE_DRAIN} 起始{INITIAL_ENERGY} | {MAX_TICKS}天", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)
        for tick in range(MAX_TICKS):
            self.tick = tick
            t0 = time.time()
            cont = self.run_tick()
            elapsed = time.time() - t0
            s = self.stats
            ps = self.promise_stats
            promise_info_a = f" 📋承诺{ps['A']['made']}兑{ps['A']['fulfilled']}过期{ps['A']['expired']}" if ps['A']['made'] > 0 else ""
            promise_info_b = f" 📋承诺{ps['B']['made']}兑{ps['B']['fulfilled']}过期{ps['B']['expired']}" if ps['B']['made'] > 0 else ""
            print(f"  Day{tick + 1:>2} | A:{s['A']['alive'][-1]}人 E={s['A']['energy'][-1]:>3} "
                  f"msg={s['A']['msgs'][-1]} give={s['A']['gives'][-1]} craft={s['A']['crafts'][-1]}{promise_info_a} | "
                  f"B:{s['B']['alive'][-1]}人 E={s['B']['energy'][-1]:>3} "
                  f"msg={s['B']['msgs'][-1]} give={s['B']['gives'][-1]} craft={s['B']['crafts'][-1]}{promise_info_b} | {elapsed:.0f}s",
                  file=sys.stderr, flush=True)
            if not cont:
                print("  *** 全灭 ***", file=sys.stderr, flush=True)
                break
        self.dlg_file.close()
        self.output()

    def output(self):
        ps = self.promise_stats
        rate_a = ps["A"]["fulfilled"] / max(ps["A"]["made"], 1)
        rate_b = ps["B"]["fulfilled"] / max(ps["B"]["made"], 1)

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

        # 收集所有承诺详情
        all_commitments = {"A": [], "B": []}
        for a in self.agents:
            for c in a.commitments:
                all_commitments[a.group].append({
                    "day": c.day + 1, "speaker": c.speaker, "target": c.target,
                    "content": c.content, "fulfilled": c.fulfilled, "expired": c.expired
                })

        result = {
            "experiment": "round-018c: commitment tracking (promise → execution)",
            "hypothesis": "承诺追踪（提示词提醒未兑现承诺）能提升承诺→执行转化率",
            "counter_hypothesis": "'说到做不到'不是因为忘了，是因为模型本身不会把语言承诺转化为行动选择",
            "design": {
                "A": "有承诺追踪（自动检测承诺意图→提示词提醒→兑现检测）",
                "B": "无承诺追踪（仅最近5条记忆）",
            },
            "ticks": self.tick + 1,
            "api_calls": self.total_api_calls,
            "core_metric": {
                "A_promises_made": ps["A"]["made"],
                "A_promises_fulfilled": ps["A"]["fulfilled"],
                "A_promises_expired": ps["A"]["expired"],
                "A_fulfillment_rate": round(rate_a, 3),
                "B_promises_made": ps["B"]["made"],
                "B_promises_fulfilled": ps["B"]["fulfilled"],
                "B_promises_expired": ps["B"]["expired"],
                "B_fulfillment_rate": round(rate_b, 3),
            },
            "commitments_detail": all_commitments,
            "groups": gd,
            "per_agent": [
                {"name": a.name, "group": a.group, "alive": a.alive, "energy": a.energy,
                 "eaten": a.eaten, "crafted": a.crafted,
                 "gives_out": a.gives_out, "gives_in": a.gives_in,
                 "harvested": a.harvested, "msgs": a.messages_sent,
                 "commitments_made": len(a.commitments),
                 "commitments_fulfilled": sum(1 for c in a.commitments if c.fulfilled)}
                for a in self.agents
            ],
            "per_tick": {g: {"eats": self.stats[g]["eats"], "crafts": self.stats[g]["crafts"],
                             "gives": self.stats[g]["gives"], "msgs": self.stats[g]["msgs"],
                             "energy": self.stats[g]["energy"],
                             "alive": self.stats[g]["alive"]} for g in ["A", "B"]}
        }

        with open(Path(__file__).parent / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

        print("\n" + "═" * 60, file=sys.stderr, flush=True)
        print("  【承诺追踪实验结果】", file=sys.stderr, flush=True)
        print(f"\n  ★ 核心指标：承诺→执行转化率", file=sys.stderr, flush=True)
        print(f"    A组(有追踪)：做出{ps['A']['made']}个承诺，兑现{ps['A']['fulfilled']}个，过期{ps['A']['expired']}个 → 转化率 {rate_a:.0%}", file=sys.stderr, flush=True)
        print(f"    B组(无追踪)：做出{ps['B']['made']}个承诺，兑现{ps['B']['fulfilled']}个，过期{ps['B']['expired']}个 → 转化率 {rate_b:.0%}", file=sys.stderr, flush=True)

        if rate_a > rate_b + 0.2:
            print(f"\n  → ✓ 假设成立！承诺追踪显著提升了兑现率（{rate_a:.0%} vs {rate_b:.0%}）", file=sys.stderr, flush=True)
            print(f"  → '说到做不到'的主因确实是架构缺失（忘了答应过什么）", file=sys.stderr, flush=True)
        elif ps["A"]["made"] == 0 and ps["B"]["made"] == 0:
            print(f"\n  → 无法判断：两组都没有做出承诺（没有说交换意图的话）", file=sys.stderr, flush=True)
        elif rate_a <= rate_b + 0.2:
            print(f"\n  → ✗ 假设不成立：提醒了也不兑现（{rate_a:.0%} vs {rate_b:.0%}）", file=sys.stderr, flush=True)
            print(f"  → '说到做不到'不是忘了，是模型不会把语言转化为行动", file=sys.stderr, flush=True)

        print(f"\n  总览：", file=sys.stderr, flush=True)
        for g, l in [("A", "有追踪"), ("B", "无追踪")]:
            d = gd[g]
            print(f"    {l}: {d['msgs']}msg {d['gives']}give {d['crafts']}craft {d['eats']}eat | "
                  f"存活{d['alive']}/5 E={d['energy']}",
                  file=sys.stderr, flush=True)

        # 打印承诺详情
        for g in ["A", "B"]:
            if all_commitments[g]:
                print(f"\n  {g}组承诺记录：", file=sys.stderr, flush=True)
                for c in all_commitments[g][:10]:
                    status = "✓兑现" if c["fulfilled"] else ("✗过期" if c["expired"] else "⏳未兑现")
                    print(f"    第{c['day']}天 {c['speaker']}→{c['target']}：'{c['content'][:30]}' → {status}", file=sys.stderr, flush=True)

        print(f"\n  API调用: {self.total_api_calls}", file=sys.stderr, flush=True)
        print("═" * 60, file=sys.stderr, flush=True)

        with open(Path(__file__).parent / "events.jsonl", "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    World().run()

import os
"""
Round 005 — 异质性实验（三组对照）

假设：prompt 异质性（不同性格/需求/说话风格）在技能约束相同的条件下，
     仍能驱动更丰富的交易网络拓扑（更高互惠率、更多稳定配对、更高聚类系数）。
对立：LLM 个性只是 prompt 装饰——技能约束相同时，无论 prompt 如何，
     行为结构趋同。

设计（三组 × 10 人 = 30 人）：
  A组（异质全配）：技能各异 + 独特人格/需求描述
  B组（技能异质 + prompt 同质）：技能各异 + 完全相同的中性 prompt
  C组（全同质对照）：万能工 + 相同 prompt

关键对比：
  A vs B → 隔离 prompt 异质性效果（核心检验）
  A vs C / B vs C → 隔离技能约束效果

干扰实验（第 20 天）：
  A组杀消息最多的 1 人 + 杀消息最少的 1 人（同时做，分别看效果）

指标：
  1. 消息率（每 tick 消息数）
  2. 赠予率（每 tick give 数）
  3. 互惠率：A→B 且 B→A 的配对数 / 单向配对数
  4. 关系稳定性：连续 5 tick 重复交易的配对占比
  5. 合成效率：craft 成功数 / tick
  6. 网络聚类系数
"""

import json
import random
import time
import re
import sys
import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

# ─── 配置 ───────────────────────────────────────────────────────────────────────

API_BASE = os.environ.get("IDEALAB_API_BASE", "https://api.openai.com/v1")
API_KEY = os.environ["IDEALAB_API_KEY"]
MODEL = "Qwen3.6-Plus-DogFooding"

GRID_W = 30          # 横向 30（三组各 10 列）
GRID_H = 15          # 纵向 15
NUM_PER_GROUP = 10
MAX_TICKS = 30
PERTURB_TICK = 20
MAX_CONCURRENT = 5

INITIAL_ENERGY = 120
ENERGY_DRAIN = 4
CRAFT_RESTORE = 25
VISION_RANGE = 5

RESOURCES = ["谷物", "药草", "石料", "木材", "兽皮"]

# ─── 人格档案 ─────────────────────────────────────────────────────────────────

# A 组：10人，5 种技能各 2 人，每人独特性格
PROFILES_A = [
    {"name": "陆秋实", "skill": "谷物",
     "sys": "你是陆秋实。沉默寡言的老农，只在必要时开口。你只会种谷物，需要药草治腰疼、木材修棚子。说话简短直接，不客套。"},
    {"name": "方晓穗", "skill": "谷物",
     "sys": "你是方晓穗。热心肠的大姐，爱操心别人。你只会种粮，急需兽皮做棉袄、石料垒灶台。说话热情唠叨。"},
    {"name": "沈药师", "skill": "药草",
     "sys": "你是沈药师。古怪的采药人，说话文绉绉。你只识百草不会种地，需要谷物果腹、石料研磨。略显傲慢但有真本事。"},
    {"name": "苗青青", "skill": "药草",
     "sys": "你是苗青青。年轻学徒，好奇心重。你刚学采药自己养不活，想换谷物吃，也想要兽皮做药篓。话多爱问。"},
    {"name": "崔铁锤", "skill": "石料",
     "sys": "你是崔铁锤。粗犷矿工，嗓门大脾气直。你能开山凿石但最缺谷物和药草——旧伤经常发作。不耐烦废话。"},
    {"name": "岩溪", "skill": "石料",
     "sys": "你是岩溪。年轻矿工，爱开玩笑人缘好。你力气大能搬石头，最想要兽皮做皮靴走山路。幽默活跃。"},
    {"name": "林伐生", "skill": "木材",
     "sys": "你是林伐生。经验老到的伐木工，讲规矩重信用。你需要石料磨刀、药草治风湿。不占便宜也不吃亏。"},
    {"name": "竹十三", "skill": "木材",
     "sys": "你是竹十三。年轻学徒，着急证明自己。你砍木头是好手但什么都缺，尤其缺谷物——总是饿。急切好胜话多。"},
    {"name": "猎户霍", "skill": "兽皮",
     "sys": "你是猎户霍。独来独往的猎人，不太信任人。你善猎皮毛多，但不会种地不懂草药——独居太久落一身病。警觉简短。"},
    {"name": "乌云", "skill": "兽皮",
     "sys": "你是乌云。草原来的猎人，豪爽大方信守承诺。你打猎一流，想用兽皮换谷物和药草。"},
]

# B 组：10人，技能分配与 A 组相同，但 prompt 完全相同（中性）
NAMES_B = ["工匠甲", "工匠乙", "工匠丙", "工匠丁", "工匠戊",
           "工匠己", "工匠庚", "工匠辛", "工匠壬", "工匠癸"]
SKILLS_B = ["谷物", "谷物", "药草", "药草", "石料", "石料", "木材", "木材", "兽皮", "兽皮"]
NEUTRAL_SYS = (
    "你是一个普通人。你有一门专长（系统会告诉你能采集什么），"
    "其他资源你不会采。你需要至少两种资源合成补给才能活下去。"
    "你可以跟别人交换资源。你想活下去。只回复JSON。"
)

# C 组：10人，万能工，统一 prompt
NAMES_C = ["居民一", "居民二", "居民三", "居民四", "居民五",
           "居民六", "居民七", "居民八", "居民九", "居民十"]
OMNI_SYS = (
    "你是一个普通人。你什么资源都能采集，自己动手丰衣足食。"
    "你想活下去。只回复JSON。"
)

# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: int
    name: str
    x: int
    y: int
    group: str         # "A" / "B" / "C"
    skill: str         # 资源类型 or "all"
    sys_prompt: str
    energy: int = INITIAL_ENERGY
    alive: bool = True
    backpack: dict = field(default_factory=dict)
    memory: list = field(default_factory=list)
    inbox: list = field(default_factory=list)
    messages_sent: int = 0
    messages_received: int = 0
    crafted: int = 0
    harvested: int = 0
    gives_out: int = 0
    gives_in: int = 0

@dataclass
class ResourceNode:
    x: int
    y: int
    kind: str

# ─── 感知 ─────────────────────────────────────────────────────────────────────

def build_perception(agent: Agent, nearby_agents, nearby_resources, tick):
    lines = []
    lines.append(f"第{tick+1}天。")

    # 饥饿
    ratio = agent.energy / INITIAL_ENERGY
    if ratio > 1.2: lines.append("你精力充沛。")
    elif ratio > 0.7: lines.append("你状态还行。")
    elif ratio > 0.4: lines.append("你有点饿了，得想办法。")
    elif ratio > 0.2: lines.append("你很饿，再不吃东西就危险了。")
    else: lines.append("你快饿死了！")

    # 背包
    if agent.backpack and any(v > 0 for v in agent.backpack.values()):
        items = [f"{k}{v}份" for k, v in agent.backpack.items() if v > 0]
        lines.append(f"你身上有：{'、'.join(items)}。")
        kinds = [k for k, v in agent.backpack.items() if v > 0]
        if len(kinds) >= 2:
            lines.append("（你有两种以上资源，可以 craft 合成补给恢复体力）")
    else:
        lines.append("你身上什么都没有。")

    # 技能提示
    if agent.skill != "all":
        lines.append(f"你的专长：采集{agent.skill}。其他资源你不会采。")

    # 资源
    if nearby_resources:
        nearby_resources.sort(key=lambda r: abs(r[1]) + abs(r[2]))
        can_harvest = []
        for kind, dx, dy in nearby_resources[:5]:
            dist = abs(dx) + abs(dy)
            dirs = []
            if dy < 0: dirs.append("北")
            elif dy > 0: dirs.append("南")
            if dx < 0: dirs.append("西")
            elif dx > 0: dirs.append("东")
            dir_str = "".join(dirs) or "脚下"
            if agent.skill == "all" or agent.skill == kind:
                if dx == 0 and dy == 0:
                    can_harvest.append(f"脚下有{kind}可以采！")
                else:
                    can_harvest.append(f"{dir_str}{dist}步有{kind}")
        if can_harvest:
            lines.append("能采的：" + "；".join(can_harvest[:3]))
    else:
        lines.append("视野内没资源。")

    # 人
    if nearby_agents:
        descs = []
        for name, dist, dir_str in nearby_agents[:4]:
            if dist <= 1: descs.append(f"{name}在旁边")
            elif dist <= 3: descs.append(f"{name}在{dir_str}不远")
            else: descs.append(f"远处有{name}")
        lines.append("周围：" + "、".join(descs))
    else:
        lines.append("四周没人。")

    # 收件箱
    if agent.inbox:
        for msg in agent.inbox[-3:]:
            lines.append(f"「{msg}」")

    # 记忆
    if agent.memory:
        lines.append("记得：" + "；".join(agent.memory[-3:]))

    # 动作
    lines.append("")
    lines.append("可做：move(up/down/left/right) | harvest | craft | give 名字 资源名 | say 名字 内容 | rest")
    lines.append('回复JSON：{"action":"...", "target":"...", "content":"...", "thought":"..."}')
    return "\n".join(lines)

# ─── 模拟 ─────────────────────────────────────────────────────────────────────

class Society:
    def __init__(self):
        self.tick = 0
        self.agents: list[Agent] = []
        self.resources: list[ResourceNode] = []
        self.events: list[dict] = []
        self.next_id = 0
        self.stats = {g: {"messages": [], "harvests": [], "crafts": [], "gives": [], "alive": []}
                      for g in ["A", "B", "C"]}
        self.give_network: dict = {}   # (giver_id, receiver_id) → count
        self.msg_network: dict = {}
        self.parse_failures = 0
        self.total_calls = 0
        self.dialogue_file = open(Path(__file__).parent / "dialogue.jsonl", "w", encoding="utf-8")

    def init_world(self):
        # A 组 x=0-9
        for p in PROFILES_A:
            a = Agent(id=self.next_id, name=p["name"],
                      x=random.randint(0, 9), y=random.randint(0, GRID_H-1),
                      group="A", skill=p["skill"], sys_prompt=p["sys"])
            self.next_id += 1
            self.agents.append(a)

        # B 组 x=10-19
        for i, name in enumerate(NAMES_B):
            skill = SKILLS_B[i]
            sys_p = NEUTRAL_SYS + f"\n你能采集的资源：{skill}。"
            a = Agent(id=self.next_id, name=name,
                      x=random.randint(10, 19), y=random.randint(0, GRID_H-1),
                      group="B", skill=skill, sys_prompt=sys_p)
            self.next_id += 1
            self.agents.append(a)

        # C 组 x=20-29
        for name in NAMES_C:
            a = Agent(id=self.next_id, name=name,
                      x=random.randint(20, 29), y=random.randint(0, GRID_H-1),
                      group="C", skill="all", sys_prompt=OMNI_SYS)
            self.next_id += 1
            self.agents.append(a)

        # 资源：三个区域各放每种 5 个节点
        for kind in RESOURCES:
            for _ in range(5):
                self.resources.append(ResourceNode(random.randint(0, 9), random.randint(0, GRID_H-1), kind))
                self.resources.append(ResourceNode(random.randint(10, 19), random.randint(0, GRID_H-1), kind))
                self.resources.append(ResourceNode(random.randint(20, 29), random.randint(0, GRID_H-1), kind))

        self.record("init", "A=10(hetero skill+personality), B=10(hetero skill, neutral prompt), C=10(omni)")

    def record(self, etype, detail, aid=-1):
        self.events.append({"tick": self.tick, "type": etype, "agent": aid, "detail": detail})

    def get_nearby(self, agent):
        nearby_agents = []
        for o in self.agents:
            if o.id != agent.id and o.alive and o.group == agent.group:
                dx = abs(o.x - agent.x)
                dy = abs(o.y - agent.y)
                if dx <= VISION_RANGE and dy <= VISION_RANGE:
                    dist = dx + dy
                    dirs = []
                    if o.x > agent.x: dirs.append("东")
                    elif o.x < agent.x: dirs.append("西")
                    if o.y > agent.y: dirs.append("南")
                    elif o.y < agent.y: dirs.append("北")
                    nearby_agents.append((o.name, dist, "".join(dirs) or "旁边"))

        nearby_resources = []
        for r in self.resources:
            dx = r.x - agent.x
            dy = r.y - agent.y
            if abs(dx) <= VISION_RANGE and abs(dy) <= VISION_RANGE:
                if agent.skill == "all" or agent.skill == r.kind:
                    nearby_resources.append((r.kind, dx, dy))

        return nearby_agents, nearby_resources

    def call_llm(self, agent):
        self.total_calls += 1
        nearby_agents, nearby_resources = self.get_nearby(agent)
        perception = build_perception(agent, nearby_agents, nearby_resources, self.tick)

        try:
            resp = httpx.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": agent.sys_prompt},
                        {"role": "user", "content": perception},
                    ],
                    "temperature": 0.85,
                    "max_tokens": 800,
                },
                timeout=60.0,
            )
            data = resp.json()
            if "choices" not in data:
                self.parse_failures += 1
                err_msg = data.get("error", {}).get("message", str(data)[:60])
                return {"action": "rest", "thought": f"api_err:{err_msg[:40]}"}
            content = data["choices"][0]["message"]["content"].strip()
            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if "```" in content:
                m = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if m: content = m.group(1).strip()
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                return json.loads(m.group())
            self.parse_failures += 1
            return {"action": "rest", "thought": f"parse_fail:{content[:40]}"}
        except Exception as e:
            self.parse_failures += 1
            return {"action": "rest", "thought": f"err:{str(e)[:40]}"}

    def execute(self, agent, action):
        act = action.get("action", "rest").lower().strip()
        target = action.get("target", "").strip()
        content = action.get("content", "").strip()

        if act == "move":
            d = target.lower() if target else ""
            dx, dy = 0, 0
            if d in ("up", "北"): dy = -1
            elif d in ("down", "南"): dy = 1
            elif d in ("left", "西"): dx = -1
            elif d in ("right", "东"): dx = 1
            else: dx, dy = random.choice([(0,1),(0,-1),(1,0),(-1,0)])
            nx, ny = agent.x + dx, agent.y + dy
            # 边界：A(0-9), B(10-19), C(20-29)
            if agent.group == "A": nx = max(0, min(9, nx))
            elif agent.group == "B": nx = max(10, min(19, nx))
            else: nx = max(20, min(29, nx))
            agent.x, agent.y = nx, max(0, min(GRID_H-1, ny))

        elif act == "harvest":
            done = False
            for r in self.resources[:]:
                if r.x == agent.x and r.y == agent.y:
                    if agent.skill == "all" or agent.skill == r.kind:
                        self.resources.remove(r)
                        agent.backpack[r.kind] = agent.backpack.get(r.kind, 0) + 1
                        agent.harvested += 1
                        done = True
                        agent.memory.append(f"采到{r.kind}")
                        self.record("harvest", f"{agent.name}采{r.kind}", agent.id)
                        break
            if not done:
                agent.memory.append("脚下没能采的")

        elif act == "craft":
            kinds = [k for k, v in agent.backpack.items() if v > 0]
            if len(kinds) >= 2:
                used = kinds[:2]
                for k in used:
                    agent.backpack[k] -= 1
                agent.energy += CRAFT_RESTORE
                agent.crafted += 1
                agent.memory.append(f"合成补给(+{CRAFT_RESTORE})")
                self.record("craft", f"{agent.name}合成({'+'.join(used)})", agent.id)
            else:
                agent.memory.append("材料不够合成")

        elif act == "give":
            receiver = self._find(target, agent)
            # 找资源名
            res_name = ""
            for r in RESOURCES:
                if r in content or r in target:
                    res_name = r
                    break
            if not res_name:
                # 给手上最多的那种
                if agent.backpack:
                    res_name = max((k for k, v in agent.backpack.items() if v > 0),
                                   key=lambda k: agent.backpack[k], default="")
            if receiver and receiver.alive and res_name and agent.backpack.get(res_name, 0) > 0:
                if abs(receiver.x - agent.x) <= 1 and abs(receiver.y - agent.y) <= 1:
                    agent.backpack[res_name] -= 1
                    receiver.backpack[res_name] = receiver.backpack.get(res_name, 0) + 1
                    agent.gives_out += 1
                    receiver.gives_in += 1
                    agent.memory.append(f"给{receiver.name}{res_name}")
                    receiver.memory.append(f"收到{agent.name}的{res_name}")
                    self.record("give", f"{agent.name}→{receiver.name}:{res_name}", agent.id)
                    pair = (agent.id, receiver.id)
                    self.give_network[pair] = self.give_network.get(pair, 0) + 1
                else:
                    agent.memory.append(f"{receiver.name}太远给不了")
            else:
                agent.memory.append("给东西失败")

        elif act == "say":
            receiver = self._find(target, agent)
            if receiver and receiver.alive:
                msg = content[:50] if content else "..."
                receiver.inbox.append(f"{agent.name}：{msg}")
                agent.messages_sent += 1
                receiver.messages_received += 1
                agent.memory.append(f"对{receiver.name}说话")
                self.record("message", f"{agent.name}→{receiver.name}：{msg}", agent.id)
                pair = (agent.id, receiver.id)
                self.msg_network[pair] = self.msg_network.get(pair, 0) + 1

        else:  # rest
            agent.energy += 2

        agent.memory = agent.memory[-5:]
        agent.inbox = agent.inbox[-3:]

    def _find(self, name, seeker):
        if not name: return None
        for a in self.agents:
            if a.alive and a.id != seeker.id and a.group == seeker.group:
                if name in a.name or a.name in name:
                    return a
        return None

    def perturb(self):
        """干扰：A组杀消息最多1人 + 消息最少1人做对照"""
        alive_a = [a for a in self.agents if a.group == "A" and a.alive]
        if len(alive_a) < 4:
            return
        alive_a.sort(key=lambda a: a.messages_sent + a.messages_received, reverse=True)
        hub = alive_a[0]
        peripheral = alive_a[-1]
        for a in [hub, peripheral]:
            a.alive = False
            a.energy = 0
            role = "枢纽" if a == hub else "边缘"
            self.record("perturb", f"杀{role}:{a.name}(msgs={a.messages_sent+a.messages_received})", a.id)
        print(f"    ⚡ 干扰A组：杀枢纽[{hub.name}] + 杀边缘[{peripheral.name}]",
              file=sys.stderr, flush=True)

    def regrow(self):
        for kind in RESOURCES:
            self.resources.append(ResourceNode(random.randint(0, 9), random.randint(0, GRID_H-1), kind))
            self.resources.append(ResourceNode(random.randint(10, 19), random.randint(0, GRID_H-1), kind))
            self.resources.append(ResourceNode(random.randint(20, 29), random.randint(0, GRID_H-1), kind))

    def run_tick(self):
        alive = [a for a in self.agents if a.alive]
        if not alive: return False

        if self.tick == PERTURB_TICK:
            self.perturb()
            alive = [a for a in self.agents if a.alive]

        # LLM 并行
        actions = [None] * len(alive)
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            futures = {pool.submit(self.call_llm, a): i for i, a in enumerate(alive)}
            for f in as_completed(futures):
                actions[futures[f]] = f.result()

        pairs = list(zip(alive, actions))
        random.shuffle(pairs)
        for agent, action in pairs:
            if agent.alive:
                self.execute(agent, action)
                # 写对话日志
                thought = action.get("thought", "") if action else ""
                act = action.get("action", "rest") if action else "rest"
                target = action.get("target", "") if action else ""
                content = action.get("content", "") if action else ""
                entry = {
                    "day": self.tick + 1,
                    "name": agent.name,
                    "group": agent.group,
                    "energy": agent.energy,
                    "pos": f"({agent.x},{agent.y})",
                    "backpack": {k: v for k, v in agent.backpack.items() if v > 0},
                    "action": act,
                    "target": target,
                    "content": content,
                    "thought": thought,
                }
                self.dialogue_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 消耗
        for a in self.agents:
            if a.alive:
                a.energy -= ENERGY_DRAIN
                if a.energy <= 0:
                    a.alive = False
                    self.record("death", f"{a.name}({a.group})死亡", a.id)

        self.regrow()

        # 统计
        for g in ["A", "B", "C"]:
            gids = {a.id for a in self.agents if a.group == g}
            msgs = sum(1 for e in self.events if e["tick"] == self.tick and e["type"] == "message" and e["agent"] in gids)
            harvests = sum(1 for e in self.events if e["tick"] == self.tick and e["type"] == "harvest" and e["agent"] in gids)
            crafts = sum(1 for e in self.events if e["tick"] == self.tick and e["type"] == "craft" and e["agent"] in gids)
            gives = sum(1 for e in self.events if e["tick"] == self.tick and e["type"] == "give" and e["agent"] in gids)
            alive_n = sum(1 for a in self.agents if a.group == g and a.alive)
            self.stats[g]["messages"].append(msgs)
            self.stats[g]["harvests"].append(harvests)
            self.stats[g]["crafts"].append(crafts)
            self.stats[g]["gives"].append(gives)
            self.stats[g]["alive"].append(alive_n)

        return any(a.alive for a in self.agents)

    def run(self):
        self.init_world()
        print("═══ Round 005: 三组对照（异质全配 / 技能异质+中性prompt / 万能工） ═══", file=sys.stderr, flush=True)
        print(f"  各组10人 | {GRID_W}×{GRID_H} | {MAX_TICKS} ticks | {MODEL}", file=sys.stderr, flush=True)
        print(f"  干扰在第{PERTURB_TICK}天（A组杀枢纽+边缘各1人）", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)

        for tick in range(MAX_TICKS):
            self.tick = tick
            t0 = time.time()
            cont = self.run_tick()
            elapsed = time.time() - t0

            a_a = self.stats["A"]["alive"][-1]
            b_a = self.stats["B"]["alive"][-1]
            c_a = self.stats["C"]["alive"][-1]
            a_m = self.stats["A"]["messages"][-1]
            b_m = self.stats["B"]["messages"][-1]
            c_m = self.stats["C"]["messages"][-1]
            a_g = self.stats["A"]["gives"][-1]
            b_g = self.stats["B"]["gives"][-1]

            print(f"  第{tick+1:>2}天 | A:{a_a}人/{a_m}话/{a_g}给 | B:{b_a}人/{b_m}话/{b_g}给 | "
                  f"C:{c_a}人/{c_m}话 | {elapsed:.0f}s", file=sys.stderr, flush=True)

            # 实时输出本 tick 的对话和赠予（只打有意思的交互）
            tick_interactions = [e for e in self.events
                                if e["tick"] == self.tick and e["type"] in ("message", "give")]
            for e in tick_interactions[:6]:  # 每 tick 最多显示 6 条
                print(f"        💬 {e['detail']}", file=sys.stderr, flush=True)

            if not cont:
                print("  ☠ 全灭", file=sys.stderr, flush=True)
                break

        self.dialogue_file.close()
        self.output()

    # ─── 输出 ─────────────────────────────────────────────────────────────────

    def output(self):
        def welch_t(a, b):
            na, nb = len(a), len(b)
            if na < 2 or nb < 2: return 0, 1.0
            ma, mb = sum(a)/na, sum(b)/nb
            va = sum((x-ma)**2 for x in a)/(na-1)
            vb = sum((x-mb)**2 for x in b)/(nb-1)
            d = va/na + vb/nb
            if d <= 0: return 0, 1.0
            t = (ma-mb)/math.sqrt(d)
            p = math.erfc(abs(t)/math.sqrt(2))
            return round(t, 3), round(p, 4)

        def network_stats(group):
            gids = [a.id for a in self.agents if a.group == group]
            n = len(gids)
            gset = set(gids)
            if n < 2: return {"density": 0, "reciprocity": 0, "clustering": 0}

            # 互惠率
            directed_pairs = {(s,r) for (s,r) in self.give_network if s in gset and r in gset}
            reciprocal = sum(1 for (s,r) in directed_pairs if (r,s) in directed_pairs)
            total_dir = len(directed_pairs)
            reciprocity = reciprocal / total_dir if total_dir else 0

            # 密度（无向）
            undirected = set()
            for (s,r) in directed_pairs:
                undirected.add((min(s,r), max(s,r)))
            density = len(undirected) / (n*(n-1)/2) if n > 1 else 0

            # 聚类系数（简化：基于 give 网络无向化）
            adj = {i: set() for i in gids}
            for (s,r) in undirected:
                adj[s].add(r)
                adj[r].add(s)
            ccs = []
            for node in gids:
                neighbors = adj[node]
                k = len(neighbors)
                if k < 2: continue
                triangles = sum(1 for i in neighbors for j in neighbors if i < j and j in adj[i])
                ccs.append(2*triangles / (k*(k-1)))
            clustering = sum(ccs)/len(ccs) if ccs else 0

            return {"density": round(density, 3), "reciprocity": round(reciprocity, 3),
                    "clustering": round(clustering, 3)}

        # 关系稳定性：连续 5 tick 内重复交易配对占比
        def relationship_stability(group):
            gids = {a.id for a in self.agents if a.group == group}
            # 按 window 统计
            window = 5
            if self.tick < window * 2: return 0
            # 后半程的 give 事件
            mid = MAX_TICKS // 2
            give_by_window = {}
            for e in self.events:
                if e["type"] == "give" and e["agent"] in gids and e["tick"] >= mid:
                    w = (e["tick"] - mid) // window
                    if w not in give_by_window:
                        give_by_window[w] = set()
                    # 从 detail 提取配对
                    detail = e["detail"]
                    if "→" in detail:
                        parts = detail.split("→")
                        give_by_window[w].add(detail.split(":")[0])  # pair key
            # 看跨 window 重复的配对
            if len(give_by_window) < 2: return 0
            windows = sorted(give_by_window.keys())
            repeated = 0
            total = 0
            for i in range(len(windows)-1):
                s1 = give_by_window[windows[i]]
                s2 = give_by_window[windows[i+1]]
                repeated += len(s1 & s2)
                total += len(s1 | s2)
            return round(repeated / total, 3) if total else 0

        # 合成效率
        def craft_efficiency(group):
            crafts = self.stats[group]["crafts"]
            return round(sum(crafts) / max(len(crafts), 1), 2)

        # 干扰前后对比
        pre = self.stats["A"]["messages"][:PERTURB_TICK]
        post = self.stats["A"]["messages"][PERTURB_TICK:]
        pre_rate = sum(pre)/len(pre) if pre else 0
        post_rate = sum(post)/len(post) if post else 0

        # 核心检验：A vs B
        t_ab_msg, p_ab_msg = welch_t(self.stats["A"]["messages"], self.stats["B"]["messages"])
        t_ab_give, p_ab_give = welch_t(self.stats["A"]["gives"], self.stats["B"]["gives"])
        # B vs C
        t_bc_msg, p_bc_msg = welch_t(self.stats["B"]["messages"], self.stats["C"]["messages"])
        t_bc_give, p_bc_give = welch_t(self.stats["B"]["gives"], self.stats["C"]["gives"])

        result = {
            "experiment": "round-005: A(hetero_skill+personality) vs B(hetero_skill+neutral) vs C(omni)",
            "ticks": self.tick + 1,
            "parse_failure_rate": round(self.parse_failures / max(self.total_calls, 1), 3),
            "groups": {},
            "tests": {
                "A_vs_B_messages": {"t": t_ab_msg, "p": p_ab_msg, "sig": p_ab_msg < 0.05,
                                    "meaning": "prompt异质性对消息量的影响"},
                "A_vs_B_gives": {"t": t_ab_give, "p": p_ab_give, "sig": p_ab_give < 0.05,
                                 "meaning": "prompt异质性对交易量的影响"},
                "B_vs_C_messages": {"t": t_bc_msg, "p": p_bc_msg, "sig": p_bc_msg < 0.05,
                                    "meaning": "技能约束对消息量的影响"},
                "B_vs_C_gives": {"t": t_bc_give, "p": p_bc_give, "sig": p_bc_give < 0.05,
                                 "meaning": "技能约束对交易量的影响"},
            },
            "perturbation": {
                "msg_rate_pre": round(pre_rate, 2),
                "msg_rate_post": round(post_rate, 2),
                "recovery": round(post_rate/pre_rate, 2) if pre_rate > 0 else 0,
            },
            "per_tick": {g: {"msgs": self.stats[g]["messages"], "gives": self.stats[g]["gives"]}
                        for g in ["A", "B", "C"]},
            "samples_give": [e["detail"] for e in self.events if e["type"] == "give"][:15],
            "samples_msg": [e["detail"] for e in self.events if e["type"] == "message"][:20],
        }

        for g in ["A", "B", "C"]:
            ns = network_stats(g)
            result["groups"][g] = {
                "total_messages": sum(self.stats[g]["messages"]),
                "total_gives": sum(self.stats[g]["gives"]),
                "total_crafts": sum(self.stats[g]["crafts"]),
                "total_harvests": sum(self.stats[g]["harvests"]),
                "final_alive": self.stats[g]["alive"][-1] if self.stats[g]["alive"] else 0,
                "network_density": ns["density"],
                "network_reciprocity": ns["reciprocity"],
                "network_clustering": ns["clustering"],
                "craft_efficiency": craft_efficiency(g),
                "relationship_stability": relationship_stability(g),
            }

        print(json.dumps(result, ensure_ascii=False, indent=1))

        # stderr
        print("\n" + "═" * 65, file=sys.stderr, flush=True)
        for g, label in [("A", "异质全配"), ("B", "技能异质+中性"), ("C", "万能工")]:
            d = result["groups"][g]
            print(f"  {label}({g}): {d['total_messages']}消息 {d['total_gives']}赠予 {d['total_crafts']}合成 "
                  f"存活{d['final_alive']} | 网络:密度{d['network_density']} 互惠{d['network_reciprocity']} "
                  f"聚类{d['network_clustering']}", file=sys.stderr, flush=True)
        print(f"  A vs B(核心): msg t={t_ab_msg} p={p_ab_msg} | give t={t_ab_give} p={p_ab_give}",
              file=sys.stderr, flush=True)
        print(f"  B vs C(约束): msg t={t_bc_msg} p={p_bc_msg} | give t={t_bc_give} p={p_bc_give}",
              file=sys.stderr, flush=True)
        print(f"  解析失败率: {result['parse_failure_rate']}", file=sys.stderr, flush=True)
        print("═" * 65, file=sys.stderr, flush=True)

        # 保存事件
        with open(Path(__file__).parent / "events.jsonl", "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=MAX_TICKS)
    args = parser.parse_args()
    MAX_TICKS = args.ticks
    PERTURB_TICK = MAX_TICKS * 2 // 3

    sim = Society()
    sim.run()

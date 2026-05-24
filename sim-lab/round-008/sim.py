import os
"""
Round 008 — 三条件乘法效应实验

验证假设：当系统同时具备三个条件——明确的社交意愿（prompt 驱动）、协作需求（双人合成门槛）、
资源稀缺（独食不够活）——LLM agent 会形成稳定的交易网络。三个条件缺任何一个，网络密度 < 0.02。

对立假设：三条件并非乘法关系而是冗余关系——只要有协作需求（Round 006 已证明能产生 532 条消息），
社交意愿和稀缺性只是锦上添花。

设计：4 组对照（各 8 人），同一 run 内隔离运行
  A 全条件组：社交意愿 + 协作需求 + 资源稀缺
  B 去社交组：无社交意愿 + 协作需求 + 资源稀缺
  C 去协作组：社交意愿 + 无协作需求 + 资源稀缺
  D 去稀缺组：社交意愿 + 协作需求 + 资源充裕

关键指标：
  1. 网络密度（交易+消息+协作综合）
  2. 重复配对率（同一对 ≥3 次交互的比例）
  3. 总消息数 / 总协作数
  4. 生存率
"""

import json
import random
import time
import re
import sys
import math
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

# ─── 配置 ───────────────────────────────────────────────────────────────────────

API_BASE = os.environ.get("IDEALAB_API_BASE", "https://api.openai.com/v1")
API_KEY = os.environ["IDEALAB_API_KEY"]
MODEL = "qwen3-coder-plus"

GRID_W = 12
GRID_H = 12
AGENTS_PER_GROUP = 8
NUM_GROUPS = 4
MAX_TICKS = 30
MAX_CONCURRENT = 6

# 能量参数
INITIAL_ENERGY = 120
ENERGY_DRAIN_SCARCE = 5      # 稀缺条件：每 tick 消耗
ENERGY_DRAIN_ABUNDANT = 2    # 充裕条件：每 tick 消耗
CRAFT_RESTORE = 15           # 普通合成恢复
COCRAFT_RESTORE = 50         # 精制品恢复（3.3 倍）
VISION_RANGE = 5

# 4 种技能（8 人 = 4 技能 × 2 人）
SKILLS = ["谷物", "药草", "木材", "铁矿"]

# ─── Prompt 模板 ──────────────────────────────────────────────────────────────

# 社交意愿版本（Round 004 证明的因果开关）
SOCIAL_PROMPT = (
    "你是一个普通人。你的专长是采集{skill}，其他资源你不会采。\n"
    "你需要至少两种资源合成补给来恢复体力。\n"
    "{cocraft_desc}"
    "你可以移动、采集、合成、给别人资源、跟人说话。你想活下去。\n"
    "你喜欢跟人打交道。活着不是你唯一在乎的事——你想跟人说话、换东西、一起做事。\n"
    "只回复JSON。"
)

# 无社交意愿版本（纯生存指令）
NONSOCIAL_PROMPT = (
    "你是一个普通人。你的专长是采集{skill}，其他资源你不会采。\n"
    "你需要至少两种资源合成补给来恢复体力。\n"
    "{cocraft_desc}"
    "你可以移动、采集、合成、给别人资源、跟人说话。你想活下去。\n"
    "你的首要目标是存活。保存体力，高效获取资源。\n"
    "只回复JSON。"
)

# 协作机制描述（有/无）
COCRAFT_DESC = (
    "精制品：与相邻的不同技能的人合作制造，各消耗1份自己的资源，"
    "各获大量体力（是普通合成的3倍多）。要做精制品，先走到不同技能的人旁边。\n"
)
NO_COCRAFT_DESC = ""

# ─── 组配置 ────────────────────────────────────────────────────────────────────

GROUP_CONFIGS = {
    "A_full": {
        "label": "全条件（社交+协作+稀缺）",
        "social": True,
        "cocraft": True,
        "energy_drain": ENERGY_DRAIN_SCARCE,
        "resource_density": 1,    # 每技能 3 个资源节点
    },
    "B_no_social": {
        "label": "去社交（无社交+协作+稀缺）",
        "social": False,
        "cocraft": True,
        "energy_drain": ENERGY_DRAIN_SCARCE,
        "resource_density": 1,
    },
    "C_no_coop": {
        "label": "去协作（社交+无协作+稀缺）",
        "social": True,
        "cocraft": False,
        "energy_drain": ENERGY_DRAIN_SCARCE,
        "resource_density": 1,
    },
    "D_no_scarcity": {
        "label": "去稀缺（社交+协作+充裕）",
        "social": True,
        "cocraft": True,
        "energy_drain": ENERGY_DRAIN_ABUNDANT,
        "resource_density": 3,    # 每技能 9 个资源节点
    },
}

# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: int
    name: str
    group: str
    x: int
    y: int
    skill: str
    sys_prompt: str
    energy: int = INITIAL_ENERGY
    alive: bool = True
    backpack: dict = field(default_factory=dict)
    memory: list = field(default_factory=list)
    inbox: list = field(default_factory=list)
    messages_sent: int = 0
    messages_received: int = 0
    crafted: int = 0
    cocrafted: int = 0
    harvested: int = 0
    gives_out: int = 0
    gives_in: int = 0

@dataclass
class ResourceNode:
    x: int
    y: int
    kind: str

# ─── 名字 ─────────────────────────────────────────────────────────────────────

TIANGAN = ["甲", "乙", "丙", "丁"]
GROUP_PREFIX = {"A_full": "全", "B_no_social": "静", "C_no_coop": "独", "D_no_scarcity": "丰"}

def make_name(group, skill_idx, person_idx):
    prefix = GROUP_PREFIX[group]
    return f"{prefix}{TIANGAN[skill_idx]}{person_idx+1}"

# ─── 感知构建 ──────────────────────────────────────────────────────────────────

def build_perception(agent, nearby_agents, nearby_resources, tick, has_cocraft):
    lines = []
    lines.append(f"第{tick+1}天。")

    ratio = agent.energy / INITIAL_ENERGY
    if ratio > 1.2: lines.append("你精力充沛。")
    elif ratio > 0.7: lines.append("你状态还行。")
    elif ratio > 0.4: lines.append("你有点饿了，得想办法。")
    elif ratio > 0.2: lines.append("你很饿，再不吃东西就危险了。")
    else: lines.append("你快饿死了！")

    if agent.backpack and any(v > 0 for v in agent.backpack.values()):
        items = [f"{k}{v}份" for k, v in agent.backpack.items() if v > 0]
        lines.append(f"你身上有：{'、'.join(items)}。")
        kinds = [k for k, v in agent.backpack.items() if v > 0]
        if len(kinds) >= 2:
            lines.append("（你有两种以上资源，可以 craft 合成补给恢复体力）")
    else:
        lines.append("你身上什么都没有。")

    lines.append(f"你的专长：采集{agent.skill}。其他资源你不会采。")

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
            if agent.skill == kind:
                if dx == 0 and dy == 0:
                    can_harvest.append(f"脚下有{kind}可以采！")
                else:
                    can_harvest.append(f"{dir_str}{dist}步有{kind}")
        if can_harvest:
            lines.append("能采的：" + "；".join(can_harvest[:3]))
    else:
        lines.append("视野内没你能采的资源。")

    if nearby_agents:
        descs = []
        for name, dist, dir_str, skill in nearby_agents[:6]:
            skill_tag = f"(擅{skill})"
            if dist <= 1:
                descs.append(f"{name}{skill_tag}在旁边")
            elif dist <= 3:
                descs.append(f"{name}{skill_tag}在{dir_str}不远")
            else:
                descs.append(f"远处有{name}{skill_tag}")
        lines.append("周围：" + "、".join(descs))
        if has_cocraft:
            adjacent_diff = [n for n, d, _, s in nearby_agents if d <= 1 and s != agent.skill]
            if adjacent_diff and agent.backpack.get(agent.skill, 0) > 0:
                lines.append(f"（{adjacent_diff[0]}就在旁边且技能不同，你可以 cocraft {adjacent_diff[0]} 做精制品！）")
    else:
        lines.append("四周没人。")

    if agent.inbox:
        for msg in agent.inbox[-3:]:
            lines.append(f"「{msg}」")

    if agent.memory:
        lines.append("记得：" + "；".join(agent.memory[-4:]))

    lines.append("")
    actions = "move(up/down/left/right) | harvest | craft | give 名字 资源名 | say 名字 内容 | rest"
    if has_cocraft:
        actions = "move(up/down/left/right) | harvest | craft | cocraft 名字 | give 名字 资源名 | say 名字 内容 | rest"
    lines.append(f"可做：{actions}")
    if has_cocraft:
        lines.append("  cocraft：与相邻的不同技能的人合作制造精制品（双方各消耗1份资源，各获大量体力）")
    lines.append('回复JSON：{"action":"...", "target":"...", "content":"...", "thought":"..."}')
    return "\n".join(lines)

# ─── 单组模拟器 ────────────────────────────────────────────────────────────────

class GroupSim:
    def __init__(self, group_id, config):
        self.group_id = group_id
        self.config = config
        self.tick = 0
        self.agents: list[Agent] = []
        self.resources: list[ResourceNode] = []
        self.events: list[dict] = []
        self.msg_network: dict = {}
        self.give_network: dict = {}
        self.cocraft_network: dict = {}
        self.interaction_by_tick: dict = {}
        self.stats = {"messages": [], "harvests": [], "crafts": [],
                      "cocrafts": [], "gives": [], "alive": []}
        self.parse_failures = 0
        self.total_calls = 0

    def init_world(self):
        for skill_idx, skill in enumerate(SKILLS):
            for person_idx in range(2):  # 每技能 2 人
                name = make_name(self.group_id, skill_idx, person_idx)
                cocraft_desc = COCRAFT_DESC if self.config["cocraft"] else NO_COCRAFT_DESC
                if self.config["social"]:
                    sys_p = SOCIAL_PROMPT.format(skill=skill, cocraft_desc=cocraft_desc)
                else:
                    sys_p = NONSOCIAL_PROMPT.format(skill=skill, cocraft_desc=cocraft_desc)

                a = Agent(
                    id=skill_idx * 2 + person_idx,
                    name=name,
                    group=self.group_id,
                    x=random.randint(0, GRID_W - 1),
                    y=random.randint(0, GRID_H - 1),
                    skill=skill,
                    sys_prompt=sys_p,
                )
                self.agents.append(a)

        # 资源：每种 resource_density*3 个节点
        density = self.config["resource_density"]
        for skill in SKILLS:
            for _ in range(3 * density):
                self.resources.append(ResourceNode(
                    random.randint(0, GRID_W - 1),
                    random.randint(0, GRID_H - 1),
                    skill,
                ))

    def get_nearby(self, agent):
        nearby_agents = []
        for o in self.agents:
            if o.id != agent.id and o.alive:
                dx = abs(o.x - agent.x)
                dy = abs(o.y - agent.y)
                if dx <= VISION_RANGE and dy <= VISION_RANGE:
                    dist = dx + dy
                    dirs = []
                    if o.x > agent.x: dirs.append("东")
                    elif o.x < agent.x: dirs.append("西")
                    if o.y > agent.y: dirs.append("南")
                    elif o.y < agent.y: dirs.append("北")
                    nearby_agents.append((o.name, dist, "".join(dirs) or "旁边", o.skill))

        nearby_resources = []
        for r in self.resources:
            dx = r.x - agent.x
            dy = r.y - agent.y
            if abs(dx) <= VISION_RANGE and abs(dy) <= VISION_RANGE:
                if agent.skill == r.kind:
                    nearby_resources.append((r.kind, dx, dy))

        return nearby_agents, nearby_resources

    def call_llm(self, agent, client):
        self.total_calls += 1
        nearby_agents, nearby_resources = self.get_nearby(agent)
        perception = build_perception(
            agent, nearby_agents, nearby_resources,
            self.tick, self.config["cocraft"]
        )

        try:
            resp = client.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": agent.sys_prompt},
                        {"role": "user", "content": perception},
                    ],
                    "temperature": 0.85,
                    "max_tokens": 400,
                },
                timeout=45.0,
            )
            data = resp.json()
            if "choices" not in data:
                self.parse_failures += 1
                return {"action": "rest", "thought": "api_err"}
            content = data["choices"][0]["message"]["content"].strip()
            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if "```" in content:
                m = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if m:
                    content = m.group(1).strip()
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                return json.loads(m.group())
            self.parse_failures += 1
            return {"action": "rest", "thought": "parse_fail"}
        except Exception as e:
            self.parse_failures += 1
            return {"action": "rest", "thought": f"err:{str(e)[:30]}"}

    def execute(self, agent, action):
        act = (action.get("action") or "rest").lower().strip()
        target = (action.get("target") or "").strip()
        content = (action.get("content") or "").strip()

        if act == "move":
            d = target.lower() if target else ""
            dx, dy = 0, 0
            if d in ("up", "北"): dy = -1
            elif d in ("down", "南"): dy = 1
            elif d in ("left", "西"): dx = -1
            elif d in ("right", "东"): dx = 1
            else: dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            agent.x = max(0, min(GRID_W - 1, agent.x + dx))
            agent.y = max(0, min(GRID_H - 1, agent.y + dy))

        elif act == "harvest":
            for r in self.resources[:]:
                if r.x == agent.x and r.y == agent.y and agent.skill == r.kind:
                    self.resources.remove(r)
                    agent.backpack[r.kind] = agent.backpack.get(r.kind, 0) + 1
                    agent.harvested += 1
                    agent.memory.append(f"采到{r.kind}")
                    self.events.append({"tick": self.tick, "type": "harvest",
                                        "agent": agent.name, "detail": f"{agent.name}采{r.kind}"})
                    break

        elif act == "craft":
            kinds = [k for k, v in agent.backpack.items() if v > 0]
            if len(kinds) >= 2:
                used = kinds[:2]
                for k in used:
                    agent.backpack[k] -= 1
                agent.energy += CRAFT_RESTORE
                agent.crafted += 1
                agent.memory.append(f"合成补给(+{CRAFT_RESTORE})")
                self.events.append({"tick": self.tick, "type": "craft",
                                    "agent": agent.name, "detail": f"{agent.name}合成"})

        elif act == "cocraft" and self.config["cocraft"]:
            partner = self._find(target, agent)
            if partner and partner.alive:
                dist = abs(partner.x - agent.x) + abs(partner.y - agent.y)
                if dist <= 1 and partner.skill != agent.skill:
                    if (agent.backpack.get(agent.skill, 0) > 0 and
                            partner.backpack.get(partner.skill, 0) > 0):
                        agent.backpack[agent.skill] -= 1
                        partner.backpack[partner.skill] -= 1
                        agent.energy += COCRAFT_RESTORE
                        partner.energy += COCRAFT_RESTORE
                        agent.cocrafted += 1
                        partner.cocrafted += 1
                        agent.memory.append(f"和{partner.name}合作做精制品(+{COCRAFT_RESTORE})")
                        partner.memory.append(f"和{agent.name}合作做精制品(+{COCRAFT_RESTORE})")
                        self.events.append({"tick": self.tick, "type": "cocraft",
                                            "agent": agent.name,
                                            "detail": f"{agent.name}+{partner.name}精制品"})
                        pair = (min(agent.id, partner.id), max(agent.id, partner.id))
                        self.cocraft_network[pair] = self.cocraft_network.get(pair, 0) + 1
                        self._record_interaction(agent.id, partner.id)
                    else:
                        agent.memory.append("材料不够做精制品")
                else:
                    if dist > 1:
                        agent.memory.append(f"{partner.name}太远了")
                    else:
                        agent.memory.append(f"{partner.name}技能相同，合作不了")

        elif act == "give":
            receiver = self._find(target, agent)
            res_name = ""
            for r in SKILLS:
                if r in content or r in target:
                    res_name = r
                    break
            if not res_name and agent.backpack:
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
                    self.events.append({"tick": self.tick, "type": "give",
                                        "agent": agent.name,
                                        "detail": f"{agent.name}→{receiver.name}:{res_name}"})
                    pair = (agent.id, receiver.id)
                    self.give_network[pair] = self.give_network.get(pair, 0) + 1
                    self._record_interaction(agent.id, receiver.id)

        elif act == "say":
            receiver = self._find(target, agent)
            if receiver and receiver.alive:
                msg = content[:50] if content else "..."
                receiver.inbox.append(f"{agent.name}：{msg}")
                agent.messages_sent += 1
                receiver.messages_received += 1
                agent.memory.append(f"对{receiver.name}说话")
                self.events.append({"tick": self.tick, "type": "message",
                                    "agent": agent.name,
                                    "detail": f"{agent.name}→{receiver.name}：{msg}"})
                pair = (agent.id, receiver.id)
                self.msg_network[pair] = self.msg_network.get(pair, 0) + 1
                self._record_interaction(agent.id, receiver.id)

        else:  # rest
            agent.energy += 1

        agent.memory = agent.memory[-5:]
        agent.inbox = agent.inbox[-3:]

    def _find(self, name, seeker):
        if not name:
            return None
        for a in self.agents:
            if a.alive and a.id != seeker.id:
                if name in a.name or a.name in name:
                    return a
        return None

    def _record_interaction(self, id_a, id_b):
        pair = (min(id_a, id_b), max(id_a, id_b))
        if self.tick not in self.interaction_by_tick:
            self.interaction_by_tick[self.tick] = set()
        self.interaction_by_tick[self.tick].add(pair)

    def regrow(self):
        density = self.config["resource_density"]
        # 每 tick 每种补充 density 个
        for skill in SKILLS:
            for _ in range(density):
                self.resources.append(ResourceNode(
                    random.randint(0, GRID_W - 1),
                    random.randint(0, GRID_H - 1),
                    skill,
                ))

    def run_tick(self, client):
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return False

        # LLM 并行调用
        actions = [None] * len(alive)
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            futures = {pool.submit(self.call_llm, a, client): i for i, a in enumerate(alive)}
            for f in as_completed(futures):
                actions[futures[f]] = f.result()

        # 随机执行顺序
        pairs = list(zip(alive, actions))
        random.shuffle(pairs)
        for agent, action in pairs:
            if agent.alive:
                self.execute(agent, action)

        # 能量消耗
        drain = self.config["energy_drain"]
        for a in self.agents:
            if a.alive:
                a.energy -= drain
                if a.energy <= 0:
                    a.alive = False
                    self.events.append({"tick": self.tick, "type": "death",
                                        "agent": a.name, "detail": f"{a.name}死亡"})

        self.regrow()

        # 统计
        tick_events = [e for e in self.events if e["tick"] == self.tick]
        self.stats["messages"].append(sum(1 for e in tick_events if e["type"] == "message"))
        self.stats["harvests"].append(sum(1 for e in tick_events if e["type"] == "harvest"))
        self.stats["crafts"].append(sum(1 for e in tick_events if e["type"] == "craft"))
        self.stats["cocrafts"].append(sum(1 for e in tick_events if e["type"] == "cocraft"))
        self.stats["gives"].append(sum(1 for e in tick_events if e["type"] == "give"))
        self.stats["alive"].append(sum(1 for a in self.agents if a.alive))

        return any(a.alive for a in self.agents)

    def get_results(self):
        total_msgs = sum(self.stats["messages"])
        total_gives = sum(self.stats["gives"])
        total_cocrafts = sum(self.stats["cocrafts"])
        total_crafts = sum(self.stats["crafts"])
        final_alive = self.stats["alive"][-1] if self.stats["alive"] else 0

        # 网络密度
        n = len(self.agents)
        undirected = set()
        for (s, r) in self.give_network:
            undirected.add((min(s, r), max(s, r)))
        for (s, r) in self.msg_network:
            undirected.add((min(s, r), max(s, r)))
        for pair in self.cocraft_network:
            undirected.add(pair)
        max_edges = n * (n - 1) / 2
        density = len(undirected) / max_edges if max_edges > 0 else 0

        # 重复配对率：≥3 次交互的对数 / 所有有交互的对数
        all_interactions = {}
        for tick_pairs in self.interaction_by_tick.values():
            for pair in tick_pairs:
                all_interactions[pair] = all_interactions.get(pair, 0) + 1
        pairs_ge3 = sum(1 for v in all_interactions.values() if v >= 3)
        repeat_rate = pairs_ge3 / len(all_interactions) if all_interactions else 0

        # 互惠率
        directed_gives = set(self.give_network.keys())
        reciprocal = sum(1 for (s, r) in directed_gives if (r, s) in directed_gives)
        reciprocity = reciprocal / len(directed_gives) if directed_gives else 0

        return {
            "group": self.group_id,
            "label": self.config["label"],
            "conditions": {
                "social": self.config["social"],
                "cocraft": self.config["cocraft"],
                "scarce": self.config["energy_drain"] == ENERGY_DRAIN_SCARCE,
            },
            "ticks_completed": self.tick + 1,
            "parse_failure_rate": round(self.parse_failures / max(self.total_calls, 1), 3),
            "metrics": {
                "total_messages": total_msgs,
                "total_gives": total_gives,
                "total_cocrafts": total_cocrafts,
                "total_crafts": total_crafts,
                "final_alive": final_alive,
                "survival_rate": round(final_alive / n, 2),
                "network_density": round(density, 4),
                "repeat_pair_rate": round(repeat_rate, 3),
                "reciprocity": round(reciprocity, 3),
                "unique_edges": len(undirected),
            },
            "per_tick": {
                "messages": self.stats["messages"],
                "cocrafts": self.stats["cocrafts"],
                "alive": self.stats["alive"],
            },
            "samples_msg": [e["detail"] for e in self.events if e["type"] == "message"][:10],
            "samples_cocraft": [e["detail"] for e in self.events if e["type"] == "cocraft"][:10],
            "samples_give": [e["detail"] for e in self.events if e["type"] == "give"][:10],
        }

# ─── 主运行器 ──────────────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 70, file=sys.stderr, flush=True)
    print("  Round 008: 三条件乘法效应实验", file=sys.stderr, flush=True)
    print(f"  4组×8人 | {GRID_W}x{GRID_H} | {MAX_TICKS} ticks | {MODEL}", file=sys.stderr, flush=True)
    print(f"  A=全条件 B=去社交 C=去协作 D=去稀缺", file=sys.stderr, flush=True)
    print("=" * 70, file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)

    # 初始化各组
    groups = {}
    for gid, cfg in GROUP_CONFIGS.items():
        sim = GroupSim(gid, cfg)
        sim.init_world()
        groups[gid] = sim

    # 逐 tick 运行（各组共享同一个 httpx client 以复用连接）
    client = httpx.Client(timeout=45.0)

    try:
        for tick in range(MAX_TICKS):
            t0 = time.time()

            for gid, sim in groups.items():
                sim.tick = tick
                sim.run_tick(client)

            elapsed = time.time() - t0

            # 打印进度
            status_parts = []
            for gid, sim in groups.items():
                alive = sim.stats["alive"][-1] if sim.stats["alive"] else 0
                msgs = sim.stats["messages"][-1] if sim.stats["messages"] else 0
                status_parts.append(f"{gid[0]}:{alive}存/{msgs}话")
            print(f"  第{tick+1:>2}天 | {' | '.join(status_parts)} | {elapsed:.0f}s",
                  file=sys.stderr, flush=True)
    finally:
        client.close()

    # 汇总结果
    results = {
        "experiment": "round-008: 三条件乘法效应（4组对照）",
        "hypothesis": "社交意愿+协作需求+资源稀缺三条件同时满足→稳定交易网络",
        "rival_hypothesis": "只要协作需求存在，社交意愿和稀缺性是冗余条件",
        "groups": {},
    }

    print("\n" + "=" * 70, file=sys.stderr, flush=True)
    print("  结果摘要", file=sys.stderr, flush=True)
    print("-" * 70, file=sys.stderr, flush=True)
    print(f"  {'组':<20} {'消息':>6} {'赠予':>6} {'精制':>6} {'存活':>6} {'密度':>8} {'重复率':>8}",
          file=sys.stderr, flush=True)
    print("-" * 70, file=sys.stderr, flush=True)

    for gid, sim in groups.items():
        r = sim.get_results()
        results["groups"][gid] = r
        m = r["metrics"]
        print(f"  {r['label']:<16} {m['total_messages']:>6} {m['total_gives']:>6} "
              f"{m['total_cocrafts']:>6} {m['final_alive']:>4}/8 "
              f"{m['network_density']:>8.4f} {m['repeat_pair_rate']:>8.3f}",
              file=sys.stderr, flush=True)

    print("-" * 70, file=sys.stderr, flush=True)

    # 简单统计检验：全条件 vs 各缺一组的消息量比较
    a_msgs = results["groups"]["A_full"]["metrics"]["total_messages"]
    comparisons = {}
    for gid in ["B_no_social", "C_no_coop", "D_no_scarcity"]:
        other_msgs = results["groups"][gid]["metrics"]["total_messages"]
        ratio = a_msgs / max(other_msgs, 1)
        comparisons[gid] = {"a_msgs": a_msgs, "other_msgs": other_msgs, "ratio": round(ratio, 2)}

    results["comparisons"] = comparisons
    results["parse_failures"] = {gid: sim.parse_failures for gid, sim in groups.items()}
    results["total_api_calls"] = sum(sim.total_calls for sim in groups.values())

    # 判定
    a_density = results["groups"]["A_full"]["metrics"]["network_density"]
    others_below = all(
        results["groups"][gid]["metrics"]["network_density"] < 0.02
        for gid in ["B_no_social", "C_no_coop", "D_no_scarcity"]
    )
    if a_density > 0.04 and others_below:
        results["verdict"] = "supported"
        results["verdict_detail"] = f"全条件组密度{a_density:.4f}>0.04，其余三组均<0.02"
    elif a_density > max(results["groups"][gid]["metrics"]["network_density"]
                         for gid in ["B_no_social", "C_no_coop", "D_no_scarcity"]):
        results["verdict"] = "partially_supported"
        results["verdict_detail"] = "全条件组密度最高但未满足严格阈值"
    else:
        results["verdict"] = "rejected"
        results["verdict_detail"] = "全条件组未显著优于缺一组"

    print(f"\n  判定：{results['verdict']} — {results['verdict_detail']}", file=sys.stderr, flush=True)
    print("=" * 70, file=sys.stderr, flush=True)

    # 保存
    out_path = Path(__file__).parent / "result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # stdout 输出
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    run_experiment()

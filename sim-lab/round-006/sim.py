import os
"""
Round 006 — 双人协作涌现实验

验证假设：当资源获取存在硬性的双人协作门槛（精制品需两个不同技能的人相邻才能合成），
         LLM agent 能否自发形成稳定配对关系，而非随机游走？

对立假设：协作门槛过高时 agent 无法通过纯信息交换找到并锁定合作伙伴，
         导致精制品产出趋近于零，最终全灭。

背景（基于 round-005 发现）：
  B组（中性 prompt + 技能异质）表现优于 A组（丰富个性 + 技能异质），
  说明简洁的行为导向 prompt 更有效。本轮全部采用中性 prompt。

设计：
  - 30 人单组（6 种技能 × 5 人）
  - 统一中性 prompt，只描述技能和协作机制，不给个性
  - 精制品（cocraft）：两个不同技能的人相邻时合作制造，各消耗 1 份自己技能资源，
    各获 3 倍能量（COCRAFT_RESTORE = 60 vs CRAFT_RESTORE = 20）
  - 干扰：第 40 天杀消息最多的 2 人（枢纽节点）

关键指标：
  1. 稳定配对数（连续 10 tick 重复互动的对）
  2. 交易网络互惠率
  3. 合成效率（精制品数量 / tick）
  4. 干扰后网络恢复速度（消息率回到干扰前水平的 tick 数）
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
MODEL = "qwen3-coder-plus"

GRID_W = 20
GRID_H = 20
NUM_AGENTS = 30
MAX_TICKS = 60
PERTURB_TICK = 40
MAX_CONCURRENT = 6

INITIAL_ENERGY = 150
ENERGY_DRAIN = 4
CRAFT_RESTORE = 20          # 普通合成恢复
COCRAFT_RESTORE = 60        # 精制品恢复（3 倍）
VISION_RANGE = 5

# 6 种基础资源，6 种技能各对应一种
RESOURCES = ["谷物", "药草", "石料", "木材", "兽皮", "铁矿"]

# 精制品名称（纯展示用，任意两种不同技能都能协作）
REFINED_NAMES = {
    ("谷物", "药草"): "药膳", ("谷物", "石料"): "石磨面", ("谷物", "木材"): "粮仓",
    ("谷物", "兽皮"): "皮囊", ("谷物", "铁矿"): "铁锅",
    ("药草", "石料"): "研磨药", ("药草", "木材"): "药柜", ("药草", "兽皮"): "药囊",
    ("药草", "铁矿"): "针灸针",
    ("石料", "木材"): "石屋", ("石料", "兽皮"): "皮甲", ("石料", "铁矿"): "铁器",
    ("木材", "兽皮"): "弓箭", ("木材", "铁矿"): "铁斧",
    ("兽皮", "铁矿"): "铁甲",
}

def get_refined_name(skill_a, skill_b):
    key = tuple(sorted([skill_a, skill_b]))
    return REFINED_NAMES.get(key, "精制品")

# ─── 人口设定 ──────────────────────────────────────────────────────────────────

SKILL_ASSIGN = []  # 6 skills × 5 people = 30
for skill in RESOURCES:
    SKILL_ASSIGN.extend([skill] * 5)

# 名字：天干 + 数字，便于 agent 互认
TIANGAN = ["甲", "乙", "丙", "丁", "戊", "己"]
NAMES = []
for i, skill in enumerate(RESOURCES):
    for j in range(1, 6):
        NAMES.append(f"{TIANGAN[i]}{j}")

# 统一中性 prompt（只描述机制，不给个性）
NEUTRAL_SYS_TEMPLATE = (
    "你是一个普通人。你的专长是采集{skill}，其他资源你不会采。\n"
    "你需要至少两种资源合成补给来恢复体力。\n"
    "另外，有些高级补给（精制品）需要两个不同技能的人站在相邻位置一起制造，"
    "效果是普通合成的三倍。要做精制品，你需要手上有自己技能的资源，"
    "然后对相邻的不同技能的人发起合作。\n"
    "你可以移动、采集、合成、给别人资源、跟人说话。你想活下去。\n"
    "只回复JSON。"
)

# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: int
    name: str
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
    crafted: int = 0          # 普通合成次数
    cocrafted: int = 0        # 协作合成次数
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

    # 体力
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
            lines.append("（你有两种以上资源，可以 craft 合成普通补给恢复体力）")
    else:
        lines.append("你身上什么都没有。")

    # 技能
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
            if agent.skill == kind:
                if dx == 0 and dy == 0:
                    can_harvest.append(f"脚下有{kind}可以采！")
                else:
                    can_harvest.append(f"{dir_str}{dist}步有{kind}")
        if can_harvest:
            lines.append("能采的：" + "；".join(can_harvest[:3]))
    else:
        lines.append("视野内没你能采的资源。")

    # 周围的人（含技能信息，帮助 agent 判断谁能协作）
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
        # 协作提示：如果旁边有不同技能的人
        adjacent_diff = [n for n, d, _, s in nearby_agents if d <= 1 and s != agent.skill]
        if adjacent_diff and agent.backpack.get(agent.skill, 0) > 0:
            lines.append(f"（{adjacent_diff[0]}就在旁边且技能不同，你可以 cocraft {adjacent_diff[0]} 做精制品！）")
    else:
        lines.append("四周没人。")

    # 收件箱
    if agent.inbox:
        for msg in agent.inbox[-3:]:
            lines.append(f"「{msg}」")

    # 记忆
    if agent.memory:
        lines.append("记得：" + "；".join(agent.memory[-4:]))

    # 动作
    lines.append("")
    lines.append("可做：move(up/down/left/right) | harvest | craft | cocraft 名字 | give 名字 资源名 | say 名字 内容 | rest")
    lines.append("  cocraft：与相邻的不同技能的人合作制造精制品（双方各消耗1份自己的资源，各获大量体力）")
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
        self.stats = {"messages": [], "harvests": [], "crafts": [],
                      "cocrafts": [], "gives": [], "alive": []}
        self.give_network: dict = {}     # (giver_id, receiver_id) → count
        self.msg_network: dict = {}      # (sender_id, receiver_id) → count
        self.cocraft_network: dict = {}  # (id_a, id_b) → count (无向，小 id 在前)
        self.interaction_by_tick: dict = {}  # tick → set of (min_id, max_id) pairs
        self.parse_failures = 0
        self.total_calls = 0
        self.dialogue_file = open(Path(__file__).parent / "dialogue.jsonl", "w", encoding="utf-8")

    def init_world(self):
        for i in range(NUM_AGENTS):
            skill = SKILL_ASSIGN[i]
            name = NAMES[i]
            sys_p = NEUTRAL_SYS_TEMPLATE.format(skill=skill)
            a = Agent(
                id=self.next_id, name=name,
                x=random.randint(0, GRID_W - 1),
                y=random.randint(0, GRID_H - 1),
                skill=skill, sys_prompt=sys_p,
            )
            self.next_id += 1
            self.agents.append(a)

        # 资源：每种 8 个节点，随机分布
        for kind in RESOURCES:
            for _ in range(8):
                self.resources.append(ResourceNode(
                    random.randint(0, GRID_W - 1),
                    random.randint(0, GRID_H - 1),
                    kind,
                ))

        self.record("init", f"30人(6技能x5) | {GRID_W}x{GRID_H} | {MAX_TICKS}ticks | cocraft机制")

    def record(self, etype, detail, aid=-1):
        self.events.append({"tick": self.tick, "type": etype, "agent": aid, "detail": detail})

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
            # 去 think 标签
            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            # 提取 JSON
            if "```" in content:
                m = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if m:
                    content = m.group(1).strip()
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                return json.loads(m.group())
            self.parse_failures += 1
            return {"action": "rest", "thought": f"parse_fail:{content[:40]}"}
        except Exception as e:
            self.parse_failures += 1
            return {"action": "rest", "thought": f"err:{str(e)[:40]}"}

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
            nx = max(0, min(GRID_W - 1, agent.x + dx))
            ny = max(0, min(GRID_H - 1, agent.y + dy))
            agent.x, agent.y = nx, ny

        elif act == "harvest":
            done = False
            for r in self.resources[:]:
                if r.x == agent.x and r.y == agent.y and agent.skill == r.kind:
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

        elif act == "cocraft":
            partner = self._find(target, agent)
            if partner and partner.alive:
                dist = abs(partner.x - agent.x) + abs(partner.y - agent.y)
                if dist <= 1 and partner.skill != agent.skill:
                    # 双方都需要有自己技能的资源
                    if agent.backpack.get(agent.skill, 0) > 0 and partner.backpack.get(partner.skill, 0) > 0:
                        agent.backpack[agent.skill] -= 1
                        partner.backpack[partner.skill] -= 1
                        agent.energy += COCRAFT_RESTORE
                        partner.energy += COCRAFT_RESTORE
                        agent.cocrafted += 1
                        partner.cocrafted += 1
                        product = get_refined_name(agent.skill, partner.skill)
                        agent.memory.append(f"和{partner.name}合作做了{product}(+{COCRAFT_RESTORE})")
                        partner.memory.append(f"和{agent.name}合作做了{product}(+{COCRAFT_RESTORE})")
                        self.record("cocraft", f"{agent.name}+{partner.name}→{product}", agent.id)
                        # 记录协作网络
                        pair = (min(agent.id, partner.id), max(agent.id, partner.id))
                        self.cocraft_network[pair] = self.cocraft_network.get(pair, 0) + 1
                        # 互动记录
                        self._record_interaction(agent.id, partner.id)
                    else:
                        if agent.backpack.get(agent.skill, 0) <= 0:
                            agent.memory.append(f"自己没{agent.skill}，做不了精制品")
                        else:
                            agent.memory.append(f"{partner.name}手上没{partner.skill}")
                elif dist > 1:
                    agent.memory.append(f"{partner.name}太远了，要相邻才能合作")
                elif partner.skill == agent.skill:
                    agent.memory.append(f"{partner.name}技能跟我一样，合作不了")
            else:
                agent.memory.append("找不到合作对象")

        elif act == "give":
            receiver = self._find(target, agent)
            res_name = ""
            for r in RESOURCES:
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
                    self.record("give", f"{agent.name}→{receiver.name}:{res_name}", agent.id)
                    pair = (agent.id, receiver.id)
                    self.give_network[pair] = self.give_network.get(pair, 0) + 1
                    self._record_interaction(agent.id, receiver.id)
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
                self._record_interaction(agent.id, receiver.id)

        else:  # rest
            agent.energy += 2

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

    def perturb(self):
        """干扰：杀消息最多的 2 人（枢纽节点）"""
        alive = [a for a in self.agents if a.alive]
        if len(alive) < 5:
            return
        alive.sort(key=lambda a: a.messages_sent + a.messages_received, reverse=True)
        killed = alive[:2]
        for a in killed:
            a.alive = False
            a.energy = 0
            self.record("perturb",
                        f"杀枢纽:{a.name}(skill={a.skill},msgs={a.messages_sent+a.messages_received})", a.id)
        names = [a.name for a in killed]
        print(f"    干扰：杀枢纽 {names}", file=sys.stderr, flush=True)

    def regrow(self):
        """每 tick 补充资源"""
        for kind in RESOURCES:
            self.resources.append(ResourceNode(
                random.randint(0, GRID_W - 1),
                random.randint(0, GRID_H - 1),
                kind,
            ))

    def run_tick(self):
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return False

        if self.tick == PERTURB_TICK:
            self.perturb()
            alive = [a for a in self.agents if a.alive]

        # LLM 并行调用
        actions = [None] * len(alive)
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            futures = {pool.submit(self.call_llm, a): i for i, a in enumerate(alive)}
            for f in as_completed(futures):
                actions[futures[f]] = f.result()

        # 随机执行顺序
        pairs = list(zip(alive, actions))
        random.shuffle(pairs)
        for agent, action in pairs:
            if agent.alive:
                self.execute(agent, action)
                # 写对话日志
                thought = action.get("thought", "") if action else ""
                act_str = action.get("action", "rest") if action else "rest"
                tgt = action.get("target", "") if action else ""
                cont = action.get("content", "") if action else ""
                entry = {
                    "day": self.tick + 1,
                    "name": agent.name,
                    "skill": agent.skill,
                    "energy": agent.energy,
                    "pos": f"({agent.x},{agent.y})",
                    "backpack": {k: v for k, v in agent.backpack.items() if v > 0},
                    "action": act_str,
                    "target": tgt,
                    "content": cont,
                    "thought": thought,
                }
                self.dialogue_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 能量消耗
        for a in self.agents:
            if a.alive:
                a.energy -= ENERGY_DRAIN
                if a.energy <= 0:
                    a.alive = False
                    self.record("death", f"{a.name}(skill={a.skill})死亡", a.id)

        self.regrow()

        # 统计本 tick
        tick_events = [e for e in self.events if e["tick"] == self.tick]
        msgs = sum(1 for e in tick_events if e["type"] == "message")
        harvests = sum(1 for e in tick_events if e["type"] == "harvest")
        crafts = sum(1 for e in tick_events if e["type"] == "craft")
        cocrafts = sum(1 for e in tick_events if e["type"] == "cocraft")
        gives = sum(1 for e in tick_events if e["type"] == "give")
        alive_n = sum(1 for a in self.agents if a.alive)
        self.stats["messages"].append(msgs)
        self.stats["harvests"].append(harvests)
        self.stats["crafts"].append(crafts)
        self.stats["cocrafts"].append(cocrafts)
        self.stats["gives"].append(gives)
        self.stats["alive"].append(alive_n)

        return any(a.alive for a in self.agents)

    def run(self):
        self.init_world()
        print("=" * 70, file=sys.stderr, flush=True)
        print("  Round 006: 双人协作涌现实验", file=sys.stderr, flush=True)
        print(f"  30人(6技能x5) | {GRID_W}x{GRID_H} | {MAX_TICKS} ticks | {MODEL}",
              file=sys.stderr, flush=True)
        print(f"  精制品=3倍能量 | 干扰第{PERTURB_TICK}天杀2枢纽",
              file=sys.stderr, flush=True)
        print("=" * 70, file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)

        for tick in range(MAX_TICKS):
            self.tick = tick
            t0 = time.time()
            cont = self.run_tick()
            elapsed = time.time() - t0

            alive_n = self.stats["alive"][-1]
            msgs = self.stats["messages"][-1]
            gives = self.stats["gives"][-1]
            cocrafts = self.stats["cocrafts"][-1]
            crafts = self.stats["crafts"][-1]

            print(f"  第{tick+1:>2}天 | 存活{alive_n} | {msgs}话 {gives}给 {crafts}合成 {cocrafts}精制 | {elapsed:.0f}s",
                  file=sys.stderr, flush=True)

            # 打印本 tick 精彩互动
            tick_interactions = [e for e in self.events
                                 if e["tick"] == self.tick and e["type"] in ("cocraft", "give", "message")]
            for e in tick_interactions[:5]:
                print(f"        {e['type']:>8} | {e['detail']}", file=sys.stderr, flush=True)

            if not cont:
                print("  全灭", file=sys.stderr, flush=True)
                break

        self.dialogue_file.close()
        self.output()

    # ─── 输出与分析 ────────────────────────────────────────────────────────────

    def output(self):
        # 1. 稳定配对数：连续 window tick 每 tick 都有互动的配对
        stable_pairs = self._calc_stable_pairs(window=10)

        # 2. 互惠率
        reciprocity = self._calc_reciprocity()

        # 3. 精制品效率
        total_cocrafts = sum(self.stats["cocrafts"])
        cocraft_efficiency = round(total_cocrafts / max(self.tick + 1, 1), 2)

        # 4. 干扰恢复速度
        recovery_ticks = self._calc_recovery_speed()

        # 5. 网络统计
        net_stats = self._network_stats()

        # 6. 协作配对排行
        top_cocraft_pairs = sorted(self.cocraft_network.items(), key=lambda x: -x[1])[:10]
        top_pairs_readable = []
        for (id_a, id_b), count in top_cocraft_pairs:
            a = next(x for x in self.agents if x.id == id_a)
            b = next(x for x in self.agents if x.id == id_b)
            top_pairs_readable.append(f"{a.name}({a.skill})+{b.name}({b.skill})={count}次")

        result = {
            "experiment": "round-006: 双人协作涌现（30人, 6技能x5, cocraft机制）",
            "hypothesis": "协作门槛驱动稳定配对自发涌现",
            "ticks_completed": self.tick + 1,
            "parse_failure_rate": round(self.parse_failures / max(self.total_calls, 1), 3),
            "summary": {
                "total_messages": sum(self.stats["messages"]),
                "total_gives": sum(self.stats["gives"]),
                "total_crafts": sum(self.stats["crafts"]),
                "total_cocrafts": total_cocrafts,
                "total_harvests": sum(self.stats["harvests"]),
                "final_alive": self.stats["alive"][-1] if self.stats["alive"] else 0,
            },
            "key_metrics": {
                "stable_pairs_10tick": stable_pairs,
                "reciprocity": reciprocity,
                "cocraft_per_tick": cocraft_efficiency,
                "recovery_ticks_after_perturb": recovery_ticks,
            },
            "network": net_stats,
            "top_cocraft_pairs": top_pairs_readable,
            "perturbation": {
                "tick": PERTURB_TICK,
                "msg_rate_pre": round(sum(self.stats["messages"][:PERTURB_TICK]) / max(PERTURB_TICK, 1), 2),
                "msg_rate_post": round(sum(self.stats["messages"][PERTURB_TICK:]) / max(self.tick + 1 - PERTURB_TICK, 1), 2),
                "cocraft_rate_pre": round(sum(self.stats["cocrafts"][:PERTURB_TICK]) / max(PERTURB_TICK, 1), 2),
                "cocraft_rate_post": round(sum(self.stats["cocrafts"][PERTURB_TICK:]) / max(self.tick + 1 - PERTURB_TICK, 1), 2),
            },
            "per_tick": {
                "messages": self.stats["messages"],
                "gives": self.stats["gives"],
                "crafts": self.stats["crafts"],
                "cocrafts": self.stats["cocrafts"],
                "alive": self.stats["alive"],
            },
            "samples_cocraft": [e["detail"] for e in self.events if e["type"] == "cocraft"][:20],
            "samples_give": [e["detail"] for e in self.events if e["type"] == "give"][:15],
            "samples_msg": [e["detail"] for e in self.events if e["type"] == "message"][:15],
        }

        # 写 result.json
        out_path = Path(__file__).parent / "result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

        # stdout 也打一份
        print(json.dumps(result, ensure_ascii=False, indent=1))

        # stderr 摘要
        print("\n" + "=" * 70, file=sys.stderr, flush=True)
        print("  结果摘要", file=sys.stderr, flush=True)
        print(f"  完成 {result['ticks_completed']} ticks | 存活 {result['summary']['final_alive']}/30",
              file=sys.stderr, flush=True)
        print(f"  消息 {result['summary']['total_messages']} | 赠予 {result['summary']['total_gives']} | "
              f"普通合成 {result['summary']['total_crafts']} | 精制品 {result['summary']['total_cocrafts']}",
              file=sys.stderr, flush=True)
        print(f"  稳定配对(10tick) {stable_pairs} | 互惠率 {reciprocity} | "
              f"精制品/tick {cocraft_efficiency}", file=sys.stderr, flush=True)
        print(f"  干扰恢复 {recovery_ticks} ticks | 解析失败率 {result['parse_failure_rate']}",
              file=sys.stderr, flush=True)
        if top_pairs_readable:
            print(f"  头部协作对: {', '.join(top_pairs_readable[:5])}", file=sys.stderr, flush=True)
        print("=" * 70, file=sys.stderr, flush=True)

        # 保存事件
        with open(Path(__file__).parent / "events.jsonl", "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def _calc_stable_pairs(self, window=10):
        """连续 window tick 每 tick 都有互动的配对数"""
        if self.tick + 1 < window:
            return 0
        # 滑动窗口，取最大值
        max_stable = 0
        for start in range(self.tick + 1 - window + 1):
            # 找在 [start, start+window) 每一天都出现的配对
            sets = []
            for t in range(start, start + window):
                sets.append(self.interaction_by_tick.get(t, set()))
            if not sets or not sets[0]:
                continue
            common = sets[0].copy()
            for s in sets[1:]:
                common = common & s
                if not common:
                    break
            max_stable = max(max_stable, len(common))
        return max_stable

    def _calc_reciprocity(self):
        """互惠率：双向赠予配对数 / 总有向赠予配对数"""
        directed = set(self.give_network.keys())
        if not directed:
            return 0.0
        reciprocal = sum(1 for (s, r) in directed if (r, s) in directed)
        return round(reciprocal / len(directed), 3)

    def _calc_recovery_speed(self):
        """干扰后消息率恢复到干扰前均值 80% 所需 tick 数"""
        if PERTURB_TICK >= self.tick:
            return -1
        pre_msgs = self.stats["messages"][:PERTURB_TICK]
        if not pre_msgs:
            return -1
        pre_avg = sum(pre_msgs) / len(pre_msgs)
        if pre_avg <= 0:
            return 0
        # 找干扰后第一个 5-tick 滑动均值 >= 80% pre_avg 的位置
        post_msgs = self.stats["messages"][PERTURB_TICK:]
        window = 5
        for i in range(len(post_msgs) - window + 1):
            avg = sum(post_msgs[i:i + window]) / window
            if avg >= pre_avg * 0.8:
                return i + window
        return -1  # 未恢复

    def _network_stats(self):
        """计算交易+消息+协作综合网络的密度、聚类系数"""
        ids = [a.id for a in self.agents]
        n = len(ids)
        id_set = set(ids)

        # 合并 give + msg + cocraft 为无向边
        undirected = set()
        for (s, r) in self.give_network:
            if s in id_set and r in id_set:
                undirected.add((min(s, r), max(s, r)))
        for (s, r) in self.msg_network:
            if s in id_set and r in id_set:
                undirected.add((min(s, r), max(s, r)))
        for pair in self.cocraft_network:
            undirected.add(pair)

        density = len(undirected) / (n * (n - 1) / 2) if n > 1 else 0

        # 聚类系数
        adj = {i: set() for i in ids}
        for (s, r) in undirected:
            adj[s].add(r)
            adj[r].add(s)
        ccs = []
        for node in ids:
            neighbors = adj[node]
            k = len(neighbors)
            if k < 2:
                continue
            triangles = sum(1 for i in neighbors for j in neighbors if i < j and j in adj[i])
            ccs.append(2 * triangles / (k * (k - 1)))
        clustering = sum(ccs) / len(ccs) if ccs else 0

        return {
            "density": round(density, 3),
            "clustering": round(clustering, 3),
            "unique_edges": len(undirected),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=MAX_TICKS)
    args = parser.parse_args()
    MAX_TICKS = args.ticks
    PERTURB_TICK = min(PERTURB_TICK, MAX_TICKS - 5)

    sim = Society()
    sim.run()

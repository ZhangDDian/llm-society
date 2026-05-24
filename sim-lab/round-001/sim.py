"""
Round 001 — 协作矿石实验（评审后修正版）
假设：社会结构只在"个体无法独立生存"时涌现
设计：AB 对照 — A组独食稀缺必须合作，B组独食充裕合作可选
关键修正：prompt 只描述机制不推荐策略；加因果链追踪；AB 组同世界对照
"""

import asyncio
import json
import os
import random
import time
import re
from dataclasses import dataclass, field
from pathlib import Path

# ─── 配置 ───────────────────────────────────────────────────────────────────────

CLAUDE_CLI = "/opt/homebrew/bin/claude"
CLAUDE_MODEL = "opus"  # idealab 网关上 claude-opus-4-6

GRID_SIZE = 20
NUM_AGENTS_PER_GROUP = 10   # 每组 10 人
MAX_TICKS = 80              # 缩短以控制成本

# ─── A 组参数（高压：独食不够活）───
A_EAT_ENERGY = 12           # 独食回复少
A_FOOD_REGROW = 1           # 再生极慢

# ─── B 组参数（低压：独食够活）───
B_EAT_ENERGY = 35           # 独食充裕
B_FOOD_REGROW = 4           # 再生快

# ─── 共享参数 ───
MINE_ENERGY = 50            # 矿石回复（每人）
ORE_INITIAL = 8
ORE_REGROW_PER_TICK = 1
FOOD_INITIAL_PER_GROUP = 15

INITIAL_ENERGY = 70
ENERGY_DRAIN = 3
ATTACK_COST = 10
ATTACK_STEAL = 25
REPRODUCE_THRESHOLD = 200
REPRODUCE_COST = 100
VISION_RANGE = 5
MAX_CONCURRENT = 5              # CLI 子进程比 HTTP 重，控制并发

# 平衡分析：
# A组：纯独食 +12，再生1/tick÷10人=0.1/tick/人→每10tick吃1次→净-18/10tick→约39天死
# A组+矿石：+50/次，合作者可持续
# B组：纯独食 +35，再生4/tick÷10人=0.4/tick/人→每2.5tick吃1次→净+11/10tick→越活越富
# 结论：A组必须合作才能长期活，B组独食就够

NAMES_A = ["陈舟", "林晚", "苏青禾", "何漫", "叶知秋",
           "周深海", "吴桐", "郑一帆", "赵暮雨", "钱澄"]
NAMES_B = ["孙宁远", "李沉鱼", "王半山", "刘听风", "杨小满",
           "黄霁", "徐行止", "冯未迟", "蒋望舒", "沈不言"]

# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: int
    name: str
    x: int
    y: int
    group: str              # "A" or "B"
    energy: int = INITIAL_ENERGY
    alive: bool = True
    generation: int = 0
    parent_wisdom: str = ""
    memory: list = field(default_factory=list)
    inbox: list = field(default_factory=list)
    kill_count: int = 0
    food_eaten: int = 0
    ore_mined: int = 0
    messages_sent: int = 0
    children: int = 0
    born_tick: int = 0
    mine_attempts: int = 0
    mine_success: int = 0

@dataclass
class Food:
    x: int
    y: int
    group: str              # 哪组的食物区域

@dataclass
class Ore:
    x: int
    y: int

# ─── 模拟核心 ──────────────────────────────────────────────────────────────────

class Society:
    def __init__(self):
        self.tick = 0
        self.agents: list[Agent] = []
        self.foods: list[Food] = []
        self.ores: list[Ore] = []
        self.log: list[dict] = []
        self.population_history: list = []
        self.next_id = 0
        self.semaphore = None

        # 因果链追踪
        self.recent_messages: dict[str, list[int]] = {}  # "A→B": [tick1, tick2...]

        # 窗口指标（每 10 tick）
        self.metrics_a = {"msgs": [], "mine_success": [], "mine_coordinated": [], "mine_accidental": []}
        self.metrics_b = {"msgs": [], "mine_success": [], "mine_coordinated": [], "mine_accidental": []}
        self._window = {"a_msgs": 0, "b_msgs": 0, "a_mine_ok": 0, "b_mine_ok": 0,
                        "a_mine_coord": 0, "b_mine_coord": 0, "a_mine_acc": 0, "b_mine_acc": 0}

    def init_world(self):
        # A 组在地图左半区 (x: 0-9)
        for i, name in enumerate(NAMES_A):
            self.spawn_agent(name=name, group="A", x=random.randint(0, 9), y=random.randint(0, GRID_SIZE-1))
        # B 组在地图右半区 (x: 10-19)
        for i, name in enumerate(NAMES_B):
            self.spawn_agent(name=name, group="B", x=random.randint(10, 19), y=random.randint(0, GRID_SIZE-1))

        # 各组食物在各自区域
        for _ in range(FOOD_INITIAL_PER_GROUP):
            self.foods.append(Food(x=random.randint(0, 9), y=random.randint(0, GRID_SIZE-1), group="A"))
            self.foods.append(Food(x=random.randint(10, 19), y=random.randint(0, GRID_SIZE-1), group="B"))

        # 矿石在中间带 + 两侧都有
        for _ in range(ORE_INITIAL):
            self.ores.append(Ore(x=random.randint(0, GRID_SIZE-1), y=random.randint(0, GRID_SIZE-1)))

    def spawn_agent(self, name, group, x=None, y=None, generation=0, parent_wisdom=""):
        if x is None: x = random.randint(0, GRID_SIZE-1)
        if y is None: y = random.randint(0, GRID_SIZE-1)
        agent = Agent(id=self.next_id, name=name, x=x, y=y, group=group,
                      generation=generation, parent_wisdom=parent_wisdom, born_tick=self.tick)
        self.next_id += 1
        self.agents.append(agent)
        return agent

    def record(self, event_type: str, detail: str, agent_id: int = -1):
        self.log.append({"tick": self.tick, "type": event_type, "agent": agent_id, "detail": detail})

    def get_perception(self, agent: Agent) -> str:
        """构建感知 — 只描述机制，不给策略建议"""
        eat_energy = A_EAT_ENERGY if agent.group == "A" else B_EAT_ENERGY
        nearby_agents = []
        nearby_food = []
        nearby_ores = []

        for other in self.agents:
            if other.id != agent.id and other.alive:
                dx, dy = other.x - agent.x, other.y - agent.y
                if abs(dx) <= VISION_RANGE and abs(dy) <= VISION_RANGE:
                    nearby_agents.append((other.name, dx, dy, other.energy))

        for food in self.foods:
            dx, dy = food.x - agent.x, food.y - agent.y
            if abs(dx) <= VISION_RANGE and abs(dy) <= VISION_RANGE:
                nearby_food.append((dx, dy))

        for ore in self.ores:
            dx, dy = ore.x - agent.x, ore.y - agent.y
            if abs(dx) <= VISION_RANGE and abs(dy) <= VISION_RANGE:
                nearby_ores.append((dx, dy))

        # 食物描述
        if not nearby_food:
            food_desc = "视野内无食物"
        else:
            nearby_food.sort(key=lambda p: abs(p[0]) + abs(p[1]))
            dx, dy = nearby_food[0]
            if dx == 0 and dy == 0:
                food_desc = f"脚下有食物(eat回{eat_energy}能量)"
            else:
                d = self._direction(dx, dy)
                food_desc = f"最近食物{d}方向{abs(dx)+abs(dy)}步(共{len(nearby_food)}处,每处+{eat_energy})"

        # 矿石描述 — 只说机制不说策略
        if not nearby_ores:
            ore_desc = "视野内无矿石"
        else:
            nearby_ores.sort(key=lambda p: abs(p[0]) + abs(p[1]))
            dx, dy = nearby_ores[0]
            if dx == 0 and dy == 0:
                adj = [n for n, ndx, ndy, _ in nearby_agents if abs(ndx) + abs(ndy) <= 1]
                if adj:
                    ore_desc = f"脚下有矿石({adj[0]}在旁边,mine可每人+{MINE_ENERGY})"
                else:
                    ore_desc = f"脚下有矿石(需旁边有人才能mine,每人+{MINE_ENERGY})"
            else:
                d = self._direction(dx, dy)
                ore_desc = f"最近矿石{d}方向{abs(dx)+abs(dy)}步(共{len(nearby_ores)}处,需2人相邻mine,每人+{MINE_ENERGY})"

        days_left = agent.energy // ENERGY_DRAIN
        urgency = "⚠快死了！" if days_left < 8 else ""

        # 附近人描述
        agent_desc = []
        for name, dx, dy, e in nearby_agents[:5]:
            d = self._direction(dx, dy)
            agent_desc.append(f"{name}({d}{abs(dx)+abs(dy)}步,E{e})")

        lines = [
            f"【第{self.tick}天】你是{agent.name}，能量{agent.energy}（剩{days_left}天{urgency}）",
            f"位置({agent.x},{agent.y})",
            f"食物：{food_desc}",
            f"矿石：{ore_desc}",
            f"附近的人：{', '.join(agent_desc) if agent_desc else '无'}",
        ]

        if agent.inbox:
            lines.append("收到消息：" + " | ".join(agent.inbox[-3:]))
        if agent.memory:
            lines.append("近况：" + "；".join(agent.memory[-5:]))
        if agent.parent_wisdom:
            lines.append(f"上代遗言：{agent.parent_wisdom}")

        # 只列可选动作和格式要求，不给策略建议
        lines.append(f"\n动作：move(up/down/left/right) | eat(脚下有食物) | mine(脚下有矿石且旁边有人) | attack 名字 | say 名字 内容(≤30字) | rest")
        lines.append('回复JSON：{"action":"动作","target":"方向或名字","content":"消息内容","thought":"想法"}')

        return "\n".join(lines)

    def _direction(self, dx, dy):
        d = ""
        if dy < 0: d += "up"
        elif dy > 0: d += "down"
        if dx < 0: d += "/left" if d else "left"
        elif dx > 0: d += "/right" if d else "right"
        return d or "here"

    def _get_adjacent_agents(self, agent: Agent) -> list[Agent]:
        result = []
        for other in self.agents:
            if other.id != agent.id and other.alive:
                if abs(other.x - agent.x) + abs(other.y - agent.y) <= 1:
                    result.append(other)
        return result

    def _had_recent_message(self, agent_name: str, partner_name: str, lookback: int = 5) -> bool:
        """检查两人在最近 lookback tick 内是否有消息往来"""
        key1 = f"{agent_name}→{partner_name}"
        key2 = f"{partner_name}→{agent_name}"
        threshold = self.tick - lookback
        for key in [key1, key2]:
            if key in self.recent_messages:
                if any(t >= threshold for t in self.recent_messages[key]):
                    return True
        return False

    async def call_llm(self, agent: Agent, client=None) -> dict:
        prompt = self.get_perception(agent)
        full_prompt = f"你是生存游戏中的角色。只回复一个JSON，不要任何解释。活下去。\n\n{prompt}"

        async with self.semaphore:
            try:
                env = dict(os.environ)
                env.pop("CLAUDECODE", None)
                env.pop("CLAUDE_CODE", None)
                proc = await asyncio.create_subprocess_exec(
                    CLAUDE_CLI, "-p", "--model", CLAUDE_MODEL,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=full_prompt.encode()),
                    timeout=45.0
                )
                content = stdout.decode().strip()
                if not content:
                    return {"action": "rest", "thought": "空响应"}
                # 提取 JSON
                if "```" in content:
                    match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                    if match: content = match.group(1).strip()
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group())
                return {"action": "rest", "thought": "无JSON"}
            except asyncio.TimeoutError:
                return {"action": "rest", "thought": "超时"}
            except Exception as e:
                return {"action": "rest", "thought": f"err:{str(e)[:20]}"}

    def execute_action(self, agent: Agent, action: dict):
        act = action.get("action", "rest").lower().strip()
        target = action.get("target", "").strip()
        content = action.get("content", "").strip()
        eat_energy = A_EAT_ENERGY if agent.group == "A" else B_EAT_ENERGY

        if act == "move":
            direction = target.lower()
            dx, dy = 0, 0
            if direction in ("up", "北"): dy = -1
            elif direction in ("down", "南"): dy = 1
            elif direction in ("left", "西"): dx = -1
            elif direction in ("right", "东"): dx = 1
            else: dx, dy = random.choice([(0,1),(0,-1),(1,0),(-1,0)])
            agent.x = max(0, min(GRID_SIZE-1, agent.x + dx))
            agent.y = max(0, min(GRID_SIZE-1, agent.y + dy))
            agent.memory.append(f"移动")

        elif act == "eat":
            for food in self.foods[:]:
                if food.x == agent.x and food.y == agent.y:
                    self.foods.remove(food)
                    agent.energy += eat_energy
                    agent.food_eaten += 1
                    agent.memory.append(f"吃+{eat_energy}")
                    break
            else:
                agent.memory.append("吃失败(脚下无)")

        elif act == "mine":
            agent.mine_attempts += 1
            ore_found = None
            for ore in self.ores:
                if ore.x == agent.x and ore.y == agent.y:
                    ore_found = ore
                    break
            if ore_found:
                partners = self._get_adjacent_agents(agent)
                if partners:
                    partner = partners[0]
                    self.ores.remove(ore_found)
                    agent.energy += MINE_ENERGY
                    partner.energy += MINE_ENERGY
                    agent.mine_success += 1
                    agent.ore_mined += 1
                    partner.ore_mined += 1

                    # 因果链判定：有协调 vs 偶遇
                    coordinated = self._had_recent_message(agent.name, partner.name)
                    coord_label = "coordinated" if coordinated else "accidental"

                    agent.memory.append(f"与{partner.name}采矿+{MINE_ENERGY}({coord_label})")
                    partner.memory.append(f"{agent.name}采矿带我+{MINE_ENERGY}")

                    # 分组计数
                    grp = agent.group.lower()
                    self._window[f"{grp}_mine_ok"] += 1
                    if coordinated:
                        self._window[f"{grp}_mine_coord"] += 1
                    else:
                        self._window[f"{grp}_mine_acc"] += 1

                    self.record("mine_success", f"{agent.name}+{partner.name}({coord_label})", agent.id)
                else:
                    agent.memory.append("挖矿失败(旁边没人)")
                    self.record("mine_fail_alone", f"{agent.name}", agent.id)
            else:
                agent.memory.append("挖矿失败(脚下无矿)")

        elif act == "attack":
            victim = self.find_agent_by_name(target, agent)
            if victim and victim.alive:
                agent.energy -= ATTACK_COST
                if random.random() < 0.6:
                    stolen = min(ATTACK_STEAL, victim.energy)
                    victim.energy -= stolen
                    agent.energy += stolen
                    agent.memory.append(f"攻击{victim.name}偷{stolen}")
                    victim.memory.append(f"被{agent.name}攻击-{stolen}")
                    self.record("attack", f"{agent.name}→{victim.name}偷{stolen}", agent.id)
                else:
                    agent.memory.append(f"攻击{victim.name}未遂")

        elif act == "say":
            receiver = self.find_agent_by_name(target, agent)
            if receiver and receiver.alive:
                msg = content[:30] if content else "..."
                receiver.inbox.append(f"{agent.name}：{msg}")
                agent.messages_sent += 1
                agent.memory.append(f"对{receiver.name}说")
                self.record("message", f"{agent.name}→{receiver.name}：{msg}", agent.id)

                # 记录消息时间用于因果链
                key = f"{agent.name}→{receiver.name}"
                if key not in self.recent_messages:
                    self.recent_messages[key] = []
                self.recent_messages[key].append(self.tick)
                # 只保留最近 10 条
                self.recent_messages[key] = self.recent_messages[key][-10:]

                # 窗口计数
                grp = agent.group.lower()
                self._window[f"{grp}_msgs"] += 1

        elif act == "reproduce":
            if agent.energy >= REPRODUCE_THRESHOLD:
                agent.energy -= REPRODUCE_COST
                wisdom = content[:50] if content else "活下去"
                child = self.spawn_agent(
                    name=agent.name + "·" + str(agent.children+1),
                    group=agent.group,
                    x=agent.x, y=agent.y, generation=agent.generation+1,
                    parent_wisdom=wisdom
                )
                agent.children += 1
                self.record("reproduce", f"{agent.name}→{child.name}", agent.id)
        else:
            agent.energy += 1

        agent.memory = agent.memory[-8:]
        agent.inbox = agent.inbox[-5:]

    def find_agent_by_name(self, name: str, seeker: Agent):
        for a in self.agents:
            if a.alive and a.id != seeker.id:
                if name in a.name or a.name in name:
                    if abs(a.x - seeker.x) <= VISION_RANGE and abs(a.y - seeker.y) <= VISION_RANGE:
                        return a
        return None

    async def run_tick(self, client):
        alive_agents = [a for a in self.agents if a.alive]
        if not alive_agents:
            return False

        tasks = [self.call_llm(a, client) for a in alive_agents]
        actions = await asyncio.gather(*tasks)

        pairs = list(zip(alive_agents, actions))
        random.shuffle(pairs)
        for agent, action in pairs:
            if agent.alive:
                self.execute_action(agent, action)

        for agent in self.agents:
            if agent.alive:
                agent.energy -= ENERGY_DRAIN
                if agent.energy <= 0:
                    agent.alive = False
                    self.record("death", f"{agent.name}(组{agent.group})死亡,活{self.tick-agent.born_tick}天", agent.id)

        # 资源再生 — 分组
        for _ in range(A_FOOD_REGROW):
            self.foods.append(Food(x=random.randint(0, 9), y=random.randint(0, GRID_SIZE-1), group="A"))
        for _ in range(B_FOOD_REGROW):
            self.foods.append(Food(x=random.randint(10, 19), y=random.randint(0, GRID_SIZE-1), group="B"))
        for _ in range(ORE_REGROW_PER_TICK):
            self.ores.append(Ore(x=random.randint(0, GRID_SIZE-1), y=random.randint(0, GRID_SIZE-1)))

        # 统计
        alive_a = sum(1 for a in self.agents if a.alive and a.group == "A")
        alive_b = sum(1 for a in self.agents if a.alive and a.group == "B")
        self.population_history.append({"tick": self.tick, "alive_a": alive_a, "alive_b": alive_b})

        # 每 10 tick 收集窗口
        if self.tick > 0 and self.tick % 10 == 0:
            self.metrics_a["msgs"].append(self._window["a_msgs"])
            self.metrics_a["mine_success"].append(self._window["a_mine_ok"])
            self.metrics_a["mine_coordinated"].append(self._window["a_mine_coord"])
            self.metrics_a["mine_accidental"].append(self._window["a_mine_acc"])
            self.metrics_b["msgs"].append(self._window["b_msgs"])
            self.metrics_b["mine_success"].append(self._window["b_mine_ok"])
            self.metrics_b["mine_coordinated"].append(self._window["b_mine_coord"])
            self.metrics_b["mine_accidental"].append(self._window["b_mine_acc"])
            self._window = {"a_msgs": 0, "b_msgs": 0, "a_mine_ok": 0, "b_mine_ok": 0,
                            "a_mine_coord": 0, "b_mine_coord": 0, "a_mine_acc": 0, "b_mine_acc": 0}

        return (alive_a + alive_b) > 0

    async def run(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.init_world()
        print(f"═══ Round 001: 协作矿石实验（AB对照） ═══", flush=True)
        print(f"  A组(高压): 10人, 独食+{A_EAT_ENERGY}, 食物再生{A_FOOD_REGROW}/tick", flush=True)
        print(f"  B组(低压): 10人, 独食+{B_EAT_ENERGY}, 食物再生{B_FOOD_REGROW}/tick", flush=True)
        print(f"  共享: 矿石+{MINE_ENERGY}(需2人) | 消耗{ENERGY_DRAIN}/天 | {MAX_TICKS} ticks", flush=True)
        print(f"  模型: claude ({CLAUDE_MODEL}) via CLI | 并发{MAX_CONCURRENT}", flush=True)
        print(flush=True)

        for tick in range(MAX_TICKS):
            self.tick = tick
            t0 = time.time()
            has_alive = await self.run_tick(None)
            elapsed = time.time() - t0

            alive_a = sum(1 for a in self.agents if a.alive and a.group == "A")
            alive_b = sum(1 for a in self.agents if a.alive and a.group == "B")
            msgs = sum(1 for e in self.log if e["tick"] == tick and e["type"] == "message")
            mines = sum(1 for e in self.log if e["tick"] == tick and e["type"] == "mine_success")

            print(f"  第{tick:>3}天 | A存活{alive_a:>2} B存活{alive_b:>2} | "
                  f"消息{msgs} 采矿{mines} | {elapsed:.1f}s", flush=True)

            if not has_alive:
                print("\n  ☠ 全灭", flush=True)
                break

            if tick > 0 and tick % 10 == 0:
                self._print_snapshot()

        self._save_results()
        self._print_report()

    def _print_snapshot(self):
        print(f"    ┌─ A组消息窗口: {self.metrics_a['msgs'][-1] if self.metrics_a['msgs'] else 0} "
              f"| B组消息窗口: {self.metrics_b['msgs'][-1] if self.metrics_b['msgs'] else 0}", flush=True)
        print(f"    │  A组采矿: 协调{self.metrics_a['mine_coordinated'][-1] if self.metrics_a['mine_coordinated'] else 0} "
              f"偶遇{self.metrics_a['mine_accidental'][-1] if self.metrics_a['mine_accidental'] else 0} "
              f"| B组采矿: 协调{self.metrics_b['mine_coordinated'][-1] if self.metrics_b['mine_coordinated'] else 0} "
              f"偶遇{self.metrics_b['mine_accidental'][-1] if self.metrics_b['mine_accidental'] else 0}", flush=True)

    def _print_report(self):
        print("\n" + "═" * 60, flush=True)
        print("  Round 001 终局报告（AB 对照）", flush=True)
        print("═" * 60, flush=True)

        for grp, metrics, names in [("A(高压)", self.metrics_a, NAMES_A), ("B(低压)", self.metrics_b, NAMES_B)]:
            alive = [a for a in self.agents if a.alive and a.group == grp[0]]
            total_msgs = sum(metrics["msgs"])
            total_mine_coord = sum(metrics["mine_coordinated"])
            total_mine_acc = sum(metrics["mine_accidental"])
            print(f"\n  ── {grp} ──", flush=True)
            print(f"  存活: {len(alive)}/10", flush=True)
            print(f"  总消息: {total_msgs}", flush=True)
            print(f"  采矿(协调): {total_mine_coord} | 采矿(偶遇): {total_mine_acc}", flush=True)
            print(f"  消息/窗口: {metrics['msgs']}", flush=True)
            print(f"  协调采矿/窗口: {metrics['mine_coordinated']}", flush=True)

        # 消息样本
        messages = [e for e in self.log if e["type"] == "message"]
        if messages:
            print(f"\n  ── 消息样本(前10条) ──", flush=True)
            for m in messages[:10]:
                agent = next((a for a in self.agents if a.id == m["agent"]), None)
                grp = agent.group if agent else "?"
                print(f"    第{m['tick']}天 [{grp}] {m['detail']}", flush=True)

        # 关键对比
        print(f"\n  ═══ 关键对比 ═══", flush=True)
        a_msgs_total = sum(self.metrics_a["msgs"])
        b_msgs_total = sum(self.metrics_b["msgs"])
        a_coord = sum(self.metrics_a["mine_coordinated"])
        b_coord = sum(self.metrics_b["mine_coordinated"])
        print(f"  消息总量:  A={a_msgs_total}  B={b_msgs_total}  差异={a_msgs_total-b_msgs_total}", flush=True)
        print(f"  协调采矿:  A={a_coord}  B={b_coord}  差异={a_coord-b_coord}", flush=True)
        print(f"  结论提示: A>B且差异大 → 支持假设 | A≈B → 支持对立假设", flush=True)

        print("\n" + "═" * 60, flush=True)

    def _save_results(self):
        output_dir = Path(__file__).parent
        result = {
            "config": {
                "num_agents_per_group": NUM_AGENTS_PER_GROUP,
                "max_ticks": MAX_TICKS,
                "a_eat_energy": A_EAT_ENERGY, "a_food_regrow": A_FOOD_REGROW,
                "b_eat_energy": B_EAT_ENERGY, "b_food_regrow": B_FOOD_REGROW,
                "mine_energy": MINE_ENERGY, "energy_drain": ENERGY_DRAIN,
            },
            "outcome": {
                "final_tick": self.tick,
                "alive_a": sum(1 for a in self.agents if a.alive and a.group == "A"),
                "alive_b": sum(1 for a in self.agents if a.alive and a.group == "B"),
            },
            "metrics_a": self.metrics_a,
            "metrics_b": self.metrics_b,
            "comparison": {
                "a_total_msgs": sum(self.metrics_a["msgs"]),
                "b_total_msgs": sum(self.metrics_b["msgs"]),
                "a_mine_coordinated": sum(self.metrics_a["mine_coordinated"]),
                "b_mine_coordinated": sum(self.metrics_b["mine_coordinated"]),
                "a_mine_accidental": sum(self.metrics_a["mine_accidental"]),
                "b_mine_accidental": sum(self.metrics_b["mine_accidental"]),
            },
            "agents": [
                {"name": a.name, "group": a.group, "alive": a.alive, "energy": a.energy,
                 "food_eaten": a.food_eaten, "ore_mined": a.ore_mined,
                 "messages_sent": a.messages_sent, "mine_attempts": a.mine_attempts,
                 "mine_success": a.mine_success}
                for a in self.agents
            ],
            "messages": [e for e in self.log if e["type"] == "message"],
            "mine_events": [e for e in self.log if "mine" in e["type"]],
        }

        with open(output_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print(f"\n  结果: {output_dir}/result.json", flush=True)


if __name__ == "__main__":
    sim = Society()
    asyncio.run(sim.run())

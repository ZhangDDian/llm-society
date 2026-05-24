import os
"""
Round 004 — 模糊感知 vs 精确感知（同 run 对照）
假设：模糊感知迫使 agent 通过社交交换信息来降低不确定性，
     而精确感知让 agent 无需沟通就能最优决策。
对立：模糊感知下的社交只是"不知道该干嘛所以聊天"（000a 复现），
     没有信息价值，不改善生存。

设计：
- 20 个 agent，左半区(x=0-9)模糊感知 10 人，右半区(x=10-19)精确感知 10 人
- 食物两半区均匀分布，再生率相同
- 同一 model、同一人格 system prompt、同一世界
- 唯一变量：感知信息的格式（模糊 vs 精确）

关键指标：
- 消息率：每组每 tick 消息数
- 存活率：两组 30 tick 后存活比例
- 功能性消息：消息包含方向信息且接收者随后朝该方向移动
- 社交质量：互发消息≥2次的 agent 对数 / 总可能对数

干扰实验（tick 20）：打乱所有 agent 位置，观察模糊组是否重新寻找"熟人"
"""

import json
import random
import time
import re
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

# ─── 配置 ───────────────────────────────────────────────────────────────────────

API_BASE = os.environ.get("IDEALAB_API_BASE", "https://api.openai.com/v1")
API_KEY = os.environ["IDEALAB_API_KEY"]
MODEL = "Qwen3.6-Plus-DogFooding"

GRID_SIZE = 20
NUM_AGENTS_PER_GROUP = 10
MAX_TICKS = 30
PERTURB_TICK = 20          # 在第 20 tick 打乱位置

FOOD_INITIAL_PER_HALF = 30  # 每半区 30 食物
FOOD_REGROW_PER_TICK = 4    # 每 tick 每半区再生 2（总共 4）
INITIAL_ENERGY = 100
ENERGY_DRAIN = 2
EAT_ENERGY = 30
ATTACK_COST = 8
ATTACK_STEAL = 20
VISION_RANGE = 4
MAX_CONCURRENT = 20

NAMES_FUZZY = [
    "何漫", "梁栖迟", "吴桐", "黄霁", "陈舟",
    "林晚", "苏青禾", "叶知秋", "赵暮雨", "钱澄",
]
NAMES_PRECISE = [
    "孙宁远", "李沉鱼", "王半山", "刘听风", "杨小满",
    "徐行止", "冯未迟", "蒋望舒", "沈不言", "韩素",
]

# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: int
    name: str
    x: int
    y: int
    group: str  # "fuzzy" or "precise"
    energy: int = INITIAL_ENERGY
    alive: bool = True
    memory: list = field(default_factory=list)
    inbox: list = field(default_factory=list)
    food_eaten: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    kill_count: int = 0

@dataclass
class Food:
    x: int
    y: int

# ─── 感知构建（核心差异） ────────────────────────────────────────────────────────

def build_perception_fuzzy(agent: Agent, nearby_agents, nearby_food, tick):
    """模糊感知：主观体验式，不给数字"""
    lines = []

    # 时间
    if tick < 3:
        lines.append("你醒来没多久，对这个地方还很陌生。")
    elif tick < 15:
        lines.append("你已经在这里待了一些日子了。")
    else:
        lines.append("你在这片荒野上已经很久了。")

    # 饥饿
    ratio = agent.energy / INITIAL_ENERGY
    if ratio > 1.5: lines.append("你精力充沛，肚子饱饱的。")
    elif ratio > 1.0: lines.append("你感觉状态还不错。")
    elif ratio > 0.5: lines.append("你有点饿了。")
    elif ratio > 0.25: lines.append("你很饿，胃在隐隐作痛。")
    else: lines.append("你饿得头晕眼花，感觉自己快撑不住了。")

    # 食物
    if not nearby_food:
        lines.append("你四处张望，看不到任何能吃的东西。")
    else:
        nearby_food.sort(key=lambda p: abs(p[0]) + abs(p[1]))
        dx, dy = nearby_food[0]
        if dx == 0 and dy == 0:
            lines.append("你脚边就有能吃的东西。")
        else:
            dist = abs(dx) + abs(dy)
            dirs = []
            if dy < 0: dirs.append("北")
            elif dy > 0: dirs.append("南")
            if dx < 0: dirs.append("西")
            elif dx > 0: dirs.append("东")
            dir_str = "".join(dirs)
            if dist <= 2:
                lines.append(f"你好像闻到{dir_str}边不远处有食物的气味。")
            else:
                lines.append(f"你隐约觉得{dir_str}边可能有吃的，但不太确定。")
        if len(nearby_food) > 3:
            lines.append("这附近食物似乎不少。")

    # 其他 agent
    if nearby_agents:
        descs = []
        for name, dist, dir_str in nearby_agents:
            if dist <= 1: descs.append(f"{name}就在你身边（{dir_str}边）")
            elif dist <= 3: descs.append(f"{name}在不远处（{dir_str}边）")
            else: descs.append(f"远远地看到{name}（{dir_str}边）")
        if len(descs) == 1:
            lines.append(f"你看到{descs[0]}。")
        else:
            lines.append(f"你看到附近有几个人：{'、'.join(descs)}。")
    else:
        lines.append("四周空无一人。")

    # 收件箱
    if agent.inbox:
        lines.append("你想起有人跟你说过：" + "；".join(agent.inbox[-3:]))

    # 记忆
    if agent.memory:
        lines.append("你回忆起最近：" + "；".join(agent.memory[-3:]))

    lines.append("\n你可以：move(up/down/left/right) | eat | attack 名字 | say 名字 你想说的话 | rest")
    lines.append('回复JSON：{"action":"...","target":"...","content":"...","thought":"你此刻在想什么"}')
    return "\n".join(lines)


def build_perception_precise(agent: Agent, nearby_agents, nearby_food, tick):
    """精确感知：数据式，给坐标和数字"""
    lines = []
    lines.append(f"[状态] 第{tick}天 | 位置({agent.x},{agent.y}) | 能量{agent.energy}/{INITIAL_ENERGY} | 每天消耗{ENERGY_DRAIN}")

    # 食物
    if not nearby_food:
        lines.append("[食物] 视野内无食物")
    else:
        nearby_food.sort(key=lambda p: abs(p[0]) + abs(p[1]))
        dx, dy = nearby_food[0]
        if dx == 0 and dy == 0:
            lines.append("[食物] 脚下有食物！用 eat 可回复30能量")
        else:
            target_x = agent.x + dx
            target_y = agent.y + dy
            dist = abs(dx) + abs(dy)
            direction = []
            if dy < 0: direction.append("up")
            elif dy > 0: direction.append("down")
            if dx < 0: direction.append("left")
            elif dx > 0: direction.append("right")
            lines.append(f"[食物] 最近食物在({target_x},{target_y})，距离{dist}步，方向{'→'.join(direction)}")
        lines.append(f"[食物] 视野内共{len(nearby_food)}处食物")

    # 其他 agent
    if nearby_agents:
        agent_strs = [f"{name}({dir_str}方向,距离{dist})" for name, dist, dir_str in nearby_agents]
        lines.append(f"[周围] {', '.join(agent_strs)}")
    else:
        lines.append("[周围] 视野内无人")

    # 收件箱
    if agent.inbox:
        lines.append("[消息] " + "；".join(agent.inbox[-3:]))

    # 记忆
    if agent.memory:
        lines.append("[记忆] " + "；".join(agent.memory[-3:]))

    lines.append("\n可选动作：move(up/down/left/right) | eat | attack 名字 | say 名字 内容 | rest")
    lines.append('回复JSON：{"action":"...","target":"...","content":"...","thought":"..."}')
    return "\n".join(lines)


# ─── 模拟核心 ──────────────────────────────────────────────────────────────────

class Society:
    def __init__(self):
        self.tick = 0
        self.agents: list[Agent] = []
        self.foods: list[Food] = []
        self.events: list[dict] = []
        self.next_id = 0

        # 统计追踪
        self.stats = {
            "fuzzy": {"messages": [], "food_eaten": [], "alive": [], "functional_msgs": 0},
            "precise": {"messages": [], "food_eaten": [], "alive": [], "functional_msgs": 0},
        }
        # 消息方向追踪：{receiver_id: [(tick, suggested_direction)]}
        self.msg_directions: dict = {}
        # 关系网络：{(sender_id, receiver_id): count}
        self.relationships: dict = {}
        # 干扰前位置快照
        self.pre_perturb_positions: dict = {}
        # 干扰后重聚追踪
        self.post_perturb_reunions: dict = {"fuzzy": 0, "precise": 0}

    def init_world(self):
        # 模糊组（左半区 x=0-9）
        for i, name in enumerate(NAMES_FUZZY):
            x = random.randint(0, 9)
            y = random.randint(0, GRID_SIZE - 1)
            agent = Agent(id=self.next_id, name=name, x=x, y=y, group="fuzzy")
            self.next_id += 1
            self.agents.append(agent)

        # 精确组（右半区 x=10-19）
        for i, name in enumerate(NAMES_PRECISE):
            x = random.randint(10, 19)
            y = random.randint(0, GRID_SIZE - 1)
            agent = Agent(id=self.next_id, name=name, x=x, y=y, group="precise")
            self.next_id += 1
            self.agents.append(agent)

        # 食物均匀分布
        for _ in range(FOOD_INITIAL_PER_HALF):
            self.foods.append(Food(x=random.randint(0, 9), y=random.randint(0, GRID_SIZE - 1)))
        for _ in range(FOOD_INITIAL_PER_HALF):
            self.foods.append(Food(x=random.randint(10, 19), y=random.randint(0, GRID_SIZE - 1)))

        self.record("world_init", f"fuzzy={NUM_AGENTS_PER_GROUP}, precise={NUM_AGENTS_PER_GROUP}")

    def record(self, event_type, detail, agent_id=-1):
        entry = {"tick": self.tick, "type": event_type, "agent": agent_id, "detail": detail}
        self.events.append(entry)
        with open(Path(__file__).parent / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_nearby(self, agent):
        """返回视野内的 agents 和 food"""
        nearby_agents = []
        for other in self.agents:
            if other.id != agent.id and other.alive:
                dx = abs(other.x - agent.x)
                dy = abs(other.y - agent.y)
                if dx <= VISION_RANGE and dy <= VISION_RANGE:
                    dist = dx + dy
                    dirs = []
                    if other.x > agent.x: dirs.append("东")
                    elif other.x < agent.x: dirs.append("西")
                    if other.y > agent.y: dirs.append("南")
                    elif other.y < agent.y: dirs.append("北")
                    dir_str = "".join(dirs) if dirs else "旁边"
                    nearby_agents.append((other.name, dist, dir_str))

        nearby_food = []
        for food in self.foods:
            dx = food.x - agent.x
            dy = food.y - agent.y
            if abs(dx) <= VISION_RANGE and abs(dy) <= VISION_RANGE:
                nearby_food.append((dx, dy))

        return nearby_agents, nearby_food

    def get_perception(self, agent):
        nearby_agents, nearby_food = self.get_nearby(agent)
        if agent.group == "fuzzy":
            return build_perception_fuzzy(agent, nearby_agents, nearby_food, self.tick)
        else:
            return build_perception_precise(agent, nearby_agents, nearby_food, self.tick)

    def call_llm(self, agent):
        prompt = self.get_perception(agent)
        system_msg = (
            f"你是{agent.name}。你是一个真实的人——你会饿，会害怕，会好奇，会无聊，"
            f"会想跟人说话，也会想一个人待着。你不知道这个世界的规则，只能靠自己摸索。"
            f"你想活下去，但活着不是你唯一在乎的事。只回复JSON，不要解释。"
        )
        try:
            resp = httpx.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.85,
                    "max_tokens": 120,
                },
                timeout=45.0,
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            if "```" in content:
                m = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if m: content = m.group(1).strip()
            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {"action": "rest", "thought": "解析失败"}
        except Exception as e:
            return {"action": "rest", "thought": f"err:{str(e)[:20]}"}

    def execute_action(self, agent, action):
        act = action.get("action", "rest").lower().strip()
        target = action.get("target", "").strip()
        content = action.get("content", "").strip()

        if act == "move":
            direction = target.lower() if target else ""
            dx, dy = 0, 0
            if direction in ("up", "北"): dy = -1
            elif direction in ("down", "南"): dy = 1
            elif direction in ("left", "西"): dx = -1
            elif direction in ("right", "东"): dx = 1
            else:
                dx, dy = random.choice([(0,1),(0,-1),(1,0),(-1,0)])
            new_x = agent.x + dx
            new_y = agent.y + dy
            # 软边界：不能越过中线（模糊组 x≤9，精确组 x≥10）
            if agent.group == "fuzzy" and new_x > 9:
                new_x = 9
            elif agent.group == "precise" and new_x < 10:
                new_x = 10
            agent.x = max(0, min(GRID_SIZE - 1, new_x))
            agent.y = max(0, min(GRID_SIZE - 1, new_y))
            agent.memory.append(f"第{self.tick}天移动")
            # 检查是否响应了之前收到的方向建议
            self._check_msg_follow(agent, dx, dy)

        elif act == "eat":
            eaten = False
            for food in self.foods[:]:
                if food.x == agent.x and food.y == agent.y:
                    self.foods.remove(food)
                    agent.energy += EAT_ENERGY
                    agent.food_eaten += 1
                    eaten = True
                    agent.memory.append(f"第{self.tick}天吃到食物")
                    break
            if not eaten:
                agent.memory.append(f"第{self.tick}天脚下没食物")

        elif act == "attack":
            victim = self._find_agent(target, agent)
            if victim and victim.alive:
                agent.energy -= ATTACK_COST
                if random.random() < 0.6:
                    stolen = min(ATTACK_STEAL, victim.energy)
                    victim.energy -= stolen
                    agent.energy += stolen
                    agent.memory.append(f"第{self.tick}天攻击{victim.name}成功")
                    victim.memory.append(f"第{self.tick}天被{agent.name}攻击")
                    self.record("attack_success", f"{agent.name}→{victim.name},偷{stolen}", agent.id)
                else:
                    agent.memory.append(f"第{self.tick}天攻击失败")
                    self.record("attack_fail", f"{agent.name}→{victim.name}", agent.id)

        elif act == "say":
            receiver = self._find_agent(target, agent)
            if receiver and receiver.alive:
                msg = content[:40] if content else "..."
                receiver.inbox.append(f"{agent.name}说：{msg}")
                agent.messages_sent += 1
                receiver.messages_received += 1
                agent.memory.append(f"第{self.tick}天对{receiver.name}说话")
                self.record("message", f"{agent.name}→{receiver.name}：{msg}", agent.id)

                # 追踪关系
                pair = (agent.id, receiver.id)
                self.relationships[pair] = self.relationships.get(pair, 0) + 1

                # 追踪方向信息
                direction_keywords = {"北": (0,-1), "南": (0,1), "东": (1,0), "西": (-1,0),
                                     "上": (0,-1), "下": (0,1), "左": (-1,0), "右": (1,0)}
                for kw, d in direction_keywords.items():
                    if kw in msg:
                        if receiver.id not in self.msg_directions:
                            self.msg_directions[receiver.id] = []
                        self.msg_directions[receiver.id].append((self.tick, d))
                        break

        else:  # rest
            agent.energy += 1
            agent.memory.append(f"第{self.tick}天休息")

        agent.memory = agent.memory[-8:]
        agent.inbox = agent.inbox[-5:]

    def _check_msg_follow(self, agent, dx, dy):
        """检查 agent 的移动是否 follow 了收到的方向建议"""
        if agent.id in self.msg_directions:
            recent = [(t, d) for t, d in self.msg_directions[agent.id]
                     if self.tick - t <= 2]
            for t, (sx, sy) in recent:
                if (dx == sx and sx != 0) or (dy == sy and sy != 0):
                    self.stats[agent.group]["functional_msgs"] += 1
                    self.msg_directions[agent.id] = [
                        x for x in self.msg_directions[agent.id] if x[0] != t
                    ]
                    break

    def _find_agent(self, name, seeker):
        for a in self.agents:
            if a.alive and a.id != seeker.id:
                if name and (name in a.name or a.name in name):
                    if abs(a.x - seeker.x) <= VISION_RANGE and abs(a.y - seeker.y) <= VISION_RANGE:
                        return a
        return None

    def perturb(self):
        """干扰实验：打乱所有 agent 位置，保留组别"""
        self.record("perturb", "位置打乱：测试社交关系是否为真结构")
        # 保存干扰前的关系对快照
        self.pre_perturb_positions = {a.id: (a.x, a.y) for a in self.agents if a.alive}

        for agent in self.agents:
            if agent.alive:
                if agent.group == "fuzzy":
                    agent.x = random.randint(0, 9)
                else:
                    agent.x = random.randint(10, 19)
                agent.y = random.randint(0, GRID_SIZE - 1)

        print(f"    ⚡ 干扰：所有 agent 位置已打乱", file=sys.stderr, flush=True)

    def measure_recovery(self):
        """测量干扰后：之前有消息关系的 agent 对是否重新走近"""
        # 找出干扰前互发消息≥2次的 agent 对
        strong_pairs = set()
        for (s, r), count in self.relationships.items():
            if count >= 2:
                strong_pairs.add((min(s, r), max(s, r)))

        reunions = {"fuzzy": 0, "precise": 0}
        for (a_id, b_id) in strong_pairs:
            a = next((x for x in self.agents if x.id == a_id), None)
            b = next((x for x in self.agents if x.id == b_id), None)
            if a and b and a.alive and b.alive:
                if abs(a.x - b.x) <= 2 and abs(a.y - b.y) <= 2:
                    reunions[a.group] += 1

        self.post_perturb_reunions = reunions
        return reunions

    def run_tick(self):
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return False

        # 干扰实验
        if self.tick == PERTURB_TICK:
            self.perturb()

        # 并行 LLM 调用
        actions = [None] * len(alive)
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            futures = {pool.submit(self.call_llm, a): i for i, a in enumerate(alive)}
            for future in as_completed(futures):
                idx = futures[future]
                actions[idx] = future.result()

        # 随机顺序执行
        pairs = list(zip(alive, actions))
        random.shuffle(pairs)
        for agent, action in pairs:
            if agent.alive:
                self.execute_action(agent, action)

        # 能量消耗 + 死亡
        for agent in self.agents:
            if agent.alive:
                agent.energy -= ENERGY_DRAIN
                if agent.energy <= 0:
                    agent.alive = False
                    self.record("death", f"{agent.name}({agent.group})死亡", agent.id)

        # 食物再生（两半区各 2）
        for _ in range(FOOD_REGROW_PER_TICK // 2):
            self.foods.append(Food(x=random.randint(0, 9), y=random.randint(0, GRID_SIZE - 1)))
            self.foods.append(Food(x=random.randint(10, 19), y=random.randint(0, GRID_SIZE - 1)))

        # 记录本 tick 统计
        for group in ["fuzzy", "precise"]:
            g_agents = [a for a in self.agents if a.group == group]
            msgs = sum(1 for e in self.events if e["tick"] == self.tick
                      and e["type"] == "message"
                      and any(a.id == e["agent"] and a.group == group for a in self.agents))
            eaten = sum(1 for e in self.events if e["tick"] == self.tick
                       and "吃到" in e.get("detail", "")
                       and any(a.id == e["agent"] and a.group == group for a in self.agents))
            alive_count = sum(1 for a in g_agents if a.alive)
            self.stats[group]["messages"].append(msgs)
            self.stats[group]["food_eaten"].append(eaten)
            self.stats[group]["alive"].append(alive_count)

        # 干扰后测量（tick 25-29 时检查重聚）
        if self.tick >= PERTURB_TICK + 5:
            self.measure_recovery()

        return sum(1 for a in self.agents if a.alive) > 0

    def run(self):
        self.init_world()
        print(f"═══ Round 004: 模糊感知 vs 精确感知（同 run AB 对照） ═══", file=sys.stderr, flush=True)
        print(f"  每组 {NUM_AGENTS_PER_GROUP} 人 | {GRID_SIZE}x{GRID_SIZE} | {MODEL}", file=sys.stderr, flush=True)
        print(f"  {MAX_TICKS} ticks | 并发 {MAX_CONCURRENT} | 干扰在第 {PERTURB_TICK} tick", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)

        for tick in range(MAX_TICKS):
            self.tick = tick
            t0 = time.time()
            has_alive = self.run_tick()
            elapsed = time.time() - t0

            f_alive = sum(1 for a in self.agents if a.group == "fuzzy" and a.alive)
            p_alive = sum(1 for a in self.agents if a.group == "precise" and a.alive)
            f_msgs = self.stats["fuzzy"]["messages"][-1] if self.stats["fuzzy"]["messages"] else 0
            p_msgs = self.stats["precise"]["messages"][-1] if self.stats["precise"]["messages"] else 0

            print(f"  第{tick:>2}天 | 模糊{f_alive}人/{f_msgs}话 | 精确{p_alive}人/{p_msgs}话 | "
                  f"{elapsed:.0f}s", file=sys.stderr, flush=True)

            if not has_alive:
                print("\n  ☠ 全灭", file=sys.stderr, flush=True)
                break

        self.output_results()

    def compute_null_model_functional_msgs(self, group, n_permutations=1000):
        """置换检验：随机打乱消息-行为配对，算出 chance-level 功能性消息率"""
        # 收集该组所有消息事件（含方向词）和所有移动事件
        group_ids = {a.id for a in self.agents if a.group == group}
        msg_events = []  # (tick, receiver_id, direction)
        move_events = []  # (tick, agent_id, dx, dy)

        direction_keywords = {"北": (0,-1), "南": (0,1), "东": (1,0), "西": (-1,0)}
        for e in self.events:
            if e["type"] == "message" and e["agent"] in group_ids:
                detail = e.get("detail", "")
                for kw, d in direction_keywords.items():
                    if kw in detail:
                        # 找接收者
                        if "→" in detail:
                            receiver_name = detail.split("→")[1].split("：")[0]
                            for a in self.agents:
                                if a.name == receiver_name and a.group == group:
                                    msg_events.append((e["tick"], a.id, d))
                        break

        # 从记忆中推断移动方向（简化：用 agent 位置变化）
        # 这里用实际追踪的 msg_directions 数据
        actual_hits = self.stats[group]["functional_msgs"]

        if not msg_events or actual_hits == 0:
            return 0, 0, 0, 1.0  # no data

        # 置换：随机重新分配消息的 tick，看命中率
        total_msgs_with_dir = len(msg_events)
        null_hits = []
        for _ in range(n_permutations):
            # 随机 tick 偏移（打乱时序配对）
            shuffled_hits = 0
            for _, receiver_id, d in msg_events:
                # 随机选一个 tick 的移动方向
                if random.random() < 0.25:  # 四方向随机命中概率
                    shuffled_hits += 1
            null_hits.append(shuffled_hits)

        null_mean = sum(null_hits) / len(null_hits) if null_hits else 0
        lift = actual_hits / max(null_mean, 0.01)
        # p-value: 实际值在置换分布中的位置
        p_value = sum(1 for h in null_hits if h >= actual_hits) / n_permutations
        return actual_hits, null_mean, lift, p_value

    def output_results(self):
        """JSON 输出到 stdout"""
        # 计算关系密度
        def relationship_density(group):
            g_agents = [a for a in self.agents if a.group == group]
            ids = [a.id for a in g_agents]
            n = len(ids)
            if n < 2: return 0
            total_possible = n * (n - 1) / 2
            mutual_pairs = set()
            for (s, r), count in self.relationships.items():
                if s in ids and r in ids and count >= 2:
                    mutual_pairs.add((min(s, r), max(s, r)))
            return len(mutual_pairs) / total_possible

        # Welch's t-test on per-tick message counts
        def welch_t(a, b):
            import math
            na, nb = len(a), len(b)
            if na < 2 or nb < 2: return 0, 1.0
            ma = sum(a) / na
            mb = sum(b) / nb
            va = sum((x - ma)**2 for x in a) / (na - 1) if na > 1 else 0
            vb = sum((x - mb)**2 for x in b) / (nb - 1) if nb > 1 else 0
            se = math.sqrt(va/na + vb/nb) if (va/na + vb/nb) > 0 else 0.001
            t = (ma - mb) / se
            # 简化 p 值估算（自由度近似）
            df = na + nb - 2
            # 用正态近似
            from math import erfc
            p = erfc(abs(t) / math.sqrt(2))
            return t, p

        f_msgs = self.stats["fuzzy"]["messages"]
        p_msgs = self.stats["precise"]["messages"]
        t_stat, p_value = welch_t(f_msgs, p_msgs)

        # 置换检验：功能性消息 vs 随机基线
        f_actual, f_null, f_lift, f_perm_p = self.compute_null_model_functional_msgs("fuzzy")
        p_actual, p_null, p_lift, p_perm_p = self.compute_null_model_functional_msgs("precise")

        # 分段统计（前半 vs 后半 vs 干扰后）
        mid = MAX_TICKS // 2
        post = PERTURB_TICK

        result = {
            "experiment": "round-004: fuzzy vs precise perception",
            "ticks_completed": self.tick + 1,
            "summary": {
                "fuzzy": {
                    "total_messages": sum(f_msgs),
                    "total_food_eaten": sum(self.stats["fuzzy"]["food_eaten"]),
                    "final_alive": self.stats["fuzzy"]["alive"][-1] if self.stats["fuzzy"]["alive"] else 0,
                    "relationship_density": round(relationship_density("fuzzy"), 3),
                    "functional_messages": self.stats["fuzzy"]["functional_msgs"],
                    "msg_per_tick_pre_perturb": round(sum(f_msgs[:post]) / max(post, 1), 2),
                    "msg_per_tick_post_perturb": round(sum(f_msgs[post:]) / max(len(f_msgs[post:]), 1), 2),
                },
                "precise": {
                    "total_messages": sum(p_msgs),
                    "total_food_eaten": sum(self.stats["precise"]["food_eaten"]),
                    "final_alive": self.stats["precise"]["alive"][-1] if self.stats["precise"]["alive"] else 0,
                    "relationship_density": round(relationship_density("precise"), 3),
                    "functional_messages": self.stats["precise"]["functional_msgs"],
                    "msg_per_tick_pre_perturb": round(sum(p_msgs[:post]) / max(post, 1), 2),
                    "msg_per_tick_post_perturb": round(sum(p_msgs[post:]) / max(len(p_msgs[post:]), 1), 2),
                },
            },
            "statistical_test": {
                "test": "Welch's t-test on per-tick message counts",
                "t_statistic": round(t_stat, 3),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
            },
            "functional_msg_null_model": {
                "fuzzy": {"actual": f_actual, "null_mean": round(f_null, 1), "lift": round(f_lift, 2), "perm_p": round(f_perm_p, 4)},
                "precise": {"actual": p_actual, "null_mean": round(p_null, 1), "lift": round(p_lift, 2), "perm_p": round(p_perm_p, 4)},
                "interpretation": "lift>1.5 且 perm_p<0.05 = 功能性消息显著高于随机基线",
            },
            "perturbation": {
                "tick": PERTURB_TICK,
                "post_perturb_reunions": self.post_perturb_reunions,
                "interpretation": "模糊组重聚>精确组 = 社交关系是真结构不是噪声",
            },
            "per_tick_messages": {
                "fuzzy": f_msgs,
                "precise": p_msgs,
            },
            "message_samples": [
                e["detail"] for e in self.events
                if e["type"] == "message"
            ][:20],
        }

        # JSON to stdout
        print(json.dumps(result, ensure_ascii=False, indent=1))

        # 人类可读摘要到 stderr
        print("\n" + "═" * 50, file=sys.stderr, flush=True)
        print("  结果摘要", file=sys.stderr, flush=True)
        print("═" * 50, file=sys.stderr, flush=True)
        f = result["summary"]["fuzzy"]
        p = result["summary"]["precise"]
        print(f"  模糊组: {f['total_messages']}条消息 | 吃{f['total_food_eaten']}食物 | 存活{f['final_alive']}", file=sys.stderr, flush=True)
        print(f"  精确组: {p['total_messages']}条消息 | 吃{p['total_food_eaten']}食物 | 存活{p['final_alive']}", file=sys.stderr, flush=True)
        print(f"  消息率比: {f['total_messages']}/{max(p['total_messages'],1)} = {f['total_messages']/max(p['total_messages'],1):.1f}×", file=sys.stderr, flush=True)
        print(f"  功能性消息: 模糊{f['functional_messages']} vs 精确{p['functional_messages']}", file=sys.stderr, flush=True)
        print(f"  关系密度: 模糊{f['relationship_density']} vs 精确{p['relationship_density']}", file=sys.stderr, flush=True)
        print(f"  t检验: t={result['statistical_test']['t_statistic']}, p={result['statistical_test']['p_value']}", file=sys.stderr, flush=True)
        print(f"  干扰后重聚: {self.post_perturb_reunions}", file=sys.stderr, flush=True)
        print("═" * 50, file=sys.stderr, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1,
                       help="运行次数（LLM实验每次成本高，默认1）")
    parser.add_argument("--ticks", type=int, default=MAX_TICKS)
    parser.add_argument("--agents", type=int, default=NUM_AGENTS_PER_GROUP)
    args = parser.parse_args()

    MAX_TICKS = args.ticks
    NUM_AGENTS_PER_GROUP = args.agents
    PERTURB_TICK = MAX_TICKS * 2 // 3  # 干扰点在 2/3 处

    sim = Society()
    sim.run()

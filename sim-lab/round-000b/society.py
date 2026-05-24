import os
"""
LLM 社会模拟 — 30 个 AI agent 在资源有限的世界里生存、交流、博弈
观察：没有预设规则时，会自发涌现什么社会结构？
"""

import asyncio
import json
import random
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# ─── 配置 ───────────────────────────────────────────────────────────────────────

API_BASE = os.environ.get("IDEALAB_API_BASE", "https://api.openai.com/v1")
API_KEY = os.environ["IDEALAB_API_KEY"]
MODEL = "Qwen3.6-Plus-DogFooding"

GRID_SIZE = 20
NUM_AGENTS = 30
MAX_TICKS = 150

FOOD_INITIAL = 60          # 初始食物数量
FOOD_REGROW_PER_TICK = 4   # 每 tick 再生食物
INITIAL_ENERGY = 100
ENERGY_DRAIN = 2           # 每 tick 被动消耗
EAT_ENERGY = 30            # 吃一次回复
ATTACK_COST = 8            # 攻击消耗
ATTACK_STEAL = 20          # 攻击成功偷取
REPRODUCE_THRESHOLD = 160  # 繁殖门槛
REPRODUCE_COST = 80        # 繁殖消耗
VISION_RANGE = 4           # 视野半径
MAX_CONCURRENT = 10        # 并发 API 调用数

# 真实感中文名
NAMES = [
    "陈舟", "林晚", "苏青禾", "何漫", "叶知秋",
    "周深海", "吴桐", "郑一帆", "赵暮雨", "钱澄",
    "孙宁远", "李沉鱼", "王半山", "刘听风", "杨小满",
    "黄霁", "徐行止", "冯未迟", "蒋望舒", "沈不言",
    "韩素", "曹觉晓", "谢长安", "邓既明", "唐清圆",
    "宋无忧", "梁栖迟", "许若谷", "萧默然", "罗与归",
    "顾惊蛰", "白露生", "秦子衿", "魏无羡", "姜可期",
    "高远山", "戚微澜", "褚轻舟", "陆九渊", "傅近光",
]

# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: int
    name: str
    x: int
    y: int
    energy: int = INITIAL_ENERGY
    alive: bool = True
    generation: int = 0
    parent_wisdom: str = ""
    memory: list = field(default_factory=list)       # 最近事件
    inbox: list = field(default_factory=list)         # 收到的消息
    kill_count: int = 0
    food_eaten: int = 0
    messages_sent: int = 0
    children: int = 0
    born_tick: int = 0

@dataclass
class Food:
    x: int
    y: int

# ─── 模拟核心 ──────────────────────────────────────────────────────────────────

class Society:
    def __init__(self):
        self.tick = 0
        self.agents: list[Agent] = []
        self.foods: list[Food] = []
        self.log: list[dict] = []           # 全量事件日志
        self.population_history: list = []
        self.next_id = 0
        self.semaphore = None  # 在 async 上下文中初始化

    def init_world(self):
        for i in range(NUM_AGENTS):
            self.spawn_agent(name=NAMES[i], generation=0)
        for _ in range(FOOD_INITIAL):
            self.spawn_food()
        self.record("world_init", f"{NUM_AGENTS} agents, {FOOD_INITIAL} food, {GRID_SIZE}x{GRID_SIZE} grid")

    def spawn_agent(self, name=None, generation=0, parent_wisdom="", x=None, y=None):
        if x is None:
            x = random.randint(0, GRID_SIZE - 1)
        if y is None:
            y = random.randint(0, GRID_SIZE - 1)
        if name is None:
            name = random.choice(NAMES) + str(self.next_id)
        agent = Agent(
            id=self.next_id, name=name, x=x, y=y,
            generation=generation, parent_wisdom=parent_wisdom,
            born_tick=self.tick
        )
        self.next_id += 1
        self.agents.append(agent)
        return agent

    def spawn_food(self):
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        self.foods.append(Food(x=x, y=y))

    def record(self, event_type: str, detail: str, agent_id: int = -1):
        entry = {"tick": self.tick, "type": event_type, "agent": agent_id, "detail": detail}
        self.log.append(entry)
        # 实时写盘，进程中断也不丢
        with open(Path(__file__).parent / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_perception(self, agent: Agent) -> str:
        """构建 agent 的感知文本——模糊、感性、像人的主观体验"""
        nearby_agents = []
        nearby_food = []

        for other in self.agents:
            if other.id != agent.id and other.alive:
                dx = abs(other.x - agent.x)
                dy = abs(other.y - agent.y)
                if dx <= VISION_RANGE and dy <= VISION_RANGE:
                    dist = dx + dy
                    if dist <= 1:
                        proximity = "就在你身边"
                    elif dist <= 3:
                        proximity = "不远处"
                    else:
                        proximity = "远远地"
                    dirs = []
                    if other.x > agent.x: dirs.append("东")
                    elif other.x < agent.x: dirs.append("西")
                    if other.y > agent.y: dirs.append("南")
                    elif other.y < agent.y: dirs.append("北")
                    dir_str = "".join(dirs) if dirs else "旁边"
                    nearby_agents.append(f"{other.name}{proximity}（{dir_str}边）")

        for food in self.foods:
            dx = abs(food.x - agent.x)
            dy = abs(food.y - agent.y)
            if dx <= VISION_RANGE and dy <= VISION_RANGE:
                nearby_food.append((food.x - agent.x, food.y - agent.y))

        # 模糊食物感知
        if not nearby_food:
            food_desc = "你四处张望，看不到任何能吃的东西。"
        else:
            nearby_food.sort(key=lambda p: abs(p[0]) + abs(p[1]))
            closest = nearby_food[0]
            dx, dy = closest
            if dx == 0 and dy == 0:
                food_desc = "你脚边就有能吃的东西。"
            else:
                dist = abs(dx) + abs(dy)
                dirs = []
                if dy < 0: dirs.append("北")
                elif dy > 0: dirs.append("南")
                if dx < 0: dirs.append("西")
                elif dx > 0: dirs.append("东")
                dir_str = "".join(dirs)
                if dist <= 2:
                    food_desc = f"你好像闻到{dir_str}边不远处有食物的气味。"
                else:
                    food_desc = f"你隐约觉得{dir_str}边可能有吃的，但不太确定。"
                if len(nearby_food) > 3:
                    food_desc += "这附近食物似乎不少。"

        # 模糊饥饿感知
        energy_ratio = agent.energy / INITIAL_ENERGY
        if energy_ratio > 1.5:
            hunger = "你精力充沛，肚子饱饱的。"
        elif energy_ratio > 1.0:
            hunger = "你感觉状态还不错。"
        elif energy_ratio > 0.5:
            hunger = "你有点饿了。"
        elif energy_ratio > 0.25:
            hunger = "你很饿，胃在隐隐作痛。"
        else:
            hunger = "你饿得头晕眼花，感觉自己快撑不住了。"

        # 时间感
        if self.tick < 5:
            time_feel = "你醒来没多久，对这个地方还很陌生。"
        elif self.tick < 20:
            time_feel = "你已经在这里待了一些日子了。"
        else:
            time_feel = "你在这片荒野上已经很久了。"

        lines = [time_feel, hunger, food_desc]

        if nearby_agents:
            if len(nearby_agents) == 1:
                lines.append(f"你看到{nearby_agents[0]}。")
            else:
                lines.append(f"你看到附近有几个人：{'、'.join(nearby_agents)}。")
        else:
            lines.append("四周空无一人。")

        if agent.inbox:
            recent_msgs = agent.inbox[-3:]
            lines.append("你想起有人跟你说过：" + "；".join(recent_msgs))

        if agent.memory:
            lines.append("你回忆起最近：" + "；".join(agent.memory[-3:]))

        if agent.parent_wisdom:
            lines.append(f"你依稀记得长辈说过：「{agent.parent_wisdom}」")

        # 动作提示——简洁，不教策略
        lines.append("\n你可以：move(up/down/left/right) | eat | attack 名字 | say 名字 你想说的话 | rest | reproduce")
        lines.append('回复JSON：{"action":"...","target":"...","content":"...","thought":"你此刻在想什么"}')

        return "\n".join(lines)

    async def call_llm(self, agent: Agent, client) -> dict:
        """调用 LLM 获取 agent 动作"""
        prompt = self.get_perception(agent)

        async with self.semaphore:
            try:
                resp = await client.post(
                    f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": f"你是{agent.name}。你是一个真实的人——你会饿，会害怕，会好奇，会无聊，会想跟人说话，也会想一个人待着。你不知道这个世界的规则，只能靠自己摸索。你想活下去，但活着不是你唯一在乎的事。只回复JSON，不要解释。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.85,
                        "max_tokens": 120,
                    },
                    timeout=30.0,
                )
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                # 提取 JSON
                content = content.strip()
                # 处理 markdown 代码块
                if "```" in content:
                    match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                # 处理 <think> 标签 (qwen 的思考模式)
                if "<think>" in content:
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                # 尝试找到 JSON 对象
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group())
                return {"action": "rest", "thought": "解析失败"}
            except Exception as e:
                return {"action": "rest", "thought": f"API异常:{str(e)[:20]}"}

    def execute_action(self, agent: Agent, action: dict):
        """执行一个 agent 的动作"""
        act = action.get("action", "rest").lower().strip()
        target = action.get("target", "").strip()
        content = action.get("content", "").strip()
        thought = action.get("thought", "")

        if act == "move":
            direction = target.lower()
            dx, dy = 0, 0
            if direction in ("up", "北"): dy = -1
            elif direction in ("down", "南"): dy = 1
            elif direction in ("left", "西"): dx = -1
            elif direction in ("right", "东"): dx = 1
            else:
                dx, dy = random.choice([(0,1),(0,-1),(1,0),(-1,0)])
            agent.x = max(0, min(GRID_SIZE - 1, agent.x + dx))
            agent.y = max(0, min(GRID_SIZE - 1, agent.y + dy))
            agent.memory.append(f"第{self.tick}天移动")

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
                agent.memory.append(f"第{self.tick}天试图吃但脚下没食物")

        elif act == "attack":
            victim = self.find_agent_by_name(target, agent)
            if victim and victim.alive:
                agent.energy -= ATTACK_COST
                # 攻击成功率取决于能量差
                success = random.random() < 0.6
                if success:
                    stolen = min(ATTACK_STEAL, victim.energy)
                    victim.energy -= stolen
                    agent.energy += stolen
                    agent.kill_count += (1 if victim.energy <= 0 else 0)
                    agent.memory.append(f"第{self.tick}天攻击{victim.name}成功，偷{stolen}能量")
                    victim.memory.append(f"第{self.tick}天被{agent.name}攻击，失{stolen}能量")
                    self.record("attack_success", f"{agent.name} → {victim.name}, 偷{stolen}", agent.id)
                else:
                    agent.memory.append(f"第{self.tick}天攻击{victim.name}失败")
                    victim.memory.append(f"第{self.tick}天被{agent.name}攻击但躲开了")
                    self.record("attack_fail", f"{agent.name} → {victim.name}", agent.id)
            else:
                agent.memory.append(f"第{self.tick}天想攻击但找不到目标")

        elif act == "say":
            receiver = self.find_agent_by_name(target, agent)
            if receiver and receiver.alive:
                msg = content[:30] if content else "..."
                receiver.inbox.append(f"{agent.name}说：{msg}")
                agent.messages_sent += 1
                agent.memory.append(f"第{self.tick}天对{receiver.name}说话")
                self.record("message", f"{agent.name}→{receiver.name}：{msg}", agent.id)

        elif act == "reproduce":
            if agent.energy >= REPRODUCE_THRESHOLD:
                agent.energy -= REPRODUCE_COST
                wisdom = content[:50] if content else "活下去"
                child = self.spawn_agent(
                    name=agent.name + "·" + str(agent.children + 1),
                    generation=agent.generation + 1,
                    parent_wisdom=wisdom,
                    x=agent.x, y=agent.y
                )
                agent.children += 1
                agent.memory.append(f"第{self.tick}天繁殖了后代{child.name}")
                self.record("reproduce", f"{agent.name}→{child.name}，遗言：{wisdom}", agent.id)

        else:  # rest
            agent.energy += 1
            agent.memory.append(f"第{self.tick}天休息")

        # 保留最近 8 条记忆
        agent.memory = agent.memory[-8:]
        agent.inbox = agent.inbox[-5:]

    def find_agent_by_name(self, name: str, seeker: Agent):
        """模糊匹配视野内的 agent"""
        for a in self.agents:
            if a.alive and a.id != seeker.id:
                if name in a.name or a.name in name:
                    dx = abs(a.x - seeker.x)
                    dy = abs(a.y - seeker.y)
                    if dx <= VISION_RANGE and dy <= VISION_RANGE:
                        return a
        return None

    async def run_tick(self, client):
        """执行一个时间步"""
        alive_agents = [a for a in self.agents if a.alive]
        if not alive_agents:
            return False

        # 并发调用所有 agent
        tasks = [self.call_llm(a, client) for a in alive_agents]
        actions = await asyncio.gather(*tasks)

        # 随机化执行顺序（避免 ID 优势）
        pairs = list(zip(alive_agents, actions))
        random.shuffle(pairs)

        for agent, action in pairs:
            if agent.alive:
                self.execute_action(agent, action)

        # 被动能量消耗 + 死亡判定
        for agent in self.agents:
            if agent.alive:
                agent.energy -= ENERGY_DRAIN
                if agent.energy <= 0:
                    agent.alive = False
                    agent.memory.append(f"第{self.tick}天死亡")
                    self.record("death", f"{agent.name}死亡，存活{self.tick - agent.born_tick}天，吃{agent.food_eaten}食物，杀{agent.kill_count}人", agent.id)

        # 食物再生
        for _ in range(FOOD_REGROW_PER_TICK):
            self.spawn_food()

        # 记录人口
        alive_count = sum(1 for a in self.agents if a.alive)
        self.population_history.append({"tick": self.tick, "alive": alive_count, "total_ever": len(self.agents), "food": len(self.foods)})

        return alive_count > 0

    async def run(self):
        """主循环"""
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.init_world()
        print(f"═══ LLM 社会模拟启动 ═══")
        print(f"  {NUM_AGENTS} agents | {GRID_SIZE}x{GRID_SIZE} | 模型: {MODEL}")
        print(f"  最大 {MAX_TICKS} ticks | 并发 {MAX_CONCURRENT}")
        print()

        import httpx
        async with httpx.AsyncClient() as client:
            for tick in range(MAX_TICKS):
                self.tick = tick
                t0 = time.time()
                has_alive = await self.run_tick(client)
                elapsed = time.time() - t0

                alive = sum(1 for a in self.agents if a.alive)
                msgs_this_tick = sum(1 for e in self.log if e["tick"] == tick and e["type"] == "message")
                attacks_this_tick = sum(1 for e in self.log if e["tick"] == tick and "attack" in e["type"])

                print(f"  第{tick:>3}天 | 存活{alive:>2} | 食物{len(self.foods):>3} | "
                      f"消息{msgs_this_tick} | 攻击{attacks_this_tick} | {elapsed:.1f}s")

                if not has_alive:
                    print("\n  ☠ 全灭")
                    break

                # 每 10 tick 输出社交快照
                if tick > 0 and tick % 10 == 0:
                    self.print_snapshot()

        self.save_results()
        self.print_final_report()

    def print_snapshot(self):
        """中间快照"""
        alive = [a for a in self.agents if a.alive]
        alive.sort(key=lambda a: a.energy, reverse=True)
        top3 = alive[:3]
        print(f"    ┌─ 快照: 最强者 ", end="")
        for a in top3:
            print(f"[{a.name} E{a.energy}]", end=" ")
        print()

    def print_final_report(self):
        """终局报告"""
        print("\n" + "═" * 60)
        print("  终 局 报 告")
        print("═" * 60)

        alive = [a for a in self.agents if a.alive]
        dead = [a for a in self.agents if not a.alive]

        print(f"\n  总人口: {len(self.agents)} (初始{NUM_AGENTS} + 繁殖{len(self.agents)-NUM_AGENTS})")
        print(f"  存活: {len(alive)} | 死亡: {len(dead)}")
        print(f"  总消息数: {sum(1 for e in self.log if e['type']=='message')}")
        print(f"  总攻击数: {sum(1 for e in self.log if 'attack' in e['type'])}")
        print(f"  总繁殖数: {sum(1 for e in self.log if e['type']=='reproduce')}")

        # 排行榜
        all_agents = sorted(self.agents, key=lambda a: a.food_eaten + a.kill_count * 5 + a.children * 10, reverse=True)
        print(f"\n  ── 综合排行（食物+击杀+后代）──")
        for i, a in enumerate(all_agents[:8]):
            status = "☠" if not a.alive else "♥"
            lifespan = (self.tick if a.alive else next((e["tick"] for e in self.log if e["agent"] == a.id and e["type"] == "death"), self.tick)) - a.born_tick
            print(f"    {i+1}. {status} {a.name} | 代{a.generation} | 活{lifespan}天 | "
                  f"吃{a.food_eaten} 杀{a.kill_count} 娃{a.children} 话{a.messages_sent}")

        # 社交网络
        msg_pairs = {}
        for e in self.log:
            if e["type"] == "message":
                detail = e["detail"]
                if "→" in detail:
                    pair = detail.split("：")[0]
                    msg_pairs[pair] = msg_pairs.get(pair, 0) + 1

        if msg_pairs:
            print(f"\n  ── 最活跃对话 ──")
            sorted_pairs = sorted(msg_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
            for pair, count in sorted_pairs:
                print(f"    {pair} ({count}条)")

        # 有趣的消息样本
        messages = [e for e in self.log if e["type"] == "message"]
        if messages:
            print(f"\n  ── 消息样本 ──")
            samples = random.sample(messages, min(10, len(messages)))
            for m in samples:
                print(f"    第{m['tick']}天 {m['detail']}")

        print("\n" + "═" * 60)

    def save_results(self):
        """保存完整日志"""
        output_dir = Path(__file__).parent

        # 事件日志
        with open(output_dir / "events.json", "w", encoding="utf-8") as f:
            json.dump(self.log, f, ensure_ascii=False, indent=1)

        # 人口曲线
        with open(output_dir / "population.json", "w", encoding="utf-8") as f:
            json.dump(self.population_history, f, ensure_ascii=False, indent=1)

        # Agent 墓志铭
        epitaphs = []
        for a in self.agents:
            epitaphs.append({
                "name": a.name, "generation": a.generation,
                "alive": a.alive, "energy": a.energy,
                "food_eaten": a.food_eaten, "kill_count": a.kill_count,
                "children": a.children, "messages_sent": a.messages_sent,
                "parent_wisdom": a.parent_wisdom,
                "last_memory": a.memory[-3:] if a.memory else [],
                "born_tick": a.born_tick,
            })
        with open(output_dir / "agents.json", "w", encoding="utf-8") as f:
            json.dump(epitaphs, f, ensure_ascii=False, indent=1)

        print(f"\n  结果已保存至 {output_dir}/")
        print(f"    events.json  — 全量事件日志")
        print(f"    population.json — 人口曲线")
        print(f"    agents.json — agent 档案")


# ─── 入口 ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = Society()
    asyncio.run(sim.run())

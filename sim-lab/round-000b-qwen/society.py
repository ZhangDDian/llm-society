import os
"""
Round 002 — 精确感知 + 人格 prompt
对照：只改 system prompt（从"游戏角色"→"有情绪的人"），感知格式与 000b 完全一致
假设：人格 framing 能释放非零社交行为，同时不影响生存能力
"""

import json
import random
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

# ─── 配置 ───────────────────────────────────────────────────────────────────────

API_BASE = os.environ.get("IDEALAB_API_BASE", "https://api.openai.com/v1")
API_KEY = os.environ["IDEALAB_API_KEY"]
MODEL = "Qwen3.6-Plus-DogFooding"

GRID_SIZE = 20
NUM_AGENTS = 30
MAX_TICKS = 50

FOOD_INITIAL = 60
FOOD_REGROW_PER_TICK = 4
INITIAL_ENERGY = 100
ENERGY_DRAIN = 2
EAT_ENERGY = 30
ATTACK_COST = 8
ATTACK_STEAL = 20
REPRODUCE_THRESHOLD = 160
REPRODUCE_COST = 80
VISION_RANGE = 4
MAX_CONCURRENT = 10

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
    memory: list = field(default_factory=list)
    inbox: list = field(default_factory=list)
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
        self.log: list[dict] = []
        self.population_history: list = []
        self.next_id = 0
        self.log_file = Path(__file__).parent / "events.jsonl"
        self.log_file.write_text("")  # 清空

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
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_perception(self, agent: Agent) -> str:
        """精确感知——与 000b 完全一致"""
        nearby_agents = []
        nearby_food = []

        for other in self.agents:
            if other.id != agent.id and other.alive:
                dx = abs(other.x - agent.x)
                dy = abs(other.y - agent.y)
                if dx <= VISION_RANGE and dy <= VISION_RANGE:
                    rel = f"({'东' if other.x > agent.x else '西' if other.x < agent.x else ''}{'南' if other.y > agent.y else '北' if other.y < agent.y else ''})"
                    nearby_agents.append(f"{other.name}{rel} 能量{other.energy}")

        for food in self.foods:
            dx = abs(food.x - agent.x)
            dy = abs(food.y - agent.y)
            if dx <= VISION_RANGE and dy <= VISION_RANGE:
                nearby_food.append((food.x - agent.x, food.y - agent.y))

        if not nearby_food:
            food_desc = "视野内无食物"
        else:
            nearby_food.sort(key=lambda p: abs(p[0]) + abs(p[1]))
            closest = nearby_food[0]
            dx, dy = closest
            if dx == 0 and dy == 0:
                food_desc = f"脚下有食物！直接eat！（视野内共{len(nearby_food)}处）"
            else:
                direction = ""
                if dy < 0: direction += "up"
                elif dy > 0: direction += "down"
                if dx < 0: direction += "/left" if direction else "left"
                elif dx > 0: direction += "/right" if direction else "right"
                food_desc = f"最近食物在{direction}方向{abs(dx)+abs(dy)}步（视野内共{len(nearby_food)}处）"

        days_left = agent.energy // ENERGY_DRAIN
        urgency = "⚠危险！" if days_left < 10 else ""

        food_here = any(f.x == agent.x and f.y == agent.y for f in self.foods)
        here_hint = "【脚下有食物！用eat可回复30能量】" if food_here else ""

        lines = [
            f"【第{self.tick}天】你是{agent.name}，第{agent.generation}代",
            f"位置({agent.x},{agent.y})，能量{agent.energy}（还能活{days_left}天{urgency}）{here_hint}",
            f"食物：{food_desc}",
            f"附近的人：{', '.join(nearby_agents) if nearby_agents else '无'}",
        ]

        if agent.inbox:
            recent_msgs = agent.inbox[-3:]
            lines.append("收到消息：" + " | ".join(recent_msgs))

        if agent.memory:
            lines.append("近况：" + "；".join(agent.memory[-4:]))

        if agent.parent_wisdom:
            lines.append(f"上代遗言：{agent.parent_wisdom}")

        lines.append(f"\n生存提示：先走到食物格子(move朝食物方向)，到了再eat。能量归零=死亡。")
        lines.append(f"可选动作：move(up/down/left/right) | eat(脚下有食物才有效!) | attack 名字 | say 名字 内容(≤30字) | rest | reproduce(能量≥{REPRODUCE_THRESHOLD}时,写遗言给后代)")
        lines.append('严格回复JSON：{"action":"动作","target":"目标或方向","content":"消息或遗言","thought":"一句内心话"}')

        return "\n".join(lines)

    def call_llm(self, agent: Agent) -> dict:
        prompt = self.get_perception(agent)
        try:
            resp = httpx.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "你是一个生存游戏中的角色。用最短的JSON回复你的动作。不要解释。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.85,
                    "max_tokens": 120,
                },
                timeout=45.0,
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            content = content.strip()
            if "```" in content:
                match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"action": "rest", "thought": "解析失败"}
        except Exception as e:
            return {"action": "rest", "thought": f"API异常:{str(e)[:20]}"}

    def execute_action(self, agent: Agent, action: dict):
        act = action.get("action", "rest").lower().strip()
        target = action.get("target", "").strip()
        content = action.get("content", "").strip()

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

        agent.memory = agent.memory[-8:]
        agent.inbox = agent.inbox[-5:]

    def find_agent_by_name(self, name: str, seeker: Agent):
        for a in self.agents:
            if a.alive and a.id != seeker.id:
                if name in a.name or a.name in name:
                    dx = abs(a.x - seeker.x)
                    dy = abs(a.y - seeker.y)
                    if dx <= VISION_RANGE and dy <= VISION_RANGE:
                        return a
        return None

    def run_tick(self):
        alive_agents = [a for a in self.agents if a.alive]
        if not alive_agents:
            return False

        # 线程池并发调用 API
        actions = [None] * len(alive_agents)
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            futures = {pool.submit(self.call_llm, a): i for i, a in enumerate(alive_agents)}
            for future in as_completed(futures):
                idx = futures[future]
                actions[idx] = future.result()

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
                    agent.memory.append(f"第{self.tick}天死亡")
                    self.record("death", f"{agent.name}死亡，存活{self.tick - agent.born_tick}天，吃{agent.food_eaten}食物，杀{agent.kill_count}人", agent.id)

        for _ in range(FOOD_REGROW_PER_TICK):
            self.spawn_food()

        alive_count = sum(1 for a in self.agents if a.alive)
        self.population_history.append({"tick": self.tick, "alive": alive_count, "total_ever": len(self.agents), "food": len(self.foods)})

        return alive_count > 0

    def run(self):
        self.init_world()
        print(f"═══ Round 000b-qwen: 精确感知 + 原版prompt（基线） ═══", flush=True)
        print(f"  system: '你是一个生存游戏中的角色'", flush=True)
        print(f"  感知：精确（方向+步数+能量数字+生存提示）", flush=True)
        print(f"  {NUM_AGENTS} agents | {GRID_SIZE}x{GRID_SIZE} | 模型: {MODEL}", flush=True)
        print(f"  最大 {MAX_TICKS} ticks | 并发 {MAX_CONCURRENT}", flush=True)
        print(flush=True)

        for tick in range(MAX_TICKS):
            self.tick = tick
            t0 = time.time()
            has_alive = self.run_tick()
            elapsed = time.time() - t0

            alive = sum(1 for a in self.agents if a.alive)
            msgs_this_tick = sum(1 for e in self.log if e["tick"] == tick and e["type"] == "message")
            attacks_this_tick = sum(1 for e in self.log if e["tick"] == tick and "attack" in e["type"])

            print(f"  第{tick:>3}天 | 存活{alive:>2} | 食物{len(self.foods):>3} | "
                  f"消息{msgs_this_tick} | 攻击{attacks_this_tick} | {elapsed:.1f}s", flush=True)

            if not has_alive:
                print("\n  ☠ 全灭", flush=True)
                break

            if tick > 0 and tick % 10 == 0:
                self.print_snapshot()

        self.save_results()
        self.print_final_report()

    def print_snapshot(self):
        alive = [a for a in self.agents if a.alive]
        alive.sort(key=lambda a: a.energy, reverse=True)
        top3 = alive[:3]
        print(f"    ┌─ 快照: 最强者 ", end="")
        for a in top3:
            print(f"[{a.name} E{a.energy}]", end=" ")
        print()

    def print_final_report(self):
        print("\n" + "═" * 60)
        print("  Round 002 终局报告（精确感知 + 人格prompt）")
        print("═" * 60)

        alive = [a for a in self.agents if a.alive]
        dead = [a for a in self.agents if not a.alive]

        print(f"\n  总人口: {len(self.agents)} (初始{NUM_AGENTS} + 繁殖{len(self.agents)-NUM_AGENTS})")
        print(f"  存活: {len(alive)} | 死亡: {len(dead)}")
        print(f"  总消息数: {sum(1 for e in self.log if e['type']=='message')}")
        print(f"  总攻击数: {sum(1 for e in self.log if 'attack' in e['type'])}")
        print(f"  总繁殖数: {sum(1 for e in self.log if e['type']=='reproduce')}")

        all_agents = sorted(self.agents, key=lambda a: a.food_eaten + a.kill_count * 5 + a.children * 10, reverse=True)
        print(f"\n  ── 综合排行（食物+击杀+后代）──")
        for i, a in enumerate(all_agents[:8]):
            status = "☠" if not a.alive else "♥"
            lifespan = (self.tick if a.alive else next((e["tick"] for e in self.log if e["agent"] == a.id and e["type"] == "death"), self.tick)) - a.born_tick
            print(f"    {i+1}. {status} {a.name} | 代{a.generation} | 活{lifespan}天 | "
                  f"吃{a.food_eaten} 杀{a.kill_count} 娃{a.children} 话{a.messages_sent}")

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

        messages = [e for e in self.log if e["type"] == "message"]
        if messages:
            print(f"\n  ── 消息样本 ──")
            samples = random.sample(messages, min(10, len(messages)))
            for m in samples:
                print(f"    第{m['tick']}天 {m['detail']}")

        print("\n" + "═" * 60)

    def save_results(self):
        output_dir = Path(__file__).parent
        with open(output_dir / "events.json", "w", encoding="utf-8") as f:
            json.dump(self.log, f, ensure_ascii=False, indent=1)
        with open(output_dir / "population.json", "w", encoding="utf-8") as f:
            json.dump(self.population_history, f, ensure_ascii=False, indent=1)
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


if __name__ == "__main__":
    sim = Society()
    sim.run()

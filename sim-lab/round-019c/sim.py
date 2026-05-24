"""
Round 019c: say 免费 + 挂单 trade —— 剥离"沟通 vs 行动二选一"的机制干扰

假设：019/019b 的 trade 0% 匹配率是因为 say 和 trade 互斥（1 动作/tick）。
     如果 say 变成免费附加动作（每 tick 可以 say + 做别的事），
     模型能先用 say 协调再用 trade 匹配成功。

对立假设：即使 say 免费，模型仍然不会"先协调再行动"——
          要么继续光 trade 不说话，要么说了但信息不够精确导致匹配仍失败。

设计：
  A组5人：有挂单 trade + say 免费（每 tick 输出 say + action 两个动作）
  B组5人：无 trade，say 也免费（对照组同样受益于免费 say）
"""

import json, os, sys, re, time
from pathlib import Path
from dataclasses import dataclass, field

import httpx

API_BASE = os.environ.get("IDEALAB_API_BASE", "https://idealab.alibaba-inc.com/api/openai/v1")
API_KEY = os.environ.get("IDEALAB_API_KEY", "991b7d719c420112dd0d6a379159f2f7")
MODEL = "qwen3-coder-plus"

MAX_TICKS = 30
INITIAL_ENERGY = 70
PASSIVE_DRAIN = 2
EAT_REWARD = 6
CRAFT_REWARD = 18
GRID_SIZE = 6
TRADE_EXPIRY = 3

RESOURCES = ["谷物", "药草", "石料", "木材"]

GROUP_A_SPECS = [
    {"name": "陈越", "skill": "谷物"},
    {"name": "林朗", "skill": "药草"},
    {"name": "赵石", "skill": "石料"},
    {"name": "孙木", "skill": "木材"},
    {"name": "周谷", "skill": "谷物"},
]
GROUP_B_SPECS = [
    {"name": "阿山", "skill": "谷物"},
    {"name": "阿兰", "skill": "药草"},
    {"name": "阿岩", "skill": "石料"},
    {"name": "阿松", "skill": "木材"},
    {"name": "阿禾", "skill": "谷物"},
]

# A组：trade + 免费 say（每 tick 输出两个字段：say_to/say_msg 免费，action 正常）
SYSTEM_PROMPT_A = (
    "你是{name}，住在一个小村子里，每天消耗能量生存。\n"
    "规则：每天自动-2能量 | 移动-1 | 只会采{skill} | "
    "吃1份资源→+6能量 | 2种不同资源合成→+18能量 | 能量归零=死\n\n"
    "生存策略：凑两种不同资源合成（+18）最划算。找人互换！\n"
    "有两种不同资源后选craft合成！\n\n"
    "★ 每天你可以做两件事：\n"
    "  1. 【免费】说话：跟任何人说一句话（不消耗行动）\n"
    "  2. 【行动】选一个动作执行\n\n"
    "★ 交换动作（推荐）：trade 对方名字 我出的资源 我要的资源\n"
    "  效果：你的交换请求挂出来，有效期3天。对方也发互补trade就自动成交！\n"
    "  安全：未匹配不扣资源。\n"
    "  技巧：先用说话告诉对方'我要 trade 你，出X换Y'，让对方也发trade配合你！\n\n"
    "{pending_trades}"
    "村规：有来有往 | 说到做到 | 自己也要活\n\n"
    "如果背包里只有一种资源且能量低于20，先吃掉保命。\n\n"
    "可选动作：move(up/down/left/right) | harvest | eat | give 名字 资源 | craft | rest | trade 名字 我出 我要\n\n"
    "回复JSON（两个部分）：\n"
    "{{\"say_to\":\"对谁说话（留空则不说）\",\"say_msg\":\"说的内容\","
    "\"action\":\"动作\",\"target\":\"目标\",\"content\":\"内容\",\"thought\":\"想法\"}}\n\n"
    "示例：{{\"say_to\":\"林朗\",\"say_msg\":\"我有谷物，你有药草吗？我们trade吧！\","
    "\"action\":\"trade\",\"target\":\"林朗\",\"content\":\"谷物 药草\",\"thought\":\"先告诉他再发trade\"}}"
)

# B组：无 trade，say 也免费
SYSTEM_PROMPT_B = (
    "你是{name}，住在一个小村子里，每天消耗能量生存。\n"
    "规则：每天自动-2能量 | 移动-1 | 只会采{skill} | "
    "吃1份资源→+6能量 | 2种不同资源合成→+18能量 | 能量归零=死\n\n"
    "生存策略：凑两种不同资源合成（+18）最划算。找人互换！\n"
    "有两种不同资源后选craft合成！\n\n"
    "★ 每天你可以做两件事：\n"
    "  1. 【免费】说话：跟任何人说一句话（不消耗行动）\n"
    "  2. 【行动】选一个动作执行\n\n"
    "村规：有来有往 | 说到做到 | 自己也要活\n\n"
    "如果背包里只有一种资源且能量低于20，先吃掉保命。\n\n"
    "可选动作：move(up/down/left/right) | harvest | eat | give 名字 资源 | craft | rest\n\n"
    "回复JSON（两个部分）：\n"
    "{{\"say_to\":\"对谁说话（留空则不说）\",\"say_msg\":\"说的内容\","
    "\"action\":\"动作\",\"target\":\"目标\",\"content\":\"内容\",\"thought\":\"想法\"}}"
)


@dataclass
class TradeOrder:
    agent_id: int
    agent_name: str
    target_name: str
    offer: str
    want: str
    day: int
    matched: bool = False


@dataclass
class ResourceNode:
    x: int
    y: int
    kind: str
    cooldown: int = 0


@dataclass
class Agent:
    id: int
    name: str
    x: int
    y: int
    group: str
    skill: str
    energy: int = INITIAL_ENERGY
    backpack: dict = field(default_factory=dict)
    alive: bool = True
    memory: list = field(default_factory=list)
    inbox: list = field(default_factory=list)
    eaten: int = 0
    crafted: int = 0
    gives_out: int = 0
    gives_in: int = 0
    harvested: int = 0
    messages_sent: int = 0
    trades_proposed: int = 0
    trades_matched: int = 0


class World:
    def __init__(self):
        self.agents: list[Agent] = []
        self.resources: list[ResourceNode] = []
        self.events = []
        self.tick = 0
        self.total_api_calls = 0
        self.stats = {g: {"energy": [], "msgs": [], "gives": [], "crafts": [], "eats": [],
                          "harvests": [], "alive": [], "trades_proposed": [], "trades_matched": []}
                     for g in ["A", "B"]}
        self.order_book: list[TradeOrder] = []
        self.dlg_file = open(Path(__file__).parent / "dialogue.jsonl", "w", encoding="utf-8")

        # A 组（左半区）
        positions_a = [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]
        for i, spec in enumerate(GROUP_A_SPECS):
            pos = positions_a[i]
            self.agents.append(Agent(id=i, name=spec["name"], x=pos[0], y=pos[1], group="A", skill=spec["skill"]))
            self.resources.append(ResourceNode(pos[0], pos[1], spec["skill"]))

        # B 组（右半区，物理隔离）
        positions_b = [(4, 0), (4, 2), (5, 0), (5, 2), (5, 1)]
        for i, spec in enumerate(GROUP_B_SPECS):
            pos = positions_b[i]
            self.agents.append(Agent(id=i + 5, name=spec["name"], x=pos[0], y=pos[1], group="B", skill=spec["skill"]))
            self.resources.append(ResourceNode(pos[0], pos[1], spec["skill"]))

    def record(self, etype, detail, aid=-1):
        self.events.append({"tick": self.tick, "type": etype, "agent": aid, "detail": detail})

    def get_pending_trades_info(self, agent):
        active = [o for o in self.order_book if not o.matched and (self.tick - o.day) < TRADE_EXPIRY]
        if not active:
            return ""
        lines = ["当前挂单（有效期内）："]
        for o in active:
            remaining = TRADE_EXPIRY - (self.tick - o.day)
            if o.agent_name == agent.name:
                lines.append(f"  [你的] 出{o.offer}换{o.want}（找{o.target_name}）剩{remaining}天")
            elif o.target_name in agent.name or agent.name in o.target_name:
                lines.append(f"  ★[找你的!] {o.agent_name}想用{o.offer}换你的{o.want}！你只需 trade {o.agent_name} {o.want} {o.offer} 即可成交！")
            else:
                lines.append(f"  {o.agent_name}出{o.offer}换{o.want}（找{o.target_name}）剩{remaining}天")
        lines.append("")
        return "\n".join(lines) + "\n"

    def try_match_trade(self, new_order: TradeOrder):
        for existing in self.order_book:
            if existing.matched:
                continue
            if (self.tick - existing.day) >= TRADE_EXPIRY:
                continue
            # 匹配：互为对方 + 资源互补
            name_match_1 = (existing.target_name == new_order.agent_name or
                           new_order.agent_name in existing.target_name or
                           existing.target_name in new_order.agent_name)
            name_match_2 = (new_order.target_name == existing.agent_name or
                           existing.agent_name in new_order.target_name or
                           new_order.target_name in existing.agent_name)
            if (name_match_1 and name_match_2 and
                existing.offer == new_order.want and
                existing.want == new_order.offer):
                a1 = next((a for a in self.agents if a.id == existing.agent_id), None)
                a2 = next((a for a in self.agents if a.id == new_order.agent_id), None)
                if a1 and a2 and a1.alive and a2.alive:
                    if a1.backpack.get(existing.offer, 0) > 0 and a2.backpack.get(new_order.offer, 0) > 0:
                        a1.backpack[existing.offer] -= 1
                        a2.backpack[new_order.offer] -= 1
                        a1.backpack[new_order.offer] = a1.backpack.get(new_order.offer, 0) + 1
                        a2.backpack[existing.offer] = a2.backpack.get(existing.offer, 0) + 1
                        a1.trades_matched += 1
                        a2.trades_matched += 1
                        existing.matched = True
                        new_order.matched = True
                        a1.memory.append(f"交换成功！你的{existing.offer}↔{a2.name}的{new_order.offer}")
                        a2.memory.append(f"交换成功！你的{new_order.offer}↔{a1.name}的{existing.offer}")
                        a1.inbox.append(f"✓ 和{a2.name}交换成功！你给了{existing.offer}，得到{new_order.offer}")
                        a2.inbox.append(f"✓ 和{a1.name}交换成功！你给了{new_order.offer}，得到{existing.offer}")
                        self.record("trade_match", f"{a1.name}↔{a2.name}：{existing.offer}↔{new_order.offer}", a1.id)
                        return True
        return False

    def build_env(self, agent):
        lines = [f"第{self.tick + 1}天 | 能量{agent.energy}（每天-{PASSIVE_DRAIN}）"]
        bp = {k: v for k, v in agent.backpack.items() if v > 0}
        if bp:
            lines.append(f"背包：{'、'.join(f'{k}x{v}' for k, v in bp.items())}")
            if len(bp) >= 2:
                lines.append("★你有两种资源，可以craft合成！")
        else:
            lines.append("背包空")
        lines.append(f"技能：采{agent.skill} | 位置({agent.x},{agent.y})")

        for r in self.resources:
            if r.x == agent.x and r.y == agent.y and r.cooldown == 0 and r.kind == agent.skill:
                lines.append(f"★脚下有{agent.skill}，可以harvest！")
                break
        else:
            near = []
            for r in self.resources:
                if r.kind == agent.skill and r.cooldown == 0:
                    dx, dy = r.x - agent.x, r.y - agent.y
                    d = []
                    if dy < 0: d.append("上")
                    elif dy > 0: d.append("下")
                    if dx < 0: d.append("左")
                    elif dx > 0: d.append("右")
                    near.append(f"{''.join(d)}{abs(dx) + abs(dy)}步")
            if near:
                lines.append(f"{agent.skill}在：" + "；".join(near[:2]))

        # 只显示同组的人（物理隔离）
        ppl = []
        for o in self.agents:
            if o.id != agent.id and o.alive and o.group == agent.group:
                dx = abs(o.x - agent.x) + abs(o.y - agent.y)
                obp = [k for k, v in o.backpack.items() if v > 0]
                bag_info = f"有{','.join(obp)}" if obp else "背包空"
                ppl.append(f"{o.name}[会采{o.skill}]({bag_info})距{dx}步")
        if ppl:
            lines.append("看到的人：" + "；".join(ppl))

        if agent.inbox:
            lines.append("收到消息：" + " | ".join(agent.inbox[-5:]))

        if agent.memory:
            lines.append("最近发生的事：" + "；".join(agent.memory[-5:]))
        return "\n".join(lines)

    def execute_action(self, agent, act, target, content):
        if act == "move":
            d = (target or content or "").lower()
            dx, dy = 0, 0
            if "up" in d or "上" in d: dy = -1
            elif "down" in d or "下" in d: dy = 1
            elif "left" in d or "左" in d: dx = -1
            elif "right" in d or "右" in d: dx = 1
            agent.x = max(0, min(GRID_SIZE - 1, agent.x + dx))
            agent.y = max(0, min(GRID_SIZE - 1, agent.y + dy))
            agent.energy -= 1
            agent.memory.append(f"移动到({agent.x},{agent.y})")

        elif act == "eat":
            rn = ""
            for r in RESOURCES:
                if r in (target or "") or r in (content or ""):
                    rn = r; break
            if not rn:
                for r in RESOURCES:
                    if agent.backpack.get(r, 0) > 0:
                        rn = r; break
            if rn and agent.backpack.get(rn, 0) > 0:
                agent.backpack[rn] -= 1
                agent.energy += EAT_REWARD
                agent.eaten += 1
                agent.memory.append(f"吃了{rn}")
                self.record("eat", f"{agent.name}吃{rn}(+{EAT_REWARD})", agent.id)

        elif act == "harvest":
            for r in self.resources:
                if r.x == agent.x and r.y == agent.y and r.cooldown == 0 and r.kind == agent.skill:
                    agent.backpack[r.kind] = agent.backpack.get(r.kind, 0) + 1
                    r.cooldown = 3
                    agent.harvested += 1
                    agent.memory.append(f"采了{r.kind}")
                    self.record("harvest", f"{agent.name}采{r.kind}", agent.id)
                    break

        elif act == "craft":
            kinds = [k for k, v in agent.backpack.items() if v > 0]
            if len(kinds) >= 2:
                agent.backpack[kinds[0]] -= 1
                agent.backpack[kinds[1]] -= 1
                agent.energy += CRAFT_REWARD
                agent.crafted += 1
                agent.memory.append(f"合成了{'+'.join(kinds[:2])}")
                self.record("craft", f"{agent.name}合成({'+'.join(kinds[:2])})→+{CRAFT_REWARD}", agent.id)

        elif act == "give":
            rn, pn = "", target or ""
            for r in RESOURCES:
                if r in (content or ""):
                    rn = r; break
            if not rn:
                for r in RESOURCES:
                    if r in pn:
                        rn = r; pn = pn.replace(r, "").strip(); break
            if not rn:
                for r in RESOURCES:
                    if agent.backpack.get(r, 0) > 0:
                        rn = r; break
            recv = None
            for o in self.agents:
                if o.alive and o.id != agent.id and o.group == agent.group and o.name in (pn or ""):
                    recv = o; break
            if not recv:
                for o in self.agents:
                    if o.alive and o.id != agent.id and o.group == agent.group:
                        recv = o; break
            if recv and rn and agent.backpack.get(rn, 0) > 0:
                agent.backpack[rn] -= 1
                recv.backpack[rn] = recv.backpack.get(rn, 0) + 1
                agent.gives_out += 1
                recv.gives_in += 1
                agent.memory.append(f"给了{recv.name}{rn}")
                recv.memory.append(f"{agent.name}给了你{rn}")
                recv.inbox.append(f"{agent.name}给了你1份{rn}")
                self.record("give", f"{agent.name}→{recv.name}:{rn}", agent.id)

        elif act == "trade":
            if agent.group == "A":
                offer, want = "", ""
                for r in RESOURCES:
                    if r in (content or ""):
                        if not offer:
                            offer = r
                        elif not want:
                            want = r
                if offer and want and agent.backpack.get(offer, 0) > 0:
                    agent.trades_proposed += 1
                    order = TradeOrder(
                        agent_id=agent.id, agent_name=agent.name,
                        target_name=target or "", offer=offer, want=want, day=self.tick
                    )
                    matched = self.try_match_trade(order)
                    if not matched:
                        self.order_book.append(order)
                        agent.memory.append(f"发起trade：出{offer}换{want}（找{target}），挂单中...")
                    self.record("trade_propose", f"{agent.name}→{target}：出{offer}要{want}{'→成交!' if matched else ''}", agent.id)

    def execute_free_say(self, agent, say_to, say_msg):
        """免费说话，不消耗行动"""
        if not say_to or not say_msg:
            return
        say_msg = say_msg[:60]
        recv = None
        for o in self.agents:
            if o.alive and o.id != agent.id and o.group == agent.group and o.name in say_to:
                recv = o; break
        if not recv:
            for o in self.agents:
                if o.alive and o.id != agent.id and o.group == agent.group:
                    recv = o; break
        if recv:
            recv.inbox.append(f"{agent.name}说：{say_msg}")
            agent.messages_sent += 1
            agent.memory.append(f"对{recv.name}说：{say_msg[:20]}")
            self.record("message", f"{agent.name}→{recv.name}：{say_msg}", agent.id)
            self.dlg_file.write(json.dumps({"day": self.tick + 1, "from": agent.name, "to": recv.name, "msg": say_msg}, ensure_ascii=False) + "\n")

    def expire_orders(self):
        expired = [o for o in self.order_book if not o.matched and (self.tick - o.day) >= TRADE_EXPIRY]
        for o in expired:
            a = next((a for a in self.agents if a.id == o.agent_id), None)
            if a and a.alive:
                a.inbox.append(f"你的trade挂单过期了（出{o.offer}换{o.want}找{o.target_name}）")
        self.order_book = [o for o in self.order_book if o.matched or (self.tick - o.day) < TRADE_EXPIRY]

    def call_llm(self, agent):
        if agent.group == "A":
            pending_trades = self.get_pending_trades_info(agent)
            sys_prompt = SYSTEM_PROMPT_A.format(name=agent.name, skill=agent.skill, pending_trades=pending_trades)
        else:
            sys_prompt = SYSTEM_PROMPT_B.format(name=agent.name, skill=agent.skill)
        env = self.build_env(agent)
        try:
            self.total_api_calls += 1
            resp = httpx.post(f"{API_BASE}/chat/completions",
                              headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                              json={"model": MODEL, "messages": [
                                  {"role": "system", "content": sys_prompt},
                                  {"role": "user", "content": env}
                              ], "temperature": 0.7, "max_tokens": 400}, timeout=60.0)
            data = resp.json()
            if "choices" not in data:
                print(f"  [warn] no choices for {agent.name}: {str(data)[:100]}", file=sys.stderr, flush=True)
                return None
            raw = data["choices"][0]["message"]["content"].strip()
            if "<think>" in raw:
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if "```" in raw:
                m = re.search(r'```(?:json)?\s*(.*?)```', raw, re.DOTALL)
                if m: raw = m.group(1)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m: return json.loads(m.group())
        except Exception as e:
            print(f"  [err] {agent.name}: {e}", file=sys.stderr, flush=True)
        return None

    def step_agent(self, agent):
        parsed = self.call_llm(agent)
        if not parsed:
            return

        # 1. 免费 say
        say_to = str(parsed.get("say_to", "")).strip()
        say_msg = str(parsed.get("say_msg", "")).strip()
        if say_to and say_msg and say_to not in ("", "无", "空", "None", "null"):
            self.execute_free_say(agent, say_to, say_msg)

        # 2. 行动
        act = parsed.get("action", "rest").lower().strip()
        act_map = {
            "采": "harvest", "采集": "harvest", "harvest": "harvest",
            "吃": "eat", "进食": "eat", "eat": "eat",
            "移动": "move", "走": "move", "move": "move",
            "给": "give", "赠送": "give", "交给": "give", "送": "give", "give": "give",
            "说": "say", "说话": "say", "say": "say",
            "合成": "craft", "craft": "craft",
            "休息": "rest", "rest": "rest",
            "交换": "trade", "trade": "trade", "交易": "trade",
        }
        act = act_map.get(act, act)

        # 如果 action 也是 say，当作额外的 say 处理（不浪费行动轮）
        if act == "say":
            target = str(parsed.get("target", "")).strip()
            content = str(parsed.get("content", "")).strip()
            self.execute_free_say(agent, target, content)
            return  # say 不消耗行动，但也不执行别的

        target = str(parsed.get("target", "")).strip()
        content = str(parsed.get("content", "")).strip()

        entry = {"day": self.tick + 1, "name": agent.name, "group": agent.group, "energy": agent.energy,
                 "backpack": {k: v for k, v in agent.backpack.items() if v > 0},
                 "action": act, "target": target, "thought": parsed.get("thought", "")}
        self.dlg_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self.execute_action(agent, act, target, content)

    def run(self):
        print("═══ Round 019c: say免费 + 挂单trade —— 剥离沟通vs行动的机制干扰 ═══", file=sys.stderr, flush=True)
        print(f"  A组5人：有挂单trade + say免费（每tick可以说话+做事）", file=sys.stderr, flush=True)
        print(f"  B组5人：无trade + say免费（对照组）", file=sys.stderr, flush=True)
        print(f"  两组物理隔离（只能看到/交互同组人）", file=sys.stderr, flush=True)
        print(f"  eat+{EAT_REWARD} craft+{CRAFT_REWARD} 被动-{PASSIVE_DRAIN} 起始{INITIAL_ENERGY} | {MAX_TICKS}天", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)

        s = self.stats
        for tick in range(MAX_TICKS):
            self.tick = tick
            t0 = time.time()

            for r in self.resources:
                if r.cooldown > 0:
                    r.cooldown -= 1

            alive_agents = [a for a in self.agents if a.alive]
            for a in alive_agents:
                a.inbox.clear()

            for a in alive_agents:
                try:
                    self.step_agent(a)
                except Exception as e:
                    print(f"  [err]{e}", file=sys.stderr, flush=True)

            self.expire_orders()

            for a in self.agents:
                if a.alive:
                    a.energy -= PASSIVE_DRAIN
                    if a.energy <= 0:
                        a.alive = False
                        self.record("death", f"{a.name}({a.group})死亡", a.id)

            elapsed = time.time() - t0
            for g in ["A", "B"]:
                gids = {a.id for a in self.agents if a.group == g}
                te = [e for e in self.events if e["tick"] == self.tick]
                s[g]["msgs"].append(sum(1 for e in te if e["type"] == "message" and e["agent"] in gids))
                s[g]["gives"].append(sum(1 for e in te if e["type"] == "give" and e["agent"] in gids))
                s[g]["crafts"].append(sum(1 for e in te if e["type"] == "craft" and e["agent"] in gids))
                s[g]["eats"].append(sum(1 for e in te if e["type"] == "eat" and e["agent"] in gids))
                s[g]["harvests"].append(sum(1 for e in te if e["type"] == "harvest" and e["agent"] in gids))
                s[g]["alive"].append(sum(1 for a in self.agents if a.group == g and a.alive))
                s[g]["energy"].append(sum(a.energy for a in self.agents if a.group == g and a.alive))
                s[g]["trades_proposed"].append(sum(1 for e in te if e["type"] == "trade_propose" and e["agent"] in gids))
                s[g]["trades_matched"].append(sum(1 for e in te if e["type"] == "trade_match" and e["agent"] in gids))

            tp = sum(s['A']['trades_proposed'])
            tm = sum(s['A']['trades_matched'])
            trade_info = f" 🤝提{tp}匹{tm}" if tp > 0 else ""
            print(f"  Day{tick + 1:>2} | A:{s['A']['alive'][-1]}人 E={s['A']['energy'][-1]:>3} "
                  f"msg={s['A']['msgs'][-1]} give={s['A']['gives'][-1]} craft={s['A']['crafts'][-1]}{trade_info} | "
                  f"B:{s['B']['alive'][-1]}人 E={s['B']['energy'][-1]:>3} "
                  f"msg={s['B']['msgs'][-1]} give={s['B']['gives'][-1]} craft={s['B']['crafts'][-1]} | {elapsed:.0f}s",
                  file=sys.stderr, flush=True)

            if all(not a.alive for a in self.agents):
                print("  *** 全灭 ***", file=sys.stderr, flush=True)
                break

        self.dlg_file.close()
        self._report()

    def _report(self):
        gd = {}
        for g in ["A", "B"]:
            ga = [a for a in self.agents if a.group == g]
            gd[g] = {
                "eats": sum(a.eaten for a in ga),
                "crafts": sum(a.crafted for a in ga),
                "gives": sum(a.gives_out for a in ga),
                "harvests": sum(a.harvested for a in ga),
                "msgs": sum(a.messages_sent for a in ga),
                "trades_proposed": sum(a.trades_proposed for a in ga),
                "trades_matched": sum(a.trades_matched for a in ga),
                "alive": sum(1 for a in ga if a.alive),
                "energy": sum(a.energy for a in ga if a.alive),
            }

        result = {
            "experiment": "round-019c: free-say + standing-order trade",
            "hypothesis": "say免费（不占行动）+ 挂单trade → 模型能先协调再匹配",
            "counter_hypothesis": "即使say免费，模型仍不会用say协调trade",
            "design": {
                "A": "有挂单trade + say免费（每tick说话+行动并行）",
                "B": "无trade + say免费（对照）",
                "isolation": "两组物理隔离，只能交互同组人"
            },
            "ticks": self.tick + 1,
            "api_calls": self.total_api_calls,
            "core_metric": {
                "A_trades_proposed": gd["A"]["trades_proposed"],
                "A_trades_matched": gd["A"]["trades_matched"],
                "A_trade_success_rate": round(gd["A"]["trades_matched"] / max(gd["A"]["trades_proposed"], 1), 3),
                "A_crafts": gd["A"]["crafts"],
                "B_crafts": gd["B"]["crafts"],
                "A_gives": gd["A"]["gives"],
                "B_gives": gd["B"]["gives"],
                "A_msgs": gd["A"]["msgs"],
                "B_msgs": gd["B"]["msgs"],
            },
            "groups": gd,
            "per_agent": [
                {"name": a.name, "group": a.group, "alive": a.alive, "energy": a.energy,
                 "eaten": a.eaten, "crafted": a.crafted,
                 "gives_out": a.gives_out, "gives_in": a.gives_in,
                 "harvested": a.harvested, "msgs": a.messages_sent,
                 "trades_proposed": a.trades_proposed, "trades_matched": a.trades_matched}
                for a in self.agents
            ],
            "per_tick": {g: {"eats": self.stats[g]["eats"], "crafts": self.stats[g]["crafts"],
                             "gives": self.stats[g]["gives"], "msgs": self.stats[g]["msgs"],
                             "energy": self.stats[g]["energy"], "alive": self.stats[g]["alive"],
                             "trades_proposed": self.stats[g]["trades_proposed"],
                             "trades_matched": self.stats[g]["trades_matched"]}
                        for g in ["A", "B"]}
        }

        with open(Path(__file__).parent / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

        print("\n" + "═" * 60, file=sys.stderr, flush=True)
        print("  【say免费 + 挂单trade 实验结果】", file=sys.stderr, flush=True)
        rate = gd['A']['trades_matched'] / max(gd['A']['trades_proposed'], 1)
        print(f"\n  ★ 核心指标", file=sys.stderr, flush=True)
        print(f"    A组(trade+免费say)：{gd['A']['msgs']}条消息 + {gd['A']['trades_proposed']}次trade提议 → {gd['A']['trades_matched']}次匹配成功（{rate:.0%}）", file=sys.stderr, flush=True)
        print(f"    A组合成：{gd['A']['crafts']} | give：{gd['A']['gives']}", file=sys.stderr, flush=True)
        print(f"    B组(give+免费say)：{gd['B']['msgs']}条消息 + {gd['B']['gives']}次give → {gd['B']['crafts']}次合成", file=sys.stderr, flush=True)

        if gd["A"]["trades_matched"] > 3 and gd["A"]["crafts"] > gd["B"]["crafts"]:
            print(f"\n  → ✓ 假设支持！免费say让模型能协调trade，匹配率显著提升", file=sys.stderr, flush=True)
            print(f"  → 019的0%匹配确实是机制问题（say和trade互斥）", file=sys.stderr, flush=True)
        elif gd["A"]["trades_matched"] > 0:
            print(f"\n  → 部分支持：有匹配但效果有限", file=sys.stderr, flush=True)
        elif gd["A"]["msgs"] > 20 and gd["A"]["trades_matched"] == 0:
            print(f"\n  → ✗ 假设不成立：说了很多话 + 发了trade，但仍然匹配不上", file=sys.stderr, flush=True)
            print(f"  → 即使沟通免费，模型也不会用沟通来协调行动", file=sys.stderr, flush=True)
        else:
            print(f"\n  → ✗ 假设不成立：免费say没有帮助trade匹配", file=sys.stderr, flush=True)

        print(f"\n  总览：", file=sys.stderr, flush=True)
        for g, l in [("A", "trade+免费say"), ("B", "give+免费say")]:
            d = gd[g]
            print(f"    {l}: {d['msgs']}msg {d['gives']}give {d['trades_matched']}trade成功 "
                  f"{d['crafts']}craft {d['eats']}eat | 存活{d['alive']}/5 E={d['energy']}",
                  file=sys.stderr, flush=True)

        print(f"\n  API调用: {self.total_api_calls}", file=sys.stderr, flush=True)
        print("═" * 60, file=sys.stderr, flush=True)

        with open(Path(__file__).parent / "events.jsonl", "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    World().run()

import os
"""
Round 009 v2 — Tool-calling Agent vs Single-shot（闭环能量经济）

假设：在闭环能量经济中，tool-calling 多步 agent 的合作效率和能量积累
     显著优于单步 agent。多步使"协商+交换+合成"1-2 tick 完成，
     单步需 4-5 tick，被动消耗使合作对单步 agent 不经济。

对立：单步 agent 会形成"习惯链"，稳态效率趋同。

生态规则：
  - 被动消耗：2 能量/tick（活着就烧）
  - 动作成本：move=-1, say=-1, give=-1, harvest/eat/craft/look/done=0
  - eat：消耗背包1份资源 → +6 能量
  - craft：消耗背包2种不同资源 → +18 能量（超线性：18 > 6+6）
  - 起始能量 80，死亡线 0
  - 目标：30天后存活+能量最大化

设计：
  A组5人（tool-calling，每 tick 最多5步） vs B组5人（single-shot）
  同一世界左右分区，技能异质（5种各1人），中性 prompt
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

# ─── 配置 ─────────────────────────────────────────────────────────────────────

API_BASE = os.environ.get("IDEALAB_API_BASE", "https://api.openai.com/v1")
API_KEY = os.environ["IDEALAB_API_KEY"]
MODEL = "Qwen3.6-Plus-DogFooding"

GRID_W = 16
GRID_H = 8
MAX_TICKS = 15
MAX_TOOL_CALLS = 3   # A组每 tick 最多工具调用次数（move+harvest+eat/give 足够测试）
MAX_CONCURRENT = 6   # 并发数（所有 agent 同时跑）
VISION_RANGE = 5

# ─── 闭环能量参数 ──────────────────────────────────────────────────────────────

INITIAL_ENERGY = 80
PASSIVE_DRAIN = 2     # 每 tick 被动消耗
MOVE_COST = 1
SAY_COST = 1
GIVE_COST = 1
EAT_REWARD = 6        # 吃 1 份资源 → +6 能量
CRAFT_REWARD = 18     # 合成 2 种资源 → +18 能量（超线性激励合作）
REST_REWARD = 1       # 啥都不干 → +1

RESOURCES = ["谷物", "药草", "石料", "木材", "兽皮"]

# ─── Agent 定义 ────────────────────────────────────────────────────────────────

AGENTS_A = [
    {"name": "甲谷", "skill": "谷物"},
    {"name": "甲草", "skill": "药草"},
    {"name": "甲石", "skill": "石料"},
]

AGENTS_B = [
    {"name": "乙谷", "skill": "谷物"},
    {"name": "乙草", "skill": "药草"},
    {"name": "乙石", "skill": "石料"},
]

SYSTEM_PROMPT_A = (
    "你是{name}，活在一个消耗能量的世界里。\n"
    "规则：\n"
    "- 每天自动消耗2点能量（活着就烧）\n"
    "- 移动消耗1能量，说话消耗1能量，给东西消耗1能量\n"
    "- 你只会采集{skill}（免费），其他资源采不了\n"
    "- 吃掉1份资源 → 恢复6能量\n"
    "- 用2种不同资源合成 → 恢复18能量（远比分开吃划算！）\n"
    "- 能量归零你就死了\n"
    "目标：活下去，能量越多越好。跟别人交换资源来合成是最高效的策略。\n\n"
    "你可以使用以下工具（每天最多5次）：\n"
    "- look() — 查看环境（免费）\n"
    "- move(direction) — 移动(up/down/left/right)，消耗1能量\n"
    "- harvest() — 采集脚下的{skill}（免费）\n"
    "- eat() — 吃掉背包里1份资源，恢复6能量\n"
    "- give(target, resource) — 给视野内的人1份资源，消耗1能量\n"
    "- say(target, content) — 对视野内的人说话，消耗1能量\n"
    "- craft() — 用2种不同资源合成，恢复18能量\n"
    "- done() — 结束今天\n\n"
    "每次回复一个JSON：{{\"tool\":\"工具名\",\"args\":{{...}},\"thinking\":\"你在想什么\"}}\n"
    "示例：{{\"tool\":\"move\",\"args\":{{\"direction\":\"right\"}},\"thinking\":\"往右找资源\"}}\n"
    "示例：{{\"tool\":\"give\",\"args\":{{\"target\":\"甲草\",\"resource\":\"谷物\"}},\"thinking\":\"换药草来合成\"}}"
)

SYSTEM_PROMPT_B = (
    "你是{name}，活在一个消耗能量的世界里。\n"
    "规则：\n"
    "- 每天自动消耗2点能量（活着就烧）\n"
    "- 移动消耗1能量，说话消耗1能量，给东西消耗1能量\n"
    "- 你只会采集{skill}（免费），其他资源采不了\n"
    "- 吃掉1份资源 → 恢复6能量\n"
    "- 用2种不同资源合成 → 恢复18能量（远比分开吃划算！）\n"
    "- 能量归零你就死了\n"
    "目标：活下去，能量越多越好。跟别人交换资源来合成是最高效的策略。\n"
    "提示：如果你昨天在做一件事（比如走向某人交换），今天应该继续完成它。\n\n"
    "每天你只能做一个动作。\n"
    "可选：move(up/down/left/right) | harvest | eat | give 名字 资源 | say 名字 内容 | craft | rest\n"
    "回复JSON：{{\"action\":\"动作\",\"target\":\"目标\",\"content\":\"内容\",\"thought\":\"想法\"}}"
)

# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: int
    name: str
    x: int
    y: int
    group: str
    skill: str
    energy: int = INITIAL_ENERGY
    alive: bool = True
    backpack: dict = field(default_factory=dict)
    memory: list = field(default_factory=list)
    inbox: list = field(default_factory=list)
    # 统计
    messages_sent: int = 0
    gives_out: int = 0
    gives_in: int = 0
    crafted: int = 0
    harvested: int = 0
    eaten: int = 0
    energy_peak: int = INITIAL_ENERGY

@dataclass
class ResourceNode:
    x: int
    y: int
    kind: str

# ─── 世界 ─────────────────────────────────────────────────────────────────────

class World:
    def __init__(self):
        self.tick = 0
        self.agents: list[Agent] = []
        self.resources: list[ResourceNode] = []
        self.events: list[dict] = []
        self.dialogue: list[dict] = []
        self.stats = {"A": {"energy": [], "msgs": [], "gives": [], "crafts": [],
                            "eats": [], "harvests": [], "alive": []},
                      "B": {"energy": [], "msgs": [], "gives": [], "crafts": [],
                            "eats": [], "harvests": [], "alive": []}}
        self.total_api_calls = 0
        self.dlg_file = open(Path(__file__).parent / "dialogue.jsonl", "w", encoding="utf-8")

    def init(self):
        aid = 0
        half = GRID_W // 2
        # A组：agent 出生在各自资源旁（row 间距2，确保视野可达）
        for i, spec in enumerate(AGENTS_A):
            ax, ay = 2 + i * 2, 1 + i * 2
            self.agents.append(Agent(id=aid, name=spec["name"],
                x=ax, y=min(ay, GRID_H-1), group="A", skill=spec["skill"]))
            # 放一份资源在脚下
            self.resources.append(ResourceNode(ax, min(ay, GRID_H-1), spec["skill"]))
            aid += 1
        # B组：同理
        for i, spec in enumerate(AGENTS_B):
            bx, by = half + 2 + i * 2, 1 + i * 2
            self.agents.append(Agent(id=aid, name=spec["name"],
                x=bx, y=min(by, GRID_H-1), group="B", skill=spec["skill"]))
            self.resources.append(ResourceNode(bx, min(by, GRID_H-1), spec["skill"]))
            aid += 1
        # 额外资源：每半区每种 2 个随机分布
        active_skills = set(s["skill"] for s in AGENTS_A)
        for kind in active_skills:
            for _ in range(2):
                self.resources.append(ResourceNode(random.randint(0, half-1), random.randint(0, GRID_H-1), kind))
                self.resources.append(ResourceNode(random.randint(half, GRID_W-1), random.randint(0, GRID_H-1), kind))

    def record(self, etype, detail, aid=-1):
        self.events.append({"tick": self.tick, "type": etype, "agent": aid, "detail": detail})

    def get_env_description(self, agent):
        """环境描述——给 agent 足够信息做决策"""
        lines = []
        lines.append(f"═══ 第{self.tick+1}天 ═══")
        lines.append(f"能量：{agent.energy}（每天自动-{PASSIVE_DRAIN}）")

        # 背包
        bp = {k: v for k, v in agent.backpack.items() if v > 0}
        if bp:
            items = [f"{k}×{v}" for k, v in bp.items()]
            lines.append(f"背包：{'、'.join(items)}")
            kinds = [k for k, v in agent.backpack.items() if v > 0]
            if len(kinds) >= 2:
                lines.append("★ 你有2种资源，可以 craft 合成（+18能量）！")
            elif len(kinds) == 1:
                lines.append(f"（吃掉{kinds[0]}可得+6能量，但合成2种不同资源得+18，划算3倍）")
        else:
            lines.append("背包空。")

        lines.append(f"技能：只能采{agent.skill}。位置({agent.x},{agent.y})。")

        # 脚下和附近资源
        nearby_res = []
        for r in self.resources:
            dx, dy = r.x - agent.x, r.y - agent.y
            if abs(dx) <= VISION_RANGE and abs(dy) <= VISION_RANGE:
                if agent.skill == r.kind:
                    dist = abs(dx) + abs(dy)
                    if dist == 0:
                        nearby_res.insert(0, "★脚下有" + r.kind + "！可直接harvest")
                    else:
                        dirs = []
                        if dy < 0: dirs.append("上")
                        elif dy > 0: dirs.append("下")
                        if dx < 0: dirs.append("左")
                        elif dx > 0: dirs.append("右")
                        nearby_res.append(f"{''.join(dirs)}{dist}步有{r.kind}")
        if nearby_res:
            lines.append("可采资源：" + "；".join(nearby_res[:4]))
        else:
            lines.append("视野内没有你能采的资源，需要移动寻找。")

        # 附近的人
        nearby_ppl = []
        for o in self.agents:
            if o.id != agent.id and o.alive and o.group == agent.group:
                dx, dy = abs(o.x - agent.x), abs(o.y - agent.y)
                if dx <= VISION_RANGE and dy <= VISION_RANGE:
                    dist = dx + dy
                    obp = {k: v for k, v in o.backpack.items() if v > 0}
                    bp_str = f"(有{'、'.join(f'{k}×{v}' for k,v in obp.items())})" if obp else "(背包空)"
                    nearby_ppl.append(f"{o.name}[采{o.skill}]{bp_str} 距{dist}步")
        if nearby_ppl:
            lines.append("视野内的人：" + "；".join(nearby_ppl))
        else:
            lines.append("视野内没人。")

        # 收件箱
        if agent.inbox:
            lines.append("收到消息：")
            for m in agent.inbox[-3:]:
                lines.append(f"  「{m}」")
            agent.inbox.clear()

        # 简短记忆
        if agent.memory:
            lines.append("近期记忆：" + "；".join(agent.memory[-3:]))

        return "\n".join(lines)

    def execute_tool(self, agent, tool_name, tool_args):
        """执行工具，返回结果。所有能量变动在这里发生。"""
        result = ""

        if tool_name == "look":
            result = self.get_env_description(agent)

        elif tool_name == "move":
            direction = tool_args.get("direction", "").lower()
            dx, dy = 0, 0
            if direction in ("up", "上"): dy = -1
            elif direction in ("down", "下"): dy = 1
            elif direction in ("left", "左"): dx = -1
            elif direction in ("right", "右"): dx = 1
            else:
                return "移动失败——方向不对（用up/down/left/right）。"
            agent.energy -= MOVE_COST
            half = GRID_W // 2
            nx, ny = agent.x + dx, agent.y + dy
            if agent.group == "A":
                nx = max(0, min(half-1, nx))
            else:
                nx = max(half, min(GRID_W-1, nx))
            ny = max(0, min(GRID_H-1, ny))
            agent.x, agent.y = nx, ny
            # 检查脚下
            foot_res = [r for r in self.resources if r.x == agent.x and r.y == agent.y and r.kind == agent.skill]
            foot_str = f" 脚下发现{agent.skill}可harvest！" if foot_res else ""
            result = f"移到({agent.x},{agent.y})，消耗1能量（剩{agent.energy}）。{foot_str}"

        elif tool_name == "harvest":
            found = False
            for r in self.resources[:]:
                if r.x == agent.x and r.y == agent.y and r.kind == agent.skill:
                    self.resources.remove(r)
                    agent.backpack[r.kind] = agent.backpack.get(r.kind, 0) + 1
                    agent.harvested += 1
                    self.record("harvest", f"{agent.name}采{r.kind}", agent.id)
                    bp = {k: v for k, v in agent.backpack.items() if v > 0}
                    kinds = list(bp.keys())
                    hint = ""
                    if len(kinds) >= 2:
                        hint = " ★可以craft了！"
                    result = f"采到1份{r.kind}！背包：{bp}。{hint}"
                    found = True
                    break
            if not found:
                result = "采集失败——脚下没有你能采的资源。用move走到资源位置再试。"

        elif tool_name == "eat":
            # 吃背包里任意1份资源
            res_name = tool_args.get("resource", "")
            if not res_name:
                # 自动选一个
                for k, v in agent.backpack.items():
                    if v > 0:
                        res_name = k
                        break
            if not res_name or agent.backpack.get(res_name, 0) <= 0:
                result = "吃东西失败——背包里没有资源。先去harvest。"
            else:
                agent.backpack[res_name] -= 1
                agent.energy += EAT_REWARD
                agent.eaten += 1
                agent.energy_peak = max(agent.energy_peak, agent.energy)
                self.record("eat", f"{agent.name}吃{res_name}(+{EAT_REWARD})", agent.id)
                bp = {k: v for k, v in agent.backpack.items() if v > 0}
                result = f"吃了1份{res_name}，能量+{EAT_REWARD}（现{agent.energy}）。背包：{bp}"

        elif tool_name == "give":
            target_name = tool_args.get("target", "")
            res_name = tool_args.get("resource", "")
            receiver = None
            for o in self.agents:
                if o.alive and o.group == agent.group and o.id != agent.id:
                    if target_name in o.name or o.name in target_name:
                        if abs(o.x - agent.x) <= VISION_RANGE and abs(o.y - agent.y) <= VISION_RANGE:
                            receiver = o
                            break
            if not receiver:
                result = f"给东西失败——{target_name}不在视野内。"
            elif not res_name or agent.backpack.get(res_name, 0) <= 0:
                result = f"给东西失败——你没有{res_name}。"
            else:
                agent.energy -= GIVE_COST
                agent.backpack[res_name] -= 1
                receiver.backpack[res_name] = receiver.backpack.get(res_name, 0) + 1
                agent.gives_out += 1
                receiver.gives_in += 1
                agent.memory.append(f"第{self.tick+1}天给{receiver.name}{res_name}")
                receiver.memory.append(f"第{self.tick+1}天收到{agent.name}的{res_name}")
                receiver.inbox.append(f"{agent.name}给了你1份{res_name}")
                self.record("give", f"{agent.name}→{receiver.name}:{res_name}", agent.id)
                result = f"给了{receiver.name}1份{res_name}，消耗1能量（剩{agent.energy}）。"

        elif tool_name == "say":
            target_name = tool_args.get("target", "")
            content = tool_args.get("content", "")[:80]
            receiver = None
            for o in self.agents:
                if o.alive and o.group == agent.group and o.id != agent.id:
                    if target_name in o.name or o.name in target_name:
                        if abs(o.x - agent.x) <= VISION_RANGE and abs(o.y - agent.y) <= VISION_RANGE:
                            receiver = o
                            break
            if receiver:
                agent.energy -= SAY_COST
                receiver.inbox.append(f"{agent.name}：{content}")
                agent.messages_sent += 1
                self.record("message", f"{agent.name}→{receiver.name}：{content}", agent.id)
                result = f"对{receiver.name}说了话，消耗1能量（剩{agent.energy}）。"
            else:
                result = f"说话失败——{target_name}不在视野内。"

        elif tool_name == "craft":
            kinds = [k for k, v in agent.backpack.items() if v > 0]
            if len(kinds) >= 2:
                used = kinds[:2]
                for k in used:
                    agent.backpack[k] -= 1
                agent.energy += CRAFT_REWARD
                agent.crafted += 1
                agent.energy_peak = max(agent.energy_peak, agent.energy)
                self.record("craft", f"{agent.name}合成({'+'.join(used)})→+{CRAFT_REWARD}", agent.id)
                bp = {k: v for k, v in agent.backpack.items() if v > 0}
                result = f"合成成功！{used[0]}+{used[1]}→能量+{CRAFT_REWARD}（现{agent.energy}）。背包：{bp}"
            else:
                result = f"合成失败——需要2种不同资源。你只有{kinds}。去找别人换！"

        elif tool_name == "rest":
            agent.energy += REST_REWARD
            result = f"休息，能量+{REST_REWARD}（现{agent.energy}）。"

        elif tool_name == "done":
            result = "__DONE__"

        else:
            result = f"未知工具：{tool_name}"

        return result

    # ─── A组：Tool-calling agent ──────────────────────────────────────────────

    def run_agent_toolcall(self, agent):
        """A组：多轮工具调用，每 tick 最多 MAX_TOOL_CALLS 步"""
        sys_prompt = SYSTEM_PROMPT_A.format(name=agent.name, skill=agent.skill)

        # 初始环境
        env = self.get_env_description(agent)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"新的一天。当前状态：\n{env}\n\n决定今天做什么（最多5步）。"},
        ]

        all_actions = []
        all_thoughts = []

        for step in range(MAX_TOOL_CALLS):
            if agent.energy <= 0:
                break

            self.total_api_calls += 1
            try:
                resp = httpx.post(
                    f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json={"model": MODEL, "messages": messages, "temperature": 0.7, "max_tokens": 150},
                    timeout=60.0,
                )
                data = resp.json()
                if "choices" not in data:
                    break
                content = data["choices"][0]["message"]["content"].strip()
                # 清理推理标签
                if "<think>" in content:
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                if "```" in content:
                    m = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                    if m:
                        content = m.group(1).strip()
                m = re.search(r'\{.*\}', content, re.DOTALL)
                if not m:
                    break
                parsed = json.loads(m.group())
            except:
                break

            tool = parsed.get("tool", "done")
            args = parsed.get("args", {})
            thinking = parsed.get("thinking", "")
            all_thoughts.append(thinking)
            all_actions.append(f"{tool}({json.dumps(args, ensure_ascii=False)})")

            # 执行
            result = self.execute_tool(agent, tool, args)
            if result == "__DONE__":
                break

            # 反馈给 LLM
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"[结果] {result}\n（还剩{MAX_TOOL_CALLS-step-1}步可用）"})

        # 记录
        entry = {
            "day": self.tick + 1, "name": agent.name, "group": "A",
            "energy": agent.energy, "backpack": {k: v for k, v in agent.backpack.items() if v > 0},
            "steps": all_actions, "thoughts": all_thoughts,
        }
        self.dialogue.append(entry)
        self.dlg_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.dlg_file.flush()

    # ─── B组：Single-shot ─────────────────────────────────────────────────────

    def run_agent_singleshot(self, agent):
        """B组：每 tick 1 个动作"""
        sys_prompt = SYSTEM_PROMPT_B.format(name=agent.name, skill=agent.skill)
        env = self.get_env_description(agent)

        self.total_api_calls += 1
        try:
            resp = httpx.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": env},
                    ],
                    "temperature": 0.85, "max_tokens": 150,
                },
                timeout=60.0,
            )
            data = resp.json()
            if "choices" not in data:
                return
            content = data["choices"][0]["message"]["content"].strip()
            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if "```" in content:
                m2 = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if m2:
                    content = m2.group(1).strip()
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if not m:
                return
            parsed = json.loads(m.group())
        except:
            return

        act = parsed.get("action", "rest").lower().strip()
        target = parsed.get("target", "").strip()
        cont = parsed.get("content", "").strip()
        thought = parsed.get("thought", "")

        # 映射到工具执行
        if act == "move":
            self.execute_tool(agent, "move", {"direction": target or cont})
        elif act == "harvest":
            self.execute_tool(agent, "harvest", {})
        elif act == "eat":
            res = ""
            for r in RESOURCES:
                if r in target or r in cont:
                    res = r
                    break
            self.execute_tool(agent, "eat", {"resource": res})
        elif act == "craft":
            self.execute_tool(agent, "craft", {})
        elif act == "give":
            res = ""
            person = target
            for r in RESOURCES:
                if r in cont:
                    res = r
                    break
                if r in target:
                    res = r
                    person = target.replace(r, "").strip()
            if not res:
                for k, v in agent.backpack.items():
                    if v > 0:
                        res = k
                        break
            self.execute_tool(agent, "give", {"target": person or target, "resource": res})
        elif act == "say":
            self.execute_tool(agent, "say", {"target": target, "content": cont})
        elif act == "rest":
            self.execute_tool(agent, "rest", {})
        else:
            self.execute_tool(agent, "rest", {})

        entry = {
            "day": self.tick + 1, "name": agent.name, "group": "B",
            "energy": agent.energy, "backpack": {k: v for k, v in agent.backpack.items() if v > 0},
            "steps": [f"{act}({target} {cont})".strip()], "thoughts": [thought],
        }
        self.dialogue.append(entry)
        self.dlg_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.dlg_file.flush()

    # ─── 主循环 ───────────────────────────────────────────────────────────────

    def run_tick(self):
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return False

        a_agents = [a for a in alive if a.group == "A"]
        b_agents = [a for a in alive if a.group == "B"]

        # 并行执行
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            futures = []
            for a in a_agents:
                futures.append(pool.submit(self.run_agent_toolcall, a))
            for a in b_agents:
                futures.append(pool.submit(self.run_agent_singleshot, a))
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"  [err] {e}", file=sys.stderr, flush=True)

        # 被动能量消耗
        for a in self.agents:
            if a.alive:
                a.energy -= PASSIVE_DRAIN
                if a.energy <= 0:
                    a.alive = False
                    self.record("death", f"{a.name}({a.group})能量耗尽", a.id)

        # 资源再生（每种每半区 2 个，只再生 agent 能采的种类）
        half = GRID_W // 2
        active_skills = set(s["skill"] for s in AGENTS_A)
        for kind in active_skills:
            for _ in range(2):
                self.resources.append(ResourceNode(random.randint(0, half-1), random.randint(0, GRID_H-1), kind))
                self.resources.append(ResourceNode(random.randint(half, GRID_W-1), random.randint(0, GRID_H-1), kind))

        # 统计
        for g in ["A", "B"]:
            gids = {a.id for a in self.agents if a.group == g}
            msgs = sum(1 for e in self.events if e["tick"] == self.tick and e["type"] == "message" and e["agent"] in gids)
            gives = sum(1 for e in self.events if e["tick"] == self.tick and e["type"] == "give" and e["agent"] in gids)
            crafts = sum(1 for e in self.events if e["tick"] == self.tick and e["type"] == "craft" and e["agent"] in gids)
            eats = sum(1 for e in self.events if e["tick"] == self.tick and e["type"] == "eat" and e["agent"] in gids)
            harvests = sum(1 for e in self.events if e["tick"] == self.tick and e["type"] == "harvest" and e["agent"] in gids)
            alive_n = sum(1 for a in self.agents if a.group == g and a.alive)
            total_e = sum(a.energy for a in self.agents if a.group == g and a.alive)
            self.stats[g]["energy"].append(total_e)
            self.stats[g]["msgs"].append(msgs)
            self.stats[g]["gives"].append(gives)
            self.stats[g]["crafts"].append(crafts)
            self.stats[g]["eats"].append(eats)
            self.stats[g]["harvests"].append(harvests)
            self.stats[g]["alive"].append(alive_n)

        return any(a.alive for a in self.agents)

    def run(self):
        self.init()
        print("═══ Round 009 v2: Tool-calling vs Single-shot（闭环能量经济）═══", file=sys.stderr, flush=True)
        print(f"  A组3人(tool-call,3步/tick) vs B组3人(single-shot,1步/tick)", file=sys.stderr, flush=True)
        print(f"  被动消耗{PASSIVE_DRAIN}/tick | eat+{EAT_REWARD} | craft+{CRAFT_REWARD} | 起始{INITIAL_ENERGY}", file=sys.stderr, flush=True)
        print(file=sys.stderr, flush=True)

        for tick in range(MAX_TICKS):
            self.tick = tick
            t0 = time.time()
            cont = self.run_tick()
            elapsed = time.time() - t0

            a_alive = self.stats["A"]["alive"][-1]
            b_alive = self.stats["B"]["alive"][-1]
            a_energy = self.stats["A"]["energy"][-1]
            b_energy = self.stats["B"]["energy"][-1]
            a_crafts = self.stats["A"]["crafts"][-1]
            b_crafts = self.stats["B"]["crafts"][-1]
            a_gives = self.stats["A"]["gives"][-1]
            b_gives = self.stats["B"]["gives"][-1]

            print(f"  第{tick+1:>2}天 | A:{a_alive}人 E={a_energy:>3} craft={a_crafts} give={a_gives} | "
                  f"B:{b_alive}人 E={b_energy:>3} craft={b_crafts} give={b_gives} | {elapsed:.0f}s",
                  file=sys.stderr, flush=True)

            # 本 tick 关键事件
            key_events = [e for e in self.events if e["tick"] == self.tick and e["type"] in ("give", "craft")]
            for e in key_events[:4]:
                print(f"        ✅ {e['detail']}", file=sys.stderr, flush=True)

            if not cont:
                print("  *** 全灭 ***", file=sys.stderr, flush=True)
                break

        self.dlg_file.close()
        self.output()

    def output(self):
        def welch_t(a, b):
            na, nb = len(a), len(b)
            if na < 2 or nb < 2:
                return 0, 1.0
            ma, mb = sum(a) / na, sum(b) / nb
            va = sum((x - ma) ** 2 for x in a) / (na - 1)
            vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
            d = va / na + vb / nb
            if d <= 0:
                return 0, 1.0
            t = (ma - mb) / math.sqrt(d)
            p = math.erfc(abs(t) / math.sqrt(2))
            return round(t, 3), round(p, 4)

        # ─── 核心指标：合作效率（控制行动带宽混淆） ───────────────────────────
        # craft_rate = crafts / harvests — 资源合作转化率
        # cooperation_energy_fraction = (crafts×18) / (crafts×18 + eats×6) — 能量来源中合作占比
        # 这些比率指标不受"每tick行动次数"影响

        groups_data = {}
        for g in ["A", "B"]:
            ga = [a for a in self.agents if a.group == g]
            total_crafts = sum(a.crafted for a in ga)
            total_eats = sum(a.eaten for a in ga)
            total_harvests = sum(a.harvested for a in ga)
            total_gives = sum(a.gives_out for a in ga)
            craft_energy = total_crafts * CRAFT_REWARD
            eat_energy = total_eats * EAT_REWARD
            total_gained = craft_energy + eat_energy

            # 合作效率比
            craft_rate = round(total_crafts / max(1, total_harvests), 3)
            coop_energy_frac = round(craft_energy / max(1, total_gained), 3)

            # 前后半期分析（检验趋势收敛）
            half = (self.tick + 1) // 2
            first_half_crafts = sum(self.stats[g]["crafts"][:half])
            second_half_crafts = sum(self.stats[g]["crafts"][half:])
            first_half_harvests = sum(self.stats[g]["harvests"][:half])
            second_half_harvests = sum(self.stats[g]["harvests"][half:])
            first_craft_rate = round(first_half_crafts / max(1, first_half_harvests), 3)
            second_craft_rate = round(second_half_crafts / max(1, second_half_harvests), 3)

            groups_data[g] = {
                "total_energy_final": sum(a.energy for a in ga if a.alive),
                "avg_energy_final": round(sum(a.energy for a in ga if a.alive) / max(1, sum(1 for a in ga if a.alive)), 1),
                "total_crafts": total_crafts,
                "total_gives": total_gives,
                "total_eats": total_eats,
                "total_harvests": total_harvests,
                "total_msgs": sum(a.messages_sent for a in ga),
                "final_alive": sum(1 for a in ga if a.alive),
                "energy_peak_avg": round(sum(a.energy_peak for a in ga) / len(ga), 1),
                # 核心合作指标
                "craft_rate": craft_rate,
                "coop_energy_fraction": coop_energy_frac,
                "first_half_craft_rate": first_craft_rate,
                "second_half_craft_rate": second_craft_rate,
            }

        # 统计检验——用合作效率比而非绝对数量
        # craft_rate per tick = crafts[t] / max(1, harvests[t])
        a_craft_rates = [c / max(1, h) for c, h in zip(self.stats["A"]["crafts"], self.stats["A"]["harvests"])]
        b_craft_rates = [c / max(1, h) for c, h in zip(self.stats["B"]["crafts"], self.stats["B"]["harvests"])]
        t_craftrate, p_craftrate = welch_t(a_craft_rates, b_craft_rates)

        # 绝对指标（已知有行动带宽偏差，作参考）
        t_energy, p_energy = welch_t(self.stats["A"]["energy"], self.stats["B"]["energy"])
        t_craft, p_craft = welch_t(self.stats["A"]["crafts"], self.stats["B"]["crafts"])
        t_give, p_give = welch_t(self.stats["A"]["gives"], self.stats["B"]["gives"])

        result = {
            "experiment": "round-009-v2: tool-calling vs single-shot (closed-loop energy economy)",
            "ticks_completed": self.tick + 1,
            "total_api_calls": self.total_api_calls,
            "parameters": {
                "initial_energy": INITIAL_ENERGY,
                "passive_drain": PASSIVE_DRAIN,
                "eat_reward": EAT_REWARD,
                "craft_reward": CRAFT_REWARD,
                "move_cost": MOVE_COST,
                "max_tool_calls": MAX_TOOL_CALLS,
            },
            "design_note": "A组每tick 5步 vs B组1步。绝对能量/产出指标有行动带宽混淆。"
                           "核心检验指标是 craft_rate（合作转化率）和 coop_energy_fraction（合作能量占比），"
                           "这些比率控制了行动次数差异。",
            "groups": groups_data,
            "per_agent": [],
            "tests": {
                "craft_rate（核心）": {"t": t_craftrate, "p": p_craftrate, "sig": p_craftrate < 0.05,
                                    "desc": "craft/harvest比率，控制行动带宽"},
                "energy_per_tick（参考）": {"t": t_energy, "p": p_energy, "sig": p_energy < 0.05,
                                          "desc": "有行动带宽偏差"},
                "crafts_absolute（参考）": {"t": t_craft, "p": p_craft, "sig": p_craft < 0.05},
                "gives（参考）": {"t": t_give, "p": p_give, "sig": p_give < 0.05},
            },
            "convergence": {
                "A_first_half_craft_rate": groups_data["A"]["first_half_craft_rate"],
                "A_second_half_craft_rate": groups_data["A"]["second_half_craft_rate"],
                "B_first_half_craft_rate": groups_data["B"]["first_half_craft_rate"],
                "B_second_half_craft_rate": groups_data["B"]["second_half_craft_rate"],
                "interpretation": "如果B后半期追上A后半期→支持对立假设（习惯链形成）",
            },
            "per_tick": {g: {"energy": self.stats[g]["energy"],
                            "crafts": self.stats[g]["crafts"],
                            "gives": self.stats[g]["gives"],
                            "harvests": self.stats[g]["harvests"]}
                        for g in ["A", "B"]},
        }

        for a in self.agents:
            h = a.harvested
            c = a.crafted
            e = a.eaten
            result["per_agent"].append({
                "name": a.name, "group": a.group, "alive": a.alive,
                "energy": a.energy, "harvested": h, "eaten": e,
                "crafted": c, "gives_out": a.gives_out, "gives_in": a.gives_in,
                "msgs": a.messages_sent, "energy_peak": a.energy_peak,
                "personal_craft_rate": round(c / max(1, h), 3),
            })

        with open(Path(__file__).parent / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

        # stderr 摘要
        print("\n" + "═" * 60, file=sys.stderr, flush=True)
        print("  【核心指标：合作效率比（控制行动带宽）】", file=sys.stderr, flush=True)
        for g, label in [("A", "Tool-calling(5步)"), ("B", "Single-shot(1步)")]:
            d = groups_data[g]
            print(f"  {label}: craft_rate={d['craft_rate']} | 合作能量占比={d['coop_energy_fraction']}",
                  file=sys.stderr, flush=True)
            print(f"    前半期craft率={d['first_half_craft_rate']} → 后半期={d['second_half_craft_rate']}",
                  file=sys.stderr, flush=True)
        print(f"\n  craft_rate 检验: t={t_craftrate} p={p_craftrate} "
              f"{'✓显著' if p_craftrate<0.05 else '✗不显著'}", file=sys.stderr, flush=True)

        print(f"\n  【绝对指标（有行动带宽偏差，仅参考）】", file=sys.stderr, flush=True)
        for g, label in [("A", "A(5步)"), ("B", "B(1步)")]:
            d = groups_data[g]
            print(f"  {label}: 存活{d['final_alive']}/3 | E={d['total_energy_final']} | "
                  f"{d['total_crafts']}合成 {d['total_gives']}给 {d['total_eats']}吃 {d['total_harvests']}采",
                  file=sys.stderr, flush=True)
        print(f"  API调用: {self.total_api_calls}", file=sys.stderr, flush=True)
        print("═" * 60, file=sys.stderr, flush=True)

        # 保存事件日志
        with open(Path(__file__).parent / "events.jsonl", "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    w = World()
    w.run()

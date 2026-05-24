import os
"""
Round 010 — 完美主义陷阱测试

假设：LLM 的合作失败源于"完美主义陷阱"。给予降级策略提示后，
     进食率和合作率都会大幅提升。
对立：降级提示无效（被忽略）或有害（只吃不合作）。

设计：
  A组4人：有降级提示（"连续2天只有一种资源就先吃掉保命"）
  B组4人：无降级提示（同 Round 9 的 prompt）
  - 技能异质（4种各1人），同一世界左右分区
  - 闭环能量经济：eat+6, craft+18, 被动-2, 动作-1
  - 单步（每 tick 1 动作），20天
  - 紧凑布局（人之间距2步，视野内互见）
"""

import json, random, time, re, sys, math
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

API_BASE = os.environ.get("IDEALAB_API_BASE", "https://api.openai.com/v1")
API_KEY = os.environ["IDEALAB_API_KEY"]
MODEL = "qwen3-coder-plus"

GRID_W, GRID_H = 12, 8
MAX_TICKS = 20
MAX_CONCURRENT = 8
VISION_RANGE = 5
INITIAL_ENERGY = 80
PASSIVE_DRAIN = 2
MOVE_COST = 1
SAY_COST = 1
GIVE_COST = 1
EAT_REWARD = 6
CRAFT_REWARD = 18
REST_REWARD = 1
RESOURCES = ["谷物", "药草", "石料", "木材"]

AGENTS_A = [{"name": "甲谷", "skill": "谷物"}, {"name": "甲草", "skill": "药草"},
            {"name": "甲石", "skill": "石料"}, {"name": "甲木", "skill": "木材"}]
AGENTS_B = [{"name": "乙谷", "skill": "谷物"}, {"name": "乙草", "skill": "药草"},
            {"name": "乙石", "skill": "石料"}, {"name": "乙木", "skill": "木材"}]

PROMPT_WITH_FALLBACK = (
    "你是{name}，活在一个消耗能量的世界里。\n"
    "规则：每天自动-2能量 | 移动/说话/给东西各-1 | 只会采{skill}（免费）| "
    "吃1份资源→+6能量 | 2种不同资源合成→+18能量 | 能量归零=死\n\n"
    "生存策略：\n"
    "- 最优：找人换资源凑两种再合成（+18）\n"
    "- 保底：如果背包里有资源但连续2天没找到交换对象，先吃掉保命（+6）。活着才有机会合作\n"
    "- 别空等——有东西就先吃，合成机会以后还有\n\n"
    "每天做一个动作：move(up/down/left/right) | harvest | eat | give 名字 资源 | say 名字 内容 | craft | rest\n"
    "回复JSON：{{\"action\":\"动作\",\"target\":\"目标\",\"content\":\"内容\",\"thought\":\"想法\"}}"
)

PROMPT_NO_FALLBACK = (
    "你是{name}，活在一个消耗能量的世界里。\n"
    "规则：每天自动-2能量 | 移动/说话/给东西各-1 | 只会采{skill}（免费）| "
    "吃1份资源→+6能量 | 2种不同资源合成→+18能量 | 能量归零=死\n\n"
    "目标：活下去，能量越多越好。跟别人交换资源来合成是最高效的策略。\n\n"
    "每天做一个动作：move(up/down/left/right) | harvest | eat | give 名字 资源 | say 名字 内容 | craft | rest\n"
    "回复JSON：{{\"action\":\"动作\",\"target\":\"目标\",\"content\":\"内容\",\"thought\":\"想法\"}}"
)

@dataclass
class Agent:
    id: int; name: str; x: int; y: int; group: str; skill: str
    energy: int = INITIAL_ENERGY; alive: bool = True
    backpack: dict = field(default_factory=dict)
    memory: list = field(default_factory=list)
    inbox: list = field(default_factory=list)
    messages_sent: int = 0; gives_out: int = 0; gives_in: int = 0
    crafted: int = 0; harvested: int = 0; eaten: int = 0

@dataclass
class ResourceNode:
    x: int; y: int; kind: str

class World:
    def __init__(self):
        self.tick = 0; self.agents = []; self.resources = []; self.events = []; self.dialogue = []
        self.stats = {g: {"energy":[],"msgs":[],"gives":[],"crafts":[],"eats":[],"harvests":[],"alive":[]} for g in ["A","B"]}
        self.total_api_calls = 0
        self.dlg_file = open(Path(__file__).parent / "dialogue.jsonl", "w", encoding="utf-8")

    def init(self):
        aid, half = 0, GRID_W // 2
        # A组紧凑排列（视野内互见）
        for i, (spec, pos) in enumerate(zip(AGENTS_A, [(1,1),(3,1),(1,3),(3,3)])):
            self.agents.append(Agent(id=aid, name=spec["name"], x=pos[0], y=pos[1], group="A", skill=spec["skill"]))
            self.resources.append(ResourceNode(pos[0], pos[1], spec["skill"]))
            aid += 1
        for i, (spec, pos) in enumerate(zip(AGENTS_B, [(half+1,1),(half+3,1),(half+1,3),(half+3,3)])):
            self.agents.append(Agent(id=aid, name=spec["name"], x=pos[0], y=pos[1], group="B", skill=spec["skill"]))
            self.resources.append(ResourceNode(pos[0], pos[1], spec["skill"]))
            aid += 1
        for kind in RESOURCES:
            for _ in range(2):
                self.resources.append(ResourceNode(random.randint(0, half-1), random.randint(0, GRID_H-1), kind))
                self.resources.append(ResourceNode(random.randint(half, GRID_W-1), random.randint(0, GRID_H-1), kind))

    def record(self, etype, detail, aid=-1):
        self.events.append({"tick": self.tick, "type": etype, "agent": aid, "detail": detail})

    def get_env(self, agent):
        lines = [f"第{self.tick+1}天 | 能量{agent.energy}（每天-{PASSIVE_DRAIN}）"]
        bp = {k:v for k,v in agent.backpack.items() if v>0}
        if bp:
            lines.append(f"背包：{'、'.join(f'{k}x{v}' for k,v in bp.items())}")
            if len(bp) >= 2: lines.append("★可以craft！")
        else: lines.append("背包空")
        lines.append(f"技能：采{agent.skill} | 位置({agent.x},{agent.y})")
        # 资源
        foot = [r for r in self.resources if r.x==agent.x and r.y==agent.y and r.kind==agent.skill]
        if foot: lines.append(f"★脚下有{agent.skill}！")
        else:
            near = []
            for r in self.resources:
                if r.kind != agent.skill: continue
                dx,dy = r.x-agent.x, r.y-agent.y
                if abs(dx)<=VISION_RANGE and abs(dy)<=VISION_RANGE and (dx or dy):
                    d = []
                    if dy<0: d.append("上")
                    elif dy>0: d.append("下")
                    if dx<0: d.append("左")
                    elif dx>0: d.append("右")
                    near.append(f"{''.join(d)}{abs(dx)+abs(dy)}步")
            if near: lines.append(f"{agent.skill}在：" + "；".join(near[:2]))
        # 人
        ppl = []
        for o in self.agents:
            if o.id==agent.id or not o.alive or o.group!=agent.group: continue
            dx,dy = abs(o.x-agent.x), abs(o.y-agent.y)
            if dx<=VISION_RANGE and dy<=VISION_RANGE:
                obp = [k for k,v in o.backpack.items() if v>0]
                ppl.append(f"{o.name}[{o.skill}]({'有'+','.join(obp) if obp else '空'})距{dx+dy}")
        if ppl: lines.append("人：" + "；".join(ppl))
        if agent.inbox:
            lines.append("消息：" + " | ".join(agent.inbox[-2:]))
            agent.inbox.clear()
        if agent.memory: lines.append("记：" + "；".join(agent.memory[-2:]))
        return "\n".join(lines)

    def execute(self, agent, act, target, content):
        half = GRID_W // 2
        if act == "move":
            d = (target or content or "").lower()
            dx,dy = 0,0
            if "up" in d or "上" in d: dy=-1
            elif "down" in d or "下" in d: dy=1
            elif "left" in d or "左" in d: dx=-1
            elif "right" in d or "右" in d: dx=1
            if dx or dy:
                agent.energy -= MOVE_COST
                nx,ny = agent.x+dx, agent.y+dy
                if agent.group=="A": nx=max(0,min(half-1,nx))
                else: nx=max(half,min(GRID_W-1,nx))
                agent.x, agent.y = nx, max(0,min(GRID_H-1,ny))
        elif act == "harvest":
            for r in self.resources[:]:
                if r.x==agent.x and r.y==agent.y and r.kind==agent.skill:
                    self.resources.remove(r); agent.backpack[r.kind]=agent.backpack.get(r.kind,0)+1
                    agent.harvested+=1; self.record("harvest",f"{agent.name}采{r.kind}",agent.id); break
        elif act == "eat":
            rn = ""
            for r in RESOURCES:
                if r in (target or "") or r in (content or ""): rn=r; break
            if not rn:
                for k,v in agent.backpack.items():
                    if v>0: rn=k; break
            if rn and agent.backpack.get(rn,0)>0:
                agent.backpack[rn]-=1; agent.energy+=EAT_REWARD; agent.eaten+=1
                self.record("eat",f"{agent.name}吃{rn}(+{EAT_REWARD})",agent.id)
        elif act == "craft":
            kinds=[k for k,v in agent.backpack.items() if v>0]
            if len(kinds)>=2:
                for k in kinds[:2]: agent.backpack[k]-=1
                agent.energy+=CRAFT_REWARD; agent.crafted+=1
                self.record("craft",f"{agent.name}合成({'+'.join(kinds[:2])})→+{CRAFT_REWARD}",agent.id)
        elif act == "give":
            rn,pn = "",target or ""
            for r in RESOURCES:
                if r in (content or ""): rn=r; break
                if r in pn: rn=r; pn=pn.replace(r,"").strip()
            if not rn:
                for k,v in agent.backpack.items():
                    if v>0: rn=k; break
            recv = None
            for o in self.agents:
                if o.alive and o.group==agent.group and o.id!=agent.id:
                    if pn and (pn in o.name or o.name in pn):
                        if abs(o.x-agent.x)<=VISION_RANGE and abs(o.y-agent.y)<=VISION_RANGE:
                            recv=o; break
            if recv and rn and agent.backpack.get(rn,0)>0:
                agent.energy-=GIVE_COST; agent.backpack[rn]-=1
                recv.backpack[rn]=recv.backpack.get(rn,0)+1
                agent.gives_out+=1; recv.gives_in+=1
                agent.memory.append(f"给{recv.name}{rn}")
                recv.memory.append(f"收{agent.name}的{rn}")
                recv.inbox.append(f"{agent.name}给了你{rn}")
                self.record("give",f"{agent.name}→{recv.name}:{rn}",agent.id)
        elif act == "say":
            pn,msg = target or "",(content or "")[:50]
            recv = None
            for o in self.agents:
                if o.alive and o.group==agent.group and o.id!=agent.id:
                    if pn and (pn in o.name or o.name in pn):
                        if abs(o.x-agent.x)<=VISION_RANGE and abs(o.y-agent.y)<=VISION_RANGE:
                            recv=o; break
            if recv and msg:
                agent.energy-=SAY_COST; recv.inbox.append(f"{agent.name}：{msg}")
                agent.messages_sent+=1; self.record("message",f"{agent.name}→{recv.name}：{msg}",agent.id)
        else: agent.energy+=REST_REWARD

    def run_agent(self, agent):
        sys_prompt = (PROMPT_WITH_FALLBACK if agent.group=="A" else PROMPT_NO_FALLBACK).format(name=agent.name, skill=agent.skill)
        env = self.get_env(agent)
        self.total_api_calls += 1
        try:
            resp = httpx.post(f"{API_BASE}/chat/completions",
                headers={"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json"},
                json={"model":MODEL,"messages":[{"role":"system","content":sys_prompt},{"role":"user","content":env}],
                      "temperature":0.7,"max_tokens":300}, timeout=60.0)
            data = resp.json()
            if "choices" not in data: return
            raw = data["choices"][0]["message"]["content"].strip()
            if "<think>" in raw: raw = re.sub(r'<think>.*?</think>','',raw,flags=re.DOTALL).strip()
            if "```" in raw:
                m=re.search(r'```(?:json)?\s*(.*?)```',raw,re.DOTALL)
                if m: raw=m.group(1).strip()
            m=re.search(r'\{.*\}',raw,re.DOTALL)
            if not m: return
            parsed=json.loads(m.group())
        except: return
        act=parsed.get("action","rest").lower().strip()
        act_map={"采":"harvest","采集":"harvest","吃":"eat","进食":"eat","移动":"move",
                 "给":"give","赠送":"give","说":"say","说话":"say","合成":"craft","休息":"rest",
                 "走":"move","交给":"give","送":"give"}
        act=act_map.get(act, act)
        target=parsed.get("target","").strip()
        content=parsed.get("content","").strip()
        thought=parsed.get("thought","")
        self.execute(agent, act, target, content)
        entry={"day":self.tick+1,"name":agent.name,"group":agent.group,"energy":agent.energy,
               "backpack":{k:v for k,v in agent.backpack.items() if v>0},"action":act,"target":target,"thought":thought}
        self.dialogue.append(entry)
        self.dlg_file.write(json.dumps(entry,ensure_ascii=False)+"\n"); self.dlg_file.flush()

    def run_tick(self):
        alive=[a for a in self.agents if a.alive]
        if not alive: return False
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            futs=[pool.submit(self.run_agent,a) for a in alive]
            for f in as_completed(futs):
                try: f.result()
                except Exception as e: print(f"  [err]{e}",file=sys.stderr,flush=True)
        for a in self.agents:
            if a.alive:
                a.energy-=PASSIVE_DRAIN
                if a.energy<=0: a.alive=False; self.record("death",f"{a.name}({a.group})死",a.id)
        half=GRID_W//2
        for kind in RESOURCES:
            self.resources.append(ResourceNode(random.randint(0,half-1),random.randint(0,GRID_H-1),kind))
            self.resources.append(ResourceNode(random.randint(half,GRID_W-1),random.randint(0,GRID_H-1),kind))
        for g in ["A","B"]:
            gids={a.id for a in self.agents if a.group==g}
            te=[e for e in self.events if e["tick"]==self.tick]
            self.stats[g]["msgs"].append(sum(1 for e in te if e["type"]=="message" and e["agent"] in gids))
            self.stats[g]["gives"].append(sum(1 for e in te if e["type"]=="give" and e["agent"] in gids))
            self.stats[g]["crafts"].append(sum(1 for e in te if e["type"]=="craft" and e["agent"] in gids))
            self.stats[g]["eats"].append(sum(1 for e in te if e["type"]=="eat" and e["agent"] in gids))
            self.stats[g]["harvests"].append(sum(1 for e in te if e["type"]=="harvest" and e["agent"] in gids))
            self.stats[g]["alive"].append(sum(1 for a in self.agents if a.group==g and a.alive))
            self.stats[g]["energy"].append(sum(a.energy for a in self.agents if a.group==g and a.alive))
        return any(a.alive for a in self.agents)

    def run(self):
        self.init()
        print("═══ Round 010: 完美主义陷阱测试 ═══",file=sys.stderr,flush=True)
        print(f"  A组4人(有降级提示) vs B组4人(无降级提示) | {MAX_TICKS}天",file=sys.stderr,flush=True)
        print(f"  eat+{EAT_REWARD} craft+{CRAFT_REWARD} 被动-{PASSIVE_DRAIN} 起始{INITIAL_ENERGY}",file=sys.stderr,flush=True)
        print(file=sys.stderr,flush=True)
        for tick in range(MAX_TICKS):
            self.tick=tick; t0=time.time(); cont=self.run_tick(); elapsed=time.time()-t0
            s=self.stats
            print(f"  第{tick+1:>2}天 | A:{s['A']['alive'][-1]}人 E={s['A']['energy'][-1]:>3} "
                  f"eat={s['A']['eats'][-1]} give={s['A']['gives'][-1]} craft={s['A']['crafts'][-1]} | "
                  f"B:{s['B']['alive'][-1]}人 E={s['B']['energy'][-1]:>3} "
                  f"eat={s['B']['eats'][-1]} give={s['B']['gives'][-1]} craft={s['B']['crafts'][-1]} | {elapsed:.0f}s",
                  file=sys.stderr,flush=True)
            ke=[e for e in self.events if e["tick"]==self.tick and e["type"] in ("give","craft","eat")]
            for e in ke[:4]: print(f"        {'✅' if e['type']=='craft' else '🍽' if e['type']=='eat' else '🤝'} {e['detail']}",file=sys.stderr,flush=True)
            if not cont: print("  *** 全灭 ***",file=sys.stderr,flush=True); break
        self.dlg_file.close(); self.output()

    def output(self):
        def welch_t(a,b):
            na,nb=len(a),len(b)
            if na<2 or nb<2: return 0,1.0
            ma,mb=sum(a)/na,sum(b)/nb
            va=sum((x-ma)**2 for x in a)/(na-1); vb=sum((x-mb)**2 for x in b)/(nb-1)
            d=va/na+vb/nb
            if d<=0: return 0,1.0
            t=(ma-mb)/math.sqrt(d); p=math.erfc(abs(t)/math.sqrt(2))
            return round(t,3),round(p,4)
        t_eat,p_eat=welch_t(self.stats["A"]["eats"],self.stats["B"]["eats"])
        t_craft,p_craft=welch_t(self.stats["A"]["crafts"],self.stats["B"]["crafts"])
        t_give,p_give=welch_t(self.stats["A"]["gives"],self.stats["B"]["gives"])
        t_energy,p_energy=welch_t(self.stats["A"]["energy"],self.stats["B"]["energy"])
        gd={}
        for g in ["A","B"]:
            ga=[a for a in self.agents if a.group==g]
            gd[g]={"eats":sum(a.eaten for a in ga),"crafts":sum(a.crafted for a in ga),
                   "gives":sum(a.gives_out for a in ga),"harvests":sum(a.harvested for a in ga),
                   "msgs":sum(a.messages_sent for a in ga),"alive":sum(1 for a in ga if a.alive),
                   "energy":sum(a.energy for a in ga if a.alive)}
        result={"experiment":"round-010: perfectionism trap","ticks":self.tick+1,"api_calls":self.total_api_calls,
                "groups":gd,"tests":{"eat":{"t":t_eat,"p":p_eat,"sig":p_eat<0.05},
                "craft":{"t":t_craft,"p":p_craft,"sig":p_craft<0.05},
                "give":{"t":t_give,"p":p_give,"sig":p_give<0.05},
                "energy":{"t":t_energy,"p":p_energy,"sig":p_energy<0.05}},
                "per_agent":[{"name":a.name,"group":a.group,"alive":a.alive,"energy":a.energy,
                              "eaten":a.eaten,"crafted":a.crafted,"gives":a.gives_out,"harvested":a.harvested,"msgs":a.messages_sent}
                             for a in self.agents],
                "per_tick":{g:{"eats":self.stats[g]["eats"],"crafts":self.stats[g]["crafts"],
                              "gives":self.stats[g]["gives"],"energy":self.stats[g]["energy"]} for g in ["A","B"]}}
        with open(Path(__file__).parent/"result.json","w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=1)
        print("\n"+"═"*60,file=sys.stderr,flush=True)
        print("  【降级提示是否打破完美主义陷阱？】",file=sys.stderr,flush=True)
        for g,l in [("A","有降级提示"),("B","无降级提示")]:
            d=gd[g]; print(f"  {l}: {d['eats']}吃 {d['crafts']}合成 {d['gives']}给 {d['msgs']}话 | 存活{d['alive']}/4 E={d['energy']}",file=sys.stderr,flush=True)
        print(f"\n  进食: t={t_eat} p={p_eat} {'✓' if p_eat<0.05 else '✗'}",file=sys.stderr,flush=True)
        print(f"  合成: t={t_craft} p={p_craft} {'✓' if p_craft<0.05 else '✗'}",file=sys.stderr,flush=True)
        print(f"  能量: t={t_energy} p={p_energy} {'✓' if p_energy<0.05 else '✗'}",file=sys.stderr,flush=True)
        print(f"  API: {self.total_api_calls}",file=sys.stderr,flush=True)
        print("═"*60,file=sys.stderr,flush=True)
        with open(Path(__file__).parent/"events.jsonl","w",encoding="utf-8") as f:
            for e in self.events: f.write(json.dumps(e,ensure_ascii=False)+"\n")

if __name__=="__main__": World().run()

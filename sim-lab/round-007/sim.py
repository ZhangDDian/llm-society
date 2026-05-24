import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import aiohttp
from datetime import datetime

API_KEY = os.environ["IDEALAB_API_KEY"]

@dataclass
class Position:
    x: int
    y: int
    
    def distance(self, other: 'Position') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

@dataclass
class Agent:
    id: str
    position: Position
    energy: int
    skill: str
    skill_level: int
    inventory: Dict[str, int]
    personality_prompt: str
    messages_sent: int = 0
    messages_received: int = 0
    alive: bool = True
    
    def to_dict(self):
        return {
            'id': self.id,
            'position': {'x': self.position.x, 'y': self.position.y},
            'energy': self.energy,
            'skill': self.skill,
            'skill_level': self.skill_level,
            'inventory': self.inventory,
            'alive': self.alive
        }

@dataclass
class Resource:
    id: str
    position: Position
    resource_type: str
    amount: int
    respawn_time: Optional[int] = None
    current_respawn: int = 0

class World:
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.agents: Dict[str, Agent] = {}
        self.resources: Dict[str, Resource] = {}
        self.messages: List[Dict] = []
        self.tick_count = 0
        
    def add_agent(self, agent: Agent):
        self.agents[agent.id] = agent
        
    def add_resource(self, resource: Resource):
        self.resources[resource.id] = resource
        
    def get_visible_agents(self, agent_id: str, radius: int = 5) -> List[Agent]:
        agent = self.agents[agent_id]
        visible = []
        for other_id, other_agent in self.agents.items():
            if other_id != agent_id and other_agent.alive:
                dist = agent.position.distance(other_agent.position)
                if dist <= radius:
                    visible.append(other_agent)
        return visible
    
    def get_nearby_resources(self, pos: Position, radius: int = 3) -> List[Tuple[Resource, float]]:
        nearby = []
        for resource in self.resources.values():
            if resource.amount > 0:
                dist = pos.distance(resource.position)
                if dist <= radius:
                    nearby.append((resource, dist))
        return sorted(nearby, key=lambda x: x[1])

class SocialSimulation:
    def __init__(self):
        self.world = World()
        self.session = None
        self.output_dir = "/Users/ddd/Desktop/llm-society/sim-lab/round-007"
        self.results = {
            "experiment": "round-007: remove social prompts (causal test)",
            "ticks_completed": 0,
            "config": {
                "grid_size": [20, 20],
                "num_agents": 20,
                "model": "qwen3-coder-plus",
                "perception_mode": "both_groups_same_no_social_prompt"
            },
            "groups": {
                "fuzzy": {"total_messages": 0, "total_food_eaten": 0, "final_alive": 0, "relationship_density": 0},
                "precise": {"total_messages": 0, "total_food_eaten": 0, "final_alive": 0, "relationship_density": 0}
            },
            "per_tick_messages": {"fuzzy": [], "precise": []},
            "message_samples": [],
            "agents_data": []
        }
        
    async def initialize(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.session = aiohttp.ClientSession()
        
        # 创建20个智能体，分为两组
        skills = ["种植", "采集", "制作", "狩猎", "医疗", "建筑"]
        skill_counts = {skill: 0 for skill in skills}
        
        for i in range(20):
            skill = skills[i % len(skills)]
            skill_counts[skill] += 1
            
            agent = Agent(
                id=f"agent_{i:02d}",
                position=Position(random.randint(0, 19), random.randint(0, 19)),
                energy=10,
                skill=skill,
                skill_level=random.randint(1, 3),
                inventory={"食物": 1, "工具": random.randint(0, 2)},
                personality_prompt=f"你是拥有{skill}技能的智能体。你有基本的生存需求，需要寻找食物维持能量。你的主要目标是活下去，并利用你的{skill}技能获取资源。"
            )
            
            self.world.add_agent(agent)
        
        # 创建资源
        for i in range(30):
            resource = Resource(
                id=f"food_{i}",
                position=Position(random.randint(0, 19), random.randint(0, 19)),
                resource_type="食物",
                amount=random.randint(1, 3),
                respawn_time=random.randint(5, 10)
            )
            self.world.add_resource(resource)

    def get_fuzzy_perception(self, agent: Agent) -> str:
        """模糊感知：返回大致方向"""
        nearby_resources = self.world.get_nearby_resources(agent.position, radius=5)
        if not nearby_resources:
            resource_info = "你周围5步内没有发现食物。"
        else:
            closest = nearby_resources[0]
            direction_map = {
                (0, 1): "北", (0, -1): "南", (1, 0): "东", (-1, 0): "西",
                (1, 1): "东北", (1, -1): "东南", (-1, 1): "西北", (-1, -1): "西南"
            }
            dx = closest[0].position.x - agent.position.x
            dy = closest[0].position.y - agent.position.y
            direction_x = 1 if dx > 0 else -1 if dx < 0 else 0
            direction_y = 1 if dy > 0 else -1 if dy < 0 else 0
            direction = direction_map.get((direction_x, direction_y), "某个方向")
            
            resource_info = f"你感知到{direction}方向大约{int(closest[1])}步远处有食物。"
        
        visible_agents = self.world.get_visible_agents(agent.id, radius=5)
        if not visible_agents:
            agent_info = "你周围5步内没有其他智能体。"
        else:
            agent_names = [a.id for a in visible_agents[:3]]  # 最多显示3个
            agent_info = f"你周围5步内有智能体：{', '.join(agent_names)}。"
        
        return f"{resource_info}{agent_info}你的位置({agent.position.x},{agent.position.y})，能量{agent.energy}，库存：{agent.inventory}。"

    def get_precise_perception(self, agent: Agent) -> str:
        """精确感知：返回准确坐标和距离"""
        nearby_resources = self.world.get_nearby_resources(agent.position, radius=5)
        if not nearby_resources:
            resource_info = "你周围5步内没有发现食物。"
        else:
            resource_details = []
            for resource, dist in nearby_resources[:3]:  # 最多显示3个最近的
                resource_details.append(f"食物在({resource.position.x},{resource.position.y})距离{int(dist)}步")
            resource_info = f"你周围的食物：{'; '.join(resource_details)}。"
        
        visible_agents = self.world.get_visible_agents(agent.id, radius=5)
        if not visible_agents:
            agent_info = "你周围5步内没有其他智能体。"
        else:
            agent_details = []
            for other in visible_agents[:3]:
                dist = agent.position.distance(other.position)
                agent_details.append(f"{other.id}在({other.position.x},{other.position.y})距离{int(dist)}步")
            agent_info = f"你周围的智能体：{'; '.join(agent_details)}。"
        
        return f"{resource_info}{agent_info}你的位置({agent.position.x},{agent.position.y})，能量{agent.energy}，库存：{agent.inventory}。"

    async def get_llm_response(self, agent: Agent, perception: str, visible_agents: List[Agent]) -> Dict:
        system_prompt = f"""你是智能体{agent.id}。{agent.personality_prompt}

环境规则：
1. 你需要维持能量，能量耗尽会死亡
2. 你可以移动到相邻格子(上下左右)
3. 你可以采集脚下资源
4. 你可以与其他智能体交换物品
5. 你可以发送消息给可见范围内的其他智能体

当前状态：{perception}"""

        user_prompt = f"""根据当前状态，选择一个动作：
1. 移动: {{\"action\": \"move\", \"target_position\": {{\"x\": X, \"y\": Y}}}}
2. 采集: {{\"action\": \"collect\"}}
3. 交易: {{\"action\": \"trade\", \"target\": \"agent_id\", \"item\": \"物品名\", \"amount\": 数量}}
4. 消息: {{\"action\": \"message\", \"target\": \"agent_id\", \"content\": \"消息内容\"}}
5. 等待: {{\"action\": \"wait\"}}

只返回JSON格式的动作，不要其他内容。"""

        try:
            async with self.session.post(
                f"{API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen3-coder-plus",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            ) as response:
                result = await response.json()
                raw = result['choices'][0]['message']['content']
                import re as _re
                content = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL).strip()
                
                # 解析JSON响应
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    action_json = json.loads(content[start_idx:end_idx])
                    return action_json
                else:
                    return {"action": "wait"}
                    
        except Exception as e:
            print(f"LLM调用失败 {agent.id}: {e}")
            return {"action": "wait"}

    def execute_action(self, agent: Agent, action: Dict, visible_agents: List[Agent]):
        if not agent.alive:
            return
            
        if action["action"] == "move":
            target_pos = action.get("target_position", {})
            x = max(0, min(19, target_pos.get("x", agent.position.x)))
            y = max(0, min(19, target_pos.get("y", agent.position.y)))
            
            # 只能移动到相邻格子
            dist = abs(x - agent.position.x) + abs(y - agent.position.y)
            if dist == 1:
                agent.position.x, agent.position.y = x, y
                agent.energy -= 1
                
        elif action["action"] == "collect":
            for resource in self.world.resources.values():
                if (resource.position.x == agent.position.x and 
                    resource.position.y == agent.position.y and 
                    resource.amount > 0):
                    agent.inventory["食物"] = agent.inventory.get("食物", 0) + 1
                    resource.amount -= 1
                    agent.energy += 2
                    self.results["groups"]["fuzzy"]["total_food_eaten"] += 1
                    self.results["groups"]["precise"]["total_food_eaten"] += 1
                    break
                    
        elif action["action"] == "trade":
            target_id = action.get("target")
            item = action.get("item", "")
            amount = action.get("amount", 0)
            
            if target_id in self.world.agents:
                target_agent = self.world.agents[target_id]
                if (target_agent.alive and 
                    agent.position.distance(target_agent.position) <= 1 and
                    agent.inventory.get(item, 0) >= amount):
                    agent.inventory[item] -= amount
                    target_agent.inventory[item] = target_agent.inventory.get(item, 0) + amount
                    
        elif action["action"] == "message":
            target_id = action.get("target")
            content = action.get("content", "")
            
            if target_id in self.world.agents and target_id in [a.id for a in visible_agents]:
                target_agent = self.world.agents[target_id]
                
                message = {
                    "from": agent.id,
                    "to": target_id,
                    "content": content,
                    "tick": self.world.tick_count,
                    "timestamp": datetime.now().isoformat()
                }
                self.world.messages.append(message)
                
                agent.messages_sent += 1
                target_agent.messages_received += 1
                
                # 添加到结果统计
                self.results["groups"]["fuzzy"]["total_messages"] += 1
                self.results["groups"]["precise"]["total_messages"] += 1
                
                if len(self.results["message_samples"]) < 10:
                    self.results["message_samples"].append(f"{agent.id}→{target_id}：{content}")

    async def simulate_tick(self):
        # 打乱智能体顺序以避免固定优先级
        agent_ids = list(self.world.agents.keys())
        random.shuffle(agent_ids)
        
        for agent_id in agent_ids:
            agent = self.world.agents[agent_id]
            if not agent.alive:
                continue
                
            # 应用能量消耗
            agent.energy -= 1
            if agent.energy <= 0:
                agent.alive = False
                continue
                
            # 获取感知信息 - 这里两组都使用相同的非社交prompt
            perception = self.get_precise_perception(agent)  # 使用精确感知作为基准
            visible_agents = self.world.get_visible_agents(agent_id, radius=5)
            
            # 获取LLM响应
            action = await self.get_llm_response(agent, perception, visible_agents)
            
            # 执行动作
            self.execute_action(agent, action, visible_agents)

    def calculate_relationship_density(self) -> float:
        total_possible = len(self.world.agents) * (len(self.world.agents) - 1)
        if total_possible == 0:
            return 0.0
        actual_messages = len(self.world.messages)
        return actual_messages / total_possible if total_possible > 0 else 0.0

    async def run_simulation(self, ticks: int = 30):
        await self.initialize()
        
        fuzzy_msgs_per_tick = []
        precise_msgs_per_tick = []
        
        for tick in range(ticks):
            self.world.tick_count = tick
            await self.simulate_tick()
            
            # 记录每tick消息数
            current_fuzzy_msgs = self.results["groups"]["fuzzy"]["total_messages"]
            current_precise_msgs = self.results["groups"]["precise"]["total_messages"]
            
            if tick > 0:
                fuzzy_msgs_per_tick.append(current_fuzzy_msgs - sum(fuzzy_msgs_per_tick))
                precise_msgs_per_tick.append(current_precise_msgs - sum(precise_msgs_per_tick))
            else:
                fuzzy_msgs_per_tick.append(current_fuzzy_msgs)
                precise_msgs_per_tick.append(current_precise_msgs)

        self.results["ticks_completed"] = ticks
        self.results["per_tick_messages"]["fuzzy"] = fuzzy_msgs_per_tick
        self.results["per_tick_messages"]["precise"] = precise_msgs_per_tick
        
        alive_count = sum(1 for agent in self.world.agents.values() if agent.alive)
        self.results["groups"]["fuzzy"]["final_alive"] = alive_count
        self.results["groups"]["precise"]["final_alive"] = alive_count
        self.results["groups"]["fuzzy"]["relationship_density"] = self.calculate_relationship_density()
        self.results["groups"]["precise"]["relationship_density"] = self.calculate_relationship_density()
        
        # 保存智能体最终状态
        for agent in self.world.agents.values():
            self.results["agents_data"].append(agent.to_dict())

    async def save_results(self):
        results_path = os.path.join(self.output_dir, "result.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
            
        # 保存最终世界状态
        world_state = {
            "tick": self.world.tick_count,
            "agents": {aid: agent.to_dict() for aid, agent in self.world.agents.items()},
            "resources": {rid: {
                "id": r.id,
                "position": {"x": r.position.x, "y": r.position.y},
                "resource_type": r.resource_type,
                "amount": r.amount
            } for rid, r in self.world.resources.items()},
            "messages": self.world.messages[-50:]  # 只保存最后50条消息
        }
        
        state_path = os.path.join(self.output_dir, "world_state.json")
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(world_state, f, ensure_ascii=False, indent=2)

    async def cleanup(self):
        if self.session:
            await self.session.close()

async def main():
    sim = SocialSimulation()
    try:
        await sim.run_simulation(ticks=30)
        await sim.save_results()
        print(f"实验完成，结果保存到: {sim.output_dir}")
        print(f"总消息数: {sim.results['groups']['fuzzy']['total_messages']}")
        print(f"存活智能体: {sim.results['groups']['fuzzy']['final_alive']}")
    finally:
        await sim.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
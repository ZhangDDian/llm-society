import os
from openai import OpenAI
from pathlib import Path

client = OpenAI(
    base_url=os.environ.get("IDEALAB_API_BASE", "https://api.openai.com/v1"),
    api_key=os.environ["IDEALAB_API_KEY"]
)

sim_code = (Path(__file__).parent / "sim.py").read_text()
summary = (Path(__file__).parent / "summary.md").read_text()

prompt = f"""
你是对立假设的辩护律师。你的工作不是评审实验好不好，
而是尽全力用对立假设来解释实验可能产生的一切结果。

## 当前假设（你要攻击的）
在闭环能量经济中，tool-calling 多步 agent 的合作完成率和能量积累将显著优于单步 agent——
因为多步能力使"协商+交换+合成"能在 1-2 tick 内完成，而单步 agent 需要 4-5 tick，
期间的被动消耗使合作变得不经济。

## 对立假设（你要辩护的）
闭环能量经济不改变结论——单步 agent 虽然慢，但会学会在相邻 tick 连续执行交易子步骤
（形成"习惯链"），最终合作效率与多步 agent 趋同。多步能力的优势只体现在前几天的
适应速度，而非稳态效率。

## 实验设计
白话摘要：
{summary}

代码（关键部分）：
- 每组5人，技能各异（5种资源各1人）
- A组每tick最多5步工具调用，B组每tick 1个动作
- 被动消耗2/tick，move/say/give各消耗1能量
- eat: 消耗1资源→+6能量；craft: 消耗2种资源→+18能量
- 起始能量80，30天
- 两组分区（A左半B右半），资源独立再生

## 你的任务
1. 假设实验跑出了"支持当前假设"的结果（A组合成远多于B组）——给出至少 2 种替代解释，
   说明同样的数据如何被对立假设解释
2. 指出实验设计中哪些参数/规则的设定本身就偏向了当前假设
3. 如果你是设计者，你会怎么改实验才能真正排除对立假设？

## 输出格式
- 替代解释（≥2 条）：
- 设计偏见（参数/规则层面）：
- 你建议的修改：
- 一句话：如果实验结果支持假设，最可能的"假阳性"原因是什么？
"""

r = client.chat.completions.create(
    model="gpt-5.5-0424-global",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=16000
)

with open(Path(__file__).parent / "red_team_review.md", "w") as f:
    f.write("## 红队（GPT-5.5）\n\n")
    f.write(r.choices[0].message.content)

print("红队评审完成")

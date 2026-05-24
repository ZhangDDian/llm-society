import os
"""红队评审：调用 GPT-5.5 作为对立假设的辩护律师"""
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
模糊感知迫使 LLM agent 通过社交交换信息来降低不确定性，而精确感知让 agent 无需沟通就能最优决策。即：感知的不确定性是社交行为的驱动力。

## 对立假设（你要辩护的）
模糊感知下的社交只是 LLM 的"不知道该干嘛所以聊天"——跟 Round 000a 一样是语言模型的默认输出，没有信息交换价值。两组存活率差异（如有）才是关键指标：模糊组社交多但活得差 = 社交无功能性。

## 实验设计
白话摘要：
{summary}

代码（关键部分）：
- 20 个 agent 分两组（模糊 vs 精确），住在同一世界的左右半区
- 同一模型（Qwen3.6-Plus-DogFooding）、同一人格 system prompt
- 唯一变量：感知信息格式（主观体验 vs 坐标数字）
- 30 ticks，第 20 tick 打乱位置测试关系是否为真结构
- 追踪"功能性消息"：消息含方向信息 + 接收者随后朝该方向移动

## 你的任务
1. 假设实验跑出了"支持当前假设"的结果（模糊组消息多、存活率不差、有功能性消息）——给出至少 2 种替代解释，说明同样的数据如何被对立假设解释
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

output = "## 红队（GPT-5.5）\n\n" + r.choices[0].message.content
Path(__file__).parent.joinpath("red_team_review.md").write_text(output, encoding="utf-8")
print("红队评审完成，结果写入 red_team_review.md")

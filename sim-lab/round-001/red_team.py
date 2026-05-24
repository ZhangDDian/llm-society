import os
"""红队评审 — 用 GPT-5.5 作为对立假设辩护律师"""
from openai import OpenAI
from pathlib import Path

client = OpenAI(
    base_url=os.environ.get("IDEALAB_API_BASE", "https://api.openai.com/v1"),
    api_key=os.environ["IDEALAB_API_KEY"]
)

sim_code = Path(__file__).parent / "sim.py"
code_content = sim_code.read_text()[:6000]  # 截取前 6000 字符避免超长

prompt = f"""
你是对立假设的辩护律师。你的工作不是评审实验好不好，
而是尽全力用对立假设来解释实验可能产生的一切结果。

## 当前假设（你要攻击的）
社会结构（交易、联盟、等级）只在"个体无法独立生存"的条件下才从 LLM agent 群体中涌现。

## 对立假设（你要辩护的）
LLM 的社交倾向是"语言模型默认行为"（训练分布中社交对话远多于独立行动），与生存压力无关——只要 prompt 不刻意抑制，agent 就会社交。

## 背景
Round 000：
- 第一局：prompt 开放，agent 疯狂社交但不吃东西，全灭
- 第二局：prompt 加了"先找食物再吃"的强指引，agent 零社交但全部存活

## 实验设计（Round 001）
白话摘要：在 20 人小镇里加入"需要两人配合才能采集"的矿石资源，同时削弱独食收益。观察社交行为是否从 0 增加。

代码核心逻辑：
{code_content[:4000]}

## 你的任务
1. 假设实验跑出了"支持当前假设"的结果（消息从0变成很多，且内容跟采矿相关）——给出至少 2 种替代解释，说明同样的数据如何被对立假设解释
2. 指出实验设计中哪些参数/规则的设定本身就偏向了当前假设
3. 如果你是设计者，你会怎么改实验才能真正排除对立假设？

## 输出格式
- 替代解释（≥2 条）：
- 设计偏见（参数/规则层面）：
- 你建议的修改：
- 一句话：如果实验结果支持假设，最可能的"假阳性"原因是什么？
"""

print("正在调用 GPT-5.5...", flush=True)
r = client.chat.completions.create(
    model="gpt-5.5-0424-global",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=16000
)

output = r.choices[0].message.content
print("红队评审完成", flush=True)

with open(Path(__file__).parent / "red_team_review.md", "w") as f:
    f.write("## 红队（GPT-5.5）\n\n")
    f.write(output)

print(f"\n已保存到 red_team_review.md")
print(f"\n--- 评审内容 ---\n{output}")

import os
"""红队评审：调用 GPT-5.5 作为对立假设辩护律师"""
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
异质性（不同技能 + 不同需求 + 不同性格）驱动自发分工与交易网络的涌现：当 30 个 agent 各有专长且互相需要对方的产出时，会自发形成稳定的交易关系、信息共享和角色分化——而同质 agent 群体不会。

## 对立假设（你要辩护的）
LLM agent 的"个性"只是 prompt 装饰——无论分配什么技能/需求/性格，它们的实际行为模式趋同（要么都社交要么都沉默），不会产生真正的功能性分工。异质 prompt 改变的是"说话内容"而非"行为结构"。

## 实验设计
白话摘要：
{summary}

代码（关键部分）：
- 异质组 15 人，5 种技能各 3 人，每人独特性格/需求描述
- 同质组 15 人，万能工，统一性格
- harvest 动作有技能检查：skill != resource.kind 时采集失败
- give 动作允许 agent 把自己的资源给旁边的人
- craft 需要 2 种不同资源
- 干扰：第 20 天杀掉异质组消息最多的 2 人
- 两组在同一世界的左右半区，资源再生率相同

## 你的任务
1. 假设实验跑出了"支持当前假设"的结果——给出至少 2 种替代解释，
   说明同样的数据如何被对立假设解释
2. 指出实验设计中哪些参数/规则的设定本身就偏向了当前假设
3. 如果你是设计者，你会怎么改实验才能真正排除对立假设？

## 输出格式
- 替代解释（≥2 条）：
- 设计偏见（参数/规则层面）：
- 你建议的修改：
- 一句话：如果实验结果支持假设，最可能的"假阳性"原因是什么？
"""

try:
    r = client.chat.completions.create(
        model="gpt-5.5-0424-global",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16000
    )
    output = r.choices[0].message.content
except Exception as e:
    output = f"[红队 API 调用失败: {e}]"

with open(Path(__file__).parent / "red_team_review.md", "w", encoding="utf-8") as f:
    f.write("## 红队（GPT-5.5）\n\n")
    f.write(output)

print(output)

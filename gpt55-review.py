"""调用 GPT-5.5 对 llm-society 项目做完整审查"""

import os
from openai import OpenAI
from pathlib import Path

client = OpenAI(
    base_url=os.environ.get("IDEALAB_API_BASE", "https://api.openai.com/v1"),
    api_key=os.environ["IDEALAB_API_KEY"],
)

base = Path(__file__).parent

# 只用核心叙事文档，不塞代码
files = {
    "README.md": (base / "README.md").read_text(),
    "sim-lab/brief.md（每轮详细过程）": (base / "sim-lab/brief.md").read_text(),
    "sim-lab/config.md（约束与假设链）": (base / "sim-lab/config.md").read_text(),
    "round-009/review.md（红蓝队评审样本）": (base / "sim-lab/round-009/review.md").read_text(),
    "round-010/summary.md": (base / "sim-lab/round-010/summary.md").read_text(),
    "round-011/summary.md": (base / "sim-lab/round-011/summary.md").read_text(),
}

project_context = "\n\n".join(
    f"=== {name} ===\n{content}" for name, content in files.items()
)

system_prompt = """你是一个跨学科战略洞察分析师。你的任务是对一个 LLM 多智能体社会模拟实验项目做深度审查。

要求：
- 全中文回复
- 理性、中立、客观
- 不客套、不重复项目已有的结论
- 重点放在项目没看到的、做错的、可以做得更好的地方
- 数据和逻辑说话，不要空泛的鼓励
- 信息密度要高，重复或低价值的内容简略，高价值部分详细展开

请按以下结构分析：

## 第一部分：发散性分析

### 1. 初始觉察与好奇
阅读材料时你注意到了什么？隐含假设、方法论盲点、结论跳跃、未言明的前提？宏观层面的好奇？

### 2. 高熵与混乱点
项目中最混乱、最不确定的部分。关键矛盾点、悖论、失败的尝试。这些是未来创新的最大机会。

### 3. 跨学科视角
从行为经济学、演化生物学、社会学、认知科学、博弈论、分布式系统等视角，有哪些类比或模型可借鉴？

### 4. 盲区、风险与机会
项目可能遗漏了什么？

## 第二部分：收敛性分析

### 5. 低熵与关键解法
项目中已出现的秩序点、高光时刻、清晰共识。

### 6. 领域专家建议
作为多智能体系统和 LLM 行为研究的顶尖专家，核心洞察是什么？关键建议？

### 7. 深度调研议程
最需要进一步调研的关键问题，每个给出具体方向和方法。"""

user_prompt = f"""以下是"LLM 社会模拟实验"的完整项目材料。研究者用 12 轮实验验证 LLM agent 是否能自发形成社会合作。请做深度审查。

{project_context}"""

print(f"材料总长度: {len(user_prompt)} 字符", flush=True)
print("正在调用 GPT-5.5...", flush=True)

response = client.chat.completions.create(
    model="gpt-5.5-0424-global",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    max_completion_tokens=32000,
)

result = response.choices[0].message.content
print(f"返回长度: {len(result) if result else 0} 字符", flush=True)

if result:
    output_path = base / "gpt55-review.md"
    output_path.write_text(f"# GPT-5.5 项目审查\n\n{result}")
    print(f"\n已保存到: {output_path}")
else:
    print("返回为空，尝试查看 finish_reason:", response.choices[0].finish_reason)

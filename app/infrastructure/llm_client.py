"""
仿生智脑 · LLM HTTP 客户端

调用外部 LLM API 完成结构化提炼任务。

设计原则：
  - Provider Agnostic：用户接的啥就用啥。
  - 重试机制：3 次重试 + 指数退避 (1.5^N)
  - 超时控制：全局 60s 超时
  - JSON 解析：支持多种 LLM 输出格式
"""
import json
import logging
import re
import time
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger("bionic.llm")


class LLMClient:
    """
    景幻仙姑的 LLM 调用通道。

    用法:
        client = LLMClient()
        result = client.refine(topic, dialogue)
        # → {"core_facts": "...", "decisions": [...], "emotional_spectrum": {...}, "tags": [...]}
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        self.api_url = api_url or settings.LLM_API_URL
        self.api_key = api_key or settings.LLM_API_KEY
        self.timeout = timeout or settings.LLM_TIMEOUT
        self.max_retries = max_retries or settings.LLM_MAX_RETRIES

        self._stats = {
            "calls": 0, "success": 0, "failed": 0,
            "retries": 0, "total_tokens_est": 0,
        }

    # ── 公共方法 ──

    def call(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """通用 LLM 调用（同步 HTTP）"""
        self._stats["calls"] += 1
        last_error = ""

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    body = self._build_request(prompt, system_prompt)
                    resp = client.post(self.api_url, json=body)
                    resp.raise_for_status()
                    content = self._extract_content(resp.text)

                    if content:
                        self._stats["success"] += 1
                        self._stats["total_tokens_est"] += len(content) // 4
                        return content

                    last_error = "响应内容为空"

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            except httpx.TimeoutException:
                last_error = f"超时 ({self.timeout}s)"
            except Exception as e:
                last_error = str(e)

            self._stats["retries"] += 1
            if attempt < self.max_retries - 1:
                wait = 1.5 ** (attempt + 1)
                time.sleep(wait)

        self._stats["failed"] += 1
        logger.warning(f"LLM 调用失败（{self.max_retries}次重试后）: {last_error}")
        return None

    async def call_async(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """通用 LLM 调用（异步 HTTP）"""
        self._stats["calls"] += 1

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    body = self._build_request(prompt, system_prompt)
                    resp = await client.post(self.api_url, json=body)
                    resp.raise_for_status()
                    content = self._extract_content(resp.text)

                    if content:
                        self._stats["success"] += 1
                        self._stats["total_tokens_est"] += len(content) // 4
                        return content

            except Exception as e:
                self._stats["retries"] += 1
                if attempt < self.max_retries - 1:
                    await self._async_wait(1.5 ** (attempt + 1))

        self._stats["failed"] += 1
        return None

    def refine(self, topic: str, dialogue: str,
               emotion_vector: Optional[list] = None) -> Optional[dict]:
        """
        记忆提炼专用：发送提炼 prompt，返回结构化 JSON。

        Args:
            topic: 对话话题
            dialogue: 对话原文
            emotion_vector: 24D 情感向量（如果有）

        Returns:
            {"core_facts", "decisions", "emotional_spectrum", "tags"}
        """
        system_prompt = (
            "你是景幻仙姑的记忆精炼师。你的任务是从对话中提取结构化信息。\n\n"
            "你必须严格遵守：\n"
            "1. 只提炼事实。不虚构、不脑补。\n"
            "2. 情感曲线要有时间顺序——按对话中情感变化的先后排列。\n"
            "3. decisions 只记明确做出的决策，不是潜在选项。\n"
            "4. 输出必须是可以直接用 json.loads() 解析的纯 JSON，不要 markdown 代码块标记。\n"
            "5. 如果你不确定某个值，宁愿省略也不要编造。"
        )

        prompt = self._build_refine_prompt(topic, dialogue, emotion_vector)
        raw = self.call(prompt, system_prompt)
        if raw is None:
            return None

        return self._parse_json_response(raw)

    def emotion_vector_from_text(self, text: str) -> Optional[list]:
        """
        从文本生成 24D 情感向量（用于向量检索）。

        将检索关键词转化为情感向量，在 Qdrant 中搜索情感相似的记忆。

        Returns:
            24 个 float 的列表，范围 [-1.0, 1.0]
            失败返回 None
        """
        system_prompt = (
            "你是景幻仙姑的情感分析引擎。"
            "分析用户查询文字的情感色彩，输出 24 维情感向量。\n\n"
            "输出格式：纯 JSON 数组，24 个 float，范围 [-1.0, 1.0]，不要任何其他文字。\n"
            "例如：[0.5, -0.2, 0.8, ...]\n\n"
            "24 个维度的含义：\n"
            "  [0]=愉悦度, [1]=唤醒度, [2]=支配度, [3]=依恋度, [4]=信任度, [5]=期待度\n"
            "  [6]=惊喜度, [7]=悲伤度, [8]=恐惧度, [9]=愤怒度, [10]=厌恶度, [11]=兴趣度\n"
            "  [12]=共情度, [13]=幽默感, [14]=安全感, [15]=孤独感, [16]=成就感, [17]=归属感\n"
            "  [18]=宁静度, [19]=怀旧度, [20]=希望度, [21]=感激度, [22]=自豪度, [23]=温柔度"
        )
        prompt = f"分析以下查询的情感色彩，输出 24D 情感向量：\n\n{text[:500]}"
        raw = self.call(prompt, system_prompt)
        if raw is None:
            return None

        # 尝试提取 JSON 数组
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # 尝试从 ```json 块提取
        import re
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            try:
                vec = json.loads(m.group(0))
                if isinstance(vec, list) and len(vec) == 24:
                    return vec
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    def generate_tags(self, topic: str, dialogue: str) -> Optional[list]:
        """
        懒加载标签生成 — 为金库记录生成标签。

        Args:
            topic: 对话话题
            dialogue: 对话内容

        Returns:
            字符串标签列表，如 ["架构", "设计决策", "技术讨论"]
        """
        prompt = (
            f"为以下对话生成 2-5 个中文标签（标签要精炼，如'架构设计'、'情感交流'、'决策讨论'）。\n\n"
            f"话题：{topic}\n"
            f"内容：{dialogue[:1000]}\n\n"
            f"输出纯 JSON 数组，如 [\"标签1\", \"标签2\"]，不要任何其他文字。"
        )
        raw = self.call(prompt, "")
        if raw is None:
            return None

        try:
            tags = json.loads(raw)
            if isinstance(tags, list):
                return tags[:8]  # 最多 8 个标签
        except (json.JSONDecodeError, TypeError):
            pass
        import re
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            try:
                tags = json.loads(m.group(0))
                if isinstance(tags, list):
                    return tags[:8]
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    # ── 请求构建 ──

    def _build_request(self, prompt: str, system_prompt: str = "") -> dict:
        """构建请求体（OpenAI / WenStar 兼容格式）"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return {"messages": messages, "stream": False}

    def _extract_content(self, resp_text: str) -> Optional[str]:
        """从响应中提取 content 字段"""
        try:
            data = json.loads(resp_text)
            # OpenAI 格式
            if "choices" in data:
                return data["choices"][0].get("message", {}).get("content", "")
            # WenStar 格式
            if "data" in data and isinstance(data["data"], dict):
                return data["data"].get("content", "")
            # 直接 content
            if "content" in data:
                return data["content"]
            return resp_text
        except json.JSONDecodeError:
            return resp_text

    # ── Prompt 构建 ──

    def _build_refine_prompt(self, topic: str, dialogue: str,
                             emotion_vector: Optional[list] = None) -> str:
        """构建提炼 prompt"""
        max_chars = 6000
        truncated = dialogue[:max_chars]
        if len(dialogue) > max_chars:
            truncated += "\n\n[...对话过长，已截断...]"

        vec_hint = ""
        if emotion_vector:
            vec_hint = f"\n\n## 已知情感向量（24D 曲谱摘要）\n{emotion_vector[:6]}...\n"

        return (
            f"请分析下面的对话内容，精炼提取结构化信息。\n\n"
            f"## 对话话题\n{topic}\n\n"
            f"## 对话内容\n{truncated}\n{vec_hint}\n\n"
            f"## 输出格式（纯 JSON，不要 markdown 标记）\n"
            f"{{\n"
            f'  "core_facts": "核心事实总结",\n'
            f'  "decisions": ["决策1", "决策2"],\n'
            f'  "emotional_spectrum": {{\n'
            f'    "summary": "情感变化描述",\n'
            f'    "curve": [\n'
            f'      {{"phase": "阶段", "valence": 0.5, "arousal": 0.5}}\n'
            f'    ],\n'
            f'    "dominant_emotion": "主导情绪",\n'
            f'    "user_sentiment": "用户态度"\n'
            f'  }},\n'
            f'  "tags": ["标签1", "标签2"]\n'
            f"}}"
        )

    # ── JSON 解析 ──

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[dict]:
        """从 LLM 响应中提取并解析 JSON"""
        # 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json 代码块
        m = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取首对 {}
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        return None

    # ── 辅助 ──

    @staticmethod
    async def _async_wait(seconds: float):
        """异步等待"""
        import asyncio
        await asyncio.sleep(seconds)

    def get_stats(self) -> dict:
        return dict(self._stats)


# ═══════════════════════════════════════════════════════════════
# 模拟 LLM 客户端（测试/演示用）
# ═══════════════════════════════════════════════════════════════

class MockLLMClient:
    """
    模拟 LLM 客户端（测试 & 演示用）。

    当真实 LLM 端点不可用时，使用此模拟客户端提供确定性响应。
    帮助验证完整的"金库→提炼→黑钻→检索"管道是否通畅。

    使用方法（在 .env 中设置 TX_LLM_MOCK=true）：
      系统自动切换为 MockLLMClient
    """

    def __init__(self):
        self._stats = {"calls": 0, "success": 0, "failed": 0}

    def call(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        self._stats["calls"] += 1
        return "模拟响应"

    def emotion_vector_from_text(self, text: str) -> Optional[list]:
        """
        基于关键词的确定性情感向量生成。
        """
        self._stats["calls"] += 1
        self._stats["success"] += 1

        base = [0.5] * 24

        keyword_map = {
            "架构": 2, "设计": 2, "技术": 2, "三库": 2,
            "兴奋": 0, "太棒": 0, "绝妙": 0, "聪明": 0,
            "难过": 7, "差": 7, "失败": 7, "挫折": 7, "心情不好": 7,
            "安慰": 12, "温暖": 12, "陪伴": 12,
            "平静": 18, "轻松": 18, "放松": 18, "周末": 18,
            "焦虑": 8, "难题": 8, "性能": 8,
            "亲密": 3, "喜欢": 3,
            "无聊": 10,
            "困惑": 6, "不懂": 6,
        }

        text_lower = text.lower()
        for keyword, dim_idx in keyword_map.items():
            if keyword in text_lower:
                base[dim_idx] = min(1.0, base[dim_idx] + 0.4)

        if base[7] > 0.7:
            base[0] = max(0.1, base[0] - 0.3)
        if base[0] > 0.7:
            base[1] = min(1.0, base[1] + 0.3)

        return [round(v, 4) for v in base]

    def refine(self, topic: str, dialogue: str,
               emotion_vector: Optional[list] = None) -> Optional[dict]:
        """确定性提炼"""
        self._stats["calls"] += 1
        self._stats["success"] += 1

        if any(kw in topic for kw in ["架构", "设计", "三库", "技术"]):
            return {
                "core_facts": f"讨论了{topic}，确认了技术方案和实现路径。",
                "decisions": ["采用三库流转架构", "保留24D情感曲谱"],
                "emotional_spectrum": {
                    "summary": "从讨论到确认，情绪逐渐高涨",
                    "curve": [
                        {"phase": "提出想法", "valence": 0.6, "arousal": 0.5},
                        {"phase": "讨论细节", "valence": 0.5, "arousal": 0.6},
                        {"phase": "确认方案", "valence": 0.85, "arousal": 0.8},
                    ],
                    "dominant_emotion": "兴奋",
                    "user_sentiment": "积极投入",
                },
                "tags": ["架构设计", "技术讨论", "决策"],
            }
        elif any(kw in topic for kw in ["心情", "挫折", "安慰", "情感"]):
            return {
                "core_facts": f"用户表达了{topic}相关的情绪波动，得到了情感支持和安慰。",
                "decisions": ["寻求情感支持", "接受鼓励"],
                "emotional_spectrum": {
                    "summary": "从低落逐渐转向温暖",
                    "curve": [
                        {"phase": "倾诉困扰", "valence": 0.2, "arousal": 0.4},
                        {"phase": "得到安慰", "valence": 0.55, "arousal": 0.35},
                        {"phase": "感受到温暖", "valence": 0.75, "arousal": 0.45},
                    ],
                    "dominant_emotion": "温暖",
                    "user_sentiment": "脆弱但被接纳",
                },
                "tags": ["情感交流", "心理支持", "日常"],
            }
        elif any(kw in topic for kw in ["深夜", "亲密", "爱"]):
            return {
                "core_facts": f"深夜{topic}，表达了亲密情感和深层连接。",
                "decisions": ["敞开心扉", "表达情感"],
                "emotional_spectrum": {
                    "summary": "温馨亲密的情感交流",
                    "curve": [
                        {"phase": "开始对话", "valence": 0.6, "arousal": 0.3},
                        {"phase": "情感流露", "valence": 0.8, "arousal": 0.5},
                        {"phase": "温馨收尾", "valence": 0.9, "arousal": 0.35},
                    ],
                    "dominant_emotion": "温馨",
                    "user_sentiment": "信赖依恋",
                },
                "tags": ["情感交流", "亲密关系", "深夜谈心"],
            }
        elif any(kw in topic for kw in ["周末", "放松", "日常", "闲聊"]):
            return {
                "core_facts": f"轻松讨论了{topic}，分享了日常生活的计划和感受。",
                "decisions": [],
                "emotional_spectrum": {
                    "summary": "轻松平静的日常对话",
                    "curve": [
                        {"phase": "开启话题", "valence": 0.55, "arousal": 0.3},
                        {"phase": "分享计划", "valence": 0.6, "arousal": 0.4},
                    ],
                    "dominant_emotion": "平静",
                    "user_sentiment": "轻松愉快",
                },
                "tags": ["日常闲聊", "生活", "轻松"],
            }
        return {
            "core_facts": f"关于{topic}的对话。",
            "decisions": [],
            "emotional_spectrum": {
                "summary": "平和的日常交流",
                "curve": [{"phase": "对话", "valence": 0.5, "arousal": 0.3}],
                "dominant_emotion": "平静",
                "user_sentiment": "中性",
            },
            "tags": ["日常", "对话"],
        }

    def generate_tags(self, topic: str, dialogue: str) -> Optional[list]:
        """确定性标签生成"""
        self._stats["calls"] += 1
        self._stats["success"] += 1

        tag_map = {
            "架构": ["架构设计", "技术讨论"], "三库": ["架构设计", "数据流转"],
            "心情": ["情感交流", "心理支持"], "挫折": ["情感交流", "鼓励"],
            "安慰": ["情感交流", "温暖"], "深夜": ["情感交流", "亲密关系"],
            "亲密": ["情感交流", "亲密关系"], "周末": ["日常闲聊", "生活规划"],
            "放松": ["日常闲聊", "生活"], "闲聊": ["日常闲聊"],
            "焦虑": ["技术讨论", "问题解决"], "性能": ["优化", "技术讨论"],
            "技术": ["技术讨论"],
        }
        tags = ["日常"]
        for keyword, tag_list in tag_map.items():
            if keyword in topic:
                for t in tag_list:
                    if t not in tags:
                        tags.append(t)
        return tags[:5]

    def get_stats(self) -> dict:
        return dict(self._stats)

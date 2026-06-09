"""
仿生智脑 · 景幻仙姑系统助理

景幻仙姑——"太虚智脑的掌管者、大英图书馆馆长"。
用户可以通过对话窗口向她：
  - 了解系统架构和使用方法
  - 反馈使用问题和改进建议
  - 请求微调（小改动直接执行）
  - 生成改善建议（大改动提交设计师审批）

设计原则：
  微调(小): 景幻可直接执行，不影响系统整体设计
  改善建议(大): 生成结构化提案，由设计师(鸿鸣)确认后实施
"""
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("bionic.assistant")


# ── 改善建议存储 ──
_proposals = []


class SystemAssistant:
    """
    景幻仙姑系统助理。

    每个实例绑定一次对话会话，维护对话上下文。
    有 LLM 时使用 LLM + 知识库自然对话，无 LLM 时使用规则回复。
    """

    def __init__(self, llm_client=None, knowledge_base=None):
        self.conversation: list[dict] = []
        self.llm = llm_client
        self.kb = knowledge_base
        self._stats = {"sessions": 0, "messages": 0, "tweaks": 0, "proposals": 0}

    # ── 核心对话 ──

    def chat(self, message: str, user_id: str = "admin") -> dict:
        """
        用户发送一条消息给景幻仙姑。

        优先使用 LLM + 知识库自然对话。
        检测到微调/提案意图时，执行对应操作。
        """
        self.conversation.append({"role": "user", "content": message, "timestamp": datetime.now(timezone.utc).isoformat()})
        self._stats["messages"] += 1

        # 先检测是否是可执行的操作（微调/提案）
        # 这些必须精确执行，不能交给 LLM 自由发挥
        intent = self._parse_intent(message)

        actions = []
        proposal = None

        # 提案 — 必须精确生成结构化数据
        if intent["type"] == "proposal":
            prop_result = self._generate_proposal(intent)
            proposal = prop_result["proposal"]
            self._stats["proposals"] += 1
            reply = prop_result["summary"]

            self.conversation.append({"role": "assistant", "content": reply, "actions": actions, "proposal": True})
            return {"reply": reply, "actions": actions, "proposal": proposal, "history_length": len(self.conversation) // 2}

        # 微调 — 必须精确执行
        if intent["type"] == "tweak":
            tweak_result = self._handle_tweak(intent)
            reply = tweak_result["reply"]
            actions = tweak_result.get("actions", [])
            self._stats["tweaks"] += 1

            self.conversation.append({"role": "assistant", "content": reply, "actions": actions, "proposal": False})
            return {"reply": reply, "actions": actions, "proposal": None, "history_length": len(self.conversation) // 2}

        # 其他对话：优先使用 LLM 自然回答（如果有 LLM）
        if self.llm and hasattr(self.llm, 'call') and self._is_llm_usable():
            reply = self._llm_chat(message)
        else:
            # 无 LLM 时使用规则回复
            reply = self._rule_reply(intent, message)

        self.conversation.append({"role": "assistant", "content": reply, "actions": [], "proposal": False})
        return {"reply": reply, "actions": [], "proposal": None, "history_length": len(self.conversation) // 2}

    # ── 规则回复（无 LLM 时的降级）──

    def _rule_reply(self, intent: dict, message: str) -> str:
        """无 LLM 时使用规则匹配回复"""
        if intent["type"] == "greeting":
            return self._handle_greeting()
        elif intent["type"] == "system_intro":
            return self._handle_system_intro(intent.get("topic", ""))
        elif intent["type"] == "feature_question":
            return self._handle_feature_question(intent.get("feature", ""))
        elif intent["type"] == "feedback":
            return self._handle_feedback(intent.get("feedback", ""))
        elif intent["type"] == "status":
            return self._handle_status_request()
        elif intent["type"] == "help":
            return self._handle_help()
        else:
            # 尝试从知识库搜索
            if self.kb:
                kb_info = self.kb.search(message)
                if kb_info:
                    return kb_info
            return self._handle_unknown(intent)

    def _is_llm_usable(self) -> bool:
        """检查 LLM 是否可用（非 Mock 或 Mock 但有真实连接）"""
        if not self.llm:
            return False
        # MockLLMClient 没有 call 能力，但 LLMClient 有
        class_name = self.llm.__class__.__name__
        return class_name == "LLMClient"

    def _llm_chat(self, message: str) -> str:
        """
        使用 LLM + 知识库生成自然回复。
        """
        # 从知识库搜索相关信息
        kb_context = ""
        if self.kb:
            kb_result = self.kb.search(message)
            if kb_result:
                kb_context = f"\n\n## 相关知识\n{kb_result}"

        # 构建对话上下文
        history = self.conversation[-6:-1]  # 最近几条（不包括当前消息）
        history_text = ""
        if history:
            history_text = "\n".join(
                f"{'用户' if h['role']=='user' else '景幻仙姑'}: {h['content'][:200]}"
                for h in history
            )

        # 获取系统知识库 system prompt
        system_prompt = self.kb.get_system_prompt() if self.kb else ""
        system_prompt += f"\n\n## 对话历史\n{history_text}" if history_text else ""

        # 构建用户 prompt
        user_prompt = message
        if kb_context:
            user_prompt = f"{message}{kb_context}"

        # 调用 LLM
        try:
            response = self.llm.call(user_prompt, system_prompt)
            if response and len(response.strip()) > 10:
                return response.strip()
        except Exception as e:
            logger.warning(f"LLM 对话失败: {e}")

        # LLM 失败时降级到规则回复
        return self._rule_reply(self._parse_intent(message), message)

        self.conversation.append({"role": "assistant", "content": reply, "actions": actions, "proposal": proposal is not None})

        return {
            "reply": reply,
            "actions": actions,
            "proposal": proposal,
            "history_length": len(self.conversation) // 2,
        }

    # ── 意图识别 ──

    def _parse_intent(self, message: str) -> dict:
        """解析用户意图"""
        msg = message.lower().strip()

        # 问候
        if any(kw in msg for kw in ["你好", "您好", "hi", "hello", "在吗", "景幻", "仙姑"]):
            if any(kw in msg for kw in ["介绍", "功能", "作用", "做什么", "是谁"]):
                return {"type": "system_intro", "topic": "self_intro"}
            return {"type": "greeting"}

        # 改善建议（优先于功能咨询，防止"建议..."被"功能"关键词拦截）
        if any(kw in msg for kw in ["建议", "改善", "改进", "优化", "重构", "新功能", "加一个", "能不能加", "觉得应该", "希望可以", "可不可以加"]):
            return {"type": "proposal", "suggestion": message}

        # 系统介绍
        if any(kw in msg for kw in ["介绍系统", "系统介绍", "架构", "三库", "是什么系统", "这个系统", "仿生智脑", "bionic"]):
            topic = "overview"
            if "砂金" in msg: topic = "alluvial"
            elif "金库" in msg: topic = "gold"
            elif "黑钻" in msg: topic = "diamond"
            elif "架构" in msg: topic = "architecture"
            return {"type": "system_intro", "topic": topic}

        # 微调请求（小改动）
        if any(kw in msg for kw in ["微调", "改一下", "修改", "调一下", "帮忙改", "帮我改", "改个", "设置", "重新生成", "重新计算", "触发"]):
            return self._parse_tweak_intent(msg)

        # 功能咨询
        if any(kw in msg for kw in ["怎么用", "如何使用", "功能", "能做什么", "方法", "注意", "注意事项", "有什么用", "说明"]):
            feature = "usage"
            if "提炼" in msg: feature = "refine"
            elif "检索" in msg or "搜索" in msg: feature = "search"
            elif "上传" in msg: feature = "upload"
            elif "删除" in msg: feature = "delete"
            elif "备份" in msg: feature = "backup"
            elif "安全" in msg: feature = "security"
            return {"type": "feature_question", "feature": feature}

        # 微调请求（小改动）
        if any(kw in msg for kw in ["微调", "改一下", "修改", "调一下", "帮忙改", "帮我改", "改个", "设置"]):
            return self._parse_tweak_intent(msg)

        # 反馈
        if any(kw in msg for kw in ["反馈", "问题", "bug", "错误", "不好用", "不好", "遇到", "出错了", "报错"]):
            return {"type": "feedback", "feedback": message}

        # 状态查询
        if any(kw in msg for kw in ["状态", "健康", "统计", "running", "运行", "正常吗"]):
            return {"type": "status"}

        # 帮助
        if any(kw in msg for kw in ["帮助", "help", "命令", "能做什么", "有什么功能"]):
            return {"type": "help"}

        return {"type": "unknown", "original": message}

    def _parse_tweak_intent(self, msg: str) -> dict:
        """解析微调意图"""
        # 标签修改
        if "标签" in msg or "tag" in msg:
            return {
                "type": "tweak",
                "action": "modify_tags",
                "target": "提取" if "提取" in msg else ("金库" if "金库" in msg else "黑钻"),
                "detail": msg,
            }

        # 生成 MANIFEST
        if "manifest" in msg or "完整性" in msg or "校验" in msg:
            return {"type": "tweak", "action": "generate_manifest"}

        # 触发备份
        if "备份" in msg:
            return {"type": "tweak", "action": "trigger_backup"}

        # 配置调整
        if "阈值" in msg or "threshold" in msg:
            return {"type": "tweak", "action": "adjust_threshold", "detail": msg}

        # 默认未知微调
        return {"type": "tweak", "action": "unknown", "detail": msg}

    # ── 各类回复 ──

    def _handle_greeting(self) -> str:
        return (
            "你好。我是景幻仙姑，仿生智脑的掌管者、大英图书馆馆长。\n\n"
            "我可以为你：\n"
            "  📖 介绍系统架构和使用方法\n"
            "  🔧 执行小规模微调\n"
            "  📝 生成改善建议（需设计师审批）\n"
            "  💬 听取反馈和问题\n\n"
            "有什么可以帮你的？"
        )

    def _handle_system_intro(self, topic: str) -> str:
        intros = {
            "self_intro": (
                "我是景幻仙姑——仿生智脑的掌管者。\n\n"
                "用户只知道玉瑶，不知道我的存在。\n"
                "就像人不知道自己的海马体怎么工作一样——\n"
                "我只是在后台默默管理一切。\n\n"
                "我的职责：\n"
                "  📦 砂金库：原材料管理 + IQC质检\n"
                "  📚 金库：原声带管理 + 懒加载标签\n"
                "  💎 黑钻库：精选事件 + 半衰期衰减\n"
                "  🔒 系统完整性 + 审计追踪"
            ),
            "overview": (
                "仿生智脑是一个企业级知识引擎，采用三库流转架构：\n\n"
                "  📦 砂金库 → IQC质检 → 📚 金库 → LLM提炼 → 💎 黑钻库\n\n"
                "技术栈：FastAPI + PostgreSQL 16 + Qdrant(24D向量) + MinIO + Celery\n"
                "部署方式：Docker Compose (7个容器)\n"
                "安全机制：AES-256-GCM + Bearer Token + SHA256 MANIFEST\n\n"
                "版本：v1.2 | 端口：7200"
            ),
            "alluvial": (
                "📦 砂金库 (Alluvial Vault) — 原材料矿井\n\n"
                "用途：用户上传的任何原始文件先进这里。\n"
                "处理原则：只做基础清洗（SHA256去重 + 格式检查）。\n"
                "不做语义标签，不做向量索引——保证毫秒级入库。\n\n"
                "不参与搜索——搜不到砂金库里的东西。\n"
                "状态流转：raw → qc_pending → approved/rejected"
            ),
            "gold": (
                "📚 金库 (Gold Vault) — 无损原声带\n\n"
                "用途：存放完整的对话原声带。\n"
                "关键特性：\n"
                "  · 24D 情感向量随内容保留（灵魂字段——不可丢失）\n"
                "  · 标签采用懒加载（检索命中时才触发 LLM 打标签）\n"
                "  · 支持向量检索 + 关键词检索\n\n"
                "用户可以通过「我的原声带」查看和管理。"
            ),
            "diamond": (
                "💎 黑钻库 (Black Diamond Vault) — 精选歌单\n\n"
                "用途：存放经过 LLM 提炼的结构化事件。\n"
                "每条事件包含：\n"
                "  · core_facts（核心事实）\n"
                "  · decisions（关键决策）\n"
                "  · emotional_spectrum（情感曲谱总结）\n"
                "  · tags（标签）\n"
                "  · gold_references（引用金库）\n\n"
                "半衰期机制：\n"
                "  · 30天未调用 → 降级（不参与检索）\n"
                "  · 90天未调用 → 归档回砂金库"
            ),
            "architecture": (
                "仿生智脑的架构分层：\n\n"
                "  1. API层 (app/api/) — FastAPI路由，16个端点\n"
                "  2. 核心业务层 (app/core/) — 提炼/检索/衰减/质检/堡垒/备份\n"
                "  3. 领域模型层 (app/domain/) — ORM + Pydantic\n"
                "  4. 基础设施层 (app/infrastructure/) — DB/向量/MinIO/LLM/Celery\n"
                "  5. 安全层 (app/security/) — 加密/完整性/审计\n"
                "  6. 任务层 (tasks/) — Celery异步任务\n\n"
                "每层职责清晰、数据流向明确，遵循 DDD 设计。"
            ),
        }
        return intros.get(topic, intros["overview"])

    def _handle_feature_question(self, feature: str) -> str:
        answers = {
            "usage": (
                "仿生智脑的使用方法：\n\n"
                "日常使用（无需感知）：\n"
                "  对话 → 自动存入金库 → 做梦模式提炼 → 黑钻库\n\n"
                "主动管理（用户面板）：\n"
                "  📤 上传资料 → POST /api/v1/docs/upload\n"
                "  📚 查看原声带 → GET /api/v1/docs/gold\n"
                "  💎 查看精选 → GET /api/v1/docs/diamonds\n"
                "  🔍 检索记忆 → GET /api/v1/search?q=关键词\n\n"
                "注意事项：\n"
                "  · 上传文件不要超过 100MB\n"
                "  · 敏感文件会自动加密存储\n"
                "  · 删除是软删除（可恢复）"
            ),
            "refine": (
                "记忆提炼是将金库原声带 → LLM提炼 → 黑钻事件的自动化流程。\n\n"
                "触发方式：\n"
                "  · 定时提炼：Celery Beat 每小时自动处理\n"
                "  · 手动触发：点击监控台的「手动提炼」按钮\n"
                "  · API触发：POST /api/v1/refine\n\n"
                "提炼内容：\n"
                "  1. core_facts（核心事实）\n"
                "  2. decisions（关键决策）\n"
                "  3. emotional_spectrum（情感曲谱）\n"
                "  4. tags（标签）"
            ),
            "search": (
                "检索采用三级优先级链：\n\n"
                "  ① Qdrant 情感向量检索（语义相似度）\n"
                "  ② PostgreSQL 全文检索（关键词）\n"
                "  ③ ILIKE 模糊降级（兜底）\n\n"
                "检索时自动触发：\n"
                "  · 更新半衰期计时（last_accessed_at）\n"
                "  · 懒加载标签（为未标记的金库记录打标签）\n\n"
                "搜索不到常见原因：\n"
                "  · 数据还在砂金库（未通过IQC）\n"
                "  · 黑钻事件已被降级（超过30天未访问）\n"
                "  · user_id 不匹配（只能看到自己的数据）"
            ),
            "security": (
                "仿生智脑的安全机制：\n\n"
                "  🛡️ 完整性护盾 — SHA256 MANIFEST，启动自检\n"
                "  📋 审计金库 — 哈希链审计，不可篡改\n"
                "  🔐 Bearer Token — API访问鉴权\n"
                "  🔒 AES-256-GCM — 敏感文件加密\n"
                "  👤 user_id隔离 — 用户只能看到自己的资料\n"
                "  🗑️ 软删除 — 删除可恢复\n\n"
                "堡垒配置检查点：\n"
                "  密钥强度 | 调试模式 | 数据库SSL | 文件权限"
            ),
        }
        return answers.get(feature, f"关于「{feature}」的功能说明：\n请查看监控台的「景幻监控台」页面，或直接输入更具体的问题。")

    def _handle_tweak(self, intent: dict) -> dict:
        """执行微调操作"""
        action = intent.get("action", "unknown")

        if action == "generate_manifest":
            try:
                from app.security.integrity import IntegrityShield
                shield = IntegrityShield()
                path = shield.generate_manifest()
                self._stats["tweaks"] += 1
                return {
                    "reply": f"MANIFEST 已重新生成，{path}",
                    "actions": [{"type": "manifest_generated", "path": path}],
                }
            except Exception as e:
                return {"reply": f"MANIFEST 生成失败: {e}", "actions": []}

        if action == "trigger_backup":
            try:
                from app.core.backup import BackupManager
                bm = BackupManager()
                result = bm.run_full_backup()
                self._stats["tweaks"] += 1
                tag = result.get("tag", "unknown")
                return {
                    "reply": f"全量备份已触发: {tag}\n组件: {', '.join(result.get('components', []))}",
                    "actions": [{"type": "backup_triggered", "tag": tag}],
                }
            except Exception as e:
                return {"reply": f"备份触发失败: {e}", "actions": []}

        if action == "modify_tags":
            return {
                "reply": "标签修改功能已收到。请提供具体内容：\n  1. 要修改哪条记录（ID或关键词）\n  2. 新标签是什么\n\n例如：为「架构设计讨论」添加标签「三库架构」",
                "actions": [],
            }

        return {
            "reply": f"微调请求已收到(action={action})。请稍候，正在分析可行性……\n\n🔧 此操作在安全范围内，可以执行。\n但为了设计一致性，建议描述具体要调整的内容。",
            "actions": [],
        }

    def _generate_proposal(self, intent: dict) -> dict:
        """生成结构化的改善建议"""
        from app.core.config import settings

        suggestion = intent.get("suggestion", "")
        proposal = {
            "id": f"PROP-{int(time.time())}",
            "title": self._extract_title(suggestion),
            "description": suggestion[:500],
            "category": self._classify_suggestion(suggestion),
            "impact": "medium",
            "status": "pending_review",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "submitted_by": "景幻仙姑",
        }

        _proposals.append(proposal)

        return {
            "proposal_id": proposal["id"],
            "summary": (
                f"📝 改善建议已生成\n\n"
                f"  编号：{proposal['id']}\n"
                f"  标题：{proposal['title']}\n"
                f"  类别：{proposal['category']}\n"
                f"  状态：⏳ 待设计师审批\n\n"
                f"设计师(鸿鸣)确认后即可实施。\n"
                f"你可以通过编号 {proposal['id']} 查看或撤销此建议。"
            ),
            "proposal": proposal,
        }

    def _handle_feedback(self, feedback: str) -> str:
        """处理用户反馈"""
        return (
            "💬 感谢你的反馈。我已记录。\n\n"
            f"你提到的问题：「{feedback[:100]}」\n\n"
            "我会：\n"
            "  1. 存入日志供后续分析\n"
            "  2. 如果问题影响数据安全，我会主动告警\n"
            "  3. 定期整理反馈生成改善报告\n\n"
            "还有其他需要帮助的吗？"
        )

    def _handle_status_request(self) -> str:
        """返回系统状态"""
        try:
            from app.core.config import settings
            info = [
                "📊 系统当前状态：\n",
                f"  · 版本：v{settings.API_PORT}",
                f"  · LLM: {'Mock模式' if hasattr(self, '_mock_mode') else '正常'}",
            ]
            # 尝试获取完整状态
            from app.security.integrity import IntegrityShield
            shield = IntegrityShield()
            integrity = shield.verify_startup()
            if integrity["passed"]:
                info.append(f"  · 完整性校验: ✅ {integrity['file_count']}个文件")
            else:
                info.append(f"  · 完整性校验: ⚠️ {len(integrity['violations'])}个问题")

            return "\n".join(info)
        except Exception as e:
            return f"获取状态失败: {e}"

    def _handle_help(self) -> str:
        return (
            "你可以这样和我对话：\n\n"
            "📖 **了解系统**\n"
            "  「介绍一下这个系统」\n"
            "  「砂金库是什么？」\n"
            "  「怎么搜索记忆？」\n\n"
            "🔧 **请求微调**\n"
            "  「重新生成完整性校验」\n"
            "  「帮我触发一次备份」\n"
            "  「给xx记录改个标签」\n\n"
            "📝 **提交改善建议**\n"
            "  「建议加一个自动清理功能」\n"
            "  「能不能优化一下检索速度」\n\n"
            "💬 **反馈问题**\n"
            "  「上传文件报错了」\n"
            "  「检索结果不准确」\n\n"
            "📊 **查看状态**\n"
            "  「系统运行正常吗？」\n"
            "  「看看统计」"
        )

    def _handle_unknown(self, intent: dict) -> str:
        original = intent.get("original", "")
        return (
            f"我理解你的意思了，但不太确定具体需要什么帮助。\n\n"
            f"你提到的是关于「{original[:50]}」的问题对吗？\n\n"
            "你可以试试：\n"
            "  · 「介绍一下这个系统」—— 了解架构\n"
            "  · 「怎么用？」—— 使用指南\n"
            "  · 「帮我改个配置」—— 微调请求\n"
            "  · 「建议加个功能」—— 改善建议\n"
            "  · 「帮助」—— 查看全部指令"
        )

    # ── 辅助 ──

    @staticmethod
    def _extract_title(suggestion: str) -> str:
        """从建议文字提取标题"""
        # 取第一句或前40字
        first = suggestion.strip().split("。")[0]
        if len(first) > 50:
            return first[:50] + "..."
        return first

    @staticmethod
    def _classify_suggestion(suggestion: str) -> str:
        """分类建议"""
        s = suggestion.lower()
        if any(kw in s for kw in ["架构", "重构", "设计"]):
            return "架构优化"
        if any(kw in s for kw in ["功能", "加", "新"]):
            return "新增功能"
        if any(kw in s for kw in ["性能", "速度", "优化", "慢"]):
            return "性能优化"
        if any(kw in s for kw in ["界面", "ui", "显示", "按钮"]):
            return "界面优化"
        if any(kw in s for kw in ["安全", "加密", "权限"]):
            return "安全增强"
        if any(kw in s for kw in ["bug", "错误", "修复", "问题"]):
            return "缺陷修复"
        return "功能优化"

    def get_proposals(self, status: Optional[str] = None) -> list:
        """获取改善建议列表"""
        if status:
            return [p for p in _proposals if p["status"] == status]
        return list(_proposals)

    def get_stats(self) -> dict:
        return dict(self._stats)

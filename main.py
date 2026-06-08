"""
仿生智脑 v1.1 · 入口 (FastAPI Application)

启动方式:
  # 开发模式（本地调试）：
  python main.py

  # 生产模式：
  uvicorn main:app --host 0.0.0.0 --port 7200 --workers 4

  # Docker（推荐）：
  docker compose up -d
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.deps import db_manager
from app.api.routes import router as api_router, init_services
from app.infrastructure.vector_store import VectorStore
from app.infrastructure.llm_client import LLMClient
from app.core.refiner import MemoryConsolidator
from app.core.retrieval import HybridSearchService

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("bionic")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """🔒 应用生命周期管理——启动时执行安全初始化"""

    # ═══════════════════════════════════════════════════════════════
    # 阶段 0：完整性校验
    # ═══════════════════════════════════════════════════════════════
    logger.info("═" * 50)
    logger.info("仿生智脑 (Bionic Cognitive Engine) v1.1 启动")
    logger.info("景幻仙姑 — 仿生智脑的掌管者、大英图书馆馆长")
    logger.info("═" * 50)

    _run_integrity_check()
    _run_fortress_check()
    _init_audit_vault()

    # ═══════════════════════════════════════════════════════════════
    # 阶段 1：数据库初始化
    # ═══════════════════════════════════════════════════════════════
    try:
        await db_manager.initialize()
        logger.info("[OK] 数据库已连接")
    except Exception as e:
        logger.warning(f"[WARN] 数据库连接失败: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 阶段 2：Qdrant 向量库
    # ═══════════════════════════════════════════════════════════════
    vs = VectorStore()
    vs_ok = vs.initialize()
    if vs_ok:
        logger.info("[OK] Qdrant 向量存储已就绪")
    else:
        logger.warning("[WARN] Qdrant 未就绪，检索将降级为纯文本")

    # ═══════════════════════════════════════════════════════════════
    # 阶段 3：业务服务初始化
    # ═══════════════════════════════════════════════════════════════
    llm = LLMClient()
    consolidator = MemoryConsolidator(llm, vs)
    searcher = HybridSearchService(vs)

    # 注入到 API 路由 + 注入安全模块
    init_services(vs, consolidator, searcher)
    _inject_security_services()

    logger.info("[OK] 业务服务已就绪")

    # ── 生成 MANIFEST（开发模式自动生成，生产模式依赖预生成）──
    if settings.API_DEBUG:
        _ensure_manifest()

    yield  # ← 应用在此运行

    # ═══════════════════════════════════════════════════════════════
    # 关闭
    # ═══════════════════════════════════════════════════════════════
    await db_manager.close()
    logger.info("仿生智脑已关闭。景幻仙姑回归虚无。")


# ═══════════════════════════════════════════════════════════════
# FastAPI 应用
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="仿生智脑 (Bionic Cognitive Engine)",
    description="景幻仙姑的大英图书馆——三库流转的知识引擎 · 加固版",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS（仅允许玉瑶跨域调用，生产环境应指定具体域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 注册路由 ──
app.include_router(api_router)


# ── 根路径 ──
@app.get("/")
async def root():
    return {
        "service": "仿生智脑 (Bionic Cognitive Engine)",
        "version": "1.1.0",
        "keeper": "景幻仙姑",
        "api": "/api/v1/health",
        "security": "/api/v1/security/status",
    }


# ═══════════════════════════════════════════════════════════════
# 安全基础设施（启动时初始化）
# ═══════════════════════════════════════════════════════════════

_audit_vault = None
_fortress = None


def _run_integrity_check():
    """执行完整性校验"""
    try:
        from app.security.integrity import IntegrityShield
        shield = IntegrityShield()
        result = shield.verify_startup()
        if result["passed"]:
            logger.info(f"[OK] 代码完整性校验通过 ({result['file_count']} 文件)")
        else:
            logger.warning(f"[WARN] 代码完整性校验: {len(result['violations'])} 个问题")
            for v in result["violations"]:
                logger.warning(f"  [{v['type']}] {v.get('file','')} - {v['detail']}")
    except ImportError:
        logger.warning("[WARN] 完整性护盾未加载（首次部署请生成 MANIFEST）")
    except Exception as e:
        logger.error(f"[ERR] 完整性校验异常: {e}")


def _run_fortress_check():
    """执行堡垒安全检查"""
    try:
        from app.core.fortress import FortressGuard
        global _fortress
        _fortress = FortressGuard()
        report = _fortress.run_all_checks()
        s = report["summary"]
        if s["errors"] > 0:
            logger.warning(f"[WARN] 堡垒检查: {s['errors']} 个致命问题")
        elif s["warnings"] > 0:
            logger.info(f"[OK] 堡垒检查: {s['passed']} 通过, {s['warnings']} 个建议")
        else:
            logger.info(f"[OK] 堡垒检查全部通过 ({s['passed']}/{s['total']})")
    except Exception as e:
        logger.error(f"[ERR] 堡垒检查异常: {e}")


def _init_audit_vault():
    """初始化审计金库"""
    try:
        from app.security.audit import AuditVault
        global _audit_vault
        _audit_vault = AuditVault()
        logger.info("[OK] 审计金库已就绪")
    except Exception as e:
        logger.error(f"[ERR] 审计金库初始化失败: {e}")


def _ensure_manifest():
    """确保 MANIFEST 存在（开发模式自动生成）"""
    manifest_path = Path(__file__).parent / "MANIFEST.json"
    if not manifest_path.exists():
        try:
            from app.security.integrity import IntegrityShield
            shield = IntegrityShield()
            shield.generate_manifest()
            logger.info("[OK] MANIFEST 已自动生成（开发模式）")
        except Exception as e:
            logger.warning(f"[WARN] MANIFEST 生成跳过: {e}")


def _inject_security_services():
    """将安全模块注入到 API 路由"""
    from app.api.routes import _set_audit_vault, _set_fortress
    if _audit_vault:
        _set_audit_vault(_audit_vault)
    if _fortress:
        _set_fortress(_fortress)


def get_audit_vault():
    """供其他模块获取审计金库实例"""
    return _audit_vault


def get_fortress():
    """供其他模块获取堡垒守卫实例"""
    return _fortress

"""
=============================================================================
全局配置常量 — RAG 系统核心配置中心
=============================================================================

此文件集中管理所有可调参数，方便快速切换不同环境（云端 API / 本地 vLLM）。

【使用说明】
  1. 云端模式（默认）：使用 DeepSeek API，无需额外配置
  2. 本地模式：启动 vLLM 服务后，修改 BASE_URL 指向本地地址即可
     $ vllm serve your-model-name --host 0.0.0.0 --port 8000
  3. 环境变量覆盖：所有 LLM 配置均支持通过环境变量覆盖

=============================================================================
"""

import os

# ============================================================
# LLM API 配置 — 核心：可轻易切换云端/本地
# ============================================================

# --------------------------------------------------------------------------
# 方案一：DeepSeek API（默认云端方案）
# --------------------------------------------------------------------------
# - 认证信息从 ~/.bashrc 中读取环境变量
# - 模型 deepseek-v4-pro 用于复杂推理，deepseek-v4-flash 用于子任务
# --------------------------------------------------------------------------
DEEPSEEK_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEEPSEEK_API_KEY = os.environ.get(
    "ZHIPU_API_KEY",
    "1fe4c37fd3264ffa9f535fec9d0fc96b.UtiuwWTVuFofYHnB"  # ← 智谱 GLM-4.7-Flash 默认 Key
)
DEEPSEEK_MODEL = "glm-4.7-flash"  # 智谱 GLM-4.7-Flash 免费模型

# --------------------------------------------------------------------------
# 方案二：本地 vLLM 自建 OpenAI 兼容服务（成本零，需本地 GPU）
# --------------------------------------------------------------------------
# 如需使用本地 vLLM 服务，请：
#   1. 启动 vLLM: vllm serve /path/to/model --host 0.0.0.0 --port 8000
#   2. 将下方 BASE_URL 改为 "http://localhost:8000/v1"
#   3. 将 API_KEY 改为 "not-needed"（vLLM 默认不验证 Key）
#   4. 将 MODEL_NAME 改为你 vLLM 部署的模型名称
# --------------------------------------------------------------------------
# BASE_URL   = "http://localhost:8000/v1"
# API_KEY    = "not-needed"
# MODEL_NAME = "your-vllm-model-name"

# ============================================================
# 生效配置（环境变量 > 下方默认值）
# ============================================================
# 当前默认：本地 vLLM OpenAI 兼容服务
# 如需切回云端 DeepSeek API，设置环境变量：
#   export LLM_BASE_URL="https://api.deepseek.com/anthropic"
#   export LLM_API_KEY="sk-your-deepseek-key"
# ============================================================
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1")
API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-AWQ")

# ============================================================
# 向量库 & 嵌入模型配置
# ============================================================

# ChromaDB 持久化目录（向量数据落盘位置）
CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vector_db"
)

# HuggingFace 嵌入模型名称
# - BAAI/bge-small-zh-v1.5: 中文专优模型，512维（当前默认 — 中文 SDK 文档场景最优）
# - all-MiniLM-L6-v2: 英文轻量模型，384维（回退备选 — 英文场景更快）
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 是否在 HuggingFaceEmbeddings 加载失败时，
# 自动回退到 ChromaDB 内置的 ONNX 方案（ONNXMiniLM_L6_V2）
# 设为 True 保证极端环境下的鲁棒性 —— 绝不升级依赖
FALLBACK_TO_ONNX = True

# ============================================================
# HuggingFace 镜像配置（国内加速）
# ============================================================
# 默认使用国内镜像 hf-mirror.com，避免访问 huggingface.co 超时
# 如需使用官方源，设置环境变量: HF_ENDPOINT=https://huggingface.co
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
# 强制执行，覆盖系统中可能存在的官方源地址
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
# 离线模式：禁止 sentence-transformers 在启动时尝试连接 huggingface.co 验证模型文件
# bge-small-zh-v1.5 模型已完整缓存于 ~/.cache/huggingface/hub/
os.environ["HF_HUB_OFFLINE"] = "1"
# 注意：不启用 HF_HUB_ENABLE_HF_TRANSFER，避免因缺少 hf_transfer 包导致下载失败

# 嵌入向量设备：优先 GPU，不可用时回退 CPU
# 注意：使用 torch.cuda.is_available() 做实际可用性检查，
# 而非仅依赖 CUDA_VISIBLE_DEVICES 环境变量（可能指向不可用 GPU）
try:
    import torch
    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    EMBEDDING_DEVICE = "cpu"

# ============================================================
# PDF 文档处理配置
# ============================================================

# PDF 文件存放目录
PDF_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)

# 文本分块参数
# chunk_size=300: 细粒度切片，每个 SDK 函数定义+示例约 200-400 字符，
#                 300 确保单个函数不会被跨切片截断，同时减少噪声信息混入
# chunk_overlap=50: 相邻块之间重叠 50 个字符 (~17%)，保证函数边界不丢失
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ── v4 Dual Indexing 切片配置 (ADR-15) ──
CHUNK_MODE = "v4_dual"  # "v4_dual" | "v3_legacy"
PARENT_CHUNK_SIZE = 1000   # H2 章节级父层切片（粗召回）
CHILD_CHUNK_SIZE = 400     # H3/H4 函数级子层切片（精匹配，API 原子）

# ── v4 GPU 批量加速 (ADR-16) ──
EMBEDDING_BATCH_SIZE = 64   # GPU 批量推理大小（A100 建议 64-128）

# ============================================================
# RAG 检索配置
# ============================================================

# 检索时返回的 Top-K 文档片段数
# k=5: 每次检索返回最相关的 5 个文本块
#      增加到 5 以提高召回覆盖率，降低关键函数漏检概率
RETRIEVAL_K = 8

# 相似度阈值 — ChromaDB cosine 距离上限
# 使用 similarity_search_with_score 进行距离过滤：
#   余弦距离范围: 0 (完全相同) ~ 2 (完全相反), 1 为正交无关
#   threshold=0.68: 在 0.78 基础上大幅放宽 0.10，解决复合查询（如"上电+回零"）
#     中多关键词导致的语义稀释问题。实测数据：
#     - OpenR6 set_robot_power_on + set_robot_arm_home 距离 ~0.65-0.72
#       在 0.68 阈值下可完整召回（0.78 会将 ~0.72 的切片误杀）
#     - Q3 get_robot_pose 距离 0.72→0.68 可召回
#     - Q1 robot_Power_on 距离 0.43→0.68 可召回
#   设为 None 可禁用阈值过滤
SIMILARITY_THRESHOLD = 0.68

# ============================================================
# 产品线动态路由配置 — Product-Aware RAG Isolation
# ============================================================
#
# 【设计目标】
# 不同产品（如 OpenR6、OpenC3）的 SDK 文档在语义上存在重叠（如上电、使能
# 等通用操作），但底层动态库、函数签名、参数结构完全不同。若将不同产品的
# 切片混合检索，极易导致 LLM 张冠李戴——用 OpenR6 的 py_dll 函数回答
# OpenC3 的六轴机械臂问题。
#
# 解决方案：入库时自动打标 product_id → 检索时物理隔离 → 未指定产品时
# 主动反问澄清。
#
# 【扩展方式】
# 新增产品线只需在 PRODUCT_MAPPING_RULES 和 PRODUCT_ROUTER_RULES 中
# 各追加一条配置即可，无需修改任何核心逻辑代码。

# ---- 产品识别打标规则（入库阶段：文件名 → product_id） ----
# 每条规则包含:
#   product_id:       产品唯一标识（存入 ChromaDB metadata）
#   filename_patterns: 文件名关键词列表（任一命中即匹配，不区分大小写）
#   content_keywords:  文档内容关键词（辅助确认，可选）
PRODUCT_MAPPING_RULES = [
    {
        "product_id": "OpenR6",
        "filename_patterns": ["OpenR6", "openr6", "R6", "windows系统"],
        "content_keywords": ["py_dll", "Robot_.*", "robot_Power_on", "windows"],
    },
    {
        "product_id": "OpenC3",
        "filename_patterns": ["OpenC3", "openc3", "六轴机械臂", "collrob", "六轴"],
        "content_keywords": ["六轴", "collrob", "OpenC3", "机械臂"],
    },
    {
        "product_id": "JAKA",
        "filename_patterns": ["JAKA", "jaka", "Zu", "MiniCab", "节卡"],
        "content_keywords": ["JAKA", "Zu", "MiniCab", "Modbus", "Profinet", "节卡"],
    },
]

# ---- 产品意图路由规则（查询阶段：用户 query → product_id） ----
# 每条规则包含:
#   product_id:    目标产品标识
#   keywords:      命中关键词列表（任一命中即锁定产品，不区分大小写）
#   priority:      优先级（数字越大越优先，用于解决关键词重叠冲突）
PRODUCT_ROUTER_RULES = [
    {
        "product_id": "OpenR6",
        "keywords": [
            "OpenR6", "openr6", "py_dll", "R6", "windows",
            "windows系统", "windows sdk",
        ],
        "priority": 10,
    },
    {
        "product_id": "OpenC3",
        "keywords": [
            "OpenC3", "openc3", "collrob", "六轴",
            "六轴机械臂", "OpenC3六轴",
        ],
        "priority": 10,
    },
    {
        "product_id": "JAKA",
        "keywords": [
            "JAKA", "jaka", "Zu", "MiniCab", "节卡", "Zu APP",
            "JAKA Zu", "Modbus", "Profinet",
            "VBrake", "vbrake", "Vbrake", "minicab",
            "TCP", "JOG", "jog", "tcp校准", "工具坐标系",
        ],
        "priority": 15,  # 🔴 最高优先：强特征词直接锁定 JAKA
    },
]

# 意图澄清回复模板（当用户未指定产品时使用）
PRODUCT_CLARIFICATION_PROMPT = (
    "请问您询问的是哪一款产品呢？（例如：{product_list}）\n"
    "不同产品的 SDK 动态库与函数接口有所不同，"
    "请告知具体型号以便为您提供准确的代码示例。"
)

# 主动澄清时的 HTTP 状态（200=正常返回让前端展示，非异常）
PRODUCT_CLARIFICATION_HTTP_STATUS = 200

# ============================================================
# Web 服务配置
# ============================================================

HOST = "0.0.0.0"  # 0.0.0.0 允许外部访问
PORT = 7860

# 最大上传文件大小（字节），默认 50MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# ============================================================
# GPU 智能自适应部署 — Dynamic GPU Detection
# ============================================================
#
# 通过 nvidia-smi 实时探测所有 GPU 的空闲显存，
# 自动选择剩余显存最大的 GPU 作为 vLLM 推理目标。
#
# 优先级：
#   1. 环境变量 VLLM_GPU_ID（手动覆盖，最高优先级）
#   2. nvidia-smi 自动探测（空闲显存最大者胜出）
#   3. 默认回退 GPU 0（nvidia-smi 不可用时）
#
# 使用方式：
#   from src.config import get_best_gpu, get_all_gpu_info
#   gpu_id = get_best_gpu()
#   gpus = get_all_gpu_info()

import logging
import subprocess
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# 最小需要的 GPU 空闲显存（MiB），低于此值认为该 GPU 不可用
MIN_FREE_MEMORY_MIB = 5120  # 5 GB — 1.5B 模型约需 3.7 GB

# 当前选定的 vLLM GPU（优先读环境变量，否则自动探测）
_VLLM_GPU_ID_FROM_ENV = os.environ.get("VLLM_GPU_ID")


def _parse_nvidia_smi() -> List[Dict[str, any]]:
    """
    调用 nvidia-smi 并解析为结构化 GPU 信息列表。

    每项包含:
        index, name, memory_used_mib, memory_total_mib, memory_free_mib,
        temperature, power_w, utilization_pct
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free,"
                "temperature.gpu,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()

        if not output:
            return []

        gpus = []
        for line in output.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": float(parts[2]),
                "memory_total_mib": float(parts[3]),
                "memory_free_mib": float(parts[4]),
                "temperature": parts[5] if parts[5] else "N/A",
                "power_w": parts[6] if parts[6] else "N/A",
                "utilization_pct": parts[7] if len(parts) > 7 and parts[7] else "N/A",
            })
        return gpus

    except FileNotFoundError:
        logger.warning("nvidia-smi 未找到，无法探测 GPU 状态")
        return []
    except Exception as e:
        logger.warning(f"nvidia-smi 调用失败: {e}")
        return []


def get_all_gpu_info() -> List[Dict[str, any]]:
    """返回所有 GPU 的结构化信息列表（供 check_status.py 等工具使用）。"""
    return _parse_nvidia_smi()


def detect_best_gpu(min_free_mib: int = MIN_FREE_MEMORY_MIB) -> int:
    """
    在所有 GPU 中查找空闲显存最大的那一张。

    算法：
      1. nvidia-smi 查询每张 GPU 的 memory.free
      2. 过滤空闲显存 < min_free_mib 的 GPU
      3. 按空闲降序，返回第一名

    Args:
        min_free_mib: 最低空闲显存门槛（MiB），低于此值的 GPU 被排除

    Returns:
        GPU 索引（int），无可用的 GPU 时返回 -1
    """
    gpus = _parse_nvidia_smi()
    if not gpus:
        return 0  # nvidia-smi 不可用时回退 GPU 0

    best_idx = -1
    best_free = -1

    for gpu in gpus:
        free_mib = gpu["memory_free_mib"]
        if free_mib >= min_free_mib and free_mib > best_free:
            best_free = free_mib
            best_idx = gpu["index"]

    if best_idx >= 0:
        logger.info(
            f"🖥️ 智能 GPU 选择: GPU {best_idx} "
            f"(空闲 {best_free:.0f} MiB / {best_free/1024:.1f} GB)"
        )
    else:
        logger.warning(
            f"⚠️  所有 GPU 空闲显存均不足 {min_free_mib} MiB，"
            f"vLLM 部署可能失败"
        )

    return best_idx if best_idx >= 0 else 0  # 无可用的 GPU 时回退 0


def get_best_gpu() -> int:
    """
    获取当前应使用的 GPU 索引。

    优先级: 环境变量 VLLM_GPU_ID > nvidia-smi 自动探测 > 默认 0
    """
    if _VLLM_GPU_ID_FROM_ENV is not None:
        try:
            gpu_id = int(_VLLM_GPU_ID_FROM_ENV)
            logger.info(f"🖥️ 使用环境变量指定的 GPU: {gpu_id} (VLLM_GPU_ID)")
            return gpu_id
        except ValueError:
            logger.warning(f"VLLM_GPU_ID 值无效 '{_VLLM_GPU_ID_FROM_ENV}'，回退自动探测")
    return detect_best_gpu()


# 模块级常量：当前选定的 vLLM GPU（首次 import 时自动探测）
VLLM_GPU_ID = get_best_gpu()

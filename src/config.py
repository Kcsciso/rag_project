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
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

# ============================================================
# 向量库 & 嵌入模型配置
# ============================================================

# ChromaDB 持久化目录（向量数据落盘位置）
CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vector_db"
)

# HuggingFace 嵌入模型名称
# - all-MiniLM-L6-v2: 英文轻量模型，384维，速度快（默认）
# - BAAI/bge-small-zh-v1.5: 中文专优模型，512维（中文场景推荐替换）
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

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
# chunk_size=600: 每个文本块最大 600 个字符——在保留上下文完整性与检索精度间平衡
#                 增大到 600 防止 API 示例代码跨切片被截断
# chunk_overlap=100: 相邻块之间重叠 100 个字符
#                    加大重叠区确保关键函数定义不会恰好落在块边界上
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# ============================================================
# RAG 检索配置
# ============================================================

# 检索时返回的 Top-K 文档片段数
# k=5: 每次检索返回最相关的 5 个文本块
#      增加到 5 以提高召回覆盖率，降低关键函数漏检概率
RETRIEVAL_K = 5

# 相似度阈值 — ChromaDB cosine 距离上限
# 使用 similarity_search_with_score 进行距离过滤：
#   余弦距离范围: 0 (完全相同) ~ 2 (完全相反), 1 为正交无关
#   threshold=0.75: 基于 Qwen2.5-7B-Instruct + 机械臂 SDK 文档实测校准
#     - Q3 get_robot_pose 距离 0.72，0.70 误杀 → 0.75 可召回
#     - Q4 摄像头无关内容距离 0.68-0.76，0.75 可过滤大部分
#   设为 None 可禁用阈值过滤
SIMILARITY_THRESHOLD = 0.75

# ============================================================
# Web 服务配置
# ============================================================

HOST = "0.0.0.0"  # 0.0.0.0 允许外部访问（配合 ngrok 使用）
PORT = 8000

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

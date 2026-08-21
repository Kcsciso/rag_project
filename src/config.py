"""
=============================================================================
全局配置常量 — RAG 系统核心配置中心 (v32 定版)
=============================================================================

集中管理所有可调参数，涵盖：
  1. 双模型微服务 (vLLM :8001 文本生成 + Qwen2-VL :8005 视觉提取)
  2. 向量库与 BGE 嵌入配置 (GPU/CPU 自适应)
  3. 双轨切片容量分配 (SDK 轨 400/1000, GUI 轨 1500/2000)
  4. 检索超参数 (RRF, Autocut, 混合检索阈值)
  5. 产品物理隔离路由规则 (OpenR6, OpenC3, JAKA)
  6. GPU 显存智能探测与自适应选择
=============================================================================
"""

import os
import logging
import subprocess
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# 1. 基础服务端口与服务配置
# ============================================================

HOST = "0.0.0.0"
PORT = 8000
FASTAPI_PORT = PORT
FRONTEND_PORT = 8501
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 最大上传 50MB

# ============================================================
# 2. LLM 与 VLM 双模型微服务 API 配置
# ============================================================

# ── 方案一：本地主文本推理模型 (vLLM @ :8001) ──
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1")
API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
MODEL_NAME = os.environ.get(
    "LLM_MODEL_NAME",
    "/home/kasm-user/LLM/mo/models/Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/master"
)

# ── 方案二：本地多模态视觉提纯模型 (Qwen2-VL @ :8005) ──
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8005/v1")
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")

# ── 方案三：云端降级备选 (智谱 GLM-4.7-Flash) ──
DEEPSEEK_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEEPSEEK_API_KEY = os.environ.get(
    "ZHIPU_API_KEY",
    "1fe4c37fd3264ffa9f535fec9d0fc96b.UtiuwWTVuFofYHnB"
)
DEEPSEEK_MODEL = "glm-4.7-flash"

# ============================================================
# 3. 向量库 & 嵌入模型配置
# ============================================================

# ChromaDB 持久化目录
CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vector_db"
)

# KV 物理属性存储目录
KV_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kv_db"
)

# 嵌入模型: BAAI/bge-small-zh-v1.5 (512维)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
FALLBACK_TO_ONNX = True
EMBEDDING_BATCH_SIZE = 64

# 国内镜像与离线模式约束
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    import torch
    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    EMBEDDING_DEVICE = "cpu"

# ============================================================
# 4. 数据摄入与双轨切片容量配置 (ADR-15, v23, v32)
# ============================================================

PDF_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)

JAKA_MARKDOWN_PATH = os.path.join(
    PDF_DATA_DIR,
    "jaka_markdown/JAKA_Manual/auto/JAKA_Manual.md"
)

CHUNK_MODE = "v4_dual"

# ── SDK 专轨切片大小 ──
SDK_PARENT_CHUNK_SIZE = 1000   # H2 章节级父层（粗召回）
SDK_CHILD_CHUNK_SIZE = 400     # H3/H4 函数级子层（API 原子）

# ── GUI / JAKA 专轨切片大小 ──
GUI_PARENT_CHUNK_SIZE = 2000
GUI_CHILD_CHUNK_SIZE = 1500

# ── 通用向下兼容别名（避免导入报错） ──
PARENT_CHUNK_SIZE = SDK_PARENT_CHUNK_SIZE
CHILD_CHUNK_SIZE = SDK_CHILD_CHUNK_SIZE

# 通用默认回退切片大小
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ============================================================
# 5. RAG 混合检索与过滤配置
# ============================================================

RETRIEVAL_K = 10
SIMILARITY_THRESHOLD = 0.68

_AUTOCUT_MIN_K = 8
_AUTOCUT_MAX_K = 15
_MIN_SUB_QUERY_LEN = 2
_MAX_CONTEXT_CHARS_SDK = 8000
_MAX_CONTEXT_CHARS_GENERAL = 4000

# ============================================================
# 6. 产品线动态路由与隔离规则 (Product-Aware Isolation)
# ============================================================

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
        "priority": 15,
    },
]

PRODUCT_CLARIFICATION_PROMPT = (
    "请问您询问的是哪一款产品呢？（当前已支持：JAKA、OpenC3、OpenR6）\n"
    "不同产品的 SDK 接口与操作逻辑有所不同，请告知具体型号以便为您准确解答。"
)
PRODUCT_CLARIFICATION_HTTP_STATUS = 200

# ============================================================
# 7. GPU 显存智能自适应探测
# ============================================================

MIN_FREE_MEMORY_MIB = 5120  # 最低空闲要求 5GB
_VLLM_GPU_ID_FROM_ENV = os.environ.get("VLLM_GPU_ID")

def _parse_nvidia_smi() -> List[Dict[str, any]]:
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
    except Exception as e:
        logger.warning(f"nvidia-smi 探测不可用: {e}")
        return []

def get_all_gpu_info() -> List[Dict[str, any]]:
    return _parse_nvidia_smi()

def detect_best_gpu(min_free_mib: int = MIN_FREE_MEMORY_MIB) -> int:
    gpus = _parse_nvidia_smi()
    if not gpus:
        return 0

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
    return best_idx if best_idx >= 0 else 0

def get_best_gpu() -> int:
    if _VLLM_GPU_ID_FROM_ENV is not None:
        try:
            return int(_VLLM_GPU_ID_FROM_ENV)
        except ValueError:
            pass
    return detect_best_gpu()

VLLM_GPU_ID = get_best_gpu()
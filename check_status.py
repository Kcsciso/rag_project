#!/usr/bin/env python3
"""
=============================================================================
比邻星 (ProximaRAG) 统一服务健康状态检查脚本 (v4)
=============================================================================

检查范围：
  1. 本地 vLLM 推理服务（端口 8001）— 在线状态、已加载模型名称
  2. FastAPI 后端（比邻星，端口 7860）— 在线状态、向量库文档数
  3. v4 Parent-Child 双索引 — rag_v4_parent / rag_v4_child chunk 数量
  4. CUDA GPU 显存占用 — GPU 0 & GPU 1 的实时显存、温度、功率

使用方式：
  python check_status.py              # 一次性检查
  python check_status.py --watch 10   # 每 10 秒刷新（Ctrl+C 退出）
  python check_status.py --watch 5    # 每 5 秒刷新

依赖：
  - httpx（HTTP 请求）
  - nvidia-smi（GPU 状态查询）
  - chromadb（v4 索引检查）
=============================================================================
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# ============================================================
# 颜色输出（终端 ANSI）
# ============================================================

class Color:
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

def green(s):  return f"{Color.GREEN}{s}{Color.RESET}"
def red(s):    return f"{Color.RED}{s}{Color.RESET}"
def yellow(s): return f"{Color.YELLOW}{s}{Color.RESET}"
def cyan(s):   return f"{Color.CYAN}{s}{Color.RESET}"
def blue(s):   return f"{Color.BLUE}{s}{Color.RESET}"
def bold(s):   return f"{Color.BOLD}{s}{Color.RESET}"
def dim(s):    return f"{Color.DIM}{s}{Color.RESET}"

STATUS_OK  = green("● 在线")
STATUS_DOWN = red("● 离线")

# ============================================================
# 配置
# ============================================================

VLLM_BASE_URL      = "http://localhost:8001"
FASTAPI_BASE_URL   = "http://localhost:8000"

TIMEOUT_CONNECT = 3.0   # HTTP 连接超时（秒）
TIMEOUT_READ    = 5.0   # HTTP 读取超时（秒）

# ============================================================
# HTTP 客户端
# ============================================================

import httpx

_http_client: Optional[httpx.Client] = None

def get_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        # httpx 0.28+ requires explicit default timeout when overriding specific params
        _http_client = httpx.Client(
            timeout=httpx.Timeout(10.0, connect=TIMEOUT_CONNECT, read=TIMEOUT_READ),
        )
    return _http_client

# ============================================================
# 检查 vLLM 服务
# ============================================================

def check_vllm() -> Dict[str, Any]:
    """
    检查 vLLM OpenAI 兼容服务 (http://localhost:8001)。

    Returns:
        {
            "online": bool,
            "url": str,
            "model": str | None,
            "models_available": [str, ...],
            "gpu_id": int | None,       # vLLM 实际绑定的 GPU 索引
            "gpu_name": str | None,     # 该 GPU 的名称
            "error": str | None,
            "latency_ms": float,
        }
    """
    result = {
        "online": False,
        "url": f"{VLLM_BASE_URL}/v1",
        "model": None,
        "models_available": [],
        "gpu_id": None,
        "gpu_name": None,
        "error": None,
        "latency_ms": 0,
    }

    t0 = time.monotonic()
    try:
        resp = get_client().get(f"{VLLM_BASE_URL}/v1/models")
        result["latency_ms"] = round((time.monotonic() - t0) * 1000)

        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            result["online"] = True
            result["models_available"] = [m.get("id", "?") for m in models]
            result["model"] = result["models_available"][0] if result["models_available"] else None

            # --- 检测 vLLM 实际运行在哪个 GPU 上 ---
            gpu_id, gpu_name = _detect_vllm_process_gpu()
            result["gpu_id"] = gpu_id
            result["gpu_name"] = gpu_name
        else:
            result["error"] = f"HTTP {resp.status_code}"
    except httpx.ConnectError:
        result["error"] = "连接被拒绝（服务未启动或端口错误）"
    except httpx.TimeoutException:
        result["error"] = f"连接超时（>{TIMEOUT_CONNECT}s）"
    except Exception as e:
        result["error"] = str(e)

    return result


def _detect_vllm_process_gpu() -> Tuple[Optional[int], Optional[str]]:
    """
    通过查找 vLLM 进程的 CUDA_VISIBLE_DEVICES 环境变量，
    判断 vLLM 实际绑定到哪张 GPU。

    流程：
      1. 用 ss 找到占用端口 8001 的进程 PID
      2. 读取 /proc/<pid>/environ 中的 CUDA_VISIBLE_DEVICES 值
      3. 回退策略：如果 vLLM 不在线，则从环境变量 VLLM_GPU_ID 读取预期值

    Returns:
        (gpu_index, gpu_name) — 如 (1, "NVIDIA A100-PCIE-40GB")
    """
    try:
        # 查找占用 vLLM 端口的进程
        port_pid_output = subprocess.check_output(
            ["ss", "-tlnp"],
            text=True, timeout=5,
        )
        vllm_pid = None
        for line in port_pid_output.split("\n"):
            # VLLM_BASE_URL 格式: "http://localhost:8001" → 提取端口 "8001"
            vllm_port = VLLM_BASE_URL.split(":")[-1]
            if f":{vllm_port}" in line and "pid=" in line:
                match = re.search(r'pid=(\d+)', line)
                if match:
                    vllm_pid = match.group(1)
                    break

        # 读取进程环境变量中的 CUDA_VISIBLE_DEVICES
        cuda_devices = None
        if vllm_pid:
            try:
                with open(f"/proc/{vllm_pid}/environ", "rb") as f:
                    environ_raw = f.read().decode("utf-8", errors="replace")
                    for item in environ_raw.split("\x00"):
                        if item.startswith("CUDA_VISIBLE_DEVICES="):
                            cuda_devices = item.split("=", 1)[1]
                            break
            except (PermissionError, FileNotFoundError):
                pass  # 权限不足或进程已退出

        # 回退：尝试环境变量
        if cuda_devices is None:
            cuda_devices = os.environ.get("VLLM_GPU_ID") or os.environ.get("CUDA_VISIBLE_DEVICES")

        if cuda_devices is None:
            return None, None

        # 解析 GPU 索引（支持 "0", "0,1", "GPU-xxx" 等格式）
        gpu_idx_str = cuda_devices.split(",")[0].strip()
        try:
            gpu_idx = int(gpu_idx_str)
        except ValueError:
            return None, None

        # 查询该 GPU 的名称
        gpu_name = _get_gpu_name_by_index(gpu_idx)
        return gpu_idx, gpu_name

    except Exception:
        return None, None


def _get_gpu_name_by_index(index: int) -> Optional[str]:
    """通过 nvidia-smi 查询指定索引 GPU 的名称。"""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
        for line in output.split("\n"):
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) >= 2 and parts[0] == str(index):
                return parts[1]
    except Exception:
        pass
    return None


# ============================================================
# 检查 FastAPI 后端
# ============================================================

def check_fastapi() -> Dict[str, Any]:
    """
    检查 比邻星 (ProximaRAG) FastAPI 后端 (http://localhost:8000)。

    Returns:
        {
            "online": bool,
            "url": str,
            "ready": bool,
            "document_count": int,
            "error": str | None,
            "latency_ms": float,
        }
    """
    result = {
        "online": False,
        "url": FASTAPI_BASE_URL,
        "ready": False,
        "document_count": 0,
        "error": None,
        "latency_ms": 0,
    }

    t0 = time.monotonic()
    try:
        resp = get_client().get(f"{FASTAPI_BASE_URL}/api/status")
        result["latency_ms"] = round((time.monotonic() - t0) * 1000)

        if resp.status_code == 200:
            data = resp.json()
            result["online"] = True
            result["ready"] = data.get("ready", False)
            result["document_count"] = data.get("document_count", 0)
        else:
            result["error"] = f"HTTP {resp.status_code}"
    except httpx.ConnectError:
        result["error"] = "连接被拒绝（服务未启动或端口错误）"
    except httpx.TimeoutException:
        result["error"] = f"连接超时（>{TIMEOUT_CONNECT}s）"
    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================
# v4 双索引检查
# ============================================================

def check_v4_collections() -> Dict[str, Any]:
    """
    检查 v4 Parent-Child 双索引 ChromaDB Collections。

    Returns:
        {
            "v4_available": bool,
            "parent_count": int,
            "child_count": int,
            "legacy_count": int,
            "error": str | None,
        }
    """
    result = {
        "v4_available": False,
        "parent_count": 0,
        "child_count": 0,
        "legacy_count": 0,
        "error": None,
    }
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path="vector_db",
            settings=Settings(anonymized_telemetry=False),
        )
        for name in ["rag_v4_parent", "rag_v4_child", "rag_documents"]:
            try:
                coll = client.get_collection(name)
                count = coll.count()
                if name == "rag_v4_parent":
                    result["parent_count"] = count
                elif name == "rag_v4_child":
                    result["child_count"] = count
                else:
                    result["legacy_count"] = count
            except Exception:
                pass
        result["v4_available"] = result["parent_count"] > 0 and result["child_count"] > 0
    except Exception as e:
        result["error"] = str(e)[:80]
    return result


# ============================================================
# CUDA GPU 显存查询
# ============================================================

def parse_memory(mib_str: str) -> float:
    """将 '12345 MiB' 转换为 GB 浮点数"""
    try:
        return float(mib_str.replace("MiB", "").strip()) / 1024
    except (ValueError, AttributeError):
        return 0.0

def check_gpu_memory() -> List[Dict[str, Any]]:
    """
    通过 nvidia-smi 查询 GPU 显存占用。

    Returns:
        [
            {
                "index": int,
                "name": str,
                "memory_used_gb": float,
                "memory_total_gb": float,
                "memory_free_gb": float,
                "memory_pct": float,
                "temperature": str,
                "power": str,
                "utilization_gpu": str,
            },
            ...
        ]
    """
    try:
        # nvidia-smi 查询：索引、名称、已用显存、总显存、空闲显存、温度、功率、GPU 利用率
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free,temperature.gpu,power.draw,utilization.gpu",
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

            index      = int(parts[0])
            name       = parts[1]
            mem_used   = float(parts[2]) / 1024   # MiB → GB
            mem_total  = float(parts[3]) / 1024
            mem_free   = float(parts[4]) / 1024
            temp       = parts[5] if parts[5] else "N/A"
            power      = parts[6] if parts[6] else "N/A"
            util       = parts[7] if len(parts) > 7 and parts[7] else "N/A"

            mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0

            gpus.append({
                "index": index,
                "name": name,
                "memory_used_gb": round(mem_used, 2),
                "memory_total_gb": round(mem_total, 2),
                "memory_free_gb": round(mem_free, 2),
                "memory_pct": round(mem_pct, 1),
                "temperature": f"{temp}°C" if temp != "N/A" else "N/A",
                "power": f"{power}W" if power != "N/A" else "N/A",
                "utilization_gpu": f"{util}%" if util != "N/A" else "N/A",
            })

        return gpus

    except FileNotFoundError:
        return []
    except Exception as e:
        return [{"index": -1, "name": f"nvidia-smi 查询失败: {e}"}]


# ============================================================
# 格式化输出
# ============================================================

def print_header():
    """打印报告头部"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print(bold(blue("╔══════════════════════════════════════════════════════════╗")))
    print(bold(blue("║")) + bold("      比邻星 (ProximaRAG) 系统健康检查报告               ") + bold(blue("║")))
    print(bold(blue("╠══════════════════════════════════════════════════════════╣")))
    print(bold(blue("║")) + dim(f"  检查时间: {now}") + " " * (26 - len(now)) + bold(blue("║")))
    print(bold(blue("╚══════════════════════════════════════════════════════════╝")))
    print()

def print_gpu_section(gpus: List[Dict[str, Any]], vllm_gpu_id: Optional[int] = None):
    """打印 GPU 显存占用表格，标记 vLLM 部署的 GPU"""
    print(bold("┌─ GPU 资源状态 " + "─" * 43 + "┐"))

    if not gpus:
        print(f"│  {yellow('⚠️  nvidia-smi 不可用，无法查询 GPU 状态')}" + " " * 18 + "│")
        print("└" + "─" * 59 + "┘")
        return

    # 表头
    print(f"│ {'GPU':<5} {'名称':<22} {'显存占用':<16} {'温度':<7} {'功率':<8} │")
    print(f"│ {'':<5} {'':<22} {'Used / Total (GB)':<16} {'':<7} {'':<8} │")
    print("│" + "─" * 59 + "│")

    for gpu in gpus:
        idx   = gpu["index"]
        name  = gpu.get("name", "?")[:22]
        mem_bar = _gpu_bar(gpu["memory_pct"])
        mem_str = f"{gpu['memory_used_gb']:.1f} / {gpu['memory_total_gb']:.1f} GB {mem_bar}"
        temp  = gpu.get("temperature", "N/A")
        power = gpu.get("power", "N/A")

        # 标记 vLLM 部署的 GPU
        tag = ""
        if vllm_gpu_id is not None and idx == vllm_gpu_id:
            tag = f" {green('◀ vLLM')}"

        print(f"│ [{idx}]  {name:<20} {mem_str:<16} {temp:<7} {power:<8}{tag} │")

    # 图例
    if vllm_gpu_id is not None:
        print(f"│ {dim('  ◀ vLLM  = 本地推理服务部署在此 GPU 上'):<58} │")

    print("└" + "─" * 59 + "┘")
    print()

def _gpu_bar(pct: float, width: int = 8) -> str:
    """绘制简易显存占用进度条"""
    filled = int(pct / 100 * width)
    empty  = width - filled

    if pct > 90:
        bar = red("█" * filled + "░" * empty)
    elif pct > 70:
        bar = yellow("█" * filled + "░" * empty)
    else:
        bar = green("█" * filled + "░" * empty)

    return f"[{bar}] {pct:.0f}%"

def print_service_section(vllm: Dict, fastapi: Dict, v4: Dict = None):
    """打印服务状态表格（v4 含双索引信息）"""
    print(bold("┌─ 核心服务状态 " + "─" * 42 + "┐"))

    # vLLM
    vllm_status = STATUS_OK if vllm["online"] else STATUS_DOWN
    vllm_model  = vllm.get("model") or "-"
    vllm_lat    = f"{vllm.get('latency_ms', 0)}ms"
    vllm_err    = vllm.get("error", "")
    vllm_gpu_id = vllm.get("gpu_id")
    vllm_gpu_name = vllm.get("gpu_name")

    print(f"│ {'vLLM 推理服务':<16} {vllm_status:<18} {dim(f'({vllm_lat})'):<10} │")
    print(f"│   └ 端口          {cyan(f'{VLLM_BASE_URL}/v1'):<30} │")
    if vllm["online"] and vllm_model:
        print(f"│   └ 已加载模型    {green(vllm_model):<30} │")
    if vllm_gpu_id is not None:
        gpu_label = f"GPU {vllm_gpu_id}" + (f" ({vllm_gpu_name})" if vllm_gpu_name else "")
        print(f"│   └ 部署 GPU      {green(gpu_label):<30} │")
    elif not vllm["online"]:
        expected_gpu = os.environ.get("VLLM_GPU_ID", "自动检测")
        print(f"│   └ 目标 GPU      {dim(f'[{expected_gpu}] — vLLM 离线'):<30} │")
    if not vllm["online"] and vllm_err:
        print(f"│   └ 错误          {red(vllm_err[:48]):<48} │")

    print("│" + " " * 59 + "│")

    # FastAPI
    api_status  = STATUS_OK if fastapi["online"] else STATUS_DOWN
    api_ready   = green("✅ 向量库就绪") if fastapi["ready"] else yellow("⚠️  向量库未初始化")
    api_docs    = f"{fastapi['document_count']} 个文档片段"
    api_lat     = f"{fastapi.get('latency_ms', 0)}ms"
    api_err     = fastapi.get("error", "")

    print(f"│ {'比邻星 后端':<16} {api_status:<18} {dim(f'({api_lat})'):<10} │")
    print(f"│   └ 端口          {cyan(f'{FASTAPI_BASE_URL}'):<30} │")
    if fastapi["online"]:
        print(f"│   └ 向量库        {api_ready:<24} {api_docs:<15} │")
    if not fastapi["online"] and api_err:
        print(f"│   └ 错误          {red(api_err[:48]):<48} │")

    # v4 双索引
    if v4:
        print("│" + " " * 59 + "│")
        v4_status = green("✅ v4 双索引就绪") if v4.get("v4_available") else (yellow("⚠️  v4 未构建") if v4.get("error") else dim("  未检测"))
        print(f"│ {'v4 双索引 (ADR-15)':<16} {v4_status:<36} │")
        if v4.get("parent_count") or v4.get("child_count"):
            print(f"│   └ Parent       {green(str(v4['parent_count'])):<6} chunks (H2 章节级粗召回)        │")
            print(f"│   └ Child        {green(str(v4['child_count'])):<6} chunks (H3/H4 函数级精匹配)        │")
        if v4.get("legacy_count", 0) > 0:
            print(f"│   └ Legacy (v3)  {dim(str(v4['legacy_count'])):<6} chunks (旧索引，兼容保留)         │")

    print("└" + "─" * 59 + "┘")
    print()

def print_summary(vllm: Dict, fastapi: Dict, gpus: List[Dict]):
    """打印综合评估"""
    all_ok = vllm["online"] and fastapi["online"] and fastapi["ready"]
    degraded = (fastapi["online"] and not vllm["online"])  # Layer 2/3 模式下可用

    print(bold("┌─ 综合评估 " + "─" * 46 + "┐"))

    if all_ok:
        vllm_gpu = vllm.get("gpu_id", "?")
        print(f"│  {green('✅ 系统完全健康 — vLLM 就绪 (GPU ' + str(vllm_gpu) + ')，向量库已加载，所有通道正常')} │")
    elif degraded:
        print(f"│  {yellow('⚠️  降级运行中')}" + " " * 45 + "│")
        print(f"│  本地 vLLM 不可用，对话将自动降级至云端 API 或纯检索直出模式   │")
    elif fastapi["online"] and not fastapi["ready"]:
        print(f"│  {yellow('⚠️  后端在线但向量库为空 — 请上传 PDF 文档')}" + " " * 17 + "│")
    else:
        print(f"│  {red('❌ 系统不可用 — 比邻星 后端离线')}" + " " * 36 + "│")

    # GPU 建议（所有 GPU 均不足时发出警告）
    if gpus:
        low_gpus = [g for g in gpus if g.get("memory_free_gb", 0) < 5]
        if low_gpus and not vllm["online"]:
            gpu_list = ", ".join(str(g["index"]) for g in low_gpus)
            print(f"│  {yellow('⚠️  GPU ' + gpu_list + ' 空闲显存不足 5GB — 不建议此时启动本地 vLLM')} │")

    print("└" + "─" * 59 + "┘")
    print()

def print_layer_status(vllm_online: bool, fastapi_online: bool):
    """打印当前有效的容灾层级"""
    print(bold("┌─ 四层容灾可用性 " + "─" * 40 + "┐"))

    layers = [
        ("Layer 1", "本地 vLLM 推理",       vllm_online,               "零成本、低延迟"),
        ("Layer 2", "智谱 GLM-4.7-Flash",   fastapi_online,            "云端备份、自动切换"),
        ("Layer 3", "纯检索智能直出",        fastapi_online,            "CPU-only、零API费用"),
        ("Layer 4", "优雅错误提示",           True,                      "最终防线（始终可用）"),
    ]

    for name, desc, available, note in layers:
        if available:
            marker = green("✓")
        else:
            marker = red("✗")
        print(f"│  {marker}  {name:<10} {desc:<24} {dim(note):<22} │")

    print("└" + "─" * 59 + "┘")
    print()


# ============================================================
# 主流程
# ============================================================

def run_check():
    """执行一次完整健康检查并打印报告"""
    # ---- 收集数据 ----
    vllm_result    = check_vllm()
    fastapi_result = check_fastapi()
    v4_result      = check_v4_collections()
    gpus           = check_gpu_memory()

    # ---- 打印报告 ----
    print_header()
    print_service_section(vllm_result, fastapi_result, v4_result)
    print_gpu_section(gpus, vllm_gpu_id=vllm_result.get("gpu_id"))
    print_layer_status(vllm_result["online"], fastapi_result["online"])
    print_summary(vllm_result, fastapi_result, gpus)


def main():
    parser = argparse.ArgumentParser(
        description="比邻星 (ProximaRAG) 统一服务健康状态检查脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python check_status.py              # 一次性检查
  python check_status.py --watch 10   # 每 10 秒自动刷新
  python check_status.py -w 5         # 每 5 秒刷新（简写）
        """,
    )
    parser.add_argument(
        "-w", "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="持续监控模式，每隔 N 秒刷新一次（Ctrl+C 退出）",
    )
    args = parser.parse_args()

    if args.watch > 0:
        interval = args.watch
        print(f"{cyan('🔄 持续监控模式')} — 每 {interval} 秒刷新一次（按 Ctrl+C 退出）")
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                run_check()
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n{green('✅ 监控已停止')}")
    else:
        run_check()


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# =============================================================================
# 比邻星 (ProximaRAG) 自动化启动脚本 (8000 端口 + 显存自适应恢复版)
# =============================================================================

set -euo pipefail

# ============================================================
# 1. 终端颜色定义
# ============================================================
GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
CYAN='\033[96m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ============================================================
# 2. 全局配置
# ============================================================
CONDA_ENV="rag_agent"
VLLM_PORT=8001
FASTAPI_PORT=8000
VLLM_GPU_ID="${VLLM_GPU_ID:-}"  # 空 = 自动检测，非空 = 手动覆盖

# ── 动态模型选择候选池 ──
MODEL_CANDIDATES=(
    "/home/kasm-user/LLM/mo/models/Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/master|0.25|8192|--quantization awq"
    "/home/kasm-user/LLM/mo/models/Qwen--Qwen2.5-3B-Instruct|0.20|8192"
    "/home/kasm-user/LLM/mo/models/Qwen--Qwen2.5-1.5B-Instruct|0.20|4096"
)
MIN_FREE_MEMORY_MIB=5120   # 5 GB 最低门槛

VLLM_READY_TIMEOUT=180
VLLM_READY_INTERVAL=3
VLLM_PID=""
FASTAPI_PID=""

# ============================================================
# 辅助函数
# ============================================================

log_info()    { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "${CYAN}[STEP]${NC}  ${BOLD}$1${NC}"; }
log_detail()  { echo -e "        ${DIM}$1${NC}"; }

banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}${BOLD}     比邻星 (ProximaRAG) 服务启动脚本              ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}${DIM}     湖南比邻星科技 — 文档智能问答系统               ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

check_port() {
    local port=$1
    if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
        return 1
    else
        return 0
    fi
}

detect_best_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "-1"
        return 2
    fi

    local gpu_data
    gpu_data=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)

    if [ -z "$gpu_data" ]; then
        echo "-1"
        return 1
    fi

    local best_idx=-1
    local best_free=0
    local all_info=""

    while IFS=',' read -r idx free_mib; do
        idx=$(echo "$idx" | xargs)
        free_mib=$(echo "$free_mib" | xargs)

        if [ -z "$idx" ] || [ -z "$free_mib" ]; then
            continue
        fi

        local free_gb
        free_gb=$(awk "BEGIN {printf \"%.1f\", $free_mib / 1024}")

        if [ "$free_mib" -ge "$MIN_FREE_MEMORY_MIB" ]; then
            all_info="${all_info}  ${GREEN}✓${NC} GPU ${idx}: ${free_gb} GB 空闲 ${GREEN}(可用)${NC}\n"
            if [ "$free_mib" -gt "$best_free" ]; then
                best_free=$free_mib
                best_idx=$idx
            fi
        else
            all_info="${all_info}  ${RED}✗${NC} GPU ${idx}: ${free_gb} GB 空闲 ${DIM}(不足 ${MIN_FREE_MEMORY_MIB} MiB)${NC}\n"
        fi
    done <<< "$gpu_data"

    log_detail "GPU 空闲显存扫描结果:" >&2
    echo -e "$all_info" | while IFS= read -r line; do
        [ -n "$line" ] && echo -e "        $line" >&2
    done

    if [ "$best_idx" -ge 0 ]; then
        local best_gb
        best_gb=$(awk "BEGIN {printf \"%.1f\", $best_free / 1024}")
        log_info "自动选择 GPU: ${BOLD}${best_idx}${NC}（空闲 ${GREEN}${best_gb} GB${NC}，所有候选 GPU 中最大）" >&2
        echo "$best_idx"
        return 0
    fi

    log_error "所有 GPU 空闲显存均不足 ${MIN_FREE_MEMORY_MIB} MiB，无法部署 vLLM" >&2
    echo "-1"
    return 1
}

export_gpu_env() {
    local gpu_id=$1
    export VLLM_GPU_ID="$gpu_id"
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    log_detail "已设置: CUDA_VISIBLE_DEVICES=${gpu_id}, VLLM_GPU_ID=${gpu_id}" >&2
}

wait_for_vllm() {
    log_info "等待 vLLM 推理服务就绪（最长 ${VLLM_READY_TIMEOUT}s）..."

    local elapsed=0
    local url="http://localhost:${VLLM_PORT}/v1/models"

    while [ $elapsed -lt $VLLM_READY_TIMEOUT ]; do
        if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log_error "vLLM 进程已意外退出（PID: $VLLM_PID）"
            return 1
        fi

        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$url" 2>/dev/null || echo "000")

        if [ "$http_code" = "200" ]; then
            local model_name
            model_name=$(curl -s "$url" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "?")
            log_info "vLLM 就绪！已加载模型: ${GREEN}${model_name}${NC}（耗时 ${elapsed}s）"
            return 0
        fi

        if [ $((elapsed % 15)) -eq 0 ] && [ $elapsed -gt 0 ]; then
            log_detail "仍在等待... (${elapsed}s/${VLLM_READY_TIMEOUT}s)"
        fi

        sleep $VLLM_READY_INTERVAL
        elapsed=$((elapsed + VLLM_READY_INTERVAL))
    done

    log_error "vLLM 启动超时（${VLLM_READY_TIMEOUT}s），请检查日志"
    return 1
}

cleanup() {
    echo ""
    log_warn "正在关闭服务..."
    [ -n "$FASTAPI_PID" ] && kill "$FASTAPI_PID" 2>/dev/null || true
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
    log_info "服务已清理"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================================
# 启动逻辑
# ============================================================

start_vllm() {
    log_step "第 1 步：启动本地 vLLM 推理服务"

    if ! check_port $VLLM_PORT; then
        log_info "vLLM 服务已在 8001 端口运行，跳过启动"
        return 0
    fi

    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"

    local selected_gpu
    if [ -n "$VLLM_GPU_ID" ]; then
        selected_gpu="$VLLM_GPU_ID"
        log_info "使用手动指定的 GPU: ${BOLD}${selected_gpu}${NC}"
    else
        log_step "智能 GPU 检测：扫描所有 GPU 空闲显存..."
        selected_gpu=$(detect_best_gpu)
        local detect_rc=$?
        if [ "$detect_rc" -ne 0 ] || [ "$selected_gpu" -lt 0 ]; then
            log_error "未找到可用的 GPU，无法启动 vLLM"
            return 1
        fi
    fi
    export_gpu_env "$selected_gpu"

    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_HUB_OFFLINE=1
    export PYTHONUNBUFFERED=1

    # 动态模型选择：根据 GPU 空闲显存匹配模型
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | grep "^${selected_gpu}," | cut -d',' -f2 | xargs)
    local free_gb
    free_gb=$(awk "BEGIN {printf \"%.1f\", ${free_mib:-0} / 1024}")

    local selected_model="" selected_gpu_mem="" selected_max_len="" selected_extra_flags=""
    for candidate in "${MODEL_CANDIDATES[@]}"; do
        IFS='|' read -r model_name gpu_mem max_len extra_flags <<< "$candidate"
        local min_free_mib=0
        case "$model_name" in
            *7B*)  min_free_mib=14336 ;;   # > 14 GB
            *3B*)   min_free_mib=8192  ;;   # > 8 GB
            *1.5B*) min_free_mib=5120 ;;   # > 5 GB
        esac
        if [ "${free_mib:-0}" -ge "$min_free_mib" ]; then
            selected_model="$model_name"
            selected_gpu_mem="$gpu_mem"
            selected_max_len="$max_len"
            selected_extra_flags="$extra_flags"
            log_info "🎯 动态模型选择: ${BOLD}${selected_model}${NC} (空闲: ${free_gb} GB ≥ ${min_free_mib} MiB)"
            break
        fi
    done

    if [ -z "$selected_model" ]; then
        log_error "GPU ${selected_gpu} 空闲显存不足 (${free_gb} GB < 5 GB)，无法部署任何模型"
        return 1
    fi

    log_info "启动 vLLM 推理服务..."
    log_detail "GPU:        CUDA_VISIBLE_DEVICES=${selected_gpu}"
    log_detail "空闲显存:   ${free_gb} GB"
    log_detail "模型:       ${selected_model}"
    log_detail "端口:       ${VLLM_PORT}"
    log_detail "显存限制:   ${selected_gpu_mem} (gpu-memory-utilization)"
    log_detail "上下文长度: ${selected_max_len} (max-model-len)"

    local vllm_log="/home/kasm-user/LLM/logs/vllm_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p /home/kasm-user/LLM/logs

    CUDA_VISIBLE_DEVICES=${selected_gpu} python -m vllm.entrypoints.openai.api_server \
        --model "${selected_model}" \
        --served-model-name "${selected_model}" \
        --max-model-len "${selected_max_len}" \
        --port "${VLLM_PORT}" \
        --gpu-memory-utilization "${selected_gpu_mem}" \
        --trust-remote-code \
        ${selected_extra_flags} \
        > "$vllm_log" 2>&1 &
    VLLM_PID=$!

    log_info "vLLM 进程已启动 (PID: ${VLLM_PID})"
    log_detail "查看日志: tail -f ${vllm_log}"

    if ! wait_for_vllm; then
        log_error "vLLM 启动失败"
        return 1
    fi

    echo ""
}

start_fastapi() {
    log_step "第 2 步：启动 比邻星 FastAPI 主服务 (端口 ${FASTAPI_PORT})"
    if ! check_port $FASTAPI_PORT; then
        log_warn "端口 ${FASTAPI_PORT} 被占用，强制释放..."
        fuser -k ${FASTAPI_PORT}/tcp 2>/dev/null || true
        sleep 2
    fi

    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"

    log_info "启动服务中..."
    log_info "🎉 服务就绪！"
    log_info "👉 对话主页: ${CYAN}http://localhost:8000${NC}"
    log_info "👉 Debug 调试文档: ${CYAN}http://localhost:8000/docs${NC}"

    python app.py
}

main() {
    banner
    start_vllm
    start_fastapi
}

main "$@"
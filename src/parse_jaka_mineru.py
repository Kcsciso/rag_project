
# --- 终极防御：拦截高版本 transformers 的毒药参数 ---
try:
    from unimernet.models.mbart.modeling_mbart import UnimerMBartForCausalLM
    if not hasattr(UnimerMBartForCausalLM, "_is_patched"):
        _old_forward = UnimerMBartForCausalLM.forward
        def _new_forward(self, *args, **kwargs):
            kwargs.pop("cache_position", None)
            kwargs.pop("num_logits_to_keep", None) # 预防其他新参数
            return _old_forward(self, *args, **kwargs)
        UnimerMBartForCausalLM.forward = _new_forward
        UnimerMBartForCausalLM._is_patched = True
except ImportError:
    pass
# --------------------------------------------------

import os
import json
import subprocess
from pathlib import Path

# 注：magic-pdf 以 subprocess 方式调用（独立 Python 进程），
# 父进程内的 monkeypatch 无法生效，故不在本脚本中打补丁。

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_FILE_PATH = PROJECT_ROOT / "data" / "JAKA_Manual.pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "jaka_markdown"
MODEL_DOWNLOAD_DIR = "/home/kasm-user/LLM/MinerU_Models"
CONFIG_FILE_PATH = os.path.expanduser("~/magic-pdf.json")

def setup_mineru_models_and_config():
    """自动化步骤一：下载视觉大模型权重并写入系统配置"""
    print("🚀 [Step 1] 正在检查/下载 MinerU 视觉大模型权重 (基于 ModelScope)...")
    try:
        from modelscope import snapshot_download
        # 自动跳过已下载的内容，无脑拉取最新权重
        model_dir = snapshot_download('OpenDataLab/PDF-Extract-Kit-1.0', local_dir=MODEL_DOWNLOAD_DIR)
        print(f"✅ 权重就绪，路径: {model_dir}")
    except ImportError:
        raise RuntimeError("❌ 缺少 modelscope 库，请确保你的环境中已正确安装。")
    except Exception as e:
        raise RuntimeError(f"❌ 下载权重失败: {str(e)}")

    print("⚙️ [Step 2] 正在自动生成 magic-pdf.json 配置文件...")
    # MinerU 实际需要的模型路径通常是在下载目录的 models 子目录下
    target_models_dir = os.path.join(model_dir, "models")
    
    config_data = {
        "models-dir": target_models_dir,
        "device-mode": "cuda",  # 强行启用 GPU 满血解析
        # 🔴 关键修复: 显式指定 doclayout_yolo。
        # magic-pdf 1.3.12 在缺省 layout-config 时回退到 layoutlmv3，
        # 而 LayoutLMv3 依赖 detectron2/fvcore（与 torch 2.4 不兼容），
        # 会在 fvcore load_yaml_with_base 中抛
        # "TypeError: argument of type 'NoneType' is not iterable"。
        "layout-config": {
            "model": "doclayout_yolo"
        },
        "table-config": {
            # 合法值: tablemaster / rapid_table / struct_eqtable
            # （带连字符的 "table-master" 是非法名称，会触发 exit(1)；当前采用 rapid_table）
            "model": "rapid_table",
            "enable": True,
            "max_time": 400
        }
    }
    
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        print(f"✅ 配置文件已写入: {CONFIG_FILE_PATH}")
    except Exception as e:
        raise RuntimeError(f"❌ 配置文件写入失败: {str(e)}")


def _pick_free_gpu():
    """选择空闲显存最大的 GPU（与项目 start_services.sh 同一自适应策略）。

    仅通过子进程环境变量生效，不污染全局环境。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        free = [int(x) for x in out.stdout.splitlines() if x.strip().isdigit()]
        return free.index(max(free)) if free else 0
    except Exception:
        return 0  # 探测失败则不设限制，走 torch 默认设备


def run_mineru_parsing():
    """自动化步骤二：执行 PDF 智能解析"""
    print("\n📄 [Step 3] 准备解析 JAKA 复杂手册...")
    
    if not PDF_FILE_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到源文件: {PDF_FILE_PATH}\n💡 请先将你要测试的 JAKA 手册重命名为 JAKA_Manual.pdf 并放入 data/ 目录下！")
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ 输出目录就绪: {OUTPUT_DIR}")
    
    # 封装命令调用 magic-pdf 核心引擎
    # -p: pdf 路径 | -o: 输出路径 | -m auto: 自动识别版面结构
    cmd = [
        "magic-pdf", 
        "-p", str(PDF_FILE_PATH), 
        "-o", str(OUTPUT_DIR), 
        "-m", "auto"
    ]
    
    print(f"⏳ 正在启动 GPU 视觉解析，该过程可能需要几分钟，请耐心等待...")
    # vLLM 常驻 GPU 时自动让位：只对子进程生效
    gpu_id = _pick_free_gpu()
    print(f"🖥️  自动选择空闲显存最大的 GPU: {gpu_id}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    try:
        # 使用 subprocess 运行，将输出实时打在控制台
        subprocess.run(cmd, check=True, env=env)
        print(f"\n🎉 解析完美完成！")
        print(f"👉 请前往 {OUTPUT_DIR} 目录查看生成的 Markdown 文件和提取的表格/图片。")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 解析过程中断，请检查上方报错日志。退出码: {e.returncode}")
    except FileNotFoundError:
        print("\n❌ 找不到 magic-pdf 命令。请确认已成功执行 `pip install \"magic-pdf[full]\"`")

if __name__ == "__main__":
    print("=== ProximaRAG MinerU 离线处理管线启动 ===")
    setup_mineru_models_and_config()
    run_mineru_parsing()
"""
=============================================================================
ngrok 内网穿透启动脚本
=============================================================================

功能：将本地 FastAPI 服务（http://localhost:8000）通过 ngrok 隧道暴露到公网。

【前置条件】
  1. 安装 pyngrok: pip install pyngrok （已预装）
  2. （可选）注册 ngrok 免费账号：https://dashboard.ngrok.com/signup
     获取 authtoken 以获得更长的隧道存活时间
  3. 本地服务已启动：python app.py 或 uvicorn app:app

【使用方式】
  # 方式一：无需认证（匿名隧道，有速率限制）
  python tunnel.py

  # 方式二：使用 authtoken（推荐，更稳定）
  python tunnel.py --token YOUR_NGROK_AUTHTOKEN

  # 方式三：通过环境变量
  export NGROK_AUTHTOKEN="YOUR_NGROK_AUTHTOKEN"
  python tunnel.py

【注意事项】
  - 免费版 ngrok 隧道有时间限制和速率限制
  - 隧道 URL 每次重启会变化（免费版），PRO 版支持固定域名
  - 生产环境建议使用 frp / Cloudflare Tunnel 等更稳定的方案

=============================================================================
"""

import argparse
import os
import sys
import time


def start_tunnel(port: int = 8000, auth_token: str = None):
    """
    启动 ngrok 隧道，将本地端口暴露到公网。

    Args:
        port: 本地服务端口
        auth_token: ngrok authtoken（可选）
    """
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("❌ pyngrok 未安装。请运行: pip install pyngrok")
        sys.exit(1)

    # ---- 认证 ----
    token = auth_token or os.environ.get("NGROK_AUTHTOKEN")
    if token:
        conf.get_default().auth_token = token
        print("🔑 已配置 ngrok authtoken")
    else:
        print("⚠️  未提供 authtoken，使用匿名模式（有速率限制）")
        print("   获取免费 token: https://dashboard.ngrok.com/signup")

    # ---- 创建 HTTP 隧道 ----
    print(f"🌐 正在创建 ngrok 隧道 → http://localhost:{port} ...")

    try:
        # 建立 HTTP 隧道
        public_url = ngrok.connect(port, "http")
        print()
        print("=" * 60)
        print("  ✅ 隧道已建立！外部网络可通过以下地址访问：")
        print(f"  🔗 {public_url}")
        print("=" * 60)
        print()
        print("按 Ctrl+C 停止隧道...")

        # 保持隧道活跃
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 正在关闭隧道...")
    except Exception as e:
        print(f"\n❌ 隧道启动失败: {e}")
        print("\n常见原因：")
        print("  1. 本地服务未启动 — 请先运行 python app.py")
        print("  2. 网络防火墙限制")
        print("  3. ngrok 账户未激活（检查邮箱验证）")
        sys.exit(1)
    finally:
        ngrok.disconnect(public_url)
        print("隧道已关闭。")


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NewsPage — ngrok 内网穿透工具"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="本地服务端口（默认: 8000）"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="ngrok authtoken（也可通过环境变量 NGROK_AUTHTOKEN 设置）"
    )
    args = parser.parse_args()

    start_tunnel(port=args.port, auth_token=args.token)

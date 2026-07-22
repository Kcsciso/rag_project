"""
=============================================================================
NewsPage Streamlit 前端 — 端口 8501
=============================================================================
"""
import streamlit as st
import requests
import json

API_BASE = "http://localhost:7860"

st.set_page_config(
    page_title="NewsPage — 湖南比邻星科技",
    page_icon="⬡",
    layout="wide",
)

# ── 初始化 session state ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None

# ── 侧边栏 ──
with st.sidebar:
    st.image("https://via.placeholder.com/60x60.png?text=⬡", width=60)
    st.title("NewsPage")
    st.caption("湖南比邻星科技 · 文档智能问答")

    # 产品标签
    try:
        resp = requests.get(f"{API_BASE}/api/products", timeout=3)
        products = resp.json().get("products", [])
    except Exception:
        products = []

    if products:
        st.subheader("🏷️ 设备知识库")
        cols = st.columns(len(products) + 1)
        for i, pid in enumerate(products):
            with cols[i]:
                if st.button(
                    pid,
                    key=f"prod_{pid}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_product == pid else "secondary",
                ):
                    st.session_state.selected_product = (
                        None if st.session_state.selected_product == pid else pid
                    )
                    st.rerun()
        with cols[-1]:
            if st.button("✕ 全部", key="prod_clear", use_container_width=True):
                st.session_state.selected_product = None
                st.rerun()

    # 状态
    try:
        resp = requests.get(f"{API_BASE}/api/status", timeout=3)
        status = resp.json()
        st.metric("已索引片段", status.get("document_count", 0))
    except Exception:
        st.warning("后端未连接")

    st.divider()
    st.caption("支持 OpenR6 / OpenC3 SDK 文档查询")

# ── 主区域：对话历史 ──
st.title("⬡ NewsPage 文档智能问答")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 输入框 ──
query = st.chat_input("输入你的问题，基于已上传文档获取准确回答…")

if query:
    # 用户消息
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 调用 API
    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                fd = {"query": query, "stream": "false"}
                if st.session_state.selected_product:
                    fd["product_id"] = st.session_state.selected_product

                resp = requests.post(
                    f"{API_BASE}/api/chat",
                    data=fd,
                    timeout=90,
                )
                data = resp.json()
                answer = data.get("answer", "服务异常，请稍后重试")
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                if data.get("sources"):
                    st.caption(f"📎 来源: {', '.join(data['sources'])}")
                if data.get("needs_clarification"):
                    st.info("💡 请在上方选择对应的设备型号以获得更精准的回答")

            except Exception as e:
                st.error(f"请求失败: {e}")

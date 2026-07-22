/**
 * ============================================================================
 * NewsPage — 湖南比邻星科技文档智能问答系统
 * 前端交互逻辑：SSE 流式对话 · Markdown 渲染 · 代码高亮 · PDF 上传
 * ============================================================================
 */

// ============================================================
// marked.js 配置：集成 highlight.js 代码高亮
// ============================================================
if (typeof marked !== "undefined") {
    marked.setOptions({
        highlight: function (code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (_) { /* fall through */ }
            }
            try {
                return hljs.highlightAuto(code).value;
            } catch (_) {
                return code;
            }
        },
        breaks: true,        // 单个换行转为 <br>
        gfm: true,           // GitHub Flavored Markdown（表格、任务列表等）
    });
}

// ============================================================
// DOM 元素引用
// ============================================================
const chatMessages   = document.getElementById("chatMessages");
const chatForm       = document.getElementById("chatForm");
const queryInput     = document.getElementById("queryInput");
const sendButton     = document.getElementById("sendButton");
const fileInput      = document.getElementById("fileInput");
const uploadButton   = document.getElementById("uploadButton");
const uploadArea     = document.getElementById("uploadArea");
const uploadProgress = document.getElementById("uploadProgress");
const progressFill   = document.getElementById("progressFill");
const progressText   = document.getElementById("progressText");
const uploadResult   = document.getElementById("uploadResult");
const statusDot      = document.getElementById("statusDot");
const statusText     = document.getElementById("statusText");
const kbReady        = document.getElementById("kbReady");
const kbCount        = document.getElementById("kbCount");
const refreshStatusBtn = document.getElementById("refreshStatus");

// 产品标签栏（嵌入右侧边栏"知识库状态"卡片内）
const productTagsSection   = document.getElementById("productTagsSection");
const productTagsContainer = document.getElementById("productTagsContainer");

// ============================================================
// 状态
// ============================================================
let chatHistory        = [];
let isWaiting          = false;
let selectedProductId  = null;  // 🏷️ 当前选中的产品 ID（null=未筛选）

// ============================================================
// 初始化 — 自执行（script 在 </body> 前，DOM 已完全解析）
// ============================================================
// 不使用 DOMContentLoaded 事件监听，因为 script 位于 <body> 末尾时
// DOM 已解析完毕，DOMContentLoaded 可能已错过或存在竞态。
// 直接自执行初始化，100% 可靠。
(function initApp() {
    checkStatus();
    loadProducts();
})();

// ============================================================
// 知识库状态
// ============================================================
async function checkStatus() {
    try {
        const resp = await fetch("/api/status");
        const data = await resp.json();
        if (data.ready) {
            statusDot.classList.add("ready");
            statusText.textContent = `就绪 · ${data.document_count} 条索引`;
        } else {
            statusDot.classList.remove("ready");
            statusText.textContent = "未初始化";
        }
        kbReady.textContent = data.ready ? "✅ 就绪" : "⚠️ 未就绪";
        kbCount.textContent = data.document_count;
    } catch (_) {
        statusText.textContent = "状态查询失败";
        kbReady.textContent   = "❌ 连接失败";
    }
}

// ============================================================
// 产品知识库标签
// ============================================================
async function loadProducts() {
    // 🔴 防御：DOM 元素可能因浏览器缓存旧版 HTML 而不存在
    if (!productTagsSection || !productTagsContainer) {
        console.warn("产品标签栏 DOM 元素未就绪，跳过加载");
        return;
    }
    try {
        const resp = await fetch("/api/products");
        const data = await resp.json();

        // 🔴 前后端双重去重：Set 去重 + 过滤空值，防止重复标签
        const uniqueProducts = [...new Set((data.products || []).filter(p => p && p !== "unknown"))];

        if (uniqueProducts.length === 0) {
            productTagsSection.style.display = "none";
            return;
        }

        productTagsSection.style.display = "block";
        // 🔴 渲染前强制清空容器，防止多次调用时累积重复
        productTagsContainer.innerHTML = "";

        uniqueProducts.forEach((pid) => {
            const tag = document.createElement("span");
            tag.className = "product-tag";
            tag.dataset.productId = pid;
            tag.innerHTML = `<span class="tag-dot"></span>${escapeHtml(pid)}`;
            tag.title = `点击仅检索 ${pid} 的知识库`;
            tag.addEventListener("click", () => toggleProductTag(pid, tag));
            productTagsContainer.appendChild(tag);
        });

        // 如果之前已选中产品但该产品已不在列表中，清除选中
        if (selectedProductId && !uniqueProducts.includes(selectedProductId)) {
            selectedProductId = null;
        }

        // 恢复选中状态
        if (selectedProductId) {
            const activeTag = productTagsContainer.querySelector(`[data-product-id="${selectedProductId}"]`);
            if (activeTag) activeTag.classList.add("active");
        }
    } catch (e) {
        console.error("加载产品列表失败:", e);
        if (productTagsSection) productTagsSection.style.display = "none";
    }
}

function toggleProductTag(productId, tagEl) {
    if (selectedProductId === productId) {
        // 取消选中 → 恢复全设备检索
        selectedProductId = null;
        if (productTagsContainer) {
            productTagsContainer.querySelectorAll(".product-tag").forEach(t => t.classList.remove("active"));
        }
    } else {
        // 选中新产品
        selectedProductId = productId;
        if (productTagsContainer) {
            productTagsContainer.querySelectorAll(".product-tag").forEach(t => t.classList.remove("active"));
        }
        tagEl.classList.add("active");
    }
}

// ============================================================
// 聊天
// ============================================================
chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    await sendMessage();
});

queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event("submit"));
    }
});

queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
});

async function sendMessage() {
    const query = queryInput.value.trim();
    if (!query || isWaiting) return;

    isWaiting = true;
    sendButton.disabled = true;
    queryInput.value = "";
    queryInput.style.height = "auto";

    // 清除欢迎消息
    const wm = chatMessages.querySelector(".welcome-message");
    if (wm) wm.remove();

    // 用户消息
    addMessage("user", query);
    chatHistory.push({ role: "user", content: query });

    // AI 占位
    const aiEl = addMessage("assistant", "", true);
    const contentEl = aiEl.querySelector(".message-content");

    try {
        const fd = new FormData();
        fd.append("query", query);
        fd.append("history", JSON.stringify(chatHistory.slice(0, -1)));
        fd.append("stream", "true");
        if (selectedProductId) {
            fd.append("product_id", selectedProductId);
        }

        const resp = await fetch("/api/chat", { method: "POST", body: fd });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "请求失败");
        }

        const reader  = resp.body.getReader();
        const decoder = new TextDecoder();
        let fullText  = "";
        let buffer    = "";
        let renderPending = false;
        const RENDER_THROTTLE_MS = 50;  // 最多每 50ms 刷新一次渲染（20fps）

        removeTypingIndicator(contentEl);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split("\n\n");
            buffer = events.pop();  // 保留不完整片段

            for (const ev of events) {
                if (!ev.startsWith("data: ")) continue;
                try {
                    const data = JSON.parse(ev.slice(6));
                    if (data.done) break;
                    if (data.error) {
                        contentEl.innerHTML = `<span style="color:var(--color-error)">错误: ${escapeHtml(data.error)}</span>`;
                        return;
                    }
                    if (data.delta) {
                        fullText += data.delta;
                        // 节流渲染：避免每个 token 都触发完整 Markdown 重渲染
                        if (!renderPending) {
                            renderPending = true;
                            setTimeout(() => {
                                contentEl.innerHTML = renderMarkdown(fullText);
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                                renderPending = false;
                            }, RENDER_THROTTLE_MS);
                        }
                    }
                } catch (_) { /* 忽略不完整 JSON */ }
            }
        }

        // 确保最后一次渲染
        contentEl.innerHTML = renderMarkdown(fullText);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        chatHistory.push({ role: "assistant", content: fullText });

    } catch (err) {
        contentEl.innerHTML = `<span style="color:var(--color-error)">错误: ${escapeHtml(err.message)}</span>`;
        console.error("Chat error:", err);
    } finally {
        isWaiting = false;
        sendButton.disabled = false;
        queryInput.focus();
    }
}

// ============================================================
// 消息渲染
// ============================================================

/** 将文本渲染为安全的 HTML。优先使用 marked.js，回退时用纯文本转义 */
function renderMarkdown(text) {
    if (!text) return "";
    if (typeof marked !== "undefined") {
        try {
            return marked.parse(text);
        } catch (_) { /* 回退 */ }
    }
    // 纯文本回退
    return escapeHtml(text).replace(/\n/g, "<br>");
}

/** HTML 实体转义 */
function escapeHtml(str) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return str.replace(/[&<>"']/g, (c) => map[c]);
}

function addMessage(role, content, streaming) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    const av = document.createElement("div");
    av.className = "message-avatar";
    // 用户用文字标识，AI 用六边形图标
    av.textContent = role === "user" ? "👤" : "⬡";

    const body = document.createElement("div");
    body.className = "message-content";

    if (streaming) {
        body.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    } else {
        body.innerHTML = renderMarkdown(content);
    }

    div.appendChild(av);
    div.appendChild(body);
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function removeTypingIndicator(el) {
    const ti = el.querySelector(".typing-indicator");
    if (ti) ti.remove();
}

// ============================================================
// PDF 上传
// ============================================================
uploadButton.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
    const f = fileInput.files[0];
    if (f) uploadFile(f);
});

uploadArea.addEventListener("dragover", (e) => { e.preventDefault(); uploadArea.classList.add("drag-over"); });
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f && f.name.toLowerCase().endsWith(".pdf")) {
        uploadFile(f);
    } else {
        showUploadResult("仅支持 PDF 文件格式", "error");
    }
});

async function uploadFile(file) {
    uploadArea.style.display = "none";
    uploadProgress.style.display = "block";
    uploadResult.textContent = "";
    progressFill.style.width = "30%";
    progressText.textContent = "正在上传…";

    const fd = new FormData();
    fd.append("file", file);

    try {
        progressFill.style.width = "60%";
        progressText.textContent = "解析 PDF 并重建向量库…";

        const resp = await fetch("/api/upload", { method: "POST", body: fd });
        const data = await resp.json();

        if (resp.ok && data.success) {
            progressFill.style.width = "100%";
            progressText.textContent = "完成！";
            showUploadResult(`✅ ${data.message}（${data.document_count} 个片段）`, "success");
            await checkStatus();
            await loadProducts();  // 刷新产品标签
        } else {
            throw new Error(data.detail || data.message || "上传失败");
        }
    } catch (err) {
        progressFill.style.width = "100%";
        showUploadResult(`❌ ${err.message}`, "error");
    } finally {
        setTimeout(() => {
            uploadProgress.style.display = "none";
            uploadArea.style.display = "";
            progressFill.style.width = "0%";
        }, 2500);
    }
}

function showUploadResult(msg, type) {
    uploadResult.textContent = msg;
    uploadResult.className = `upload-result ${type}`;
}

// ============================================================
// 刷新按钮
// ============================================================
refreshStatusBtn.addEventListener("click", checkStatus);

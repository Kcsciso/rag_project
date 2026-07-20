/**
 * ============================================================================
 * NewsPage — 前端交互逻辑
 * ============================================================================
 *
 * 功能概览：
 *   1. 聊天消息的发送与流式接收（SSE）
 *   2. PDF 文件的拖拽上传与点击上传
 *   3. 知识库状态的实时查询与展示
 *   4. 多轮对话历史的维护
 */

// ============================================================
// DOM 元素引用
// ============================================================
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const sendButton = document.getElementById("sendButton");
const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("uploadButton");
const uploadArea = document.getElementById("uploadArea");
const uploadProgress = document.getElementById("uploadProgress");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const uploadResult = document.getElementById("uploadResult");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const kbReady = document.getElementById("kbReady");
const kbCount = document.getElementById("kbCount");
const refreshStatusBtn = document.getElementById("refreshStatus");

// ============================================================
// 状态管理
// ============================================================

// 对话历史：[{role: "user"|"assistant", content: "..."}, ...]
let chatHistory = [];

// 当前是否正在等待 AI 回复
let isWaiting = false;

// ============================================================
// 初始化
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    checkStatus();
});

// ============================================================
// 知识库状态查询
// ============================================================

async function checkStatus() {
    try {
        const response = await fetch("/api/status");
        const data = await response.json();

        // 更新顶部状态指示器
        if (data.ready) {
            statusDot.classList.add("ready");
            statusText.textContent = `知识库就绪（${data.document_count} 个片段）`;
        } else {
            statusDot.classList.remove("ready");
            statusText.textContent = "知识库未初始化";
        }

        // 更新侧边栏状态
        kbReady.textContent = data.ready ? "✅ 就绪" : "⚠️ 未就绪";
        kbCount.textContent = data.document_count;
    } catch (error) {
        statusText.textContent = "状态查询失败";
        kbReady.textContent = "❌ 连接失败";
    }
}

// ============================================================
// 聊天功能
// ============================================================

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendMessage();
});

// Enter 发送，Shift+Enter 换行
queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.dispatchEvent(new Event("submit"));
    }
});

// 自动调整输入框高度
queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
});

async function sendMessage() {
    const query = queryInput.value.trim();
    if (!query || isWaiting) return;

    // ---- UI 更新 ----
    isWaiting = true;
    sendButton.disabled = true;
    queryInput.value = "";
    queryInput.style.height = "auto";

    // 移除欢迎消息
    const welcomeMsg = chatMessages.querySelector(".welcome-message");
    if (welcomeMsg) welcomeMsg.remove();

    // 添加用户消息到界面
    addMessage("user", query);

    // 更新对话历史
    chatHistory.push({ role: "user", content: query });

    // 创建 AI 消息占位符
    const assistantMsgEl = addMessage("assistant", "", true);
    const contentEl = assistantMsgEl.querySelector(".message-content");

    try {
        // ---- 发送请求（流式 SSE） ----
        const formData = new FormData();
        formData.append("query", query);
        formData.append("history", JSON.stringify(chatHistory.slice(0, -1))); // 不含当前消息
        formData.append("stream", "true");

        const response = await fetch("/api/chat", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "请求失败");
        }

        // ---- 读取 SSE 流 ----
        // Server-Sent Events 格式: "data: <JSON>\n\n"
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let aiResponse = "";
        let buffer = "";

        // 移除打字动画（流式内容开始到达）
        removeTypingIndicator(contentEl);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // SSE 事件以 "\n\n" 分隔
            const events = buffer.split("\n\n");
            buffer = events.pop(); // 最后一个可能不完整

            for (const event of events) {
                if (!event.startsWith("data: ")) continue;
                const jsonStr = event.slice(6); // 去掉 "data: " 前缀
                try {
                    const data = JSON.parse(jsonStr);
                    if (data.done) {
                        // 流式输出完成
                        break;
                    }
                    if (data.delta) {
                        aiResponse += data.delta;
                        // 将 markdown 中的换行转为 <br>（简单处理）
                        contentEl.innerHTML = formatMessage(aiResponse);
                        // 滚动到底部
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                    if (data.error) {
                        contentEl.innerHTML = `<span style="color:#dc2626;">错误: ${data.error}</span>`;
                    }
                } catch (e) {
                    // 忽略解析错误（可能是不完整的 JSON）
                }
            }
        }

        // 更新对话历史
        chatHistory.push({ role: "assistant", content: aiResponse });

    } catch (error) {
        contentEl.innerHTML = `<span style="color:#dc2626;">错误: ${error.message}</span>`;
        console.error("Chat error:", error);
    } finally {
        isWaiting = false;
        sendButton.disabled = false;
        queryInput.focus();
    }
}

/**
 * 添加消息气泡到聊天区域
 * @param {"user"|"assistant"} role - 消息角色
 * @param {string} content - 消息内容
 * @param {boolean} isStreaming - 是否正在流式接收（显示打字动画）
 * @returns {HTMLElement} 消息元素
 */
function addMessage(role, content, isStreaming = false) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "👤" : "🤖";

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    if (isStreaming) {
        // 显示打字动画，表示 AI 正在思考
        contentDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    } else {
        contentDiv.innerHTML = formatMessage(content);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // 滚动到底部
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageDiv;
}

/**
 * 移除打字动画指示器
 */
function removeTypingIndicator(contentEl) {
    const indicator = contentEl.querySelector(".typing-indicator");
    if (indicator) indicator.remove();
}

/**
 * 简单的消息格式化：转义 HTML + 换行转 <br>
 */
function formatMessage(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\n/g, "<br>");
}

// ============================================================
// PDF 上传功能
// ============================================================

uploadButton.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) uploadFile(file);
});

// ---- 拖拽上传 ----
uploadArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadArea.classList.remove("drag-over");
    const file = event.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith(".pdf")) {
        uploadFile(file);
    } else {
        showUploadResult("仅支持 PDF 文件格式", "error");
    }
});

async function uploadFile(file) {
    // ---- UI 更新：显示进度条 ----
    uploadArea.style.display = "none";
    uploadProgress.style.display = "block";
    uploadResult.textContent = "";
    progressFill.style.width = "30%";
    progressText.textContent = "正在上传...";

    const formData = new FormData();
    formData.append("file", file);

    try {
        progressFill.style.width = "60%";
        progressText.textContent = "正在解析 PDF 并重建向量库...";

        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (response.ok && data.success) {
            progressFill.style.width = "100%";
            progressText.textContent = "完成！";
            showUploadResult(
                `✅ ${data.message}（共 ${data.document_count} 个片段）`,
                "success"
            );
            // 刷新状态
            await checkStatus();
        } else {
            throw new Error(data.detail || data.message || "上传失败");
        }
    } catch (error) {
        progressFill.style.width = "100%";
        showUploadResult(`❌ ${error.message}`, "error");
    } finally {
        // 延迟后恢复到上传区域
        setTimeout(() => {
            uploadProgress.style.display = "none";
            uploadArea.style.display = "";
            progressFill.style.width = "0%";
        }, 2000);
    }
}

function showUploadResult(message, type) {
    uploadResult.textContent = message;
    uploadResult.className = `upload-result ${type}`;
}

// ============================================================
// 刷新按钮
// ============================================================
refreshStatusBtn.addEventListener("click", checkStatus);

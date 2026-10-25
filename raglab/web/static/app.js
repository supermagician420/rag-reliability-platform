"use strict";

const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const resultPanel = document.getElementById("result-panel");
const statusBadge = document.getElementById("status-badge");
const answerText = document.getElementById("answer");
const metaText = document.getElementById("meta");
const sourcesBox = document.getElementById("sources");
const healthText = document.getElementById("health-text");
const healthDot = document.querySelector("#health .dot");
const docList = document.getElementById("documents");
const docCount = document.getElementById("doc-count");

async function loadHealth() {
  try {
    const response = await fetch("/api/health", { headers: { "Accept": "application/json" } });
    const data = await response.json();
    if (!response.ok) throw new Error("health failed");
    healthDot.classList.add("ok");
    healthText.textContent = `服务正常 · ${data.documents} 篇文档 · ${data.llm === "mock" ? "Mock 模式" : "LLM 模式"}`;
  } catch (error) {
    healthText.textContent = "服务不可用";
  }
}

async function loadDocuments() {
  try {
    const response = await fetch("/api/documents", { headers: { "Accept": "application/json" } });
    const data = await response.json();
    if (!response.ok) throw new Error("documents failed");
    docCount.textContent = `${data.documents.length} 篇`;
    docList.replaceChildren();
    for (const doc of data.documents) {
      const item = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = doc.title;
      const detail = document.createElement("span");
      detail.textContent = `${doc.chunks} 个分块 · ${new Date(doc.created_at).toLocaleString()}`;
      item.append(title, detail);
      docList.append(item);
    }
  } catch (error) {
    docList.replaceChildren();
    docCount.textContent = "不可用";
  }
}

function addSource(source, index) {
  const box = document.createElement("div");
  box.className = "source";
  const title = document.createElement("strong");
  title.textContent = `[S${index + 1}] ${source.title}`;
  const snippet = document.createElement("p");
  snippet.textContent = source.snippet;
  box.append(title, snippet);
  sourcesBox.append(box);
}

async function ask() {
  const question = questionInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }
  askButton.disabled = true;
  askButton.textContent = "检索中…";
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({ question })
    });
    const data = await response.json();
    resultPanel.hidden = false;
    resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    statusBadge.className = "badge";
    sourcesBox.replaceChildren();

    if (!response.ok && data.status !== "error") {
      statusBadge.textContent = "请求失败";
      statusBadge.classList.add("error");
      answerText.textContent = data.error || "请求失败";
      metaText.textContent = "";
      return;
    }

    statusBadge.textContent = data.status === "answered" ? "已回答" : data.status === "refused" ? "已拒答" : "服务错误";
    if (data.status === "refused") statusBadge.classList.add("refused");
    if (data.status === "error") statusBadge.classList.add("error");
    answerText.textContent = data.answer;
    const confidence = typeof data.confidence === "number" ? `${Math.round(data.confidence * 100)}%` : "—";
    metaText.textContent = `状态：${data.status} · 置信度：${confidence}${data.reason ? " · " + data.reason : ""}`;
    (data.sources || []).forEach(addSource);
  } catch (error) {
    resultPanel.hidden = false;
    statusBadge.className = "badge error";
    statusBadge.textContent = "网络错误";
    answerText.textContent = "请求没有完成，请检查服务是否运行。";
    metaText.textContent = "";
    sourcesBox.replaceChildren();
  } finally {
    askButton.disabled = false;
    askButton.textContent = "提问";
  }
}

askButton.addEventListener("click", ask);
questionInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") ask();
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    questionInput.value = chip.dataset.question || "";
    questionInput.focus();
  });
});

loadHealth();
loadDocuments();

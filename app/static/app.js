/* ============================================================
   D16 前端逻辑
   ============================================================ */
"use strict";

const STORAGE_KEY = "copilot_sessions";
const CURRENT_KEY = "copilot_current";

/* ---------- 长期记忆用的用户 id ----------
 *
 * 【为什么要这个东西】
 *   短期记忆按 thread_id 存（一次会话），长期记忆必须按「人」存。
 *   项目没有登录体系，所以用浏览器本地生成的一个匿名 id 代替。
 *
 * 【为什么存在 localStorage】
 *   关掉浏览器再打开还是同一个人 —— 这正是"长期记忆"要的效果。
 *   如果放内存变量，刷新页面就变成新用户，记忆全丢。
 *
 * 【怎么用】
 *   每个 /api/* 请求带一个 "X-User-Id" 头，后端据此隔离记忆。
 */
const USER_ID_KEY = "copilot_uid";

function getUserId() {
  let uid = localStorage.getItem(USER_ID_KEY);

  // 没有就生成一个。crypto.randomUUID 不是所有环境都有，兜底用时间戳 + 随机数
  if (!uid) {
    if (window.crypto && crypto.randomUUID) {
      uid = crypto.randomUUID();
    } else {
      uid = "u" + Date.now() + "-" + Math.random().toString(16).slice(2);
    }
    localStorage.setItem(USER_ID_KEY, uid);
    console.log("[memory] 生成新的用户 id:", uid);
  }

  return uid;
}

const USER_ID = getUserId();

let currentThreadId = null;
let streaming = false;
let abortCtrl = null;

const elChat = document.getElementById("chat");
const elInput = document.getElementById("input");
const elSend = document.getElementById("send");
const elSessions = document.getElementById("sessions");
const elNew = document.getElementById("new-session");
const elUploadBtn = document.getElementById("upload-btn");
const elFileInput = document.getElementById("file-input");
const elAttachBar = document.getElementById("attach-bar");
const elAttachName = document.getElementById("attach-name");
const elAttachCheck = document.getElementById("attach-check");
const elAttachRemove = document.getElementById("attach-remove");
const elExamples = document.getElementById("examples");

/* ============================================================
   一、SSE 解析
   ============================================================ */

function createSSEParser(onEvent) {
  let buffer = "";

  return function feed(text) {
    buffer += text;
    buffer = buffer.replace(/\r\n/g, "\n");

    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const rawBlock = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      dispatchBlock(rawBlock);
    }
  };

  function dispatchBlock(rawBlock) {
    if (!rawBlock.trim()) return;

    let eventName = "message";
    const dataLines = [];

    const lines = rawBlock.split("\n");
    for (const line of lines) {
      if (line.startsWith(":")) continue;

      const colon = line.indexOf(":");
      let field, value;
      if (colon === -1) {
        field = line;
        value = "";
      } else {
        field = line.slice(0, colon);
        value = line.slice(colon + 1);
        if (value.startsWith(" ")) value = value.slice(1);
      }

      if (field === "event") eventName = value;
      else if (field === "data") dataLines.push(value);
    }

    if (dataLines.length === 0) return;

    let payload;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch (e) {
      console.warn("data 不是合法 JSON，跳过");
      return;
    }

    onEvent(eventName, payload);
  }
}

/* ============================================================
   二、Markdown 渲染
   ============================================================ */

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* 渲染期间共享的「原始编号 -> 显示编号」映射，由 renderMarkdown 设置 */
let activeNumMap = null;

function inline(s) {
  let t = escapeHtml(s);
  t = t.replace(/\x60([^\x60]+)\x60/g, "<code>$1</code>");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // [7] 这类标记：只有真的能对应到一条法条，才做成可点角标
  // 为什么要有这个限制：
  //   1 模型偶尔会写一个没检索到的编号（幻觉），做成角标点了没反应会很迷惑
  //   2 正文里也可能出现 [2024] 这种不是引用的方括号，不该被当成引用
  t = t.replace(/\[(\d+)\]/g, function (whole, n) {
    if (activeNumMap && activeNumMap.has(n)) {
      // data-oidx 存「原始编号」（用来找对应的依据条目）
      // 显示的是「本答案内的序号」，从 1 开始
      return '<sup class="cite" data-oidx="' + n + '">[' + activeNumMap.get(n) + "]</sup>";
    }
    return whole;   // 对不上就原样保留，不做任何装饰
  });

  return t;
}

/**
 * 建立「原始编号 -> 本答案内的显示编号」
 *
 * 为什么要重编号：
 *   RefPool 的编号是跨多次检索连续的，所以答案里可能出现 [7] [1] [4] [3]。
 *   用户看到第一个角标是 [7] 会觉得莫名其妙。
 *   这里按「在本答案里第一次出现的顺序」重新编号成 1、2、3……
 *
 * 为什么只映射 citations 里有的编号：
 *   映射不到的（模型幻觉出的编号、或者 [2024] 这类非引用方括号）保持原样，
 *   不做成角标，避免点了没反应。
 */
function buildCitationMap(text, citations) {
  const known = new Set();
  for (const c of citations || []) {
    known.add(String(c.idx));
  }

  const map = new Map();
  let next = 1;
  const re = /\[(\d+)\]/g;
  let m;
  while ((m = re.exec(String(text || ""))) !== null) {
    const orig = m[1];
    if (!known.has(orig)) continue;
    if (!map.has(orig)) {
      map.set(orig, next);
      next++;
    }
  }
  return map;
}

function splitRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map(function (c) { return c.trim(); });
}

function renderMarkdown(text, numMap) {
  activeNumMap = numMap || null;
  const lines = String(text || "").split("\n");
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      const level = Math.min(h[1].length + 2, 6);
      out.push("<h" + level + ">" + inline(h[2]) + "</h" + level + ">");
      i++;
      continue;
    }

    const isRow = /^\s*\|/.test(line);
    const isSep = i + 1 < lines.length && /^\s*\|[\s:|-]+\|?\s*$/.test(lines[i + 1]);
    if (isRow && isSep) {
      const header = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      let html = "<table><thead><tr>";
      for (const c of header) html += "<th>" + inline(c) + "</th>";
      html += "</tr></thead><tbody>";
      for (const r of rows) {
        html += "<tr>";
        for (const c of r) html += "<td>" + inline(c) + "</td>";
        html += "</tr>";
      }
      html += "</tbody></table>";
      out.push(html);
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      let html = "<ul>";
      for (const it of items) html += "<li>" + inline(it) + "</li>";
      html += "</ul>";
      out.push(html);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i++;
      }
      let html = "<ol>";
      for (const it of items) html += "<li>" + inline(it) + "</li>";
      html += "</ol>";
      out.push(html);
      continue;
    }

    if (line.trim() === "") {
      i++;
      continue;
    }

    const para = [];
    while (i < lines.length) {
      const l = lines[i];
      if (l.trim() === "") break;
      if (/^(#{1,6})\s/.test(l)) break;
      if (/^\s*[-*]\s+/.test(l)) break;
      if (/^\s*\d+\.\s+/.test(l)) break;
      if (/^\s*\|/.test(l)) break;
      para.push(l);
      i++;
    }
    out.push("<p>" + para.map(inline).join("<br>") + "</p>");
  }

  activeNumMap = null;   // 渲染结束就清掉，避免影响下一次
  return out.join("\n");
}

/* ============================================================
   三、气泡
   ============================================================ */

function maybeScroll() {
  const nearBottom =
    elChat.scrollHeight - elChat.scrollTop - elChat.clientHeight < 120;
  if (nearBottom) elChat.scrollTop = elChat.scrollHeight;
}

function forceScroll() {
  elChat.scrollTop = elChat.scrollHeight;
}

function addUserBubble(text) {
  const wrap = document.createElement("div");
  wrap.className = "row user";
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  wrap.appendChild(b);
  elChat.appendChild(wrap);
  forceScroll();
}

function addAiBubble() {
  const wrap = document.createElement("div");
  wrap.className = "row ai";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const content = document.createElement("div");
  content.className = "content";
  bubble.appendChild(content);

  const tools = document.createElement("details");
  tools.className = "tools";
  tools.hidden = true;
  const toolSummary = document.createElement("summary");
  const toolBody = document.createElement("div");
  toolBody.className = "tool-body";
  tools.appendChild(toolSummary);
  tools.appendChild(toolBody);
  bubble.appendChild(tools);

  // 依据区做成 <details>：默认展开（用户刚看完答案通常想看依据），
  // 但可以点标题收起来。点角标时会自动展开。
  const cites = document.createElement("details");
  cites.className = "cites";
  cites.open = true;
  cites.hidden = true;
  const citesSummary = document.createElement("summary");
  citesSummary.className = "cites-title";
  const citesBody = document.createElement("div");
  citesBody.className = "cites-body";
  cites.appendChild(citesSummary);
  cites.appendChild(citesBody);
  bubble.appendChild(cites);

  wrap.appendChild(bubble);
  elChat.appendChild(wrap);
  forceScroll();

  let raw = "";
  let toolCount = 0;
  let progressEl = null;   // 体检进度行，第一次 setProgress 时才创建

  return {
    pendingTitle: "",

    appendToken(text) {
      raw += text;
      content.textContent = raw;
      content.classList.add("streaming");
      maybeScroll();
    },

    addTool(name, args) {
      toolCount++;
      tools.hidden = false;
      toolSummary.textContent = "调用了 " + toolCount + " 次工具";

      const row = document.createElement("div");
      row.className = "tool-row";
      let argText;
      try {
        argText = JSON.stringify(args);
      } catch (e) {
        argText = String(args);
      }
      row.textContent = name + "(" + argText + ")";
      toolBody.appendChild(row);
      maybeScroll();
    },

    finish(answer, citations) {
      content.classList.remove("streaming");

      const text = answer || raw;
      // 按「在本答案里出现的顺序」重新编号成 1..n
      const numMap = buildCitationMap(text, citations);

      content.innerHTML = renderMarkdown(text, numMap);
      renderCitations(cites, citations, numMap);
      maybeScroll();
    },

    /** 显示/更新进度行（合同体检用） */
    setProgress(text) {
      if (!progressEl) {
        progressEl = document.createElement("div");
        progressEl.className = "progress";
        const spin = document.createElement("span");
        spin.className = "spin";
        const label = document.createElement("span");
        label.className = "progress-text";
        progressEl.appendChild(spin);
        progressEl.appendChild(label);
        bubble.insertBefore(progressEl, cites);
      }
      progressEl.querySelector(".progress-text").textContent = text;
      maybeScroll();
    },

    /** 体检结束：渲染报告 markdown（不走 citations，报告里已经带了法条名） */
    finishContract(data) {
      if (progressEl) progressEl.remove();
      progressEl = null;

      let head = "";
      if (data.warning) {
        head += '<div class="warn">' + escapeHtml(data.warning) + "</div>";
      }

      const summary =
        "共 **" + data.total + "** 条条款：高风险 **" + data.high + "** 条，" +
        "中风险 " + data.medium + " 条，未发现问题的 " + data.none + " 条" +
        (data.unknown ? "，无法判断 " + data.unknown + " 条" : "") + "。";

      content.innerHTML =
        head +
        renderMarkdown(summary, null) +
        renderMarkdown(data.report, null);
      maybeScroll();
    },

    fail(message) {
      content.classList.remove("streaming");
      const err = document.createElement("div");
      err.className = "error";
      err.textContent = "出错了：" + message;
      bubble.appendChild(err);
    },
  };
}

function renderCitations(box, citations, numMap) {
  const body = box.querySelector(".cites-body");
  const summary = box.querySelector("summary");
  body.innerHTML = "";

  if (!citations || citations.length === 0) {
    box.hidden = true;
    return;
  }

  box.hidden = false;
  summary.textContent = "依据（" + citations.length + " 条）";

  for (const c of citations) {
    const item = document.createElement("div");
    item.className = "cite-item";
    item.dataset.oidx = String(c.idx);   // 用原始编号关联角标，显示编号可以变

    const head = document.createElement("div");
    head.className = "cite-head";
    // 依据列表也显示重新编过的号，和正文里的角标一致
    const shown = numMap && numMap.has(String(c.idx)) ? numMap.get(String(c.idx)) : c.idx;
    head.textContent = "[" + shown + "] 《" + c.law + "》" + c.no;
    item.appendChild(head);

    const text = document.createElement("div");
    text.className = "cite-text";
    text.textContent = c.text;
    item.appendChild(text);

    body.appendChild(item);
  }
}

elChat.addEventListener("click", function (e) {
  const sup = e.target.closest("sup.cite");
  if (!sup) return;

  const bubble = sup.closest(".bubble");
  if (!bubble) return;

  // 依据区是可折叠的，先展开再滚过去，否则滚到的是收起来的标题
  const box = bubble.querySelector(".cites");
  if (box) box.open = true;

  // 用原始编号去找依据条目（显示编号是重编过的，不能拿来查）
  const item = bubble.querySelector('.cite-item[data-oidx="' + sup.dataset.oidx + '"]');
  if (!item) return;

  item.scrollIntoView({ block: "nearest", behavior: "smooth" });
  item.classList.add("flash");
  setTimeout(function () { item.classList.remove("flash"); }, 1200);
});

/* ============================================================
   四、会话
   ============================================================ */

function loadSessionList() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    return [];
  }
}

function saveSessionList(list) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

function upsertSession(id, title) {
  const list = loadSessionList();
  const found = list.find(function (s) { return s.id === id; });
  if (found) {
    if (title) found.title = title;
  } else {
    list.unshift({ id: id, title: title || "新对话" });
  }
  saveSessionList(list);
  renderSessionList();
}

function removeSession(id) {
  saveSessionList(loadSessionList().filter(function (s) { return s.id !== id; }));
  renderSessionList();
}

function renderSessionList() {
  const list = loadSessionList();
  elSessions.innerHTML = "";
  for (const s of list) {
    const div = document.createElement("div");
    div.className = "session-item" + (s.id === currentThreadId ? " active" : "");
    div.dataset.id = s.id;

    const name = document.createElement("span");
    name.className = "session-name";
    name.textContent = s.title;

    const del = document.createElement("button");
    del.className = "session-del";
    del.textContent = "×";
    del.title = "删除会话";

    div.appendChild(name);
    div.appendChild(del);
    elSessions.appendChild(div);
  }
}

function newSession() {
  if (streaming) return;
  currentThreadId = null;
  localStorage.removeItem(CURRENT_KEY);
  elChat.innerHTML = "";
  renderSessionList();
  elInput.focus();
}

async function openSession(id) {
  if (streaming) return;
  currentThreadId = id;
  localStorage.setItem(CURRENT_KEY, id);
  renderSessionList();

  elChat.innerHTML = "";
  try {
    const resp = await fetch("/api/sessions/" + encodeURIComponent(id) + "/messages");
    const data = await resp.json();
    const msgs = data.messages || [];
    for (const m of msgs) {
      if (m.role === "human") {
        addUserBubble(m.content);
      } else if (m.role === "ai" && m.content) {
        const ui = addAiBubble();
        ui.finish(m.content, m.citations || []);
      }
    }
  } catch (e) {
    console.error("加载历史失败", e);
  }
  forceScroll();
}

/* ============================================================
   五、发送
   ============================================================ */

/**
 * 发一个 POST 请求，把返回的 SSE 流逐条解析出来
 *
 * 为什么抽成一个公共函数：
 *   聊天（/api/ask/stream）和体检（/api/contracts/check/stream）都要
 *   「fetch + getReader + 按 \n\n 切分 + JSON 解析」这一套。
 *   不抽出来就是两份几乎一样的代码，改一处忘另一处。
 *
 * 参数
 *   url      接口地址
 *   body     请求体对象（会转成 JSON）
 *   onEvent  回调 (事件名, 数据对象)
 * 返回
 *   Promise，流结束时 resolve
 */
async function streamSSE(url, body, onEvent) {
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // 长期记忆靠这个头区分「是谁」，后端用它做 Milvus 的 user_id 过滤
      "X-User-Id": USER_ID,
    },
    body: JSON.stringify(body),
    signal: abortCtrl ? abortCtrl.signal : undefined,
  });

  if (!resp.ok) {
    // 后端出错时返回的是普通 JSON（比如 404/400），把它读出来给用户看
    let detail = "";
    try {
      detail = await resp.text();
    } catch (e) {
      detail = "";
    }
    throw new Error("HTTP " + resp.status + (detail ? "：" + detail.slice(0, 120) : ""));
  }

  const reader = resp.body.getReader();
  // 坑：必须复用同一个 decoder 并传 stream:true，
  //     否则一个汉字被切在两个 chunk 里会变乱码
  const decoder = new TextDecoder("utf-8");
  const parser = createSSEParser(onEvent);

  while (true) {
    const r = await reader.read();
    if (r.done) break;
    parser(decoder.decode(r.value, { stream: true }));
  }
  // 把 decoder 里可能残留的字节吐出来
  parser(decoder.decode());
}

/** 风险等级的显示文字 */
function riskLabel(risk) {
  if (risk === "high") return "高风险";
  if (risk === "medium") return "中风险";
  if (risk === "unknown") return "无法判断";
  return "未发现问题";
}

async function send() {
  if (streaming) return;

  let question = elInput.value.trim();

  // 只点了发送、没打字，但附件在 —— 默认问一句最常见的
  if (!question && attachedFile) {
    question = "帮我看看这份合同有没有问题";
  }
  if (!question) return;

  // 【附件怎么带给模型】
  //   Agent 的 check_contract 工具只需要一个【文件名】。
  //   所以把文件名拼进问题文本里，模型看到就知道该传什么。
  //   为什么不改接口加一个 contract 字段：那要同时改 schemas、deps、prompt，
  //   而这里拼一句话就够了，且模型能读懂。
  const usedAttachment = attachedFile;
  let 发给模型的问题 = question;
  if (usedAttachment) {
    发给模型的问题 = question +
      "\n\n（附件：用户上传的合同文件 " + usedAttachment + "，放在 data/contracts 目录下）";
  }

  elInput.value = "";
  elInput.style.height = "auto";
  streaming = true;
  elSend.disabled = true;

  addUserBubble(question + (usedAttachment ? "　📄 " + usedAttachment : ""));
  const ui = addAiBubble();
  ui.pendingTitle = question.slice(0, 20);
  const isFirst = !currentThreadId;

  abortCtrl = new AbortController();

  try {
    let finished = false;

    await streamSSE("/api/ask/stream", { question: 发给模型的问题, thread_id: currentThreadId }, function (name, data) {
      if (name === "tool_call") {
        ui.addTool(data.name, data.args);
      } else if (name === "token") {
        ui.appendToken(data.text);
      } else if (name === "done") {
        finished = true;
        ui.finish(data.answer, data.citations);

        if (isFirst && data.thread_id) {
          currentThreadId = data.thread_id;
          localStorage.setItem(CURRENT_KEY, data.thread_id);
          upsertSession(data.thread_id, ui.pendingTitle);
        }
      } else if (name === "error") {
        finished = true;
        ui.fail(data.message);
      }
    });

    if (!finished) ui.fail("连接断了，没收到 done 事件");
  } catch (e) {
    if (e.name === "AbortError") ui.fail("已取消");
    else ui.fail(String(e));
  } finally {
    streaming = false;
    elSend.disabled = false;
    abortCtrl = null;
    // 附件是一次性的：这一轮用完就清掉，避免下一句话又莫名其妙带一次合同、
    // 让模型再跑一遍 13 秒的体检
    if (usedAttachment) clearAttachment();
    elInput.focus();
  }
}

/* ============================================================
   五之二、合同体检与上传
   ============================================================ */

/** 当前选中的合同文件名（选好后附件条才显示） */
let attachedFile = null;

function showAttachment(filename) {
  attachedFile = filename;
  elAttachName.textContent = filename;
  elAttachBar.hidden = false;
  // 隐藏空状态，否则会一直占着屏幕
  const empty = document.getElementById("empty");
  if (empty) empty.remove();
}

function clearAttachment() {
  attachedFile = null;
  elAttachBar.hidden = true;
  elAttachName.textContent = "";
}

/**
 * 体检一份合同（走 SSE 看进度）
 *
 * 为什么要单独一个函数，而不是复用 send()：
 *   事件类型不一样 —— 聊天是 token/tool_call，体检是 progress；
 *   渲染出来的东西也不一样 —— 聊天是问答气泡，体检是报告。
 */
async function checkContract(filename) {
  if (streaming) return;

  streaming = true;
  elSend.disabled = true;
  clearAttachment();

  addUserBubble("帮我体检一下 " + filename);
  const ui = addAiBubble();
  ui.setProgress("正在读取合同…");

  abortCtrl = new AbortController();

  try {
    let finished = false;

    await streamSSE("/api/contracts/check/stream", { filename: filename }, function (name, data) {
      if (name === "status") {
        ui.setProgress("正在读取合同…");
      } else if (name === "progress") {
        ui.setProgress(
          "正在检查第 " + data.done + "/" + data.total + " 条（" + riskLabel(data.risk) + "）"
        );
      } else if (name === "done") {
        finished = true;
        ui.finishContract(data);
      } else if (name === "error") {
        finished = true;
        ui.fail(data.message);
      }
    });

    if (!finished) ui.fail("连接断了，没收到 done 事件");
  } catch (e) {
    if (e.name === "AbortError") ui.fail("已取消");
    else ui.fail(String(e));
  } finally {
    streaming = false;
    elSend.disabled = false;
    abortCtrl = null;
  }
}

/**
 * 上传文件
 *
 * 【坑】用 FormData 时绝对不要手动设 Content-Type！
 *   浏览器要自己往 Content-Type 里塞 boundary=xxx，
 *   手动设成 "multipart/form-data" 会把 boundary 丢掉，
 *   后端就解析不出文件，报 "Missing boundary" 之类。
 */
async function uploadFile(file) {
  if (!file) return;

  if (!/\.txt$/i.test(file.name)) {
    alert("只支持 .txt 文件");
    return;
  }
  if (file.size > 512 * 1024) {
    alert("文件超过 512 KB，请精简后再上传");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  elUploadBtn.disabled = true;
  elUploadBtn.textContent = "⏳";
  try {
    const resp = await fetch("/api/contracts/upload", {
      method: "POST",
      // 用 FormData 时千万别手动设 Content-Type（会让 boundary 丢掉），
      // 但自定义头是可以加的
      headers: { "X-User-Id": USER_ID },
      body: form,
    });
    if (!resp.ok) {
      let detail = "";
      try {
        detail = await resp.text();
      } catch (e) {
        detail = "";
      }
      throw new Error(detail.slice(0, 160) || ("HTTP " + resp.status));
    }
    const data = await resp.json();
    showAttachment(data.filename);
  } catch (e) {
    alert("上传失败：" + e.message);
  } finally {
    elUploadBtn.disabled = false;
    elUploadBtn.textContent = "📎";
    elFileInput.value = "";     // 清掉，否则选同一个文件不会再触发 change
  }
}

/** 拉取示例合同，渲染成一排按钮 */
async function loadExamples() {
  if (!elExamples) return;
  try {
    const resp = await fetch("/api/contracts");
    if (!resp.ok) return;
    const data = await resp.json();
    const files = data.files || [];
    elExamples.innerHTML = "";
    for (const f of files) {
      const btn = document.createElement("button");
      btn.className = "example-btn";
      btn.textContent = f.name;
      btn.dataset.name = f.name;
      elExamples.appendChild(btn);
    }
    if (files.length === 0) {
      elExamples.innerHTML = '<span class="tip">data/contracts 下还没有合同</span>';
    }
  } catch (e) {
    console.warn("拉示例合同失败", e);
  }
}

/* ============================================================
   六、绑定
   ============================================================ */

elSend.addEventListener("click", send);

elInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});

elInput.addEventListener("input", function () {
  elInput.style.height = "auto";
  elInput.style.height = Math.min(elInput.scrollHeight, 140) + "px";
});

elNew.addEventListener("click", newSession);

/* ---- 上传：点按钮选文件 ---- */
elUploadBtn.addEventListener("click", function () {
  elFileInput.click();
});

elFileInput.addEventListener("change", function () {
  const f = elFileInput.files && elFileInput.files[0];
  if (f) uploadFile(f);
});

/* ---- 附件条：开始体检 / 取消 ---- */
elAttachCheck.addEventListener("click", function () {
  if (!attachedFile) return;
  checkContract(attachedFile);
});

elAttachRemove.addEventListener("click", clearAttachment);

/* ---- 示例合同按钮 ---- */
if (elExamples) {
  elExamples.addEventListener("click", function (e) {
    const btn = e.target.closest(".example-btn");
    if (!btn) return;
    checkContract(btn.dataset.name);
  });
}

/* ---- 拖拽上传 ---- */
// 为什么要 preventDefault：浏览器默认行为是「打开这个文件」，
// 不阻止的话拖进去页面就直接跳走了
document.addEventListener("dragover", function (e) {
  e.preventDefault();
});

document.addEventListener("drop", function (e) {
  e.preventDefault();
  const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) uploadFile(f);
});

elSessions.addEventListener("click", async function (e) {
  const del = e.target.closest(".session-del");
  const item = e.target.closest(".session-item");
  if (!item) return;

  const id = item.dataset.id;

  if (del) {
    e.stopPropagation();
    try {
      await fetch("/api/sessions/" + encodeURIComponent(id), { method: "DELETE" });
    } catch (err) {
      console.warn("删除失败", err);
    }
    removeSession(id);
    if (id === currentThreadId) {
      currentThreadId = null;
      localStorage.removeItem(CURRENT_KEY);
      elChat.innerHTML = "";
      renderSessionList();
    }
    return;
  }

  openSession(id);
});

/* ============================================================
   七、启动
   ============================================================ */

(function init() {
  renderSessionList();
  loadExamples();
  const last = localStorage.getItem(CURRENT_KEY);
  if (last) openSession(last);
  else elInput.focus();
})();

// app/ui/app.js

// --- helper shorthands ---
const $ = (s) => document.querySelector(s);
const api = (p) => `/api${p}`;
const escapeHTML = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const toast = (msg, ok = true) => {
  const el = $("#toast");
  if (!el) return;
  el.textContent = msg;
  el.style.background = ok ? "#0f172a" : "#b91c1c";
  el.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), 2200);
};

const setStatus = (t) => {
  const el = $("#statusChip");
  if (el) el.textContent = t;
};

async function fetchJSON(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let txt;
    try { txt = await r.text(); } catch { txt = `${r.status} ${r.statusText}`; }
    throw new Error(txt || `${r.status} ${r.statusText}`);
  }
  return await r.json();
}

// --- dark mode toggle ---
function wireDarkToggle() {
  const btn = $("#darkToggle");
  if (!btn) return;
  const root = document.documentElement;

  const isDark = () => root.classList.contains("dark");
  const setLabel = () => (btn.textContent = isDark() ? "Light" : "Dark");

  btn.addEventListener("click", () => {
    root.classList.toggle("dark");
    setLabel();
    localStorage.setItem("chromaseek.theme", isDark() ? "dark" : "light");
  });

  // init from storage
  const pref = localStorage.getItem("chromaseek.theme");
  if (pref === "dark") root.classList.add("dark");
  setLabel();
}

// --- stats ---
async function refreshStats() {
  try {
    const s = await fetchJSON(api("/stats"));
    if ($("#vecCount")) $("#vecCount").textContent = s.collection_count ?? 0;

    const ul = $("#pdfList");
    if (ul) {
      ul.innerHTML = "";
      const pdfs = s.pdfs || [];
      if ($("#pdfCount")) $("#pdfCount").textContent = pdfs.length;
      pdfs.forEach((p) => {
        const li = document.createElement("li");
        li.className = "flex items-center justify-between";
        const kb = (p.bytes / 1024).toFixed(0);
        li.innerHTML = `<span class="truncate">${escapeHTML(p.filename)}</span><span class="badge">${kb} KB</span>`;
        ul.appendChild(li);
      });
    }
  } catch (e) {
    console.error(e);
    toast("Failed to load stats", false);
  }
}

// --- upload ---
function wireUpload() {
  const drop = $("#dropZone");
  const input = $("#fileInput");
  const list = $("#uploadList");

  const doUpload = async (files) => {
    if (!files || !files.length) return;
    const fd = new FormData();
    [...files].forEach((f) => fd.append("files", f));

    setStatus("uploading");
    const t0 = performance.now();
    try {
      const res = await fetchJSON(api("/upload"), { method: "POST", body: fd });
      const ms = (performance.now() - t0).toFixed(0);
      if (list) {
        list.innerHTML = (res.saved || [])
          .map(
            (s) =>
              `<div class="text-sm text-slate-700">${escapeHTML(s.filename)} <span class="badge">${(s.bytes / 1024).toFixed(0)} KB</span></div>`
          )
          .join("");
      }
      toast(`Uploaded ${res.total} file(s) in ${ms} ms`);
      await refreshStats();
    } catch (e) {
      console.error(e);
      toast("Upload failed", false);
    } finally {
      setStatus("idle");
      if (input) input.value = "";
    }
  };

  if (drop) {
    drop.addEventListener("dragover", (e) => {
      e.preventDefault();
      drop.classList.add("drag");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("drag");
      if (e.dataTransfer?.files?.length) doUpload(e.dataTransfer.files);
    });
  }

  if (input) input.addEventListener("change", (e) => e.target.files?.length && doUpload(e.target.files));
  const refreshBtn = $("#refreshBtn");
  if (refreshBtn) refreshBtn.addEventListener("click", refreshStats);
}

// --- ingest ---
function wireIngest() {
  const btn = $("#ingestBtn");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    setStatus("indexing…");
    const t0 = performance.now();
    try {
      const res = await fetchJSON(api("/ingest"), { method: "POST" });
      const ms = (performance.now() - t0).toFixed(0);
      if ($("#vecCount")) $("#vecCount").textContent = res.collection_count ?? res.vectors ?? 0;
      toast(`Indexed ${res.vectors} chunks in ${ms} ms`);
      await refreshStats();
    } catch (e) {
      console.error(e);
      toast("Ingest failed", false);
    } finally {
      setStatus("idle");
    }
  });
}

// --- markdown rendering (answer) ---
function renderMarkdown(md) {
  // configure marked + highlight
  marked.setOptions({
    highlight: function (code, lang) {
      try { return hljs.highlight(code, { language: lang || "plaintext" }).value; }
      catch { return hljs.highlightAuto(code).value; }
    },
    breaks: true,
    gfm: true,
  });
  return marked.parse(md || "");
}

// --- ask ---
function wireAsk() {
  const topK = $("#topK");
  const topKVal = $("#topKVal");
  if (topK && topKVal) topK.addEventListener("input", () => (topKVal.textContent = topK.value));

  const askBtn = $("#askBtn");
  if (!askBtn) return;

  askBtn.addEventListener("click", async () => {
    const qEl = $("#q");
    const ans = $("#answer");
    const sources = $("#sources");
    const passages = $("#passages");
    const retrievedCount = $("#retrievedCount");
    const latency = $("#latency");

    const q = qEl?.value?.trim();
    if (!q) {
      toast("Type a question");
      qEl?.focus();
      return;
    }

    const body = { question: q, top_k: Number(topK?.value || 6) };

    if (ans) ans.innerHTML = `<div class="text-slate-500">Thinking…</div>`;
    if (sources) sources.innerHTML = "";
    if (passages) passages.innerHTML = "";
    if (retrievedCount) retrievedCount.textContent = "";
    if (latency) latency.textContent = "";

    setStatus("answering…");
    const t0 = performance.now();

    try {
      const res = await fetchJSON(api("/ask"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const ms = (performance.now() - t0).toFixed(0);
      if (latency) latency.textContent = `Latency: ${ms} ms`;

      // render markdown answer
      if (ans) ans.innerHTML = renderMarkdown(res.answer || "");

      // sources
      if (sources) {
        sources.innerHTML = (res.sources || [])
          .map((s) => `<span class="badge">[${escapeHTML(s)}]</span>`)
          .join(" ");
      }

      // passages
      const items = res.passages || [];
      if (retrievedCount) retrievedCount.textContent = `${res.retrieved ?? items.length} passages`;
      if (passages) {
        passages.innerHTML = items.map((p) => renderPassage(p)).join("");
      }

      toast("Answer ready");
    } catch (e) {
      console.error(e);
      if (ans) ans.innerHTML = "";
      toast("Ask failed", false);
    } finally {
      setStatus("idle");
    }
  });

  const copyAns = $("#copyAns");
  if (copyAns) {
    copyAns.addEventListener("click", async () => {
      const txt = $("#answer")?.innerText || "";
      if (!txt) return;
      try {
        await navigator.clipboard.writeText(txt);
        toast("Copied");
      } catch {
        toast("Copy failed", false);
      }
    });
  }

  const clearBtn = $("#clearBtn");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      $("#q").value = "";
      $("#answer").innerHTML = "";
      $("#sources").innerHTML = "";
      $("#passages").innerHTML = "";
      $("#retrievedCount").textContent = "";
      $("#latency").textContent = "";
    });
  }
}

// --- passage render ---
function renderPassage(p) {
  const source = escapeHTML(p.source ?? "unknown");
  const page = p.page !== null && p.page !== undefined ? ` p.${p.page}` : "";
  const chunk = escapeHTML(p.chunk_id ?? "");
  const score = (p.score ?? 0).toFixed(3);
  const text = escapeHTML(p.text ?? "");
  return `
    <details class="rounded-lg border border-slate-200 overflow-hidden">
      <summary class="px-3 py-2 bg-slate-50 cursor-pointer text-sm flex items-center justify-between">
        <span class="truncate"><b>${source}</b>${page} — <span class="text-slate-500">${chunk}</span></span>
        <span class="badge">score ${score}</span>
      </summary>
      <div class="p-3 text-sm text-slate-800">${text}</div>
    </details>
  `;
}

// --- init ---
(async function init() {
  try {
    wireDarkToggle();
    await refreshStats();
    wireUpload();
    wireIngest();
    wireAsk();
  } catch (e) {
    console.error(e);
    toast("UI init failed", false);
  }
})();

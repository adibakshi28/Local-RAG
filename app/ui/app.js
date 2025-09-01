const $ = (s) => document.querySelector(s);
const api = (p) => `/api${p}`;

const toast = (msg, ok=true) => {
  const el = $("#toast");
  el.textContent = msg;
  el.style.background = ok ? "#0f172a" : "#b91c1c";
  el.classList.remove("hidden");
  setTimeout(()=> el.classList.add("hidden"), 2000);
};

const setStatus = (t) => { $("#statusChip").textContent = t; };

async function fetchJSON(path, opts={}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

// ----- PDFs & stats -----
async function refreshStats() {
  try {
    const s = await fetchJSON(api("/stats"));
    $("#vecCount").textContent = s.collection_count ?? 0;

    const ul = $("#pdfList");
    ul.innerHTML = "";
    (s.pdfs || []).forEach(p => {
      const li = document.createElement("li");
      li.className = "flex items-center justify-between";
      li.innerHTML = `<span class="truncate">${p.filename}</span><span class="badge">${(p.bytes/1024).toFixed(0)} KB</span>`;
      ul.appendChild(li);
    });
  } catch (e) {
    console.error(e);
  }
}

// ----- Upload -----
function wireUpload() {
  const drop = $("#dropZone");
  const input = $("#fileInput");

  const doUpload = async (files) => {
    const fd = new FormData();
    [...files].forEach(f => fd.append("files", f));
    setStatus("uploading");
    const t0 = performance.now();
    try {
      const res = await fetchJSON(api("/upload"), { method:"POST", body: fd });
      const ms = (performance.now()-t0).toFixed(0);
      $("#uploadList").innerHTML =
        (res.saved || []).map(s => `<div class="text-sm text-slate-700">${s.filename} <span class="badge">${(s.bytes/1024).toFixed(0)} KB</span></div>`).join("");
      toast(`Uploaded ${res.total} file(s) in ${ms} ms`);
      await refreshStats();
    } catch (e) {
      toast("Upload failed", false);
      console.error(e);
    } finally {
      setStatus("idle");
    }
  };

  drop.addEventListener("dragover", (e)=>{ e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", ()=> drop.classList.remove("drag"));
  drop.addEventListener("drop", (e)=>{
    e.preventDefault(); drop.classList.remove("drag");
    if (e.dataTransfer.files?.length) doUpload(e.dataTransfer.files);
  });
  input.addEventListener("change", (e)=>{
    if (e.target.files?.length) doUpload(e.target.files);
  });
  $("#refreshBtn").addEventListener("click", refreshStats);
}

// ----- Ingest -----
function wireIngest() {
  $("#ingestBtn").addEventListener("click", async ()=>{
    setStatus("indexing…");
    const t0 = performance.now();
    try {
      const res = await fetchJSON(api("/ingest"), { method:"POST" });
      const ms = (performance.now()-t0).toFixed(0);
      $("#vecCount").textContent = res.collection_count ?? res.vectors ?? 0;
      toast(`Indexed ${res.vectors} chunks in ${ms} ms`);
      await refreshStats();
    } catch(e) {
      toast("Ingest failed", false);
      console.error(e);
    } finally {
      setStatus("idle");
    }
  });
}

// ----- Ask -----
function wireAsk() {
  const topK = $("#topK");
  const topKVal = $("#topKVal");
  topK.addEventListener("input", ()=> topKVal.textContent = topK.value);

  $("#askBtn").addEventListener("click", async ()=>{
    const q = $("#q").value.trim();
    if (!q) { toast("Type a question"); return; }

    const body = { question: q, top_k: Number(topK.value) };
    $("#answer").textContent = "Thinking…";
    $("#sources").innerHTML = "";
    $("#passages").innerHTML = "";
    $("#retrievedCount").textContent = "";
    setStatus("answering…");
    const t0 = performance.now();

    try {
      const res = await fetchJSON(api("/ask"), {
        method:"POST", headers: { "Content-Type":"application/json" },
        body: JSON.stringify(body)
      });
      const ms = (performance.now()-t0).toFixed(0);
      $("#latency").textContent = `Latency: ${ms} ms`;
      $("#answer").textContent = res.answer || "";

      // sources
      if (res.sources?.length) {
        $("#sources").innerHTML = res.sources.map(s => `<span class="badge">[${s}]</span>`).join(" ");
      }

      // passages
      $("#retrievedCount").textContent = `${res.retrieved ?? (res.passages?.length||0)} passages`;
      if (res.passages?.length) {
        $("#passages").innerHTML = res.passages.map(p => `
          <details class="rounded-lg border border-slate-200 overflow-hidden">
            <summary class="px-3 py-2 bg-slate-50 cursor-pointer text-sm flex items-center justify-between">
              <span class="truncate"><b>${p.source}</b> — <span class="text-slate-500">${p.chunk_id}</span></span>
              <span class="badge">score ${(p.score??0).toFixed(3)}</span>
            </summary>
            <div class="p-3 text-sm text-slate-800">${p.text.replace(/</g,"&lt;")}</div>
          </details>
        `).join("");
      }

      toast("Answer ready");
    } catch(e) {
      $("#answer").textContent = "";
      $("#latency").textContent = "";
      toast("Ask failed", false);
      console.error(e);
    } finally {
      setStatus("idle");
    }
  });

  $("#copyAns").addEventListener("click", async ()=>{
    const txt = $("#answer").textContent || "";
    if (!txt) return;
    await navigator.clipboard.writeText(txt);
    toast("Copied");
  });
}

// ----- init -----
(async function init(){
  try {
    await refreshStats();
    wireUpload();
    wireIngest();
    wireAsk();
  } catch(e) {
    console.error(e);
  }
})();

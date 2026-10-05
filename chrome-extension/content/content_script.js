// Renders normalization results near the user's selection when invoked from the context menu.
function removePopover() {
  document.getElementById("tecm-popover")?.remove();
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function showPopover(text) {
  removePopover();
  const sel = window.getSelection();
  const rect = sel && sel.rangeCount ? sel.getRangeAt(0).getBoundingClientRect() : { left: 40, bottom: 40 };
  const box = document.createElement("div");
  box.id = "tecm-popover";
  box.style.left = `${Math.max(8, rect.left)}px`;
  box.style.top = `${rect.bottom + 8}px`;
  box.innerHTML = `<span class="tecm-close">&times;</span><div class="tecm-status">Auto-detecting & Normalizing…</div>`;
  document.body.appendChild(box);
  box.querySelector(".tecm-close").onclick = removePopover;

  const { apiBase } = await chrome.storage.sync.get("apiBase");
  const base = (apiBase || "http://127.0.0.1:8000").replace(/\/$/, "");
  try {
    const res = await fetch(`${base}/omni/process`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, engine: "auto", n_best: 1, beam_size: 4 }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const norm = data.options.normalized_code_mixed || data.recommended.text;
    box.innerHTML = `
      <span class="tecm-close">&times;</span>
      <div style="font-size:10px; color:#38bdf8; font-weight:700; margin-bottom:4px;">✨ ${escapeHtml(data.detected.label)}</div>
      <div class="tecm-hyp" id="tecmResultText">${escapeHtml(norm)}</div>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
        <span class="tecm-status">${data.latency_ms}ms · ${data.recommended.engine}</span>
        <button id="tecmCopyBtn" style="background:#263352; border:1px solid #38bdf8; color:#fff; border-radius:4px; padding:2px 8px; font-size:11px; cursor:pointer;">Copy</button>
      </div>
    `;
    box.querySelector(".tecm-close").onclick = removePopover;
    box.querySelector("#tecmCopyBtn").onclick = () => {
      navigator.clipboard.writeText(norm).then(() => {
        box.querySelector("#tecmCopyBtn").textContent = "Copied!";
        setTimeout(removePopover, 1200);
      });
    };
  } catch (e) {
    box.innerHTML = `<span class="tecm-close">&times;</span><div class="tecm-status">API unreachable: ${escapeHtml(e.message)}. Ensure server is running.</div>`;
    box.querySelector(".tecm-close").onclick = removePopover;
  }
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === "TECM_NORMALIZE_SELECTION" && msg.text) {
    showPopover(msg.text);
  }
});

document.addEventListener("click", (e) => {
  if (e.target.closest && !e.target.closest("#tecm-popover")) removePopover();
});

/* ══════════════════════════════════════════════════════════════════════════
   SIH 26147 — RF Analysis Assistant · Frontend
   Benchmark-Advisory HITL (§9.4) — Pure fetch + DOM, no build step.
   ══════════════════════════════════════════════════════════════════════════ */
"use strict";

// ── Tiny helpers ─────────────────────────────────────────────────────────────
const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
const esc = v =>
  String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

// ── App state ─────────────────────────────────────────────────────────────────
const state = {
  runId: makeRunId(),
  file: null,
  dataType: "",
  sampleRate: "",
  upload: null,
  analyze: null,
  hypothesize: null,
  waterfall: null,
  consult: null,
  corpus: null,
  gates: { G1: [], G2: [], G3: [] },
};

function makeRunId() {
  if (crypto && crypto.randomUUID) return "run-" + crypto.randomUUID().slice(0, 8);
  return "run-" + Math.random().toString(16).slice(2, 10);
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, type = "") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = type;
  t.hidden = false;
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.hidden = true), 3000);
}

// ── Status dot ────────────────────────────────────────────────────────────────
function setStatus(s) {
  const d = $("#statusDot");
  d.className = "status-dot " + s;
  d.title = { ok: "API connected", busy: "Running…", error: "API error", "": "Idle" }[s] || s;
}

// ── Loading banner ────────────────────────────────────────────────────────────
function setLoading(on, msg = "Processing…") {
  const b = $("#loadingBanner");
  b.hidden = !on;
  if (on) {
    $("#loadingMsg").textContent = msg;
    setStatus("busy");
  } else {
    setStatus("ok");
  }
}

// ── Heatmap color ─────────────────────────────────────────────────────────────
function heat(v, clamp = 1) {
  const x = Math.max(0, Math.min(clamp, v == null ? 0 : v)) / clamp;
  const hue = Math.round(120 * x);
  return { bg: `hsl(${hue} 70% 36% / 0.9)`, fg: x > 0.5 ? "#051a05" : "#e0ffe0" };
}

function cellHTML(v, title = "") {
  const h = heat(v);
  const vv = v == null ? "—" : Number(v).toFixed(3);
  return `<span class="cell" style="background:${h.bg};color:${h.fg}" title="${esc(title)}">${vv}</span>`;
}

// ── Number formatters ─────────────────────────────────────────────────────────
function fmtHz(v) {
  if (v == null || !isFinite(v)) return "—";
  if (Math.abs(v) >= 1e9) return (v / 1e9).toFixed(3) + " GHz";
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(3) + " MHz";
  if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(2) + " kHz";
  return v.toFixed(0) + " Hz";
}

function fmtNum(v, places = 2) {
  return v == null || !isFinite(v) ? "—" : Number(v).toFixed(places);
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────
async function apiUpload(url, extra = {}) {
  const fd = new FormData();
  fd.append("file", state.file);
  const q = new URLSearchParams();
  if (state.dataType) q.set("data_type", state.dataType);
  if (state.sampleRate) q.set("sample_rate", state.sampleRate);
  for (const [k, v] of Object.entries(extra)) if (v != null) q.set(k, v);
  const res = await fetch(url + (q.toString() ? "?" + q : ""), { method: "POST", body: fd });
  return handle(res);
}

async function apiGet(url) {
  return handle(await fetch(url));
}

async function apiPostJSON(url, body) {
  return handle(await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }));
}

async function handle(res) {
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const msg = (data && (data.detail || data.message)) || res.statusText;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    await apiGet("/api/health");
    setStatus("ok");
  } catch {
    setStatus("error");
  }
}

// ── Chain label helpers ───────────────────────────────────────────────────────
function deintStr(d) {
  if (!d) return "none";
  return d.type === "block" ? `block(${d.block_n}, ${d.block_m})` : "none";
}

function chainSpec(ch) {
  if (!ch) return "";
  const fec = ch.fec ? ch.fec.mode : "none";
  const di = ch.deinterleaver ? deintStr(ch.deinterleaver) : "none";
  const fr = ch.frame ? ch.frame.frame_name : "none";
  return `${fec}|${di}|${fr}`;
}

function chainLabel(chain) {
  if (!chain) return "—";
  return `${chain.modulation} | ${chainSpec(chain)}`;
}

// ── Nav ───────────────────────────────────────────────────────────────────────
function nav(sec) {
  $$("#nav .pill").forEach(p => p.classList.toggle("active", p.dataset.sec === sec));
  $$("main .sec").forEach(s => s.classList.toggle("active", s.id === sec));
}

// ── Modal ─────────────────────────────────────────────────────────────────────
function openModal(title, bodyHTML) {
  $("#modalTitle").innerHTML = title;
  $("#modalBody").innerHTML = bodyHTML;
  $("#modalOverlay").hidden = false;
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 1 — Signal input & analysis
// ══════════════════════════════════════════════════════════════════════════════
function renderSignalInfo(res) {
  state.upload = res;
  const items = [
    { k: "File", v: res.filename },
    { k: "Format", v: res.data_type },
    { k: "Samples", v: (res.num_samples || 0).toLocaleString() },
    { k: "Sample Rate", v: res.sample_rate ? fmtHz(res.sample_rate) : "—" },
    { k: "Center Freq", v: res.center_frequency != null ? fmtHz(res.center_frequency) : "—" },
    { k: "Duration", v: res.duration_seconds ? fmtNum(res.duration_seconds, 4) + " s" : "—" },
  ];
  $("#signalInfo").innerHTML = `
    <div class="card">
      <div class="card-title">Parsed Signal</div>
      <div class="kv">
        ${items.map(i => `
          <div class="kv-item">
            <div class="k">${esc(i.k)}</div>
            <div class="v">${esc(i.v)}</div>
          </div>`).join("")}
      </div>
    </div>`;
  $("#stepButtons").hidden = false;
  markStep("stepAnalyze", false);
}

function renderAnalysis(res) {
  state.analyze = res;
  const md = res.metadata || {};
  const snr = res.snr || {};
  const ch = res.characterization || {};
  const mod = res.modulation_classification || {};
  const fv = ch.features || {};
  const bw = res.bandwidth || {};
  const sr = ch.symbol_rate || {};
  const cfo = ch.freq_offset || {};

  const items = [
    { k: "SNR (PSD)", v: fmtNum(snr.snr_db) + " dB", cls: snr.snr_db > 10 ? "green" : snr.snr_db > 0 ? "accent" : "red" },
    { k: "Noise Floor", v: fmtNum(snr.noise_floor_db) + " dBFS" },
    { k: "BW (-3dB)", v: fmtHz(bw.bandwidth_hz) },
    { k: "Symbol Rate", v: sr.symbol_rate_hz ? fmtHz(sr.symbol_rate_hz) + "Bd" : "—" },
    { k: "CFO", v: fmtHz(cfo.freq_offset_hz) },
    { k: "C₄₂ / C₂₁", v: `${fmtNum(fv.C42_norm)} / ${fmtNum(fv.C21_sym)}` },
    { k: "PAPR", v: fmtNum(fv.papr_db) + " dB" },
    { k: "Top Candidate", v: mod.top_candidate || "—", cls: "accent" },
  ];

  const candidates = mod.candidates || [];
  const maxScore = candidates.length ? Math.max(...candidates.map(c => c.score || 0)) : 1;

  $("#analysisHead").hidden = false;
  $("#analysisHead").innerHTML = `
    <div class="card" style="margin-top:14px">
      <div class="card-title">Analysis Results</div>
      ${statusBanner(res)}
      <div class="kv">
        ${items.map(i => `
          <div class="kv-item">
            <div class="k">${esc(i.k)}</div>
            <div class="v ${i.cls || ""}">${esc(i.v)}</div>
          </div>`).join("")}
      </div>
      <div style="margin-top:14px">
        <div class="card-title">Modulation Candidates</div>
        <ul class="cand-list">
          ${candidates.map((c, i) => `
            <li class="cand-item ${i === 0 ? "top" : ""}">
              <span class="cand-name">${esc(c.modulation)}</span>
              <div class="cand-bar-wrap">
                <div class="cand-bar" style="width:${Math.round(100 * (c.score || 0) / maxScore)}%"></div>
              </div>
              <span class="cand-score">score ${fmtNum(c.score, 3)}</span>
              <span class="cand-conf">conf ${fmtNum(c.confidence, 3)}</span>
            </li>`).join("") || "<li style='padding:8px;color:var(--muted)'>No candidates</li>"}
        </ul>
        ${g1TriggerBanner(mod)}
      </div>
    </div>`;

  markStep("stepAnalyze", true);
  renderGates();
  fetchAndRenderWaterfall();
}

function statusBanner(res) {
  return `<div class="banner ok"><span class="banner-icon">✅</span>Detection &amp; characterization complete</div>`;
}

function g1TriggerBanner(mod) {
  const cands = mod.candidates || [];
  if (cands.length < 2) return "";
  const margin = cands[0].score - cands[1].score;
  const fires = margin < 0.10 || (mod.confidence || 0) < 0.5;
  const cls = fires ? "warn" : "ok";
  const icon = fires ? "⚠️" : "✅";
  const msg = fires
    ? `G1 gate fires — top margin ${fmtNum(margin, 3)} &lt; 0.10. Human decision required.`
    : `G1 automatic — margin ${fmtNum(margin, 3)} ≥ 0.10`;
  return `<div class="banner ${cls}" style="margin-top:10px"><span class="banner-icon">${icon}</span>${msg}</div>`;
}

// ── Fetch & render waterfall/PSD ──────────────────────────────────────────────
async function fetchAndRenderWaterfall() {
  if (!state.file) return;
  try {
    const res = await apiUpload("/api/waterfall", { fft_size: 512, max_time_bins: 256 });
    state.waterfall = res;
    renderWaterfall(res);
    renderPSD(state.analyze);
  } catch (e) {
    console.warn("waterfall fetch failed:", e.message);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// CANVAS RENDERERS
// ══════════════════════════════════════════════════════════════════════════════

// ── Waterfall ─────────────────────────────────────────────────────────────────
function renderWaterfall(data) {
  const canvas = $("#canvasWaterfall");
  const placeholder = $("#wfPlaceholder");
  if (!canvas) return;

  const matrix = data.spectrogram_db;
  if (!matrix || !matrix.length) return;

  const rows = matrix.length;
  const cols = matrix[0].length;
  placeholder.style.display = "none";
  canvas.style.display = "block";
  canvas.width = canvas.offsetWidth || 800;
  canvas.height = 220;

  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  // Find global min/max for normalization
  let gMin = Infinity, gMax = -Infinity;
  for (const row of matrix) {
    for (const v of row) {
      if (isFinite(v)) { gMin = Math.min(gMin, v); gMax = Math.max(gMax, v); }
    }
  }
  const range = gMax - gMin || 1;

  const imgData = ctx.createImageData(W, H);
  const pixW = W / cols;
  const pixH = H / rows;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const norm = Math.max(0, Math.min(1, (matrix[r][c] - gMin) / range));
      const [rr, gg, bb] = waterfallColor(norm);
      const px = Math.round(c * pixW);
      const py = Math.round(r * pixH);
      const pw = Math.max(1, Math.round(pixW));
      const ph = Math.max(1, Math.round(pixH));
      for (let dy = 0; dy < ph; dy++) {
        for (let dx = 0; dx < pw; dx++) {
          const idx = ((py + dy) * W + (px + dx)) * 4;
          imgData.data[idx]   = rr;
          imgData.data[idx+1] = gg;
          imgData.data[idx+2] = bb;
          imgData.data[idx+3] = 255;
        }
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);

  // Draw segment overlays if available
  const segs = state.analyze && state.analyze.segments && state.analyze.segments.segments;
  if (segs && segs.length) {
    ctx.strokeStyle = "rgba(45,232,158,0.8)";
    ctx.lineWidth = 1.5;
    const sr = state.upload && state.upload.sample_rate;
    const dur = state.upload && state.upload.duration_seconds;
    if (sr && dur) {
      const totalSamples = state.upload.num_samples;
      segs.forEach(seg => {
        const x1 = Math.round((seg.start_sample / totalSamples) * W);
        const x2 = Math.round((seg.end_sample / totalSamples) * W);
        ctx.strokeRect(x1, 0, x2 - x1, H);
      });
    }
  }

  // Axis labels
  const freqs = data.frequencies_hz;
  const times = data.time_bins_seconds;
  if (freqs && freqs.length && times && times.length) {
    $("#wfInfo").textContent =
      `${fmtHz(freqs[0])} – ${fmtHz(freqs[freqs.length-1])} · ${fmtNum(times[times.length-1]*1000,2)} ms`;
  }
}

// Classic thermal / turbo colormap
function waterfallColor(t) {
  const stops = [
    [0,    [2,  6,  18]],
    [0.15, [10, 30, 80]],
    [0.3,  [20, 90, 140]],
    [0.45, [20, 180, 200]],
    [0.6,  [40, 220, 100]],
    [0.75, [255, 220, 0]],
    [0.88, [255, 100, 0]],
    [1.0,  [255, 20,  20]],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const s0 = stops[i-1], s1 = stops[i];
      const f = (t - s0[0]) / (s1[0] - s0[0]);
      return s0[1].map((c, j) => Math.round(c + f * (s1[1][j] - c)));
    }
  }
  return [255, 20, 20];
}

// ── PSD spectrum ──────────────────────────────────────────────────────────────
function renderPSD(data) {
  const canvas = $("#canvasPSD");
  const placeholder = $("#psdPlaceholder");
  if (!canvas || !data) return;

  const psd = data.psd;
  if (!psd || !psd.power_db || !psd.frequencies_hz) return;

  placeholder.style.display = "none";
  canvas.style.display = "block";
  canvas.width = canvas.offsetWidth || 400;
  canvas.height = 220;

  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const PAD = { top: 10, right: 10, bottom: 26, left: 44 };
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;

  const freqs = psd.frequencies_hz;
  const power = psd.power_db;
  const noiseDb = (data.noise_floor || {}).noise_floor_db;

  let minP = Math.min(...power.filter(isFinite));
  let maxP = Math.max(...power.filter(isFinite));
  const padDb = (maxP - minP) * 0.08 || 5;
  minP -= padDb; maxP += padDb;
  const rng = maxP - minP || 1;

  const toX = i => PAD.left + (i / (freqs.length - 1)) * pw;
  const toY = v => PAD.top + ph - ((v - minP) / rng) * ph;

  // Background
  ctx.fillStyle = "#0c1118";
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = "rgba(30,42,56,0.8)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = PAD.top + (i / 5) * ph;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
    const label = fmtNum(maxP - i * (rng / 5), 0) + " dB";
    ctx.fillStyle = "rgba(77,98,120,0.9)";
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.fillText(label, 2, y + 3);
  }

  // Noise floor line
  if (noiseDb != null && isFinite(noiseDb)) {
    const yn = toY(noiseDb);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(245,166,35,0.7)";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(PAD.left, yn); ctx.lineTo(W - PAD.right, yn); ctx.stroke();
    ctx.fillStyle = "rgba(245,166,35,0.9)";
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.fillText("noise floor " + fmtNum(noiseDb, 1) + " dB", PAD.left + 4, yn - 3);
    ctx.restore();
  }

  // Bandwidth markers
  const bw3 = (data.bandwidth || {}).bandwidth_hz;
  const fc = (data.metadata || {}).center_frequency || 0;
  if (bw3 && freqs.length > 1) {
    const fRange = freqs[freqs.length - 1] - freqs[0];
    const fMin = fc - bw3 / 2;
    const fMax = fc + bw3 / 2;
    const x1 = PAD.left + ((fMin - freqs[0]) / fRange) * pw;
    const x2 = PAD.left + ((fMax - freqs[0]) / fRange) * pw;
    ctx.save();
    ctx.fillStyle = "rgba(59,158,255,0.1)";
    ctx.fillRect(x1, PAD.top, x2 - x1, ph);
    ctx.strokeStyle = "rgba(59,158,255,0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x1, PAD.top); ctx.lineTo(x1, PAD.top + ph); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2, PAD.top); ctx.lineTo(x2, PAD.top + ph); ctx.stroke();
    ctx.restore();
  }

  // PSD fill
  const grad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + ph);
  grad.addColorStop(0, "rgba(59,158,255,0.5)");
  grad.addColorStop(1, "rgba(59,158,255,0.02)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(power[0]));
  for (let i = 1; i < power.length; i++) ctx.lineTo(toX(i), toY(isFinite(power[i]) ? power[i] : minP));
  ctx.lineTo(toX(power.length - 1), PAD.top + ph);
  ctx.lineTo(toX(0), PAD.top + ph);
  ctx.closePath();
  ctx.fill();

  // PSD line
  ctx.strokeStyle = "#3b9eff";
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(power[0]));
  for (let i = 1; i < power.length; i++) ctx.lineTo(toX(i), toY(isFinite(power[i]) ? power[i] : minP));
  ctx.stroke();

  // Freq axis labels
  ctx.fillStyle = "rgba(77,98,120,0.9)";
  ctx.font = "9px JetBrains Mono, monospace";
  ctx.textAlign = "center";
  [0, 0.25, 0.5, 0.75, 1].forEach(t => {
    const fi = Math.round(t * (freqs.length - 1));
    ctx.fillText(fmtHz(freqs[fi]), PAD.left + t * pw, H - 6);
  });
  ctx.textAlign = "left";

  // Info badge
  $("#psdInfo").textContent = `${freqs.length} bins · ${fmtNum(psd.resolution_hz || 0, 0)} Hz/bin`;
}

// ── Constellation diagram ─────────────────────────────────────────────────────
function renderConstellation(symbols, modulation) {
  const canvas = $("#canvasConst");
  const placeholder = $("#constPlaceholder");
  if (!canvas || !symbols || !symbols.length) return;

  placeholder.style.display = "none";
  canvas.style.display = "block";
  canvas.width = canvas.offsetWidth || 300;
  canvas.height = 220;

  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const cx = W / 2, cy = H / 2;

  // Find scale
  let maxVal = 0;
  symbols.forEach(s => maxVal = Math.max(maxVal, Math.abs(s[0]), Math.abs(s[1])));
  if (!maxVal) maxVal = 1;
  const scale = (Math.min(W, H) / 2 - 16) / (maxVal * 1.1);

  ctx.fillStyle = "#080c10";
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = "rgba(30,42,56,0.6)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();

  // Unit circle
  ctx.strokeStyle = "rgba(59,158,255,0.15)";
  ctx.beginPath(); ctx.arc(cx, cy, scale * maxVal * 0.7, 0, Math.PI * 2); ctx.stroke();

  // Decision boundaries
  drawDecisionBoundaries(ctx, cx, cy, scale, modulation);

  // Points
  const maxPts = Math.min(symbols.length, 2000);
  const sample = maxPts < symbols.length
    ? symbols.filter((_, i) => i % Math.ceil(symbols.length / maxPts) === 0)
    : symbols;

  sample.forEach(s => {
    const x = cx + s[0] * scale;
    const y = cy - s[1] * scale;
    ctx.fillStyle = "rgba(59,158,255,0.7)";
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, Math.PI * 2);
    ctx.fill();
  });

  // EVM ring (if demod available)
  const hyps = state.hypothesize && state.hypothesize.chains;
  if (hyps && hyps.length) {
    const top = hyps[0].chain;
    const evm = top && top.demodulation_evidence &&
      top.demodulation_evidence.find(e => e.measurement === "evm_pct");
    if (evm && isFinite(evm.value)) {
      const evmFrac = evm.value / 100;
      ctx.strokeStyle = "rgba(45,232,158,0.4)";
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, scale * maxVal * evmFrac, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Axis labels
  ctx.fillStyle = "rgba(77,98,120,0.8)";
  ctx.font = "9px JetBrains Mono, monospace";
  ctx.fillText("I", W - 14, cy - 4);
  ctx.fillText("Q", cx + 4, 12);

  // Info
  $("#constInfo").textContent = `${esc(modulation || "?")} · ${sample.length} symbols`;
}

function drawDecisionBoundaries(ctx, cx, cy, scale, mod) {
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 4]);
  if (mod === "QPSK" || mod === "BPSK") {
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, cy * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(cx * 2, cy); ctx.stroke();
  } else if (mod === "16QAM") {
    const d = scale * 0.65;
    [-1, 0, 1].forEach(i => {
      ctx.beginPath(); ctx.moveTo(cx + i * d, 0); ctx.lineTo(cx + i * d, cy * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, cy + i * d); ctx.lineTo(cx * 2, cy + i * d); ctx.stroke();
    });
  }
  ctx.setLineDash([]);
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 2 — Score Ledger Heatmap
// ══════════════════════════════════════════════════════════════════════════════
const LEDGER_COLS = [
  { key: "modulation_score",   label: "Mod",    w: "0.25" },
  { key: "demodulation_score", label: "Demod",  w: "0.25" },
  { key: "decoding_score",     label: "Decode", w: "0.35" },
];

function renderLedger(res) {
  state.hypothesize = res;
  const rows = res.chains || [];

  if (!rows.length) {
    $("#ledger").innerHTML = `<div class="banner warn"><span class="banner-icon">⚠️</span>No chains returned by the hypothesis engine</div>`;
    return;
  }

  const head = `
    <tr>
      <th>#</th><th>Chain</th>
      ${LEDGER_COLS.map(c => `<th>${c.label} (w${c.w})</th>`).join("")}
      <th>Composite</th><th>Uncertainty</th><th>CRC</th>
    </tr>`;

  const body = rows.map((r, i) => {
    const ch = r.chain;
    const hasCrc = (ch.decoding_results || []).some(d => d.crc_valid);
    const crcTag = hasCrc
      ? `<span class="tag green">PASS</span>`
      : `<span class="tag amber">FAIL</span>`;
    return `<tr class="clickable ${i === 0 ? "winner" : ""}" data-chain="${i}">
      <td style="color:var(--muted)">${i + 1}</td>
      <td><code style="font-size:11px">${esc(chainLabel(ch))}</code></td>
      ${LEDGER_COLS.map(c =>
        `<td data-stage="${c.key}">${cellHTML(ch[c.key], "Click: stage evidence items")}</td>`
      ).join("")}
      <td data-stage="composite">${cellHTML(ch.composite_score, "Click: full WHY report")}</td>
      <td data-stage="uncertainty">${cellHTML(1 - (ch.score_uncertainty || 0))}</td>
      <td>${crcTag}</td>
    </tr>`;
  }).join("");

  $("#ledger").innerHTML = `
    <div class="banner ok"><span class="banner-icon">📋</span>${rows.length} chains scored (weights: mod 0.25 · demod 0.25 · decode 0.35)</div>
    <div class="ledger-wrap">
      <table><thead>${head}</thead><tbody>${body}</tbody></table>
    </div>`;

  $$("#ledger tr[data-chain]").forEach(tr => {
    tr.addEventListener("click", ev => {
      const cell = ev.target.closest("td[data-stage]");
      const chainIdx = +tr.dataset.chain;
      if (cell) openEvidenceModal(chainIdx, cell.dataset.stage);
      else openChainModal(chainIdx);
    });
  });

  markStep("stepHypothesize", true);
  renderGates();

  // Try to render constellation from top chain
  const topChain = rows[0] && rows[0].chain;
  if (topChain) {
    const demodEv = topChain.demodulation_evidence || [];
    const symItem = demodEv.find(e => e.measurement === "num_symbols");
    if (symItem) {
      // Fetch demod symbols for constellation
      fetchConstellation(topChain.modulation);
    }
  }
}

async function fetchConstellation(modulation) {
  if (!state.file) return;
  try {
    const res = await apiUpload("/api/demodulate", { modulation });
    const demod = res.demodulation || {};
    const syms = demod.symbols;
    if (syms && syms.length) {
      renderConstellation(syms, modulation);
    }
  } catch (e) {
    console.warn("constellation fetch failed:", e.message);
  }
}

// ── Evidence modal helpers ────────────────────────────────────────────────────
function openEvidenceModal(chainIdx, stage) {
  const row = state.hypothesize.chains[chainIdx];
  const ch = row.chain;
  const stageMap = {
    modulation_score:   ["modulation_evidence",   "Modulation Evidence"],
    demodulation_score: ["demodulation_evidence",  "Demodulation Evidence"],
    decoding_score:     ["decoding_evidence",      "Decoding Evidence"],
  };
  const title = `Evidence → <code>${esc(chainLabel(ch))}</code> · ${stage.replace("_score", "")}`;
  let html = "";
  if (stage === "composite") {
    html = `<div class="why-panel">${esc(row.why || "No WHY report available")}</div>`;
    html += `<p class="corpus-stat">composite ${fmtNum(ch.composite_score, 4)} · uncertainty ${fmtNum(ch.score_uncertainty, 4)}</p><hr>`;
    for (const [k, [key, label]] of Object.entries(stageMap)) {
      html += `<h4 style="margin:12px 0 6px;font-size:13px">${label} · ${fmtNum(ch[k], 3)}</h4>`;
      html += renderItems(ch[key]);
    }
  } else if (stageMap[stage]) {
    const [key, label] = stageMap[stage];
    html = `<h4 style="margin:0 0 8px;font-size:13px">${label} · value ${fmtNum(ch[stage], 3)}</h4>`;
    html += renderItems(ch[key]);
  }
  openModal(title, html);
}

function openChainModal(chainIdx) {
  const row = state.hypothesize.chains[chainIdx];
  const ch = row.chain;
  const html = `
    <div class="why-panel">${esc(row.why || "No WHY report")}</div>
    <p class="corpus-stat">composite: ${fmtNum(ch.composite_score, 4)} · uncertainty: ${fmtNum(ch.score_uncertainty, 4)}</p>
    ${ch.alternative_paths && ch.alternative_paths.length ? `<p class="corpus-stat">alternatives: ${esc(ch.alternative_paths.join("; "))}</p>` : ""}`;
  openModal(`Chain #${chainIdx + 1} · <code>${esc(chainLabel(ch))}</code>`, html);
}

function renderItems(items) {
  if (!items || !items.length) return `<p style="color:var(--muted);font-size:12px">No evidence items</p>`;
  return items.map(it => {
    const sup = it.supports;
    return `<div class="ev-item ${sup ? "supports" : "contradicts"}">
      <div class="ev-item-head">
        <b>${esc(it.measurement || it.candidate)}</b>
        <span class="ev-badge stage">${esc(it.stage)}</span>
        <span class="ev-badge ${sup ? "sup" : "contra"}">${sup ? "supports" : "contradicts"}</span>
        <span class="ev-badge stage">${esc(it.quality || "")}</span>
      </div>
      <div class="ev-detail">
        value = ${fmtNum(it.value)} · norm = ${fmtNum(it.value_norm)} · weight = ${fmtNum(it.weight)}
        · candidate: ${esc(it.candidate)} · provenance: ${esc(it.provenance)}
        ${it.source ? "· source: " + esc(it.source) : ""}
        ${it.family ? "· family: " + esc(it.family) : ""}
      </div>
    </div>`;
  }).join("");
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 3 — Evidence Trail
// ══════════════════════════════════════════════════════════════════════════════
function renderEvidence(res) {
  const g = res.evidence_graph || {};
  const hyps = (g.hypotheses || []).slice().sort((a, b) => b.score - a.score);

  const html = hyps.map(h => `
    <div class="ev-item ${h.score > 0.5 ? "supports" : ""}">
      <div class="ev-item-head">
        <b>${esc(h.name)}</b>
        <span class="ev-badge stage">${esc(h.kind)}</span>
        ${cellHTML(h.score)}
        <span class="ev-badge stage">${esc(h.quality)}</span>
        <span style="font-size:11px;color:var(--muted)">${h.num_items} evidence items</span>
      </div>
      <div class="ev-detail">
        support ${fmtNum(h.support_weight, 3)} · contra ${fmtNum(h.contra_weight, 3)} · contradiction ratio ${fmtNum(h.contradiction_ratio, 3)}
        ${h.parents && h.parents.length ? "· parents: " + esc(h.parents.join(", ")) : ""}
      </div>
    </div>`).join("");

  $("#evidenceTrail").innerHTML = `
    <div class="banner info"><span class="banner-icon">🔬</span>
      ${esc(g.name || "Evidence graph")} — ${(g.items || []).length} evidence items, ${hyps.length} hypotheses
    </div>
    ${html || `<p style="color:var(--muted)">No hypotheses. Run hypothesize first.</p>`}`;
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 4 — Advisory
// ══════════════════════════════════════════════════════════════════════════════
function renderAdvisory(consult) {
  state.consult = consult;
  const a = consult.advisory || {};
  const corpus = consult.corpus || {};

  const suppressed = a.suppressed;
  const bannerCls = suppressed ? "warn" : "ok";
  const bannerIcon = suppressed ? "⚠️" : "✅";
  const bannerMsg = suppressed
    ? `Advisory suppressed — corpus too thin. ${esc(a.suppression_reason || "")}. Never shown as authority.`
    : `${a.neighbours || 0} similar verified signals found · distance [${(a.feature_distance_range||[0,0])[0].toFixed(2)}, ${(a.feature_distance_range||[0,0])[1].toFixed(2)}]`;

  const outcomes = a.verified_outcomes || {};
  const totalN = Object.values(outcomes).reduce((s, v) => s + v, 0) || 1;
  const outBars = Object.entries(outcomes)
    .sort((x, y) => y[1] - x[1])
    .map(([m, c]) => `
      <div class="adv-bar-row">
        <span class="adv-bar-label">${esc(m)}</span>
        <div class="adv-bar"><span style="width:${Math.round(100 * c / totalN)}%"></span></div>
        <span class="adv-bar-count">${c}</span>
      </div>`).join("");

  const chainRows = (a.chains_that_validated || [])
    .map(c => `<tr>
      <td><code style="font-size:11px">${esc(c.chain)}</code></td>
      <td style="text-align:center">${c.count}</td>
      <td>${cellHTML(c.crc_pass_rate)}</td>
    </tr>`).join("");

  const mix = corpus.source_mix || {};
  const mixLine = Object.entries(mix).map(([k, v]) => `${esc(k)}: ${v}`).join(" · ") || "empty corpus";

  $("#advisoryPanel").innerHTML = `
    <div class="banner ${bannerCls}" style="margin-top:14px">
      <span class="banner-icon">${bannerIcon}</span>${bannerMsg}
    </div>
    <div class="kv" style="margin:14px 0">
      <div class="kv-item"><div class="k">Neighbours</div><div class="v accent">${a.neighbours || 0}</div></div>
      <div class="kv-item"><div class="k">Dist range</div><div class="v">[${(a.feature_distance_range||[0,0])[0].toFixed(2)}, ${(a.feature_distance_range||[0,0])[1].toFixed(2)}]</div></div>
      <div class="kv-item"><div class="k">Corpus size</div><div class="v">${corpus.size || 0}</div></div>
      <div class="kv-item"><div class="k">Source mix</div><div class="v" style="font-size:12px">${esc(mixLine)}</div></div>
    </div>
    ${a.caveat ? `<div class="banner warn"><span class="banner-icon">⚠️</span>${esc(a.caveat)}</div>` : ""}

    <details style="margin:14px 0">
      <summary>Advisory details — verified outcomes &amp; validated chains</summary>
      <div style="padding:12px 0">
        <div class="card-title" style="margin-bottom:10px">Verified modulations among neighbours</div>
        <div class="adv-bars">${outBars || "<p style='color:var(--muted)'>None</p>"}</div>
        <div class="card-title" style="margin:14px 0 8px">Chains that validated (by CRC pass rate)</div>
        <table>
          <thead><tr><th>Chain</th><th>Count</th><th>CRC rate</th></tr></thead>
          <tbody>${chainRows || "<tr><td colspan='3' style='color:var(--muted)'>None</td></tr>"}</tbody>
        </table>
      </div>
    </details>
    <p class="corpus-stat">advisory is read-only retrieval — it entered the evidence graph <b>never</b> and touched a composite score <b>never</b>.</p>`;

  renderConfusion(consult);
  markStep("stepConsult", true);
  renderGates();
}

function renderConfusion(consult) {
  const ana = state.analyze && state.analyze.modulation_classification;
  const cands = (ana && ana.candidates || []).slice(0, 5);
  const outcome = (consult.advisory || {}).verified_outcomes || {};
  const cols = Object.keys(outcome);
  if (!cols.length || !cands.length) {
    $("#confusionMatrix").innerHTML = `<p style="color:var(--muted);font-size:12px">Requires analysis + advisor consult</p>`;
    return;
  }
  const maxVal = Math.max(...Object.values(outcome));
  const head = `<tr><th>Current candidate</th>${cols.map(c => `<th>${esc(c)}</th>`).join("")}<th>Confidence</th></tr>`;
  const body = cands.map(c => {
    const cells = cols.map(col => {
      const count = outcome[col] || 0;
      const match = col === c.modulation;
      const h = heat(maxVal > 0 ? count / maxVal : 0);
      const style = match ? `background:${h.bg};color:${h.fg};font-weight:700` : "";
      return `<td style="${style}">${count}${match ? " ✓" : ""}</td>`;
    }).join("");
    return `<tr>
      <td><b>${esc(c.modulation)}</b></td>${cells}
      <td><code style="font-size:11px">${fmtNum(c.confidence, 3)}</code></td>
    </tr>`;
  }).join("");

  $("#confusionMatrix").innerHTML = `
    <div class="kv" style="margin-bottom:10px">
      <div class="kv-item"><div class="k">Rows</div><div class="v" style="font-size:12px">Current classification candidates</div></div>
      <div class="kv-item"><div class="k">Columns</div><div class="v" style="font-size:12px">Verified neighbour modulations</div></div>
      <div class="kv-item"><div class="k">Heat</div><div class="v" style="font-size:12px">Share of neighbour vote (column)</div></div>
    </div>
    <table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 5 — HITL Gates
// ══════════════════════════════════════════════════════════════════════════════
function renderGates() {
  renderGateG1();
  renderGateG2();
  renderGateG3();
  renderAudit();
}

function setGateState(gateId, gState) {
  const card = $(`#gate${gateId}`);
  const label = card && card.querySelector(".gate-state");
  if (!card || !label) return;
  card.className = "gate-card" + (gState === "fired" ? " fired" : gState === "done" ? " done" : "");
  const texts = { fired: "fired", done: "decided", ready: "ready", pending: "pending", scoped: "scoped" };
  label.textContent = texts[gState] || gState;
}

function renderGateG1() {
  const ana = state.analyze && state.analyze.modulation_classification;
  const cands = (ana && ana.candidates || []).slice(0, 4);
  const done = state.gates.G1.length > 0;
  setGateState("G1", done ? "done" : cands.length ? "fired" : "pending");

  if (!cands.length) {
    $("[data-role=G1-trigger]").textContent = "Run analyze first";
    $("#g1body").innerHTML = "";
    return;
  }

  const margin = cands.length > 1 ? cands[0].score - cands[1].score : cands[0].score;
  const fires = margin < 0.10 || (ana.confidence || 0) < 0.5;
  const trig = $("[data-role=G1-trigger]");
  trig.textContent = fires
    ? `Gate fires — margin ${fmtNum(margin, 3)} < 0.10`
    : `Margin ${fmtNum(margin, 3)} ≥ 0.10 — would be automatic`;
  trig.className = "gate-trigger" + (fires ? " fires" : " pass");

  const opts = cands.map((c, i) =>
    `<option value="${esc(c.modulation)}" ${i === 0 ? "selected" : ""}>${esc(c.modulation)} (${fmtNum(c.score, 3)})</option>`
  ).join("");

  $("#g1body").innerHTML = `
    <p style="font-size:12px;color:var(--fg2);margin-bottom:8px">Top candidate: <b>${esc(ana.top_candidate || "")}</b>. Confirm or override.</p>
    <select id="g1pick" style="width:100%;margin-bottom:8px">${opts}</select>
    <div class="row">
      <button class="ok sm" id="g1confirm">✓ Confirm ${esc(ana.top_candidate || "")}</button>
      <button class="danger sm" id="g1override">⚡ Override to selection</button>
    </div>`;

  $("#g1confirm").addEventListener("click", () =>
    gateAction("G1", "confirm", ana.top_candidate, ana.top_candidate, null));
  $("#g1override").addEventListener("click", () =>
    gateAction("G1", "override", ana.top_candidate, $("#g1pick").value, null));

  renderGateLog("g1log", state.gates.G1);
}

function renderGateG2() {
  const done = state.gates.G2.length > 0;
  setGateState("G2", done ? "scoped" : "ready");

  const ilvOpts = ["none", "block(4, 8)", "block(2, 5)"].map(v => `<option>${esc(v)}</option>`).join("");
  const fecOpts = ["none", "viterbi", "rs255_223", "rs255_239", "viterbi+rs255_223"].map(v => `<option>${esc(v)}</option>`).join("");
  const frameOpts = ["preamble+crc16", "preamble+crc32"].map(v => `<option>${esc(v)}</option>`).join("");

  $("#g2body").innerHTML = `
    <p style="font-size:12px;color:var(--fg2);margin-bottom:8px">Pin interleaver / FEC before the search, or accept the full grid.</p>
    <select id="g2ilv" style="width:100%;margin-bottom:6px">${ilvOpts}</select>
    <select id="g2fec" style="width:100%;margin-bottom:6px">${fecOpts}</select>
    <select id="g2frame" style="width:100%;margin-bottom:8px">${frameOpts}</select>
    <div class="row">
      <button class="ok sm" id="g2accept">✓ Accept full grid</button>
      <button class="danger sm" id="g2pin">📌 Pin this chain</button>
    </div>`;

  $("#g2accept").addEventListener("click", () =>
    gateAction("G2", "confirm", "full grid", "full grid", null));
  $("#g2pin").addEventListener("click", () => {
    const to = `${$("#g2fec").value}|${$("#g2ilv").value}|${$("#g2frame").value}`;
    gateAction("G2", "override", "full grid", to, to);
  });

  renderGateLog("g2log", state.gates.G2);
}

function renderGateG3() {
  const hyps = state.hypothesize && state.hypothesize.chains || [];
  const done = state.gates.G3.length > 0;
  setGateState("G3", done ? "done" : hyps.length ? "fired" : "pending");

  if (!hyps.length) {
    $("[data-role=G3-trigger]").textContent = "Run hypothesize first";
    $("#g3body").innerHTML = "";
    return;
  }

  const top = hyps[0].chain;
  const composite = top.composite_score || 0;
  const anyCrc = hyps.some(h => (h.chain.decoding_results || []).some(r => r.crc_valid));
  const fires = composite < 0.6 || !anyCrc;
  const trig = $("[data-role=G3-trigger]");
  trig.textContent = fires
    ? `Gate fires — composite ${fmtNum(composite, 3)} < 0.6 or no CRC pass`
    : `Composite ${fmtNum(composite, 3)} — automatic`;
  trig.className = "gate-trigger" + (fires ? " fires" : " pass");

  const chainOpts = hyps.slice(0, 8).map((h, i) =>
    `<option value="${i}">#${i+1} ${esc(chainLabel(h.chain))} (${fmtNum(h.chain.composite_score, 3)})</option>`
  ).join("");

  $("#g3body").innerHTML = `
    <p style="font-size:12px;color:var(--fg2);margin-bottom:8px">
      Top chain: <code style="font-size:11px">${esc(chainLabel(top))}</code> · composite <b>${fmtNum(composite, 3)}</b>
    </p>
    <select id="g3pick" style="width:100%;margin-bottom:8px">${chainOpts}</select>
    <div class="row">
      <button class="ok sm" id="g3accept">✓ Accept #1</button>
      <button class="danger sm" id="g3force">⚡ Force selected</button>
    </div>`;

  $("#g3accept").addEventListener("click", () =>
    gateAction("G3", "confirm", chainLabel(top), chainLabel(top), chainSpec(top)));
  $("#g3force").addEventListener("click", () => {
    const h = hyps[+$("#g3pick").value];
    gateAction("G3", "override", chainLabel(top), chainLabel(h.chain), chainSpec(h.chain));
  });

  renderGateLog("g3log", state.gates.G3);
}

function renderGateLog(logId, list) {
  const el = $(`#${logId}`);
  if (!el || !list.length) { if (el) el.innerHTML = ""; return; }
  el.innerHTML = `<div class="gate-log">` +
    list.map(e => `
      <div class="gate-log-item">
        <span class="action ${e.action}">${esc(e.action)}</span>
        <span class="arrow">→</span>
        <b>${esc(e.to)}</b>
        ${e.chain ? `<code style="font-size:10px;color:var(--muted)">(${esc(e.chain)})</code>` : ""}
      </div>`).join("") +
    "</div>";
}

async function gateAction(gate, action, from, to, chain) {
  if (!state.consult) { toast("Consult the advisor first (step 4)", "warn"); return; }
  const payload = {
    run_id: state.runId,
    gate, action,
    from_value: from,
    to_value: to,
    chain,
    snr_db: (state.consult.physical_estimates || {}).snr_db,
    crc_validated: false,
    feature_vector: state.consult.query_vector || [],
  };
  try {
    const res = await apiPostJSON("/api/hitl/gate", payload);
    state.gates[gate].push({ gate, action, from, to, chain });
    const corpus = res.corpus || {};
    toast(`Gate ${gate} ${action}: "${to}" · corpus now ${corpus.size} entries`, "ok");
    renderGates();
    refreshCorpus();
  } catch (e) {
    toast("Gate failed: " + e.message, "bad");
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 6 — Audit matrices
// ══════════════════════════════════════════════════════════════════════════════
function renderAudit() {
  const stages = ["ingestion", "detection", "characterization", "classification", "demodulation", "decoding"];
  const gates = ["G1", "G2", "G3"];

  const statusOf = (g) => {
    const actions = state.gates[g] || [];
    if (!actions.length) return { t: "auto", cls: "color:var(--muted)" };
    const last = actions[actions.length - 1];
    if (last.action === "override") return { t: `override → ${last.to}`, cls: "color:var(--amber)" };
    return { t: "confirmed", cls: "color:var(--green)" };
  };

  const head = `<tr><th>Stage</th>${gates.map(g => `<th>${g}</th>`).join("")}<th>Machine / Human</th></tr>`;
  const body = stages.map(s => {
    const cells = gates.map(g => {
      const st = statusOf(g);
      return `<td style="font-family:var(--mono);font-size:11px;${st.cls}">${esc(st.t)}</td>`;
    }).join("");
    const human = (state.gates.G1.length + state.gates.G2.length + state.gates.G3.length) > 0;
    return `<tr><td>${esc(s)}</td>${cells}<td>${human ? "<b>human arbitrated</b>" : "machine"}</td></tr>`;
  }).join("");

  const advisor = state.consult ? "consulted" : "not consulted";
  const advN = state.consult ? ` · ${state.consult.advisory && state.consult.advisory.neighbours} neighbours` : "";

  $("#auditMatrix").innerHTML = `
    <table><thead>${head}</thead><tbody>${body}</tbody></table>
    <p class="corpus-stat" style="margin-top:8px">Advisor: <b>${advisor}</b>${advN} · run audit answers "was that the machine or the human?"</p>`;
}

async function refreshCorpus() {
  try {
    const c = await apiGet("/api/hitl/corpus");
    state.corpus = c;
    renderCorpus(c);
  } catch (e) { /* ignore — corpus endpoint may not be seeded yet */ }
}

function renderCorpus(c) {
  const mix = c.source_mix || {};
  const rows = (c.per_chain || []).map(r => {
    const srcCell = Object.entries(r.sources || {})
      .map(([k, v]) => v ? `${esc(k)}: ${v}` : "").filter(Boolean).join(" · ") || "—";
    return `<tr>
      <td><code style="font-size:11px">${esc(r.chain)}</code></td>
      <td style="text-align:center">${r.count}</td>
      <td>${cellHTML(r.crc_pass_rate)}</td>
      <td style="font-size:11px;color:var(--muted)">${esc(srcCell)}</td>
    </tr>`;
  }).join("");

  const mixLine = Object.entries(mix).map(([k, v]) => `${esc(k)}: ${v}`).join(" · ") || "empty";

  $("#crcMatrix").innerHTML = `
    <table>
      <thead><tr><th>Chain</th><th>Runs</th><th>CRC pass rate</th><th>Truth source</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="4" style="color:var(--muted)">Empty corpus — generate or confirm entries at gates</td></tr>`}</tbody>
    </table>`;

  $("#corpusStat").textContent =
    `Corpus: ${c.size} verified entries · source mix: ${mixLine} · human_confirmed weighted 0.35 vs 1.0 for verified ground truth`;
}

// ── Step badge helper ─────────────────────────────────────────────────────────
function markStep(stepId, done) {
  const el = $(`#${stepId}`);
  if (!el) return;
  el.classList.toggle("done", done);
}

// ══════════════════════════════════════════════════════════════════════════════
// WIRING — event listeners
// ══════════════════════════════════════════════════════════════════════════════
function wire() {
  // Nav
  $("#nav").addEventListener("click", e => {
    const b = e.target.closest(".pill");
    if (b && b.dataset.sec) nav(b.dataset.sec);
  });

  // Drag-and-drop on dropzone
  const dropzone = $("#dropzone");
  dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("drag-over"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) { state.file = f; state.runId = makeRunId(); $("#runIdLabel").textContent = state.runId; updateDropLabel(f); }
  });

  // File input
  $("#fileInput").addEventListener("change", e => {
    const f = e.target.files[0];
    if (!f) return;
    state.file = f;
    state.runId = makeRunId();
    $("#runIdLabel").textContent = state.runId;
    updateDropLabel(f);
  });

  function updateDropLabel(f) {
    const icon = dropzone.querySelector(".drop-icon");
    const label = dropzone.querySelector(".drop-label");
    if (icon) icon.textContent = "📁";
    if (label) label.textContent = f.name + ` (${(f.size / 1024).toFixed(1)} KB)`;
  }

  $("#dataType").addEventListener("change", e => (state.dataType = e.target.value));
  $("#sampleRate").addEventListener("change", e => (state.sampleRate = e.target.value.trim()));

  // Upload
  $("#btnUpload").addEventListener("click", async () => {
    if (!state.file) return toast("Choose a file first", "warn");
    setLoading(true, "Parsing signal file…");
    try {
      renderSignalInfo(await apiUpload("/api/upload"));
      toast("Signal parsed ✓", "ok");
    } catch (e) {
      toast("Upload failed: " + e.message, "bad");
      setStatus("error");
    } finally { setLoading(false); }
  });

  // Analyze
  $("#btnAnalyze").addEventListener("click", async () => {
    if (!state.file) return toast("Upload a file first", "warn");
    setLoading(true, "Running detection & characterization…");
    try {
      renderAnalysis(await apiUpload("/api/analyze"));
      toast("Analysis complete ✓", "ok");
    } catch (e) {
      toast("Analysis failed: " + e.message, "bad");
      setStatus("error");
    } finally { setLoading(false); }
  });

  // Hypothesize
  $("#btnHypothesize").addEventListener("click", async () => {
    if (!state.file) return toast("Upload a file first", "warn");
    setLoading(true, "Running hypothesis engine — this may take 20–90 s…");
    toast("Hypothesis engine running…", "");
    try {
      const res = await apiUpload("/api/hypothesize");
      renderLedger(res);
      renderEvidence(res);
      toast(`${res.num_chains} chains scored ✓`, "ok");
    } catch (e) {
      toast("Hypothesize failed: " + e.message, "bad");
      setStatus("error");
    } finally { setLoading(false); }
  });

  // Consult advisor
  $("#btnConsult").addEventListener("click", async () => {
    if (!state.file) return toast("Upload a file first", "warn");
    setLoading(true, "Consulting benchmark corpus…");
    try {
      const res = await apiUpload(`/api/advisor/consult?run_id=${state.runId}`, {});
      renderAdvisory(res);
      toast("Advisory ready ✓", "ok");
      nav("advisorySec");
    } catch (e) {
      toast("Consult failed: " + e.message, "bad");
      setStatus("error");
    } finally { setLoading(false); }
  });

  // Corpus refresh
  $("#btnCorpus").addEventListener("click", async () => {
    setLoading(true, "Refreshing corpus…");
    await refreshCorpus();
    setLoading(false);
    nav("auditSec");
    toast("Corpus refreshed", "ok");
  });

  // Modal close
  $("#modalClose").addEventListener("click", () => ($("#modalOverlay").hidden = true));
  $("#modalOverlay").addEventListener("click", ev => {
    if (ev.target.id === "modalOverlay") $("#modalOverlay").hidden = true;
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") $("#modalOverlay").hidden = true;
  });

  // Resize canvases on window resize
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.waterfall) renderWaterfall(state.waterfall);
      if (state.analyze) renderPSD(state.analyze);
    }, 200);
  });

  // Initial corpus load & health check
  refreshCorpus();
  checkHealth();
  renderGates();
}

wire();
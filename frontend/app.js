const STORAGE_KEY = "ec3_chat_threads_v2";
const THINKING_MODE_KEY = "ec3_thinking_mode";

const GRAPH_NODES = [
  { id: "user",         label: "User",          icon: "person",  col: 0,   row: 1 },
  { id: "database",     label: "Database",      icon: "book",    col: 1,   row: 0 },
  { id: "tools",        label: "Tools",         icon: "wrench",  col: 2,   row: 0 },
  { id: "orchestrator", label: "Orchestrator",  icon: "brain",   col: 1.5, row: 1 },
  { id: "fea_analyst",  label: "FEA Analyst",   icon: "cube",    col: 1.5, row: 2 },
  { id: "response",     label: "Response",      icon: "check",   col: 3,   row: 1 },
];

const GRAPH_EDGES = [
  { id: "u_o",  from: "user",         to: "orchestrator" },
  { id: "o_d",  from: "orchestrator", to: "database"     },
  { id: "d_o",  from: "database",     to: "orchestrator" },
  { id: "o_t",  from: "orchestrator", to: "tools"        },
  { id: "t_o",  from: "tools",        to: "orchestrator" },
  { id: "o_fa", from: "orchestrator", to: "fea_analyst"  },
  { id: "fa_o", from: "fea_analyst",  to: "orchestrator" },
  { id: "o_r",  from: "orchestrator", to: "response"     },
];

// Tool classification for agent-mode flow graph bridge
const _DB_TOOLS = new Set(["eurocode_search", "read_clause"]);
const _CALC_TOOLS = new Set(["math_calculator", "section_lookup", "material_lookup"]);

const NODE_ICONS = {
  person: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
  brain: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3c-1.8-1.4-4.8-1.2-6.5.4-1.8 1.7-2 4.8-.5 6.8-.9 1.3-.9 3.2.1 4.5.8 1 2.1 1.6 3.4 1.6.7 2 2.6 3.4 4.5 3.4"/><path d="M12 3c1.8-1.4 4.8-1.2 6.5.4 1.8 1.7 2 4.8.5 6.8.9 1.3.9 3.2-.1 4.5-.8 1-2.1 1.6-3.4 1.6-.7 2-2.6 3.4-4.5 3.4"/><path d="M12 3v16"/><path d="M8.2 7.2c.8-.8 2-.9 2.8-.1"/><path d="M15.8 7.2c-.8-.8-2-.9-2.8-.1"/><path d="M7.6 11c1-.7 2.3-.7 3.2.1"/><path d="M16.4 11c-1-.7-2.3-.7-3.2.1"/><path d="M8.6 14.7c.9-.4 1.7-.3 2.4.3"/><path d="M15.4 14.7c-.9-.4-1.7-.3-2.4.3"/></svg>`,
  book: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>`,
  wrench: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg>`,
  citation: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>`,
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
  cube: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>`,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const form = $("#chat-form");
const input = $("#prompt-input");
const thinkingModeSelect = $("#thinking-mode");
const thinkingModeTrigger = $("#thinking-mode-trigger");
const thinkingModeMenu = $("#thinking-mode-menu");
const thinkingModeLabel = $("#thinking-mode-label");
const sendBtn = $("#send-btn");
const messagesEl = $("#messages");
const template = $("#message-template");
const welcome = $("#welcome");
const threadList = $("#thread-list");
const newChatBtn = $("#new-chat-btn");
const chatSearch = $("#chat-search");
const sidebarToggle = $("#sidebar-toggle");
const sidebar = $("#sidebar");
const signInBtn = $("#signin-btn");
const registerBtn = $("#register-btn");
const sidebarSigninBtn = $("#sidebar-signin-btn");
const sidebarSignupBtn = $("#sidebar-signup-btn");
const attachTrigger = $("#attach-trigger");
const attachMenu = $("#attach-menu");
const photoInput = $("#photo-input");
const fileInput = $("#file-input");
const attachmentsPreview = $("#attachments-preview");
const webSearchToggle = $("#web-search-toggle");
webSearchToggle?.addEventListener("click", () => {
  state.webSearchEnabled = !state.webSearchEnabled;
  webSearchToggle.classList.toggle("active", state.webSearchEnabled);
  webSearchToggle.title = state.webSearchEnabled ? "Web search on" : "Web search off";
});


const SEND_ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="6 11 12 5 18 11"></polyline></svg>';
const STOP_ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="6" y="6" width="12" height="12" rx="2"></rect></svg>';

const state = {
  threads: [],
  activeThreadId: null,
  guestThread: null,
  filter: "",
  thinkingMode: "thinking",
  attachments: [],
  abortController: null,
  webSearchEnabled: false,
};

function uid() {
  return crypto?.randomUUID?.() || `id_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
}
function now() { return new Date().toISOString(); }
function emptyThread(title = "New chat") { return { id: uid(), title, createdAt: now(), updatedAt: now(), messages: [] }; }

function fmtTime(iso) {
  return new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function truncTitle(v) {
  const c = (v || "").trim().replace(/\s+/g, " ");
  return !c ? "New chat" : c.length > 50 ? c.slice(0, 50) + "..." : c;
}

function clamp(v, max = 42) {
  const c = String(v || "").replace(/\s+/g, " ").trim();
  return c.length > max ? c.slice(0, max - 1) + "..." : c;
}

function renderMd(text) {
  if (typeof marked === "undefined") return `<pre>${escHtml(text || "")}</pre>`;
  let input = text || "";
  const mathBlocks = [];

  // Extract display math $$...$$ before Marked can mangle underscores/braces
  input = input.replace(/\$\$([\s\S]*?)\$\$/g, (_m, tex) => {
    const id = `\x00MATH${mathBlocks.length}\x00`;
    mathBlocks.push({ id, tex, display: true });
    return id;
  });
  // Extract inline math $...$
  input = input.replace(/\$([^\$\n]+?)\$/g, (_m, tex) => {
    const id = `\x00MATH${mathBlocks.length}\x00`;
    mathBlocks.push({ id, tex, display: false });
    return id;
  });

  let html;
  try { html = marked.parse(input); } catch { return `<pre>${escHtml(text || "")}</pre>`; }

  // Replace placeholders with KaTeX-rendered HTML
  for (const b of mathBlocks) {
    let rendered;
    if (typeof katex !== "undefined") {
      try {
        rendered = katex.renderToString(b.tex.trim(), {
          displayMode: b.display,
          throwOnError: false,
        });
      } catch { rendered = `<code>${escHtml(b.tex)}</code>`; }
    } else {
      rendered = `<code>${escHtml(b.tex)}</code>`;
    }
    html = html.split(b.id).join(rendered);
  }
  return html;
}

function renderLatex(tex, displayMode = false) {
  const source = String(tex || "").trim();
  if (!source) return "";
  if (typeof katex === "undefined") {
    return `<code>${escHtml(source)}</code>`;
  }
  try {
    return katex.renderToString(source, {
      displayMode,
      throwOnError: false,
    });
  } catch {
    return `<code>${escHtml(source)}</code>`;
  }
}

function renderInlineMd(text) {
  const html = renderMd(text || "");
  return html.replace(/^<p>([\s\S]*)<\/p>\s*$/i, "$1");
}

function renderCalcNoteHtml(note) {
  if (note && typeof note === "object" && typeof note.latex === "string" && note.latex.trim()) {
    return renderLatex(note.latex, true);
  }
  const text = typeof note === "string" ? note : String(note?.text ?? note ?? "");
  return escHtml(text);
}

const _ENGINEERING_GREEK_BASES = {
  "α": "\\alpha",
  "β": "\\beta",
  "γ": "\\gamma",
  "χ": "\\chi",
  "φ": "\\phi",
  "Φ": "\\Phi",
  "λ": "\\lambda",
  "ψ": "\\psi",
  "ε": "\\epsilon",
  "ρ": "\\rho",
  "μ": "\\mu",
  "η": "\\eta",
};

const _ENGINEERING_NAME_OVERRIDES = {
  "M_Ed": "M_{Ed}",
  "M_Rd": "M_{Rd}",
  "M_cr": "M_{cr}",
  "M_b,Rd": "M_{b,Rd}",
  "M_c,Rd": "M_{c,Rd}",
  "M_pl,Rd": "M_{pl,Rd}",
  "M_el,Rd": "M_{el,Rd}",
  "N_Ed": "N_{Ed}",
  "N_Rd": "N_{Rd}",
  "N_b,Rd": "N_{b,Rd}",
  "N_pl,Rd": "N_{pl,Rd}",
  "N_t,Rd": "N_{t,Rd}",
  "V_Ed": "V_{Ed}",
  "V_Rd": "V_{Rd}",
  "V_pl,Rd": "V_{pl,Rd}",
  "W_y": "W_y",
  "W_pl,y": "W_{pl,y}",
  "W_el,y": "W_{el,y}",
  "W_eff,y": "W_{eff,y}",
  "f_y": "f_y",
  "f_u": "f_u",
  "χ_LT": "\\chi_{LT}",
  "γ_M0": "\\gamma_{M0}",
  "γ_M1": "\\gamma_{M1}",
  "γ_M2": "\\gamma_{M2}",
  "Φ_LT": "\\Phi_{LT}",
  "φ_LT": "\\phi_{LT}",
  "λ_LT": "\\lambda_{LT}",
  "L_cr": "L_{cr}",
};

function engineeringBaseToLatex(base) {
  if (_ENGINEERING_GREEK_BASES[base]) return _ENGINEERING_GREEK_BASES[base];
  if (/^[A-Za-z]$/.test(base)) return base;
  return null;
}

function engineeringTokenToLatex(token) {
  const normalized = String(token || "").trim().replace(/\s+/g, "");
  if (!normalized) return null;
  if (_ENGINEERING_NAME_OVERRIDES[normalized]) return _ENGINEERING_NAME_OVERRIDES[normalized];
  const match = normalized.match(/^([A-Za-z\u0370-\u03FF]+)(?:_([A-Za-z0-9,]+))?$/u);
  if (!match) return null;
  const [, baseRaw, subRaw] = match;
  const base = engineeringBaseToLatex(baseRaw);
  if (!base) return null;
  if (!subRaw) return base;
  return `${base}_{${subRaw}}`;
}

function eurocodeExpressionToLatex(text) {
  let latex = String(text || "").trim();
  if (!latex) return "";
  latex = latex.replace(/([0-9]),([0-9])/g, "$1.$2");
  latex = latex
    .replace(/≤/g, " \\le ")
    .replace(/≥/g, " \\ge ")
    .replace(/×/g, " \\times ")
    .replace(/·/g, " \\cdot ")
    .replace(/[−–]/g, "-");
  latex = latex.replace(/([A-Za-z\u0370-\u03FF]+_[A-Za-z0-9,]+)/gu, (token) => {
    return engineeringTokenToLatex(token) || token;
  });
  latex = latex.replace(/\s+\((\d+\.\d+)\)\s*$/, String.raw`\qquad ($1)`);
  return latex.replace(/\s+/g, " ").trim();
}

function bestEffortReferenceMd(text) {
  let md = String(text || "");
  if (!md.trim()) return "";

  const placeholders = [];
  const stash = (markdown) => {
    const id = `\x00EUROCODE${placeholders.length}\x00`;
    placeholders.push({ id, markdown });
    return id;
  };

  md = md.replace(
    /(:\s*)([^:\n]{0,260}?(?:[A-Za-z\u0370-\u03FF]+_[A-Za-z0-9,]+)[^:\n]{0,260}?(?:≤|≥|=|<|>)[^:\n]{0,180}?\(\d+\.\d+\))(?=\s+(?:where|NOTE|\([0-9]+\)|$))/gu,
    (_match, prefix, expr) => `${prefix}\n\n${stash(`$$${eurocodeExpressionToLatex(expr)}$$`)}\n\n`,
  );

  md = md.replace(
    /([A-Za-z\u0370-\u03FF]+_[A-Za-z0-9,]+)\s*=\s*([A-Za-z\u0370-\u03FF]+_[A-Za-z0-9,]+)/gu,
    (match, lhs, rhs) => {
      const left = engineeringTokenToLatex(lhs);
      const right = engineeringTokenToLatex(rhs);
      if (!left || !right) return match;
      return stash(`$${left} = ${right}$`);
    },
  );

  md = md.replace(/([A-Za-z\u0370-\u03FF]+_[A-Za-z0-9,]+)/gu, (token) => {
    const latex = engineeringTokenToLatex(token);
    return latex ? stash(`$${latex}$`) : token;
  });

  for (const { id, markdown } of placeholders) {
    md = md.split(id).join(markdown);
  }
  return md;
}

function escHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function splitToolContextBlock(text) {
  const raw = typeof text === "string" ? text : String(text || "");
  const match = raw.match(/\s*(<tool-context>[\s\S]*?<\/tool-context>)\s*$/i);
  if (match) {
    return {
      visible: raw.slice(0, match.index).replace(/\s+$/, ""),
      toolContext: match[1].trim(),
    };
  }
  const rawMarker = raw.search(/(?:^|\n)\s*(?=(?:<tool-context>|\[tool_call\]|\[tool_result\]))/i);
  if (rawMarker >= 0) {
    return {
      visible: raw.slice(0, rawMarker).replace(/\s+$/, ""),
      toolContext: raw.slice(rawMarker).trim(),
    };
  }
  return { visible: raw, toolContext: "" };
}

function getSessionMemory(payload) {
  const memory = payload?.session_memory;
  return memory && typeof memory === "object" ? memory : null;
}

function getFEASessionMemory(payload) {
  const memory = getSessionMemory(payload);
  if (memory?.fea_session && typeof memory.fea_session === "object") {
    return memory;
  }
  if (payload?.fea_session && typeof payload.fea_session === "object") {
    return { state: "final", fea_session: payload.fea_session };
  }
  return null;
}

function formatPendingAskUserText(askUser) {
  if (!askUser || typeof askUser !== "object" || !askUser.question) return "";
  let text = "I need more information to continue.";
  text += `\n\n${askUser.question}`;
  if (askUser.context) text += `\n\n${askUser.context}`;
  return text;
}

function getAssistantDisplayContent(message) {
  const payload = message?.responsePayload || message?.response_payload || null;
  const { visible } = splitToolContextBlock(message?.content || "");
  const memory = getSessionMemory(payload);
  if (memory?.state === "waiting_for_user") {
    return visible || formatPendingAskUserText(memory.ask_user);
  }
  if (typeof payload?.answer === "string" && payload.answer) {
    return splitToolContextBlock(payload.answer).visible;
  }
  return visible;
}

function serializeHistoryMessage(message) {
  return {
    role: message.role,
    content: message.content || "",
    response_payload: message.responsePayload || null,
  };
}

function serializeHistoryMessages(messages) {
  const items = Array.isArray(messages) ? messages : [];
  let lastAssistantIdx = -1;
  for (let i = items.length - 1; i >= 0; i -= 1) {
    if (items[i]?.role === "assistant") {
      lastAssistantIdx = i;
      break;
    }
  }
  return items.map((message, idx) => ({
    role: message.role,
    content: message.content || "",
    response_payload: idx === lastAssistantIdx ? (message.responsePayload || null) : null,
  }));
}

// ---- Calculation Tabs (below response, one per calc tool result) ----

const _CALC_TOOL_LABELS = {
  math_calculator: "Calculation",
  section_lookup: "Section Properties",
  material_lookup: "Material Properties",
};

// ---- Key / value formatting (mirrors backend response_formatter.py) ----

const _UNIT_SUFFIXES = [
  ["_kNm", " (kNm)"], ["_kN", " (kN)"], ["_MPa", " (MPa)"], ["_GPa", " (GPa)"],
  ["_mm3", " (mm³)"], ["_mm2", " (mm²)"], ["_mm", " (mm)"],
  ["_cm2", " (cm²)"], ["_cm3", " (cm³)"], ["_cm4", " (cm⁴)"],
  ["_m2", " (m²)"], ["_m", " (m)"],
  ["_deg", " (°)"], ["_rad", " (rad)"],
];
const _KEY_SUBSCRIPTS = {
  M_pl_Rd: "M<sub>pl,Rd</sub>", M_el_Rd: "M<sub>el,Rd</sub>",
  M_b_Rd: "M<sub>b,Rd</sub>", M_c_Rd: "M<sub>c,Rd</sub>",
  M_Rd: "M<sub>Rd</sub>", N_Rd: "N<sub>Rd</sub>", V_Rd: "V<sub>Rd</sub>",
  M_Ed: "M<sub>Ed</sub>", N_Ed: "N<sub>Ed</sub>", V_Ed: "V<sub>Ed</sub>",
  N_b_Rd: "N<sub>b,Rd</sub>", N_t_Rd: "N<sub>t,Rd</sub>",
  N_pl_Rd: "N<sub>pl,Rd</sub>", V_pl_Rd: "V<sub>pl,Rd</sub>",
  Fv_Rd: "F<sub>v,Rd</sub>", Fb_Rd: "F<sub>b,Rd</sub>", Fv: "F<sub>v</sub>",
  fy: "f<sub>y</sub>", fu: "f<sub>u</sub>", fub: "f<sub>ub</sub>",
  Wpl: "W<sub>pl</sub>", Wel: "W<sub>el</sub>", W_pl: "W<sub>pl</sub>", W_el: "W<sub>el</sub>",
  W_pl_y: "W<sub>pl,y</sub>", W_el_y: "W<sub>el,y</sub>",
  W_pl_z: "W<sub>pl,z</sub>", W_el_z: "W<sub>el,z</sub>",
  I_y: "I<sub>y</sub>", I_z: "I<sub>z</sub>", I_w: "I<sub>w</sub>", I_t: "I<sub>t</sub>",
  L_cr: "L<sub>cr</sub>", M_cr: "M<sub>cr</sub>", Mcr: "M<sub>cr</sub>",
  gamma_M0: "γ<sub>M0</sub>", gamma_M1: "γ<sub>M1</sub>", gamma_M2: "γ<sub>M2</sub>",
  alpha_v: "α<sub>v</sub>", alpha_LT: "α<sub>LT</sub>",
  chi_LT: "χ<sub>LT</sub>", chi: "χ", lambda_LT: "λ̄<sub>LT</sub>",
  Phi_LT: "Φ<sub>LT</sub>", A_net: "A<sub>net</sub>",
};
function prettyKey(key) {
  let unit = "", base = key;
  for (const [suf, label] of _UNIT_SUFFIXES) {
    if (base.endsWith(suf)) { unit = label; base = base.slice(0, -suf.length); break; }
  }
  const sorted = Object.entries(_KEY_SUBSCRIPTS).sort((a, b) => b[0].length - a[0].length);
  for (const [pat, repl] of sorted) {
    if (base === pat) return repl + unit;
    if (base.startsWith(pat + "_")) return repl + " " + base.slice(pat.length + 1).replace(/_/g, " ") + unit;
  }
  return base.replace(/_/g, " ") + unit;
}
function _guessUnit(key) {
  const k = key.toLowerCase();
  if (k.includes("knm")) return "kNm"; if (k.includes("kn")) return "kN";
  if (k.includes("mpa")) return "MPa"; if (k.includes("gpa")) return "GPa";
  if (k.includes("mm3")) return "mm³"; if (k.includes("mm2")) return "mm²";
  if (k.includes("_mm")) return "mm"; if (k.endsWith("_m")) return "m";
  if (k.includes("cm2")) return "cm²"; if (k.includes("cm3")) return "cm³"; if (k.includes("cm4")) return "cm⁴";
  if (k.includes("m2")) return "m²";
  if (k.includes("_deg")) return "°"; if (k.includes("_rad")) return "rad";
  return "";
}
function formatValue(key, val) {
  if (typeof val === "boolean") return val ? "PASS ✓" : "FAIL ✗";
  if (typeof val === "number") {
    const unit = _guessUnit(key);
    if (Number.isInteger(val) || (val === Math.round(val) && Math.abs(val) < 1e6))
      return `${Math.round(val)} ${unit}`.trim();
    return `${val.toFixed(2)} ${unit}`.trim();
  }
  return String(val);
}

function buildCalcTabs(msgNode) {
  const calcContainer = msgNode.querySelector(".calc-tabs");
  if (!calcContainer) return;
  calcContainer.innerHTML = "";
  const results = msgNode.__calcResults || [];
  if (!results.length) {
    calcContainer.classList.add("hidden");
    return;
  }

  calcContainer.classList.remove("hidden");

  // ── Group results: merge all math_calculator calls into ONE tab,
  //    keep section_lookup / material_lookup as separate tabs each ──
  const mathResults = results.filter(r => r.tool === "math_calculator");
  const otherResults = results.filter(r => r.tool !== "math_calculator");

  // Merged inputs & outputs across all math_calculator invocations
  const mergedInputs = {};
  const mergedOutputs = {};
  const mergedNotes = [];
  for (const r of mathResults) {
    const res = r.result;
    if (!res || typeof res !== "object") continue;
    // Collect variables (inputs)
    const vars = res.inputs_used?.variables || {};
    for (const [k, v] of Object.entries(vars)) {
      if (typeof v !== "object" || v === null) mergedInputs[k] = v;
    }
    // Collect outputs
    const outs = res.outputs || {};
    for (const [k, v] of Object.entries(outs)) {
      if (typeof v !== "object" || v === null) mergedOutputs[k] = v;
    }
    // Prefer structured LaTeX steps for math notes; fall back to plain text notes.
    const steps = Array.isArray(res.intermediate?.steps) ? res.intermediate.steps : [];
    const latexNotes = steps
      .filter(step => step && typeof step === "object" && typeof step.latex === "string" && step.latex.trim())
      .map(step => ({ latex: step.latex }));
    if (latexNotes.length) {
      mergedNotes.push(...latexNotes);
    } else {
      const notes = res.notes || [];
      for (const n of notes) {
        mergedNotes.push(typeof n === "string" ? n : String(n));
      }
    }
  }

  // Build groups to render: one merged "Calculation" + one per other tool
  const groups = [];
  if (Object.keys(mergedOutputs).length) {
    groups.push({ label: "Calculation", inputs: mergedInputs, outputs: mergedOutputs, notes: mergedNotes });
  }
  for (const r of otherResults) {
    const res = r.result;
    if (!res || typeof res !== "object") continue;
    const label = _CALC_TOOL_LABELS[r.tool] || r.tool.replace(/_/g, " ");
    const inputs = r.args || {};
    let outputs = {};
    const skip = new Set(["inputs_used", "intermediate", "notes", "clause_references", "error"]);
    for (const [k, v] of Object.entries(res)) {
      if (!skip.has(k) && (typeof v === "number" || typeof v === "string" || typeof v === "boolean"))
        outputs[k] = v;
    }
    if (!Object.keys(outputs).length) continue;
    groups.push({
      label,
      inputs,
      outputs,
      notes: (res.notes || []).map(note => (typeof note === "string" ? note : String(note))),
    });
  }

  // Render each group as a <details class="tool-result">
  for (const g of groups) {
    const details = document.createElement("details");
    details.className = "tool-result";

    const summary = document.createElement("summary");
    summary.innerHTML = `<strong>${escHtml(g.label)}</strong> detailed results`;
    details.appendChild(summary);

    // ── Inputs table ──
    const inputEntries = Object.entries(g.inputs).filter(([, v]) => typeof v !== "object" || v === null);
    if (inputEntries.length) {
      const tbl = document.createElement("table");
      tbl.className = "tool-io-table";
      tbl.innerHTML = `<thead><tr><th scope="col">Parameter</th><th scope="col">Value</th></tr></thead>`;
      const tbody = document.createElement("tbody");
      for (const [k, v] of inputEntries) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${prettyKey(k)}</td><td>${escHtml(formatValue(k, v))}</td>`;
        tbody.appendChild(tr);
      }
      tbl.appendChild(tbody);
      details.appendChild(tbl);
    }

    // ── Outputs table ──
    const outputEntries = Object.entries(g.outputs).filter(([, v]) => typeof v !== "object" || v === null);
    if (outputEntries.length) {
      const tbl = document.createElement("table");
      tbl.className = "tool-io-table tool-output";
      tbl.innerHTML = `<thead><tr><th scope="col">Output</th><th scope="col">Value</th></tr></thead>`;
      const tbody = document.createElement("tbody");
      for (const [k, v] of outputEntries) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${prettyKey(k)}</td><td><strong>${escHtml(formatValue(k, v))}</strong></td>`;
        tbody.appendChild(tr);
      }
      tbl.appendChild(tbody);
      details.appendChild(tbl);
    }

    // ── Notes ──
    if (g.notes.length) {
      const notesDiv = document.createElement("div");
      notesDiv.className = "tool-notes";
      const titleEl = document.createElement("strong");
      titleEl.textContent = "Notes";
      const notesList = document.createElement("ul");
      notesList.className = "tool-note-list";
      for (const note of g.notes) {
        const li = document.createElement("li");
        li.className = "tool-note-item";
        if (note && typeof note === "object" && typeof note.latex === "string" && note.latex.trim()) {
          li.classList.add("tool-note-equation");
        }
        li.innerHTML = renderCalcNoteHtml(note);
        notesList.appendChild(li);
      }
      notesDiv.appendChild(titleEl);
      notesDiv.appendChild(notesList);
      details.appendChild(notesDiv);
    }

    calcContainer.appendChild(details);
  }
}

// ---- Expandable References (Eurocode clauses used) ----

function buildReferences(msgNode) {
  const refSection = msgNode.querySelector(".ref-section");
  if (!refSection) return;
  refSection.innerHTML = "";
  const allClauses = msgNode.__refClauses || [];
  if (!allClauses.length) {
    refSection.classList.add("hidden");
    return;
  }

  // Only show clauses that were selected (green) in the search matrix
  const selectedKeys = msgNode.__selectedRefKeys;
  const clauses = selectedKeys && selectedKeys.size > 0
    ? allClauses.filter(c => selectedKeys.has(`${c.standard}:${c.clause_id}`))
    : allClauses; // fallback: show all if no matrix exists
  const dedupedClauses = [];
  const seenClauseKeys = new Set();
  for (const clause of clauses) {
    const clauseKey = `${clause.standard || ""}:${clause.clause_id || ""}`;
    if (!clauseKey || seenClauseKeys.has(clauseKey)) continue;
    seenClauseKeys.add(clauseKey);
    dedupedClauses.push(clause);
  }
  if (!dedupedClauses.length) {
    refSection.classList.add("hidden");
    return;
  }

  refSection.classList.remove("hidden");

  // Header
  const header = document.createElement("div");
  header.className = "ref-header";
  header.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg><span>References</span><span class="ref-count">${dedupedClauses.length}</span>`;
  refSection.appendChild(header);

  // Each clause as an expandable card
  for (const c of dedupedClauses) {
    const details = document.createElement("details");
    details.className = "ref-clause";

    const label = c.clause_id.startsWith("Table")
      ? `${c.standard} ${c.clause_id}`
      : `${c.standard} Cl. ${c.clause_id}`;

    const summary = document.createElement("summary");
    summary.innerHTML = `<span class="ref-clause-id">${escHtml(label)}</span>${c.title ? `<span class="ref-clause-title">${escHtml(c.title)}</span>` : ""}`;
    details.appendChild(summary);

    if (c.text) {
      const textDiv = document.createElement("div");
      textDiv.className = "clause-text";
      // Split text into paragraphs
      const paragraphs = c.text.split(/\n\n+/);
      for (const p of paragraphs) {
        if (!p.trim()) continue;
        const pEl = document.createElement("div");
        pEl.className = "clause-p";
        pEl.innerHTML = renderInlineMd(bestEffortReferenceMd(p.trim()));
        textDiv.appendChild(pEl);
      }
      details.appendChild(textDiv);
    }

    refSection.appendChild(details);
  }
}

// ---- Attachments ----
function fmtFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function isImageFile(file) {
  return file.type.startsWith("image/");
}

function closeAttachMenu() {
  if (!attachMenu) return;
  attachMenu.classList.add("hidden");
  attachTrigger?.setAttribute("aria-expanded", "false");
  attachTrigger?.closest(".attach-wrap")?.classList.remove("open");
}

async function addAttachments(files) {
  for (const file of files) {
    if (state.attachments.length >= 10) break;
    const att = {
      id: uid(),
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      isImage: isImageFile(file),
      dataUrl: null,
    };
    if (att.isImage) {
      try { att.dataUrl = await readFileAsDataUrl(file); } catch { /* skip preview */ }
    }
    state.attachments.push(att);
  }
  renderAttachmentsPreview();
}

function removeAttachment(id) {
  state.attachments = state.attachments.filter(a => a.id !== id);
  renderAttachmentsPreview();
}

function clearAttachments() {
  state.attachments = [];
  renderAttachmentsPreview();
}

function renderAttachmentsPreview() {
  if (!attachmentsPreview) return;
  attachmentsPreview.innerHTML = "";
  if (!state.attachments.length) {
    attachmentsPreview.classList.add("hidden");
    return;
  }
  attachmentsPreview.classList.remove("hidden");
  for (const att of state.attachments) {
    const chip = document.createElement("div");
    chip.className = att.isImage ? "attachment-chip photo-chip" : "attachment-chip";

    if (att.isImage && att.dataUrl) {
      const img = document.createElement("img");
      img.src = att.dataUrl;
      img.alt = att.name;
      chip.appendChild(img);
    } else {
      const icon = document.createElement("span");
      icon.className = "att-icon";
      icon.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`;
      const name = document.createElement("span");
      name.className = "att-name";
      name.textContent = att.name;
      name.title = att.name;
      const size = document.createElement("span");
      size.className = "att-size";
      size.textContent = fmtFileSize(att.size);
      chip.appendChild(icon);
      chip.appendChild(name);
      chip.appendChild(size);
    }

    const removeBtn = document.createElement("button");
    removeBtn.className = "att-remove";
    removeBtn.type = "button";
    removeBtn.innerHTML = "&times;";
    removeBtn.title = "Remove";
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      removeAttachment(att.id);
    });
    chip.appendChild(removeBtn);

    attachmentsPreview.appendChild(chip);
  }
}

function buildAttachmentHtml(attachments) {
  if (!attachments || !attachments.length) return "";
  let html = '<div class="msg-attachments">';
  for (const att of attachments) {
    if (att.isImage && att.dataUrl) {
      html += `<div class="msg-photo-thumb" data-src="${escHtml(att.dataUrl)}"><img src="${escHtml(att.dataUrl)}" alt="${escHtml(att.name)}" /></div>`;
    } else {
      html += `<div class="msg-file-tag"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><span class="file-tag-name" title="${escHtml(att.name)}">${escHtml(att.name)}</span><span class="file-tag-size">${fmtFileSize(att.size)}</span></div>`;
    }
  }
  html += "</div>";
  return html;
}

function attachLightboxListeners(container) {
  const thumbs = container.querySelectorAll(".msg-photo-thumb");
  for (const thumb of thumbs) {
    thumb.addEventListener("click", () => {
      const src = thumb.dataset.src || thumb.querySelector("img")?.src;
      if (!src) return;
      const overlay = document.createElement("div");
      overlay.className = "photo-lightbox";
      const img = document.createElement("img");
      img.src = src;
      overlay.appendChild(img);
      overlay.addEventListener("click", () => overlay.remove());
      document.body.appendChild(overlay);
    });
  }
}

function isValidThinkingMode(mode) {
  return mode === "standard" || mode === "thinking" || mode === "extended";
}

function thinkingModeLabelText(mode) {
  if (mode === "standard") return "Standard";
  if (mode === "extended") return "Extended Thinking";
  return "Thinking";
}

function closeThinkingModeMenu() {
  if (!thinkingModeMenu) return;
  thinkingModeMenu.classList.add("hidden");
  thinkingModeTrigger?.setAttribute("aria-expanded", "false");
  thinkingModeTrigger?.closest(".thinking-mode-wrap")?.classList.remove("open");
}

function syncThinkingModeUi() {
  if (thinkingModeSelect && thinkingModeSelect.value !== state.thinkingMode) {
    thinkingModeSelect.value = state.thinkingMode;
  }
  if (thinkingModeLabel) {
    thinkingModeLabel.textContent = thinkingModeLabelText(state.thinkingMode);
  }
  for (const option of $$(".thinking-mode-option")) {
    const selected = option.dataset.mode === state.thinkingMode;
    option.classList.toggle("selected", selected);
    option.setAttribute("aria-selected", selected ? "true" : "false");
  }
}

function setThinkingModeDisabled(disabled) {
  if (thinkingModeSelect) thinkingModeSelect.disabled = disabled;
  if (thinkingModeTrigger) thinkingModeTrigger.disabled = disabled;
  for (const option of $$(".thinking-mode-option")) {
    option.disabled = disabled;
  }
  if (disabled) closeThinkingModeMenu();
}

function loadThinkingModePreference() {
  const saved = localStorage.getItem(THINKING_MODE_KEY);
  if (isValidThinkingMode(saved)) {
    state.thinkingMode = saved;
  }
  syncThinkingModeUi();
}

function setThinkingMode(mode) {
  state.thinkingMode = isValidThinkingMode(mode) ? mode : "thinking";
  localStorage.setItem(THINKING_MODE_KEY, state.thinkingMode);
  syncThinkingModeUi();
}

// ---- State persistence ----
function canUseStoredThreads() {
  if (!auth.ready) return false;
  return !!auth.user;
}

function storageKey() {
  if (!auth.user) return STORAGE_KEY;
  const identity = auth.user.user_id || auth.user.email || "user";
  return `${STORAGE_KEY}:${identity}`;
}

function resetThreadState() {
  state.threads = [];
  state.activeThreadId = null;
}

function resetGuestThread() {
  state.guestThread = emptyThread("Temporary chat");
}

function save() {
  if (!canUseStoredThreads()) return;
  if (auth.threadsSync) return;
  localStorage.setItem(storageKey(), JSON.stringify({ threads: state.threads, activeThreadId: state.activeThreadId }));
}

async function load() {
  resetThreadState();
  if (!canUseStoredThreads()) return;
  if (auth.threadsSync) {
    try {
      const res = await fetchWithAuth("/api/threads");
      if (!res.ok) return;
      const data = await res.json();
      const threads = (data.threads || []).map((t) => ({
        id: t.id,
        title: t.title || "New chat",
        createdAt: t.createdAt,
        updatedAt: t.updatedAt,
        messages: [],
      }));
      state.threads = threads;
      if (threads.length && !state.activeThreadId) {
        state.activeThreadId = threads[0].id;
      }
      if (state.activeThreadId) {
        const full = await loadThreadFromApi(state.activeThreadId);
        if (full) {
          const idx = state.threads.findIndex((t) => t.id === full.id);
          if (idx >= 0) state.threads[idx] = full;
        }
      }
    } catch {
      state.threads = [];
    }
    return;
  }
  try {
    const key = storageKey();
    const current = localStorage.getItem(key);
    const fallback = key !== STORAGE_KEY ? localStorage.getItem(STORAGE_KEY) : null;
    const raw = current || fallback;
    if (!raw) return;
    const p = JSON.parse(raw);
    if (p?.threads) { state.threads = p.threads; state.activeThreadId = p.activeThreadId || null; }
    if (!current && fallback && key !== STORAGE_KEY) {
      localStorage.setItem(key, fallback);
    }
  } catch { state.threads = []; state.activeThreadId = null; }
}

async function addMessageToApi(threadId, role, content, responsePayload = null) {
  const res = await fetchWithAuth(`/api/threads/${threadId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, content, response_payload: responsePayload }),
  });
  return res.ok;
}

async function truncateThreadApi(threadId, keepCount, updatedContent = null) {
  const res = await fetchWithAuth(`/api/threads/${threadId}/truncate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ keep_count: keepCount, updated_content: updatedContent }),
  });
  return res.ok;
}

async function updateThreadTitleApi(threadId, title) {
  const res = await fetchWithAuth(`/api/threads/${threadId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return res.ok;
}

async function loadThreadFromApi(threadId) {
  const res = await fetchWithAuth(`/api/threads/${threadId}`);
  if (!res.ok) return null;
  const t = await res.json();
  return {
    id: t.id,
    title: t.title || "New chat",
    createdAt: t.createdAt,
    updatedAt: t.updatedAt,
    messages: (t.messages || []).map((m) => ({
      id: m.id,
      role: m.role,
      content: m.content || "",
      responsePayload: m.responsePayload,
      createdAt: m.createdAt,
    })),
  };
}

async function createThread(title = "New chat") {
  if (!canUseStoredThreads()) {
    state.guestThread = emptyThread("Temporary chat");
    return state.guestThread;
  }
  if (auth.threadsSync) {
    try {
      const res = await fetchWithAuth("/api/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title || "New chat" }),
      });
      if (!res.ok) throw new Error("Failed to create thread");
      const t = await res.json();
      const thread = {
        id: t.id,
        title: t.title || "New chat",
        createdAt: t.createdAt,
        updatedAt: t.updatedAt,
        messages: [],
      };
      state.threads.unshift(thread);
      state.activeThreadId = thread.id;
      return thread;
    } catch {
      const t = emptyThread(title);
      state.threads.unshift(t);
      state.activeThreadId = t.id;
      save();
      return t;
    }
  }
  const t = emptyThread(title);
  state.threads.unshift(t);
  state.activeThreadId = t.id;
  save();
  return t;
}

function activeThread() { return state.threads.find((t) => t.id === state.activeThreadId) || null; }
function currentThread() { return canUseStoredThreads() ? activeThread() : state.guestThread; }
async function ensureThread() { return currentThread() || await createThread(canUseStoredThreads() ? "New chat" : "Temporary chat"); }

async function setActive(id) {
  if (!canUseStoredThreads()) return;
  if (!state.threads.some((t) => t.id === id)) return;
  state.activeThreadId = id;
  if (auth.threadsSync) {
    const full = await loadThreadFromApi(id);
    if (full) {
      const idx = state.threads.findIndex((t) => t.id === id);
      if (idx >= 0) state.threads[idx] = full;
    }
  } else {
    save();
  }
  renderThreadList();
  renderMessages();
}

function updateWelcome() {
  const a = currentThread();
  const hasMessages = !!(a && a.messages.length);
  welcome.classList.toggle("hidden", hasMessages);
  document.body.classList.toggle("chat-started", hasMessages);
  document.body.classList.toggle("pre-chat", !hasMessages);
}

// ---- Thread list ----
function renderThreadList() {
  const chatsSection = $("#sidebar-chats-section");
  const ctaSection = $("#sidebar-chats-cta");
  const authDisabledSection = $("#sidebar-auth-disabled");

  if (!auth.ready) {
    threadList.innerHTML = '<li class="empty-threads">Loading...</li>';
    if (chatsSection) chatsSection.classList.remove("hidden");
    if (ctaSection) ctaSection.classList.add("hidden");
    if (authDisabledSection) authDisabledSection.classList.add("hidden");
    return;
  }

  if (!auth.enabled) {
    if (chatsSection) chatsSection.classList.add("hidden");
    if (ctaSection) ctaSection.classList.add("hidden");
    if (authDisabledSection) authDisabledSection.classList.remove("hidden");
    threadList.innerHTML = "";
    return;
  }

  if (!canUseStoredThreads()) {
    if (chatsSection) chatsSection.classList.add("hidden");
    if (ctaSection) ctaSection.classList.remove("hidden");
    if (authDisabledSection) authDisabledSection.classList.add("hidden");
    threadList.innerHTML = "";
    return;
  }

  if (chatsSection) chatsSection.classList.remove("hidden");
  if (ctaSection) ctaSection.classList.add("hidden");
  if (authDisabledSection) authDisabledSection.classList.add("hidden");

  threadList.innerHTML = "";
  const f = state.filter.trim().toLowerCase();
  const vis = state.threads.filter(t => !f || t.title.toLowerCase().includes(f));
  if (!vis.length) {
    threadList.innerHTML = '<li class="empty-threads">No chats yet</li>';
    return;
  }
  for (const t of vis) {
    const li = document.createElement("li");
    li.className = "thread-item" + (t.id === state.activeThreadId ? " active" : "");
    li.innerHTML = `<span class="thread-title">${escHtml(t.title || "New chat")}</span><span class="thread-meta">${fmtTime(t.updatedAt || t.createdAt)}</span>`;
    li.addEventListener("click", () => setActive(t.id));
    threadList.appendChild(li);
  }
}

// ---- Flow graph ----
const GRID = {
  colW: 82,
  rowH: 56,
  padX: 8,
  padY: 20,
  nodeW: 60,
  nodeH: 34,
  popupGap: 5,
  popupMaxRows: 2,
  popupMaxItems: 20,
  popupRowH: 14,
  popupRowGap: 3,
  edgePadY: 5,
};

function popupLaneReserve() {
  return GRID.edgePadY
    + GRID.popupGap
    + (GRID.popupMaxRows * GRID.popupRowH)
    + ((GRID.popupMaxRows - 1) * GRID.popupRowGap);
}

function getGraphLayout() {
  const minCol = Math.min(...GRAPH_NODES.map(n => n.col));
  const maxCol = Math.max(...GRAPH_NODES.map(n => n.col));
  const minRow = Math.min(...GRAPH_NODES.map(n => n.row));
  const maxRow = Math.max(...GRAPH_NODES.map(n => n.row));
  const popupReserveY = popupLaneReserve();

  const originX = GRID.padX - minCol * GRID.colW;
  const originY = GRID.padY + popupReserveY - minRow * GRID.rowH;
  const spanCols = maxCol - minCol;
  const spanRows = maxRow - minRow;

  return {
    originX,
    originY,
    totalW: originX + spanCols * GRID.colW + GRID.nodeW + GRID.padX,
    totalH: originY + spanRows * GRID.rowH + GRID.nodeH + GRID.padY,
  };
}

function getNodePos(node, layout = { originX: GRID.padX, originY: GRID.padY }) {
  return {
    x: layout.originX + node.col * GRID.colW,
    y: layout.originY + node.row * GRID.rowH,
    cx: layout.originX + node.col * GRID.colW + GRID.nodeW / 2,
    cy: layout.originY + node.row * GRID.rowH + GRID.nodeH / 2,
  };
}

function edgePath(fromNode, toNode, layout) {
  const f = getNodePos(fromNode, layout);
  const t = getNodePos(toNode, layout);

  if (fromNode.row !== toNode.row) {
    // Different rows: connect through top/bottom (horizontal) sides
    const downward = fromNode.row < toNode.row;
    const startX = f.cx;
    const endX = t.cx;
    const startY = downward ? f.y + GRID.nodeH : f.y;
    const endY = downward ? t.y : t.y + GRID.nodeH;
    if (fromNode.col === toNode.col) {
      return `M ${startX} ${startY} L ${endX} ${endY}`;
    }
    const dy = endY - startY;
    return `M ${startX} ${startY} C ${startX} ${startY + dy * 0.5} ${endX} ${endY - dy * 0.5} ${endX} ${endY}`;
  }

  // Same row: connect through left/right (vertical) sides
  const rightward = fromNode.col < toNode.col;
  const startX = rightward ? f.x + GRID.nodeW : f.x;
  const endX = rightward ? t.x : t.x + GRID.nodeW;
  const startY = f.cy;
  const endY = t.cy;
  const dx = endX - startX;
  return `M ${startX} ${startY} C ${startX + dx * 0.5} ${startY} ${endX - dx * 0.5} ${endY} ${endX} ${endY}`;
}

function initFlowGraph(msgNode, prompt) {
  const diagramPanel = msgNode.querySelector(".diagram-panel");
  const thinkingPanel = msgNode.querySelector(".thinking-panel");
  const graph = msgNode.querySelector(".flow-graph");
  if (!graph) return;

  // Move diagram panel to body level so it floats as a popup
  if (diagramPanel && diagramPanel.parentNode === msgNode) {
    document.body.appendChild(diagramPanel);
  }
  if (diagramPanel) diagramPanel.classList.remove("hidden");
  if (thinkingPanel) {
    thinkingPanel.classList.remove("hidden");
    thinkingPanel.classList.add("collapsed");
  }

  // Wire up close button
  const closeBtn = diagramPanel?.querySelector(".diagram-close-btn");
  if (closeBtn && !closeBtn.__wired) {
    closeBtn.__wired = true;
    closeBtn.addEventListener("click", () => {
      diagramPanel.classList.add("hidden");
    });
  }

  // Wire up collapse button on thinking panel
  const collapseBtn = msgNode.querySelector(".panel-collapse-btn");
  if (collapseBtn && !collapseBtn.__wired) {
    collapseBtn.__wired = true;
    collapseBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      thinkingPanel.classList.toggle("collapsed");
    });
  }
  // Clicking the collapsed panel header also expands it
  if (thinkingPanel && !thinkingPanel.__clickWired) {
    thinkingPanel.__clickWired = true;
    thinkingPanel.addEventListener("click", (e) => {
      if (thinkingPanel.classList.contains("collapsed") && !e.target.closest(".panel-collapse-btn")) {
        thinkingPanel.classList.remove("collapsed");
      }
    });
  }

  // Add scroll detection for sticky header border
  if (thinkingPanel && !thinkingPanel.__scrollWired) {
    thinkingPanel.__scrollWired = true;
    thinkingPanel.addEventListener("scroll", () => {
      thinkingPanel.classList.toggle("scrolled", thinkingPanel.scrollTop > 2);
    });
  }

  setThinkingState(msgNode, true);
  graph.innerHTML = "";

  const canvas = document.createElement("div");
  canvas.className = "flow-canvas";

  const layout = getGraphLayout();
  const totalW = layout.totalW;
  const totalH = layout.totalH;
  canvas.style.width = totalW + "px";
  canvas.style.height = totalH + "px";

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("class", "flow-svg");
  svg.setAttribute("viewBox", `0 0 ${totalW} ${totalH}`);

  const nodeMap = Object.fromEntries(GRAPH_NODES.map(n => [n.id, n]));
  for (const e of GRAPH_EDGES) {
    const fromN = nodeMap[e.from], toN = nodeMap[e.to];
    const path = document.createElementNS(svgNS, "path");
    path.setAttribute("d", edgePath(fromN, toN, layout));
    path.setAttribute("class", "flow-edge idle");
    path.dataset.edge = e.id;

    const markerId = `arrow-${e.id}`;
    const defs = svg.querySelector("defs") || svg.insertBefore(document.createElementNS(svgNS, "defs"), svg.firstChild);
    const marker = document.createElementNS(svgNS, "marker");
    marker.setAttribute("id", markerId);
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "9");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "5");
    marker.setAttribute("markerHeight", "5");
    marker.setAttribute("orient", "auto-start-reverse");
    const arrow = document.createElementNS(svgNS, "path");
    arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    arrow.setAttribute("fill", "currentColor");
    arrow.setAttribute("class", "flow-arrow");
    marker.appendChild(arrow);
    defs.appendChild(marker);
    path.setAttribute("marker-end", `url(#${markerId})`);

    svg.appendChild(path);
  }
  canvas.appendChild(svg);

  const nodeEls = {};
  for (const n of GRAPH_NODES) {
    const pos = getNodePos(n, layout);
    const el = document.createElement("div");
    el.className = "flow-node idle";
    el.dataset.node = n.id;
    el.style.left = pos.x + "px";
    el.style.top = pos.y + "px";
    el.style.width = GRID.nodeW + "px";
    el.style.height = GRID.nodeH + "px";

    const icon = document.createElement("div");
    icon.className = "flow-node-icon";
    icon.innerHTML = NODE_ICONS[n.icon] || "";
    const title = document.createElement("div");
    title.className = "flow-node-title";
    title.textContent = n.label;
    el.appendChild(icon);
    el.appendChild(title);

    canvas.appendChild(el);
    nodeEls[n.id] = el;
  }

  const docPos = getNodePos(nodeMap.database, layout);
  const toolPos = getNodePos(nodeMap.tools, layout);

  const docPopups = document.createElement("div");
  docPopups.className = "flow-popups above-left";
  docPopups.style.left = (docPos.x + GRID.nodeW) + "px";
  docPopups.style.top = (docPos.y - GRID.popupGap) + "px";
  canvas.appendChild(docPopups);

  const toolPopups = document.createElement("div");
  toolPopups.className = "flow-popups above-right";
  toolPopups.style.left = toolPos.x + "px";
  toolPopups.style.top = (toolPos.y - GRID.popupGap) + "px";
  canvas.appendChild(toolPopups);

  graph.appendChild(canvas);

  const ns = {}, es = {};
  GRAPH_NODES.forEach(n => { ns[n.id] = "idle"; });
  GRAPH_EDGES.forEach(e => { es[e.id] = "idle"; });

  ns["user"] = "done";
  ns["orchestrator"] = "active";
  es["u_o"] = "active";

  msgNode.__flow = {
    ns,
    es,
    nodeEls,
    popupLanes: { docs: docPopups, tools: toolPopups },
    popupRefs: { docs: new Map(), tools: new Map() },
    diagramPanel,
  };
  msgNode.__thinkStart = Date.now();
  msgNode.__stepCount = 0;
  applyFlow(msgNode);
  initSearchMatrix(msgNode);
}

function setNS(f, id, s) {
  const c = f.ns[id] || "idle";
  if (c === "error" && s !== "error") return;
  f.ns[id] = s;
}
function setES(f, id, s) {
  const c = f.es[id] || "idle";
  if (c === "error" && s !== "error") return;
  f.es[id] = s;
}

function triggerPopupPulse(chip) {
  if (!chip) return;
  chip.classList.remove("pulse");
  void chip.offsetWidth;
  chip.classList.add("pulse");
}

function addPopupChip(f, lane, key, label) {
  if (!lane || !key || !label) return;
  const laneEl = f.popupLanes?.[lane];
  const refs = f.popupRefs?.[lane];
  if (!laneEl || !refs) return;

  const existing = refs.get(key);
  if (existing) {
    triggerPopupPulse(existing);
    return;
  }

  const chip = document.createElement("div");
  chip.className = "flow-popup-chip";
  chip.textContent = clamp(label, 28);
  laneEl.appendChild(chip);
  refs.set(key, chip);
  requestAnimationFrame(() => chip.classList.add("show"));
  triggerPopupPulse(chip);

  while (laneEl.children.length > GRID.popupMaxItems) {
    const first = laneEl.firstElementChild;
    if (!first) break;
    laneEl.removeChild(first);
    for (const [k, el] of refs.entries()) {
      if (el === first) refs.delete(k);
    }
  }
}

function formatDocBadge(entry) {
  if (!entry || typeof entry !== "object") return "";
  const rawDoc = String(entry.doc_id || "").trim();
  // Shorten: "ec3.en1993-1-1.2005.json" → "EN1993-1-1"
  let short = rawDoc.replace(/\.json$/i, "").replace(/^ec3\./i, "");
  short = short.replace(/\.\d{4}$/, "").replace(/^en/i, "EN");
  if (!short) short = "doc";
  const clause = String(entry.clause_id || "").trim();
  if (!clause) return short;
  const prefix = /^\d/.test(clause) ? "Cl." : "";
  return `${short} ${prefix}${clause}`;
}

function pushDocBadges(f, entries) {
  if (!Array.isArray(entries)) return;
  for (const entry of entries) {
    const label = formatDocBadge(entry);
    if (!label) continue;
    const key = `${entry.doc_id || "unknown"}:${entry.clause_id || "?"}`;
    addPopupChip(f, "docs", key, label);
  }
}

function pushToolBadge(f, tool) {
  const raw = normTool(tool);
  if (!raw) return;
  addPopupChip(f, "tools", raw.toLowerCase(), raw);
}

function applyFlow(n) {
  const f = n.__flow;
  if (!f) return;
  for (const nd of GRAPH_NODES) {
    const el = f.nodeEls[nd.id], s = f.ns[nd.id] || "idle";
    el.classList.remove("idle", "active", "done", "error");
    el.classList.add(s);
  }
  // flow-edge elements are inside diagramPanel which may be at body level
  const edgeContainer = f.diagramPanel || n;
  const arrowColors = { idle: "#3d4149", done: "var(--accent)", active: "var(--accent-2)", error: "var(--danger)" };
  edgeContainer.querySelectorAll(".flow-edge").forEach(e => {
    const s = f.es[e.dataset.edge] || "idle";
    e.classList.remove("idle", "active", "done", "error");
    e.classList.add(s);
    const marker = edgeContainer.querySelector(`#arrow-${e.dataset.edge} .flow-arrow`);
    if (marker) marker.setAttribute("fill", arrowColors[s] || arrowColors.idle);
  });
}

function clearPopupLane(f, lane) {
  const laneEl = f.popupLanes?.[lane];
  const refs = f.popupRefs?.[lane];
  if (!laneEl) return;
  laneEl.innerHTML = "";
  if (refs) refs.clear();
}

function normTool(n) { return String(n || "").replace(/_ec3/g, "").replace(/_/g, " ").trim(); }

// ---- Search Matrix (Database Tunnel Visualization) ----

const _SM_SEARCH_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>`;

function initSearchMatrix(msgNode) {
  const panel = msgNode.__flow?.diagramPanel || msgNode;
  const container = panel.querySelector(".search-matrix");
  if (!container) return;
  container.innerHTML = "";
  container.classList.add("hidden");

  const header = document.createElement("div");
  header.className = "search-matrix-header";
  header.innerHTML = `<span class="sm-icon">${_SM_SEARCH_ICON}</span><span>Database Tunnel</span><span class="sm-count"></span>`;
  container.appendChild(header);

  const body = document.createElement("div");
  body.className = "search-matrix-body";
  container.appendChild(body);

  const summary = document.createElement("div");
  summary.className = "sm-summary hidden";
  container.appendChild(summary);

  msgNode.__searchMatrix = {
    container,
    body,
    summary,
    countEl: header.querySelector(".sm-count"),
    columns: [],
    allClauses: new Map(), // clauseKey → {id, title, score, selected}
    totalHits: 0,
  };
}

function _smAddColumn(msgNode, phase, query) {
  const sm = msgNode.__searchMatrix;
  if (!sm) return -1;

  sm.container.classList.remove("hidden");

  // Deactivate previous column
  const prev = sm.body.querySelector(".sm-col.active");
  if (prev) {
    prev.classList.remove("active");
    prev.classList.add("done");
  }

  // Arrow between columns
  if (sm.columns.length > 0) {
    const arrow = document.createElement("div");
    arrow.className = "sm-arrow";
    arrow.textContent = "\u203A"; // single right-pointing angle quotation mark
    sm.body.appendChild(arrow);
    requestAnimationFrame(() => arrow.classList.add("show"));
  }

  const col = document.createElement("div");
  col.className = "sm-col active";
  col.innerHTML = `<div class="sm-col-head"><span class="sm-col-phase">${escHtml(phase)}</span>${query ? `<span class="sm-col-query" title="${escHtml(query)}">${escHtml(clamp(query, 18))}</span>` : ""}</div><div class="sm-col-rows"></div>`;
  sm.body.appendChild(col);

  const idx = sm.columns.length;
  sm.columns.push({ el: col, phase, query, rows: [] });

  requestAnimationFrame(() => {
    col.classList.add("show");
    sm.body.scrollLeft = sm.body.scrollWidth;
  });

  return idx;
}

function _smPopulateColumn(msgNode, colIdx, clauses, isFetch) {
  const sm = msgNode.__searchMatrix;
  if (!sm || colIdx < 0 || colIdx >= sm.columns.length) return;

  const colData = sm.columns[colIdx];
  const rowsEl = colData.el.querySelector(".sm-col-rows");
  if (!rowsEl) return;

  const maxScore = Math.max(...clauses.map(c => c.score || 0), 1);

  clauses.forEach((c, i) => {
    const cid = c.clause_id || c.id || "?";
    const score = c.score || 0;
    const normScore = maxScore > 0 ? score / maxScore : 0;
    const tier = normScore >= 0.8 ? 1 : normScore >= 0.5 ? 2 : 3;
    // Use backend's selected flag; fetched clauses are always selected
    const isSelected = isFetch || !!c.selected;

    // Build label with Eurocode reference
    const stdShort = (c.standard || "").replace("EN 1993-", "EC3-").replace("EN ", "");
    const label = stdShort ? `${stdShort} ${cid}` : cid;

    const row = document.createElement("div");
    row.className = `sm-row tier-${tier}${isSelected ? " selected" : ""}`;
    row.innerHTML = `<span class="sm-row-id" title="${escHtml(c.title || cid)}">${escHtml(clamp(label, 22))}</span><span class="sm-row-bar" style="width:0;min-width:${Math.max(4, Math.round(normScore * 28))}px"></span>`;
    rowsEl.appendChild(row);
    colData.rows.push(row);

    // Track in global map
    const key = `${c.standard || ""}:${cid}`;
    if (!sm.allClauses.has(key) || score > (sm.allClauses.get(key).score || 0)) {
      sm.allClauses.set(key, { id: cid, title: c.title, score, selected: isSelected });
    }

    // Staggered reveal animation
    setTimeout(() => {
      row.classList.add("show");
      const bar = row.querySelector(".sm-row-bar");
      if (bar) bar.style.width = `${Math.max(4, Math.round(normScore * 28))}px`;
    }, 60 + i * 50);
  });

  sm.totalHits += clauses.length;
  if (sm.countEl) sm.countEl.textContent = `${sm.totalHits} hits`;
}

function _smFinalizeColumn(msgNode, colIdx) {
  const sm = msgNode.__searchMatrix;
  if (!sm || colIdx < 0 || colIdx >= sm.columns.length) return;
  const col = sm.columns[colIdx];
  col.el.classList.remove("active");
  col.el.classList.add("done");
}

function _smShowSummary(msgNode, selectedClauses) {
  const sm = msgNode.__searchMatrix;
  if (!sm || !selectedClauses.length) return;
  if (!sm.shownChipIds) sm.shownChipIds = new Set();
  sm.summary.classList.remove("hidden");

  const existingCount = sm.summary.children.length;
  let added = 0;
  selectedClauses.forEach((c) => {
    const id = c.clause_id || c.id || "?";
    const chipKey = `${c.standard || ""}:${id}`;
    if (sm.shownChipIds.has(chipKey)) return;
    sm.shownChipIds.add(chipKey);
    const stdShort = (c.standard || "").replace("EN 1993-", "EC3-").replace("EN ", "");
    const chip = document.createElement("span");
    chip.className = "sm-summary-chip";
    chip.textContent = stdShort ? `${stdShort} ${id}` : id;
    chip.title = c.title || "";
    sm.summary.appendChild(chip);
    setTimeout(() => chip.classList.add("show"), 80 + (existingCount + added) * 60);
    added++;
  });
}

/** Bridge: called from streaming handler when a search tool starts */
function onSearchToolStart(msgNode, toolName, args) {
  const isSearch = toolName === "eurocode_search";
  const isRead = toolName === "read_clause";
  if (!isSearch && !isRead) return;

  const phase = isSearch ? "SEARCH" : "FETCH";
  const query = isSearch ? (args?.query || "") : (args?.clause_id || "");
  const colIdx = _smAddColumn(msgNode, phase, query);
  // Stash the column index for later population
  msgNode.__smPendingCol = colIdx;
}

/** Bridge: called from streaming handler when a search tool finishes */
function onSearchToolResult(msgNode, toolName, result) {
  const sm = msgNode.__searchMatrix;
  if (!sm) return;
  const colIdx = msgNode.__smPendingCol;
  if (colIdx === undefined || colIdx < 0) return;

  const parsed = typeof result === "object" ? result : {};
  const clauses = parsed.clauses || [];

  if (clauses.length > 0) {
    // read_clause returns no score — give targeted fetches a high default
    const isFetch = toolName === "read_clause";
    const mapped = clauses.map(c => ({
      clause_id: c.clause_id,
      title: c.title,
      standard: c.standard,
      score: c.score || (isFetch ? 10 : 0),
      selected: isFetch || !!c.selected,  // backend LLM marks selected; fetches always selected
    }));
    _smPopulateColumn(msgNode, colIdx, mapped, isFetch);
    _smFinalizeColumn(msgNode, colIdx);

    // Track selected (green) clause keys — these are the only ones that become references
    const selected = mapped.filter(c => c.selected);
    if (msgNode.__selectedRefKeys) {
      for (const c of selected) {
        msgNode.__selectedRefKeys.add(`${c.standard || ""}:${c.clause_id || ""}`);
      }
    }

    // Update summary — only the actually-selected (green) clauses
    _smShowSummary(msgNode, selected);
  } else {
    // Empty result — show "no hits"
    const col = sm.columns[colIdx];
    if (col) {
      const rowsEl = col.el.querySelector(".sm-col-rows");
      if (rowsEl) {
        rowsEl.innerHTML = `<div class="sm-row show"><span class="sm-row-id" style="color:var(--muted);font-style:italic">no hits</span></div>`;
      }
    }
    _smFinalizeColumn(msgNode, colIdx);
  }

  delete msgNode.__smPendingCol;
}

// ---- Flow Graph Event Processing ----

function processEvent(f, ev) {
  const s = ev.status || "active", node = ev.node, m = ev.meta || {};
  if (node === "intake") {
    if (s === "active") {
      setNS(f, "user", "done");
      setNS(f, "orchestrator", "active");
      setES(f, "u_o", "active");
      return;
    }
    if (s === "error") {
      setNS(f, "user", "done");
      setNS(f, "orchestrator", "error");
      setES(f, "u_o", "error");
      return;
    }
    setNS(f, "orchestrator", "done");
    setES(f, "u_o", "done");
  } else if (node === "plan") {
    setNS(f, "orchestrator", s === "done" ? "done" : "active");
  } else if (node === "inputs") {
    setNS(f, "orchestrator", s === "done" ? "done" : "active");
  } else if (node === "retrieval") {
    const skipped = m.skipped === true || /skipped/i.test(String(ev.detail || ""));
    setNS(f, "orchestrator", "done");
    if (skipped) {
      setNS(f, "database", "idle");
      setES(f, "o_d", "idle");
      setES(f, "d_o", "idle");
      return;
    }
    if (s === "active") {
      clearPopupLane(f, "docs");
      setES(f, "o_d", "active");
      setES(f, "d_o", "idle");
    } else if (s === "done") {
      setES(f, "o_d", "done");
      setES(f, "d_o", "done");
    } else {
      setES(f, "o_d", "error");
      setES(f, "d_o", "error");
    }
    setNS(f, "database", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    if (m.top?.length) pushDocBadges(f, m.top);
    if (m.top_clauses?.length) pushDocBadges(f, m.top_clauses);
  } else if (node === "tools") {
    setNS(f, "orchestrator", "done");
    if (ev.skipped) {
      setNS(f, "tools", "idle");
      setES(f, "o_t", "idle");
      setES(f, "t_o", "idle");
    } else if (s === "active") {
      setES(f, "o_t", "active");
      setES(f, "t_o", "idle");
      setNS(f, "tools", "active");
    } else if (s === "done") {
      setES(f, "o_t", "done");
      setES(f, "t_o", "done");
      setNS(f, "tools", "done");
    } else {
      setES(f, "o_t", "error");
      setES(f, "t_o", "error");
      setNS(f, "tools", "error");
    }
    if (m.tool) pushToolBadge(f, m.tool);
  } else if (node === "compose") {
    setNS(f, "orchestrator", s === "done" ? "done" : (s === "error" ? "error" : "active"));
    if (s === "error") setNS(f, "response", "error");
    setES(f, "o_r", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    if (m.used_tools?.length) {
      setNS(f, "tools", "done");
      for (const t of m.used_tools) pushToolBadge(f, t);
    }
  } else if (node === "fea_analyst") {
    setNS(f, "orchestrator", "done");
    setNS(f, "fea_analyst", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    setES(f, "o_fa", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    if (s === "done") {
      setES(f, "fa_o", "done");
      setNS(f, "orchestrator", "done");
      setES(f, "o_r", "done");
      setNS(f, "response", "done");
      setNS(f, "user", "done");
    }
  } else if (node === "output") {
    setNS(f, "orchestrator", "done");
    setNS(f, "response", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    setES(f, "o_r", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    setNS(f, "user", "done");
  }
}

function appendLog(msgNode, text) {
  const log = msgNode.querySelector(".machine-log");
  if (!log) return;
  const li = document.createElement("li");
  li.className = "log-item";
  const ts = document.createElement("span");
  ts.className = "log-ts";
  ts.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const msg = document.createElement("span");
  msg.className = "log-msg";
  msg.textContent = clamp(text, 260);
  li.appendChild(ts);
  li.appendChild(msg);
  log.prepend(li);
  while (log.children.length > 12) log.removeChild(log.lastChild);
}

// ---- Witty rotating phrases while thinking ----
const _THINKING_PHRASES = [
  "Sharpening the pencil",
  "Warming up the calculator",
  "Flipping through the textbook",
  "Putting on the hard hat",
  "Rolling up the sleeves",
  "Brewing a strong coffee",
  "Dusting off the slide rule",
  "Squinting at the fine print",
  "Stretching before the heavy lifting",
  "Cracking the knuckles",
  "Gathering the evidence",
  "Connecting the dots",
  "Thinking really hard",
  "Almost there, probably",
  "Doing the engineering thing",
  "Consulting the sacred texts",
];

function _startPhraseRotation(msgNode) {
  if (msgNode.__phraseTimer) return;
  let idx = Math.floor(Math.random() * _THINKING_PHRASES.length);
  const label = msgNode.querySelector(".thinking-label");
  if (!label) return;
  label.textContent = _THINKING_PHRASES[idx];
  msgNode.__phraseTimer = setInterval(() => {
    idx = (idx + 1) % _THINKING_PHRASES.length;
    label.classList.add("phrase-fade");
    setTimeout(() => {
      label.textContent = _THINKING_PHRASES[idx];
      label.classList.remove("phrase-fade");
    }, 200);
  }, 2800);
}

function _stopPhraseRotation(msgNode) {
  if (msgNode.__phraseTimer) {
    clearInterval(msgNode.__phraseTimer);
    msgNode.__phraseTimer = null;
  }
}

function updateThinkingLabel(msgNode, text) {
  _stopPhraseRotation(msgNode);
  const label = msgNode.querySelector(".thinking-label");
  if (label) label.textContent = clamp(text, 120);
}

function setThinkingState(msgNode, active) {
  const panel = msgNode.querySelector(".thinking-panel");
  if (!panel) return;
  panel.classList.toggle("is-thinking", !!active);
  if (active) {
    _startPhraseRotation(msgNode);
  } else {
    _stopPhraseRotation(msgNode);
  }
}

function setComposerStreaming(active) {
  if (!sendBtn) return;
  if (active) {
    sendBtn.disabled = false;
    sendBtn.innerHTML = STOP_ICON;
    sendBtn.setAttribute("aria-label", "Stop");
    sendBtn.classList.add("stop-mode");
    setThinkingModeDisabled(true);
    return;
  }
  sendBtn.innerHTML = SEND_ICON;
  sendBtn.setAttribute("aria-label", "Send");
  sendBtn.classList.remove("stop-mode");
  sendBtn.disabled = false;
  setThinkingModeDisabled(false);
}

function previewPairs(obj, max = 3) {
  if (!obj || typeof obj !== "object") return "";
  const pairs = Object.entries(obj)
    .filter(([, v]) => v !== null && v !== undefined && String(v).trim() !== "")
    .slice(0, max)
    .map(([k, v]) => `${k}=${String(v)}`);
  return pairs.join(", ");
}

function previewClauses(entries, max = 3) {
  if (!Array.isArray(entries)) return "";
  return entries
    .map((e) => String(e?.clause_id || "").trim())
    .filter(Boolean)
    .slice(0, max)
    .join(", ");
}

function describeMachineStep(ev) {
  if (!ev || !ev.node) return "";
  const s = ev.status || "active";
  const m = ev.meta || {};

  if (ev.node === "intake") {
    if (s === "active") return "Read your request and started orchestrator intake.";
    if (s === "done") return "Parsed the request and moved to planning.";
    return "Stopped during intake due to an error.";
  }

  if (ev.node === "plan") {
    const thinking = String(m.thinking_mode || "");
    const mode = String(m.mode || "retrieval_only").replace(/_/g, " ");
    const tools = Array.isArray(m.tools) && m.tools.length ? m.tools.map(normTool).join(" -> ") : "no tools";
    const modeLabel = thinking ? `${thinking} mode` : "default mode";
    return `Planned ${mode} path (${modeLabel}) with tool chain: ${tools}.`;
  }

  if (ev.node === "inputs") {
    if (ev.skipped) return null;  // no tools → no inputs step needed
    if (s === "active") return "Resolving user-provided values and defaults.";
    if (s === "done") {
      const provided = Object.keys(m.user_inputs || {}).length;
      const defaulted = Object.keys(m.assumed_inputs || {}).length;
      const sample = previewPairs(m.user_inputs);
      return sample
        ? `Resolved inputs (${provided} provided, ${defaulted} defaulted). Key values: ${sample}.`
        : `Resolved inputs (${provided} provided, ${defaulted} defaulted).`;
    }
    return "Input resolution failed.";
  }

  if (ev.node === "retrieval") {
    const skipped = m.skipped === true || /skipped/i.test(String(ev.detail || ""));
    if (skipped) return "Skipped database retrieval for calculator-only path.";
    if (s === "active") {
      const iteration = m.iteration?.iteration || m.iteration?.pass || "";
      const clauses = previewClauses(m.top || m.top_clauses);
      if (iteration && clauses) return `Search pass ${iteration}: top EC3 clauses ${clauses}.`;
      if (iteration) return `Search pass ${iteration}: updating EC3 evidence ranking.`;
      return "Searching EC3 clauses and ranking relevance.";
    }
    if (s === "done") {
      const count = Number(m.retrieved_count || 0);
      const clauses = previewClauses(m.top_clauses || m.top);
      return clauses
        ? `Selected ${count} relevant clause(s). Top hits: ${clauses}.`
        : `Selected ${count} relevant clause(s) for evidence.`;
    }
    return "Retrieval step failed.";
  }

  if (ev.node === "tools") {
    if (ev.skipped) return null;  // no tools used — suppress log line
    const toolName = normTool(m.tool || "");
    if (s === "error") {
      return toolName ? `Tool ${toolName} failed; answer support may be limited.` : "Tool execution failed.";
    }
    if (toolName && m.status === "ok") return `Tool ${toolName} completed successfully.`;
    if (toolName && s === "active") return `Running ${toolName} with resolved inputs.`;
    if (s === "done") return "Tool execution finished.";
    return "Executing tool chain.";
  }

  if (ev.node === "compose") {
    const usedTools = Array.isArray(m.used_tools) ? m.used_tools.length : 0;
    const usedSources = Array.isArray(m.used_sources) ? m.used_sources.length : 0;
    if (s === "active") {
      return (usedTools || usedSources)
        ? "Composing response from tool outputs and retrieved clauses."
        : (ev.detail || "Generating response...");
    }
    if (s === "done") {
      return (usedTools || usedSources)
        ? `Draft complete with ${usedTools} tool(s) and ${usedSources} source citation(s).`
        : (ev.detail || "Response ready.");
    }
    return "Could not fully ground the draft in available evidence.";
  }

  if (ev.node === "fea_analyst") {
    if (s === "active") return ev.detail || "FEA Analyst initializing...";
    if (s === "done") return ev.detail || "FEA analysis complete.";
    return ev.detail || "FEA Analyst error.";
  }

  if (ev.node === "output") {
    if (s === "active") return "Streaming response to chat.";
    if (s === "done") return "Response delivered.";
    return "Output stage failed.";
  }

  return ev.detail || `${ev.node}: ${s}`;
}

function updateFlow(msgNode, ev) {
  const f = msgNode.__flow;
  if (!f || !ev?.node) return;
  msgNode.__stepCount = (msgNode.__stepCount || 0) + 1;
  processEvent(f, ev);
  applyFlow(msgNode);
  const detail = describeMachineStep(ev);
  // Don't call updateThinkingLabel here — the phrase rotator handles it
  if (detail !== null) {
    appendLog(msgNode, detail || ev.detail || `${ev.node}: ${ev.status}`);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function finalizeThinking(msgNode, payload) {
  const f = msgNode.__flow;
  if (!f) return;
  setNS(f, "response", payload.supported ? "done" : "error");
  setNS(f, "user", "done");
  setES(f, "o_r", payload.supported ? "done" : "error");
  applyFlow(msgNode);
  setThinkingState(msgNode, false);

  const elapsed = ((Date.now() - (msgNode.__thinkStart || Date.now())) / 1000).toFixed(1);
  const steps = msgNode.__stepCount || 0;
  const meta = msgNode.querySelector(".thinking-meta");
  if (meta) meta.textContent = `${steps} steps \u00B7 ${elapsed}s`;
  updateThinkingLabel(msgNode, "Reasoning complete. Expand to review steps.");
}

function setTrace(msgNode, payload) {
  const trace = msgNode.querySelector(".trace");
  const body = msgNode.querySelector(".trace-body");
  if (!trace || !body) return;
  const lines = [];
  if (payload.what_i_used?.length) lines.push(...payload.what_i_used.map(i => `• ${i}`));
  if (payload.tool_trace?.length) {
    lines.push("", "Tool chain:");
    for (const s of payload.tool_trace) lines.push(`  ${s.status === "ok" ? "✓" : "✗"} ${s.tool_name}: ${s.status}`);
  }
  if (payload.assumptions?.length) {
    lines.push("", "Assumptions:");
    for (const a of payload.assumptions) lines.push(`  → ${a}`);
  }
  if (!lines.length) return;
  body.textContent = lines.join("\n");
  trace.classList.remove("hidden");
}

// ---- Activity Feed (Agent Mode) ----

const _ACTIVITY_ICONS = {
  thinking: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>`,
  plan: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/></svg>`,
  search: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>`,
  tool: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>`,
};

const _STEP_ICONS = { pending: "\u25CB", in_progress: "\u25B6", done: "\u2713", error: "\u2717" };

function _getActivityFeed(msgNode) {
  return msgNode.querySelector(".activity-feed");
}

function _makeCard(type, label, body) {
  const card = document.createElement("div");
  card.className = `activity-card ${type}-card`;
  card.innerHTML = `<div class="card-icon">${_ACTIVITY_ICONS[type] || _ACTIVITY_ICONS.thinking}</div>
    <div class="card-body"><div class="card-label">${escHtml(label)}</div><div class="card-content">${body}</div></div>`;
  requestAnimationFrame(() => requestAnimationFrame(() => card.classList.add("show")));
  return card;
}

function _autoScrollThinkingPanel(msgNode) {
  const panel = msgNode.querySelector(".thinking-panel");
  if (panel && !panel.classList.contains("collapsed")) {
    requestAnimationFrame(() => {
      panel.scrollTop = panel.scrollHeight;
    });
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendThinkingCard(msgNode, content) {
  const feed = _getActivityFeed(msgNode);
  if (!feed) return;
  const card = _makeCard("thinking", "Thinking", escHtml(content));
  feed.appendChild(card);
  _autoScrollThinkingPanel(msgNode);
}

function renderPlanCard(msgNode, steps) {
  const feed = _getActivityFeed(msgNode);
  if (!feed) return;
  // Reuse existing plan card if one exists (update in place)
  let card = feed.querySelector(".activity-card.plan-card");
  const safeSteps = Array.isArray(steps)
    ? steps
        .map(s => {
          const id = String(s?.id || "").trim();
          const text = typeof s?.text === "string" ? s.text.trim() : "";
          if (!id || !text) return null;
          return { id, text, status: s.status || "pending" };
        })
        .filter(Boolean)
    : [];
  if (!safeSteps.length) {
    card?.remove();
    return;
  }
  const items = safeSteps.map(s => {
    const st = s.status || "pending";
    const icon = _STEP_ICONS[st] || _STEP_ICONS.pending;
    return `<div class="plan-step ${st}" data-step-id="${s.id}"><span class="step-icon">${icon}</span><span>${escHtml(s.text)}</span></div>`;
  }).join("");
  if (card) {
    card.querySelector(".plan-checklist").innerHTML = items;
  } else {
    card = _makeCard("plan", "Plan", `<div class="plan-checklist">${items}</div>`);
    feed.appendChild(card);
  }
  _autoScrollThinkingPanel(msgNode);
}

function updatePlanStep(msgNode, stepId, status) {
  const feed = _getActivityFeed(msgNode);
  if (!feed) return;
  const step = feed.querySelector(`.plan-step[data-step-id="${stepId}"]`);
  if (!step) return;
  step.className = `plan-step ${status}`;
  step.querySelector(".step-icon").textContent = _STEP_ICONS[status] || _STEP_ICONS.pending;
  // Update thinking label to reflect current task
  if (status === "in_progress") {
    const taskText = step.querySelector("span:last-child")?.textContent || "";
    if (taskText) updateThinkingLabel(msgNode, taskText);
  }
}

/* ── Ask-user popup ─────────────────────────────────────────────── */

/**
 * Show the ask-user popup. When the user answers, `onAnswer(value)`
 * is called instead of creating a new chat message.
 */
function showAskUserPopup(question, options, context, onAnswer) {
  const overlay = document.getElementById("ask-user-overlay");
  const qEl = document.getElementById("ask-user-popup-question");
  const ctxEl = document.getElementById("ask-user-popup-context");
  const optsEl = document.getElementById("ask-user-popup-options");
  const inputEl = document.getElementById("ask-user-popup-input");
  const sendEl = document.getElementById("ask-user-popup-send");
  const popupEl = overlay?.querySelector(".ask-user-popup");
  if (!overlay) return;

  qEl.innerHTML = renderInlineMd(question || "");
  ctxEl.innerHTML = context ? renderMd(context) : "";
  optsEl.innerHTML = "";

  if (options && options.length) {
    options.forEach(o => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.value = o.value || o.label;
      btn.innerHTML =
        `<span class="ask-option-label">${renderInlineMd(o.label || "")}</span>` +
        (o.description ? `<span class="ask-option-desc">${renderInlineMd(o.description)}</span>` : "");
      btn.addEventListener("click", () => {
        inputEl.value = btn.dataset.value;
        inputEl.focus();
        optsEl.querySelectorAll("button").forEach(b => b.classList.remove("selected"));
        btn.classList.add("selected");
      });
      optsEl.appendChild(btn);
    });
  }

  inputEl.value = "";
  overlay.classList.remove("hidden");
  positionPopup();
  inputEl.focus();

  function submit() {
    const val = inputEl.value.trim();
    if (!val) return;
    overlay.classList.add("hidden");
    cleanup();
    if (onAnswer) onAnswer(val);
  }

  function cleanup() {
    sendEl.removeEventListener("click", submit);
    inputEl.removeEventListener("keydown", onKey);
    window.removeEventListener("resize", positionPopup);
    overlay.style.removeProperty("--ask-user-left");
    overlay.style.removeProperty("--ask-user-width");
    overlay.style.removeProperty("--ask-user-bottom");
  }

  function onKey(e) {
    if (e.key === "Enter") { e.preventDefault(); submit(); }
    if (e.key === "Escape") { overlay.classList.add("hidden"); cleanup(); }
  }

  function positionPopup() {
    if (!popupEl) return;
    const composerEl = document.getElementById("chat-form");
    if (!composerEl) return;
    const rect = composerEl.getBoundingClientRect();
    const gap = 12;
    const maxWidth = Math.min(rect.width, window.innerWidth - 24);
    const left = Math.min(
      Math.max(12, rect.left),
      Math.max(12, window.innerWidth - maxWidth - 12),
    );
    const bottom = Math.max(18, window.innerHeight - rect.top + gap);
    overlay.style.setProperty("--ask-user-left", `${Math.round(left)}px`);
    overlay.style.setProperty("--ask-user-width", `${Math.round(maxWidth)}px`);
    overlay.style.setProperty("--ask-user-bottom", `${Math.round(bottom)}px`);
  }

  sendEl.addEventListener("click", submit);
  inputEl.addEventListener("keydown", onKey);
  window.addEventListener("resize", positionPopup);
}

function appendToolCard(msgNode, toolName, args, status) {
  const feed = _getActivityFeed(msgNode);
  if (!feed) return;
  const isSearch = toolName === "search" || toolName === "retrieval";
  const type = isSearch ? "search" : "tool";
  const displayName = isSearch ? "Searching standards" : toolName.replace(/_/g, " ");
  const argsPreview = args ? previewPairs(args, 4) : "";
  const statusClass = status || "running";
  const statusIcon = statusClass === "running" ? `<span class="spinner"></span>` : "";
  const body = `<span>${escHtml(displayName)}</span>${argsPreview ? `<span class="tool-args">${escHtml(argsPreview)}</span>` : ""}`;
  const card = _makeCard(type, type === "search" ? "Retrieval" : "Tool", body);
  card.dataset.toolName = toolName;
  card.innerHTML += `<div class="card-status ${statusClass}">${statusIcon}</div>`;
  feed.appendChild(card);
  _autoScrollThinkingPanel(msgNode);
}

function updateToolCard(msgNode, toolName, result, status, summary) {
  const feed = _getActivityFeed(msgNode);
  if (!feed) return;
  // Find last card matching this tool name
  const cards = feed.querySelectorAll(`.activity-card[data-tool-name="${toolName}"]`);
  const card = cards[cards.length - 1];
  if (!card) return;
  // Update status indicator
  const statusEl = card.querySelector(".card-status");
  if (statusEl) {
    statusEl.className = `card-status ${status}`;
    statusEl.innerHTML = status === "ok" ? _STEP_ICONS.done : _STEP_ICONS.error;
  }
  // Add summary/result detail
  if (summary || result) {
    const body = card.querySelector(".card-content");
    if (body && summary) {
      body.innerHTML += `<div class="tool-summary">${escHtml(summary)}</div>`;
    }
    if (result) {
      const details = document.createElement("details");
      details.className = "card-details";
      details.innerHTML = `<summary>Details</summary><pre class="card-detail-pre">${escHtml(typeof result === "string" ? result : JSON.stringify(result, null, 2))}</pre>`;
      card.querySelector(".card-body")?.appendChild(details);
    }
  }
}

function finalizeAgentThinking(msgNode, taskCount) {
  setThinkingState(msgNode, false);
  const elapsed = ((Date.now() - (msgNode.__thinkStart || Date.now())) / 1000).toFixed(1);
  const meta = msgNode.querySelector(".thinking-meta");
  if (meta) meta.textContent = `${taskCount} task${taskCount !== 1 ? "s" : ""} \u00B7 ${elapsed}s`;
  updateThinkingLabel(msgNode, "Completed. Expand to review steps.");
}

function pauseAgentThinking(msgNode, taskCount) {
  const f = msgNode.__flow;
  if (f) {
    setNS(f, "orchestrator", "done");
    setNS(f, "response", "done");
    setNS(f, "user", "active");
    setES(f, "u_o", "done");
    setES(f, "o_r", "done");
    applyFlow(msgNode);
  }
  setThinkingState(msgNode, false);
  const elapsed = ((Date.now() - (msgNode.__thinkStart || Date.now())) / 1000).toFixed(1);
  const meta = msgNode.querySelector(".thinking-meta");
  if (meta) meta.textContent = `${taskCount} task${taskCount !== 1 ? "s" : ""} \u00B7 ${elapsed}s`;
  updateThinkingLabel(msgNode, "Waiting for your answer.");
}

function pauseThinking(msgNode) {
  const f = msgNode.__flow;
  if (f) {
    setNS(f, "orchestrator", "done");
    setNS(f, "response", "done");
    setNS(f, "user", "active");
    setES(f, "u_o", "done");
    setES(f, "o_r", "done");
    applyFlow(msgNode);
  }
  setThinkingState(msgNode, false);
  const elapsed = ((Date.now() - (msgNode.__thinkStart || Date.now())) / 1000).toFixed(1);
  const steps = msgNode.__stepCount || 0;
  const meta = msgNode.querySelector(".thinking-meta");
  if (meta) meta.textContent = `${steps} steps \u00B7 ${elapsed}s`;
  updateThinkingLabel(msgNode, "Waiting for your answer.");
}

// ---- FEA Panel Integration ----

async function initFEAPanel(msgNode) {
  const feaSection = msgNode.querySelector(".fea-panel");
  if (!feaSection) return;

  feaSection.classList.remove("hidden");
  feaSection.classList.add("collapsed");

  try {
    const { FEAPanelController } = await import("/static/fea_panel.js");
    const panel = new FEAPanelController(feaSection);
    msgNode.__feaPanel = panel;
  } catch (err) {
    console.error("Failed to init FEA panel:", err);
    feaSection.innerHTML = `<div class="fea-error">FEA panel failed to load: ${err.message}</div>`;
  }
}

async function restoreFEAPanelFromPayload(msgNode, payload) {
  const memory = getFEASessionMemory(payload);
  const feaSession = memory?.fea_session;
  if (!feaSession || typeof feaSession !== "object") return;
  if (!feaSession.model_snapshot || typeof feaSession.model_snapshot !== "object") return;

  if (!msgNode.__feaPanelReady) {
    msgNode.__feaPanelReady = initFEAPanel(msgNode);
  }
  try {
    await msgNode.__feaPanelReady;
    if (msgNode.__feaPanel) {
      await msgNode.__feaPanel.restoreSessionSnapshot(feaSession);
    }
  } catch (err) {
    console.error("Failed to restore FEA panel snapshot:", err);
  }
}

/**
 * Show a popup for the FEA analyst to ask the user a clarifying question.
 * Reuses the main ask-user popup so FEA follows the same human-in-the-loop UX.
 */
function showFEAQueryPopup(msgNode, sessionId, question, options, context, onAnswerSent) {
  const normalizedOptions = Array.isArray(options)
    ? options
        .map(option => {
          if (typeof option !== "string") return null;
          const value = option.trim();
          return value ? { label: value, value } : null;
        })
        .filter(Boolean)
    : [];

  showAskUserPopup(question, normalizedOptions, context, async (answer) => {
    const value = String(answer || "").trim();
    if (!value) return;
    try {
      await fetch("/api/fea/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, answer: value }),
      });
      appendThinkingCard(msgNode, `You answered: ${value}`);
      if (typeof onAnswerSent === "function") onAnswerSent(value);
    } catch (err) {
      console.error("Failed to send FEA answer:", err);
      appendLog(msgNode, `Failed to send FEA answer: ${err.message}`);
    }
  });
}

// ---- Messages ----
function createMsg(role, content = "", opts = {}) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  node.querySelector(".role").textContent = role === "assistant" ? "Assistant" : "You";

  const contentEl = node.querySelector(".content");
  if (role === "assistant") {
    contentEl.innerHTML = renderMd(content);
    // Assistant messages: hide edit button, keep copy
    node.querySelector(".edit-btn")?.remove();
    if (opts.showThinking !== false) {
      initFlowGraph(node, opts.prompt || "");
    } else {
      node.querySelector(".diagram-panel")?.classList.add("hidden");
      node.querySelector(".thinking-panel")?.classList.add("hidden");
    }
    if (opts.responsePayload) {
      setTrace(node, opts.responsePayload);
      restoreFEAPanelFromPayload(node, opts.responsePayload).catch(err => {
        console.error("Stored FEA panel restore failed:", err);
      });
    }
  } else {
    const attHtml = buildAttachmentHtml(opts.attachments);
    if (attHtml) {
      contentEl.innerHTML = attHtml;
      const textNode = document.createElement("span");
      textNode.textContent = content;
      contentEl.appendChild(textNode);
      attachLightboxListeners(contentEl);
    } else {
      contentEl.textContent = content;
    }
    node.querySelector(".diagram-panel")?.remove();
    node.querySelector(".thinking-panel")?.remove();
    node.querySelector(".trace")?.remove();

    // Store original prompt + attachments for edit & resubmit
    node.__editContent = content;
    node.__editAttachments = opts.attachments || [];
  }

  const copyBtn = node.querySelector(".copy-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const text = role === "assistant" ? (contentEl.innerText || contentEl.textContent) : content;
      navigator.clipboard?.writeText(text);
      copyBtn.title = "Copied!";
      setTimeout(() => { copyBtn.title = "Copy to clipboard"; }, 1500);
    });
  }

  // Edit & resubmit for user messages
  const editBtn = node.querySelector(".edit-btn");
  if (editBtn && role === "user") {
    editBtn.addEventListener("click", () => editAndResubmit(node));
  }

  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

// ---- Inline Edit & Resubmit ----
function editAndResubmit(userMsgNode) {
  // Don't allow editing while a request is in flight or already editing
  if (sendBtn.disabled) return;
  if (userMsgNode.classList.contains("editing")) return;

  const contentEl = userMsgNode.querySelector(".content");
  const originalHtml = contentEl.innerHTML;
  const prompt = userMsgNode.__editContent || "";
  const attachments = userMsgNode.__editAttachments || [];

  userMsgNode.classList.add("editing");

  // Build inline editor
  const textarea = document.createElement("textarea");
  textarea.className = "inline-edit-input";
  textarea.value = prompt;

  const btnRow = document.createElement("div");
  btnRow.className = "inline-edit-actions";

  const saveBtn = document.createElement("button");
  saveBtn.className = "inline-edit-save";
  saveBtn.textContent = "Save & Submit";
  saveBtn.type = "button";

  const cancelBtn = document.createElement("button");
  cancelBtn.className = "inline-edit-cancel";
  cancelBtn.textContent = "Cancel";
  cancelBtn.type = "button";

  btnRow.appendChild(saveBtn);
  btnRow.appendChild(cancelBtn);

  // Replace content with textarea + buttons
  contentEl.innerHTML = "";
  contentEl.appendChild(textarea);
  contentEl.appendChild(btnRow);

  // Auto-size textarea to fit content
  textarea.style.height = "auto";
  textarea.style.height = textarea.scrollHeight + "px";
  textarea.focus();
  textarea.selectionStart = textarea.selectionEnd = textarea.value.length;

  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
  });

  // Cancel → restore original content
  cancelBtn.addEventListener("click", () => {
    contentEl.innerHTML = originalHtml;
    userMsgNode.classList.remove("editing");
    if (attachments.length) attachLightboxListeners(contentEl);
  });

  // Save → resubmit from this point
  saveBtn.addEventListener("click", async () => {
    const newText = textarea.value.trim();
    if (!newText && !attachments.length) return;

    userMsgNode.classList.remove("editing");

    // Update the displayed content
    const attHtml = buildAttachmentHtml(attachments);
    if (attHtml) {
      contentEl.innerHTML = attHtml;
      const span = document.createElement("span");
      span.textContent = newText;
      contentEl.appendChild(span);
      attachLightboxListeners(contentEl);
    } else {
      contentEl.textContent = newText;
    }

    // Update stored edit content
    userMsgNode.__editContent = newText;

    // Remove all messages AFTER this one from the DOM
    const allMsgs = [...messagesEl.querySelectorAll(".message")];
    const idx = allMsgs.indexOf(userMsgNode);
    if (idx < 0) return;
    const toRemove = allMsgs.slice(idx + 1);
    for (const el of toRemove) el.remove();

    // Update thread: keep this message (with new content), drop everything after
    const thread = currentThread();
    if (thread && thread.messages.length > idx) {
      thread.messages[idx].content = newText;
      thread.messages[idx].attachments = attachments;
      thread.messages.length = idx + 1;
      thread.updatedAt = now();
    }

    // Build full prompt with attachment markers
    let fullPrompt = newText;
    if (attachments.length) {
      const fileDescs = attachments.map(a =>
        a.isImage ? `[Attached image: ${a.name}]` : `[Attached file: ${a.name} (${fmtFileSize(a.size)})]`
      ).join(" ");
      fullPrompt = fullPrompt ? `${fileDescs}\n\n${fullPrompt}` : fileDescs;
    }

    // Lock UI before any async work to prevent concurrent edits/submissions
    sendBtn.disabled = true;
    setThinkingModeDisabled(true);

    if (canUseStoredThreads()) {
      if (auth.threadsSync) {
        try {
          await truncateThreadApi(thread.id, idx + 1, newText);
        } catch (e) {
          console.warn("Thread truncation sync failed:", e);
        }
      } else {
        save();
      }
    }

    // Create assistant node and stream response
    _cleanupFloatingDiagrams();
    const assistantNode = createMsg("assistant", "", { showThinking: true, prompt: fullPrompt });

    try {
      await streamChat(fullPrompt, assistantNode, thread, state.thinkingMode, attachments, { isEdit: true });
    } catch (err) {
      const errMsg = `Error: ${err.message}`;
      setThinkingState(assistantNode, false);
      updateThinkingLabel(assistantNode, "Reasoning failed.");
      const editErrEl = assistantNode.querySelector(".content");
      editErrEl.classList.remove("streaming");
      editErrEl.innerHTML = `<div class="error-msg">${escHtml(errMsg)}</div>`;
      appendLog(assistantNode, `Transport error: ${err.message}`);
      thread.messages.push({ id: uid(), role: "assistant", content: errMsg, responsePayload: null, createdAt: now() });
      thread.updatedAt = now();
      if (canUseStoredThreads()) {
        if (auth.threadsSync) {
          await addMessageToApi(thread.id, "assistant", errMsg, null);
        } else {
          save();
        }
      }
    } finally {
      sendBtn.disabled = false;
      setThinkingModeDisabled(false);
    }
  });

  // Keyboard shortcuts: Ctrl/Cmd+Enter to submit, Escape to cancel
  textarea.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      saveBtn.click();
    }
    if (e.key === "Escape") {
      e.preventDefault();
      cancelBtn.click();
    }
  });
}

function _cleanupFloatingDiagrams() {
  // Remove any diagram panels that were moved to body level
  document.querySelectorAll("body > .diagram-panel").forEach(el => el.remove());
}

// ---- Context Usage Circle ----
function renderContextUsage(usage) {
  const btn = document.getElementById("context-usage-btn");
  const ring = document.getElementById("context-usage-ring");
  const primary = document.getElementById("context-usage-primary");
  const secondary = document.getElementById("context-usage-secondary");
  if (!btn || !ring) return;

  const radius = 8.5;
  const circumference = 2 * Math.PI * radius;
  ring.style.strokeDasharray = `${circumference}`;

  if (!usage) {
    ring.style.strokeDashoffset = `${circumference}`;
    btn.classList.remove("low", "medium", "high", "critical");
    if (primary) primary.textContent = "No usage data";
    if (secondary) secondary.textContent = "\u2014";
    return;
  }

  const usedPercent = Math.max(0, Number(usage.used_percent) || 0);
  const fraction = Math.min(1, usedPercent / 100);
  ring.style.strokeDashoffset = `${circumference * (1 - fraction)}`;

  btn.classList.remove("low", "medium", "high", "critical");
  const level = String(usage.level || "low").toLowerCase();
  btn.classList.add(level);

  const usedRounded = Math.round(usedPercent);
  const leftPercent = Math.max(0, 100 - usedRounded);
  if (primary) primary.textContent = `${usedRounded}% used (${leftPercent}% left)`;

  const fmt = n => n >= 1_000_000 ? (n / 1_000_000).toFixed(1) + "M" : n >= 1000 ? (n / 1000).toFixed(0) + "k" : String(n);
  if (secondary) secondary.textContent = `${fmt(usage.estimated_tokens || 0)} / ${fmt(usage.context_window || 0)} tokens`;
}

function renderMessages() {
  _cleanupFloatingDiagrams();
  messagesEl.innerHTML = "";
  const t = currentThread();
  if (!t) { updateWelcome(); return; }
  for (const m of t.messages || []) {
    if (m.role === "assistant") {
      createMsg("assistant", getAssistantDisplayContent(m), { showThinking: false, responsePayload: m.responsePayload });
    } else {
      createMsg("user", m.content || "", { attachments: m.attachments });
    }
  }
  updateWelcome();
}

// ---- Streaming ----
async function streamChat(prompt, assistantNode, thread, thinkingMode = "thinking", attachments = [], { isEdit = false, isContinuation = false } = {}) {
  const contentEl = assistantNode.querySelector(".content");
  contentEl.innerHTML = "";
  contentEl.classList.add("streaming");
  let accumulated = "";
  assistantNode.__accumulatedMd = "";
  let renderTimer = null;
  let lastRenderLen = 0;

  function scheduleRender() {
    if (renderTimer) return;
    // Shorter debounce for snappier token feel
    renderTimer = setTimeout(() => {
      if (accumulated.length !== lastRenderLen) {
        contentEl.innerHTML = renderMd(accumulated);
        lastRenderLen = accumulated.length;
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
      renderTimer = null;
    }, 30);
  }

  const threadMsgs = thread.messages || [];
  const prevMsgs = threadMsgs.slice(0, -1);
  const history = serializeHistoryMessages(prevMsgs);

  // Build attachments payload for the API (with base64 data for images)
  const apiAttachments = attachments.map(a => ({
    name: a.name,
    type: a.type || "",
    size: a.size || 0,
    is_image: !!a.isImage,
    data_url: a.isImage ? (a.dataUrl || null) : null,
  }));

  const abortController = new AbortController();
  state.abortController = abortController;

  const res = await fetchWithAuth("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: prompt,
        history,
        thinking_mode: thinkingMode,
        attachments: apiAttachments,
        is_edit: isEdit,
        is_ask_user_reply: isContinuation,
        web_search: state.webSearchEnabled,
      }),
    signal: abortController.signal,
  });
  if (!res.ok || !res.body) throw new Error(`Request failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalized = false;
  let lastPayload = null;
  let isAgentMode = false;
  let agentTaskCount = 0;
  // Accumulators for "What I used" trace section
  if (!isContinuation) {
    assistantNode.__toolTrace = [];
    assistantNode.__usedClauses = [];
    assistantNode.__calcResults = [];
    assistantNode.__refClauses = [];
    assistantNode.__selectedRefKeys = new Set();
  } else {
    assistantNode.__toolTrace = assistantNode.__toolTrace || [];
    assistantNode.__usedClauses = assistantNode.__usedClauses || [];
    assistantNode.__calcResults = assistantNode.__calcResults || [];
    assistantNode.__refClauses = assistantNode.__refClauses || [];
    assistantNode.__selectedRefKeys = assistantNode.__selectedRefKeys || new Set();
  }
  assistantNode.__planSteps = assistantNode.__planSteps || [];
  assistantNode.__pendingAskUser = null;
  assistantNode.__pendingCalcArgs = null; // stash tool_start args for calc tools

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      let event;
      try { event = JSON.parse(line); } catch { continue; }

      // Legacy pipeline events
      if (event.type === "machine") updateFlow(assistantNode, event);

      // Agent activity events
      if (event.type === "thinking") {
        appendThinkingCard(assistantNode, event.content);
        agentTaskCount = (agentTaskCount || 0);
        isAgentMode = true;
        // Keep orchestrator active while thinking
        if (assistantNode.__flow) {
          const f = assistantNode.__flow;
          if (f.ns["orchestrator"] !== "active") {
            setNS(f, "orchestrator", "active");
            applyFlow(assistantNode);
          }
        }
      }
      if (event.type === "plan") {
        renderPlanCard(assistantNode, event.steps);
        assistantNode.__planSteps = Array.isArray(event.steps)
          ? event.steps.map(step => ({ ...step }))
          : [];
        agentTaskCount = event.steps?.length || 0;
        isAgentMode = true;
      }
      if (event.type === "plan_update") {
        updatePlanStep(assistantNode, event.step_id, event.status);
        assistantNode.__planSteps = (assistantNode.__planSteps || []).map(step => (
          step?.id === event.step_id ? { ...step, status: event.status } : step
        ));
      }
      if (event.type === "ask_user") {
        assistantNode.__pendingAskUser = {
          question: event.question || "",
          options: Array.isArray(event.options) ? event.options : [],
          context: event.context || "",
        };
        showAskUserPopup(event.question, event.options || [], event.context || "", async (answer) => {
          // Add the user's answer to thread history so the agent sees it
          thread.messages.push({ id: uid(), role: "user", content: answer, createdAt: now() });
          thread.updatedAt = now();
          if (canUseStoredThreads()) {
            if (auth.threadsSync) {
              await addMessageToApi(thread.id, "user", answer);
            } else {
              save();
            }
          }
          // Re-activate thinking panel for the continuation
          setThinkingState(assistantNode, true);
          updateThinkingLabel(assistantNode, "Continuing with your answer...");
          // Continue streaming into the SAME assistant bubble
          try {
            setComposerStreaming(true);
            if (assistantNode.__flow) {
              const f = assistantNode.__flow;
              setNS(f, "user", "done");
              setNS(f, "orchestrator", "active");
              setNS(f, "response", "active");
              setES(f, "u_o", "active");
              setES(f, "o_r", "active");
              applyFlow(assistantNode);
            }
            setThinkingState(assistantNode, true);
            appendLog(assistantNode, "Received your answer. Continuing the same run.");
            await streamChat(answer, assistantNode, thread, thinkingMode, [], { isContinuation: true });
          } catch (err) {
            if (err.name !== "AbortError") {
              const errContentEl = assistantNode.querySelector(".content");
              errContentEl.innerHTML += `<div class="error-msg">${escHtml(err.message)}</div>`;
            }
          } finally {
            state.abortController = null;
            setComposerStreaming(false);
          }
        });
      }
      if (event.type === "tool_start") {
        // Skip activity card for planning tool — the plan card handles it
        if (event.tool !== "todo_write") {
          appendToolCard(assistantNode, event.tool, event.args, "running");
          appendLog(assistantNode, `Running ${event.tool}...`);
        }
        assistantNode.__toolTrace.push({ tool_name: event.tool, status: "running" });
        // ── Agent→Flow bridge: highlight the right node ──
        if (assistantNode.__flow) {
          const f = assistantNode.__flow;
          if (event.tool === "todo_write") {
            // Planning tool — keep orchestrator active, don't highlight any external node
          } else if (_DB_TOOLS.has(event.tool)) {
            // Database tool → highlight database
            clearPopupLane(f, "docs");
            setNS(f, "orchestrator", "done");
            setNS(f, "database", "active");
            setES(f, "o_d", "active");
            setES(f, "d_o", "idle");
            applyFlow(assistantNode);
            // Search matrix column
            onSearchToolStart(assistantNode, event.tool, event.args);
          } else if (_CALC_TOOLS.has(event.tool)) {
            // Calc/engineering tool → highlight tools
            setNS(f, "orchestrator", "done");
            setNS(f, "tools", "active");
            setES(f, "o_t", "active");
            setES(f, "t_o", "idle");
            applyFlow(assistantNode);
            pushToolBadge(f, event.tool);
            // Stash args so tool_result can use them
            assistantNode.__pendingCalcArgs = event.args || {};
          } else {
            // Other tool → also highlight tools node
            setNS(f, "orchestrator", "done");
            setNS(f, "tools", "active");
            setES(f, "o_t", "active");
            applyFlow(assistantNode);
            pushToolBadge(f, event.tool);
          }
        }
      }
      if (event.type === "tool_result") {
        if (event.tool !== "todo_write") {
          updateToolCard(assistantNode, event.tool, event.result, event.status, event.summary);
          appendLog(assistantNode, `${event.tool}: ${event.status}${event.summary ? " — " + event.summary : ""}`);
        }
        // Update tool trace status
        const traceEntry = assistantNode.__toolTrace.findLast(t => t.tool_name === event.tool && t.status === "running");
        if (traceEntry) traceEntry.status = event.status === "error" ? "error" : "ok";
        // ── Agent→Flow bridge: mark node done, return to orchestrator ──
        if (assistantNode.__flow && event.tool !== "todo_write") {
          const f = assistantNode.__flow;
          const isDone = event.status !== "error";
          const st = isDone ? "done" : "error";
          if (_DB_TOOLS.has(event.tool)) {
            setNS(f, "database", st);
            setES(f, "o_d", st);
            setES(f, "d_o", st);
            // Push doc badges from search results
            let parsed = typeof event.result === "object" ? event.result : {};
            if (typeof event.result === "string") { try { parsed = JSON.parse(event.result); } catch {} }
            const clauses = parsed.clauses || [];
            const topDocs = clauses.slice(0, 5).map(c => ({
              doc_id: c.standard || "", clause_id: c.clause_id || "",
            }));
            if (topDocs.length) pushDocBadges(f, topDocs);
            // Collect clauses for "What I used" trace + expandable references
            for (const c of clauses) {
              assistantNode.__usedClauses.push(`${c.standard || "EC3"} — ${c.clause_id || "?"}: ${c.title || ""}`);
              // Store full clause for expandable references (deduplicate by clause_id)
              const refKey = `${c.standard || ""}:${c.clause_id || ""}`;
              if (!assistantNode.__refClauses.find(r => `${r.standard}:${r.clause_id}` === refKey)) {
                assistantNode.__refClauses.push({
                  standard: c.standard || "EC3",
                  clause_id: c.clause_id || "",
                  title: c.title || "",
                  text: c.text || "",
                  pointer: c.pointer || "",
                });
              }
            }
            // Search matrix population
            onSearchToolResult(assistantNode, event.tool, parsed);
          } else if (_CALC_TOOLS.has(event.tool)) {
            setNS(f, "tools", st);
            setES(f, "o_t", st);
            setES(f, "t_o", st);
            // Collect calc results for tabs below response
            if (isDone) {
              let calcParsed = typeof event.result === "object" ? event.result : {};
              if (typeof event.result === "string") { try { calcParsed = JSON.parse(event.result); } catch {} }
              assistantNode.__calcResults.push({
                tool: event.tool,
                args: assistantNode.__pendingCalcArgs || {},
                result: calcParsed,
                summary: event.summary || "",
              });
              assistantNode.__pendingCalcArgs = null;
            }
          } else {
            setNS(f, "tools", st);
            setES(f, "o_t", st);
            setES(f, "t_o", st);
          }
          // Re-activate orchestrator for next step
          setNS(f, "orchestrator", "active");
          applyFlow(assistantNode);
        }
      }

      if (event.type === "delta") {
        accumulated += event.delta || event.content || "";
        assistantNode.__accumulatedMd = accumulated;
        scheduleRender();
        // ── Agent→Flow bridge: first delta → light up response path ──
        if (assistantNode.__flow && !assistantNode.__deltaStarted) {
          assistantNode.__deltaStarted = true;
          const f = assistantNode.__flow;
          setNS(f, "orchestrator", "done");
          setNS(f, "response", "active");
          setES(f, "o_r", "active");
          applyFlow(assistantNode);
        }
      }

      // ── FEA events ──────────────────────────────────
      if (event.type === "fea_session_created") {
        assistantNode.__feaSessionId = event.session_id;
        if (!assistantNode.__feaPanelReady) {
          assistantNode.__feaPanelReady = initFEAPanel(assistantNode);
        }
      }

      if (event.type === "fea_state_restored") {
        if (!assistantNode.__feaPanelReady) {
          assistantNode.__feaPanelReady = initFEAPanel(assistantNode);
        }
        if (assistantNode.__feaPanelReady) {
          assistantNode.__feaPanelReady.then(() => {
            if (assistantNode.__feaPanel) {
              assistantNode.__feaPanel.restoreSessionSnapshot({
                model_snapshot: event.model_snapshot,
                results_snapshot: event.results_snapshot,
                model_summary: event.model_summary,
              }).catch(e => console.error("FEA restore error:", e));
            }
          }).catch(e => console.error("FEA panel init error on restore:", e));
        }
      }

      if (event.type === "fea_thinking") {
        appendThinkingCard(assistantNode, event.content);
        isAgentMode = true;
      }

      if (event.type === "fea_tool_call") {
        if (event.tool !== "todo_write") {
          appendToolCard(assistantNode, event.tool, event.args, "running");
          appendLog(assistantNode, `FEA: ${event.tool}(${JSON.stringify(event.args).slice(0, 80)}...)`);
        }
        assistantNode.__toolTrace.push({ tool_name: event.tool, status: "running" });
        isAgentMode = true;
      }

      if (event.type === "fea_tool_result") {
        if (event.tool !== "todo_write") {
          updateToolCard(assistantNode, event.tool, event.result, event.status, event.summary);
          appendLog(assistantNode, `FEA ${event.tool}: ${event.status}${event.summary ? " — " + event.summary : ""}`);
        }
        const traceEntry = assistantNode.__toolTrace.findLast(t => t.tool_name === event.tool && t.status === "running");
        if (traceEntry) traceEntry.status = event.status === "error" ? "error" : "ok";
        isAgentMode = true;
      }

      if (event.type === "fea_command") {
        // Initialize FEA panel lazily — must await before sending commands
        if (!assistantNode.__feaPanelReady) {
          assistantNode.__feaPanelReady = initFEAPanel(assistantNode);
        }
        if (event.commands) {
          assistantNode.__feaPanelReady.then(() => {
            if (assistantNode.__feaPanel) {
              assistantNode.__feaPanel.handleCommands(event.commands).catch(e => console.error("FEA handleCommands error:", e));
            }
          }).catch(e => console.error("FEA panel init error:", e));
        }
      }

      if (event.type === "fea_solve_request") {
        appendThinkingCard(assistantNode, "Solving structural model...");
        if (assistantNode.__feaPanelReady) {
          assistantNode.__feaPanelReady.then(() => {
            if (assistantNode.__feaPanel) {
              assistantNode.__feaPanel.handleSolveRequest(event).catch(e => console.error("FEA solve error:", e));
            }
          }).catch(e => console.error("FEA panel init error on solve:", e));
        } else {
          console.warn("FEA solve: no __feaPanelReady");
        }
      }

      if (event.type === "fea_view_command") {
        if (assistantNode.__feaPanelReady) {
          assistantNode.__feaPanelReady.then(() => {
            if (assistantNode.__feaPanel) {
              assistantNode.__feaPanel.handleViewCommand(event).catch(e => console.error("FEA view error:", e));
            }
          }).catch(e => console.error("FEA panel init error on view:", e));
        }
      }

      if (event.type === "fea_user_query") {
        assistantNode.__pendingAskUser = {
          question: event.question || "",
          options: Array.isArray(event.options) ? event.options : [],
          context: event.context || "",
        };
        setThinkingState(assistantNode, false);
        updateThinkingLabel(assistantNode, "Waiting for your answer.");
        showFEAQueryPopup(
          assistantNode,
          event.session_id,
          event.question,
          event.options || [],
          event.context || "",
          () => {
            assistantNode.__pendingAskUser = null;
            setThinkingState(assistantNode, true);
            updateThinkingLabel(assistantNode, "Continuing with your answer...");
            appendLog(assistantNode, "Received your answer. Continuing FEA analysis.");
          },
        );
      }

      if (event.type === "fea_complete") {
        const askOverlay = document.getElementById("ask-user-overlay");
        askOverlay?.classList.add("hidden");
        document.querySelector(".fea-query-overlay")?.remove();
        // Show the summary in the response area
        if (event.summary) {
          accumulated += event.summary;
          scheduleRender();
        }
        // Finalize thinking
        setThinkingState(assistantNode, false);
        const elapsed = ((Date.now() - (assistantNode.__thinkStart || Date.now())) / 1000).toFixed(1);
        const steps = assistantNode.__stepCount || 0;
        const meta = assistantNode.querySelector(".thinking-meta");
        if (meta) meta.textContent = `${steps} steps \u00B7 ${elapsed}s`;
        updateThinkingLabel(assistantNode, "FEA analysis complete. Expand to review steps.");

        const feaContent = accumulated || event.summary || "FEA analysis complete.";
        const panelSnapshot = assistantNode.__feaPanel?.getSessionSnapshot
          ? assistantNode.__feaPanel.getSessionSnapshot()
          : null;
        const baseSessionMemory = event.session_memory && typeof event.session_memory === "object"
          ? { ...event.session_memory }
          : { state: "final", fea_session: {} };
        if (!baseSessionMemory.state) baseSessionMemory.state = "final";
        if (!baseSessionMemory.answer_summary) baseSessionMemory.answer_summary = feaContent;
        if ((!Array.isArray(baseSessionMemory.plan) || !baseSessionMemory.plan.length) && assistantNode.__planSteps?.length) {
          baseSessionMemory.plan = assistantNode.__planSteps.map(step => ({ ...step }));
        }
        if (!Array.isArray(baseSessionMemory.assumptions) || !baseSessionMemory.assumptions.length) {
          baseSessionMemory.assumptions = Array.isArray(event.assumptions) ? [...event.assumptions] : [];
        }
        const feaSessionMemory = baseSessionMemory.fea_session && typeof baseSessionMemory.fea_session === "object"
          ? { ...baseSessionMemory.fea_session }
          : {};
        if (panelSnapshot?.model_snapshot) feaSessionMemory.model_snapshot = panelSnapshot.model_snapshot;
        if (panelSnapshot?.results_snapshot) feaSessionMemory.results_snapshot = panelSnapshot.results_snapshot;
        if (!feaSessionMemory.model_summary && panelSnapshot?.model_summary) {
          feaSessionMemory.model_summary = panelSnapshot.model_summary;
        }
        baseSessionMemory.fea_session = feaSessionMemory;
        const tracePayload = {
          answer: feaContent,
          assumptions: Array.isArray(event.assumptions) ? event.assumptions : [],
          tool_trace: assistantNode.__toolTrace || [],
          session_memory: baseSessionMemory,
        };
        setTrace(assistantNode, tracePayload);
        assistantNode.__pendingAskUser = null;

        // Save to thread
        if (!finalized) {
          thread.messages.push({
            id: uid(),
            role: "assistant",
            content: feaContent,
            responsePayload: tracePayload,
            createdAt: now(),
          });
          thread.updatedAt = now();
          if (canUseStoredThreads()) {
            if (auth.threadsSync) {
              await addMessageToApi(thread.id, "assistant", feaContent, tracePayload);
            } else {
              save();
            }
          }
          renderThreadList();
          finalized = true;
        }
        contentEl.classList.remove("streaming");
      }
      // ── End FEA events ──────────────────────────────

      if (event.type === "final") {
        if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
        contentEl.classList.remove("streaming");
        const payload = event.response;
        const pendingAskUser = assistantNode.__pendingAskUser || null;
        const waitingForUser = !!pendingAskUser;
        const rawAnswer = payload.answer || (waitingForUser ? formatPendingAskUserText(pendingAskUser) : "");
        const displayText = splitToolContextBlock(accumulated || rawAnswer).visible;
        lastPayload = payload;
        contentEl.innerHTML = renderMd(displayText);
        buildCalcTabs(assistantNode);
        buildReferences(assistantNode);
        // Build trace from accumulated tool data
        const tracePayload = {
          ...payload,
          what_i_used: assistantNode.__usedClauses || [],
          tool_trace: assistantNode.__toolTrace || [],
        };
        setTrace(assistantNode, tracePayload);
        // ── Agent→Flow bridge: finalize response ──
        if (assistantNode.__flow && !waitingForUser) {
          const f = assistantNode.__flow;
          const ok = payload.supported !== false;
          setNS(f, "orchestrator", "done");
          setNS(f, "response", ok ? "done" : "error");
          setES(f, "u_o", "done");
          setES(f, "o_r", ok ? "done" : "error");
          setNS(f, "user", "done");
          applyFlow(assistantNode);
        }
        if (waitingForUser) {
          if (isAgentMode) {
            pauseAgentThinking(assistantNode, agentTaskCount || 1);
          } else {
            pauseThinking(assistantNode);
          }
        } else if (isAgentMode) {
          finalizeAgentThinking(assistantNode, agentTaskCount || 1);
        } else {
          finalizeThinking(assistantNode, payload);
        }
        appendLog(assistantNode, waitingForUser ? "Waiting for your answer." : "Response complete.");

        if (!finalized) {
          const toolCtx = event.tool_context || "";
          const eventSessionMemory = event.session_memory && typeof event.session_memory === "object"
            ? event.session_memory
            : {};
          const storedPayload = {
            ...payload,
            answer: displayText,
            session_memory: {
              ...eventSessionMemory,
              tool_context: toolCtx || eventSessionMemory.tool_context || "",
              state: waitingForUser ? "waiting_for_user" : (eventSessionMemory.state || "final"),
              ask_user: pendingAskUser || eventSessionMemory.ask_user || null,
            },
          };
          if (isContinuation) {
            const lastAst = [...thread.messages].reverse().find(m => m.role === "assistant");
            if (lastAst) {
              lastAst.content = displayText;
              lastAst.responsePayload = storedPayload;
              lastAst.updatedAt = now();
            } else {
              thread.messages.push({
                id: uid(),
                role: "assistant",
                content: displayText,
                responsePayload: storedPayload,
                createdAt: now(),
              });
            }
          } else {
            thread.messages.push({
              id: uid(),
              role: "assistant",
              content: displayText,
              responsePayload: storedPayload,
              createdAt: now(),
            });
          }
          thread.updatedAt = now();
          if (canUseStoredThreads()) {
            if (auth.threadsSync) {
              if (isContinuation) {
                save();
              } else {
                await addMessageToApi(thread.id, "assistant", displayText, storedPayload);
              }
            } else {
              save();
            }
          }
          renderThreadList();
          finalized = true;
        }
        assistantNode.__pendingAskUser = null;
      }

      if (event.type === "error") {
        if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
        contentEl.classList.remove("streaming");
        const errMsg = `Error: ${event.detail || "Unknown error"}`;
        // ── Agent→Flow bridge: error state ──
        if (assistantNode.__flow) {
          const f = assistantNode.__flow;
          f.ns["orchestrator"] = "error";
          f.ns["response"] = "error";
          f.es["o_r"] = "error";
          applyFlow(assistantNode);
        }
        setThinkingState(assistantNode, false);
        updateThinkingLabel(assistantNode, "Reasoning failed.");
        contentEl.innerHTML = `<div class="error-msg">${escHtml(errMsg)}</div>`;
        appendLog(assistantNode, errMsg);
        if (!finalized) {
          thread.messages.push({ id: uid(), role: "assistant", content: errMsg, responsePayload: null, createdAt: now() });
          thread.updatedAt = now();
          if (canUseStoredThreads()) {
            if (auth.threadsSync) {
              await addMessageToApi(thread.id, "assistant", errMsg, null);
            } else {
              save();
            }
          }
          renderThreadList();
          finalized = true;
        }
      }

      // Context usage update (circle indicator)
      if (event.type === "context_usage") {
        renderContextUsage(event);
      }
    }
  }
}

// ---- Auth ----
const auth = {
  token: null,
  user: null,
  refreshToken: null,
  expiresAt: null,
  enabled: true,
  threadsSync: false,
  ready: false,
  mode: "login",
};

function syncAuthControls() {
  const signedIn = !!auth.user;
  const showAuthArea = auth.ready && auth.enabled && !signedIn;

  const authArea = $("#sidebar-auth-area");
  const userArea = $("#sidebar-user-area");
  if (authArea) authArea.classList.toggle("hidden", !showAuthArea);
  if (userArea) userArea.classList.toggle("hidden", showAuthArea);

  if (chatSearch) {
    const canSearch = canUseStoredThreads();
    chatSearch.disabled = !canSearch;
    chatSearch.placeholder = !auth.ready
      ? "Loading chats..."
      : (!auth.enabled
        ? "History unavailable without auth"
        : (chatSearch.disabled ? "Sign in to search saved chats..." : "Search chats..."));
    if (chatSearch.disabled) {
      chatSearch.value = "";
      state.filter = "";
    }
  }
}

async function applyAuthState() {
  if (canUseStoredThreads()) {
    await load();
    await ensureThread();
  } else {
    resetThreadState();
    resetGuestThread();
  }
  updateUserPill();
  syncAuthControls();
  renderThreadList();
  renderMessages();
}

async function checkAuthStatus() {
  auth.ready = false;
  try {
    const res = await fetch("/api/auth/status");
    const data = await res.json();
    auth.enabled = data.enabled === true;
    auth.threadsSync = data.threads_sync === true;
  } catch { auth.enabled = false; auth.threadsSync = false; }

  if (!auth.enabled) {
    auth.ready = true;
    return;
  }

  const saved = sessionStorage.getItem("ec3_auth");
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      auth.token = parsed.access_token || null;
      auth.user = { user_id: parsed.user_id, email: parsed.email };
      auth.refreshToken = parsed.refresh_token || null;
      auth.expiresAt = parsed.expires_at || null;
      await maybeRefreshSession();
    } catch {
      sessionStorage.removeItem("ec3_auth");
    }
  }
  auth.ready = true;
}

function persistAuth(data) {
  auth.token = data.access_token;
  auth.refreshToken = data.refresh_token ?? auth.refreshToken;
  auth.expiresAt = data.expires_at ?? auth.expiresAt;
  if (data.user_id || data.email) {
    auth.user = { ...auth.user, user_id: data.user_id || auth.user?.user_id, email: data.email || auth.user?.email };
  }
  const toStore = {
    access_token: auth.token,
    refresh_token: auth.refreshToken,
    expires_at: auth.expiresAt,
    user_id: auth.user?.user_id,
    email: auth.user?.email,
  };
  sessionStorage.setItem("ec3_auth", JSON.stringify(toStore));
}

function isTokenExpiringSoon() {
  if (!auth.expiresAt) return false;
  const secLeft = auth.expiresAt - Math.floor(Date.now() / 1000);
  return secLeft < 300;
}

async function refreshSession() {
  if (!auth.refreshToken) return false;
  try {
    const res = await fetch("/api/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: auth.refreshToken }),
    });
    if (!res.ok) return false;
    const data = await res.json();
    persistAuth({
      access_token: data.access_token,
      refresh_token: data.refresh_token ?? auth.refreshToken,
      expires_at: data.expires_at,
      user_id: auth.user?.user_id,
      email: auth.user?.email,
    });
    return true;
  } catch {
    return false;
  }
}

async function maybeRefreshSession() {
  if (isTokenExpiringSoon() && auth.refreshToken) {
    await refreshSession();
  }
}

async function fetchWithAuth(url, opts = {}) {
  await maybeRefreshSession();
  opts.headers = { ...authHeaders(), ...opts.headers };
  let res = await fetch(url, opts);
  if (res.status === 401 && auth.refreshToken) {
    const ok = await refreshSession();
    if (ok) {
      opts.headers = { ...authHeaders(), ...opts.headers };
      res = await fetch(url, opts);
    }
  }
  return res;
}

function showAuthOverlay(mode = "login") {
  if (!auth.enabled) return;
  switchAuthTab(mode);
  $("#auth-error")?.classList.add("hidden");
  $("#auth-overlay")?.classList.remove("hidden");
  if (mode === "login") $("#auth-email-login")?.focus();
  else $("#auth-email-signup")?.focus();
}

function hideAuthOverlay() {
  $("#auth-overlay")?.classList.add("hidden");
}

function updateSidebarUser() {
  const avatarEl = $("#sidebar-user-avatar");
  const nameEl = $("#sidebar-user-name");
  if (!auth.user) return;

  const email = auth.user.email || "";
  const name = email.split("@")[0] || "User";
  const initials = name.slice(0, 2).toUpperCase();

  if (avatarEl) avatarEl.textContent = initials;
  if (nameEl) nameEl.textContent = name;
}

function updateUserPill() {
  updateSidebarUser();
}

function initAuth() {
  const overlay = $("#auth-overlay");
  const errorEl = $("#auth-error");
  const closeBtn = $("#auth-close");
  const tabLogin = $("#auth-tab-login");
  const tabSignup = $("#auth-tab-signup");
  const formLogin = $("#auth-form-login");
  const formSignup = $("#auth-form-signup");
  const forgotLink = $("#auth-forgot-password");

  // Sidebar: open auth modal
  signInBtn?.addEventListener("click", () => showAuthOverlay("login"));
  registerBtn?.addEventListener("click", () => showAuthOverlay("signup"));
  sidebarSigninBtn?.addEventListener("click", () => showAuthOverlay("login"));
  sidebarSignupBtn?.addEventListener("click", () => showAuthOverlay("signup"));

  // Sidebar user area: toggle menu & logout (event delegation)
  sidebar?.addEventListener("click", (e) => {
    if (e.target.closest("#sidebar-user-btn")) {
      if (e.target.closest("#sidebar-logout-btn")) return;
      const btn = $("#sidebar-user-btn");
      const menu = $("#sidebar-user-menu");
      const expanded = btn?.getAttribute("aria-expanded") === "true";
      btn?.setAttribute("aria-expanded", !expanded);
      menu?.classList.toggle("hidden", expanded);
    }
    if (e.target.closest("#sidebar-logout-btn")) {
      (async () => {
        try { await fetch("/api/auth/logout", { method: "POST" }); } catch {}
        sessionStorage.removeItem("ec3_auth");
        auth.token = null;
        auth.user = null;
        auth.refreshToken = null;
        auth.expiresAt = null;
        $("#sidebar-user-menu")?.classList.add("hidden");
        hideAuthOverlay();
        applyAuthState();
      })();
    }
  });

  // Auth modal: close
  closeBtn?.addEventListener("click", hideAuthOverlay);
  overlay?.addEventListener("click", (e) => {
    if (e.target === overlay) hideAuthOverlay();
  });

  // Auth tabs
  tabLogin?.addEventListener("click", () => switchAuthTab("login"));
  tabSignup?.addEventListener("click", () => switchAuthTab("signup"));

  // Forgot password
  forgotLink?.addEventListener("click", (e) => {
    e.preventDefault();
    handleForgotPassword();
  });

  // Login form
  formLogin?.addEventListener("submit", (e) => {
    e.preventDefault();
    handleAuthSubmit("login");
  });

  // Signup form
  formSignup?.addEventListener("submit", (e) => {
    e.preventDefault();
    handleAuthSubmit("signup");
  });
}

function switchAuthTab(mode) {
  auth.mode = mode;
  const isLogin = mode === "login";
  $("#auth-tab-login")?.classList.toggle("active", isLogin);
  $("#auth-tab-signup")?.classList.toggle("active", !isLogin);
  $("#auth-form-login")?.classList.toggle("hidden", !isLogin);
  $("#auth-form-signup")?.classList.toggle("hidden", isLogin);
  const errEl = $("#auth-error");
  errEl?.classList.add("hidden");
  errEl?.classList.remove("auth-success");
}

async function handleForgotPassword() {
  const emailInput = $("#auth-email-login");
  const email = emailInput?.value?.trim();
  if (!email) {
    $("#auth-error").textContent = "Enter your email address.";
    $("#auth-error").classList.remove("hidden");
    return;
  }
  const btn = $("#auth-submit-login");
  const btnText = btn?.querySelector(".auth-btn-text");
  const spinner = btn?.querySelector(".auth-btn-spinner");
  btnText && (btnText.textContent = "Sending...");
  spinner?.classList.remove("hidden");
  btn?.setAttribute("disabled", "true");
  const errEl = $("#auth-error");
  try {
    const res = await fetch("/api/auth/forgot-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });
    const data = await res.json().catch(() => ({}));
    errEl.textContent = data.message || "If an account exists, you will receive a reset link.";
    errEl.classList.remove("hidden");
    errEl.classList.add("auth-success");
  } catch (err) {
    errEl.textContent = err.message || "Failed to send reset email.";
    errEl.classList.remove("hidden");
    errEl.classList.remove("auth-success");
  } finally {
    btnText && (btnText.textContent = "Sign in");
    spinner?.classList.add("hidden");
    btn?.removeAttribute("disabled");
  }
}

async function handleAuthSubmit(mode) {
  const isLogin = mode === "login";
  const emailInput = isLogin ? $("#auth-email-login") : $("#auth-email-signup");
  const passInput = isLogin ? $("#auth-password-login") : $("#auth-password-signup");
  const passConfirm = $("#auth-password-confirm");
  const submitBtn = isLogin ? $("#auth-submit-login") : $("#auth-submit-signup");
  const btnText = submitBtn?.querySelector(".auth-btn-text");
  const spinner = submitBtn?.querySelector(".auth-btn-spinner");

  const email = emailInput?.value?.trim();
  const password = passInput?.value;
  if (!email || !password) return;

  if (!isLogin) {
    const confirmVal = passConfirm?.value;
    if (password !== confirmVal) {
      $("#auth-error").textContent = "Passwords do not match.";
      $("#auth-error").classList.remove("hidden");
      return;
    }
  }

  $("#auth-error")?.classList.add("hidden");
  submitBtn?.setAttribute("disabled", "true");
  btnText && (btnText.classList.add("hidden"));
  spinner?.classList.remove("hidden");

  const endpoint = isLogin ? "/api/auth/login" : "/api/auth/signup";
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Auth failed");
    if (!data.access_token) throw new Error("Account created. Confirm your email, then sign in.");

    auth.token = data.access_token;
    auth.user = { user_id: data.user_id, email: data.email };
    auth.refreshToken = data.refresh_token || null;
    auth.expiresAt = data.expires_at || null;
    persistAuth(data);
    applyAuthState();
    hideAuthOverlay();
  } catch (err) {
    $("#auth-error").textContent = err.message;
    $("#auth-error").classList.remove("hidden");
  } finally {
    submitBtn?.removeAttribute("disabled");
    btnText?.classList.remove("hidden");
    spinner?.classList.add("hidden");
  }
}

function authHeaders() {
  if (auth.token) return { Authorization: `Bearer ${auth.token}` };
  return {};
}

async function handleAuthRedirectFromHash() {
  const hash = window.location.hash;
  if (!hash) return;
  const params = new URLSearchParams(hash.slice(1));
  const accessToken = params.get("access_token");
  if (!accessToken) return;

  const refreshToken = params.get("refresh_token") || "";
  const expiresAt = params.get("expires_at");
  let user = { user_id: params.get("user_id") || "", email: params.get("email") || "" };

  auth.token = accessToken;
  auth.refreshToken = refreshToken;
  auth.expiresAt = expiresAt ? parseInt(expiresAt, 10) : null;
  auth.user = user;

  if (!user.user_id || !user.email) {
    try {
      const res = await fetch("/api/auth/me", { headers: { Authorization: `Bearer ${accessToken}` } });
      if (res.ok) {
        const data = await res.json();
        user = { user_id: data.user_id || "", email: data.email || "" };
        auth.user = user;
      }
    } catch {}
  }

  const toStore = {
    access_token: accessToken,
    refresh_token: refreshToken,
    expires_at: auth.expiresAt,
    user_id: user.user_id,
    email: user.email,
  };
  sessionStorage.setItem("ec3_auth", JSON.stringify(toStore));

  window.history.replaceState(null, "", window.location.pathname + window.location.search);
  applyAuthState();
}


// ---- Init ----
async function initialize() {
  resetGuestThread();
  renderThreadList();
  renderMessages();
  initAuth();
  syncAuthControls();
  loadThinkingModePreference();

  fetch("/api/tools").then(r => r.json()).then(tools => {
    const el = document.getElementById("tool-count");
    if (el) el.textContent = `${tools.length} tools available`;
  }).catch(() => {});

  thinkingModeSelect?.addEventListener("change", (e) => {
    setThinkingMode(e.target.value);
  });
  thinkingModeTrigger?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (thinkingModeTrigger.disabled || !thinkingModeMenu) return;
    closeAttachMenu();
    const open = thinkingModeMenu.classList.contains("hidden");
    if (open) {
      thinkingModeMenu.classList.remove("hidden");
      thinkingModeTrigger.setAttribute("aria-expanded", "true");
      thinkingModeTrigger.closest(".thinking-mode-wrap")?.classList.add("open");
      return;
    }
    closeThinkingModeMenu();
  });
  thinkingModeMenu?.addEventListener("click", (e) => {
    e.stopPropagation();
  });
  for (const option of $$(".thinking-mode-option")) {
    option.addEventListener("click", () => {
      setThinkingMode(option.dataset.mode || "thinking");
      closeThinkingModeMenu();
    });
  }
  document.addEventListener("click", (e) => {
    const wrap = thinkingModeTrigger?.closest(".thinking-mode-wrap");
    if (!wrap) return;
    if (!(e.target instanceof Node)) return;
    if (!wrap.contains(e.target)) closeThinkingModeMenu();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeThinkingModeMenu();
      closeAttachMenu();
    }
  });

  // ---- Attach button ----
  attachTrigger?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (!attachMenu) return;
    const isOpen = !attachMenu.classList.contains("hidden");
    closeThinkingModeMenu();
    if (isOpen) {
      closeAttachMenu();
    } else {
      attachMenu.classList.remove("hidden");
      attachTrigger.setAttribute("aria-expanded", "true");
      attachTrigger.closest(".attach-wrap")?.classList.add("open");
    }
  });
  attachMenu?.addEventListener("click", (e) => {
    e.stopPropagation();
  });
  for (const opt of $$(".attach-option")) {
    opt.addEventListener("click", () => {
      const type = opt.dataset.type;
      closeAttachMenu();
      if (type === "photo") {
        photoInput?.click();
      } else if (type === "file") {
        fileInput?.click();
      }
    });
  }
  photoInput?.addEventListener("change", () => {
    if (photoInput.files?.length) addAttachments(photoInput.files);
    photoInput.value = "";
  });
  fileInput?.addEventListener("change", () => {
    if (fileInput.files?.length) addAttachments(fileInput.files);
    fileInput.value = "";
  });
  document.addEventListener("click", (e) => {
    const wrap = attachTrigger?.closest(".attach-wrap");
    if (wrap && e.target instanceof Node && !wrap.contains(e.target)) closeAttachMenu();
  });

  // Drag-and-drop on the composer
  const composerEl = $(".composer");
  if (composerEl) {
    let dragCounter = 0;
    composerEl.addEventListener("dragenter", (e) => {
      e.preventDefault();
      dragCounter++;
      composerEl.style.borderColor = "var(--accent)";
      composerEl.style.boxShadow = "0 0 0 2px var(--accent-glow), 0 8px 24px rgba(0,0,0,0.25)";
    });
    composerEl.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        composerEl.style.borderColor = "";
        composerEl.style.boxShadow = "";
      }
    });
    composerEl.addEventListener("dragover", (e) => {
      e.preventDefault();
    });
    composerEl.addEventListener("drop", (e) => {
      e.preventDefault();
      dragCounter = 0;
      composerEl.style.borderColor = "";
      composerEl.style.boxShadow = "";
      if (e.dataTransfer?.files?.length) {
        addAttachments(e.dataTransfer.files);
      }
    });
  }

  await handleAuthRedirectFromHash();

  newChatBtn.addEventListener("click", async () => {
    await createThread();
    renderThreadList();
    renderMessages();
    input.focus();
  });

  chatSearch.addEventListener("input", e => {
    state.filter = e.target.value || "";
    renderThreadList();
  });

  sidebarToggle?.addEventListener("click", () => {
    sidebar.classList.toggle("sidebar-open");
  });

  for (const btn of $$(".example-btn")) {
    btn.addEventListener("click", () => {
      const p = btn.dataset.prompt;
      if (p) { input.value = p; form.requestSubmit(); }
    });
  }

  checkAuthStatus().finally(() => {
    applyAuthState();
  });

  setInterval(() => maybeRefreshSession(), 45 * 60 * 1000);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    // If streaming, treat submit as stop
    if (state.abortController) {
      state.abortController.abort();
      return;
    }
    const prompt = input.value.trim();
    const hasAttachments = state.attachments.length > 0;
    if (!prompt && !hasAttachments) return;
    const thinkingMode = thinkingModeSelect?.value || state.thinkingMode;
    setThinkingMode(thinkingMode);

    // Capture attachments before clearing
    const currentAttachments = state.attachments.map(a => ({
      id: a.id, name: a.name, size: a.size, type: a.type, isImage: a.isImage, dataUrl: a.dataUrl,
    }));

    // Build a prompt that includes file context for the AI
    let fullPrompt = prompt;
    if (currentAttachments.length) {
      const fileDescs = currentAttachments.map(a =>
        a.isImage ? `[Attached image: ${a.name}]` : `[Attached file: ${a.name} (${fmtFileSize(a.size)})]`
      ).join(" ");
      fullPrompt = fullPrompt ? `${fileDescs}\n\n${fullPrompt}` : fileDescs;
    }

    const thread = await ensureThread();
    if (canUseStoredThreads() && (!thread.messages.length || thread.title === "New chat")) {
      thread.title = truncTitle(prompt || currentAttachments[0]?.name || "New chat");
      if (auth.threadsSync) {
        await updateThreadTitleApi(thread.id, thread.title);
        const idx = state.threads.findIndex((t) => t.id === thread.id);
        if (idx >= 0) state.threads[idx].title = thread.title;
      }
    } else if (!canUseStoredThreads() && (!thread.messages.length || thread.title === "Temporary chat")) {
      thread.title = truncTitle(prompt || currentAttachments[0]?.name || "Temporary chat");
    }

    thread.messages.push({ id: uid(), role: "user", content: prompt, attachments: currentAttachments, createdAt: now() });
    thread.updatedAt = now();
    if (canUseStoredThreads()) {
      if (auth.threadsSync) {
        await addMessageToApi(thread.id, "user", fullPrompt);
      } else {
        save();
      }
    }
    renderThreadList();

    input.value = "";
    clearAttachments();
    closeAttachMenu();
    setComposerStreaming(true);
    createMsg("user", prompt, { attachments: currentAttachments });
    _cleanupFloatingDiagrams();
    const assistantNode = createMsg("assistant", "", { showThinking: true, prompt: fullPrompt });
    updateWelcome();

    try {
      await streamChat(fullPrompt, assistantNode, thread, state.thinkingMode, currentAttachments);
    } catch (err) {
      if (err.name === "AbortError") {
        setThinkingState(assistantNode, false);
        updateThinkingLabel(assistantNode, "Stopped by user.");
        appendLog(assistantNode, "Stopped by user.");
        // Clean up diagram popup
        const abortFlow = assistantNode.__flow;
        if (abortFlow?.diagramPanel) abortFlow.diagramPanel.classList.add("hidden");
        const contentEl = assistantNode.querySelector(".content");
        contentEl.classList.remove("streaming");
        const partial = contentEl.innerHTML;
        if (!partial || partial === "<p></p>") {
          contentEl.innerHTML = '<div class="error-msg">Stopped.</div>';
        }
        thread.messages.push({ id: uid(), role: "assistant", content: contentEl.textContent || "Stopped.", responsePayload: null, createdAt: now() });
        thread.updatedAt = now();
      } else {
        const errMsg = `Error: ${err.message}`;
        setThinkingState(assistantNode, false);
        updateThinkingLabel(assistantNode, "Reasoning failed.");
        const errContentEl = assistantNode.querySelector(".content");
        errContentEl.classList.remove("streaming");
        errContentEl.innerHTML = `<div class="error-msg">${escHtml(errMsg)}</div>`;
        appendLog(assistantNode, `Transport error: ${err.message}`);
        thread.messages.push({ id: uid(), role: "assistant", content: errMsg, responsePayload: null, createdAt: now() });
        thread.updatedAt = now();
      }
      if (canUseStoredThreads()) {
        if (auth.threadsSync) {
          const lastMsg = thread.messages[thread.messages.length - 1];
          await addMessageToApi(thread.id, "assistant", lastMsg.content, null);
        } else {
          save();
        }
      }
      renderThreadList();
    } finally {
      state.abortController = null;
      setComposerStreaming(false);
      input.focus();
    }
  });
}

initialize();

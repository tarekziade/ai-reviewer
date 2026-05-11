(() => {
  // URL shape: /reviews/{owner}/{repo}/{number}/{jobId}
  const pathMatch = window.location.pathname.match(
    /^\/reviews\/([^/]+)\/([^/]+)\/(\d+)\/([a-f0-9]+)/
  );
  if (!pathMatch) {
    document.body.textContent = "Invalid review URL.";
    return;
  }
  const owner = pathMatch[1];
  const repo = pathMatch[2];
  const number = pathMatch[3];
  const jobId = pathMatch[4];
  const apiBase = `/reviews/${owner}/${repo}/${number}/${jobId}`;
  const consoleEl = document.getElementById("console");
  const statusEl = document.getElementById("status");
  const targetEl = document.getElementById("target");
  const banner = document.getElementById("error-banner");
  const editSection = document.getElementById("edit-section");
  const draftMeta = document.getElementById("draft-meta");
  const summaryEl = document.getElementById("summary");
  const eventEl = document.getElementById("event");
  const commentsListEl = document.getElementById("comments-list");
  const commentsCountEl = document.getElementById("comments-count");
  const publishBtn = document.getElementById("publish-btn");
  const discardBtn = document.getElementById("discard-btn");

  let draft = null;
  let infoCache = null;

  // Per-job localStorage key. Stores the user's in-progress edits so a
  // refresh doesn't wipe their work. Cleared on publish or discard.
  const editsKey = `serge:edits:${jobId}`;

  function loadStoredEdits() {
    try {
      const raw = localStorage.getItem(editsKey);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }
  function saveEdits() {
    if (!draft) return;
    try {
      localStorage.setItem(editsKey, JSON.stringify(collectEdits()));
    } catch {
      // quota exceeded / storage disabled — silently fall through
    }
  }
  function clearStoredEdits() {
    try {
      localStorage.removeItem(editsKey);
    } catch {}
  }

  const STEP_ORDER = ["clone", "fetch", "context", "llm", "done"];
  const progressFill = document.getElementById("progress-fill");
  const llmDetailEl = document.getElementById("llm-detail");
  const tokInEl = document.getElementById("tok-in");
  const tokOutEl = document.getElementById("tok-out");
  const tokRateEl = document.getElementById("tok-rate");
  const tokToolsEl = document.getElementById("tok-tools");
  const tokElapsedEl = document.getElementById("tok-elapsed");
  const rerunBtn = document.getElementById("rerun-btn");

  function formatNumber(n) {
    return n.toLocaleString();
  }

  function handleMetrics(raw) {
    let m;
    try {
      m = typeof raw === "string" ? JSON.parse(raw) : raw;
    } catch {
      return;
    }
    if (typeof m.in === "number") tokInEl.textContent = formatNumber(m.in);
    if (typeof m.out === "number") tokOutEl.textContent = formatNumber(m.out);
    if (typeof m.rate === "number") tokRateEl.textContent = m.rate.toFixed(1);
    if (typeof m.tools === "number") tokToolsEl.textContent = formatNumber(m.tools);
    if (typeof m.seconds === "number") tokElapsedEl.textContent = m.seconds.toFixed(1);
  }

  function showError(msg) {
    banner.textContent = msg;
    banner.classList.add("visible");
  }

  function setStatus(s) {
    statusEl.textContent = s;
    statusEl.className = `status-badge ${s}`;
  }

  function handleStep(raw) {
    // Values: "clone", "fetch", "context", "llm", "llm:N/M", "done", "error"
    if (raw === "error") {
      // Mark the current active step as errored.
      const active = document.querySelector(".progress .step.active");
      if (active) {
        active.classList.remove("active");
        active.classList.add("error");
      }
      progressFill.classList.add("error");
      return;
    }
    let [name, detail] = raw.split(":");
    if (!STEP_ORDER.includes(name)) return;
    const idx = STEP_ORDER.indexOf(name);
    for (let i = 0; i < STEP_ORDER.length; i++) {
      const el = document.querySelector(
        `.progress .step[data-step="${STEP_ORDER[i]}"]`
      );
      if (!el) continue;
      el.classList.remove("pending", "active", "done");
      if (i < idx) el.classList.add("done");
      else if (i === idx) el.classList.add(name === "done" ? "done" : "active");
      else el.classList.add("pending");
    }
    if (name === "llm" && detail) {
      llmDetailEl.textContent = `· turn ${detail}`;
    } else if (name !== "llm") {
      // Leave the LLM detail if we've moved past it; clear if not yet there.
      if (idx < STEP_ORDER.indexOf("llm")) llmDetailEl.textContent = "";
    }
    // Width: number of "done" or "active" steps, normalized.
    const doneCount = idx + (name === "done" ? 1 : 0);
    const pct = Math.min(100, (doneCount / (STEP_ORDER.length - 1)) * 100);
    progressFill.style.width = `${pct}%`;
  }

  // Tracks whether the previous console line was terminated. Streamed
  // tokens/reasoning come without newlines so a tool/log/error event
  // landing right after them would glue onto the end of the partial
  // line ("Let me check things.⚙ grep(...)"). We insert a leading
  // newline in that case.
  let consoleAtLineStart = true;

  function appendConsole(kind, text) {
    const span = document.createElement("span");
    span.className = kind;
    const streamed = kind === "token" || kind === "reasoning";
    let prefix = streamed || consoleAtLineStart ? "" : "\n";
    if (kind === "log") prefix += "› ";
    else if (kind === "tool") prefix += "⚙ ";
    else if (kind === "error") prefix += "✗ ";
    const body = prefix + text + (streamed ? "" : "\n");
    span.textContent = body;
    consoleEl.appendChild(span);
    consoleEl.scrollTop = consoleEl.scrollHeight;
    consoleAtLineStart = body.endsWith("\n");
  }

  async function loadInfo() {
    const r = await fetch(`${apiBase}/info`);
    if (r.status === 404) {
      showError(
        "This review job is no longer in the server's memory — it was either " +
        "garbage-collected (1h TTL) or the server has restarted since you " +
        "started it. Start a new review."
      );
      return null;
    }
    if (!r.ok) {
      showError(`Could not load job info: ${r.status}`);
      return null;
    }
    const info = await r.json();
    targetEl.textContent = info.target;
    setStatus(info.status);
    infoCache = info;
    return info;
  }

  async function loadDraft() {
    const r = await fetch(`${apiBase}/draft`);
    if (!r.ok) {
      showError(`Could not load draft: ${r.status}`);
      return;
    }
    const data = await r.json();
    setStatus(data.status);
    if (data.error) {
      showError(data.error);
    }
    if (!data.draft) {
      return;
    }
    draft = data.draft;
    renderDraftForm();
  }

  function renderDraftForm() {
    if (!draft) return;
    draftMeta.textContent = `${draft.owner}/${draft.repo}#${draft.number} · ${draft.metrics_line || ""}`;
    summaryEl.value = draft.summary || "";
    eventEl.value = draft.event || "COMMENT";
    commentsCountEl.textContent = draft.comments.length;
    commentsListEl.innerHTML = "";
    for (const c of draft.comments) {
      commentsListEl.appendChild(buildCommentCard(c));
    }
    applyStoredEdits();
    editSection.style.display = "";

    // If the job is already in a terminal state (e.g. user reloaded
    // after publishing), keep the draft visible but disable the action
    // buttons so they don't double-submit.
    const finished =
      infoCache && (infoCache.status === "published" || infoCache.status === "discarded");
    publishBtn.disabled = !!finished;
    discardBtn.disabled = !!finished;
    if (finished) {
      publishBtn.textContent =
        infoCache.status === "published" ? "Already published" : "Already discarded";
    }
  }

  function buildCommentCard(c) {
    const card = document.createElement("div");
    card.className = "comment-card";
    card.dataset.id = c.id;

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${c.path}:${c.line} (${c.side})`;
    card.appendChild(meta);

    if (c.diff_hunk && c.diff_hunk.length) {
      card.appendChild(buildDiffSnippet(c.diff_hunk));
    }

    const ta = document.createElement("textarea");
    ta.rows = 4;
    ta.value = c.body;
    ta.dataset.id = c.id;
    ta.dataset.role = "body";
    ta.addEventListener("input", saveEdits);
    card.appendChild(ta);

    const discardRow = document.createElement("div");
    discardRow.className = "discard-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = `discard-${c.id}`;
    cb.dataset.id = c.id;
    cb.dataset.role = "discard";
    cb.addEventListener("change", () => {
      card.classList.toggle("discarded", cb.checked);
      ta.disabled = cb.checked;
      saveEdits();
    });
    const lbl = document.createElement("label");
    lbl.htmlFor = cb.id;
    lbl.textContent = "Discard this comment";
    discardRow.appendChild(cb);
    discardRow.appendChild(lbl);
    card.appendChild(discardRow);

    return card;
  }

  function buildDiffSnippet(hunk) {
    const table = document.createElement("table");
    table.className = "diff-snippet";
    for (const line of hunk) {
      const tr = document.createElement("tr");
      tr.className = "diff-line";
      if (line.op === "+") tr.classList.add("add");
      else if (line.op === "-") tr.classList.add("del");
      else tr.classList.add("context");
      if (line.is_target) tr.classList.add("target");

      const old = document.createElement("td");
      old.className = "lineno";
      old.textContent = line.old != null ? line.old : "";
      const nw = document.createElement("td");
      nw.className = "lineno";
      nw.textContent = line.new != null ? line.new : "";
      const op = document.createElement("td");
      op.className = "op";
      op.textContent = line.op;
      const text = document.createElement("td");
      text.className = "text";
      text.textContent = line.text;

      tr.appendChild(old);
      tr.appendChild(nw);
      tr.appendChild(op);
      tr.appendChild(text);
      table.appendChild(tr);
    }
    return table;
  }

  function applyStoredEdits() {
    const stored = loadStoredEdits();
    if (!stored) return;
    if (typeof stored.summary === "string") summaryEl.value = stored.summary;
    if (stored.event) eventEl.value = stored.event;
    const discarded = new Set(stored.discarded_comment_ids || []);
    const overrides = stored.comment_overrides || {};
    for (const c of draft.comments) {
      const ta = commentsListEl.querySelector(
        `textarea[data-role="body"][data-id="${c.id}"]`
      );
      const cb = commentsListEl.querySelector(
        `input[data-role="discard"][data-id="${c.id}"]`
      );
      if (cb && discarded.has(c.id)) {
        cb.checked = true;
        cb.dispatchEvent(new Event("change"));
      }
      if (ta && Object.prototype.hasOwnProperty.call(overrides, c.id)) {
        ta.value = overrides[c.id];
      }
    }
  }

  summaryEl.addEventListener("input", saveEdits);
  eventEl.addEventListener("change", saveEdits);

  function collectEdits() {
    const summary = summaryEl.value;
    const event = eventEl.value;
    const comment_overrides = {};
    const discarded_comment_ids = [];
    for (const c of draft.comments) {
      const ta = commentsListEl.querySelector(
        `textarea[data-role="body"][data-id="${c.id}"]`
      );
      const cb = commentsListEl.querySelector(
        `input[data-role="discard"][data-id="${c.id}"]`
      );
      if (cb && cb.checked) {
        discarded_comment_ids.push(c.id);
        continue;
      }
      if (ta && ta.value !== c.body) {
        comment_overrides[c.id] = ta.value;
      }
    }
    return { summary, event, comment_overrides, discarded_comment_ids };
  }

  publishBtn.addEventListener("click", async () => {
    publishBtn.disabled = true;
    discardBtn.disabled = true;
    banner.classList.remove("visible");
    try {
      const r = await fetch(`${apiBase}/publish`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectEdits()),
      });
      if (!r.ok) {
        const body = await r.text();
        throw new Error(`${r.status}: ${body}`);
      }
      clearStoredEdits();
      setStatus("published");
      editSection.style.display = "none";
      const prUrl = `https://github.com/${owner}/${repo}/pull/${number}`;
      appendConsole("log", `Review published — redirecting to ${prUrl}`);
      // Brief pause so the user sees the confirmation before we navigate.
      setTimeout(() => {
        window.location.href = prUrl;
      }, 800);
    } catch (err) {
      showError(`Publish failed — ${err.message}`);
      publishBtn.disabled = false;
      discardBtn.disabled = false;
    }
  });

  rerunBtn.addEventListener("click", async () => {
    if (!infoCache) return;
    rerunBtn.disabled = true;
    try {
      const r = await fetch("/reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pr: infoCache.target,
          comment: infoCache.trigger_comment,
        }),
      });
      if (!r.ok) {
        const body = await r.text();
        throw new Error(`${r.status}: ${body}`);
      }
      const { url } = await r.json();
      window.location.href = url;
    } catch (err) {
      showError(`Could not re-run review — ${err.message}`);
      rerunBtn.disabled = false;
    }
  });

  discardBtn.addEventListener("click", async () => {
    if (!confirm("Discard the entire draft? Nothing will be posted to GitHub.")) return;
    publishBtn.disabled = true;
    discardBtn.disabled = true;
    try {
      const r = await fetch(`${apiBase}/discard`, { method: "POST" });
      if (!r.ok) throw new Error(`${r.status}`);
      clearStoredEdits();
      setStatus("discarded");
      editSection.style.display = "none";
      appendConsole("log", "Draft discarded.");
    } catch (err) {
      showError(`Discard failed — ${err.message}`);
      publishBtn.disabled = false;
      discardBtn.disabled = false;
    }
  });

  function attachStream() {
    const es = new EventSource(`${apiBase}/stream`);
    es.onmessage = (ev) => {
      try {
        appendConsole("log", JSON.parse(ev.data));
      } catch {
        appendConsole("log", ev.data);
      }
    };
    es.addEventListener("token", (ev) => {
      try {
        appendConsole("token", JSON.parse(ev.data));
      } catch {
        appendConsole("token", ev.data);
      }
    });
    es.addEventListener("tool", (ev) => {
      try {
        appendConsole("tool", JSON.parse(ev.data));
      } catch {
        appendConsole("tool", ev.data);
      }
    });
    es.addEventListener("reasoning", (ev) => {
      try {
        appendConsole("reasoning", JSON.parse(ev.data));
      } catch {
        appendConsole("reasoning", ev.data);
      }
    });
    es.addEventListener("metrics", (ev) => {
      try {
        handleMetrics(JSON.parse(ev.data));
      } catch {
        handleMetrics(ev.data);
      }
    });
    es.addEventListener("step", (ev) => {
      try {
        handleStep(JSON.parse(ev.data));
      } catch {
        handleStep(ev.data);
      }
    });
    es.addEventListener("error", (ev) => {
      // ``error`` fires both on payload-carrying server errors AND when
      // the server closes the stream normally (e.g., the generator
      // returns after a finished job's replay). EventSource will
      // auto-reconnect by default — which on a finished job means the
      // server replays the full history again, the client appends it
      // again, and we end up in a tight loop that spins the CPU and
      // crashes the tab. Close unconditionally; if the user wants to
      // see fresh state they can reload.
      if (ev.data) {
        try {
          appendConsole("error", JSON.parse(ev.data));
        } catch {
          appendConsole("error", ev.data);
        }
      }
      es.close();
    });
    es.addEventListener("done", () => {
      es.close();
      const details = document.getElementById("stream-details");
      if (details) details.open = false;
      loadDraft();
    });
  }

  (async () => {
    const info = await loadInfo();
    if (!info) return;
    if (info.status === "running") {
      attachStream();
    } else {
      // Refresh of an already-finished job. Render the form straight
      // away so the page isn't blank while the stream replay catches
      // up. Then attach the stream so the console still gets the
      // transcript.
      await loadDraft();
      const details = document.getElementById("stream-details");
      if (details) details.open = false;
      attachStream();
    }
  })();
})();

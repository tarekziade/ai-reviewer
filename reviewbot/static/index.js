(() => {
  const form = document.getElementById("submit-form");
  const btn = document.getElementById("submit-btn");
  const banner = document.getElementById("error-banner");
  const commentEl = document.getElementById("comment");
  const jobsSection = document.getElementById("jobs-section");
  const jobsTbody = document.getElementById("jobs-tbody");
  const jobsCount = document.getElementById("jobs-count");

  const PRESETS = {
    "first-pass":
      "@serge first-pass review. Flag clear correctness, security, and " +
      "API-shape problems; skip style nits and speculative concerns. " +
      "Use the browse tools to verify any claim before flagging it.",
    "new-model":
      "@serge this PR adds a new model. Verify the modular file structure, " +
      "tokenizer/config/processor wiring, that tests exist and cover the " +
      "model meaningfully, and that the docs entry is in place. Be strict " +
      "about consistency with sibling model implementations — use the " +
      "browse tools to compare.",
    "bugfix":
      "@serge this is a bug fix. Verify the change actually addresses the " +
      "root cause (not just the symptom), confirm a regression test was " +
      "added (or call out clearly that one is missing), and flag any " +
      "unrelated changes that slipped in.",
    "docs":
      "@serge this is a documentation change. Focus on accuracy, clarity, " +
      "and whether code samples are runnable. Ignore changes outside of " +
      "docs and docstrings; do not nit on prose style unless it's actively " +
      "misleading.",
  };

  function showError(msg) {
    banner.textContent = msg;
    banner.classList.add("visible");
  }

  function relTime(epochSeconds) {
    const delta = Date.now() / 1000 - epochSeconds;
    if (delta < 60) return `${Math.floor(delta)}s ago`;
    if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
    return `${Math.floor(delta / 86400)}d ago`;
  }

  function renderJobs(jobs) {
    jobsTbody.replaceChildren();
    jobsCount.textContent = jobs.length ? `(${jobs.length})` : "";
    if (!jobs.length) {
      jobsSection.style.display = "none";
      return;
    }
    for (const j of jobs) {
      const tr = document.createElement("tr");

      const tdStatus = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = `status-badge ${j.status}`;
      badge.textContent = j.status;
      tdStatus.appendChild(badge);

      const tdPr = document.createElement("td");
      const link = document.createElement("a");
      link.href = j.url;
      link.textContent = `${j.owner}/${j.repo}#${j.number}`;
      tdPr.appendChild(link);

      const tdAgo = document.createElement("td");
      tdAgo.className = "ago";
      tdAgo.textContent = relTime(j.created_at);
      tdAgo.title = new Date(j.created_at * 1000).toLocaleString();

      tr.appendChild(tdStatus);
      tr.appendChild(tdPr);
      tr.appendChild(tdAgo);
      jobsTbody.appendChild(tr);
    }
    jobsSection.style.display = "";
  }

  async function loadJobs() {
    try {
      const r = await fetch("/reviews");
      if (!r.ok) return;
      const data = await r.json();
      renderJobs(data.jobs || []);
    } catch {
      // Non-fatal — the form still works.
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    btn.disabled = true;
    banner.classList.remove("visible");
    const pr = document.getElementById("pr").value.trim();
    const comment = document.getElementById("comment").value.trim();
    try {
      const r = await fetch("/reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pr, comment }),
      });
      if (!r.ok) {
        const body = await r.text();
        throw new Error(`${r.status}: ${body}`);
      }
      const { url } = await r.json();
      window.location.href = url;
    } catch (err) {
      showError(`Could not start review — ${err.message}`);
      btn.disabled = false;
    }
  });

  for (const b of document.querySelectorAll("button.preset")) {
    b.addEventListener("click", () => {
      const text = PRESETS[b.dataset.preset];
      if (!text) return;
      commentEl.value = text;
      commentEl.focus();
    });
  }

  loadJobs();
  // Soft-refresh every 5s so running reviews tick over to "done" /
  // "published" without the user having to reload.
  setInterval(loadJobs, 5000);
})();

/* colabllm Control Plane — vanilla JS, no build step. Talks only to the local `colabllm ui` server. */
(() => {
  'use strict';

  // ------------------------------------------------------------------ token & storage
  const store = {
    get(area, key) { try { return window[area].getItem(key); } catch { return null; } },
    set(area, key, value) { try { window[area].setItem(key, value); } catch { /* storage unavailable */ } },
  };
  let token = store.get('sessionStorage', 'colabllm.token') || '';
  // A fresh `colabllm ui` link pasted into an open tab only changes the hash, so this also runs on hashchange.
  function adoptHashToken() {
    const m = /(?:^#|&)token=([^&]+)/.exec(location.hash);
    if (!m) return false;
    token = decodeURIComponent(m[1]);
    store.set('sessionStorage', 'colabllm.token', token);
    history.replaceState(null, '', location.pathname + '#/overview');
    return true;
  }
  adoptHashToken();

  // ------------------------------------------------------------------ state
  const S = {
    route: 'overview', data: null, usage: undefined, doctor: null, doctorBusy: false,
    lastOk: 0, error: null, search: '', instFilter: 'all', catFilter: 'All',
    drawer: null, pollTimer: null, selected: new Set(), optionSchemas: {}, lastHtml: '',
  };
  const $ = (sel, root = document) => root.querySelector(sel);
  const content = $('#content');

  // ------------------------------------------------------------------ helpers
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const icon = (name, cls = '') => `<svg class="i ${cls}"><use href="#i-${name}"/></svg>`;
  const CAT_ICON = { LLM: 'llm', Image: 'image', Video: 'video', Audio: 'audio', Document: 'doc', Training: 'train', Other: 'cpu' };
  const svcIcon = (cat, cls = '') => `<span class="svc-icon cat-${esc(cat)} ${cls}">${icon(CAT_ICON[cat] || 'cpu')}</span>`;
  const STATUS_LABEL = { ready: 'Ready', starting: 'Provisioning', failed: 'Failed', stopped: 'Stopped', running: 'Running', succeeded: 'Succeeded', pass: 'Pass', fail: 'Fail', warn: 'Warning', skip: 'Skipped' };
  const pill = (status) => `<span class="pill ${esc(status)}">${esc(STATUS_LABEL[status] || status)}</span>`;
  const fmtNum = (n, d = 0) => (n == null || Number.isNaN(n)) ? '—' : Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

  function fmtDuration(sec) {
    if (sec == null) return '—';
    sec = Math.max(0, Math.floor(sec));
    const d = Math.floor(sec / 86400), h = Math.floor(sec % 86400 / 3600), m = Math.floor(sec % 3600 / 60), s = sec % 60;
    if (d) return `${d}d ${h}h`;
    if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
    if (m) return `${m}m ${String(s).padStart(2, '0')}s`;
    return `${s}s`;
  }
  function fmtAgo(ts) {
    if (!ts) return '—';
    const sec = Date.now() / 1000 - ts;
    if (sec < 10) return 'just now';
    if (sec < 60) return `${Math.floor(sec)}s ago`;
    if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
    if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
    return new Date(ts * 1000).toLocaleDateString();
  }
  const fmtTime = (ts) => ts ? new Date(ts * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—';
  const isAlive = (r) => r.status === 'ready' || r.status === 'starting';
  const matches = (...fields) => !S.search || fields.some((f) => String(f || '').toLowerCase().includes(S.search));

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      method: opts.method || 'GET',
      headers: { 'X-Colabllm-Token': token, ...(opts.body ? { 'Content-Type': 'application/json' } : {}) },
      body: opts.body ? JSON.stringify(opts.body) : undefined,
      cache: 'no-store',
    });
    let payload = null;
    try { payload = await res.json(); } catch { /* empty */ }
    if (res.status === 401) { showLocked(); throw new Error('Unauthorized'); }
    if (!res.ok) throw new Error((payload && payload.error) || `HTTP ${res.status}`);
    return payload;
  }

  async function copy(text, label = 'Value') {
    try {
      await navigator.clipboard.writeText(text);
      toast('ok', `${label} copied`, 'Copied to clipboard');
    } catch {
      toast('bad', 'Copy failed', 'Clipboard access was denied by the browser');
    }
  }

  function toast(kind, title, body = '', ms = 4200) {
    const el = document.createElement('div');
    el.className = `toast ${kind}`;
    el.innerHTML = `${icon(kind === 'ok' ? 'check' : kind === 'bad' ? 'xcircle' : 'zap')}<div><strong>${esc(title)}</strong>${body ? `<span>${esc(body)}</span>` : ''}</div>`;
    $('#toasts').appendChild(el);
    setTimeout(() => el.remove(), ms);
  }

  function showLocked() { $('#locked').hidden = false; }

  // ------------------------------------------------------------------ data
  async function refresh() {
    try {
      S.data = await api('/api/overview');
      const known = new Set(S.data.instances.map((r) => r.id));
      [...S.selected].forEach((id) => { if (!known.has(id)) S.selected.delete(id); });
      S.lastOk = Date.now();
      S.error = null;
    } catch (err) {
      S.error = err.message;
    }
    renderChrome();
    renderPage();
    schedulePoll();
  }
  async function refreshUsage(force = false) {
    try {
      const r = await api('/api/usage' + (force ? '?refresh=1' : ''));
      S.usage = r.usage;
    } catch { S.usage = null; }
    if (S.route === 'overview') renderPage();
  }
  function schedulePoll() {
    clearTimeout(S.pollTimer);
    const busy = S.data && (S.data.operations.some((o) => o.state === 'running') || S.data.instances.some((i) => i.status === 'starting'));
    S.pollTimer = setTimeout(refresh, document.hidden ? 15000 : busy ? 2500 : 6000);
  }

  // ------------------------------------------------------------------ chrome
  const TITLES = { overview: 'Overview', instances: 'Instances', catalog: 'Service catalog', operations: 'Operations', diagnostics: 'Diagnostics' };
  function renderChrome() {
    document.querySelectorAll('.nav a').forEach((a) => a.classList.toggle('active', a.dataset.route === S.route));
    $('#page-title').textContent = TITLES[S.route];
    const d = S.data;
    const dot = $('#conn-dot'), label = $('#conn-label');
    if (S.error) { dot.className = 'dot dot-down'; label.textContent = 'Connection lost'; }
    else if (d) { dot.className = 'dot dot-live'; label.textContent = 'Local agent online'; }
    if (!d) return;
    $('#env-transport').textContent = d.environment.transport;
    $('#env-version').textContent = d.environment.version;
    const running = d.instances.filter(isAlive).length;
    const navRun = $('#nav-running');
    navRun.hidden = !running; navRun.textContent = running;
    const ops = d.operations.filter((o) => o.state === 'running').length;
    const navOps = $('#nav-ops');
    navOps.hidden = !ops; navOps.textContent = ops;
  }
  function tickUpdated() {
    const el = $('#updated');
    if (S.error) { el.textContent = 'Reconnecting…'; return; }
    if (!S.lastOk) return;
    const s = Math.floor((Date.now() - S.lastOk) / 1000);
    el.textContent = s < 2 ? 'Updated just now' : `Updated ${s}s ago`;
  }

  // ------------------------------------------------------------------ shared fragments
  function instanceActions(r) {
    const alive = isAlive(r);
    return `<div class="actions-inner">
      ${r.url ? `<a class="icon-btn sm" href="${esc(r.url)}" target="_blank" rel="noopener noreferrer" title="Open endpoint">${icon('external')}</a>` : `<button class="icon-btn sm" disabled title="No endpoint">${icon('external')}</button>`}
      <button class="icon-btn sm" data-action="secrets" data-id="${esc(r.id)}" title="Credentials" ${r.has_secrets ? '' : 'disabled'}>${icon('key')}</button>
      <button class="icon-btn sm" data-action="logs" data-id="${esc(r.id)}" title="Runtime logs" ${r.has_log ? '' : 'disabled'}>${icon('logs')}</button>
      <button class="icon-btn sm" data-action="health" data-id="${esc(r.id)}" title="Health probe" ${r.status === 'ready' ? '' : 'disabled'}>${icon('pulse')}</button>
      ${alive
        ? `<button class="icon-btn sm danger" data-action="stop" data-id="${esc(r.id)}" title="Stop instance">${icon('stop')}</button>`
        : `<button class="icon-btn sm" data-action="deploy" data-service="${esc(r.service)}" data-instance="${esc(r.instance_arg)}" title="Start again">${icon('play')}</button>`}
      <button class="icon-btn sm danger" data-action="delete" data-id="${esc(r.id)}" title="${alive ? 'Stop the instance before deleting it' : 'Delete instance'}" ${alive ? 'disabled' : ''}>${icon('trash')}</button>
    </div>`;
  }

  function instanceTable(rows, { compact = false, selectable = false } = {}) {
    const allOn = rows.length > 0 && rows.every((r) => S.selected.has(r.id));
    return `<div class="table-wrap"><table class="data">
      <thead><tr>${selectable ? `<th class="cb"><input type="checkbox" data-action="select-all" aria-label="Select all" ${allOn ? 'checked' : ''} ${rows.length ? '' : 'disabled'}></th>` : ''}<th>Instance</th><th>Status</th><th>Accelerator</th><th>Endpoint</th>${compact ? '' : '<th class="hide-md">Transport</th>'}<th>Uptime</th><th></th></tr></thead>
      <tbody>${rows.map((r) => `<tr class="${S.selected.has(r.id) ? 'selected' : ''}">
        ${selectable ? `<td class="cb"><input type="checkbox" data-action="select" data-id="${esc(r.id)}" aria-label="Select ${esc(r.id)}" ${S.selected.has(r.id) ? 'checked' : ''}></td>` : ''}
        <td><div class="cell-main">${svcIcon(r.category)}<div><div class="cell-title">${esc(r.id)}</div><div class="cell-sub">${esc(r.service)} · ${r.instance === 'default' ? 'default instance' : 'instance ' + esc(r.instance)}</div></div></div></td>
        <td>${pill(r.status)}</td>
        <td><span class="tag" title="${r.gpu_recorded ? 'Accelerator recorded at start' : 'Service default accelerator'}">${icon('cpu')}${esc(r.gpu)}</span></td>
        <td>${r.url ? `<span class="endpoint"><a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.url.replace(/^https?:\/\//, ''))}</a><button class="icon-btn sm" data-action="copy" data-value="${esc(r.url)}" data-label="Endpoint" title="Copy URL">${icon('copy')}</button></span>` : '<span class="muted">—</span>'}</td>
        ${compact ? '' : `<td class="hide-md"><span class="tag plain">${icon('globe')}${esc(r.transport)}</span></td>`}
        <td class="num">${isAlive(r) ? fmtDuration(r.uptime) : '<span class="muted">—</span>'}</td>
        <td class="actions">${instanceActions(r)}</td>
      </tr>`).join('')}</tbody></table></div>`;
  }

  function emptyState(iconName, title, text, actionHtml = '') {
    return `<div class="empty"><div class="empty-icon">${icon(iconName, 'i-lg')}</div><h4>${esc(title)}</h4><p>${esc(text)}</p>${actionHtml}</div>`;
  }

  function opsFeed(ops) {
    if (!ops.length) return emptyState('activity', 'No operations yet', 'Deployments and shutdowns started here appear in this feed.');
    return `<ul class="feed">${ops.map((o) => `<li data-action="op" data-id="${esc(o.id)}">
      <span class="feed-icon ${esc(o.state)}">${icon(o.state === 'running' ? 'refresh' : o.state === 'succeeded' ? 'check' : 'xcircle', o.state === 'running' ? 'spin' : '')}</span>
      <div class="feed-main"><div class="feed-title">${esc(o.summary)}</div><div class="feed-sub">${esc(o.last_line || o.command)}</div></div>
      <span class="feed-time">${fmtAgo(o.started)}</span></li>`).join('')}</ul>`;
  }

  // ------------------------------------------------------------------ pages
  function pageOverview() {
    const d = S.data, u = S.usage;
    const rows = d.instances.filter((r) => matches(r.id, r.service, r.url));
    const alive = d.instances.filter(isAlive);
    const ready = alive.filter((r) => r.status === 'ready').length;
    const starting = alive.length - ready;
    const totalUptime = alive.reduce((s, r) => s + (r.uptime || 0), 0);
    const longest = alive.slice().sort((a, b) => (b.uptime || 0) - (a.uptime || 0))[0];
    const longRunning = alive.filter((r) => (r.uptime || 0) > 4 * 3600);
    const failed = d.instances.filter((r) => r.status === 'failed');
    const runway = u && u.rate > 0 ? u.balance / u.rate : null;

    const banners = [];
    if (u && u.balance != null && u.balance < 50) banners.push(`<div class="banner bad">${icon('alert')}<div><strong>Compute balance is low</strong><p>${fmtNum(u.balance, 2)} compute units remain. New deployments may fail to get an accelerator.</p></div></div>`);
    if (longRunning.length) banners.push(`<div class="banner">${icon('clock')}<div><strong>${longRunning.length} instance${longRunning.length > 1 ? 's have' : ' has'} been running for more than 4 hours</strong><p>${longRunning.map((r) => esc(r.id)).join(', ')} ${longRunning.length > 1 ? 'keep' : 'keeps'} consuming compute units until stopped.</p></div><a class="btn btn-sm" href="#/instances">Review</a></div>`);
    if (failed.length) banners.push(`<div class="banner bad">${icon('xcircle')}<div><strong>${failed.length} runner${failed.length > 1 ? 's' : ''} exited with an error</strong><p>${failed.map((r) => esc(r.id)).join(', ')} — check the runtime logs before starting again.</p></div><a class="btn btn-sm" href="#/instances">Inspect</a></div>`);

    const gpuCounts = {};
    alive.forEach((r) => { gpuCounts[r.gpu] = (gpuCounts[r.gpu] || 0) + 1; });
    const gpuMax = Math.max(1, ...Object.values(gpuCounts));
    const gpuRows = Object.entries(gpuCounts).sort((a, b) => b[1] - a[1]);

    const kpi = (ic, cls, title, value, foot) => `<div class="kpi"><div class="kpi-head"><span class="kpi-icon ${cls}">${icon(ic)}</span>${title}</div><div class="kpi-value">${value}</div><div class="kpi-foot">${foot}</div></div>`;
    const usageLoading = u === undefined;
    const usageNA = u === null;

    return `
      <div class="page-head"><div><h1>Overview</h1><p>Live state of your Colab GPU services, tunnels and compute budget.</p></div>
        <div class="page-actions"><a class="btn" href="#/catalog">${icon('catalog')}Browse catalog</a></div></div>
      <div class="banners">${banners.join('')}</div>
      <section class="kpis">
        ${kpi('wallet', '', 'Compute balance',
          usageLoading ? '<span class="spinner"></span>' : usageNA ? '—' : `${fmtNum(u.balance, 2)}<small>CU</small>`,
          usageNA ? '<span>Unavailable — run <code>colab usage</code> to check sign-in</span>' : usageLoading ? 'Querying Colab account…' : `<span class="trend ${runway != null && runway < 24 ? 'warn' : 'ok'}">${runway != null ? fmtDuration(runway * 3600) + ' runway' : 'Idle'}</span><span>at current burn rate</span>`)}
        ${kpi('server', 'ok', 'Active instances',
          `${alive.length}<small>/ ${d.instances.length}</small>`,
          `<span class="trend ok">${ready} ready</span>${starting ? `<span class="trend warn">${starting} provisioning</span>` : ''}<span>${d.services.length} services available</span>`)}
        ${kpi('gauge', 'warn', 'Burn rate',
          usageLoading ? '<span class="spinner"></span>' : usageNA ? '—' : `${fmtNum(u.rate, 2)}<small>CU/h</small>`,
          usageNA || usageLoading ? '<span>Reported by Colab</span>' : `<span>≈ <strong>${fmtNum(u.rate * 24, 1)}</strong> CU per day</span><span class="trend muted">${fmtNum(u.assignments)} assignment${u.assignments === 1 ? '' : 's'}</span>`)}
        ${kpi('clock', 'info', 'Accelerator time',
          `${totalUptime ? fmtDuration(totalUptime) : '0m'}`,
          longest ? `<span>Longest: <strong>${esc(longest.id)}</strong> · ${fmtDuration(longest.uptime)}</span>` : '<span>No instances running</span>')}
      </section>
      <div class="grid-2">
        <div class="stack">
          <section class="card">
            <div class="card-head"><div><h3>Active instances <span class="count-badge">${alive.length}</span></h3><p>Runners attached to a Colab session on this machine</p></div><div class="page-actions">${alive.length > 1 ? `<button class="btn btn-sm btn-danger-outline" data-action="stop-all">${icon('stop')}Stop all</button>` : ''}<a class="btn btn-sm btn-ghost" href="#/instances">View all${icon('chevron', 'i-sm')}</a></div></div>
            ${(() => {
              const live = rows.filter(isAlive);
              if (live.length) return instanceTable(live, { compact: true });
              return emptyState('server', S.search ? 'No matching instances' : 'No instances running', S.search ? 'Try a different search term.' : 'Deploy a service from the catalog to provision a GPU runtime.', S.search ? '' : `<button class="btn btn-primary" data-action="deploy">${icon('plus')}Deploy service</button>`);
            })()}
          </section>
          <section class="card">
            <div class="card-head"><div><h3>Recent operations</h3><p>Lifecycle commands issued from this console</p></div><a class="btn btn-sm btn-ghost" href="#/operations">View all${icon('chevron', 'i-sm')}</a></div>
            ${opsFeed(d.operations.slice(0, 5))}
          </section>
        </div>
        <div class="stack">
          <section class="card">
            <div class="card-head"><div><h3>Accelerator allocation</h3><p>Running instances by GPU class</p></div></div>
            <div class="card-body">${gpuRows.length ? `<div class="bars">${gpuRows.map(([g, n]) => `<div class="bar-row"><span class="tag">${esc(g)}</span><div class="bar-track"><div class="bar-fill" data-w="${(n / gpuMax) * 100}"></div></div><span class="bar-val">${n} running</span></div>`).join('')}</div>` : '<div class="muted">No accelerators allocated.</div>'}</div>
          </section>
          <section class="card">
            <div class="card-head"><div><h3>Environment</h3><p>Tunnel and storage configuration</p></div><a class="btn btn-sm btn-ghost" href="#/diagnostics">Diagnostics${icon('chevron', 'i-sm')}</a></div>
            <div class="card-body"><dl class="kv">
              <dt>Transport</dt><dd><span class="tag plain">${icon('globe')}${esc(d.environment.transport)}</span></dd>
              <dt>Base domain</dt><dd>${d.environment.base_domain ? `<code>${esc(d.environment.base_domain)}</code>` : '<span class="muted">Dynamic trycloudflare URLs</span>'}</dd>
              ${d.environment.frp_server ? `<dt>FRP server</dt><dd><code>${esc(d.environment.frp_server)}</code></dd>` : ''}
              <dt>State directory</dt><dd><code>${esc(d.environment.home)}</code></dd>
              <dt>Runtime</dt><dd>colabllm ${esc(d.environment.version)} · Python ${esc(d.environment.python)}</dd>
            </dl></div>
          </section>
        </div>
      </div>`;
  }

  function pageInstances() {
    const d = S.data;
    const all = d.instances.filter((r) => matches(r.id, r.service, r.url, r.gpu));
    const counts = { all: all.length, running: all.filter(isAlive).length, stopped: all.filter((r) => r.status === 'stopped').length, failed: all.filter((r) => r.status === 'failed').length };
    const rows = all.filter((r) => S.instFilter === 'all' || (S.instFilter === 'running' ? isAlive(r) : r.status === S.instFilter));
    const tab = (k, label) => `<button class="tab ${S.instFilter === k ? 'active' : ''}" data-action="inst-filter" data-value="${k}">${label}<em>${counts[k]}</em></button>`;
    return `
      <div class="page-head"><div><h1>Instances</h1><p>Every service instance with local state under <code>${esc(d.environment.home)}</code>.</p></div>
        <div class="page-actions">${counts.running ? `<button class="btn btn-danger-outline" data-action="stop-all">${icon('stop')}Stop all running</button>` : ''}<button class="btn btn-primary" data-action="deploy">${icon('plus')}New instance</button></div></div>
      <section class="card">
        ${S.selected.size ? (() => {
            const picked = d.instances.filter((r) => S.selected.has(r.id));
            const running = picked.filter(isAlive).length, stopped = picked.length - running;
            return `<div class="toolbar bulkbar"><div class="bulk-count"><span class="count-badge">${S.selected.size}</span> selected</div>
            <div class="page-actions"><button class="btn btn-sm btn-ghost" data-action="clear-selection">Clear</button>
              <button class="btn btn-sm btn-danger-outline" data-action="delete-selected" ${stopped ? '' : 'disabled'} title="${stopped ? '' : 'Only stopped instances can be deleted'}">${icon('trash')}Delete${stopped ? ` ${stopped}` : ''}</button>
              <button class="btn btn-sm btn-danger" data-action="stop-selected" ${running ? '' : 'disabled'}>${icon('stop')}Stop${running ? ` ${running}` : ''}</button></div></div>`;
          })()
          : `<div class="toolbar"><div class="tabs">${tab('all', 'All')}${tab('running', 'Running')}${tab('stopped', 'Stopped')}${tab('failed', 'Failed')}</div>
          <span class="cell-sub">Select instances to stop or delete them together.</span></div>`}
        ${rows.length ? instanceTable(rows, { selectable: true }) : emptyState('server', 'Nothing here', S.search ? 'No instances match your search.' : 'No instances in this state.')}
      </section>`;
  }

  function pageCatalog() {
    const d = S.data;
    const cats = ['All', ...Array.from(new Set(d.services.map((s) => s.category)))];
    const runningBy = {};
    d.instances.filter(isAlive).forEach((r) => { runningBy[r.service] = (runningBy[r.service] || 0) + 1; });
    const list = d.services.filter((s) => (S.catFilter === 'All' || s.category === S.catFilter) && matches(s.name, s.summary, s.category, s.default_gpu));
    const authLabel = { caddy: 'Caddy auth', builtin: 'Built-in auth', none: 'No auth' };
    return `
      <div class="page-head"><div><h1>Service catalog</h1><p>${d.services.length} packaged runtimes. Each deploys a notebook to a Colab accelerator and publishes it through a secured tunnel.</p></div></div>
      <div class="catalog-filters"><div class="chips" role="tablist">${cats.map((c) => `<button class="chip ${S.catFilter === c ? 'active' : ''}" data-action="cat-filter" data-value="${esc(c)}">${esc(c)}</button>`).join('')}</div><span class="cell-sub">${list.length} of ${d.services.length} services</span></div>
      <div class="catalog" id="catalog-grid">${list.map((s) => `
        <article class="card svc-card">
          <div class="svc-card-head">${svcIcon(s.category)}<div><h3>${esc(s.name)}</h3><div class="cell-sub">${esc(s.category)}</div></div>
            ${runningBy[s.name] ? `<span class="pill ready running-flag">${runningBy[s.name]} running</span>` : ''}</div>
          <p class="summary">${esc(s.summary || 'No description.')}</p>
          <div class="svc-meta"><span class="tag">${icon('cpu')}${esc(s.default_gpu)}</span><span class="tag plain">${icon('key')}${esc(authLabel[s.auth] || s.auth)}</span>${s.high_mem ? '<span class="tag plain">High-RAM</span>' : ''}<span class="tag plain">${icon('clock')}≤ ${Math.round(s.ready_timeout / 60)} min boot</span></div>
          <div class="svc-card-foot"><a class="btn btn-sm btn-ghost" href="${esc(s.docs)}" target="_blank" rel="noopener noreferrer">${icon('book')}Docs</a><button class="btn btn-sm btn-primary" data-action="deploy" data-service="${esc(s.name)}">${icon('play')}Deploy</button></div>
        </article>`).join('') || `<div class="card">${emptyState('catalog', 'No services found', 'Try another search or category.')}</div>`}
      </div>`;
  }

  function pageOperations() {
    const ops = S.data.operations.filter((o) => matches(o.summary, o.command, o.target));
    return `
      <div class="page-head"><div><h1>Operations</h1><p>Audit trail of lifecycle commands started from this console session.</p></div></div>
      <section class="card">${ops.length ? `<div class="table-wrap"><table class="data">
        <thead><tr><th>Operation</th><th>Status</th><th>Command</th><th>Started</th><th>Duration</th><th>Exit</th></tr></thead>
        <tbody>${ops.map((o) => `<tr class="clickable" data-action="op" data-id="${esc(o.id)}">
          <td><div class="cell-title">${esc(o.summary)}</div><div class="cell-sub">${esc(o.id)}</div></td>
          <td>${pill(o.state)}</td>
          <td><code class="cell-sub">${esc(o.command)}</code></td>
          <td class="num">${fmtTime(o.started)}</td>
          <td class="num">${fmtDuration((o.finished || Date.now() / 1000) - o.started)}</td>
          <td class="num">${o.exit_code ?? '<span class="muted">—</span>'}</td></tr>`).join('')}</tbody></table></div>`
        : emptyState('activity', 'No operations yet', 'Operations are kept in memory for as long as this console runs. The CLI history is unaffected.')}</section>`;
  }

  function pageDiagnostics() {
    const d = S.data, doc = S.doctor;
    const iconFor = { pass: 'check', fail: 'xcircle', warn: 'alert', skip: 'alert' };
    const body = !doc ? `<div class="boot"><div class="spinner"></div><span>Running checks…</span></div>` : `<div class="table-wrap"><table class="data"><thead><tr><th></th><th>Check</th><th>Result</th><th>Details</th><th>Required</th></tr></thead><tbody>
      ${doc.checks.map((c) => `<tr class="check-row"><td><span class="check-icon ${esc(c.status)}">${icon(iconFor[c.status] || 'alert', 'i-sm')}</span></td>
        <td class="cell-title">${esc(c.name)}</td><td>${pill(c.status)}</td><td class="cell-sub">${esc(c.message)}</td><td>${c.required ? 'Yes' : '<span class="muted">Optional</span>'}</td></tr>`).join('')}
      </tbody></table></div>`;
    const failed = doc ? doc.checks.filter((c) => c.status === 'fail' && c.required).length : 0;
    return `
      <div class="page-head"><div><h1>Diagnostics</h1><p>Read-only checks of local tooling, credentials and configuration (<code>colabllm doctor</code>).</p></div>
        <div class="page-actions">
          <button class="btn" data-action="doctor" ${S.doctorBusy ? 'disabled' : ''}>${icon('refresh', S.doctorBusy ? 'spin' : '')}Re-run</button>
          <button class="btn btn-primary" data-action="doctor-online" ${S.doctorBusy ? 'disabled' : ''}>${icon('shield')}Run online checks</button></div></div>
      ${doc ? (doc.ok ? `<div class="banner info">${icon('check')}<div><strong>All required checks passed</strong><p>${doc.online ? 'Including online authorization checks.' : 'Offline checks only — online checks verify that Colab and Cloudflare sign-ins are still valid.'}</p></div></div>`
        : `<div class="banner bad">${icon('alert')}<div><strong>${failed} required check${failed === 1 ? '' : 's'} failing</strong><p>Deployments will likely fail until these are resolved.</p></div></div>`) : ''}
      <section class="card"><div class="card-head"><div><h3>Health checks</h3><p>Never opens a browser, creates a session or mounts Drive</p></div></div>${body}</section>
      <section class="card"><div class="card-head"><div><h3>Configuration</h3></div></div><div class="card-body"><dl class="kv">
        <dt>Transport</dt><dd>${esc(d.environment.transport)}</dd>
        <dt>Base domain</dt><dd>${esc(d.environment.base_domain || '—')}</dd>
        <dt>State directory</dt><dd><code>${esc(d.environment.home)}</code></dd>
        <dt>Version</dt><dd>colabllm ${esc(d.environment.version)} · Python ${esc(d.environment.python)}</dd>
      </dl></div></section>`;
  }

  const PAGES = { overview: pageOverview, instances: pageInstances, catalog: pageCatalog, operations: pageOperations, diagnostics: pageDiagnostics };
  function renderPage() {
    if (!S.data) {
      if (S.error) content.innerHTML = `<div class="card">${emptyState('alert', 'Cannot reach the local agent', S.error + ' — is `colabllm ui` still running?')}</div>`;
      return;
    }
    const html = PAGES[S.route]();
    // Skip identical re-renders so polling never swaps a button out from under a click.
    if (html === S.lastHtml && content.firstElementChild) return;
    S.lastHtml = html;
    const y = window.scrollY;
    content.innerHTML = html;
    content.querySelectorAll('[data-w]').forEach((el) => { el.style.width = el.dataset.w + '%'; });
    window.scrollTo(0, y);
  }

  async function runDoctor(online = false) {
    S.doctorBusy = true; renderPage();
    try { S.doctor = { ...(await api('/api/doctor' + (online ? '?online=1' : ''))), online }; }
    catch (err) { toast('bad', 'Diagnostics failed', err.message); }
    S.doctorBusy = false;
    if (S.route === 'diagnostics') renderPage();
  }

  // ------------------------------------------------------------------ modal
  function openModal(title, bodyHtml, footHtml, size = '') {
    $('#modal .modal-panel').classList.toggle('wide', size === 'wide');
    $('#modal-title').textContent = title;
    $('#modal-body').innerHTML = bodyHtml;
    $('#modal-foot').innerHTML = footHtml;
    const m = $('#modal');
    m.classList.add('open'); m.setAttribute('aria-hidden', 'false');
    setTimeout(() => { const f = m.querySelector('input:not([type=checkbox]), select, .btn-primary, .btn-danger'); f && f.focus(); }, 60);
  }
  function closeModal() { const m = $('#modal'); m.classList.remove('open'); m.setAttribute('aria-hidden', 'true'); }

  // Shell-style preview of one argv element (display only; the server never uses a shell).
  const shq = (v) => /^[\w@%+=:,./-]+$/.test(v) ? v : `'${String(v).replace(/'/g, `'\\''`)}'`;

  function optionArgs(schema, values) {
    const args = [];
    (schema || []).forEach((o) => {
      const v = values[o.key];
      if (o.kind === 'flag') { if (v === true) args.push(o.flag); }
      else if (o.kind === 'tristate') { if (v === 'on') args.push(o.flag); else if (v === 'off') args.push(o.flag.replace(/^--/, '--no-')); }
      else if (o.kind === 'list') { String(v || '').split('\n').map((x) => x.trim()).filter(Boolean).forEach((x) => args.push(`${o.flag}=${x}`)); }
      else if (v != null && String(v).trim() !== '') args.push(`${o.flag}=${String(v).trim()}`);
    });
    return args;
  }

  function optionField(o, value) {
    const id = 'opt-' + o.key.replace(/[^\w-]/g, '');
    const help = o.help ? `<small>${esc(o.help)}</small>` : '';
    const label = `<label for="${id}"><code>${esc(o.flag)}</code></label>`;
    if (o.kind === 'flag') {
      return `<label class="check opt-flag" data-opt-row="${esc(o.key)}"><input type="checkbox" id="${id}" data-opt="${esc(o.key)}" ${value === true ? 'checked' : ''}><div><strong><code>${esc(o.flag)}</code></strong><span>${esc(o.help)}</span></div></label>`;
    }
    if (o.kind === 'tristate' || o.kind === 'choice') {
      const opts = o.kind === 'tristate' ? [['', 'Service default'], ['on', 'On'], ['off', 'Off']]
        : [['', o.default != null ? `Default (${o.default})` : 'Service default'], ...o.choices.map((c) => [c, c])];
      return `<div class="field opt-field" data-opt-row="${esc(o.key)}">${label}<select class="select" id="${id}" data-opt="${esc(o.key)}">${opts.map(([v, t]) => `<option value="${esc(v)}" ${String(value ?? '') === v ? 'selected' : ''}>${esc(t)}</option>`).join('')}</select>${help}</div>`;
    }
    if (o.kind === 'list') {
      return `<div class="field opt-field opt-wide" data-opt-row="${esc(o.key)}">${label}<textarea class="input textarea" id="${id}" data-opt="${esc(o.key)}" rows="2" spellcheck="false" placeholder="One value per line">${esc(value || '')}</textarea>${help}</div>`;
    }
    const ph = o.metavar || (o.default != null && o.default !== '' ? `default: ${o.default}` : 'Service default');
    return `<div class="field opt-field" data-opt-row="${esc(o.key)}">${label}<input class="input" id="${id}" data-opt="${esc(o.key)}" ${o.kind === 'number' ? 'inputmode="numeric"' : ''} value="${esc(value ?? '')}" placeholder="${esc(ph)}" autocomplete="off" spellcheck="false">${help}</div>`;
  }

  async function loadSchema(service) {
    if (!S.optionSchemas[service]) {
      const r = await api(`/api/services/${encodeURIComponent(service)}/options`);
      S.optionSchemas[service] = r.options;
    }
    return S.optionSchemas[service];
  }

  function deployModal(serviceName, instanceArg = '') {
    const d = S.data;
    const svc = d.services.find((s) => s.name === serviceName) || d.services[0];
    const form = { service: svc.name, instance: instanceArg, gpu: svc.default_gpu, quick: false, drive: true, rotate: false, opts: {}, filter: '', open: true };
    const quickOnly = d.environment.transport === 'quick';
    const current = () => d.services.find((x) => x.name === form.service);
    const schema = () => S.optionSchemas[form.service];
    const valid = () => !form.instance || /^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$/.test(form.instance);
    const argv = () => [form.service, 'up', '--detach', ...(form.instance ? ['-i', form.instance] : []),
      ...(form.gpu !== current().default_gpu ? ['--gpu', form.gpu] : []), ...(form.quick ? ['--quick-tunnel'] : []),
      ...(!form.drive ? ['--no-drive'] : []), ...(form.rotate ? ['--new-credentials'] : []), ...optionArgs(schema(), form.opts)];

    const body = () => {
      const s = current();
      return `
        <div class="row-2">
          <div class="field"><label for="f-svc">Service</label><select class="select" id="f-svc">${d.services.map((x) => `<option value="${esc(x.name)}" ${x.name === form.service ? 'selected' : ''}>${esc(x.name)}</option>`).join('')}</select></div>
          <div class="field"><label for="f-inst">Instance ID <span class="muted">(optional)</span></label><input class="input" id="f-inst" value="${esc(form.instance)}" placeholder="default" maxlength="64" autocomplete="off" spellcheck="false"><small id="f-inst-msg">Run parallel copies with IDs like 2 or eval.</small></div>
        </div>
        <div class="field"><span class="field-label">Accelerator</span><div class="seg tall" id="f-gpu">${d.gpus.map((g) => `<button type="button" data-gpu="${g}" class="${form.gpu === g ? 'active' : ''}">${g === 'none' ? 'CPU' : g}${g === s.default_gpu ? '<span class="def">default</span>' : ''}</button>`).join('')}</div></div>
        <div class="field"><span class="field-label">Network</span>
          ${quickOnly ? '<div class="note">' + icon('globe') + '<span>No base domain is configured, so a temporary <code>trycloudflare.com</code> URL is issued.</span></div>'
            : `<label class="check"><input type="checkbox" id="f-quick" ${form.quick ? 'checked' : ''}><div><strong>Use a quick tunnel</strong><span>Ignore the configured ${esc(d.environment.transport)} transport and issue a temporary trycloudflare.com URL.</span></div></label>`}
        </div>
        <div class="row-2">
          <label class="check"><input type="checkbox" id="f-drive" ${form.drive ? 'checked' : ''}><div><strong>Mount Google Drive</strong><span>Persist models and outputs under MyDrive/colab-llm.</span></div></label>
          <label class="check"><input type="checkbox" id="f-rotate" ${form.rotate ? 'checked' : ''}><div><strong>Rotate credentials</strong><span>Issue a new password and API token.</span></div></label>
        </div>
        <section class="opt-group" id="f-opts"></section>
        <div class="field"><span class="field-label">Equivalent command</span><div class="cmd-preview">${icon('terminal')}<span id="f-cmd"></span></div></div>
        <div class="note">${icon('alert')}<span>The runner starts in the background and keeps consuming compute units until you stop it. Boot time is usually several minutes.</span></div>`;
    };

    const updatePreview = () => {
      const cmd = $('#f-cmd');
      if (cmd) cmd.textContent = 'colabllm ' + argv().map(shq).join(' ');
      const badge = $('#f-opts-count');
      if (badge) { const n = Object.keys(form.opts).length; badge.textContent = n ? `${n} set` : ''; badge.hidden = !n; }
      const submit = $('#modal-submit');
      if (submit) submit.disabled = !valid();
    };

    const renderOptions = () => {
      const box = $('#f-opts');
      if (!box) return;
      const list = schema();
      if (list === undefined) { box.innerHTML = `<div class="opt-head"><span class="field-label">Service options</span></div><div class="boot sm"><div class="spinner"></div><span>Reading ${esc(form.service)} options…</span></div>`; return; }
      if (list === null) { box.innerHTML = `<div class="opt-head"><span class="field-label">Service options</span></div><div class="note">${icon('alert')}<span>Could not read this service's options. Core settings above still apply.</span></div>`; return; }
      if (!list.length) { box.innerHTML = `<div class="opt-head"><span class="field-label">Service options</span></div><div class="note">${icon('check')}<span><code>${esc(form.service)}</code> has no service-specific start options.</span></div>`; return; }
      const fields = list.filter((o) => o.kind !== 'flag'), flags = list.filter((o) => o.kind === 'flag');
      box.innerHTML = `
        <button type="button" class="opt-head opt-toggle" id="f-opts-toggle" aria-expanded="${form.open}">
          ${icon('chevron', 'opt-chevron')}<span class="field-label">Service options</span><span class="count-badge">${list.length}</span><span class="pill starting opt-set" id="f-opts-count" hidden></span>
          <span class="opt-hint">${esc(form.service)} start arguments, validated by the CLI parser</span>
        </button>
        <div class="opt-body" ${form.open ? '' : 'hidden'}>
          ${list.length > 8 ? `<div class="opt-filter">${icon('search')}<input class="input" id="f-opt-filter" placeholder="Filter ${list.length} options" value="${esc(form.filter)}" autocomplete="off" spellcheck="false"></div>` : ''}
          <div class="opt-grid">${fields.map((o) => optionField(o, form.opts[o.key])).join('')}</div>
          ${flags.length ? `<div class="opt-flags">${flags.map((o) => optionField(o, form.opts[o.key])).join('')}</div>` : ''}
          <div class="opt-empty muted" hidden>No options match.</div>
        </div>`;
      applyFilter();
      updatePreview();
    };

    const applyFilter = () => {
      const q = form.filter.toLowerCase();
      let shown = 0;
      document.querySelectorAll('#f-opts [data-opt-row]').forEach((row) => {
        const o = schema().find((x) => x.key === row.dataset.optRow);
        const hit = !q || (o.flag + ' ' + o.help).toLowerCase().includes(q);
        row.hidden = !hit; if (hit) shown++;
      });
      const empty = $('#f-opts .opt-empty'); if (empty) empty.hidden = shown > 0;
    };

    const ensureSchema = async () => {
      const name = form.service;
      if (S.optionSchemas[name] !== undefined) return renderOptions();
      renderOptions();
      try { await loadSchema(name); } catch (err) { S.optionSchemas[name] = null; toast('bad', 'Cannot read service options', err.message); }
      if (form.service === name) renderOptions();
    };

    const rerender = () => { $('#modal-body').innerHTML = body(); bind(); ensureSchema(); updatePreview(); };

    const bind = () => {
      $('#f-svc').onchange = (e) => { form.service = e.target.value; form.gpu = current().default_gpu; form.opts = {}; form.filter = ''; rerender(); };
      $('#f-inst').oninput = (e) => {
        form.instance = e.target.value.trim();
        const ok = valid(), msg = $('#f-inst-msg');
        msg.textContent = ok ? 'Run parallel copies with IDs like 2 or eval.' : 'Use letters, digits, _ or -, not starting with -.';
        msg.classList.toggle('bad-text', !ok);
        updatePreview();
      };
      $('#f-gpu').onclick = (e) => {
        const b = e.target.closest('[data-gpu]'); if (!b) return;
        form.gpu = b.dataset.gpu;
        document.querySelectorAll('#f-gpu [data-gpu]').forEach((x) => x.classList.toggle('active', x === b));
        updatePreview();
      };
      const q = $('#f-quick'); if (q) q.onchange = (e) => { form.quick = e.target.checked; updatePreview(); };
      $('#f-drive').onchange = (e) => { form.drive = e.target.checked; updatePreview(); };
      $('#f-rotate').onchange = (e) => { form.rotate = e.target.checked; updatePreview(); };
      const box = $('#f-opts');
      const onOpt = (e) => {
        const el = e.target;
        if (el.id === 'f-opt-filter') { form.filter = el.value; applyFilter(); return; }
        const key = el.dataset && el.dataset.opt; if (!key) return;
        const value = el.type === 'checkbox' ? el.checked : el.value;
        if (value === '' || value === false) delete form.opts[key]; else form.opts[key] = value;
        updatePreview();
      };
      box.oninput = onOpt; box.onchange = onOpt;
      box.onclick = (e) => {
        if (!e.target.closest('#f-opts-toggle')) return;
        form.open = !form.open;
        $('#f-opts-toggle').setAttribute('aria-expanded', form.open);
        $('#f-opts .opt-body').hidden = !form.open;
      };
    };

    openModal('Deploy service', body(), `<button class="btn" data-close="modal">Cancel</button><button class="btn btn-primary" id="modal-submit">${icon('play')}Deploy</button>`, 'wide');
    bind(); ensureSchema(); updatePreview();
    $('#modal-submit').onclick = async (e) => {
      if (!valid()) return;
      const btn = e.currentTarget;
      btn.disabled = true;
      try {
        const s = current();
        const options = {};
        (schema() || []).forEach((o) => {
          const v = form.opts[o.key];
          if (v === undefined) return;
          options[o.key] = o.kind === 'list' ? String(v).split('\n').map((x) => x.trim()).filter(Boolean) : v;
        });
        const op = await api(`/api/services/${encodeURIComponent(form.service)}/deploy`, { method: 'POST', body: {
          instance: form.instance, gpu: form.gpu !== s.default_gpu ? form.gpu : null, quick_tunnel: form.quick,
          no_drive: !form.drive, new_credentials: form.rotate, options } });
        closeModal();
        toast('info', 'Deployment started', `${op.target} — follow progress in Operations`);
        await refresh();
        openOpDrawer(op.id);
      } catch (err) {
        btn.disabled = false;
        toast('bad', 'Deployment rejected', err.message, 7000);
      }
    };
  }

  function deleteModal(ids) {
    const rows = S.data.instances.filter((r) => ids.includes(r.id) && !isAlive(r));
    const skipped = ids.length - rows.length;
    if (!rows.length) { toast('info', 'Nothing to delete', 'Running instances must be stopped first'); return; }
    const tunnels = rows.filter((r) => r.owns_tunnel);
    const n = rows.length, plural = n > 1 ? 's' : '';
    openModal(n === 1 ? `Delete ${rows[0].id}?` : `Delete ${n} instances?`,
      `<p>This permanently removes the local state of ${n === 1 ? 'this instance' : 'these instances'}: saved credentials, <code>run.log</code> and the notebook copy under <code>${esc(S.data.environment.home)}/services/</code>. Models, outputs and voices on Google Drive are not touched.</p>
       <div class="bulk-list">${rows.map((r) => `<div class="bulk-item">${svcIcon(r.category)}<div class="bulk-main"><div class="cell-title">${esc(r.id)}</div><div class="cell-sub">${pill(r.status)}${r.started ? ` · last started ${esc(r.started)}` : ''}</div></div><span class="tag">${icon('cpu')}${esc(r.gpu)}</span></div>`).join('')}</div>
       ${skipped ? `<div class="note">${icon('alert')}<span>${skipped} running instance${skipped > 1 ? 's are' : ' is'} skipped. Stop ${skipped > 1 ? 'them' : 'it'} first.</span></div>` : ''}
       ${tunnels.length ? `<label class="check"><input type="checkbox" id="f-deltun" checked><div><strong>Also delete named tunnel${tunnels.length > 1 ? 's' : ''} (${tunnels.length})</strong><span>${tunnels.map((r) => esc(r.hostname || r.id)).join(', ')} and ${tunnels.length > 1 ? 'their' : 'its'} DNS route${tunnels.length > 1 ? 's are' : ' is'} removed. Uncheck to leave ${tunnels.length > 1 ? 'them' : 'it'} in Cloudflare/FRP.</span></div></label>` : ''}
       <div class="note">${icon('shield')}<span>Any leftover Colab session is stopped first. If Colab cannot be reached, nothing is deleted.</span></div>
       <div class="cmd-preview">${icon('terminal')}<span>${rows.map((r) => `colabllm ${esc(r.service)} purge${r.instance_arg ? ' -i ' + esc(r.instance_arg) : ''}`).join(' && ')}</span></div>`,
      `<button class="btn" data-close="modal">Cancel</button><button class="btn btn-danger" id="modal-submit">${icon('trash')}Delete ${n === 1 ? 'instance' : `${n} instances`}</button>`);
    $('#modal-submit').onclick = async (e) => {
      const btn = e.currentTarget; // currentTarget is null once the handler awaits
      btn.disabled = true;
      const del = $('#f-deltun');
      const body = { delete_tunnel: del ? del.checked : true };
      try {
        let results;
        if (rows.length === 1) {
          const op = await api(`/api/instances/${encodeURIComponent(rows[0].id)}/delete`, { method: 'POST', body });
          results = [{ id: rows[0].id, ok: true, operation: op }];
        } else {
          results = (await api('/api/instances/delete', { method: 'POST', body: { ...body, ids: rows.map((r) => r.id) } })).results;
        }
        closeModal();
        const ok = results.filter((r) => r.ok), bad = results.filter((r) => !r.ok);
        if (ok.length) toast('info', `Deleting ${ok.length} instance${ok.length > 1 ? 's' : ''}`, 'Progress is shown in Operations');
        bad.forEach((r) => toast('bad', `Could not delete ${r.id}`, r.error, 7000));
        rows.forEach((r) => S.selected.delete(r.id));
        refresh();
      } catch (err) {
        btn.disabled = false;
        toast('bad', 'Delete rejected', err.message, 7000);
      }
    };
  }

  function bulkStopModal(ids) {
    const rows = S.data.instances.filter((r) => ids.includes(r.id) && isAlive(r));
    if (!rows.length) { toast('info', 'Nothing to stop', 'No running instances selected'); return; }
    const named = rows.filter((r) => r.hostname);
    const gpus = {};
    rows.forEach((r) => { gpus[r.gpu] = (gpus[r.gpu] || 0) + 1; });
    openModal(`Stop ${rows.length} instance${rows.length > 1 ? 's' : ''}?`,
      `<p>Each instance's runner and Colab session is stopped and its accelerator released: ${Object.entries(gpus).map(([g, n]) => `<strong>${n}× ${esc(g)}</strong>`).join(', ')}. Drive data is kept.</p>
       <div class="bulk-list">${rows.map((r) => `<div class="bulk-item">${svcIcon(r.category)}<div class="bulk-main"><div class="cell-title">${esc(r.id)}</div><div class="cell-sub">${pill(r.status)} · up ${fmtDuration(r.uptime)}</div></div><span class="tag">${icon('cpu')}${esc(r.gpu)}</span></div>`).join('')}</div>
       ${named.length ? `<label class="check"><input type="checkbox" id="f-deltun"><div><strong>Also delete named tunnels (${named.length})</strong><span>${named.map((r) => esc(r.hostname)).join(', ')} and their DNS routes are removed. Leave unchecked to reuse the same URLs.</span></div></label>` : ''}
       <div class="note">${icon('activity')}<span>Each instance runs its own <code>down</code> operation, so one failure does not block the others.</span></div>`,
      `<button class="btn" data-close="modal">Cancel</button><button class="btn btn-danger" id="modal-submit">${icon('stop')}Stop ${rows.length} instance${rows.length > 1 ? 's' : ''}</button>`);
    $('#modal-submit').onclick = async (e) => {
      const btn = e.currentTarget; // currentTarget is null once the handler awaits
      btn.disabled = true;
      const del = $('#f-deltun');
      try {
        const res = await api('/api/instances/stop', { method: 'POST', body: { ids: rows.map((r) => r.id), delete_tunnel: !!(del && del.checked) } });
        closeModal();
        const ok = res.results.filter((r) => r.ok), bad = res.results.filter((r) => !r.ok);
        if (ok.length) toast('info', `Stopping ${ok.length} instance${ok.length > 1 ? 's' : ''}`, ok.map((r) => r.id).join(', '));
        bad.forEach((r) => toast('bad', `Could not stop ${r.id}`, r.error, 7000));
        S.selected.clear();
        refresh();
      } catch (err) {
        btn.disabled = false;
        toast('bad', 'Bulk stop rejected', err.message);
      }
    };
  }

  function stopModal(id) {
    const r = S.data.instances.find((x) => x.id === id);
    if (!r) return;
    openModal(`Stop ${r.id}?`,
      `<p>This stops the runner and the Colab session <code>${esc(r.id)}</code>, releasing its <strong>${esc(r.gpu)}</strong> accelerator. Unsaved in-memory work on the runtime is lost; Drive data is kept.</p>
       ${r.hostname ? `<label class="check"><input type="checkbox" id="f-deltun"><div><strong>Also delete the named tunnel</strong><span>Removes <code>${esc(r.hostname)}</code> and its DNS route. Leave unchecked to reuse the same URL next time.</span></div></label>` : ''}
       <div class="cmd-preview">${icon('terminal')}<span>colabllm ${esc(r.service)} down${r.instance_arg ? ' -i ' + esc(r.instance_arg) : ''}</span></div>`,
      `<button class="btn" data-close="modal">Cancel</button><button class="btn btn-danger" id="modal-submit">${icon('stop')}Stop instance</button>`);
    $('#modal-submit').onclick = async (e) => {
      const btn = e.currentTarget; // currentTarget is null once the handler awaits
      btn.disabled = true;
      const del = $('#f-deltun');
      try {
        const op = await api(`/api/instances/${encodeURIComponent(id)}/stop`, { method: 'POST', body: { delete_tunnel: !!(del && del.checked) } });
        closeModal();
        toast('info', 'Shutdown started', op.target);
        refresh();
      } catch (err) {
        btn.disabled = false;
        toast('bad', 'Stop rejected', err.message);
      }
    };
  }

  async function secretsModal(id) {
    let data;
    try { data = await api(`/api/instances/${encodeURIComponent(id)}/secrets`); }
    catch (err) { toast('bad', 'Cannot read credentials', err.message); return; }
    const LABELS = { url: 'Endpoint', auth_user: 'Username', auth_pass: 'Password', api_token: 'API token', login_user: 'Login user', login_pass: 'Login password' };
    const sensitive = new Set(['auth_pass', 'api_token', 'login_pass']);
    let shown = false;
    const render = () => Object.entries(data).map(([k, v]) => `<div class="secret"><span class="k">${esc(LABELS[k] || k)}</span>
      <span class="v ${sensitive.has(k) && !shown ? 'masked' : ''}">${sensitive.has(k) && !shown ? '•'.repeat(18) : esc(v)}</span>
      <button class="icon-btn sm" data-action="copy" data-value="${esc(v)}" data-label="${esc(LABELS[k] || k)}" title="Copy">${icon('copy')}</button></div>`).join('');
    openModal(`Credentials · ${id}`,
      `<div class="secret-list" id="secret-list">${render()}</div>
       <div class="note">${icon('shield')}<span>Read from the local state file. Browsers log in with the username and password; API clients send <code>Authorization: Bearer &lt;token&gt;</code>.</span></div>`,
      `<button class="btn" id="reveal-btn">${icon('eye')}Reveal</button><button class="btn btn-primary" data-close="modal">Done</button>`);
    $('#reveal-btn').onclick = (e) => {
      shown = !shown;
      $('#secret-list').innerHTML = render();
      e.currentTarget.innerHTML = shown ? `${icon('eyeoff')}Hide` : `${icon('eye')}Reveal`;
    };
  }

  // ------------------------------------------------------------------ drawer
  function openDrawer(kicker, title, tools, bodyHtml, flush = false) {
    closeDrawerTimers();
    $('#drawer-kicker').textContent = kicker;
    $('#drawer-title').textContent = title;
    $('#drawer-tools').innerHTML = tools;
    const b = $('#drawer-body');
    b.className = 'drawer-body' + (flush ? ' flush' : '');
    b.innerHTML = bodyHtml;
    const dr = $('#drawer'); dr.classList.add('open'); dr.setAttribute('aria-hidden', 'false');
  }
  function closeDrawerTimers() { if (S.drawer && S.drawer.timer) clearTimeout(S.drawer.timer); S.drawer = null; }
  function closeDrawer() { closeDrawerTimers(); const dr = $('#drawer'); dr.classList.remove('open'); dr.setAttribute('aria-hidden', 'true'); }

  function colorize(text) {
    return text.split('\n').map((line) => {
      const e = esc(line);
      if (/\b(error|traceback|exception|failed|fatal)\b/i.test(line)) return `<span class="ln-err">${e}</span>`;
      if (/\bwarn(ing)?\b/i.test(line)) return `<span class="ln-warn">${e}</span>`;
      if (/CT_READY|\bready\b|\bsuccess/i.test(line)) return `<span class="ln-ok">${e}</span>`;
      if (/^\s*(\[colabsvc\]|==>|\$ )/.test(line)) return `<span class="ln-dim">${e}</span>`;
      return e;
    }).join('\n');
  }

  function logDrawer(id) {
    const r = S.data.instances.find((x) => x.id === id);
    openDrawer('Runtime logs', id, `<button class="icon-btn" id="log-copy" title="Copy log">${icon('copy')}</button>`,
      `<div class="logbar"><span class="dot ${r && isAlive(r) ? 'dot-live' : ''}"></span><span id="log-meta">Loading…</span><span class="spacer"></span><label><input type="checkbox" id="log-follow" checked> Follow</label></div><pre class="logview" id="logview"></pre>`, true);
    const st = S.drawer = { kind: 'logs', id, text: '' };
    const load = async () => {
      if (S.drawer !== st) return;
      try {
        const res = await api(`/api/instances/${encodeURIComponent(id)}/logs`);
        if (S.drawer !== st) return;
        if (res.text !== st.text) {
          st.text = res.text;
          const v = $('#logview');
          v.innerHTML = res.text ? colorize(res.text) : '<span class="ln-dim">Log is empty.</span>';
          if ($('#log-follow').checked) v.scrollTop = v.scrollHeight;
        }
        $('#log-meta').textContent = `${(res.size / 1024).toFixed(1)} KB${res.truncated ? ' · showing last 256 KB' : ''} · run.log`;
      } catch (err) { $('#log-meta').textContent = err.message; }
      st.timer = setTimeout(load, 2000);
    };
    $('#log-copy').onclick = () => copy(st.text, 'Log');
    load();
  }

  function openOpDrawer(opId) {
    openDrawer('Operation', 'Loading…', '', `<div class="spinner"></div>`);
    const st = S.drawer = { kind: 'op', id: opId };
    const load = async () => {
      if (S.drawer !== st) return;
      let op;
      try { op = await api(`/api/operations/${encodeURIComponent(opId)}`); }
      catch (err) { $('#drawer-body').innerHTML = emptyState('alert', 'Operation not found', err.message); return; }
      if (S.drawer !== st) return;
      $('#drawer-title').textContent = op.summary;
      const prev = $('#op-log');
      const follow = !prev || prev.scrollTop + prev.clientHeight >= prev.scrollHeight - 30;
      const instance = S.data && S.data.instances.find((x) => x.id === op.target);
      $('#drawer-body').innerHTML = `
        <dl class="kv">
          <dt>Status</dt><dd>${pill(op.state)}</dd>
          <dt>Command</dt><dd><code>${esc(op.command)}</code></dd>
          <dt>Started</dt><dd>${fmtTime(op.started)}</dd>
          <dt>Duration</dt><dd class="num">${fmtDuration((op.finished || Date.now() / 1000) - op.started)}</dd>
          ${op.exit_code != null ? `<dt>Exit code</dt><dd class="num">${op.exit_code}</dd>` : ''}
        </dl>
        ${op.state === 'succeeded' && op.action === 'deploy' ? `<div class="banner info">${icon('zap')}<div><strong>Runner started in the background</strong><p>The model keeps booting on Colab; the instance turns <em>Ready</em> once the notebook reports CT_READY.</p></div>${instance && instance.has_log ? `<button class="btn btn-sm" data-action="logs" data-id="${esc(op.target)}">Follow logs</button>` : ''}</div>` : ''}
        <pre class="logview" id="op-log">${op.output ? colorize(op.output) : '<span class="ln-dim">Waiting for output…</span>'}</pre>`;
      const v = $('#op-log');
      v.style.minHeight = '320px'; v.style.borderRadius = '10px';
      if (follow) v.scrollTop = v.scrollHeight;
      if (op.state === 'running') st.timer = setTimeout(load, 1200);
    };
    load();
  }

  // ------------------------------------------------------------------ events
  document.addEventListener('click', async (e) => {
    const closer = e.target.closest('[data-close]');
    if (closer) { closer.dataset.close === 'modal' ? closeModal() : closeDrawer(); return; }
    if (e.target.id === 'modal') { closeModal(); return; }
    if (e.target.id === 'drawer') { closeDrawer(); return; }
    const el = e.target.closest('[data-action]');
    if (!el || el.disabled) return;
    const { action, id } = el.dataset;
    if (action === 'select') {
      if (el.checked) S.selected.add(id); else S.selected.delete(id);
      renderPage(); return;
    }
    if (action === 'select-all') {
      const visible = [...document.querySelectorAll('[data-action=select]:not(:disabled)')].map((x) => x.dataset.id);
      if (el.checked) visible.forEach((v) => S.selected.add(v)); else visible.forEach((v) => S.selected.delete(v));
      renderPage(); return;
    }
    if (action === 'clear-selection') { S.selected.clear(); renderPage(); return; }
    if (action === 'stop-selected') { bulkStopModal([...S.selected]); return; }
    if (action === 'delete-selected') { deleteModal([...S.selected]); return; }
    if (action === 'delete') { deleteModal([id]); return; }
    if (action === 'stop-all') { bulkStopModal(S.data.instances.filter(isAlive).map((r) => r.id)); return; }
    if (action === 'copy') { e.stopPropagation(); copy(el.dataset.value, el.dataset.label || 'Value'); }
    else if (action === 'deploy') deployModal(el.dataset.service, el.dataset.instance || '');
    else if (action === 'stop') stopModal(id);
    else if (action === 'secrets') secretsModal(id);
    else if (action === 'logs') logDrawer(id);
    else if (action === 'op') openOpDrawer(id);
    else if (action === 'inst-filter') { S.instFilter = el.dataset.value; renderPage(); }
    else if (action === 'cat-filter') { S.catFilter = el.dataset.value; renderPage(); }
    else if (action === 'doctor') runDoctor(false);
    else if (action === 'doctor-online') runDoctor(true);
    else if (action === 'health') {
      el.disabled = true; el.classList.add('spin');
      try {
        const h = await api(`/api/instances/${encodeURIComponent(id)}/health`);
        toast(h.ok ? 'ok' : 'bad', h.ok ? `${id} is healthy` : `${id} probe failed`, `HTTP ${h.status}${h.latency_ms != null ? ` · ${h.latency_ms} ms` : ''}`);
      } catch (err) { toast('bad', 'Probe failed', err.message); }
      el.disabled = false; el.classList.remove('spin');
    }
  });

  $('#deploy-btn').onclick = () => S.data && deployModal();
  $('#refresh-btn').onclick = async (e) => {
    e.currentTarget.classList.add('spin');
    await Promise.all([refresh(), refreshUsage(true)]);
    setTimeout(() => $('#refresh-btn').classList.remove('spin'), 300);
  };
  $('#search').oninput = (e) => { S.search = e.target.value.trim().toLowerCase(); renderPage(); };
  $('#menu-btn').onclick = () => $('#app').classList.toggle('nav-open');
  $('#scrim').onclick = () => $('#app').classList.remove('nav-open');

  function applyTheme(t) {
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
    const dark = t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    $('#theme-btn use').setAttribute('href', dark ? '#i-sun' : '#i-moon');
  }
  $('#theme-btn').onclick = () => {
    const current = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    store.set('localStorage', 'colabllm.theme', next);
    applyTheme(next);
  };
  applyTheme(store.get('localStorage', 'colabllm.theme'));

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { if ($('#modal').classList.contains('open')) closeModal(); else closeDrawer(); }
    if (e.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { e.preventDefault(); $('#search').focus(); }
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });

  function route() {
    const r = (location.hash.match(/^#\/(\w+)/) || [])[1];
    S.route = PAGES[r] ? r : 'overview';
    $('#app').classList.remove('nav-open');
    renderChrome();
    renderPage();
    if (S.route === 'diagnostics' && !S.doctor && !S.doctorBusy && S.data) runDoctor(false);
    window.scrollTo(0, 0);
  }
  window.addEventListener('hashchange', () => {
    if (adoptHashToken()) {
      // New server run: forget anything cached from the previous one.
      $('#locked').hidden = true;
      S.optionSchemas = {}; S.lastHtml = ''; S.doctor = null; S.error = null;
      start();
    }
    route();
  });

  // ------------------------------------------------------------------ boot
  let started = false;
  function start() {
    route();
    refresh().then(() => { if (S.route === 'diagnostics' && !S.doctor) runDoctor(false); });
    refreshUsage();
    if (started) return;
    started = true;
    setInterval(() => refreshUsage(), 60000);
    setInterval(tickUpdated, 1000);
  }
  if (token) start(); else showLocked();
})();

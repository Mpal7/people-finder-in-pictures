/* Family Photo Finder - frontend (vanilla, no external resources) */
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

async function api(url, body) {
  const opt = body !== undefined
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(url, opt);
  if (!r.ok) { let t = await r.text(); try { t = JSON.parse(t).detail || t; } catch {} throw new Error(t); }
  return r.json();
}

async function apiDelete(url) {
  const r = await fetch(url, { method: 'DELETE' });
  if (!r.ok) { let t = await r.text(); try { t = JSON.parse(t).detail || t; } catch {} throw new Error(t); }
  return r.json();
}

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtH = s => { if (s == null) return ''; s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return h ? `${h}h ${m}m` : (m ? `${m}m ${s % 60}s` : `${s}s`); };
const fmtMB = b => (b / 2 ** 20).toFixed(1) + ' MB';

let S = null, view = 'dashboard', person = null, personSlug = null;
const ui = { models: new Set(['w600k']), flip: true, det: '1280', topFaces: 1000, scanSrc: '',
             match: { fusion: 'mean', minZ: '', weights: '', negWeight: 0.5, groupSimilar: true,
                      advanced: false },
             faces: { min_faces: 1, min_det: 0.5, min_face_pct: 0, folder: '', sort: 'faces',
                      shots: 'hide', dest: 'found/people', items: [], total: 0, bytes: 0, faces: 0 } };
window.ui = ui;
window.uiModel = el => { el.checked ? ui.models.add(el.value) : ui.models.delete(el.value); };

function progBar(p, extraClass = '') {
  if (!p || !p.total) return '';
  const pct = Math.min(100, 100 * (p.done || 0) / p.total);
  return `<div class="bar ${p.finished ? 'done' : ''} ${extraClass}"><div style="width:${pct}%"></div></div>`;
}

function jobTable(jobs, showPerson = false) {
  if (!jobs.length) return '<div class="muted">no jobs yet</div>';
  return `<table><thead><tr><th>Job</th><th>Status</th><th></th></tr></thead><tbody>${jobs.map(j => {
    const p = j.progress_data || {};
    const eta = p.eta_s ? ` &middot; ETA ${fmtH(p.eta_s)}` : '';
    const cnt = p.total ? `${p.done || 0}/${p.total}${eta}` : '';
    const detail = p.detail ? ` &middot; ${esc(p.detail)}` : '';
    const log = (j.log_tail && j.log_tail.length)
      ? `<details><summary class="muted">log</summary><pre class="log">${esc(j.log_tail.join('\n'))}</pre></details>` : '';
    const stop = j.status === 'running' ? `<button class="secondary" onclick="stopJob('${j.id}')">Stop</button>` : '';
    const who = showPerson && j.meta && j.meta.person ? ` <span class="pill">${esc(j.meta.person)}</span>` : '';
    return `<tr><td>${esc(j.title)}${who}<div class="muted">${esc(j.kind)} &middot; ${j.started}${j.ended ? ' &rarr; ' + j.ended : ''}${detail}</div>${progBar(p)}<div class="muted">${cnt}</div>${log}</td><td><span class="badge ${j.status}">${j.status}</span></td><td>${stop}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ---------------------------------------------------------------- views -- */
function renderHeader() {
  if (!S) return;
  $('#rootInfo').innerHTML = S.root_ok
    ? `collection: <span title="${esc(S.root)}">${esc(S.root.split(/[\\/]/).pop())}</span> &middot; ${S.index.images_ok}/${S.index.total || '?'} indexed &middot; ${S.index.faces} faces`
    : '<b style="color:#e88">collection drive not mounted</b>';
}

function renderDashboard() {
  const idx = S.index;
  const cfg = idx.config || {};
  const detMismatch = cfg.det_sizes && String(ui.det).replace(/\s+/g, '') !== cfg.det_sizes.join(',');
  const flipMismatch = cfg.det_sizes && ui.flip !== !!cfg.flip;
  const cfgLine = cfg.det_sizes
    ? `current index: models ${(cfg.models || []).join(', ') || '-'} &middot; det ${cfg.det_sizes.join(',')} &middot; flip ${cfg.flip ? 'on' : 'off'} &middot; max side ${cfg.max_side || '-'}`
    : 'no index config yet';
  const cfgWarn = (detMismatch || flipMismatch)
    ? '<div class="muted" style="color:#e0a44a">Changing detection size or flip TTA re-detects every image on the next index run.</div>'
    : '';
  let progress = '';
  if (idx.active_job && idx.active_job.progress_data && idx.active_job.progress_data.total) progress = progBar(idx.active_job.progress_data) + `<div class="muted">${idx.active_job.progress_data.done}/${idx.active_job.progress_data.total} &middot; ${idx.running ? 'ETA ' + fmtH(idx.eta_s) : ''}</div>`;
  else if (idx.running && idx.total) progress = progBar({ done: idx.images_ok, total: idx.total, finished: false }) + `<div class="muted">${idx.images_ok}/${idx.total} &middot; ETA ${fmtH(idx.eta_s)}</div>`;

  const cov = Object.entries(S.coverage).map(([m, c]) =>
    `<tr><td>${m}</td><td>${c.count.toLocaleString()}</td><td>${c.pct}%</td><td style="width:40%">${progBar({ done: c.pct, total: 100, finished: c.pct > 99.5 })}</td></tr>`).join('');

  const modelChecks = ['w600k', 'adaface', 'lvface'].map(m =>
    `<label><input type="checkbox" class="mdl" value="${m}" ${ui.models.has(m) ? 'checked' : ''} onchange="uiModel(this)"> ${m}</label>`).join(' ');

  $('#view-dashboard').innerHTML = `
  <div class="card">
    <div class="row" style="justify-content: space-between">
      <b>Face index</b>
      ${idx.running
        ? `<button class="danger" onclick="stopIndex()">Stop indexing</button>`
        : `<div class="row">${modelChecks}
             <label><input type="checkbox" id="flip" ${ui.flip ? 'checked' : ''} onchange="ui.flip=this.checked"> flip TTA</label>
             <label>det <input type="number" id="detSizes" value="${esc(ui.det)}" style="width:80px" onchange="ui.det=this.value"></label>
             <button onclick="startIndex()">Index / add models</button>
             <button class="danger" onclick="resetIndex()">Reset index...</button></div>`}
    </div>
    ${progress}
    <div class="muted" style="margin-top:8px">${cfgLine}</div>
    ${cfgWarn}
    <div class="kv" style="margin-top:10px">
      <div><b>${idx.images_ok.toLocaleString()}${idx.total ? ' / ' + idx.total.toLocaleString() : ''}</b><span>images indexed</span></div>
      <div><b>${idx.faces.toLocaleString()}</b><span>faces indexed</span></div>
      <div><b>${idx.images_error}</b><span>unreadable files</span></div>
      <div><b>${idx.db_mb} MB</b><span>database</span></div>
      <div><b>${S.found_total}</b><span>photos in found/</span></div>
    </div>
    <div class="muted" style="margin-top:6px">Adding a model later only computes the missing embeddings (no re-detection); an interrupted run resumes where it stopped.</div>
  </div>

  <div class="card">
    <b>Embedding coverage</b>
    <table><thead><tr><th>model</th><th>faces</th><th>%</th><th></th></tr></thead><tbody>${cov}</tbody></table>
    <div class="muted" style="margin-top:6px">w600k is the base; AdaFace/LVFace sharpen the ranking, especially for old/low-quality photos.</div>
  </div>

  <div class="card">
    <b>Settings</b>
    <div class="row" style="margin-top:8px">
      <label style="flex:1">collection root <input type="text" id="rootInput" value="${esc(S.root)}" style="width:70%"></label>
      <button class="secondary" onclick="saveRoot()">Save</button>
    </div>
    <div class="muted">e.g. the NTFS drive mounted read-only, folder "Foto".</div>
    <div class="row" style="margin-top:10px;border-top:1px solid #262b34;padding-top:8px">
      <span class="muted">thumbnail cache: ${fmtMB((S.thumbs_cache || {}).bytes || 0)} / ${fmtMB((S.thumbs_cache || {}).max_bytes || 0)} &middot; ${(S.thumbs_cache || {}).files || 0} files</span>
      <span style="flex:1"></span>
      <button class="secondary" onclick="clearThumbs()">Clear cache</button>
    </div>
  </div>

  <h2>Recent jobs</h2>
  <div class="card">${jobTable(S.jobs.slice(0, 8))}</div>`;
}

function renderPeople() {
  const cards = S.people.map(p => `
    <div class="card person-card">
      <div><b>${esc(p.name)}</b> <span class="pill">${p.slug}</span>
        <div class="muted">${p.runs} match run(s)</div></div>
      <button onclick="openPerson('${p.slug}')">Open</button>
    </div>`).join('') || '<div class="muted">no people yet</div>';

  $('#view-people').innerHTML = `
  <div class="card">
    <b>Add a person</b>
    <div class="row" style="margin-top:8px">
      <input type="text" id="personName" placeholder="name (e.g. Alice)">
      <button onclick="createPerson()">Create</button>
    </div>
    <div class="muted">After creating: drop 5–20 photos of the person into <code>people/&lt;slug&gt;/refs_src/</code> then click Scan.</div>
  </div>
  ${cards}`;
}

function renderPerson() {
  const p = person;
  if (!p) { $('#view-person').innerHTML = '<div class="muted">loading...</div>'; return; }
  const steps = [];
  steps.push(`<div class="step ${p.refs_count ? 'ok' : ''}">
    <h3>1. Reference photos (${p.refs_count})</h3>
    <div class="muted">folder: <code>${esc(p.refs_dir)}</code><br>drop photos into <code>people/${p.slug}/refs_src/</code>, then scan.</div>
    <div class="row" style="margin-top:8px">
      <input type="text" id="scanSrc" placeholder="people/${p.slug}/refs_src" value="${esc(ui.scanSrc)}" style="width:60%" onchange="ui.scanSrc=this.value">
      <button onclick="scanRefs('${p.slug}')">Scan folder</button>
    </div>
    <div class="muted" style="margin-top:6px">look-alike negatives: <b>${p.neg_refs_count || 0}</b> in <code>people/${p.slug}/refs_neg/</code> (optional; used automatically by match)</div></div>`);
  steps.push(`<div class="step ${p.pick_selection.exists ? 'ok' : ''}">
    <h3>2. Pick the face</h3>
    <div class="muted">${p.picker_exists ? 'picker page ready' : 'scan first'}
      ${p.pick_selection.exists ? ` &middot; ${p.pick_selection.picked} picked (saved)` : ''}</div>
    <div class="row" style="margin-top:8px">
      <button ${p.picker_exists ? '' : 'disabled'} onclick="openPicker('${p.slug}')">Open picker</button>
      <button ${p.pick_selection.exists ? '' : 'disabled'} onclick="applyPicks('${p.slug}')">Apply picks</button>
    </div></div>`);
  const m = ui.match;
  steps.push(`<div class="step ${p.runs.length ? 'ok' : ''}">
    <h3>3. Match</h3>
    <div class="muted">models: ${(S.coverage && Object.entries(S.coverage).filter(([, c]) => c.count > 0).map(([name]) => name)) || []}</div>
    <div class="row" style="margin-top:8px">
      <label>top faces <input type="number" id="topFaces" value="${esc(ui.topFaces)}" style="width:90px" onchange="ui.topFaces=this.value"></label>
      <label>fusion <select onchange="ui.match.fusion=this.value">
        ${[['mean', 'mean'], ['max', 'max']].map(([v, l]) => `<option value="${v}" ${m.fusion === v ? 'selected' : ''}>${l}</option>`).join('')}</select></label>
      <label>min z <input type="number" step="0.5" value="${esc(m.minZ)}" style="width:80px" onchange="ui.match.minZ=this.value" placeholder="none"></label>
      <label><input type="checkbox" ${m.groupSimilar ? 'checked' : ''} onchange="ui.match.groupSimilar=this.checked"> group similar photos</label>
    </div>
    <details style="margin-top:6px" ${m.advanced ? 'open' : ''} ontoggle="ui.match.advanced=this.open"><summary class="muted">advanced</summary>
      <div class="row">
        <label>weights <input type="text" placeholder="w600k=1,adaface=1" value="${esc(m.weights)}" style="width:190px" onchange="ui.match.weights=this.value"></label>
        <label>negative weight <input type="number" step="0.1" min="0" value="${esc(m.negWeight)}" style="width:70px" onchange="ui.match.negWeight=this.value"></label>
      </div>
      <div class="muted">Negative weight > 0 suppresses faces that resemble the photos in refs_neg/. 0 disables the penalty.</div>
    </details>
    <div class="row" style="margin-top:8px">
      <button ${p.refs_count ? '' : 'disabled'} onclick="runMatch('${p.slug}')">Run match</button>
    </div></div>`);

  const runs = p.runs.slice().reverse().map(r => {
    const fin = r.meta && r.meta.finished ? ` &middot; ${r.meta.faces_kept} faces / ${r.meta.candidate_images} images` : '';
    const status = r.failed
      ? '<span class="badge error">failed</span>'
      : (r.job_status === 'running' || r.job_status === 'stopping')
        ? '<span class="badge running">running</span>'
        : (r.review_exists ? '<span class="badge done">review ready</span>' : '<span class="muted">-</span>');
    return `<tr>
      <td>${r.ts} ${status}<div class="muted">top ${r.top_faces} &middot; ${r.fusion || 'mean'}${r.negatives ? ` &middot; ${r.negatives} neg` : ''}${fin}</div></td>
      <td>${r.review_exists ? `<a href="${esc(r.review_url || `/files/${r.dir}/review/index.html`)}" target="_blank">open review</a>` : '<span class="muted">not ready</span>'}</td>
      <td>${r.selection_n ? `${r.selection_n} selected` : '<span class="muted">-</span>'}</td>
      <td>${r.selection_n ? `<button onclick="exportRun('${p.slug}','${r.ts}')">Copy to found/</button>` : ''}</td>
      <td><button class="secondary" onclick="deleteRun('${p.slug}','${r.ts}')">Delete</button></td>
    </tr>`;
  }).join('');

  $('#view-person').innerHTML = `
  <div class="card row" style="justify-content: space-between">
    <div><b style="font-size:16px">${esc(p.name)}</b>
      <div class="muted">${p.refs_count} refs (${p.kps_count} with landmarks) &middot; ${p.neg_refs_count || 0} negatives &middot; ${p.found_count} photos exported</div></div>
    <div class="row">
      <button class="secondary" onclick="go('people')">Back</button>
      <button class="danger" onclick="deletePerson('${p.slug}')">Delete person</button>
    </div>
  </div>
  <div class="steps">${steps.join('')}</div>
  <h2>Match runs</h2>
  <div class="card">${runs ? `<table><thead><tr><th>run</th><th>review</th><th>selection</th><th></th><th></th></tr></thead><tbody>${runs}</tbody></table>` : '<div class="muted">none yet</div>'}</div>
  <h2>Jobs for this person</h2>
  <div class="card">${jobTable(p.jobs)}</div>
  <div id="exportMsg" class="muted"></div>`;
}

function facesParams() {
  const f = ui.faces;
  return `min_faces=${f.min_faces}&min_det=${f.min_det}&min_face_pct=${f.min_face_pct}` +
         `&folder=${encodeURIComponent(f.folder)}&sort=${f.sort}&shots=${f.shots}`;
}

function renderFaces() {
  const f = ui.faces;
  const cards = f.items.map(it => {
    const shot = it.shot_class === 2 ? '<span class="pill shot">screenshot</span>'
      : (it.shot_class === 1 ? '<span class="pill">possible</span>' : '');
    return `
    <div class="fcard">
      <a href="/api/faces/view?path=${encodeURIComponent(it.path)}" target="_blank">
        <img loading="lazy" src="/api/faces/thumb?path=${encodeURIComponent(it.path)}&px=320" alt=""></a>
      <div class="fmeta">
        <span class="pill">${it.n_faces} face${it.n_faces > 1 ? 's' : ''}</span>
        <span class="pill">${it.max_face_pct}%</span>
        <span class="pill">${(it.size / 2 ** 20).toFixed(1)} MB</span>
        ${shot}
      </div>
      <div class="muted fpath" title="${esc(it.path)}${it.shot_reason ? ' | ' + esc(it.shot_reason) : ''}">${esc(it.path)}</div>
    </div>`;
  }).join('');
  const left = f.total - f.items.length;
  $('#view-faces').innerHTML = `
  <div class="card">
    <b>Photos with people</b> <span class="muted">&mdash; filter the indexed collection for photos containing faces</span>
    <div class="row" style="margin-top:8px">
      <label>min faces <input type="number" id="fmin" min="1" value="${f.min_faces}" style="width:70px" onchange="ui.faces.min_faces=+this.value"></label>
      <label>min detection score <select onchange="ui.faces.min_det=+this.value">
        ${[0.3, 0.5, 0.65].map(v => `<option value="${v}" ${f.min_det === v ? 'selected' : ''}>${v}</option>`).join('')}</select></label>
      <label>face size &ge; <select onchange="ui.faces.min_face_pct=+this.value">
        ${[0, 3, 5, 8, 12, 20].map(v => `<option value="${v}" ${f.min_face_pct === v ? 'selected' : ''}>${v}%</option>`).join('')}</select> of frame</label>
      <label>folder contains <input type="text" id="ffolder" value="${esc(f.folder)}" style="width:170px" onchange="ui.faces.folder=this.value" placeholder="e.g. 2024"></label>
      <label>sort <select onchange="ui.faces.sort=this.value;facesApply()">
        ${[['faces', 'most faces'], ['recent', 'newest'], ['path', 'path']].map(([v, l]) => `<option value="${v}" ${f.sort === v ? 'selected' : ''}>${l}</option>`).join('')}</select></label>
      <label>screenshots <select onchange="ui.faces.shots=this.value;facesApply()">
        ${[['hide', 'hide'], ['hide2', 'hide + possible'], ['all', 'show all'], ['only', 'only screenshots']].map(([v, l]) => `<option value="${v}" ${f.shots === v ? 'selected' : ''}>${l}</option>`).join('')}</select></label>
      <button onclick="facesApply()">Apply filters</button>
    </div>
    <div class="row" style="margin-top:10px">
      <span class="muted"><b>${f.total.toLocaleString()}</b> photos &middot; ${f.faces.toLocaleString()} faces &middot; ${fmtMB(f.bytes)}</span>
      <span style="flex:1"></span>
      <label>copy to folder <input type="text" id="fdest" value="${esc(f.dest)}" style="width:230px" onchange="ui.faces.dest=this.value" placeholder="found/people  or  /absolute/path"></label>
      <button onclick="facesExport()" ${f.total ? '' : 'disabled'}>Copy ${f.total.toLocaleString()} photos</button>
      <a href="/api/faces/list.csv?${facesParams()}" target="_blank"><button class="secondary">CSV list</button></a>
    </div>
    <div class="muted" style="margin-top:4px">Relative paths land in the project folder (e.g. <code>found/people</code>); absolute paths (e.g. <code>/home/you/Pictures/people</code>) are used as-is. Copies only, originals never moved. Big exports run as a job (Jobs tab).</div>
    ${shotStatusRow()}
  </div>
  <div class="fgrid">${cards}</div>
  ${left > 0 ? `<div style="text-align:center;margin:14px"><button onclick="facesMore()">Load more (${left.toLocaleString()} left)</button></div>` : ''}`;
}

function renderJobs() {
  $('#view-jobs').innerHTML = `<div class="card">${jobTable(S.jobs, true)}</div>`;
}

function shotStatusRow() {
  const sh = (S && S.screenshots) || {};
  if (!sh.total) return '';
  let status, buttons;
  if (sh.running) {
    status = `scanning... (${sh.analyzed}/${sh.total} analyzed)`;
    buttons = '';
  } else if (sh.analyzed < sh.total) {
    status = `screenshot scan: ${sh.analyzed}/${sh.total} analyzed`;
    buttons = `<button class="secondary" onclick="shotsScan(false)">Run screenshot scan</button>`;
  } else {
    status = `screenshot scan: <b>${sh.screenshots}</b> screenshots, <b>${sh.possible}</b> possible`;
    buttons = sh.possible ? `<button class="secondary" onclick="shotsScan(true)">Refine possible (visual pass)</button>` : '';
  }
  return `<div class="row" style="margin-top:8px;border-top:1px solid #262b34;padding-top:8px">
    <span class="muted">${status}</span><span style="flex:1"></span>${buttons}</div>`;
}

function render() {
  if (!S) return;
  renderHeader();
  $$('nav button').forEach(b => b.classList.toggle('active', b.dataset.view === view));
  $('#view-dashboard').hidden = view !== 'dashboard';
  $('#view-people').hidden = view !== 'people';
  $('#view-faces').hidden = view !== 'faces';
  $('#view-person').hidden = view !== 'person';
  $('#view-jobs').hidden = view !== 'jobs';
  const ae = document.activeElement;
  const typing = ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA');
  if (typing) return;
  if (view === 'dashboard') renderDashboard();
  else if (view === 'people') renderPeople();
  else if (view === 'faces') renderFaces();
  else if (view === 'person') renderPerson();
  else if (view === 'jobs') renderJobs();
}

async function tick() {
  try { S = await api('/api/state'); } catch (e) { console.error(e); return; }
  if (view === 'person' && personSlug) {
    try { person = await api('/api/people/' + personSlug); } catch {}
  }
  render();
}

/* -------------------------------------------------------------- actions -- */
async function wrapped(fn) { try { await fn(); } catch (e) { alert('Error: ' + e.message); } }

window.go = v => {
  view = v;
  if (v === 'faces' && ui.faces.items.length === 0 && !ui.faces.loading) facesApply();
  render();
};

window.startIndex = () => wrapped(async () => {
  const models = [...ui.models];
  if (!models.length) return alert('select at least one model');
  await api('/api/index/start', { models, flip: ui.flip, det_sizes: ui.det || '1280' });
  tick();
});
window.stopIndex = () => wrapped(async () => { await api('/api/index/stop', {}); tick(); });
window.saveRoot = () => wrapped(async () => { await api('/api/config', { root: $('#rootInput').value.trim() }); tick(); });

window.stopJob = jid => wrapped(async () => { await api(`/api/jobs/${jid}/stop`, {}); tick(); });

window.createPerson = () => wrapped(async () => {
  const name = $('#personName').value.trim();
  if (!name) return;
  const r = await api('/api/people', { name });
  openPerson(r.person.slug);
});

window.openPerson = async slug => {
  personSlug = slug; view = 'person';
  person = await api('/api/people/' + slug);
  render();
};

window.scanRefs = slug => wrapped(async () => {
  const src = (ui.scanSrc || '').trim();
  await api(`/api/people/${slug}/refs/scan`, { src: src || null });
  tick();
});

window.openPicker = slug => window.open(`/files/people/${slug}/refs_pick/index.html`, '_blank');

window.applyPicks = slug => wrapped(async () => { await api(`/api/people/${slug}/refs/apply`, {}); tick(); });

window.runMatch = slug => wrapped(async () => {
  const top = parseInt(ui.topFaces || '1000', 10);
  const m = ui.match;
  const body = {
    top_faces: top, fusion: m.fusion, weights: (m.weights || '').trim(),
    neg_weight: parseFloat(m.negWeight) || 0, group_similar: !!m.groupSimilar,
  };
  if (String(m.minZ).trim() !== '') body.min_z = parseFloat(m.minZ);
  await api(`/api/people/${slug}/match`, body);
  tick();
});

window.deleteRun = (slug, ts) => wrapped(async () => {
  if (!confirm(`Delete match run ${ts}? Its review page and selection will be removed. Photos already copied to found/ are kept.`)) return;
  await apiDelete(`/api/people/${slug}/runs/${ts}`);
  person = await api('/api/people/' + slug);
  tick();
});

window.deletePerson = slug => wrapped(async () => {
  if (!confirm(`Delete person "${slug}"? This removes people/${slug}/ (references, picks and all runs). Photos already copied to found/${slug}/ are kept.`)) return;
  await apiDelete(`/api/people/${slug}`);
  view = 'people'; person = null; personSlug = null;
  tick();
});

window.resetIndex = () => wrapped(async () => {
  if (!confirm('Delete the face index? All indexed faces/embeddings are removed. Your photos, references and runs are untouched.')) return;
  if (!confirm('Really reset the index? This cannot be undone (a rebuild will take hours).')) return;
  await api('/api/index/reset', { confirm: true });
  tick();
});

window.clearThumbs = () => wrapped(async () => {
  const r = await api('/api/cache/clear', {});
  alert(`Removed ${r.removed} cached thumbnails.`);
  tick();
});

window.exportRun = (slug, ts) => wrapped(async () => {
  const msg = $('#exportMsg');
  msg.textContent = 'starting copy job...';
  const r = await api(`/api/people/${slug}/export`, { ts });
  msg.textContent = `copy job started -> ${r.dest} (watch the Jobs tab)`;
  tick();
});

window.facesApply = () => wrapped(async () => {
  const f = ui.faces;
  if (f.loading) return;
  f.loading = true;
  try {
    const r = await api(`/api/faces/images?${facesParams()}&limit=60&offset=0`);
    f.items = r.items; f.total = r.total; f.bytes = r.bytes_total; f.faces = r.faces_total;
  } finally { f.loading = false; }
  render();
});

window.facesMore = () => wrapped(async () => {
  const f = ui.faces;
  if (f.loading) return;
  f.loading = true;
  try {
    const r = await api(`/api/faces/images?${facesParams()}&limit=60&offset=${f.items.length}`);
    f.items = f.items.concat(r.items);
  } finally { f.loading = false; }
  render();
});

window.facesExport = () => wrapped(async () => {
  const f = ui.faces;
  if (!f.total) return;
  if (!confirm(`Copy ${f.total.toLocaleString()} photos (${fmtMB(f.bytes)}) to:\n${f.dest}\n\nOriginals stay untouched.`)) return;
  const r = await api('/api/faces/export', {
    min_faces: f.min_faces, min_det: f.min_det, min_face_pct: f.min_face_pct,
    folder: f.folder, sort: f.sort, shots: f.shots, dest: f.dest,
  });
  alert(`Copying ${r.count.toLocaleString()} photos to ${r.dest} - watch the Jobs tab.`);
  go('jobs');
});

window.shotsScan = visual => wrapped(async () => {
  await api('/api/screenshots/analyze', { visual });
  alert(visual ? 'Visual refinement started - watch the Jobs tab.' : 'Screenshot scan started - watch the Jobs tab.');
  tick();
});

/* ----------------------------------------------------------------- init -- */
$$('nav button').forEach(b => b.addEventListener('click', () => go(b.dataset.view)));
tick();
setInterval(tick, 2000);

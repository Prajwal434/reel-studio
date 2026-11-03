// Reel Studio UI. Talks to the Rust shell (invoke) which runs the Python engine and streams JSON-line events.
const { invoke, convertFileSrc } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const dialog = window.__TAURI__.dialog;

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const sep = navigator.userAgent.includes('Windows') ? '\\' : '/';
const join = (...a) => a.join(sep).replace(/[\\/]+/g, sep);

let settings = {};
let project = null; // current project.json (+ dir)
let saveTimer = null;

// ---------------------------------------------------------------- engine bridge
const jobs = new Map();
listen('engine', ({ payload }) => {
  const j = jobs.get(payload.job);
  if (!j) return;
  if (payload.stderr) { j.stderr = payload.stderr; return; }
  let ev;
  try { ev = JSON.parse(payload.line); } catch { return; }
  if (ev.type === 'done') j.result = ev;
  else if (ev.type === 'error') j.error = ev.msg;
  j.onEvent?.(ev);
});

async function engine(args, onEvent) {
  const job = Math.random().toString(36).slice(2);
  const j = { onEvent };
  jobs.set(job, j);
  try {
    const code = await invoke('run_engine', { job, args });
    if (code !== 0 || j.error) throw new Error(j.error || lastLine(j.stderr) || `Engine stopped (code ${code})`);
    return j.result;
  } finally { jobs.delete(job); }
}
const lastLine = (s) => (s || '').trim().split('\n').filter(Boolean).slice(-1)[0];

// ---------------------------------------------------------------- views
function show(view) {
  $$('.view').forEach((v) => v.classList.toggle('hidden', v.id !== `view-${view}`));
  if (view !== 'project') $$('.plist button').forEach((b) => b.classList.remove('on'));
}

async function refreshProjects() {
  const list = settings.projects_dir ? await invoke('list_projects', { dir: settings.projects_dir }) : [];
  const nav = $('#projectList');
  nav.innerHTML = '';
  for (const p of list) {
    const b = document.createElement('button');
    b.innerHTML = `${esc(p.name || 'Untitled')}<small>${{ cover: 'Singing cover', talk: 'Talking reel', skit: 'AI skit' }[p.template]}${p.analyzed ? '' : ' · not analyzed'}</small>`;
    b.onclick = () => openProject(p.dir, b);
    if (project && project.dir === p.dir) b.classList.add('on');
    nav.appendChild(b);
  }
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---------------------------------------------------------------- settings
async function loadSettings() {
  settings = await invoke('load_settings');
  // packaged app: always use the engine inside the app; development build: this checkout's engine + your Python
  const d = await invoke('engine_defaults');
  if (d.bundled || !settings.python) settings.python = d.python;
  if (d.bundled || !settings.engine || !(await invoke('path_exists', { path: settings.engine }))) settings.engine = d.engine;
  settings.projects_dir ||= d.projects_dir;
  settings.provider ||= 'none';
  if (d.bundled) document.querySelectorAll('#sPython, #sEngine').forEach((e) => (e.closest('.field').style.display = 'none'));
}

function fillSettings() {
  $('#sProvider').value = settings.provider;
  $('#sKey').value = settings.api_key || '';
  $('#sBase').value = settings.base_url || '';
  $('#sPython').value = settings.python;
  $('#sEngine').value = settings.engine;
  $('#sProjects').value = settings.projects_dir;
  if (settings.provider === 'anthropic' && settings.model) $('#sModelClaude').value = settings.model;
  if (settings.provider === 'openai') $('#sModelOther').value = settings.model || '';
  syncProviderFields();
}

function syncProviderFields() {
  const p = $('#sProvider').value;
  $$('.ai').forEach((e) => e.classList.toggle('hidden', p === 'none'));
  $('.openai-only').classList.toggle('hidden', p !== 'openai');
  $('#sModelClaude').classList.toggle('hidden', p !== 'anthropic');
  $('#sModelOther').classList.toggle('hidden', p !== 'openai');
}

function readSettingsForm() {
  const provider = $('#sProvider').value;
  return {
    ...settings, provider,
    api_key: $('#sKey').value.trim(),
    model: provider === 'anthropic' ? $('#sModelClaude').value : $('#sModelOther').value.trim(),
    base_url: $('#sBase').value.trim(),
    python: $('#sPython').value.trim(),
    engine: $('#sEngine').value.trim(),
    projects_dir: $('#sProjects').value.trim(),
  };
}

async function saveSettings() {
  settings = readSettingsForm();
  await invoke('save_settings', { settings });
  $('#savedMsg').textContent = 'Saved';
  setTimeout(() => ($('#savedMsg').textContent = ''), 1500);
  refreshProjects();
}

async function checkEngine(quiet) {
  try {
    const r = await engine(['check']);
    $('#engineDot').className = 'dot ' + (r.ok ? 'ok' : 'bad');
    $('#engineDot').title = r.ok ? 'Engine ready' : 'Engine incomplete: open Settings';
    if (quiet && r.ok === false && !r.separator_model && Object.values(r.modules).every(Boolean)) { fillSettings(); show('settings'); $('#setupMsg').textContent = 'First run: download the AI models to start editing.'; }
    if (!quiet) {
      const missing = [...Object.entries(r.tools), ...Object.entries(r.modules)].filter(([, v]) => !v).map(([k]) => k);
      if (!r.separator) missing.push('audio-separator');
      if (!r.separator_model) missing.push('separation model');
      $('#engineResult').innerHTML = r.ok ? '<span style="color:var(--ok)">Engine ready</span>' : `<span class="error">Missing: ${missing.join(', ')}</span>`;
    }
  } catch (e) {
    $('#engineDot').className = 'dot bad';
    if (!quiet) $('#engineResult').innerHTML = `<span class="error">${esc(e.message)}</span>`;
  }
}

// ---------------------------------------------------------------- new project
let newVideo = null;
const STEP_LABELS = { probe: 'Reading the video', separate: 'Separating voice and instrument', transcribe: 'Listening for the words',
  face: 'Finding your face', ai: 'AI checks the words', mix: 'Polishing the sound' };

function renderSteps(state) {
  const tpl = document.querySelector('input[name=tpl]:checked').value;
  const ai = settings.provider !== 'none' && settings.api_key ? ['ai'] : [];
  const order = tpl === 'skit' ? ['probe', 'transcribe', 'face', ...ai] : ['probe', ...(tpl === 'cover' ? ['separate'] : []), 'transcribe', ...ai, 'mix'];
  $('#steps').innerHTML = order.map((k) => {
    const s = state[k] || {};
    const cls = s.fail ? 'fail' : s.pct >= 1 ? 'done' : s.pct != null ? 'run' : '';
    const icon = s.fail ? '✕' : s.pct >= 1 ? '✓' : s.pct != null ? '◌' : '·';
    const pct = s.pct != null && s.pct < 1 && s.pct > 0 ? `${Math.round(s.pct * 100)}%` : '';
    return `<li class="${cls}">${icon} ${STEP_LABELS[k]}<span class="pct">${pct}</span></li>`;
  }).join('');
}

async function createProject() {
  $('#newError').textContent = '';
  if (!newVideo) return ($('#newError').textContent = 'Choose a video first.');
  const template = document.querySelector('input[name=tpl]:checked').value;
  const name = $('#projName').value.trim() || newVideo.split(/[\\/]/).pop().replace(/\.[^.]+$/, '');
  const slug = name.replace(/[^A-Za-z0-9 _-]+/g, '').trim().replace(/\s+/g, '-') || 'reel';
  let dir = join(settings.projects_dir, slug);
  for (let i = 2; await invoke('path_exists', { path: dir }); i++) dir = join(settings.projects_dir, `${slug}-${i}`);
  const p = { name, video: newVideo, template, title_card: true };
  if (template === 'cover') p.song = { title: $('#songTitle').value.trim(), artist: $('#songArtist').value.trim() };
  if (template === 'skit') { p.ai_name = $('#aiName').value.trim() || 'the AI'; p.vocab = $('#vocab').value.trim(); }
  await invoke('write_json', { path: join(dir, 'project.json'), value: p });
  refreshProjects();
  $('#createBtn').disabled = true;
  $('#analyzeBox').classList.remove('hidden');
  const state = {};
  renderSteps(state);
  try {
    await engine(['analyze', dir], (ev) => {
      if (ev.type !== 'progress') return;
      for (const k of Object.keys(STEP_LABELS)) if (k === ev.step) break; else if (state[k]?.pct != null && state[k].pct < 1) state[k].pct = 1;
      state[ev.step] = { pct: ev.pct };
      renderSteps(state);
    });
    await openProject(dir);
  } catch (e) {
    const cur = Object.keys(state).pop();
    if (cur) state[cur].fail = true;
    renderSteps(state);
    $('#newError').textContent = e.message;
  } finally {
    $('#createBtn').disabled = false;
  }
}

function resetNew() {
  newVideo = null;
  $('#videoName').textContent = 'No video chosen';
  ['#songTitle', '#songArtist', '#projName', '#aiName', '#vocab'].forEach((s) => ($(s).value = ''));
  $('#analyzeBox').classList.add('hidden');
  $('#newError').textContent = '';
  syncTemplate();
  show('new');
}
function syncTemplate() {
  const tpl = document.querySelector('input[name=tpl]:checked').value;
  $$('#view-new .cover-only').forEach((e) => e.classList.toggle('hidden', tpl !== 'cover'));
  $$('#view-new .skit-only').forEach((e) => e.classList.toggle('hidden', tpl !== 'skit'));
}

// ---------------------------------------------------------------- project editor
const player = $('#player');
const LOOK_CSS = { daylight: 'brightness(.93) contrast(1.12) saturate(.94)', moody: 'saturate(.7) brightness(.86) contrast(1.04) sepia(.06)', warm_film: 'sepia(.22) saturate(.95) brightness(.96)',
  clean: 'contrast(1.05) saturate(1.04)', cool_night: 'saturate(.68) brightness(.86) hue-rotate(-8deg)', noir: 'grayscale(1) contrast(1.15) brightness(.95)' };
const LOOKS = [
  ['moody', 'Moody', 'Muted, teal shadows, warm skin, faded blacks'],
  ['warm_film', 'Warm film', 'Golden, soft contrast, light grain'],
  ['clean', 'Clean', 'Natural colour, a little contrast'],
  ['cool_night', 'Cool night', 'Blue, dim, quiet'],
  ['daylight', 'Daylight fix', 'Bright window take: recovers the wall, adds depth'],
  ['noir', 'Black & white', 'Deep blacks and grain'],
];

async function openProject(dir, btn) {
  const p = await invoke('read_json', { path: join(dir, 'project.json') });
  p.dir = dir;
  project = p;
  $$('.plist button').forEach((b) => b.classList.remove('on'));
  btn?.classList.add('on');
  if (!p.analyzed) {
    // project was created but analysis didn't finish: offer to run it again
    show('new');
    newVideo = p.video; $('#videoName').textContent = p.video.split(/[\\/]/).pop();
    $('#projName').value = p.name;
    $('#newError').textContent = 'This project was not analyzed yet. Press Create & analyze to run it.';
    return;
  }
  show('project');
  $('#pName').textContent = p.name;
  const song = p.song?.title ? ` · ${p.song.title}${p.song.artist ? ' – ' + p.song.artist : ''}` : '';
  const kind = { cover: 'Singing cover', talk: 'Talking reel', skit: 'AI skit' }[p.template];
  $('#pMeta').textContent = `${kind}${song} · ${p.duration.toFixed(1)} s`;
  const skit = p.template === 'skit';
  $$('.skit-tab').forEach((b) => b.classList.toggle('hidden', !skit));
  $$('.noskit-tab').forEach((b) => b.classList.toggle('hidden', skit));
  if (skit) { renderReplies(); fillDetails(); }
  $('#aiNote').textContent = p.ai_note || '';
  $$('#view-project .cover-only').forEach((e) => e.classList.toggle('hidden', p.template !== 'cover'));
  $('#ovLine').classList.toggle('talk', p.template !== 'cover');
  player.src = convertFileSrc(p.video);
  player.style.filter = LOOK_CSS[p.look] || '';
  renderLines();
  fillTrim();
  renderOutputs();
  $('#capText').value = p.caption || '';
  selectTab('words');
  refreshProjects();
}

function saveProject() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const { dir, ...data } = project;
    invoke('write_json', { path: join(dir, 'project.json'), value: data });
  }, 300);
}

function selectTab(t) {
  $$('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === t));
  $$('.tab').forEach((e) => e.classList.toggle('hidden', e.id !== `tab-${t}`));
  if (t === 'look') renderLooks();
}

// words
function renderLines() {
  const box = $('#lines');
  box.innerHTML = '';
  project.lines.sort((a, b) => a.start - b.start);
  project.lines.forEach((l, i) => {
    const r = document.createElement('div');
    const skit = project.template === 'skit';
    r.className = 'lrow' + (skit ? ' skit' : '') + (l.skip ? ' skipped' : '');
    r.innerHTML = `<input class="t" type="number" step="0.05" value="${l.start}" title="Start (s)" />
      <input class="t" type="number" step="0.05" value="${l.end}" title="End (s)" />
      <input value="${esc(l.text)}" placeholder="(no text: nothing shown)" />
      ${skit ? `<input type="checkbox" title="Cut this line out of the video" ${l.skip ? '' : 'checked'} />` : ''}
      <button title="Delete line">✕</button>`;
    if (skit) r.querySelector('input[type=checkbox]').onchange = (ev) => { l.skip = !ev.target.checked; r.classList.toggle('skipped', l.skip); saveProject(); };
    const [s, e, t] = r.querySelectorAll('input:not([type=checkbox])');
    s.onchange = () => { l.start = +s.value; saveProject(); };
    e.onchange = () => { l.end = +e.value; saveProject(); };
    t.oninput = () => { l.text = t.value; saveProject(); };
    r.querySelector('button').onclick = () => { project.lines.splice(i, 1); saveProject(); renderLines(); };
    r.onclick = (ev) => { if (ev.target.tagName !== 'BUTTON' && ev.target !== t && ev.target.type !== 'checkbox') { player.currentTime = l.start; } };
    box.appendChild(r);
  });
}

function addLine() {
  const t = +player.currentTime.toFixed(2);
  project.lines.push({ start: t, end: +(t + 2.5).toFixed(2), text: '' });
  saveProject(); renderLines();
}

// live overlay preview
function tick() {
  if (project && !$('#view-project').classList.contains('hidden')) {
    const t = player.currentTime;
    const lines = project.lines.filter((l) => l.text.trim() && !l.skip);
    let cur = null;
    lines.forEach((l, i) => {
      const nxt = lines[i + 1]?.start ?? 1e9;
      const a = project.template === 'cover' ? l.start - 0.15 : l.start;
      const b = project.template === 'cover' ? Math.min(l.end + 0.45, nxt - 0.05) : l.end;
      if (t >= a && t <= b) cur = l;
    });
    const ov = $('#ovLine');
    ov.textContent = cur ? cur.text : '';
    ov.style.opacity = cur ? 1 : 0;
    $$('.lrow').forEach((r, i) => r.classList.toggle('now', project.lines[i] === cur));
    const rel = t - project.trim.in;
    const showTitle = project.template === 'cover' && project.title_card && project.song?.title && rel > 0.4 && rel < 6;
    $('#ovTitle').classList.toggle('hidden', !showTitle);
    if (showTitle) {
      $('#ovTitle i').textContent = project.song.title;
      $('#ovTitle small').textContent = `${project.song.artist ? project.song.artist + ' · ' : ''}cover`;
    }
    // preview only the trimmed range
    if (!player.paused && t >= project.trim.out) { player.pause(); player.currentTime = project.trim.out; }
  }
  requestAnimationFrame(tick);
}

// look
async function renderLooks() {
  const box = $('#looks');
  box.innerHTML = '';
  const t = project.lines[0] ? (project.lines[0].start + project.lines[0].end) / 2 : project.duration / 3;
  const cards = LOOKS.map(([id, label, desc]) => {
    const b = document.createElement('button');
    b.className = 'look' + (project.look === id ? ' on' : '');
    b.innerHTML = `<div class="img">…</div><div class="t"><b>${label}</b><span>${desc}</span></div>`;
    b.onclick = () => {
      project.look = id; saveProject();
      player.style.filter = LOOK_CSS[id];
      $$('.look').forEach((x) => x.classList.toggle('on', x === b));
    };
    box.appendChild(b);
    return [id, b.querySelector('.img')];
  });
  for (const [id, img] of cards) {
    const path = join(project.dir, 'stills', `${id}.jpg`);
    if (!(await invoke('path_exists', { path }))) {
      try { await engine(['still', project.dir, id, String(t)]); } catch (e) { img.textContent = 'Preview failed'; continue; }
    }
    img.textContent = '';
    img.style.backgroundImage = `url("${convertFileSrc(path)}?v=${Date.now()}")`;
  }
}

// trim & sound
function fillTrim() {
  $('#trimIn').value = project.trim.in;
  $('#trimOut').value = project.trim.out;
  $('#fadeOut').value = project.trim.fade_out;
  $('#titleCard').checked = project.title_card !== false;
  $('#instDb').value = project.inst_db ?? -3;
  $('#instVal').textContent = `${project.inst_db ?? -3} dB`;
}

// export
async function render(target) {
  await flushSave();
  $$('[data-render]').forEach((b) => (b.disabled = true));
  $('#renderBox').classList.remove('hidden');
  const set = (pct, msg) => { $('#renderBar').style.width = `${Math.round(pct * 100)}%`; if (msg) $('#renderMsg').textContent = msg; };
  const W = project.template === 'skit'
    ? { voice: [0, 0.06], mix: [0.06, 0.12], render: [0.12, 0.97], mux: [0.97, 1] }
    : { mix: [0, 0.08], render: [0.08, 0.95], mux: [0.95, 1] };
  set(0, target === 'yt' ? 'Rendering 4K: this takes a few minutes' : 'Rendering');
  try {
    const r = await engine(['render', project.dir, target], (ev) => {
      if (ev.type === 'progress' && W[ev.step]) set(W[ev.step][0] + (W[ev.step][1] - W[ev.step][0]) * ev.pct, ev.msg);
    });
    set(1, `Done · ${(r.size / 1e6).toFixed(1)} MB`);
    project.exports = [...new Set([r.path, ...(project.exports || [])])];
    saveProject(); renderOutputs();
  } catch (e) {
    $('#renderMsg').innerHTML = `<span class="error">${esc(e.message)}</span>`;
  } finally {
    $$('[data-render]').forEach((b) => (b.disabled = false));
  }
}

function renderOutputs() {
  $('#outputs').innerHTML = '';
  for (const path of project.exports || []) {
    const d = document.createElement('div');
    d.className = 'out';
    d.innerHTML = `<span>${esc(path.split(/[\\/]/).pop())}</span><button class="btn">Show in folder</button>`;
    d.querySelector('button').onclick = () => invoke('reveal', { path });
    $('#outputs').appendChild(d);
  }
}

async function writeCaption() {
  $('#capBtn').disabled = true;
  $('#capText').value = 'Writing…';
  try {
    const r = await engine(['caption', project.dir, $('#capPlatform').value]);
    $('#capText').value = r.text;
    project.caption = r.text; saveProject();
  } catch (e) {
    $('#capText').value = '';
    alert(e.message);
  } finally { $('#capBtn').disabled = false; }
}

// ---------------------------------------------------------------- AI skit: replies + details
const VOICES = [['am_michael', 'Michael (US, calm)'], ['am_adam', 'Adam (US)'], ['am_eric', 'Eric (US)'], ['am_onyx', 'Onyx (US, deep)'],
  ['bm_george', 'George (UK)'], ['bm_lewis', 'Lewis (UK)'], ['af_heart', 'Heart (US, female)'], ['af_bella', 'Bella (US, female)'],
  ['bf_emma', 'Emma (UK, female)'], ['hm_omega', 'Omega (Hindi accent)']];
$('#aiVoice').innerHTML = VOICES.map(([v, l]) => `<option value="${v}">AI voice: ${l}</option>`).join('');

function renderReplies() {
  $('#aiVoice').value = project.ai_voice || 'am_michael';
  const box = $('#replies');
  box.innerHTML = '';
  (project.replies || []).sort((a, b) => a.start - b.start).forEach((r, i) => {
    const d = document.createElement('div');
    const mode = r.mode || 'reply';
    d.className = `reply ${mode}`;
    d.innerHTML = `<div class="rhead"><span>Pause at ${r.start.toFixed(1)} s · ${(r.end - r.start).toFixed(1)} s long</span>
      <select><option value="reply">AI reply</option><option value="demo">Product demo</option><option value="cut">Cut this pause</option></select>
      <span class="sp fit"></span><button class="btn">▶</button></div>
      <textarea placeholder="What the AI says here…">${esc(r.text || '')}</textarea>`;
    const sel = d.querySelector('select'), ta = d.querySelector('textarea'), fit = d.querySelector('.fit'), play = d.querySelector('button');
    sel.value = mode;
    ta.classList.toggle('hidden', mode !== 'reply'); play.classList.toggle('hidden', mode !== 'reply');
    // rough length check before synthesis: ~2.9 words/s at the app's pace
    const est = () => { const n = (ta.value.match(/\S+/g) || []).length; const secs = n / 2.9; const room = r.end - r.start - 0.3;
      fit.textContent = n ? (secs <= room ? `fits (~${secs.toFixed(1)} s)` : `long (~${secs.toFixed(1)} s): will be sped up`) : '';
      fit.className = 'sp fit ' + (n && secs > room ? 'fit-bad' : 'fit-ok'); };
    est();
    sel.onchange = () => { r.mode = sel.value; if (r.mode === 'demo') project.replies.forEach((o) => { if (o !== r && o.mode === 'demo') o.mode = 'reply'; });
      saveProject(); renderReplies(); };
    ta.oninput = () => { r.text = ta.value; est(); saveProject(); };
    d.querySelector('.rhead span').onclick = () => (player.currentTime = Math.max(0, r.start - 2));
    play.onclick = async () => {
      play.disabled = true; play.textContent = '…';
      await flushSave();
      try {
        const res = await engine(['voice', project.dir, String(i)]);
        fit.textContent = res.fits ? `fits (${res.seconds} s)` : `${res.seconds} s: too long for this pause, will be sped up`;
        fit.className = 'sp fit ' + (res.fits ? 'fit-ok' : 'fit-bad');
        const a = $('#voicePlayer'); a.src = `${convertFileSrc(res.path)}?v=${Date.now()}`; a.play();
      } catch (e) { fit.textContent = e.message; fit.className = 'sp fit fit-bad'; }
      play.disabled = false; play.textContent = '▶';
    };
    box.appendChild(d);
  });
}

function fillDetails() {
  const dm = (project.demo ||= { video: '', texts: ['', '', '', '', ''] });
  const ec = (project.endcard ||= {});
  $('#dHeader').value = project.header || '';
  $('#demoName').textContent = dm.video ? dm.video.split(/[\\/]/).pop() : 'None';
  $('#dZoom').value = dm.zoom ?? 0.32; $('#zoomVal').textContent = `${Math.round((dm.zoom ?? 0.32) * 100)}% of the width`;
  $('#eTitle').value = ec.title || ''; $('#eSub').value = ec.subtitle || ''; $('#eCta').value = ec.cta || ''; $('#eUrl').value = ec.url || '';
  $('#sIn').value = project.trim.in; $('#sOut').value = project.trim.out;
  const shots = $('#demoShots');
  shots.innerHTML = '';
  for (let k = 0; k < 5; k++) {
    const row = document.createElement('div'); row.className = 'shot';
    row.innerHTML = `<input type="number" step="0.5" min="0" placeholder="auto" title="Start time in the recording (s)" value="${dm.starts?.[k] ?? ''}" />
      <input placeholder="Text for shot ${k + 1}" value="${esc(dm.texts?.[k] || '')}" />
      <input type="number" step="0.05" min="0.15" max="1" placeholder="zoom" title="Zoom for this shot (blank = default)" value="${dm.zooms?.[k] ?? ''}" />`;
    const [st, tx, zm] = row.querySelectorAll('input');
    st.onchange = () => { dm.starts ||= []; dm.starts[k] = st.value === '' ? null : +st.value; if (dm.starts.every((v) => v == null)) delete dm.starts; saveProject(); };
    tx.oninput = () => { dm.texts ||= ['', '', '', '', '']; dm.texts[k] = tx.value; saveProject(); };
    zm.onchange = () => { dm.zooms ||= []; dm.zooms[k] = zm.value === '' ? null : +zm.value; saveProject(); };
    shots.appendChild(row);
  }
}
// demo starts: the engine treats a missing start as "auto"; fill gaps so the list stays aligned
function flushSave() {
  clearTimeout(saveTimer);
  const { dir, ...data } = project;
  if (data.demo?.starts) {
    if (data.demo.starts.some((v) => v == null)) data.demo = { ...data.demo, starts: undefined };
  }
  return invoke('write_json', { path: join(dir, 'project.json'), value: data });
}

// ---------------------------------------------------------------- wire up
$('#newBtn').onclick = resetNew;
$('#welcomeNew').onclick = resetNew;
$('#settingsBtn').onclick = () => { fillSettings(); show('settings'); };
$$('input[name=tpl]').forEach((r) => (r.onchange = syncTemplate));
$('#pickVideo').onclick = async () => {
  const f = await dialog.open({ multiple: false, filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'm4v', 'MOV', 'MP4'] }] });
  if (f) { newVideo = f; $('#videoName').textContent = f.split(/[\\/]/).pop(); }
};
$('#createBtn').onclick = createProject;
$('#sProvider').onchange = syncProviderFields;
$('#saveSettings').onclick = saveSettings;
$('#pickProjects').onclick = async () => { const d = await dialog.open({ directory: true }); if (d) $('#sProjects').value = d; };
$('#checkEngine').onclick = async () => { await saveSettings(); $('#engineResult').textContent = 'Checking…'; checkEngine(false); };
$('#testAi').onclick = async () => {
  await saveSettings();
  $('#aiResult').textContent = 'Testing…';
  try { const r = await engine(['llm-test']); $('#aiResult').innerHTML = `<span style="color:var(--ok)">Connected: “${esc(r.reply)}”</span>`; }
  catch (e) { $('#aiResult').innerHTML = `<span class="error">${esc(e.message)}</span>`; }
};
$$('.tabs button').forEach((b) => (b.onclick = () => selectTab(b.dataset.tab)));
$('#addLine').onclick = addLine;
$$('[data-set]').forEach((b) => (b.onclick = () => { const id = b.dataset.set; $('#' + id).value = player.currentTime.toFixed(2); $('#' + id).onchange(); }));
$('#trimIn').onchange = () => { project.trim.in = Math.max(0, +$('#trimIn').value); saveProject(); };
$('#trimOut').onchange = () => { project.trim.out = Math.min(project.duration, +$('#trimOut').value); saveProject(); };
$('#fadeOut').onchange = () => { project.trim.fade_out = +$('#fadeOut').value; saveProject(); };
$('#titleCard').onchange = () => { project.title_card = $('#titleCard').checked; saveProject(); };
$('#instDb').oninput = () => { project.inst_db = +$('#instDb').value; $('#instVal').textContent = `${project.inst_db} dB`; saveProject(); };
$$('[data-render]').forEach((b) => (b.onclick = () => render(b.dataset.render)));
$('#capBtn').onclick = writeCaption;
$('#aiVoice').onchange = () => { project.ai_voice = $('#aiVoice').value; saveProject(); };
$('#draftReplies').onclick = async () => {
  if (settings.provider === 'none' || !settings.api_key) { $('#draftMsg').textContent = 'Add an AI key in Settings to draft replies.'; return; }
  $('#draftMsg').textContent = 'Writing…'; await flushSave();
  try {
    const r = await engine(['suggest-replies', project.dir]);
    project.replies.sort((a, b) => a.start - b.start).forEach((x, i) => { if ((x.mode || 'reply') === 'reply' && r.replies[i]) x.text = r.replies[i]; });
    saveProject(); renderReplies(); $('#draftMsg').textContent = 'Drafted. Edit anything you like.';
  } catch (e) { $('#draftMsg').textContent = e.message; }
};
$('#dHeader').oninput = () => { project.header = $('#dHeader').value; saveProject(); };
$('#dZoom').oninput = () => { project.demo.zoom = +$('#dZoom').value; $('#zoomVal').textContent = `${Math.round(project.demo.zoom * 100)}% of the width`; saveProject(); };
[['#eTitle', 'title'], ['#eSub', 'subtitle'], ['#eCta', 'cta'], ['#eUrl', 'url']].forEach(([id, k]) => ($(id).oninput = () => { project.endcard[k] = $(id).value; saveProject(); }));
$('#pickDemo').onclick = async () => {
  const f = await dialog.open({ multiple: false, filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'm4v', 'MOV', 'MP4'] }] });
  if (f) { project.demo.video = f; $('#demoName').textContent = f.split(/[\\/]/).pop(); saveProject(); }
};
$('#sIn').onchange = () => { project.trim.in = +$('#sIn').value; saveProject(); };
$('#sOut').onchange = () => { project.trim.out = +$('#sOut').value; saveProject(); };
$$('[data-set2]').forEach((b) => (b.onclick = () => { const id = b.dataset.set2; $('#' + id).value = player.currentTime.toFixed(2); $('#' + id).onchange(); }));
$('#setupBtn').onclick = async () => {
  $('#setupBtn').disabled = true;
  try {
    await engine(['setup'], (ev) => { if (ev.type === 'progress') { $('#setupBar').style.width = `${ev.pct * 100}%`; if (ev.msg) $('#setupMsg').textContent = ev.msg; } });
    $('#setupMsg').textContent = 'All models ready'; checkEngine(true);
  } catch (e) { $('#setupMsg').innerHTML = `<span class="error">${esc(e.message)}</span>`; }
  $('#setupBtn').disabled = false;
};
$('#capCopy').onclick = () => navigator.clipboard.writeText($('#capText').value);
player.addEventListener('play', () => { if (player.currentTime < project.trim.in || player.currentTime >= project.trim.out) player.currentTime = project.trim.in; });

(async () => {
  await loadSettings();
  await refreshProjects();
  show('welcome');
  checkEngine(true);
  requestAnimationFrame(tick);
})();

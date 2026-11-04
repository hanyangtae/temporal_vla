'use strict';
// Fold27 reward viewer front-end. No external libraries; all text set via textContent.
// Series identity (color + dash) is fixed per key everywhere; toggles hide/show a key in every decomposition chart.
const SERIES = {
  basic: { label: 'basic', color: '#2a78d6', dash: '' },
  mix: { label: 'MIX', color: '#eb6834', dash: '6 3' },
  alpha_term: { label: 'α항 = α·(anchor − Φ)', color: '#1baf7a', dash: '' },
  minus_beta_cost: { label: '−βC (실제 차감)', color: '#eda100', dash: '6 3' },
  delta: { label: 'Δ = MIX − basic (합계)', color: '#e87ba4', dash: '2 3' },
  phi: { label: 'Φ (저장값)', color: '#6250d6', dash: '' },
};
const TOGGLEABLE = ['alpha_term', 'minus_beta_cost', 'delta'];
const NS = 'http://www.w3.org/2000/svg';
const $ = (id) => document.getElementById(id);
const state = { meta: null, fold: null, episodes: [], ep: null, means: null, hidden: new Set(), cursors: [], raf: 0 };

function el(tag, attrs, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
function sv(tag, attrs) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  return e;
}
const fmt = (v) => (v === null || v === undefined || Number.isNaN(v)) ? '—'
  : (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(3));
async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(url + ' → ' + r.status);
  return r.json();
}
function setOptions(sel, items, keep) {
  const prev = keep ? sel.value : null;
  sel.replaceChildren(...items.map(([v, t]) => el('option', { value: v }, t)));
  if (prev && items.some(([v]) => v === prev)) sel.value = prev;
}
const cumOn = () => $('chk-cum').checked;
const cumsum = (v) => { let s = 0; return v.map((x) => (s += x)); };

// ---------------------------------------------------------------- toggles (shared, URL hash: #hide=a,b)
function readHash() {
  const m = /(?:^|[#&])hide=([a-z_,]*)/.exec(location.hash);
  state.hidden = new Set(m ? m[1].split(',').filter((k) => TOGGLEABLE.includes(k)) : []);
}
function toggle(key) {
  if (state.hidden.has(key)) state.hidden.delete(key); else state.hidden.add(key);
  history.replaceState(null, '', state.hidden.size ? '#hide=' + [...state.hidden].join(',') : location.pathname);
  renderMeans(); if (state.ep) renderCharts();
}

// ---------------------------------------------------------------- generic step chart
// o: {title, sub, keys, values:{key:[...]}, x0:[...], x1:[...], xlabel, zero, term (index|-1), toggle, nt, tipHead(i)}
function chart(o) {
  const box = el('div', { class: 'chart' });
  box.append(el('h3', {}, o.title), el('p', { class: 'sub' }, o.sub));
  const L = el('div', { class: 'legend' });
  for (const k of o.keys) {
    const s = SERIES[k], hidden = o.toggle && state.hidden.has(k);
    const item = o.toggle ? el('button', { class: 'tog' + (hidden ? ' off' : ''), type: 'button', 'data-key': k,
      'aria-pressed': hidden ? 'false' : 'true', title: hidden ? '보이기' : '가리기' }) : el('span');
    const sw = sv('svg', { width: 22, height: 8 });
    sw.append(sv('line', { x1: 0, y1: 4, x2: 22, y2: 4, stroke: s.color, 'stroke-width': 2, 'stroke-dasharray': s.dash }));
    item.append(sw, document.createTextNode((o.toggle ? (hidden ? '보이기: ' : '가리기: ') : '') + s.label));
    if (o.toggle) item.addEventListener('click', () => toggle(k));
    L.append(item);
  }
  if (o.keys.length > 1 || o.toggle) box.append(L);
  const data = o.keys.filter((k) => !(o.toggle && state.hidden.has(k)) && o.values[k]).map((k) => [k, o.values[k]]);
  const n = o.x0.length;
  if (!data.length || !n) { box.append(el('p', { class: 'missing' }, n ? '표시할 시리즈 없음 (모두 숨김)' : '데이터 없음')); return box; }
  let lo = Infinity, hi = -Infinity;
  for (const [, v] of data) for (const y of v) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
  if (o.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const W = 1000, H = 190, ml = 56, mr = 170, mt = 8, mb = 26, xmax = o.x1[n - 1];
  const X = (x) => ml + (W - ml - mr) * x / xmax, Y = (y) => mt + (H - mt - mb) * (hi - y) / (hi - lo);
  const g = sv('svg', { class: 'plot', 'data-xmax': xmax, viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', role: 'img', 'aria-label': o.title });
  for (let t = 0; t <= 4; t++) {
    const y = lo + (hi - lo) * t / 4;
    g.append(sv('line', { class: 'grid', x1: ml, x2: W - mr, y1: Y(y), y2: Y(y) }));
    const tx = sv('text', { x: ml - 6, y: Y(y) + 4, 'text-anchor': 'end' }); tx.textContent = fmt(y); g.append(tx);
  }
  const step = xmax > 400 ? 100 : xmax > 200 ? 50 : xmax > 80 ? 20 : 10;
  for (let x = 0; x <= xmax; x += step) {
    const tx = sv('text', { x: X(x), y: H - 8, 'text-anchor': 'middle' }); tx.textContent = String(x); g.append(tx);
  }
  const xl = sv('text', { x: W - mr, y: H - 8, 'text-anchor': 'start' }); xl.textContent = ' ' + o.xlabel; g.append(xl);
  if (o.zero && lo < 0 && hi > 0) g.append(sv('line', { class: 'zero', x1: ml, x2: W - mr, y1: Y(0), y2: Y(0) }));
  if (o.term >= 0) {
    g.append(sv('line', { class: 'term', x1: X(o.x0[o.term]), x2: X(o.x0[o.term]), y1: mt, y2: H - mb }));
    const tt = sv('text', { x: X(o.x0[o.term]) - 4, y: mt + 10, 'text-anchor': 'end' });
    tt.textContent = o.termLabel || 'terminal'; g.append(tt);
  }
  const labels = [];
  for (const [k, v] of data) {
    const s = SERIES[k]; let d = '';
    for (let i = 0; i < n; i++) d += (i ? 'L' : 'M') + X(o.x0[i]).toFixed(2) + ',' + Y(v[i]).toFixed(2) + 'L' + X(o.x1[i]).toFixed(2) + ',' + Y(v[i]).toFixed(2);
    g.append(sv('path', { class: 'series', 'data-key': k, d, stroke: s.color, 'stroke-dasharray': s.dash, 'vector-effect': 'non-scaling-stroke' }));
    labels.push([Y(v[n - 1]), s.label + ' ' + fmt(v[n - 1])]);
  }
  labels.sort((a, b) => a[0] - b[0]);
  let last = -Infinity;
  for (const [y, t] of labels) {
    const yy = Math.max(y + 4, last + 13); last = yy;
    const tx = sv('text', { x: W - mr + 6, y: yy }); tx.textContent = t; g.append(tx);
  }
  const cross = sv('line', { class: 'cross', x1: 0, x2: 0, y1: mt, y2: H - mb, visibility: 'hidden' });
  const hit = sv('rect', { x: ml, y: mt, width: W - ml - mr, height: H - mt - mb, fill: 'transparent' });
  g.append(cross, hit);
  const idxAt = (ev) => {
    const r = g.getBoundingClientRect();
    const es = ((ev.clientX - r.left) / r.width * W - ml) / (W - ml - mr) * xmax;
    const i = o.x1.findIndex((e1) => es < e1);
    return i < 0 ? n - 1 : i;          // the actual inference (chunk) under the pointer
  };
  if (o.cursor) {   // video cursor: individual rollout charts only (never on outcome means)
    const vc = sv('line', { class: 'vcur', x1: 0, x2: 0, y1: mt, y2: H - mb, visibility: 'hidden' });
    g.insertBefore(vc, hit);
    state.cursors.push((xv) => {
      if (xv === null) { vc.setAttribute('visibility', 'hidden'); return; }
      vc.setAttribute('x1', X(xv)); vc.setAttribute('x2', X(xv)); vc.setAttribute('visibility', 'visible');
    });
    if (o.onPick) hit.addEventListener('click', (ev) => o.onPick(idxAt(ev)));
  }
  box.append(g);
  let strip = null;
  if (o.nt) {   // N_t on its own axis (count, not reward): separate strip, same x mapping
    const SH = 64, nmax = Math.max(...o.nt);
    const YN = (c) => 6 + (SH - 22) * (1 - c / nmax);
    strip = sv('svg', { class: 'plot strip', viewBox: `0 0 ${W} ${SH}`, preserveAspectRatio: 'none', role: 'img', 'aria-label': 'N_t' });
    let d = '';
    for (let i = 0; i < n; i++) d += (i ? 'L' : 'M') + X(o.x0[i]).toFixed(2) + ',' + YN(o.nt[i]).toFixed(2) + 'L' + X(o.x1[i]).toFixed(2) + ',' + YN(o.nt[i]).toFixed(2);
    strip.append(sv('line', { class: 'grid', x1: ml, x2: W - mr, y1: YN(0), y2: YN(0) }),
      sv('path', { class: 'series nt', d, stroke: '#52514e', 'vector-effect': 'non-scaling-stroke' }));
    for (const [c, yy] of [[nmax, YN(nmax) + 4], [0, YN(0) + 4]]) {
      const tx = sv('text', { x: ml - 6, y: yy, 'text-anchor': 'end' }); tx.textContent = String(c); strip.append(tx);
    }
    const lab = sv('text', { x: W - mr + 6, y: YN(nmax) + 10 }); lab.textContent = 'N_t (t에 존재하는 rollout 수)'; strip.append(lab);
    box.append(strip);
  }
  const tip = $('tip');
  hit.addEventListener('mousemove', (ev) => {
    const i = idxAt(ev);
    cross.setAttribute('x1', X(o.x0[i])); cross.setAttribute('x2', X(o.x0[i])); cross.setAttribute('visibility', 'visible');
    tip.replaceChildren(el('div', {}, o.tipHead(i)));
    for (const [k, v] of data) {
      const row = el('div'); const sw = el('span', { class: 'sw' }); sw.style.background = SERIES[k].color;
      row.append(sw, document.createTextNode(`${SERIES[k].label}: ${fmt(v[i])}`)); tip.append(row);
    }
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 320) + 'px'; tip.style.top = (ev.clientY + 12) + 'px';
  });
  hit.addEventListener('mouseleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); });
  return box;
}

// ---------------------------------------------------------------- selection
function foldLabel(f) { return `${f.fold}  (target j${f.target_j} 제외 fit ${f.episodes}판, 성공 ${f.success})`; }
function fillTasks() {
  setOptions($('sel-task'), Object.keys(state.meta.tasks).sort().map((t) => [t, t]), true);
  fillInstr();
}
function fillInstr(noLoad) {
  const ins = state.meta.tasks[$('sel-task').value] || {};
  setOptions($('sel-instr'), Object.keys(ins).sort().map((i) => [i, i]), !noLoad);
  if (!noLoad) fillFolds();
}
function fillFolds(noLoad) {
  const fl = (state.meta.tasks[$('sel-task').value] || {})[$('sel-instr').value] || [];
  setOptions($('sel-fold'), fl.map((f) => [f.fold, foldLabel(f)]), !noLoad);
  if (!noLoad) loadFold();
}
async function loadFold(wantEp) {
  const fold = $('sel-fold').value;
  if (!fold) return;
  const [d, m] = await Promise.all([getJSON('/api/fold?fold=' + encodeURIComponent(fold)),
    getJSON('/api/fold_means?fold=' + encodeURIComponent(fold))]);
  state.fold = fold; state.episodes = d.episodes; state.means = m;
  renderCoef(); renderMeans(); renderSha();
  fillEpisodes(wantEp);
}
function fillEpisodes(wantEp) {
  const oc = $('sel-outcome').value;
  let eps = state.episodes.filter((e) => oc === 'all' || (oc === 'success') === e.success);
  if (wantEp !== undefined && !eps.some((e) => e.ep === wantEp)) { $('sel-outcome').value = 'all'; eps = state.episodes; }
  setOptions($('sel-ep'), eps.map((e) => [String(e.ep),
    `j${e.j} n${e.n} · ${e.success ? '성공' : '실패'} · ${e.env_steps} env step · ${e.decisions} decision`]), wantEp === undefined);
  if (wantEp !== undefined) $('sel-ep').value = String(wantEp);
  loadEpisode();
}
async function loadEpisode() {
  const ep = $('sel-ep').value;
  stopVideo();
  if (ep === '') { state.ep = null; $('video').replaceChildren(el('p', { class: 'missing' }, '선택된 rollout 없음')); $('charts').replaceChildren(el('p', { class: 'missing' }, '해당 조건의 rollout 없음')); return; }
  state.ep = await getJSON(`/api/episode?fold=${encodeURIComponent(state.fold)}&ep=${encodeURIComponent(ep)}`);
  renderHead(); renderVideo(); renderCharts(); renderTable(); renderDups();
}

// ---------------------------------------------------------------- fold-level panels
function renderCoef() {
  const c = state.means.coefficients, box = $('coef');
  if (c.alpha === null || c.beta === null || c.alpha === undefined || c.beta === undefined) {
    box.replaceChildren(el('b', { class: 'bad' }, 'α / β: 미확인 (receipt에 기록 없음 — 추정하지 않음)'));
    return;
  }
  box.replaceChildren(el('span', { class: 'coef-v' }, `α = ${c.alpha}`), el('span', { class: 'coef-v' }, `β = ${c.beta}`),
    el('span', { class: 'coef-src' }, `fold ${state.fold} · 출처: ${c.source}`));
}
function renderMeans() {
  const box = $('means');
  box.hidden = !$('chk-means').checked;      // user toggle; kept across fold/outcome/cumulative changes
  if (!state.means) return;
  const cum = cumOn();
  const out = [el('h2', {}, `fold ${state.fold} 성공/실패 평균 (inference step 축)`),
    el('p', { class: 'sub' }, (cum ? '누적: rollout별로 먼저 누적합을 구한 뒤, t에 실제 존재하는 rollout만 평균. '
      : 'chunk별 값: t에 실제 존재하는 rollout만 평균. ') +
      '종료 후 0 채움·마지막 값 연장 없음. 후반 t는 표본(N_t)이 줄고 긴 rollout만 남음 — 길이 혼입이므로 N_t와 함께 읽을 것. 원래 시간 정렬(t=0부터) 유지.')];
  for (const [name, ko] of [['success', '성공'], ['failure', '실패']]) {
    const gr = state.means.groups[name];
    if (!gr.total_n) { out.push(el('p', { class: 'missing' }, `${ko}: rollout 0개 — 평균 없음`)); continue; }
    const v = cum ? gr.cum_mean : gr.mean;
    const x0 = gr.t, x1 = gr.t.map((t) => t + 1);
    const head = `${ko} 평균 · 집단 total N=${gr.total_n} · 길이 ${gr.length.min}–${gr.length.max} decision`;
    const tipHead = (i) => `${ko} · inference step ${gr.t[i]} · N_t=${gr.n_t[i]}/${gr.total_n}`;
    out.push(chart({ title: head + ' — basic vs MIX', sub: cum ? '누적 평균' : '평균', keys: ['basic', 'mix'], values: v,
      x0, x1, xlabel: 'inference step', zero: false, term: -1, toggle: false, nt: gr.n_t, tipHead }));
    out.push(chart({ title: head + ' — MIX − basic 분해', sub: (cum ? '누적 평균' : '평균') + ' · 버튼으로 시리즈 보이기/가리기',
      keys: TOGGLEABLE, values: v, x0, x1, xlabel: 'inference step', zero: true, term: -1, toggle: true, nt: gr.n_t, tipHead }));
  }
  box.replaceChildren(...out);
}

// ---------------------------------------------------------------- single rollout
function epValues(keys) {
  const c = state.ep.chunks, v = {};
  for (const k of keys) {
    const raw = k === 'phi' ? state.ep.phi : c[k];
    v[k] = raw && cumOn() && k !== 'phi' ? cumsum(raw) : raw;
  }
  return v;
}
function renderHead() {
  const e = state.ep, h = $('ep-head');
  const kv = el('div', { class: 'kv' });
  const add = (k, v, cls) => { const d = el('div', {}, k + ' '); d.append(el('b', cls ? { class: cls } : {}, v)); kv.append(d); };
  add('episode', e.episode_id);
  add('outcome', e.success ? '성공' : '실패 (timeout)', e.success ? 'ok' : 'bad');
  add('env steps', `${e.env_steps} / max ${e.max_episode_steps}`);
  add('decision', String(e.decisions));
  add('마지막 chunk m', String(e.last_chunk_m) + (e.last_chunk_m < 5 ? ' (부분 chunk)' : ''));
  add('terminal outcome anchor', e.terminal_anchor > 0 ? '+1' : '−1');
  add('cluster ID', '미저장 (Φ로 역추론하지 않음)');
  const c = e.chunks, sum = (a) => a.reduce((x, y) => x + y, 0);
  add('Σbasic', fmt(sum(c.basic))); add('ΣMIX', fmt(sum(c.mix))); add('Σα항', fmt(sum(c.alpha_term)));
  add('Σ(−βC)', fmt(sum(c.minus_beta_cost))); add('ΣΔ', fmt(sum(c.delta)));
  h.replaceChildren(el('h2', {}, `개별 rollout · ${e.instruction} · s${e.scene} · fold ${e.fold}`), kv);
}
function renderCharts() {
  const c = state.ep.chunks, envX = $('sel-x').value === 'env';
  const x0 = envX ? c.env_step_start : c.step, x1 = envX ? c.env_step_start.map((s, i) => s + c.m[i]) : c.step.map((s) => s + 1);
  const term = c.final.indexOf(true);
  const termLabel = `terminal (anchor ${state.ep.terminal_anchor > 0 ? '+1' : '−1'}, m=${c.m[term]})`;
  const tipHead = (i) => `inference step ${c.step[i]} · env step ${c.env_step_start[i]}–${c.env_step_start[i] + c.m[i] - 1} (m=${c.m[i]})${c.final[i] ? ' · terminal' : ''}`;
  const cum = cumOn() ? '누적 합(할인 없음) · ' : 'chunk별 값 · ';
  const xlabel = envX ? 'env step' : 'inference step';
  state.cursors = [];
  const vs = { cursor: true, onPick: seekToInference };
  const out = [
    chart({ title: '개별 rollout — basic vs MIX', sub: cum + '칸 = decision 1회(실행 env step 보통 5, 마지막은 m)',
      keys: ['basic', 'mix'], values: epValues(['basic', 'mix']), x0, x1, xlabel, zero: false, term, termLabel, toggle: false, tipHead, ...vs }),
    chart({ title: '개별 rollout — MIX − basic 분해', sub: cum + 'α항과 실제 차감 −βC 분리, Δ = α항 + (−βC) · 버튼으로 보이기/가리기',
      keys: TOGGLEABLE, values: epValues(TOGGLEABLE), x0, x1, xlabel, zero: true, term, termLabel, toggle: true, tipHead, ...vs }),
  ];
  if (state.ep.phi) out.push(chart({ title: 'Φ (저장된 fold potential 값)', sub: 'cluster ID 미저장(역추론 안 함) · terminal 행 anchor는 Φ 대신 outcome ±1',
    keys: ['phi'], values: epValues(['phi']), x0, x1, xlabel, zero: true, term, termLabel, toggle: false, tipHead, ...vs }));
  else out.push(el('p', { class: 'missing' }, 'Φ 배열 없음 — 표시하지 않음'));
  $('charts').replaceChildren(...out);
  updateCursor();
}
function renderTable() {
  const w = $('table-wrap');
  w.hidden = !$('chk-table').checked;
  if (w.hidden || !state.ep) return;
  const c = state.ep.chunks, keys = ['basic', 'mix', ...TOGGLEABLE];
  const vals = epValues(keys);
  const t = el('table'), hr = el('tr');
  for (const h of ['inference step', 'env step', 'm', ...keys.map((k) => SERIES[k].label), 'Φ', 'terminal']) hr.append(el('th', {}, h));
  t.append(hr);
  for (let i = 0; i < c.step.length; i++) {
    const tr = el('tr');
    tr.append(el('td', {}, String(c.step[i])), el('td', {}, `${c.env_step_start[i]}–${c.env_step_start[i] + c.m[i] - 1}`), el('td', {}, String(c.m[i])));
    for (const k of keys) tr.append(el('td', {}, fmt(vals[k][i])));
    tr.append(el('td', {}, state.ep.phi ? fmt(state.ep.phi[i]) : '—'), el('td', {}, c.final[i] ? 'terminal' : ''));
    t.append(tr);
  }
  w.replaceChildren(el('h2', {}, '표 보기' + (cumOn() ? ' (누적)' : '')), t);
}
function renderDups() {
  const d = $('dups'), e = state.ep;
  d.replaceChildren(el('h2', {}, `fold 간 중복 (${e.duplicates.length}개 fold fit에 포함)`),
    el('p', { class: 'sub missing' }, '같은 rollout: basic은 fold 간 동일, MIX/Φ는 fold별 potential이라 다를 수 있음'));
  const ul = el('div', { class: 'kv' });
  for (const x of e.duplicates) {
    const b = el('button', { class: 'link', type: 'button' }, x.fold + (x.fold === e.fold ? ' (현재)' : ''));
    if (x.fold === e.fold) b.disabled = true;
    b.addEventListener('click', () => goto(x.fold, x.ep));
    ul.append(b);
  }
  d.append(ul);
}
function goto(fold, ep) {
  for (const [t, ins] of Object.entries(state.meta.tasks)) for (const [i, fl] of Object.entries(ins)) {
    if (fl.some((f) => f.fold === fold)) {
      $('sel-task').value = t; fillInstr(true); $('sel-instr').value = i; fillFolds(true); $('sel-fold').value = fold;
      loadFold(ep); return;
    }
  }
}
function stopVideo() {
  // stop + unload the previous rollout's video before anything else is shown (no stale playback/frame/cursor)
  cancelAnimationFrame(state.raf); state.raf = 0;
  for (const v of $('video').querySelectorAll('video')) { v.pause(); v.removeAttribute('src'); v.load(); }
  for (const fn of state.cursors) fn(null);
}
const curVideo = () => $('video').querySelector('video');
function videoFrame() {
  // verified mapping only (probe_video_sync): frame f shown for t in [f/fps, (f+1)/fps); no duration-ratio scaling
  const v = curVideo(), sy = state.ep && state.ep.video_sync;
  if (!v || !sy || v.readyState < 1) return null;
  const f = v.ended ? sy.frames - 1 : Math.min(sy.frames - 1, Math.max(0, Math.floor(v.currentTime * sy.fps + 1e-6)));
  const e = sy.first_frame_env_step + sy.steps_per_render * f;     // observation right after 0-based env step e
  const c = state.ep.chunks;
  let i = 0;
  while (i + 1 < c.env_step_start.length && c.env_step_start[i + 1] <= e) i++;
  return { f, e, i };
}
function updateCursor() {
  const p = videoFrame(), c = state.ep && state.ep.chunks, envX = $('sel-x').value === 'env';
  const lab = $('video').querySelector('.sync-pos');
  if (!p) { for (const fn of state.cursors) fn(null); if (lab) lab.textContent = ''; return; }
  const xv = envX ? p.e + 1 : c.step[p.i] + (p.e + 1 - c.env_step_start[p.i]) / c.m[p.i];   // end of env step e
  for (const fn of state.cursors) fn(xv);
  if (lab) lab.textContent = `영상 frame ${p.f + 1}/${state.ep.video_sync.frames} = env step ${p.e} 실행 직후 · inference step ${c.step[p.i]}`;
}
function tick() {
  updateCursor();
  const v = curVideo();
  state.raf = v && !v.paused && !v.ended ? requestAnimationFrame(tick) : 0;
}
function seekToInference(i) {
  const v = curVideo(), sy = state.ep && state.ep.video_sync;
  if (!v || !sy) return;                                      // unverified mapping: no seek
  const c = state.ep.chunks;
  let f = Math.ceil((c.env_step_start[i] - sy.first_frame_env_step) / sy.steps_per_render);
  f = Math.min(sy.frames - 1, Math.max(0, f));                // first frame at/after this inference's first env step
  let t = (f + 0.5) / sy.fps;                                 // middle of that frame's display interval
  if (Number.isFinite(v.duration)) t = Math.min(t, Math.max(0, v.duration - 0.25 / sy.fps));
  v.currentTime = t;
  updateCursor();
}
function renderVideo() {
  stopVideo();
  const v = $('video'), e = state.ep;
  v.replaceChildren(el('h2', {}, `rollout 영상 · j${e.j} n${e.n} · ${e.success ? '성공' : '실패'}`));
  if (e.video && e.video_id !== null) {
    const vid = el('video', { controls: '', preload: 'metadata', playsinline: '', class: 'rollout-video',
      src: '/video?id=' + e.video_id, 'data-episode': e.episode_id });
    for (const evn of ['loadedmetadata', 'timeupdate', 'seeked', 'pause', 'ended']) vid.addEventListener(evn, updateCursor);
    vid.addEventListener('play', () => { cancelAnimationFrame(state.raf); state.raf = requestAnimationFrame(tick); });
    const sy = e.video_sync;
    v.append(vid, el('p', { class: 'sub missing' }, '원본 archive 영상 · 경로·sha256 검증된 파일만 제공'),
      sy ? el('p', { class: 'sub sync-ok' }, `동기화 검증됨(probe): ${sy.fps} fps · ${sy.frames} frames · frame f = env step ${sy.first_frame_env_step}+${sy.steps_per_render}·f 실행 직후 · 그래프 클릭 → 해당 inference step으로 seek`)
         : el('p', { class: 'sub bad' }, `동기화 미검증 — 세로 커서·클릭 seek 비활성 (${e.video_sync_reason || '검증 기록 없음'})`),
      el('p', { class: 'sub sync-pos' }));
  } else {
    v.append(el('p', { class: 'missing' }, '누락 — 이 rollout의 검증된 영상 없음 (rollout pkl은 열지 않음)'));
  }
}
function renderSha() {
  const s = state.meta.folds[state.fold], box = $('sha');
  const o = s['원본기록_DONE_fingerprint'], r = s['재구성값'];
  const rows = [['rows', o.rows, r.rows], ['basic rewards_sha256', o.basic_rewards_sha256, r.basic_rewards_sha256],
    ['MIX rewards_sha256', o.mix_rewards_sha256, r.mix_rewards_sha256],
    ['observations_sha256 (basic)', o.basic_observations_sha256, r.observations_sha256],
    ['observations_sha256 (MIX)', o.mix_observations_sha256, r.mix_observations_sha256]];
  const t = el('table'), hr = el('tr');
  for (const h of ['항목', '원본 기록 (학습 DONE fingerprint)', '재구성 값', '일치']) hr.append(el('th', {}, h));
  t.append(hr);
  for (const [k, a, b] of rows) {
    const tr = el('tr'); const same = String(a) === String(b);
    tr.append(el('td', {}, k), el('td', { class: 'mono' }, String(a)), el('td', { class: 'mono' }, String(b)),
      el('td', { class: same ? 'ok' : 'bad' }, same ? '✓ 일치' : '✗ 불일치'));
    t.append(tr);
  }
  box.replaceChildren(el('h2', {}, `SHA — fold ${state.fold}`),
    el('p', { class: 'sub missing' }, `재구성 host ${s.host} · 길이/성공 스칼라 출처 ${s.episode_scalars} · npz sha256 ${s.npz_sha256}`), t);
}

// ---------------------------------------------------------------- wiring
async function init() {
  readHash();
  try {
    state.meta = await getJSON('/api/meta');
  } catch (err) { $('prov').textContent = '메타데이터 로드 실패: ' + err.message; return; }
  $('prov').textContent = state.meta.provenance;
  $('sel-task').addEventListener('change', () => fillInstr());
  $('sel-instr').addEventListener('change', () => fillFolds());
  $('sel-fold').addEventListener('change', () => loadFold());
  $('sel-outcome').addEventListener('change', () => fillEpisodes());
  $('sel-ep').addEventListener('change', loadEpisode);
  $('sel-x').addEventListener('change', () => state.ep && renderCharts());
  $('chk-cum').addEventListener('change', () => { renderMeans(); if (state.ep) { renderCharts(); renderTable(); } });
  $('chk-table').addEventListener('change', renderTable);
  $('chk-means').addEventListener('change', () => { $('means').hidden = !$('chk-means').checked; });
  fillTasks();
}
init();

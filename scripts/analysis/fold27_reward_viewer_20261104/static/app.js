'use strict';
// Fold27 reward viewer front-end. No external libraries; all text set via textContent.
const SERIES = {
  basic: { label: 'basic', color: '#2a78d6', dash: '' },
  mix: { label: 'MIX', color: '#eb6834', dash: '6 3' },
  alpha_term: { label: 'α항 = α·(anchor − Φ)', color: '#1baf7a', dash: '' },
  minus_beta_cost: { label: '−βC (실제 차감)', color: '#eda100', dash: '6 3' },
  delta: { label: 'Δ = MIX − basic', color: '#e87ba4', dash: '2 3' },
  phi: { label: 'Φ (저장값)', color: '#6250d6', dash: '' },
};
const NS = 'http://www.w3.org/2000/svg';
const $ = (id) => document.getElementById(id);
const state = { meta: null, fold: null, episodes: [], ep: null };

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

// ---------------------------------------------------------------- selection
function fillTasks() {
  setOptions($('sel-task'), Object.keys(state.meta.tasks).sort().map((t) => [t, t]), true);
  fillInstr();
}
function fillInstr() {
  const ins = state.meta.tasks[$('sel-task').value] || {};
  setOptions($('sel-instr'), Object.keys(ins).sort().map((i) => [i, i]), true);
  fillFolds();
}
function fillFolds() {
  const fl = (state.meta.tasks[$('sel-task').value] || {})[$('sel-instr').value] || [];
  setOptions($('sel-fold'), fl.map((f) => [f.fold, `${f.fold}  (target j${f.target_j} 제외 fit ${f.episodes}판, 성공 ${f.success})`]), true);
  loadFold();
}
async function loadFold(wantEp) {
  const fold = $('sel-fold').value;
  if (!fold) return;
  const d = await getJSON('/api/fold?fold=' + encodeURIComponent(fold));
  state.fold = fold; state.episodes = d.episodes;
  fillEpisodes(wantEp);
  renderSha();
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
  if (ep === '') { $('charts').replaceChildren(el('p', { class: 'missing' }, '해당 조건의 rollout 없음')); return; }
  state.ep = await getJSON(`/api/episode?fold=${encodeURIComponent(state.fold)}&ep=${encodeURIComponent(ep)}`);
  renderAll();
}

// ---------------------------------------------------------------- rendering
function series(key) {
  const c = state.ep.chunks;
  const v = key === 'phi' ? state.ep.phi : c[key];
  if (!v) return null;
  if (!$('chk-cum').checked || key === 'phi') return v;
  let s = 0; return v.map((x) => (s += x));
}
function renderAll() {
  renderHead(); renderCharts(); renderTable(); renderDups(); renderVideo();
}
function renderHead() {
  const e = state.ep, h = $('ep-head');
  const kv = el('div', { class: 'kv' });
  const add = (k, v, cls) => { const d = el('div', {}, k + ' '); const b = el('b', cls ? { class: cls } : {}, v); d.append(b); kv.append(d); };
  add('episode', e.episode_id);
  add('outcome', e.success ? '성공' : '실패 (timeout)', e.success ? 'ok' : 'bad');
  add('env steps', `${e.env_steps} / max ${e.max_episode_steps}`);
  add('decision', String(e.decisions));
  add('마지막 chunk m', String(e.last_chunk_m) + (e.last_chunk_m < 5 ? ' (부분 chunk)' : ''));
  add('terminal outcome anchor', (e.terminal_anchor > 0 ? '+1' : '−1'));
  add('cluster ID', '미저장 (Φ로 역추론하지 않음)');
  const c = e.chunks, sum = (a) => a.reduce((x, y) => x + y, 0);
  add('Σbasic', fmt(sum(c.basic))); add('ΣMIX', fmt(sum(c.mix))); add('Σα항', fmt(sum(c.alpha_term)));
  add('Σ(−βC)', fmt(sum(c.minus_beta_cost))); add('ΣΔ', fmt(sum(c.delta)));
  h.replaceChildren(el('h2', {}, `${e.instruction} · s${e.scene} · fold ${e.fold}`), kv);
}
function legend(keys) {
  const L = el('div', { class: 'legend' });
  for (const k of keys) {
    const s = SERIES[k], sp = el('span');
    const sw = sv('svg', { width: 22, height: 8 });
    sw.append(sv('line', { x1: 0, y1: 4, x2: 22, y2: 4, stroke: s.color, 'stroke-width': 2, 'stroke-dasharray': s.dash }));
    sp.append(sw, document.createTextNode(s.label)); L.append(sp);
  }
  return L;
}
function chart(title, sub, keys, opts) {
  const box = el('div', { class: 'chart' });
  box.append(el('h3', {}, title), el('p', { class: 'sub' }, sub));
  if (keys.length > 1) box.append(legend(keys));
  const c = state.ep.chunks, n = c.step.length;
  const envX = $('sel-x').value === 'env';
  const xs0 = envX ? c.env_step_start : c.step, xs1 = envX ? c.env_step_start.map((s, i) => s + c.m[i]) : c.step.map((s) => s + 1);
  const data = keys.map((k) => [k, series(k)]).filter(([, v]) => v);
  let lo = Infinity, hi = -Infinity;
  for (const [, v] of data) for (const y of v) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
  if (opts && opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const W = 1000, H = 190, ml = 56, mr = 150, mt = 8, mb = 26;
  const xmax = xs1[n - 1];
  const X = (x) => ml + (W - ml - mr) * x / xmax, Y = (y) => mt + (H - mt - mb) * (hi - y) / (hi - lo);
  const g = sv('svg', { class: 'plot', viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', role: 'img', 'aria-label': title });
  for (let t = 0; t <= 4; t++) {
    const y = lo + (hi - lo) * t / 4;
    g.append(sv('line', { class: 'grid', x1: ml, x2: W - mr, y1: Y(y), y2: Y(y) }));
    const tx = sv('text', { x: ml - 6, y: Y(y) + 4, 'text-anchor': 'end' }); tx.textContent = fmt(y); g.append(tx);
  }
  const step = envX ? (xmax > 400 ? 100 : 50) : (xmax > 80 ? 20 : 10);
  for (let x = 0; x <= xmax; x += step) {
    const tx = sv('text', { x: X(x), y: H - 8, 'text-anchor': 'middle' }); tx.textContent = String(x); g.append(tx);
  }
  const xl = sv('text', { x: W - mr, y: H - 8, 'text-anchor': 'start' }); xl.textContent = envX ? ' env step' : ' inference step'; g.append(xl);
  if (opts && opts.zero && lo < 0 && hi > 0) g.append(sv('line', { class: 'zero', x1: ml, x2: W - mr, y1: Y(0), y2: Y(0) }));
  const fi = c.final.indexOf(true);
  if (fi >= 0) {
    g.append(sv('line', { class: 'term', x1: X(xs0[fi]), x2: X(xs0[fi]), y1: mt, y2: H - mb }));
    const tt = sv('text', { x: X(xs0[fi]) - 4, y: mt + 10, 'text-anchor': 'end' });
    tt.textContent = `terminal (anchor ${state.ep.terminal_anchor > 0 ? '+1' : '−1'}, m=${c.m[fi]})`; g.append(tt);
  }
  const labels = [];
  for (const [k, v] of data) {
    const s = SERIES[k]; let d = '';
    for (let i = 0; i < n; i++) d += (i ? 'L' : 'M') + X(xs0[i]).toFixed(2) + ',' + Y(v[i]).toFixed(2) + 'L' + X(xs1[i]).toFixed(2) + ',' + Y(v[i]).toFixed(2);
    g.append(sv('path', { class: 'series', d, stroke: s.color, 'stroke-dasharray': s.dash, 'vector-effect': 'non-scaling-stroke' }));
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
  const tip = $('tip');
  hit.addEventListener('mousemove', (ev) => {
    const r = g.getBoundingClientRect();
    const x = (ev.clientX - r.left) / r.width * W;
    const es = (x - ml) / (W - ml - mr) * xmax;
    let i = xs1.findIndex((e1) => es < e1); if (i < 0) i = n - 1;
    cross.setAttribute('x1', X(xs0[i])); cross.setAttribute('x2', X(xs0[i])); cross.setAttribute('visibility', 'visible');
    tip.replaceChildren(el('div', {}, `inference step ${c.step[i]} · env step ${c.env_step_start[i]}–${c.env_step_start[i] + c.m[i] - 1} (m=${c.m[i]})${c.final[i] ? ' · terminal' : ''}`));
    for (const [k, v] of data) {
      const row = el('div'); const sw = el('span', { class: 'sw' }); sw.style.background = SERIES[k].color;
      row.append(sw, document.createTextNode(`${SERIES[k].label}: ${fmt(v[i])}`)); tip.append(row);
    }
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 300) + 'px'; tip.style.top = (ev.clientY + 12) + 'px';
  });
  hit.addEventListener('mouseleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); });
  box.append(g);
  return box;
}
function renderCharts() {
  const cum = $('chk-cum').checked ? '누적 합(할인 없음) · ' : 'chunk별 값 · ';
  const out = [
    chart('basic vs MIX', cum + '칸 = decision 1회(실행 env step 보통 5, 마지막은 m)', ['basic', 'mix'], {}),
    chart('MIX − basic 분해', cum + 'α항과 실제 차감 −βC를 분리; Δ = α항 + (−βC)', ['alpha_term', 'minus_beta_cost', 'delta'], { zero: true }),
  ];
  if (state.ep.phi) out.push(chart('Φ (저장된 fold potential 값)', 'decision latent의 Φ · cluster ID 미저장(역추론 안 함) · terminal 행 anchor는 Φ 대신 outcome ±1', ['phi'], { zero: true }));
  else out.push(el('p', { class: 'missing' }, 'Φ 배열 없음 — 표시하지 않음'));
  $('charts').replaceChildren(...out);
}
function renderTable() {
  const w = $('table-wrap');
  w.hidden = !$('chk-table').checked;
  if (w.hidden) return;
  const c = state.ep.chunks, keys = ['basic', 'mix', 'alpha_term', 'minus_beta_cost', 'delta'];
  const t = el('table'), hr = el('tr');
  for (const h of ['inference step', 'env step', 'm', ...keys.map((k) => SERIES[k].label), 'Φ', 'terminal']) hr.append(el('th', {}, h));
  t.append(hr);
  const cols = keys.map((k) => series(k));
  for (let i = 0; i < c.step.length; i++) {
    const tr = el('tr');
    tr.append(el('td', {}, String(c.step[i])), el('td', {}, `${c.env_step_start[i]}–${c.env_step_start[i] + c.m[i] - 1}`), el('td', {}, String(c.m[i])));
    for (const col of cols) tr.append(el('td', {}, fmt(col[i])));
    tr.append(el('td', {}, state.ep.phi ? fmt(state.ep.phi[i]) : '—'), el('td', {}, c.final[i] ? 'terminal' : ''));
    t.append(tr);
  }
  w.replaceChildren(el('h2', {}, '표 보기' + ($('chk-cum').checked ? ' (누적)' : '')), t);
}
function renderDups() {
  const d = $('dups'), e = state.ep;
  d.replaceChildren(el('h2', {}, `fold 간 중복 (${e.duplicates.length}개 fold fit에 포함)`),
    el('p', { class: 'sub missing' }, '같은 rollout: basic은 fold 간 동일, MIX/Φ는 fold별 potential이라 다를 수 있음'));
  const ul = el('div', { class: 'kv' });
  for (const x of e.duplicates) {
    const b = el('button', { class: 'link' }, x.fold + (x.fold === e.fold ? ' (현재)' : ''));
    if (x.fold === e.fold) b.disabled = true;
    b.addEventListener('click', () => goto(x.fold, x.ep));
    ul.append(b);
  }
  d.append(ul);
}
function goto(fold, ep) {
  for (const [t, ins] of Object.entries(state.meta.tasks)) for (const [i, fl] of Object.entries(ins)) {
    if (fl.some((f) => f.fold === fold)) {
      $('sel-task').value = t; fillInstrNoLoad(); $('sel-instr').value = i; fillFoldsNoLoad(); $('sel-fold').value = fold;
      loadFold(ep); return;
    }
  }
}
function fillInstrNoLoad() {
  const ins = state.meta.tasks[$('sel-task').value] || {};
  setOptions($('sel-instr'), Object.keys(ins).sort().map((i) => [i, i]), false);
}
function fillFoldsNoLoad() {
  const fl = (state.meta.tasks[$('sel-task').value] || {})[$('sel-instr').value] || [];
  setOptions($('sel-fold'), fl.map((f) => [f.fold, `${f.fold}  (target j${f.target_j} 제외 fit ${f.episodes}판, 성공 ${f.success})`]), false);
}
function renderVideo() {
  const v = $('video'), e = state.ep;
  v.replaceChildren(el('h2', {}, '원본 영상'));
  if (e.video && e.video_id !== null) {
    const vid = el('video', { controls: '', preload: 'metadata', width: '100%', src: '/video?id=' + e.video_id });
    v.append(vid, el('p', { class: 'sub missing' }, '검증된 경로·sha256 일치 파일만 제공'));
  } else {
    v.append(el('p', { class: 'missing' }, '누락 — 검증된 영상 경로 없음 (rollout pkl은 열지 않음)'));
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
  try {
    state.meta = await getJSON('/api/meta');
  } catch (err) { $('prov').textContent = '메타데이터 로드 실패: ' + err.message; return; }
  $('prov').textContent = state.meta.provenance;
  $('sel-task').addEventListener('change', fillInstr);
  $('sel-instr').addEventListener('change', fillFolds);
  $('sel-fold').addEventListener('change', () => loadFold());
  $('sel-outcome').addEventListener('change', () => fillEpisodes());
  $('sel-ep').addEventListener('change', loadEpisode);
  $('chk-cum').addEventListener('change', () => state.ep && renderAll());
  $('sel-x').addEventListener('change', () => state.ep && renderCharts());
  $('chk-table').addEventListener('change', () => state.ep && renderTable());
  fillTasks();
}
init();

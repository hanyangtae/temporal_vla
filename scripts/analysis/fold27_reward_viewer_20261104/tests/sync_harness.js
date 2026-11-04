// Browser fixture for the video<->chart sync (loaded only by tests/test_viewer.py via a test-only static entry).
// Drives the real UI in real time; every stage is reported via GET /harness-result (test-only route).
(async () => {
  const out = {}, $ = (id) => document.getElementById(id);
  const pre = document.createElement('pre'); pre.id = 'harness-out'; document.body.append(pre);
  const flush = (stage) => {   // report each stage to the test-only route (real time; no --dump-dom race)
    out.stage = stage; pre.textContent = JSON.stringify(out);
    fetch('/harness-result?d=' + encodeURIComponent(pre.textContent)).catch(() => {});
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const until = async (fn, ms = 4000) => { for (let t = 0; t < ms; t += 25) { if (fn()) return true; await sleep(25); } return false; };
  const once = (el, ev) => new Promise((r) => el.addEventListener(ev, r, { once: true }));
  const vid = () => document.querySelector('#video video');
  const curs = () => [...document.querySelectorAll('#charts line.vcur')].map((l) => [l.getAttribute('visibility'), +l.getAttribute('x1')]);
  const lab = () => (document.querySelector('#video .sync-pos') || {}).textContent || '';
  const seek = async (t) => { const v = vid(), p = once(v, 'seeked'); v.currentTime = t; await p; await sleep(50); };
  const clickAt = async (chartIdx, xUnits) => {   // click inside the plot rect at data-x = xUnits
    const svg = document.querySelectorAll('#charts svg.plot')[chartIdx], r = svg.getBoundingClientRect();
    const xmax = +svg.dataset.xmax, W = 1000, ml = 56, mr = 170;
    const px = r.left + (ml + (W - ml - mr) * xUnits / xmax) / W * r.width;
    const hit = svg.querySelector('rect[fill="transparent"]'), p = once(vid(), 'seeked');
    hit.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: px, clientY: r.top + r.height / 2 }));
    await p; await sleep(50);
  };
  try {
    flush('start');
    out.ready = await until(() => vid() && vid().readyState >= 1 && document.querySelectorAll('#charts line.vcur').length);
    out.rs = vid() ? vid().readyState : -1; out.err = vid() && vid().error ? vid().error.code : null; flush('ready');
    out.first_ep = vid().dataset.episode;
    flush('first_ep');
    out.verified_text = !!document.querySelector('#video .sync-ok');
    flush('verified_text');
    out.means_cursors = document.querySelectorAll('#means line.vcur').length;
    flush('means_cursors');
    out.n_cursors = curs().length;
    flush('n_cursors');
    out.duration = vid().duration;
    flush('duration');
    await seek(4.5 / 20);                       // frame 4 -> env step 8 -> chunk 1
    out.seek_frame4 = { curs: curs(), lab: lab() };
    flush('seek_frame4');
    await clickAt(0, 2.5);                      // pointer inside inference 2 -> frame ceil(10/2)=5
    out.click_inf2 = { t: vid().currentTime, curs: curs(), lab: lab() };
    flush('click_inf2');
    await clickAt(0, 3.99);                     // last inference (3): frame ceil(15/2)=8 = last frame
    out.click_last = { t: vid().currentTime, lab: lab() };
    flush('click_last');
    $('sel-x').value = 'env'; $('sel-x').dispatchEvent(new Event('change'));
    await sleep(50);
    out.env_axis = { curs: curs() };
    flush('env_axis');
    await clickAt(0, 7.0);                      // env axis: x=7 lies in chunk 1 (env 5..9) -> frame ceil(5/2)=3
    out.env_click = { t: vid().currentTime, curs: curs(), lab: lab() };
    flush('env_click');
    $('chk-cum').checked = true; $('chk-cum').dispatchEvent(new Event('change'));
    await sleep(50);
    out.cum = { curs: curs() };
    flush('cum');
    document.querySelector('#charts .tog').click();      // hide a decomposition series -> charts re-render
    await sleep(50);
    out.hidden_series = { curs: curs(), hash: location.hash };
    flush('hidden_series');
    vid().pause(); await sleep(50);
    out.paused = { curs: curs() };
    flush('paused');
    vid().muted = true; const endP = once(vid(), 'ended');
    vid().currentTime = vid().duration - 0.03; await vid().play().catch(() => {});
    out.ended_fired = await Promise.race([endP.then(() => true), sleep(3000).then(() => false)]);
    flush('ended_fired');
    await sleep(50);
    out.ended = { curs: curs(), lab: lab() };
    flush('ended');
    $('sel-ep').value = '1'; $('sel-ep').dispatchEvent(new Event('change'));     // rollout change -> unverified video
    await until(() => vid() && vid().dataset.episode !== out.first_ep && document.querySelector('#video .bad'));
    await sleep(100);
    out.changed = { curs: curs(), unverified_text: (document.querySelector('#video .bad') || {}).textContent || '',
                    lab: lab(), t0: vid().currentTime };
    const svg0 = document.querySelectorAll('#charts svg.plot')[0];
    svg0.querySelector('rect[fill="transparent"]').dispatchEvent(new MouseEvent('click', { bubbles: true,
      clientX: svg0.getBoundingClientRect().left + svg0.getBoundingClientRect().width / 2, clientY: svg0.getBoundingClientRect().top + 50 }));
    await sleep(200);
    out.unverified_click = { t: vid().currentTime, curs: curs() };
    flush('unverified_click');
  } catch (e) { out.error = String(e && e.stack || e); }
  out.done = true; flush('done');
})();

(function () {
  const $ = (id) => document.getElementById(id);
  const pad = $("pad");
  const ctx = pad.getContext("2d", { willReadFrequently: true });

  // ---------- 색상 ----------
  // 활성화 값(0~1) → 어두운 남색 → 청록 → 노랑 (밝을수록 강하게 반응)
  const STOPS = [[13, 16, 40], [40, 60, 140], [30, 150, 170], [120, 210, 110], [255, 230, 70]];
  function heat(t) {
    t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
    const i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i;
    const a = STOPS[i], b = STOPS[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  const rgb = (c) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;

  function drawMap(canvas, data, offset, size, max, mode) {
    const g = canvas.getContext("2d");
    const img = g.createImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const v = data[offset + i] / (max || 1);
      const c = mode === "gray" ? [v * 255, v * 255, v * 255] : heat(Math.pow(v, 0.75));
      img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  }
  function maxOf(arr) { let m = 0; for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i]; return m; }

  // ---------- 가운데 패널 준비 ----------
  function makeMaps(id, count, size, layer) {
    const box = $(id);
    const list = [];
    for (let i = 0; i < count; i++) {
      const c = document.createElement("canvas");
      c.width = c.height = size;
      c.title = `${layer} · ${i + 1}번째 특징 지도`;
      c.addEventListener("click", () => openMap(layer, i));
      box.appendChild(c);
      list.push(c);
    }
    return list;
  }
  const mapCanvases = {
    conv1: makeMaps("maps-conv1", 32, 26, "합성곱층 1"),
    pool1: makeMaps("maps-pool1", 32, 13, "풀링층 1"),
    conv2: makeMaps("maps-conv2", 64, 11, "합성곱층 2"),
    pool2: makeMaps("maps-pool2", 64, 5, "풀링층 2"),
  };
  const neurons = [];
  for (let i = 0; i < 128; i++) { const n = document.createElement("i"); $("neurons").appendChild(n); neurons.push(n); }
  const outCells = [], bars = [];
  for (let d = 0; d < 10; d++) {
    const cell = document.createElement("div");
    cell.className = "out-cell";
    cell.innerHTML = `<div class="ball">${d}</div><span>0%</span>`;
    $("outputs").appendChild(cell);
    outCells.push(cell);
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.innerHTML = `<b>${d}</b><div class="track"><div class="fill"></div></div><span>0.0%</span>`;
    $("bars").appendChild(bar);
    bars.push(bar);
  }
  $("acc").textContent = (CNN.testAccuracy * 100).toFixed(2) + "%";

  // ---------- 그리기 ----------
  let strokes = [];      // [{width, points:[{x,y,t}]}]
  let current = null;
  let penWidth = +$("pen").value;
  let replaying = false;

  function resetCanvas() {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, pad.width, pad.height);
  }
  function drawSegment(stroke, from, to) {
    ctx.strokeStyle = "#111";
    ctx.fillStyle = "#111";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = stroke.width;
    const pts = stroke.points;
    if (from === 0) {
      ctx.beginPath();
      ctx.arc(pts[0].x, pts[0].y, stroke.width / 2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.beginPath();
    ctx.moveTo(pts[Math.max(0, from - 1)].x, pts[Math.max(0, from - 1)].y);
    for (let i = from; i <= to; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.stroke();
  }
  function redrawAll() {
    resetCanvas();
    strokes.forEach((s) => drawSegment(s, 0, s.points.length - 1));
  }
  function pointFrom(e) {
    const r = pad.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (pad.width / r.width), y: (e.clientY - r.top) * (pad.height / r.height), t: performance.now() };
  }

  pad.addEventListener("pointerdown", (e) => {
    if (replaying) return;
    pad.setPointerCapture(e.pointerId);
    current = { width: penWidth, points: [pointFrom(e)] };
    strokes.push(current);
    drawSegment(current, 0, 0);
    $("placeholder").hidden = true;
    schedule();
  });
  pad.addEventListener("pointermove", (e) => {
    if (!current) return;
    const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    const start = current.points.length;
    events.forEach((ev) => current.points.push(pointFrom(ev)));
    drawSegment(current, start, current.points.length - 1);
    schedule();
  });
  function endStroke() {
    if (!current) return;
    current = null;
    const r = runNow();
    if (r) addTimeline(r);
  }
  pad.addEventListener("pointerup", endStroke);
  pad.addEventListener("pointercancel", endStroke);

  $("pen").addEventListener("input", (e) => { penWidth = +e.target.value; });
  $("btn-clear").addEventListener("click", clearAll);
  $("btn-undo").addEventListener("click", () => {
    if (replaying || !strokes.length) return;
    strokes.pop();
    timeline.pop();
    renderTimeline();
    redrawAll();
    if (!strokes.length) clearAll(); else runNow();
  });
  $("btn-replay").addEventListener("click", replay);

  function clearAll() {
    if (replaying) return;
    strokes = [];
    timeline = [];
    renderTimeline();
    resetCanvas();
    $("placeholder").hidden = false;
    showEmpty();
  }

  // ---------- 쓰는 과정 다시보기 ----------
  function replay() {
    if (replaying || !strokes.length) return;
    replaying = true;
    document.querySelectorAll(".tools button").forEach((b) => (b.disabled = true));
    const saved = strokes;
    // 획 사이의 쉬는 시간은 최대 0.4초로 줄여서 재생
    const events = [];
    let clock = 0, last = null;
    saved.forEach((s, si) => s.points.forEach((p, pi) => {
      const gap = last === null ? 0 : Math.min(p.t - last, pi === 0 ? 400 : 60);
      clock += gap; last = p.t;
      events.push({ si, pi, at: clock });
    }));
    strokes = [];
    timeline = [];
    renderTimeline();
    resetCanvas();
    showEmpty();
    const t0 = performance.now();
    let idx = 0;
    (function frame() {
      const now = performance.now() - t0;
      while (idx < events.length && events[idx].at <= now) {
        const ev = events[idx++];
        if (ev.pi === 0) strokes.push({ width: saved[ev.si].width, points: [] });
        const s = strokes[ev.si];
        s.points.push(saved[ev.si].points[ev.pi]);
        drawSegment(s, s.points.length - 1, s.points.length - 1);
        const next = events[idx];
        if (!next || next.si !== ev.si) { const r = runNow(); if (r) addTimeline(r); }
      }
      runNow();
      if (idx < events.length) requestAnimationFrame(frame);
      else {
        replaying = false;
        document.querySelectorAll(".tools button").forEach((b) => (b.disabled = false));
      }
    })();
  }

  // ---------- 인식 ----------
  let pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; runNow(); });
  }

  let lastResult = null;
  function runNow() {
    const prep = Preprocess.process(pad);
    if (!prep) { showEmpty(); return null; }
    const t0 = performance.now();
    const r = CNN.predict(prep.input);
    const ms = performance.now() - t0;
    const knn = KNN.classify(prep.input, 5);
    r.prep = prep; r.knn = knn;
    r.pred = r.probs.indexOf(Math.max(...r.probs));
    lastResult = r;
    render(r, ms);
    return r;
  }

  function showEmpty() {
    lastResult = null;
    $("pipeline").classList.add("empty");
    $("answer-digit").textContent = "?";
    $("answer-conf").textContent = "숫자를 써 주세요";
    bars.forEach((b) => { b.classList.remove("top"); b.querySelector(".fill").style.width = "0"; b.querySelector("span").textContent = "0.0%"; });
    outCells.forEach((c) => { c.classList.remove("top"); c.querySelector(".ball").style.background = ""; c.querySelector("span").textContent = "0%"; });
    $("knn-result").textContent = "-";
    $("neighbors").innerHTML = "";
  }

  let flashTimer = 0, lastFlash = 0;
  function flashPipeline() {
    const now = performance.now();
    if (now - lastFlash < 900) return;
    lastFlash = now;
    const stages = document.querySelectorAll(".stage");
    clearTimeout(flashTimer);
    let i = 0;
    (function step() {
      stages.forEach((s, j) => s.classList.toggle("flash", j === i));
      i++;
      if (i <= stages.length) flashTimer = setTimeout(step, 70);
    })();
  }

  function render(r, ms) {
    $("pipeline").classList.remove("empty");
    $("ms").textContent = ms.toFixed(1);
    const eng = $("engine");
    eng.classList.remove("live"); void eng.offsetWidth; eng.classList.add("live");
    flashPipeline();

    // ① 전처리
    const crop = $("prep-crop").getContext("2d");
    const sq = r.prep.square;
    crop.fillStyle = "#fff"; crop.fillRect(0, 0, 120, 120);
    crop.drawImage(pad, sq.x, sq.y, sq.w, sq.h, 0, 0, 120, 120);
    drawMap($("prep-small"), r.prep.small, 0, 20, 1, "gray");
    drawMap($("prep-input"), r.input, 0, 28, 1, "gray");
    drawPixelGrid($("pixel-grid"), r.input);
    if (!$("pixel-modal").hidden) drawPixelGrid($("pixel-big"), r.input);

    // ②~⑤ 특징 지도
    for (const [name, ch, size] of [["conv1", 32, 26], ["pool1", 32, 13], ["conv2", 64, 11], ["pool2", 64, 5]]) {
      const data = r[name], max = maxOf(data);
      mapCanvases[name].forEach((c, i) => drawMap(c, data, i * size * size, size, max));
    }
    // ⑥ 평탄화
    const flatCtx = $("flat").getContext("2d");
    const fmax = maxOf(r.flat);
    const img = flatCtx.createImageData(1600, 1);
    for (let i = 0; i < 1600; i++) {
      const c = heat(Math.pow(r.flat[i] / (fmax || 1), 0.75));
      img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
    }
    flatCtx.putImageData(img, 0, 0);
    // ⑦ 완전 연결층
    const nmax = maxOf(r.fc1);
    neurons.forEach((n, i) => { n.style.background = rgb(heat(Math.pow(r.fc1[i] / (nmax || 1), 0.75))); });
    // ⑧ 출력층 + 결과
    r.probs.forEach((p, d) => {
      const cell = outCells[d];
      cell.classList.toggle("top", d === r.pred);
      cell.querySelector(".ball").style.background = rgb(heat(p));
      cell.querySelector(".ball").style.color = p > 0.6 ? "#1d2433" : "#fff";
      cell.querySelector("span").textContent = Math.round(p * 100) + "%";
      const bar = bars[d];
      bar.classList.toggle("top", d === r.pred);
      bar.querySelector(".fill").style.width = (p * 100).toFixed(1) + "%";
      bar.querySelector("span").textContent = (p * 100).toFixed(1) + "%";
    });
    const conf = r.probs[r.pred];
    $("answer-digit").textContent = r.pred;
    $("answer-conf").textContent = `${(conf * 100).toFixed(1)}% 확신 · ` +
      (conf > 0.9 ? "확실해요!" : conf > 0.6 ? "아마 이 숫자일 거예요" : "헷갈려요… 🤔");

    // k-최근접 이웃
    const k = r.knn;
    $("knn-result").innerHTML = `예측: <b>${k.label}</b> <span class="muted small">(이웃 5개 중 ${k.votes[k.label]}표)</span>` +
      (k.label === r.pred ? ' <span class="small" style="color:var(--good)">딥러닝과 같아요</span>' : ' <span class="small" style="color:var(--bad)">딥러닝과 달라요!</span>');
    const nb = $("neighbors");
    if (nb.children.length !== 5) {
      nb.innerHTML = "";
      for (let i = 0; i < 5; i++) {
        const f = document.createElement("figure");
        f.innerHTML = `<canvas width="28" height="28"></canvas><b></b><div></div>`;
        nb.appendChild(f);
      }
    }
    k.neighbors.forEach((n, i) => {
      const f = nb.children[i];
      const g = f.querySelector("canvas").getContext("2d");
      const im = g.createImageData(28, 28);
      for (let j = 0; j < 784; j++) { im.data[j * 4] = im.data[j * 4 + 1] = im.data[j * 4 + 2] = n.image[j]; im.data[j * 4 + 3] = 255; }
      g.putImageData(im, 0, 0);
      f.querySelector("b").textContent = n.label;
      f.querySelector("div").textContent = "거리 " + n.dist.toFixed(1);
    });

    if (openedMap) drawModal();
  }

  function drawPixelGrid(canvas, input) {
    const g = canvas.getContext("2d");
    const cell = canvas.width / 28;
    g.font = `${Math.floor(cell * 0.42)}px ui-monospace, Menlo, monospace`;
    g.textAlign = "center"; g.textBaseline = "middle";
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const v = Math.round(input[y * 28 + x] * 255);
      g.fillStyle = `rgb(${v},${v},${v})`;
      g.fillRect(x * cell, y * cell, cell, cell);
      g.fillStyle = v > 128 ? "#000" : v > 0 ? "#fff" : "#3a3f4a";
      g.fillText(String(v), x * cell + cell / 2, y * cell + cell / 2);
    }
  }

  // ---------- 특징 지도 확대 ----------
  let openedMap = null;
  function openMap(layer, index) {
    if (!lastResult) return;
    openedMap = { layer, index };
    $("modal").hidden = false;
    drawModal();
  }
  function drawModal() {
    const r = lastResult;
    if (!r) return;
    const { layer, index } = openedMap;
    const info = {
      "합성곱층 1": { key: "conv1", size: 26, in: "input", inSize: 28, desc: "왼쪽 3×3 필터를 입력 이미지 위에서 한 칸씩 옮기며 곱해서 더한 결과예요. 필터 모양과 비슷한 부분이 있는 곳이 밝게 나타나요." },
      "풀링층 1": { key: "pool1", size: 13, in: "conv1", inSize: 26, desc: "합성곱층 1의 같은 번호 특징 지도에서 2×2 칸마다 가장 큰 값만 남긴 결과예요. 크기는 절반이지만 밝은 부분(특징)은 그대로 남아 있어요." },
      "합성곱층 2": { key: "conv2", size: 11, in: "pool1", inSize: 13, desc: "풀링층 1의 특징 지도 32장을 모두 조합해서 만든 새로운 특징이에요. 첫 번째 층보다 더 복잡한 모양에 반응해요." },
      "풀링층 2": { key: "pool2", size: 5, in: "conv2", inSize: 11, desc: "합성곱층 2의 같은 번호 특징 지도를 2×2 최댓값 풀링으로 줄인 결과예요." },
    }[layer];
    $("modal-title").textContent = `${layer} · ${index + 1}번째 특징 지도 (${info.size}×${info.size})`;
    $("modal-desc").textContent = info.desc;
    const filterFig = $("modal-filter").parentElement;
    const isFirst = info.key === "conv1";
    filterFig.style.display = isFirst ? "" : "none";
    filterFig.nextElementSibling.style.display = isFirst ? "" : "none";
    if (isFirst) {
      const f = CNN.filters1.subarray(index * 9, index * 9 + 9);
      const fm = Math.max(...Array.from(f, Math.abs));
      const g = $("modal-filter").getContext("2d");
      for (let i = 0; i < 9; i++) {
        const v = f[i] / fm;
        g.fillStyle = v >= 0 ? `rgb(255,${255 - v * 180},${255 - v * 200})` : `rgb(${255 + v * 200},${255 + v * 150},255)`;
        g.fillRect(i % 3, (i / 3) | 0, 1, 1);
      }
    }
    // 입력 쪽 그림: 첫 층은 입력 이미지, 풀링층은 같은 번호의 이전 지도, 합성곱층 2는 이전 층 전체 평균
    const inCanvas = $("modal-input");
    inCanvas.width = inCanvas.height = info.inSize;
    if (info.in === "input") drawMap(inCanvas, r.input, 0, 28, 1, "gray");
    else if (info.key === "conv2") {
      const avg = new Float32Array(169);
      for (let c = 0; c < 32; c++) for (let i = 0; i < 169; i++) avg[i] += r.pool1[c * 169 + i] / 32;
      drawMap(inCanvas, avg, 0, 13, maxOf(avg));
    } else drawMap(inCanvas, r[info.in], index * info.inSize * info.inSize, info.inSize, maxOf(r[info.in]));
    inCanvas.nextElementSibling.textContent = info.in === "input" ? "입력" : info.key === "conv2" ? "입력(풀링층 1의 32장 평균)" : "입력(이전 층 같은 번호)";
    const out = $("modal-map");
    out.width = out.height = info.size;
    drawMap(out, r[info.key], index * info.size * info.size, info.size, maxOf(r[info.key]));
  }
  function closeModals() { $("modal").hidden = true; $("pixel-modal").hidden = true; openedMap = null; }
  $("modal-close").addEventListener("click", closeModals);
  document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeModals));
  document.querySelectorAll(".modal").forEach((m) => m.addEventListener("click", (e) => { if (e.target === m) closeModals(); }));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModals(); });
  $("pixel-grid").addEventListener("click", () => {
    if (!lastResult) return;
    $("pixel-modal").hidden = false;
    drawPixelGrid($("pixel-big"), lastResult.input);
  });

  // ---------- 쓰는 동안 예측 변화 ----------
  let timeline = [];
  function addTimeline(r) {
    timeline.push({ d: r.pred, p: r.probs[r.pred] });
    renderTimeline();
  }
  function renderTimeline() {
    const box = $("stroke-timeline");
    if (!timeline.length) { box.innerHTML = '<span class="muted small">아직 기록이 없어요</span>'; return; }
    box.innerHTML = timeline.map((t, i) =>
      `${i ? '<span class="sep">→</span>' : ""}<span class="chip" title="${i + 1}번째 획 후">${t.d}<small>${Math.round(t.p * 100)}%</small></span>`).join("");
  }

  // ---------- 실험 기록 ----------
  const STORE = "hw-lab-log";
  let log = [];
  try { log = JSON.parse(localStorage.getItem(STORE) || "[]"); } catch (e) { log = []; }
  function saveLog() { try { localStorage.setItem(STORE, JSON.stringify(log)); } catch (e) { /* 저장 불가 환경 */ } }
  for (let d = 0; d < 10; d++) {
    const b = document.createElement("button");
    b.textContent = d;
    b.addEventListener("click", () => record(d));
    $("label-buttons").appendChild(b);
  }
  function record(truth) {
    if (!lastResult || replaying) return;
    const img = Array.from(lastResult.input, (v) => Math.round(v * 255));
    log.push({ truth, cnn: lastResult.pred, knn: lastResult.knn.label, img: btoa(String.fromCharCode(...img)) });
    saveLog();
    renderLog();
    clearAll();
    $("draw-panel").scrollTo({ top: 0, behavior: "smooth" });
  }
  function renderLog() {
    const n = log.length;
    const cnnOk = log.filter((l) => l.cnn === l.truth).length;
    const knnOk = log.filter((l) => l.knn === l.truth).length;
    $("score").innerHTML = n
      ? `<span>시도 <b>${n}</b>회</span><span>딥러닝 정확도 <b>${Math.round(cnnOk / n * 100)}%</b></span><span>k-최근접 이웃 정확도 <b>${Math.round(knnOk / n * 100)}%</b></span>`
      : '<span class="muted">아직 기록이 없어요</span>';
    const box = $("log");
    box.innerHTML = "";
    log.slice(-30).forEach((l) => {
      const item = document.createElement("div");
      item.className = "log-item" + (l.cnn === l.truth ? "" : " wrong");
      item.title = `쓴 숫자 ${l.truth} / 딥러닝 ${l.cnn} / k-NN ${l.knn}`;
      const c = document.createElement("canvas");
      c.width = c.height = 28;
      const g = c.getContext("2d"), im = g.createImageData(28, 28);
      const raw = atob(l.img);
      for (let j = 0; j < 784; j++) { const v = raw.charCodeAt(j); im.data[j * 4] = im.data[j * 4 + 1] = im.data[j * 4 + 2] = v; im.data[j * 4 + 3] = 255; }
      g.putImageData(im, 0, 0);
      item.appendChild(c);
      item.append(`${l.truth}→${l.cnn}`);
      box.appendChild(item);
    });
  }
  $("btn-reset-log").addEventListener("click", () => {
    if (!log.length || !confirm("실험 기록을 모두 지울까요?")) return;
    log = []; saveLog(); renderLog();
  });

  resetCanvas();
  renderLog();
  showEmpty();
})();

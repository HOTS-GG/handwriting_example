// 딥러닝 학습 과정 실험실: 작은 다층 퍼셉트론을 브라우저에서 직접 학습시키며 그 과정을 보여 준다.
(function () {
  const $ = (id) => document.getElementById(id);
  const DOMAIN = 6;           // 좌표 범위 [-6, 6]
  const GRID = 50;            // 히트맵 해상도
  const N_POINTS = 300;       // 데이터 점 개수 (절반 학습 / 절반 시험)
  const BATCH = 10;
  const ACT_NAME = { tanh: "Tanh", relu: "ReLU", sigmoid: "Sigmoid", linear: "Linear" };

  // ---------------- 입력 특성 ----------------
  const FEATURES = [
    { label: "X₁", f: (x, y) => x },
    { label: "X₂", f: (x, y) => y },
    { label: "X₁²", f: (x, y) => x * x },
    { label: "X₂²", f: (x, y) => y * y },
    { label: "X₁X₂", f: (x, y) => x * y },
    { label: "sin(X₁)", f: (x, y) => Math.sin(x) },
    { label: "sin(X₂)", f: (x, y) => Math.sin(y) },
  ];

  // ---------------- 상태 ----------------
  let dataset = "circle", noise = 0;
  let enabled = [true, true, false, false, false, false, false];
  let hidden = [4, 2];
  let actName = "tanh";
  let lr = 0.03;
  let train = [], test = [];
  let net = null;
  let epoch = 0, history = [];
  let playing = false;
  let step = null; // 데이터 하나 따라가기 상태

  // ---------------- 데이터 만들기 ----------------
  const rand = (a, b) => a + Math.random() * (b - a);
  function gaussRand() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function makeData(kind, noiseLevel, n = N_POINTS) {
    const pts = [];
    const nz = noiseLevel / 100;
    const jitter = () => rand(-1, 1) * nz * 5;
    if (kind === "circle") {
      for (let i = 0; i < n; i++) {
        const inside = i < n / 2;
        const r = inside ? rand(0, 2.5) : rand(3.5, 5);
        const t = rand(0, 2 * Math.PI);
        pts.push({ x: r * Math.sin(t) + jitter(), y: r * Math.cos(t) + jitter(), label: inside ? 1 : 0 });
      }
    } else if (kind === "xor") {
      for (let i = 0; i < n; i++) {
        let x = rand(-5, 5), y = rand(-5, 5);
        x += x > 0 ? 0.3 : -0.3; y += y > 0 ? 0.3 : -0.3;
        pts.push({ x: x + jitter(), y: y + jitter(), label: x * y >= 0 ? 1 : 0 });
      }
    } else if (kind === "gauss") {
      const sd = 0.7 + nz * 3;
      for (let i = 0; i < n; i++) {
        const blue = i < n / 2, c = blue ? 2 : -2;
        pts.push({ x: c + gaussRand() * sd, y: c + gaussRand() * sd, label: blue ? 1 : 0 });
      }
    } else {
      const half = n / 2;
      for (const [delta, label] of [[0, 1], [Math.PI, 0]]) {
        for (let i = 0; i < half; i++) {
          const r = (i / half) * 5, t = 1.75 * (i / half) * 2 * Math.PI + delta;
          pts.push({ x: r * Math.sin(t) + jitter(), y: r * Math.cos(t) + jitter(), label });
        }
      }
    }
    for (let i = pts.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [pts[i], pts[j]] = [pts[j], pts[i]]; }
    return pts;
  }
  function regenerate() {
    const pts = makeData(dataset, noise);
    train = pts.slice(0, N_POINTS / 2);
    test = pts.slice(N_POINTS / 2);
  }

  // ---------------- 신경망 ----------------
  const ACT = {
    tanh: { f: Math.tanh, d: (z, a) => 1 - a * a },
    relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
    sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), d: (z, a) => a * (1 - a) },
    linear: { f: (z) => z, d: () => 1 },
  };
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const activeFeatures = () => FEATURES.map((_, i) => i).filter((i) => enabled[i]);

  function buildNet() {
    const sizes = [activeFeatures().length, ...hidden, 1];
    const W = [], b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      W.push(Float64Array.from({ length: sizes[l + 1] * sizes[l] }, () => rand(-0.5, 0.5)));
      b.push(new Float64Array(sizes[l + 1]).fill(0.1));
    }
    net = { sizes, W, b };
    epoch = 0;
    history = [];
  }
  function inputsOf(p) { return activeFeatures().map((i) => FEATURES[i].f(p.x, p.y)); }

  // 순전파: 층마다 가중합(z)과 활성화 값(a)을 모두 남겨 둔다
  function forward(input) {
    const as = [Float64Array.from(input)], zs = [];
    const L = net.W.length, act = ACT[actName];
    for (let l = 0; l < L; l++) {
      const nIn = net.sizes[l], nOut = net.sizes[l + 1], W = net.W[l], prev = as[l];
      const z = new Float64Array(nOut), a = new Float64Array(nOut);
      for (let j = 0; j < nOut; j++) {
        let s = net.b[l][j];
        for (let i = 0; i < nIn; i++) s += W[j * nIn + i] * prev[i];
        z[j] = s;
        a[j] = l === L - 1 ? sigmoid(s) : act.f(s);
      }
      zs.push(z); as.push(a);
    }
    return { as, zs };
  }

  // 역전파: 출력층 오차에서 시작해 각 뉴런의 δ(오차 신호)를 거꾸로 계산
  function backward(cache, y) {
    const L = net.W.length, act = ACT[actName];
    const deltas = new Array(L);
    const out = cache.as[L][0];
    deltas[L - 1] = Float64Array.of((out - y) * out * (1 - out));
    for (let l = L - 2; l >= 0; l--) {
      const n = net.sizes[l + 1], nNext = net.sizes[l + 2], Wn = net.W[l + 1];
      const d = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < nNext; j++) s += Wn[j * n + i] * deltas[l + 1][j];
        d[i] = s * act.d(cache.zs[l][i], cache.as[l + 1][i]);
      }
      deltas[l] = d;
    }
    return deltas;
  }
  function accumulate(grad, cache, deltas) {
    for (let l = 0; l < net.W.length; l++) {
      const nIn = net.sizes[l];
      for (let j = 0; j < net.sizes[l + 1]; j++) {
        grad.b[l][j] += deltas[l][j];
        for (let i = 0; i < nIn; i++) grad.W[l][j * nIn + i] += deltas[l][j] * cache.as[l][i];
      }
    }
  }
  const zeroGrad = () => ({ W: net.W.map((w) => new Float64Array(w.length)), b: net.b.map((b) => new Float64Array(b.length)) });
  function applyGrad(grad, scale) {
    for (let l = 0; l < net.W.length; l++) {
      for (let k = 0; k < net.W[l].length; k++) net.W[l][k] -= scale * grad.W[l][k];
      for (let k = 0; k < net.b[l].length; k++) net.b[l][k] -= scale * grad.b[l][k];
    }
  }
  function trainEpoch() {
    const order = train.map((_, i) => i).sort(() => Math.random() - 0.5);
    for (let s = 0; s < order.length; s += BATCH) {
      const grad = zeroGrad();
      const batch = order.slice(s, s + BATCH);
      for (const idx of batch) {
        const p = train[idx];
        const cache = forward(inputsOf(p));
        accumulate(grad, cache, backward(cache, p.label));
      }
      applyGrad(grad, lr / batch.length);
    }
    epoch++;
  }
  function evaluate(pts) {
    let loss = 0, correct = 0;
    for (const p of pts) {
      const out = forward(inputsOf(p)).as[net.W.length][0];
      loss += 0.5 * (out - p.label) ** 2;
      if ((out >= 0.5 ? 1 : 0) === p.label) correct++;
    }
    return { loss: loss / pts.length, acc: correct / pts.length };
  }

  // ---------------- 색 ----------------
  const ORANGE = [245, 147, 34], BLUE = [8, 119, 189];
  function color(v) { // v: -1(주황) ~ 0(흰색) ~ 1(파랑)
    v = Math.max(-1, Math.min(1, v));
    const c = v < 0 ? ORANGE : BLUE, t = Math.abs(v);
    return [255 + (c[0] - 255) * t, 255 + (c[1] - 255) * t, 255 + (c[2] - 255) * t];
  }

  // ---------------- 신경망 그림 ----------------
  const netEl = $("net"), svg = $("links");
  let layout = null; // {nodes:[[{x,y,el,canvas}]], links:[[path]]}
  const NODE_GAP = 54, TOP = 64;

  function buildDiagram() {
    netEl.querySelectorAll(".node, .col-title, .col-ctl").forEach((e) => e.remove());
    svg.innerHTML = "";
    const cols = [FEATURES.length, ...hidden, 1];
    const width = Math.max(520, netEl.parentElement.clientWidth);
    const maxN = Math.max(...cols);
    netEl.style.height = TOP + maxN * NODE_GAP + 10 + "px";
    const left = 90, right = width - 50;
    const xs = cols.map((_, c) => left + (right - left) * (cols.length === 1 ? 0 : c / (cols.length - 1)));
    const nodes = [];
    const titles = ["입력층<br><small>특성(FEATURES)</small>", ...hidden.map((n, i) => `은닉층 ${i + 1}`), "출력층"];
    cols.forEach((n, c) => {
      const t = document.createElement("div");
      t.className = "col-title";
      t.style.left = xs[c] + "px";
      t.innerHTML = titles[c];
      netEl.appendChild(t);
      if (c > 0 && c < cols.length - 1) {
        const ctl = document.createElement("div");
        ctl.className = "col-ctl";
        ctl.style.left = xs[c] + "px";
        ctl.style.top = "22px";
        ctl.innerHTML = `<button class="mini" data-d="-1">−</button><span>${n}개</span><button class="mini" data-d="1">+</button>`;
        ctl.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => changeNeurons(c - 1, +b.dataset.d)));
        netEl.appendChild(ctl);
      }
      const col = [];
      for (let k = 0; k < n; k++) {
        const el = document.createElement("div");
        el.className = "node" + (c === 0 ? " input" : "") + (c === cols.length - 1 ? " output" : "");
        const y = TOP + k * NODE_GAP + (c === cols.length - 1 ? NODE_GAP : 0);
        el.style.left = xs[c] + "px";
        el.style.top = y + 18 + "px";
        const cv = document.createElement("canvas");
        cv.width = cv.height = GRID;
        el.appendChild(cv);
        if (c === 0) {
          el.innerHTML += `<span class="lbl">${FEATURES[k].label}</span>`;
          el.classList.toggle("off", !enabled[k]);
          el.title = "눌러서 이 입력 특성을 켜거나 끄기";
          el.addEventListener("click", () => toggleFeature(k));
        } else if (c === cols.length - 1) {
          el.title = "출력: 파랑(1)일 확률";
        } else {
          el.title = `은닉층 ${c}의 ${k + 1}번째 뉴런`;
        }
        netEl.appendChild(el);
        col.push({ x: xs[c], y: y + 18, el, canvas: el.querySelector("canvas") });
      }
      nodes.push(col);
    });
    // 연결선: 꺼진 입력 특성은 빼고 그린다
    const act = activeFeatures();
    const links = [];
    for (let l = 0; l < net.W.length; l++) {
      const fromNodes = l === 0 ? act.map((i) => nodes[0][i]) : nodes[l];
      const toNodes = nodes[l + 1];
      const arr = [];
      toNodes.forEach((to, j) => fromNodes.forEach((from, i) => {
        const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const x1 = from.x + 19, x2 = to.x - (l === net.W.length - 1 ? 22 : 19), mx = (x1 + x2) / 2;
        p.setAttribute("d", `M${x1},${from.y} C${mx},${from.y} ${mx},${to.y} ${x2},${to.y}`);
        p.setAttribute("class", "link");
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        p.appendChild(title);
        svg.appendChild(p);
        arr.push({ path: p, title, l, j, i, from, to });
      }));
      links.push(arr);
    }
    layout = { nodes, links, featureNodes: act.map((i) => nodes[0][i]) };
    updateLinks();
  }
  function updateLinks() {
    if (!layout) return;
    layout.links.forEach((arr, l) => arr.forEach((lk) => {
      const w = net.W[l][lk.j * net.sizes[l] + lk.i];
      lk.path.setAttribute("stroke", w >= 0 ? "#0877bd" : "#f59322");
      lk.path.setAttribute("stroke-width", (0.6 + Math.min(Math.abs(w), 4) * 1.5).toFixed(2));
      lk.path.setAttribute("stroke-opacity", (0.3 + 0.6 * Math.min(Math.abs(w) / 2, 1)).toFixed(2));
      lk.title.textContent = `가중치 w = ${w.toFixed(3)}`;
    }));
  }

  // ---------------- 히트맵 계산과 그리기 ----------------
  const outCanvas = $("output"), outCtx = outCanvas.getContext("2d");
  const off = document.createElement("canvas"); off.width = off.height = GRID;
  function render() {
    const L = net.W.length;
    const featIdx = activeFeatures();
    // 모든 노드의 값 기록: [열][노드] → Float32Array(GRID*GRID)
    const vals = [FEATURES.map(() => new Float32Array(GRID * GRID)), ...net.sizes.slice(1).map((n) => Array.from({ length: n }, () => new Float32Array(GRID * GRID)))];
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const x = -DOMAIN + (gx + 0.5) / GRID * 2 * DOMAIN, y = DOMAIN - (gy + 0.5) / GRID * 2 * DOMAIN;
        const k = gy * GRID + gx;
        FEATURES.forEach((f, i) => { vals[0][i][k] = f.f(x, y); });
        const c = forward(featIdx.map((i) => vals[0][i][k]));
        for (let l = 1; l <= L; l++) c.as[l].forEach((v, j) => { vals[l][j][k] = v; });
      }
    }
    const paint = (canvas, arr, mode) => {
      let maxAbs = 0;
      for (let k = 0; k < arr.length; k++) maxAbs = Math.max(maxAbs, Math.abs(arr[k]));
      const g = canvas.getContext("2d"), img = g.createImageData(GRID, GRID);
      for (let k = 0; k < arr.length; k++) {
        let v = arr[k];
        if (mode === "prob") v = 2 * v - 1;
        else if (mode === "sig") v = 2 * v - 1;
        else if (maxAbs > 1) v /= maxAbs;
        const c = color(v);
        img.data[k * 4] = c[0]; img.data[k * 4 + 1] = c[1]; img.data[k * 4 + 2] = c[2]; img.data[k * 4 + 3] = 255;
      }
      g.putImageData(img, 0, 0);
    };
    layout.nodes[0].forEach((n, i) => paint(n.canvas, vals[0][i]));
    for (let l = 1; l <= L; l++) {
      layout.nodes[l].forEach((n, j) => paint(n.canvas, vals[l][j], l === L ? "prob" : actName === "sigmoid" ? "sig" : ""));
    }
    // 출력 결정 경계
    const outVals = vals[L][0];
    const discrete = $("discretize").checked;
    const img = off.getContext("2d").createImageData(GRID, GRID);
    for (let k = 0; k < outVals.length; k++) {
      let v = 2 * outVals[k] - 1;
      if (discrete) v = v >= 0 ? 0.75 : -0.75;
      const c = color(v * 0.85);
      img.data[k * 4] = c[0]; img.data[k * 4 + 1] = c[1]; img.data[k * 4 + 2] = c[2]; img.data[k * 4 + 3] = 255;
    }
    off.getContext("2d").putImageData(img, 0, 0);
    const W = outCanvas.width;
    outCtx.imageSmoothingEnabled = !discrete;
    outCtx.drawImage(off, 0, 0, W, W);
    // 축
    outCtx.strokeStyle = "rgba(0,0,0,.12)"; outCtx.lineWidth = 1;
    outCtx.beginPath(); outCtx.moveTo(W / 2, 0); outCtx.lineTo(W / 2, W); outCtx.moveTo(0, W / 2); outCtx.lineTo(W, W / 2); outCtx.stroke();
    // 점
    const pts = $("show-test").checked ? [...train.map((p) => [p, false]), ...test.map((p) => [p, true])] : train.map((p) => [p, false]);
    for (const [p, isTest] of pts) {
      const [px, py] = toPx(p);
      outCtx.beginPath(); outCtx.arc(px, py, 5, 0, Math.PI * 2);
      outCtx.fillStyle = p.label ? "#0877bd" : "#f59322";
      outCtx.fill();
      outCtx.lineWidth = isTest ? 2 : 1;
      outCtx.strokeStyle = isTest ? "#1d2433" : "#fff";
      outCtx.stroke();
    }
    if (step) {
      const [px, py] = toPx(step.point);
      outCtx.beginPath(); outCtx.arc(px, py, 12, 0, Math.PI * 2);
      outCtx.lineWidth = 3; outCtx.strokeStyle = "#e5484d"; outCtx.stroke();
    }
    updateLinks();
  }
  const toPx = (p) => [(p.x + DOMAIN) / (2 * DOMAIN) * outCanvas.width, (DOMAIN - p.y) / (2 * DOMAIN) * outCanvas.height];

  // ---------------- 오차 그래프 ----------------
  const chart = $("loss-chart"), cctx = chart.getContext("2d");
  function updateStats() {
    const tr = evaluate(train), te = evaluate(test);
    history.push({ train: tr.loss, test: te.loss });
    if (history.length > 600) history = history.filter((_, i) => i % 2 === 0);
    $("train-loss").textContent = tr.loss.toFixed(3);
    $("test-loss").textContent = te.loss.toFixed(3);
    $("test-acc").textContent = Math.round(te.acc * 100) + "%";
    $("epoch").textContent = String(epoch).padStart(6, "0").replace(/(\d{3})(\d{3})$/, "$1,$2");
    const W = chart.width, H = chart.height;
    cctx.clearRect(0, 0, W, H);
    const max = Math.max(0.05, ...history.map((h) => Math.max(h.train, h.test)));
    for (const [key, col, w] of [["train", "#8f99ab", 2], ["test", "#1d2433", 2.5]]) {
      cctx.beginPath();
      history.forEach((h, i) => {
        const x = history.length === 1 ? 0 : (i / (history.length - 1)) * (W - 8) + 4;
        const y = H - 6 - (h[key] / max) * (H - 14);
        i ? cctx.lineTo(x, y) : cctx.moveTo(x, y);
      });
      cctx.strokeStyle = col; cctx.lineWidth = w; cctx.stroke();
    }
  }

  // ---------------- 학습 루프 ----------------
  let lastSlow = 0;
  function loop() {
    if (!playing) return;
    const speed = $("speed").value;
    if (speed === "slow") {
      // 느리게: 초당 약 8 에포크
      if (performance.now() - lastSlow < 120) return requestAnimationFrame(loop);
      lastSlow = performance.now();
      trainEpoch();
    } else if (speed === "fast") {
      // 빠르게: 한 화면을 그리는 동안 약 12ms 만큼 여러 에포크를 학습
      const t0 = performance.now();
      do trainEpoch(); while (performance.now() - t0 < 12);
    } else trainEpoch();
    render();
    updateStats();
    requestAnimationFrame(loop);
  }
  function setPlaying(on) {
    playing = on;
    $("btn-play").textContent = on ? "⏸" : "▶";
    $("btn-play").classList.toggle("on", on);
    if (on) { exitStep(); requestAnimationFrame(loop); }
  }
  function resetAll() {
    setPlaying(false);
    exitStep();
    buildNet();
    buildDiagram();
    render();
    updateStats();
  }

  // ---------------- 구조 바꾸기 ----------------
  function changeLayers(d) {
    const next = Math.max(0, Math.min(5, hidden.length + d));
    if (next === hidden.length) return;
    hidden = d > 0 ? [...hidden, 2] : hidden.slice(0, -1);
    $("layer-count").textContent = hidden.length;
    resetAll();
  }
  function changeNeurons(layer, d) {
    const n = Math.max(1, Math.min(8, hidden[layer] + d));
    if (n === hidden[layer]) return;
    hidden[layer] = n;
    resetAll();
  }
  function toggleFeature(i) {
    if (enabled[i] && activeFeatures().length === 1) return; // 최소 1개
    enabled[i] = !enabled[i];
    resetAll();
  }

  // ---------------- 데이터 하나 따라가기 ----------------
  const fmt = (v) => (v === 0 ? "0" : Math.abs(v) >= 0.01 ? v.toFixed(3) : Math.abs(v) >= 1e-5 ? v.toPrecision(2) : v.toExponential(1));
  const nodeName = (l, j) => (l === 0 ? FEATURES[activeFeatures()[j]].label : l === net.sizes.length - 1 ? "출력" : `은닉${l}-${j + 1}`);
  function nodeAt(l, j) { return l === 0 ? layout.featureNodes[j] : layout.nodes[l][j]; }
  function clearBadges() {
    netEl.querySelectorAll(".badge").forEach((b) => b.remove());
    netEl.querySelectorAll(".glow-f, .glow-b").forEach((n) => n.classList.remove("glow-f", "glow-b"));
  }
  function badge(l, j, text, kind) {
    const el = nodeAt(l, j).el;
    el.querySelector(".badge")?.remove();
    const b = document.createElement("span");
    b.className = "badge " + kind;
    b.textContent = text;
    el.appendChild(b);
  }
  function pulse(l, dir) {
    layout.links[l].forEach((lk) => {
      const p = lk.path.cloneNode(false);
      p.setAttribute("class", "pulse " + dir);
      p.removeAttribute("stroke"); p.removeAttribute("stroke-width"); p.removeAttribute("stroke-opacity");
      svg.appendChild(p);
      setTimeout(() => p.remove(), 900);
    });
  }
  let timers = [];
  const later = (ms, fn) => timers.push(setTimeout(fn, ms));
  function stopTimers() { timers.forEach(clearTimeout); timers = []; }

  function setPhase(p) {
    document.querySelectorAll("#phases li").forEach((li) => {
      const n = +li.dataset.p;
      li.className = n < p ? "done" : n === p ? "now" : "";
    });
  }
  function startStep(point) {
    setPlaying(false);
    stopTimers();
    clearBadges();
    const L = net.W.length;
    step = { point, phase: 0 };
    setPhase(0);
    const inp = inputsOf(point);
    inp.forEach((v, i) => badge(0, i, fmt(v), "a"));
    $("explain").innerHTML =
      `고른 점: <span class="f">X₁ = ${point.x.toFixed(2)}, X₂ = ${point.y.toFixed(2)}</span>, 정답은 ` +
      (point.label ? '<b style="color:var(--blue)">파랑(1)</b>' : '<b style="color:var(--orange)">주황(0)</b>') +
      `이에요.<br>입력층에는 이 점의 특성값 ${inp.length}개가 들어가요. 이제 신경망이 이 점을 어떻게 판단하는지 순서대로 따라가 봐요.`;
    $("btn-next").disabled = false;
    $("btn-next").textContent = "① 순전파 ▶";
    $("btn-exit").hidden = false;
    render();
  }
  function exitStep() {
    if (!step) return;
    stopTimers();
    step = null;
    clearBadges();
    setPhase(0);
    $("btn-next").disabled = true;
    $("btn-next").textContent = "다음 단계 ▶";
    $("btn-exit").hidden = true;
    $("explain").innerHTML = "학습은 이 네 단계를 아주 많이 반복하는 거예요. <b>점 하나를 골라</b> 신경망이 실제로 어떤 계산을 하는지 숫자로 확인해 봐요.";
    render();
  }
  function nextPhase() {
    if (!step) return;
    stopTimers();
    const L = net.W.length, p = step.point, y = p.label;
    const phase = step.phase === 4 ? 1 : step.phase + 1;
    step.phase = phase;
    setPhase(phase);
    const btn = $("btn-next");
    btn.disabled = true;
    const enable = (ms, text) => later(ms, () => { btn.disabled = false; btn.textContent = text; });

    if (phase === 1) {
      clearBadges();
      const inp = inputsOf(p);
      inp.forEach((v, i) => badge(0, i, fmt(v), "a"));
      const c = (step.cache = forward(inp));
      for (let l = 0; l < L; l++) {
        later(l * 700, () => pulse(l, "fwd"));
        later(l * 700 + 650, () => c.as[l + 1].forEach((v, j) => { badge(l + 1, j, fmt(v), "a"); nodeAt(l + 1, j).el.classList.add("glow-f"); }));
        later(l * 700 + 1300, () => layout.nodes[l + 1].forEach((n) => n.el.classList.remove("glow-f")));
      }
      // 첫 번째 계산 예시 (은닉층이 있으면 은닉1-1, 없으면 출력)
      const nIn = net.sizes[0], W = net.W[0];
      const terms = inp.map((x, i) => `(${fmt(W[i])}×${fmt(x)})`).join(" + ");
      const z = c.zs[0][0], a = c.as[1][0];
      const out = c.as[L][0];
      const isOut = L === 1;
      $("explain").innerHTML =
        `<b>순전파</b>: 입력값에 가중치를 곱해 모두 더하고(가중합), 활성화 함수를 통과시켜 다음 층으로 보내요.<br>` +
        `예) <b>${nodeName(1, 0)}</b> 뉴런: 가중합 <span class="f">${terms} + 편향(${fmt(net.b[0][0])}) = ${fmt(z)}</span> → ` +
        `${isOut ? "Sigmoid" : ACT_NAME[actName]} 함수 → <span class="f">${fmt(a)}</span><br>` +
        `모든 뉴런을 이렇게 계산하면 마지막 출력은 <span class="f">ŷ = ${fmt(out)}</span>, 즉 <b>파랑일 확률 ${Math.round(out * 100)}%</b>라고 예측했어요.`;
      enable(L * 700 + 400, "② 오차 계산 ▶");
    } else if (phase === 2) {
      const out = step.cache.as[L][0];
      const loss = 0.5 * (out - y) ** 2;
      step.lossBefore = loss;
      const right = (out >= 0.5 ? 1 : 0) === y;
      nodeAt(L, 0).el.classList.add("glow-b");
      badge(L, 0, `오차 ${fmt(loss)}`, "d");
      $("explain").innerHTML =
        `<b>오차(손실) 계산</b>: 예측 <span class="f">ŷ = ${fmt(out)}</span>, 정답 <span class="f">y = ${y}</span><br>` +
        `오차 <span class="f">= ½ × (ŷ − y)² = ½ × (${fmt(out)} − ${y})² = ${fmt(loss)}</span><br>` +
        (right ? `<span class="good">분류는 맞혔어요.</span> 하지만 정답(${y})과 완전히 같지는 않으니 오차를 더 줄일 수 있어요.`
               : `<span class="bad">분류를 틀렸어요!</span> 오차가 크니 가중치를 많이 고쳐야 해요.`);
      enable(500, "③ 역전파 ▶");
    } else if (phase === 3) {
      clearBadges();
      const deltas = (step.deltas = backward(step.cache, y));
      for (let l = L - 1; l >= 0; l--) {
        const t = (L - 1 - l) * 700;
        later(t, () => deltas[l].forEach((d, j) => { badge(l + 1, j, "δ " + fmt(d), "d"); nodeAt(l + 1, j).el.classList.add("glow-b"); }));
        later(t + 50, () => pulse(l, "bwd"));
        later(t + 1200, () => layout.nodes[l + 1].forEach((n) => n.el.classList.remove("glow-b")));
      }
      const out = step.cache.as[L][0];
      const gEx = deltas[L - 1][0] * step.cache.as[L - 1][0];
      $("explain").innerHTML =
        `<b>역전파</b>: 출력층의 오차를 입력층 방향으로 <b>거꾸로</b> 전달하며, 각 뉴런이 오차에 얼마나 책임이 있는지(δ)를 계산해요.<br>` +
        `출력층: <span class="f">δ = (ŷ − y) × ŷ(1 − ŷ) = (${fmt(out)} − ${y}) × ${fmt(out * (1 - out))} = ${fmt(deltas[L - 1][0])}</span><br>` +
        `각 가중치의 <b>기울기</b> = 뒤 뉴런의 δ × 앞 뉴런의 출력. 예) <b>${nodeName(L - 1, 0)} → 출력</b> 연결: ` +
        `<span class="f">${fmt(deltas[L - 1][0])} × ${fmt(step.cache.as[L - 1][0])} = ${fmt(gEx)}</span>`;
      enable(L * 700 + 400, "④ 가중치 수정 ▶");
    } else if (phase === 4) {
      clearBadges();
      const grad = zeroGrad();
      accumulate(grad, step.cache, step.deltas);
      const before = net.W.map((w) => Float64Array.from(w));
      applyGrad(grad, lr);
      // 많이 바뀐 가중치 5개
      const changes = [];
      net.W.forEach((w, l) => w.forEach((v, k) => {
        const nIn = net.sizes[l], j = Math.floor(k / nIn), i = k % nIn;
        changes.push({ l, i, j, old: before[l][k], g: grad.W[l][k], now: v });
      }));
      changes.sort((a, b) => Math.abs(b.now - b.old) - Math.abs(a.now - a.old));
      const after = forward(inputsOf(p)).as[L][0];
      const lossAfter = 0.5 * (after - y) ** 2;
      render();
      const rows = changes.slice(0, 5).map((c) =>
        `<tr><td>${nodeName(c.l, c.i)} → ${nodeName(c.l + 1, c.j)}</td><td>${c.old.toFixed(5)}</td><td>${fmt(c.g)}</td><td>${(c.now - c.old >= 0 ? "+" : "") + fmt(c.now - c.old)}</td><td><b>${c.now.toFixed(5)}</b></td></tr>`).join("");
      const better = lossAfter < step.lossBefore;
      $("explain").innerHTML =
        `<b>가중치 수정</b>: <span class="f">새 가중치 = 가중치 − 학습률(${lr}) × 기울기</span> 로 모든 가중치를 조금씩 고쳐요. (경사 하강법)` +
        `<table><tr><th>연결 (가장 많이 바뀐 5개)</th><th>이전</th><th>기울기</th><th>변화량</th><th>새 값</th></tr>${rows}</table>` +
        `같은 점을 다시 계산하면 오차: <span class="f">${step.lossBefore.toPrecision(4)} → ${lossAfter.toPrecision(4)}</span> ` +
        (better ? `<span class="good">줄었어요! ↓</span>` : `<span class="bad">(학습률이 너무 크면 오히려 커질 수도 있어요)</span>`) +
        `<br><span class="muted small">이 네 단계를 학습 데이터 ${train.length}개 모두에 대해 반복하면 <b>1 에포크</b>예요. 학습률을 크게 바꾸면 변화가 더 잘 보여요.</span>`;
      enable(300, "같은 점으로 한 번 더 ▶");
    }
  }

  // ---------------- 화면 연결 ----------------
  const DATASETS = [["circle", "원형"], ["xor", "XOR"], ["gauss", "가우시안"], ["spiral", "나선"]];
  DATASETS.forEach(([key, name]) => {
    const b = document.createElement("button");
    b.dataset.key = key;
    b.innerHTML = `<canvas width="60" height="60"></canvas>${name}`;
    const g = b.querySelector("canvas").getContext("2d");
    g.fillStyle = "#fff"; g.fillRect(0, 0, 60, 60);
    makeData(key, 0, 160).forEach((p) => {
      g.fillStyle = p.label ? "#0877bd" : "#f59322";
      g.beginPath(); g.arc((p.x + 6) * 5, (6 - p.y) * 5, 1.6, 0, Math.PI * 2); g.fill();
    });
    b.addEventListener("click", () => selectDataset(key));
    $("datasets").appendChild(b);
  });
  function selectDataset(key) {
    dataset = key;
    document.querySelectorAll(".datasets button").forEach((b) => b.classList.toggle("on", b.dataset.key === key));
    regenerate();
    resetAll();
  }

  const MISSIONS = [
    { t: "실습 1-② 단층 퍼셉트론 · 단순한 분류", d: "은닉층 0개 + 가우시안 데이터. 금방 Test loss가 0에 가까워질까?", data: "gauss", hidden: [], feats: [0, 1], act: "tanh", lr: 0.03 },
    { t: "실습 1-③ 단층 퍼셉트론 · 복잡한 분류", d: "은닉층 0개 + 나선 데이터. 절반이 틀리는 상황을 확인해요.", data: "spiral", hidden: [], feats: [0, 1], act: "tanh", lr: 0.03 },
    { t: "실습 2-② 다층 퍼셉트론 · 단순한 분류", d: "은닉층 2개(4개, 2개 뉴런) + 가우시안 데이터.", data: "gauss", hidden: [4, 2], feats: [0, 1], act: "tanh", lr: 0.03 },
    { t: "실습 2-③ 다층 퍼셉트론 · 복잡한 분류", d: "ReLU + 모든 특성 + 은닉층·뉴런 늘리기. 나선도 풀 수 있을까? (빠르게 학습)", data: "spiral", hidden: [8, 8, 6], feats: [0, 1, 2, 3, 4, 5, 6], act: "relu", lr: 0.1, speed: "fast" },
    { t: "도전 · XOR 문제", d: "단층 퍼셉트론으로는 못 풀어요. 은닉층을 넣거나 X₁X₂ 특성을 켜 보세요.", data: "xor", hidden: [], feats: [0, 1], act: "tanh", lr: 0.03 },
  ];
  MISSIONS.forEach((m) => {
    const b = document.createElement("button");
    b.innerHTML = `<b>${m.t}</b><span>${m.d}</span>`;
    b.addEventListener("click", () => {
      hidden = [...m.hidden];
      enabled = FEATURES.map((_, i) => m.feats.includes(i));
      actName = m.act; $("act").value = m.act;
      lr = m.lr; $("lr").value = String(m.lr);
      $("speed").value = m.speed || "normal";
      $("layer-count").textContent = hidden.length;
      selectDataset(m.data);
      setPlaying(true);
    });
    $("missions").appendChild(b);
  });

  $("btn-play").addEventListener("click", () => setPlaying(!playing));
  $("btn-reset").addEventListener("click", resetAll);
  $("btn-step-epoch").addEventListener("click", () => { setPlaying(false); exitStep(); trainEpoch(); render(); updateStats(); });
  $("lr").addEventListener("change", (e) => { lr = +e.target.value; });
  $("act").addEventListener("change", (e) => { actName = e.target.value; resetAll(); });
  $("layer-plus").addEventListener("click", () => changeLayers(1));
  $("layer-minus").addEventListener("click", () => changeLayers(-1));
  $("noise").addEventListener("input", (e) => { noise = +e.target.value; $("noise-val").textContent = noise; });
  $("noise").addEventListener("change", () => { regenerate(); resetAll(); });
  $("btn-regen").addEventListener("click", () => { regenerate(); resetAll(); });
  $("show-test").addEventListener("change", render);
  $("discretize").addEventListener("change", render);
  $("btn-pick").addEventListener("click", () => startStep(train[(Math.random() * train.length) | 0]));
  $("btn-next").addEventListener("click", nextPhase);
  $("btn-exit").addEventListener("click", exitStep);
  outCanvas.addEventListener("click", (e) => {
    const r = outCanvas.getBoundingClientRect();
    const mx = (e.clientX - r.left) * (outCanvas.width / r.width), my = (e.clientY - r.top) * (outCanvas.height / r.height);
    let best = null, bd = 18 * 18;
    for (const p of train) {
      const [px, py] = toPx(p);
      const d = (px - mx) ** 2 + (py - my) ** 2;
      if (d < bd) { bd = d; best = p; }
    }
    if (best) startStep(best);
  });
  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { const s = step; buildDiagram(); render(); if (s) startStep(s.point); }, 150);
  });

  // 시작
  selectDataset("circle");
})();

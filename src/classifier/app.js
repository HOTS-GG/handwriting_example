// 이미지 분류 실험실: 미리 학습된 MobileNet을 브라우저 안에서 실행하고 층별 계산 결과를 보여 준다.
(async function () {
  const $ = (id) => document.getElementById(id);
  const LABELS = window.IMAGENET_LABELS;

  // 보여 줄 중간 층들
  const STAGES = [
    { layer: "conv1_relu", title: "② 합성곱층 1", sub: "Conv2D · 3×3 필터 32개", desc: "가장 단순한 특징을 찾아요: 밝고 어두운 경계선, 색의 변화 같은 것들이에요." },
    { layer: "conv_pw_1_relu", title: "③ 합성곱 블록 1", sub: "깊이별 분리 합성곱", desc: "선과 색을 조합해 조금 더 다양한 무늬를 찾아요." },
    { layer: "conv_pw_3_relu", title: "④ 합성곱 블록 3", sub: "깊이별 분리 합성곱", desc: "크기를 절반으로 줄이며(풀링과 비슷한 역할) 질감과 반복되는 무늬를 찾아요." },
    { layer: "conv_pw_5_relu", title: "⑤ 합성곱 블록 5", sub: "깊이별 분리 합성곱", desc: "눈, 귀, 바퀴, 잎처럼 물체의 '부분'에 반응하는 특징이 나타나기 시작해요." },
    { layer: "conv_pw_11_relu", title: "⑥ 합성곱 블록 11", sub: "깊이별 분리 합성곱", desc: "부분들을 조합한 더 복잡한 모양에 반응해요. 사람이 보기엔 점점 알아보기 어려워져요." },
    { layer: "conv_pw_13_relu", title: "⑦ 합성곱 블록 13 (마지막)", sub: "깊이별 분리 합성곱", desc: "물체 전체에 가까운 '고급 특징'이에요. 7×7 칸 중 물체가 있는 곳이 밝게 켜져요." },
  ];
  const SHOW = 12; // 층마다 보여 줄 특징 지도 수 (가장 강하게 반응한 순)

  // ---------------- 모델 준비 ----------------
  function b64ToBytes(b64) {
    const bin = atob(b64), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function decodeWeights() {
    const specs = [], arrays = [];
    let total = 0;
    for (const w of window.MOBILENET.weights) {
      const n = w.shape.reduce((a, b) => a * b, 1);
      let arr;
      if (w.q) {
        // 채널별 8비트 → 실수로 복원 (채널 축은 메모리 순서상 항상 가장 빨리 바뀌는 축)
        const q = new Int8Array(b64ToBytes(w.q).buffer);
        const s = new Float32Array(b64ToBytes(w.s).buffer);
        const C = w.shape[w.axis];
        arr = new Float32Array(n);
        for (let i = 0; i < n; i++) arr[i] = q[i] * s[i % C];
      } else {
        arr = new Float32Array(b64ToBytes(w.f).buffer);
      }
      specs.push({ name: w.name, shape: w.shape, dtype: "float32" });
      arrays.push(arr);
      total += n;
    }
    const buf = new Float32Array(total);
    let off = 0;
    arrays.forEach((a) => { buf.set(a, off); off += a.length; });
    return { specs, buffer: buf.buffer };
  }

  let model, viz, predW, predB;
  try {
    $("loading-detail").textContent = "가중치 약 420만 개를 푸는 중";
    await new Promise((r) => setTimeout(r, 30));
    const { specs, buffer } = decodeWeights();
    try { await tf.setBackend("webgl"); } catch (e) { await tf.setBackend("cpu"); }
    await tf.ready();
    $("loading-detail").textContent = "신경망 층 28개를 쌓는 중";
    model = await tf.loadLayersModel(tf.io.fromMemory({ modelTopology: window.MOBILENET.topology, weightSpecs: specs, weightData: buffer }));
    const outputs = [...STAGES.map((s) => model.getLayer(s.layer).output), model.getLayer("global_average_pooling2d_1").output, model.getLayer("act_softmax").output];
    viz = tf.model({ inputs: model.inputs, outputs });
    const [k, b] = model.getLayer("conv_preds").getWeights();
    predW = await k.reshape([1024, 1000]).data();
    predB = await b.data();
    // 첫 실행은 느리므로 미리 한 번 돌려 둔다
    tf.tidy(() => viz.predict(tf.zeros([1, 224, 224, 3])));
  } catch (e) {
    $("loading").innerHTML = `<p>모델을 불러오지 못했어요 😢</p><p class="muted small">${e.message}</p>`;
    throw e;
  }
  const backendName = tf.getBackend() === "webgl" ? "그래픽카드(WebGL)" : "CPU";
  $("engine").classList.add("ready");
  $("engine-text").innerHTML = `내 기기에서 딥러닝 계산 중 · ${backendName} · <b id="ms">-</b> ms`;
  $("loading").hidden = true;

  // ---------------- 가운데 단계 만들기 ----------------
  const stageEls = STAGES.map((s, i) => {
    const t = model.getLayer(s.layer).outputShape; // [null, H, W, C]
    const el = document.createElement("article");
    el.className = "stage";
    el.innerHTML = `<header><h3>${s.title} <small>${s.sub}</small></h3><span class="shape">${t[1]} × ${t[2]} × ${t[3]}</span></header>
      <p class="desc">${s.desc} <span class="muted">(특징 지도 ${t[3]}장 중 가장 강하게 반응한 ${SHOW}장)</span></p>
      <div class="maps${t[1] <= 14 ? " px" : ""}"></div>`;
    const box = el.querySelector(".maps");
    const canvases = [];
    for (let k = 0; k < SHOW; k++) {
      const f = document.createElement("figure");
      const c = document.createElement("canvas");
      c.width = t[2]; c.height = t[1];
      f.appendChild(c);
      const cap = document.createElement("figcaption");
      f.appendChild(cap);
      box.appendChild(f);
      canvases.push({ c, cap });
    }
    $("stages").appendChild(el);
    return { el, canvases, H: t[1], W: t[2], C: t[3] };
  });

  const bars = [];
  for (let i = 0; i < 5; i++) {
    const d = document.createElement("div");
    d.className = "bar";
    d.innerHTML = `<div class="row"><b>-</b><span>0%</span></div><div class="track"><div class="fill"></div></div>`;
    $("bars").appendChild(d);
    bars.push(d);
  }

  // ---------------- 색 ----------------
  const STOPS = [[13, 16, 40], [40, 60, 140], [30, 150, 170], [120, 210, 110], [255, 230, 70]];
  function heat(t) {
    t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
    const i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i, a = STOPS[i], b = STOPS[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  function jet(t) { // CAM용: 파랑 → 초록 → 노랑 → 빨강
    t = Math.max(0, Math.min(1, t));
    const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
    const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
    const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
    return [r * 255, g * 255, b * 255];
  }

  // ---------------- 이미지 입력 ----------------
  const inCanvas = $("in-full"), inCtx = inCanvas.getContext("2d", { willReadFrequently: true });
  function drawSquare(src, w, h) {
    const side = Math.min(w, h);
    inCtx.drawImage(src, (w - side) / 2, (h - side) / 2, side, side, 0, 0, 224, 224);
  }

  let busy = false, lastTop = null;
  async function classify() {
    if (busy) return;
    busy = true;
    const t0 = performance.now();
    const outs = tf.tidy(() => {
      const x = tf.browser.fromPixels(inCanvas).toFloat().div(127.5).sub(1).expandDims(0);
      return viz.predict(x);
    });
    const data = await Promise.all(outs.map((t) => t.data()));
    outs.forEach((t) => t.dispose());
    const ms = performance.now() - t0;
    render(data, ms);
    busy = false;
  }

  // ---------------- 화면 그리기 ----------------
  function render(data, ms) {
    $("pipeline").classList.remove("empty");
    $("ms").textContent = ms.toFixed(0);
    const eng = $("engine"); eng.classList.remove("live"); void eng.offsetWidth; eng.classList.add("live");

    // ① RGB 채널
    const img = inCtx.getImageData(0, 0, 224, 224);
    ["in-r", "in-g", "in-b"].forEach((id, ch) => {
      const g = $(id).getContext("2d"), out = g.createImageData(224, 224);
      for (let i = 0; i < 224 * 224; i++) {
        const v = img.data[i * 4 + ch];
        out.data[i * 4] = ch === 0 ? v : 0; out.data[i * 4 + 1] = ch === 1 ? v : 0; out.data[i * 4 + 2] = ch === 2 ? v : 0; out.data[i * 4 + 3] = 255;
      }
      g.putImageData(out, 0, 0);
    });

    // ②~⑦ 특징 지도: 채널별 평균이 가장 큰 순서로
    stageEls.forEach((st, si) => {
      const a = data[si], { H, W, C } = st, HW = H * W;
      const mean = new Float32Array(C);
      for (let p = 0; p < HW; p++) for (let c = 0; c < C; c++) mean[c] += a[p * C + c];
      const top = Array.from(mean.keys()).sort((x, y) => mean[y] - mean[x]).slice(0, SHOW);
      top.forEach((ch, k) => {
        let max = 1e-6;
        for (let p = 0; p < HW; p++) max = Math.max(max, a[p * C + ch]);
        const { c, cap } = st.canvases[k];
        const g = c.getContext("2d"), im = g.createImageData(W, H);
        for (let p = 0; p < HW; p++) {
          const col = heat(Math.pow(a[p * C + ch] / max, 0.8));
          im.data[p * 4] = col[0]; im.data[p * 4 + 1] = col[1]; im.data[p * 4 + 2] = col[2]; im.data[p * 4 + 3] = 255;
        }
        g.putImageData(im, 0, 0);
        cap.textContent = "#" + (ch + 1);
      });
    });

    // ⑧ 전역 평균 풀링
    const gapVec = data[STAGES.length];
    let gmax = 1e-6;
    gapVec.forEach((v) => { gmax = Math.max(gmax, v); });
    const gctx = $("gap").getContext("2d"), gim = gctx.createImageData(1024, 1);
    for (let i = 0; i < 1024; i++) {
      const col = heat(Math.pow(gapVec[i] / gmax, 0.8));
      gim.data[i * 4] = col[0]; gim.data[i * 4 + 1] = col[1]; gim.data[i * 4 + 2] = col[2]; gim.data[i * 4 + 3] = 255;
    }
    gctx.putImageData(gim, 0, 0);

    // ⑨ 1000개 확률
    const probs = data[STAGES.length + 1];
    const order = Array.from(probs.keys()).sort((x, y) => probs[y] - probs[x]);
    const top5 = order.slice(0, 5);
    const pc = $("probs"), pctx = pc.getContext("2d");
    pctx.clearRect(0, 0, pc.width, pc.height);
    const pmax = Math.max(probs[top5[0]], 0.05);
    for (let i = 0; i < 1000; i++) {
      const h = (probs[i] / pmax) * (pc.height - 18);
      pctx.fillStyle = top5.includes(i) ? (i === top5[0] ? "#2f6fed" : "#9db8f5") : "#c9d0dc";
      pctx.fillRect(i, pc.height - h, 1.2, h);
    }
    pctx.fillStyle = "#2f6fed";
    pctx.font = "bold 13px sans-serif";
    const tx = Math.min(Math.max(top5[0], 40), 960);
    pctx.textAlign = "center";
    pctx.fillText(LABELS.ko[top5[0]], tx, 13);
    pctx.fillStyle = "#e2e6ee";
    pctx.fillRect(397, 0, 1, pc.height);

    // 결과
    const p0 = probs[top5[0]];
    $("answer-name").textContent = LABELS.ko[top5[0]];
    $("answer-en").textContent = LABELS.en[top5[0]];
    $("answer-conf").textContent = `${(p0 * 100).toFixed(1)}% 확신 · ` + (p0 > 0.7 ? "확실해요!" : p0 > 0.35 ? "아마 이것 같아요" : "잘 모르겠어요… 🤔");
    top5.forEach((c, i) => {
      bars[i].querySelector("b").textContent = LABELS.ko[c];
      bars[i].querySelector("b").title = LABELS.en[c];
      bars[i].querySelector("span").textContent = (probs[c] * 100).toFixed(1) + "%";
      bars[i].querySelector(".fill").style.width = (probs[c] * 100).toFixed(1) + "%";
    });

    // 어디를 보고 판단했을까? (CAM: 마지막 특징 지도 × 1등 종류의 가중치)
    const last = data[STAGES.length - 1]; // 7×7×1024
    const cls = top5[0];
    cam = new Float32Array(49);
    for (let p = 0; p < 49; p++) {
      let s = 0;
      for (let k = 0; k < 1024; k++) s += last[p * 1024 + k] * predW[k * 1000 + cls];
      cam[p] = s;
    }
    drawCam();
    lastTop = cls;
  }

  let cam = null;
  const camSmall = document.createElement("canvas"); camSmall.width = camSmall.height = 7;
  function drawCam() {
    const cv = $("cam"), g = cv.getContext("2d");
    g.globalAlpha = 1;
    g.drawImage(inCanvas, 0, 0);
    if (!cam) return;
    let mn = Infinity, mx = -Infinity;
    cam.forEach((v) => { mn = Math.min(mn, v); mx = Math.max(mx, v); });
    const sg = camSmall.getContext("2d"), im = sg.createImageData(7, 7);
    for (let p = 0; p < 49; p++) {
      const col = jet((cam[p] - mn) / (mx - mn || 1));
      im.data[p * 4] = col[0]; im.data[p * 4 + 1] = col[1]; im.data[p * 4 + 2] = col[2]; im.data[p * 4 + 3] = 255;
    }
    sg.putImageData(im, 0, 0);
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";
    g.globalAlpha = +$("cam-alpha").value / 100;
    g.drawImage(camSmall, 0, 0, 224, 224);
    g.globalAlpha = 1;
  }
  $("cam-alpha").addEventListener("input", drawCam);

  // ---------------- 입력 방법들 ----------------
  const preview = $("preview"), video = $("video");
  function showImage(src) {
    stopCam();
    const im = new Image();
    im.onload = () => {
      preview.src = src;
      preview.hidden = false;
      $("drop-msg").hidden = true;
      drawSquare(im, im.naturalWidth, im.naturalHeight);
      classify();
    };
    im.onerror = () => alert("이 파일은 이미지로 열 수 없어요.");
    im.src = src;
  }
  function loadFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const r = new FileReader();
    r.onload = () => showImage(r.result);
    r.readAsDataURL(file);
  }
  $("file").addEventListener("change", (e) => { loadFile(e.target.files[0]); e.target.value = ""; });
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) loadFile(item.getAsFile());
  });

  (window.SAMPLE_IMAGES || []).forEach((s) => {
    const b = document.createElement("button");
    b.title = s.name.replace(/^\d+_/, "");
    b.innerHTML = `<img src="${s.src}" alt="${b.title}">`;
    b.addEventListener("click", () => showImage(s.src));
    $("samples").appendChild(b);
  });

  // 웹캠: 실시간으로 계속 분류
  let stream = null, frozen = false;
  async function startCam() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 640 } }, audio: false });
    } catch (e) {
      $("cam-msg").textContent = "웹캠을 켤 수 없어요. 브라우저 주소창의 카메라 권한을 확인해 주세요. (" + e.name + ")";
      return;
    }
    $("cam-msg").textContent = "";
    video.srcObject = stream;
    await video.play();
    video.hidden = false; preview.hidden = true; $("drop-msg").hidden = true;
    $("live-badge").hidden = false;
    $("btn-cam").textContent = "■ 웹캠 끄기"; $("btn-cam").classList.add("on");
    $("btn-snap").hidden = false; $("btn-snap").textContent = "⏸ 이 장면 멈추기";
    frozen = false;
    camLoop();
  }
  function stopCam() {
    if (!stream) return;
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
    video.hidden = true; $("live-badge").hidden = true;
    $("btn-cam").textContent = "📷 웹캠 켜기"; $("btn-cam").classList.remove("on");
    $("btn-snap").hidden = true;
    if (preview.src && !frozen) preview.hidden = false; else if (!frozen) $("drop-msg").hidden = false;
  }
  async function camLoop() {
    if (!stream || frozen) return;
    if (video.videoWidth) {
      drawSquare(video, video.videoWidth, video.videoHeight);
      await classify();
    }
    setTimeout(() => requestAnimationFrame(camLoop), 120);
  }
  $("btn-cam").addEventListener("click", () => (stream ? stopCam() : startCam()));
  $("btn-snap").addEventListener("click", () => {
    frozen = !frozen;
    $("btn-snap").textContent = frozen ? "▶ 다시 실시간으로" : "⏸ 이 장면 멈추기";
    $("live-badge").hidden = frozen;
    if (frozen) video.pause(); else { video.play(); camLoop(); }
  });

  // ---------------- 1000가지 목록 ----------------
  function renderList(q) {
    q = (q || "").trim().toLowerCase();
    $("class-list").innerHTML = LABELS.ko.map((k, i) => [k, i])
      .filter(([k, i]) => !q || k.toLowerCase().includes(q) || LABELS.en[i].toLowerCase().includes(q))
      .map(([k, i]) => `<div title="${LABELS.en[i]}"><small>${i}</small>${k}</div>`).join("") || '<p class="muted">찾는 종류가 없어요. 이 모델은 그 물체를 모를 수 있어요!</p>';
  }
  $("btn-classes").addEventListener("click", () => { renderList(""); $("classes-modal").hidden = false; $("class-search").focus(); });
  $("class-search").addEventListener("input", (e) => renderList(e.target.value));
  const closeModal = () => { $("classes-modal").hidden = true; };
  document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeModal));
  $("classes-modal").addEventListener("click", (e) => { if (e.target.id === "classes-modal") closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
})();

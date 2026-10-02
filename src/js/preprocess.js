// 캔버스에 그린 손글씨를 MNIST와 같은 형식(28x28, 검은 배경에 흰 글씨)으로 바꾼다.
// MNIST 방식: 글씨 부분만 잘라 20x20 안에 비율 유지해 넣고, 무게중심을 28x28의 가운데에 맞춘다.
(function () {
  function inkMap(canvas) {
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const { width, height } = canvas;
    const data = ctx.getImageData(0, 0, width, height).data;
    const ink = new Float32Array(width * height);
    let minX = width, minY = height, maxX = -1, maxY = -1;
    for (let i = 0, p = 0; p < ink.length; i += 4, p++) {
      // 흰 종이(255) 위의 검은 잉크(0) → 잉크 진하기 0~1
      const v = (255 - (data[i] + data[i + 1] + data[i + 2]) / 3) / 255;
      ink[p] = v;
      if (v > 0.1) {
        const x = p % width, y = (p / width) | 0;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    if (maxX < 0) return null;
    return { ink, width, height, box: { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 } };
  }

  // 영역 평균으로 축소 (src의 사각 영역 → dw x dh)
  function resizeArea(src, sw, box, dw, dh) {
    const out = new Float32Array(dw * dh);
    const sx = box.w / dw, sy = box.h / dh;
    for (let y = 0; y < dh; y++) {
      const y0 = box.y + y * sy, y1 = y0 + sy;
      for (let x = 0; x < dw; x++) {
        const x0 = box.x + x * sx, x1 = x0 + sx;
        let sum = 0, area = 0;
        for (let yy = Math.floor(y0); yy < Math.ceil(y1); yy++) {
          const wy = Math.min(yy + 1, y1) - Math.max(yy, y0);
          if (wy <= 0) continue;
          for (let xx = Math.floor(x0); xx < Math.ceil(x1); xx++) {
            const wx = Math.min(xx + 1, x1) - Math.max(xx, x0);
            if (wx <= 0) continue;
            sum += src[yy * sw + xx] * wx * wy;
            area += wx * wy;
          }
        }
        out[y * dw + x] = area ? sum / area : 0;
      }
    }
    return out;
  }

  function process(canvas) {
    const m = inkMap(canvas);
    if (!m) return null;
    // 1) 글씨를 감싸는 정사각형 영역 (비율 유지를 위해 짧은 쪽을 늘림)
    const side = Math.max(m.box.w, m.box.h);
    const cx = m.box.x + m.box.w / 2, cy = m.box.y + m.box.h / 2;
    const sq = { x: cx - side / 2, y: cy - side / 2, w: side, h: side };
    // 캔버스 밖은 0으로 보고 패딩된 잉크 지도를 만든다
    const pad = Math.ceil(side);
    const pw = m.width + pad * 2;
    const padded = new Float32Array(pw * (m.height + pad * 2));
    for (let y = 0; y < m.height; y++) padded.set(m.ink.subarray(y * m.width, (y + 1) * m.width), (y + pad) * pw + pad);
    const box = { x: sq.x + pad, y: sq.y + pad, w: sq.w, h: sq.h };

    // 2) 20x20으로 축소
    const small = resizeArea(padded, pw, box, 20, 20);

    // 3) 28x28 가운데에 놓고 무게중심을 (14,14)로 이동
    let sum = 0, mx = 0, my = 0;
    for (let y = 0; y < 20; y++) for (let x = 0; x < 20; x++) {
      const v = small[y * 20 + x];
      sum += v; mx += v * x; my += v * y;
    }
    mx = mx / sum; my = my / sum;
    const ox = Math.round(13.5 - mx), oy = Math.round(13.5 - my);
    const out = new Float32Array(784);
    for (let y = 0; y < 20; y++) for (let x = 0; x < 20; x++) {
      const tx = x + ox, ty = y + oy;
      if (tx >= 0 && tx < 28 && ty >= 0 && ty < 28) out[ty * 28 + tx] = Math.min(1, small[y * 20 + x]);
    }
    return { input: out, crop: m.box, square: sq, small };
  }

  window.Preprocess = { process };
})();

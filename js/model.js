// 교과서 '딥러닝 실습'과 같은 구조의 합성곱 신경망(CNN)을 순수 자바스크립트로 계산한다.
// 서버 없이 각 학생의 브라우저 안에서 바로 실행된다.
(function () {
  function decode(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  // 합성곱층 (3x3 필터, 패딩 없음) + ReLU
  function conv3x3Relu(input, inC, size, w, b, outC) {
    const out = size - 2;
    const result = new Float32Array(outC * out * out);
    for (let o = 0; o < outC; o++) {
      const base = o * out * out;
      for (let y = 0; y < out; y++) {
        for (let x = 0; x < out; x++) {
          let sum = b[o];
          for (let c = 0; c < inC; c++) {
            const wi = (o * inC + c) * 9;
            const ii = c * size * size + y * size + x;
            sum += input[ii] * w[wi] + input[ii + 1] * w[wi + 1] + input[ii + 2] * w[wi + 2]
              + input[ii + size] * w[wi + 3] + input[ii + size + 1] * w[wi + 4] + input[ii + size + 2] * w[wi + 5]
              + input[ii + 2 * size] * w[wi + 6] + input[ii + 2 * size + 1] * w[wi + 7] + input[ii + 2 * size + 2] * w[wi + 8];
          }
          result[base + y * out + x] = sum > 0 ? sum : 0;
        }
      }
    }
    return result;
  }

  // 풀링층 (2x2 최댓값 풀링)
  function maxPool2(input, ch, size) {
    const out = Math.floor(size / 2);
    const result = new Float32Array(ch * out * out);
    for (let c = 0; c < ch; c++) {
      for (let y = 0; y < out; y++) {
        for (let x = 0; x < out; x++) {
          const i = c * size * size + 2 * y * size + 2 * x;
          result[c * out * out + y * out + x] = Math.max(input[i], input[i + 1], input[i + size], input[i + size + 1]);
        }
      }
    }
    return result;
  }

  // 완전 연결층
  function dense(input, w, b, outN, relu) {
    const inN = input.length;
    const result = new Float32Array(outN);
    for (let o = 0; o < outN; o++) {
      let sum = b[o];
      const row = o * inN;
      for (let i = 0; i < inN; i++) sum += input[i] * w[row + i];
      result[o] = relu && sum < 0 ? 0 : sum;
    }
    return result;
  }

  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = Array.from(logits, (v) => Math.exp(v - max));
    const total = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / total);
  }

  const W = window.MODEL_WEIGHTS;
  const p = {};
  for (const layer of ["conv1", "conv2", "fc1", "fc2"]) {
    p[layer] = { w: decode(W[layer].w), b: decode(W[layer].b) };
  }

  // input: 28x28 Float32Array (0~1, 검은 배경에 흰 글씨)
  function predict(input) {
    const conv1 = conv3x3Relu(input, 1, 28, p.conv1.w, p.conv1.b, 32); // 26x26x32
    const pool1 = maxPool2(conv1, 32, 26);                              // 13x13x32
    const conv2 = conv3x3Relu(pool1, 32, 13, p.conv2.w, p.conv2.b, 64); // 11x11x64
    const pool2 = maxPool2(conv2, 64, 11);                              // 5x5x64
    const flat = pool2;                                                 // 1600
    const fc1 = dense(flat, p.fc1.w, p.fc1.b, 128, true);               // 128
    const logits = dense(fc1, p.fc2.w, p.fc2.b, 10, false);             // 10
    return { input, conv1, pool1, conv2, pool2, flat, fc1, logits, probs: softmax(logits) };
  }

  window.CNN = {
    predict,
    filters1: p.conv1.w, // 32개의 3x3 필터 (첫 번째 합성곱층)
    testAccuracy: W.testAccuracy,
  };
})();

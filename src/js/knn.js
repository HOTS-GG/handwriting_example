// 교과서 '기계학습 실습'의 k-최근접 이웃 분류를 브라우저에서 계산한다.
(function () {
  const S = window.KNN_SAMPLES;
  function bytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  const images = bytes(S.images);
  const labels = bytes(S.labels);

  function classify(input, k = 5) {
    const dists = new Float32Array(S.count);
    for (let n = 0; n < S.count; n++) {
      let d = 0;
      const off = n * 784;
      for (let i = 0; i < 784; i++) {
        const diff = input[i] - images[off + i] / 255;
        d += diff * diff;
      }
      dists[n] = d;
    }
    const order = Array.from(dists.keys()).sort((a, b) => dists[a] - dists[b]).slice(0, k);
    const votes = new Array(10).fill(0);
    order.forEach((n) => votes[labels[n]]++);
    // 동점이면 더 가까운 이웃의 숫자를 고른다
    let best = labels[order[0]];
    votes.forEach((v, d) => { if (v > votes[best]) best = d; });
    return {
      label: best,
      votes,
      neighbors: order.map((n) => ({ label: labels[n], dist: Math.sqrt(dists[n]), image: images.subarray(n * 784, (n + 1) * 784) })),
    };
  }

  window.KNN = { classify, count: S.count };
})();

"""교과서(그림 II-35, 딥러닝 실습)와 같은 구조의 CNN을 MNIST로 학습하고
웹페이지에서 쓸 가중치 파일(src/js/model-weights.js)과 KNN 비교용 샘플(src/js/knn-samples.js)을 만든다.
만든 뒤에는 python tools/build.py 로 단일 index.html 을 다시 만든다.

구조: Conv2D(32,3x3,relu) → MaxPool(2) → Conv2D(64,3x3,relu) → MaxPool(2)
      → Flatten → Dense(128,relu) → Dense(10,softmax)

사용법:  python tools/train.py path/to/mnist.npz
(mnist.npz: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)
"""
import base64, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))

ROOT = Path(__file__).resolve().parent.parent
data = np.load(sys.argv[1] if len(sys.argv) > 1 else "mnist.npz")
x_train = torch.tensor(data["x_train"], dtype=torch.float32).unsqueeze(1) / 255.0
y_train = torch.tensor(data["y_train"], dtype=torch.long)
x_test = torch.tensor(data["x_test"], dtype=torch.float32).unsqueeze(1) / 255.0
y_test = torch.tensor(data["y_test"], dtype=torch.long)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def augment(x):
    """손으로 그린 글씨는 MNIST보다 기울기·크기·위치·굵기가 제각각이라 약간 흔들어서 학습한다."""
    n = x.shape[0]
    ang = (torch.rand(n) - 0.5) * 2 * math.radians(12)
    scale = 1 + (torch.rand(n) - 0.5) * 0.25
    shear = (torch.rand(n) - 0.5) * 0.3
    tx = (torch.rand(n) - 0.5) * 0.2
    ty = (torch.rand(n) - 0.5) * 0.2
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([
        torch.stack([cos, -sin + shear, tx], 1),
        torch.stack([sin, cos, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    # 획 굵기 변화: 일부는 굵게(팽창), 일부는 얇게(침식)
    r = torch.rand(n)
    thick = F.max_pool2d(x, 3, 1, 1)
    thin = -F.max_pool2d(-x, 2, 1, 0)
    thin = F.pad(thin, (0, 1, 0, 1))
    x = torch.where((r < 0.2).view(-1, 1, 1, 1), thick, x)
    x = torch.where((r > 0.9).view(-1, 1, 1, 1), thin, x)
    return x


model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS, BS = 12, 128
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * math.ceil(len(x_train) / BS))

for epoch in range(EPOCHS):
    model.train()
    t0 = time.time()
    perm = torch.randperm(len(x_train))
    for i in range(0, len(x_train), BS):
        idx = perm[i:i + BS]
        xb = augment(x_train[idx])
        loss = F.cross_entropy(model(xb), y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x_test).argmax(1) == y_test).float().mean().item()
    print(f"epoch {epoch + 1}/{EPOCHS}  loss {loss.item():.4f}  test acc {acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)


def pack(t):
    return base64.b64encode(t.detach().numpy().astype("<f4").tobytes()).decode()


sd = model.state_dict()
weights = {
    "testAccuracy": round(acc, 4),
    "conv1": {"w": pack(sd["conv1.weight"]), "b": pack(sd["conv1.bias"])},   # [32][1][3][3]
    "conv2": {"w": pack(sd["conv2.weight"]), "b": pack(sd["conv2.bias"])},   # [64][32][3][3]
    "fc1": {"w": pack(sd["fc1.weight"]), "b": pack(sd["fc1.bias"])},         # [128][1600] (CHW 평탄화)
    "fc2": {"w": pack(sd["fc2.weight"]), "b": pack(sd["fc2.bias"])},         # [10][128]
}
(ROOT / "src" / "js" / "model-weights.js").write_text(
    "// tools/train.py 로 생성된 파일입니다. (MNIST 학습 CNN 가중치, float32 base64)\n"
    "window.MODEL_WEIGHTS = " + json.dumps(weights) + ";\n")

# KNN(k-최근접 이웃) 비교용: 숫자별 200장씩, 총 2000장
PER = 200
xs, ys = [], []
for d in range(10):
    idx = np.where(data["y_train"] == d)[0][:PER]
    xs.append(data["x_train"][idx])
    ys.append(np.full(PER, d, dtype=np.uint8))
xs = np.concatenate(xs).astype(np.uint8)
ys = np.concatenate(ys)
(ROOT / "src" / "js" / "knn-samples.js").write_text(
    "// tools/train.py 로 생성된 파일입니다. (MNIST 학습 데이터 일부, 28x28 uint8 base64)\n"
    "window.KNN_SAMPLES = " + json.dumps({
        "count": int(len(ys)),
        "images": base64.b64encode(xs.tobytes()).decode(),
        "labels": base64.b64encode(ys.tobytes()).decode(),
    }) + ";\n")

# JS 구현 검증용: 테스트 이미지 몇 장과 파이토치 출력
with torch.no_grad():
    logits = model(x_test[:5])
(ROOT / "tools" / "reference.json").write_text(json.dumps({
    "images": [base64.b64encode(data["x_test"][i].astype(np.uint8).tobytes()).decode() for i in range(5)],
    "probs": F.softmax(logits, 1).tolist(),
}))
print("saved", acc)

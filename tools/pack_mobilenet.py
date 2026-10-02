"""구글이 공개한 MobileNet v1 (ImageNet 1000종, tfjs 형식)을 내려받아
8비트(채널별 int8)로 압축해 src/classifier/model-data.js 로 저장한다.

사용법:  python tools/pack_mobilenet.py [샘플이미지폴더]
  - 한국어 분류 이름: tools/imagenet_ko.tsv
  - 샘플 이미지(선택): 지정한 폴더의 jpg/png 를 320px 로 줄여 src/classifier/samples.js 에 저장
"""
import base64, io, json, sys, urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "classifier"
BASE = "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/"
CACHE = ROOT / "tools" / ".cache" / "mobilenet_v1_1.0_224"


def fetch(name):
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("download", name)
        urllib.request.urlretrieve(BASE + name, path)
    return path.read_bytes()


model = json.loads(fetch("model.json"))
b64 = lambda a: base64.b64encode(a.tobytes()).decode()
specs = []
raw_total = 0
for group in model["weightsManifest"]:
    buf = b"".join(fetch(p) for p in group["paths"])
    offset = 0
    for w in group["weights"]:
        n = int(np.prod(w["shape"]))
        arr = np.frombuffer(buf, dtype="<f4", count=n, offset=offset).reshape(w["shape"])
        offset += n * 4
        raw_total += n * 4
        spec = {"name": w["name"], "shape": w["shape"]}
        if arr.ndim == 4:
            # 합성곱 커널: 출력 채널(깊이별 합성곱은 입력 채널)마다 따로 스케일을 둔다
            axis = 2 if "conv_dw" in w["name"] else 3
            moved = np.moveaxis(arr, axis, -1)
            scale = np.abs(moved).reshape(-1, moved.shape[-1]).max(0) / 127.0
            scale[scale == 0] = 1e-8
            q = np.clip(np.round(moved / scale), -127, 127).astype(np.int8)
            spec.update(q=b64(np.ascontiguousarray(q)), s=b64(scale.astype("<f4")), axis=axis)
        else:
            spec["f"] = b64(arr.astype("<f4"))
        specs.append(spec)

topology = model["modelTopology"]
labels_ko = [l.split("\t", 1)[1].strip() for l in (ROOT / "tools" / "imagenet_ko.tsv").read_text(encoding="utf-8").splitlines()]
labels_en = json.loads((ROOT / "tools" / "imagenet_en.json").read_text(encoding="utf-8"))
assert len(labels_ko) == len(labels_en) == 1000

(OUT / "model-data.js").write_text(
    "// tools/pack_mobilenet.py 로 생성된 파일입니다.\n"
    "// MobileNet v1 1.0 224 (ImageNet), Google, Apache License 2.0 — 가중치를 채널별 8비트로 압축\n"
    "window.MOBILENET = " + json.dumps({"topology": topology, "weights": specs}) + ";\n"
    "window.IMAGENET_LABELS = " + json.dumps({"ko": labels_ko, "en": labels_en}, ensure_ascii=False) + ";\n",
    encoding="utf-8")
print(f"model-data.js: {(OUT / 'model-data.js').stat().st_size / 1e6:.1f} MB (원본 {raw_total / 1e6:.1f} MB)")

if len(sys.argv) > 1:
    from PIL import Image
    samples = []
    for p in sorted(Path(sys.argv[1]).iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((320, 320))
        bio = io.BytesIO()
        im.save(bio, "JPEG", quality=82)
        samples.append({"name": p.stem, "src": "data:image/jpeg;base64," + base64.b64encode(bio.getvalue()).decode()})
    (OUT / "samples.js").write_text(
        "// tools/pack_mobilenet.py 로 생성된 파일입니다. (공개 예시 이미지, 출처는 README 참고)\n"
        "window.SAMPLE_IMAGES = " + json.dumps(samples) + ";\n", encoding="utf-8")
    print("samples.js:", len(samples), "장")

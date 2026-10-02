"""src/ 의 각 페이지(HTML·CSS·JS)를 합쳐 파일 하나짜리 HTML로 만든다.

  src/handwriting/  → index.html                  (손글씨 숫자 인식 실험실)
  src/process/      → deep-learning-process.html  (딥러닝 학습 과정 들여다보기)
  src/classifier/   → image-classifier.html       (이미지 분류 실험실)

사용법:  python tools/build.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = {
    "handwriting": "index.html",
    "process": "deep-learning-process.html",
    "classifier": "image-classifier.html",
}


def build(src: Path, out: Path):
    html = (src / "index.html").read_text(encoding="utf-8")

    def inline_css(m):
        return "<style>\n" + (src / m.group(1)).read_text(encoding="utf-8") + "</style>"

    def inline_js(m):
        js = (src / m.group(1)).read_text(encoding="utf-8").replace("</script", "<\\/script")
        return "<script>\n" + js + "</script>"

    html = re.sub(r'<link rel="stylesheet" href="([^"]+)">', inline_css, html)
    html = re.sub(r'<script src="([^"]+)"></script>', inline_js, html)
    html = html.replace("<head>", f"<head>\n<!-- 이 파일은 tools/build.py 로 src/{src.name}/ 에서 자동 생성됩니다. 수정은 src/ 에서 하세요. -->", 1)
    out.write_text(html, encoding="utf-8")
    print(f"{out.name}: {out.stat().st_size / 1024 / 1024:.1f} MB")


for folder, name in PAGES.items():
    if (ROOT / "src" / folder / "index.html").exists():
        build(ROOT / "src" / folder, ROOT / name)

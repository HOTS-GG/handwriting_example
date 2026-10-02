"""src/ 의 HTML·CSS·JS를 모두 합쳐 파일 하나짜리 index.html 을 만든다.

사용법:  python tools/build.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

html = (SRC / "index.html").read_text(encoding="utf-8")


def inline_css(m):
    css = (SRC / m.group(1)).read_text(encoding="utf-8")
    return f"<style>\n{css}</style>"


def inline_js(m):
    js = (SRC / m.group(1)).read_text(encoding="utf-8").replace("</script", "<\\/script")
    return f"<script>\n{js}</script>"


html = re.sub(r'<link rel="stylesheet" href="([^"]+)">', inline_css, html)
html = re.sub(r'<script src="([^"]+)"></script>', inline_js, html)
html = html.replace("<head>", "<head>\n<!-- 이 파일은 tools/build.py 로 src/ 에서 자동 생성됩니다. 수정은 src/ 에서 하세요. -->", 1)

out = ROOT / "index.html"
out.write_text(html, encoding="utf-8")
print(f"{out.name}: {out.stat().st_size / 1024 / 1024:.1f} MB")

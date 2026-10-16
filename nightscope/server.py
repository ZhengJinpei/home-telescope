"""手机遥控：树莓派开 Wi-Fi 热点，手机浏览器打开 http://10.42.0.1:8080 就能看画面、按按钮。

只用标准库：画面用 MJPEG（一张接一张的 JPEG），所有手机浏览器都能直接显示。
"""

from __future__ import annotations

import json
import queue
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BOUNDARY = "nightscope-frame"


class Hub:
    """主循环和网页之间传东西：最新一帧画面、当前状态、手机上按的按钮。"""

    def __init__(self, media_dir: Path):
        self.media_dir = media_dir
        self.state: dict = {}
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._clients = 0
        self._actions: queue.Queue[dict] = queue.Queue()
        self.closed = False

    # 主循环这边
    @property
    def has_viewers(self) -> bool:
        return self._clients > 0

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def pop_actions(self) -> list[dict]:
        out = []
        while not self._actions.empty():
            out.append(self._actions.get_nowait())
        return out

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    # 网页这边
    def push_action(self, action: dict) -> None:
        self._actions.put(action)

    def next_frame(self, last_seq: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or self.closed, timeout)
            return self._seq, self._jpeg

    def viewer(self, delta: int) -> None:
        with self._cond:
            self._clients += delta


_SAFE_NAME = re.compile(r"^[\w.-]+$")


def make_handler(hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NightScope/1.0"

        def log_message(self, fmt, *args):  # 不刷屏
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(hub.state)
            elif path == "/stream.mjpg":
                self._stream()
            elif path == "/media":
                files = sorted(hub.media_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                self._json([{"name": p.name, "bytes": p.stat().st_size} for p in files if p.is_file()])
            elif path.startswith("/media/"):
                name = path[len("/media/"):]
                target = hub.media_dir / name
                if not _SAFE_NAME.match(name) or not target.is_file():
                    self._json({"error": "没有这个文件"}, HTTPStatus.NOT_FOUND)
                    return
                kind = "image/jpeg" if name.endswith(".jpg") else "application/octet-stream"
                self._send(200, target.read_bytes(), kind)
            else:
                self._json({"error": "没有这个页面"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "格式不对"}, HTTPStatus.BAD_REQUEST)
                return
            if self.path == "/api/press":
                hub.push_action({"press": str(data.get("button")), "long": bool(data.get("long"))})
            elif self.path == "/api/ir":
                hub.push_action({"ir_level": int(data.get("level", 0))})
            else:
                self._json({"error": "没有这个接口"}, HTTPStatus.NOT_FOUND)
                return
            self._json({"ok": True})

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            hub.viewer(+1)
            seq = -1
            try:
                while not hub.closed:
                    new_seq, jpeg = hub.next_frame(seq)
                    if new_seq == seq or jpeg is None:  # 等超时了，没有新画面
                        continue
                    seq = new_seq
                    self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                     f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.viewer(-1)

    return Handler


def start_server(hub: Hub, host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(hub))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# 夜里看手机屏幕会晃眼：整页用暗红色，不破坏眼睛的暗适应
PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>夜视望远镜</title>
<style>
  :root { --bg:#000; --fg:#c43; --dim:#622; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px system-ui, sans-serif; }
  img { display:block; width:100%; background:#000; }
  #status { padding:8px 12px; font-size:14px; min-height:20px; }
  .row { display:flex; gap:8px; padding:0 12px 8px; }
  button { flex:1; padding:14px 0; font-size:16px; background:#100; color:var(--fg);
           border:1px solid var(--dim); border-radius:10px; }
  button.on { background:#400; }
  a { color:var(--fg); }
</style></head><body>
<img id="view" src="/stream.mjpg" alt="实时画面">
<div id="status">连接中…</div>
<div class="row">
  <button data-b="mode">白天/夜间</button>
  <button data-b="zoom">变倍</button>
  <button data-b="shutter">拍照</button>
</div>
<div class="row">
  <button data-b="shutter" data-long="1" id="rec">录像</button>
  <button data-b="mode" data-long="1">换颜色/自动</button>
</div>
<div class="row" id="ir">
  <button data-ir="0">红外 关</button><button data-ir="1">弱</button>
  <button data-ir="2">中</button><button data-ir="3">强</button>
</div>
<div class="row"><a href="/media" target="_blank">照片和录像</a></div>
<script>
const post = (url, body) => fetch(url, {method:"POST", headers:{"Content-Type":"application/json"},
                                         body:JSON.stringify(body)});
document.querySelectorAll("[data-b]").forEach(b => b.onclick = () =>
  post("/api/press", {button:b.dataset.b, long:!!b.dataset.long}));
document.querySelectorAll("[data-ir]").forEach(b => b.onclick = () =>
  post("/api/ir", {level:+b.dataset.ir}));
async function refresh() {
  try {
    const s = await (await fetch("/api/state")).json();
    const mode = s.mode === "night" ? "夜间" : "白天";
    document.getElementById("status").textContent =
      `${mode}${s.auto_mode ? "（自动）" : ""} · ${s.zoom}× · 红外 ${["关","弱","中","强"][s.ir_level]}` +
      `${s.recording ? " · ● 录像中" : ""} · ${s.message || ""}`;
    document.getElementById("rec").classList.toggle("on", s.recording);
    document.querySelectorAll("[data-ir]").forEach(b =>
      b.classList.toggle("on", +b.dataset.ir === s.ir_level));
  } catch (e) { document.getElementById("status").textContent = "连接断开，正在重连…"; }
}
setInterval(refresh, 1000); refresh();
</script></body></html>
"""

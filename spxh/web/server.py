"""SPXH 本地 Web 服务（纯标准库，无额外依赖）.

    python -m spxh serve --host 127.0.0.1 --port 8760

- /                -> spxh/web/static/index.html
- /static/<file>   -> 静态资源（限制在 static 目录内）
- /api/*           -> spxh.api 的 JSON 接口（契约见 docs/前端接口约定-v1.md）

只用标准库，是为了让"打开就能用"这件事不依赖任何额外安装；生产化（鉴权、并发、日志）
不是本项目当前阶段的目标。
"""
from __future__ import annotations

import json
import mimetypes
import posixpath
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from spxh.web.auth import AuthConfig
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, unquote, urlparse

import spxh.api as api

__all__ = ["SPXHRequestHandler", "SPXHHTTPServer", "create_server", "serve", "STATIC_DIR"]


def _safe_print(text: str) -> None:
    """日志写不出去（管道断开）时绝不能让服务本身崩掉."""
    try:
        print(text, flush=True)
    except (BrokenPipeError, OSError, ValueError):
        pass


class SPXHHTTPServer(ThreadingHTTPServer):
    """对"客户端中途断开"免疫：这类事件在浏览器刷新/关标签页时非常常见.

    实测教训：工作台曾经因为一串 ConnectionResetError（浏览器在慢请求途中断开）
    打满日志、进程直接退出 —— 平台不能因为用户按了一次刷新就整体不可用。
    """

    daemon_threads = True
    allow_reuse_address = True
    reset_count = 0
    error_count = 0

    def handle_error(self, request, client_address) -> None:  # noqa: D102
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError)):
            self.reset_count += 1
            if self.reset_count <= 3 or self.reset_count % 50 == 0:
                _safe_print("[spxh] 客户端中断连接（累计 {} 次，已忽略）".format(self.reset_count))
            return
        self.error_count += 1
        summary = "{}: {}".format(type(exc).__name__, exc) if exc is not None else "unknown"
        _safe_print("[spxh] 请求处理异常（累计 {} 次）：{}".format(self.error_count, summary))
        if not isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            try:
                traceback.print_exc()
            except (BrokenPipeError, OSError, ValueError):
                pass

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _optional_int(body: Any, key: str) -> Optional[int]:
    value = body.get(key) if hasattr(body, "get") else None
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError("参数 {} 必须是整数".format(key)) from None


def _optional_float(body: Any, key: str) -> Optional[float]:
    value = body.get(key) if hasattr(body, "get") else None
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("参数 {} 必须是数字".format(key)) from None


def _first(query: dict[str, list[str]], key: str, default: Optional[str] = None) -> Optional[str]:
    values = query.get(key)
    if not values:
        return default
    return values[0]


def _int_param(query: dict[str, list[str]], key: str, default: int) -> int:
    value = _first(query, key)
    if value is None or value == "":
        return int(default)
    return int(float(value))


def _float_param(query: dict[str, list[str]], key: str) -> Optional[float]:
    value = _first(query, key)
    if value is None or value == "":
        return None
    return float(value)


class SPXHRequestHandler(BaseHTTPRequestHandler):
    server_version = "SPXH/0.1"
    protocol_version = "HTTP/1.1"
    model_dir = "models"
    # 逐请求日志默认写文件（而不是刷 stdout）：把服务挂在后台作业里时，
    # stdout 是一条管道，写爆它会把整个进程带走（实测：工作台反复"无声退出"，
    # 同一台机器上 pytest 的大量输出也会被同样打断 —— 把日志落到文件后恢复正常）。
    quiet = True
    log_path: Optional[Path] = None
    #: 访问令牌策略（默认：只服务回环地址且无令牌 -> 不鉴权）
    auth: AuthConfig = AuthConfig()

    # ------------------------------------------------------------------ 工具
    def _safe_write(self, body: bytes) -> bool:
        """写响应；客户端已经断开时安静收场（不再抛异常、不再打 traceback）.

        浏览器刷新/关标签页会让"正在写的 HTTP 响应"瞬间失去对端，这是**常态**而不是故障。
        """
        try:
            self.wfile.write(body)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True
            return False

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True
            return
        self._safe_write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_file(self, path: Path, content_type: Optional[str] = None) -> None:
        if not path.exists() or not path.is_file():
            self._send_error_json(404, "静态资源不存在：" + str(path.name))
            return
        body = path.read_bytes()
        guessed = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if guessed.startswith("text/") or guessed in ("application/javascript", "application/json"):
            guessed += "; charset=utf-8"
        try:
            self.send_response(200)
            self.send_header("Content-Type", guessed)
            self.send_header("Content-Length", str(len(body)))
            # 静态资源一律不缓存：前端更新后（尤其是多人并行改 app.js 时）
            # 浏览器拿到旧 bundle 会表现为"页面卡在加载中"这类诡异现象
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True
            return
        self._safe_write(body)

    def _static_path(self, relative: str) -> Optional[Path]:
        cleaned = posixpath.normpath("/" + unquote(relative)).lstrip("/")
        candidate = (STATIC_DIR / cleaned).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return None
        return candidate

    # ------------------------------------------------------------------ 路由
    def _auth_ok(self, route: str) -> bool:
        if self.auth.is_public(route) or not self.auth.required:
            return True
        return self.auth.allows(self.headers)

    def _reject_unauthorized(self) -> None:
        self._send_error_json(401, "需要访问令牌（401）：请在「设置」页填入令牌，或用 Authorization: Bearer <token>")

    def do_GET(self) -> None:  # noqa: N802 - http.server 规定的名字
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                index = STATIC_DIR / "index.html"
                if not index.exists():
                    self._send_json(
                        {
                            "error": "前端静态资源尚未生成（spxh/web/static/index.html 不存在）",
                            "hint": "先完成前端构建，或直接使用 /api/* 接口",
                        },
                        status=503,
                    )
                    return
                self._send_file(index, "text/html; charset=utf-8")
                return

            if route == "/favicon.ico":
                # 浏览器会自动请求 favicon；不处理的话控制台会出现一条无害但扎眼的 404
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if route.startswith("/static/"):
                relative = route[len("/static/") :]
                target = self._static_path(relative)
                if target is None:
                    self._send_error_json(403, "非法的静态资源路径")
                    return
                if not target.exists() and "/mock/" in "/" + relative:
                    # mock 模式下前端按 <slug>.<endpoint>.json 取假数据；某些案例没有专属
                    # 假数据（例如新加的信号），这里回退到通用的 <endpoint>.json，
                    # 避免整页报 404 却看不出原因
                    name = target.name
                    if "." in name:
                        fallback = target.with_name(".".join(name.split(".")[-2:]))
                        if fallback.exists():
                            target = fallback
                self._send_file(target)
                return

            if route.startswith("/api/"):
                if not self._auth_ok(route):
                    self._reject_unauthorized()
                    return
                self._handle_api(route[len("/api/") :], query)
                return

            self._send_error_json(404, "未知路径：" + route)
        except PermissionError as exc:
            self._send_error_json(403, str(exc))
        except FileNotFoundError as exc:
            self._send_error_json(404, str(exc))
        except ValueError as exc:
            self._send_error_json(400, "参数错误：" + str(exc))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
            # 客户端在响应写完之前断开：这是常态，安静收场即可
            self.close_connection = True
            _safe_print("[spxh] 客户端提前断开：{}".format(type(exc).__name__))
        except Exception as exc:  # noqa: BLE001 - 服务端兜底，必须把原因返回给前端
            try:
                traceback.print_exc()
            except (BrokenPipeError, OSError, ValueError):
                pass
            self._send_error_json(500, "服务端异常：{}: {}".format(type(exc).__name__, exc))

    def _handle_api(self, endpoint: str, query: dict[str, list[str]]) -> None:
        path = _first(query, "path")
        needs_path = {"analyze", "spectrum", "waterfall", "constellation", "features", "classify", "demod",
                      "frame", "ssca", "narrate"}
        # /api/llm 不需要 path（是配置类接口）
        if endpoint in needs_path and not path:  # noqa: SIM108 - 保持可读性
            self._send_error_json(400, "缺少 path 参数")
            return

        if endpoint == "health":
            self._send_json(api.health(self.model_dir))
        elif endpoint == "cases":
            self._send_json(api.list_cases(_first(query, "dir", "data") or "data"))
        elif endpoint == "analyze":
            self._send_json(
                api.analyze_path(str(path), nperseg=_int_param(query, "nperseg", 1024), nfft=_first(query, "nfft") and _int_param(query, "nfft", 0))
            )
        elif endpoint == "spectrum":
            self._send_json(
                api.spectrum_path(
                    str(path),
                    nperseg=_int_param(query, "nperseg", 1024),
                    nfft=_first(query, "nfft") and _int_param(query, "nfft", 0),
                    points=_int_param(query, "points", 1200),
                )
            )
        elif endpoint == "waterfall":
            self._send_json(
                api.waterfall_path(
                    str(path),
                    nperseg=_int_param(query, "nperseg", 256),
                    max_frames=_int_param(query, "max_frames", 400),
                    max_bins=_int_param(query, "max_bins", 192),
                )
            )
        elif endpoint == "constellation":
            self._send_json(
                api.constellation_path(
                    str(path),
                    symbol_rate=_float_param(query, "symbol_rate"),
                    cfo=_float_param(query, "cfo"),
                    max_points=_int_param(query, "max_points", 3000),
                )
            )
        elif endpoint == "auth" or endpoint.startswith("auth/"):
            self._send_json(self.auth.status())
        elif endpoint == "agent" or endpoint.startswith("agent/"):
            self._send_json(api.agent_tools())
        elif endpoint == "llm" or endpoint.startswith("llm/"):
            # 契约 §17：GET /api/llm/config 与 GET /api/llm/models
            action = endpoint.split("/", 1)[1] if "/" in endpoint else ((_first(query, "action", "") or "config").strip())
            if action in ("config", ""):
                self._send_json(api.llm_config())
            elif action == "models":
                self._send_json(api.llm_models())
            else:
                self._send_error_json(404, "未知的 llm 接口：" + action)
        elif endpoint == "narrate":
            self._send_json(
                api.narrate_path(
                    str(path),
                    provider=str(_first(query, "provider", "auto") or "auto"),
                    max_tokens=_int_param(query, "max_tokens", 1200),
                )
            )
        elif endpoint == "ssca":
            self._send_json(
                api.ssca_path(
                    str(path),
                    nfft=_int_param(query, "nfft", 256),
                    max_points=_int_param(query, "max_points", 160),
                )
            )
        elif endpoint == "frame":
            self._send_json(
                api.frame_path(
                    str(path),
                    modulation=_first(query, "modulation"),
                    symbol_rate=_float_param(query, "symbol_rate"),
                    cfo=_float_param(query, "cfo"),
                    max_frames=_int_param(query, "max_frames", 32),
                    model_dir=self.model_dir,
                    blind=str(_first(query, "blind", "") or "") in ("1", "true", "yes"),
                )
            )
        elif endpoint == "fec":
            self._send_json(api.fec_report())
        elif endpoint == "demod":
            self._send_json(
                api.demod_path(
                    str(path),
                    symbol_rate=_float_param(query, "symbol_rate"),
                    cfo=_float_param(query, "cfo"),
                    modulation=_first(query, "modulation"),
                    max_symbols=_int_param(query, "max_symbols", 2000),
                    trace_points=_int_param(query, "trace_points", 400),
                    model_dir=self.model_dir,
                )
            )
        elif endpoint == "features":
            self._send_json(api.features_path(str(path), nperseg=_int_param(query, "nperseg", 1024)))
        elif endpoint == "classify":
            self._send_json(
                api.classify_path(
                    str(path),
                    model_dir=self.model_dir,
                    nperseg=_int_param(query, "nperseg", 1024),
                    threshold=_float_param(query, "threshold"),
                )
            )
        else:
            self._send_error_json(404, "未知接口：/api/" + endpoint)

    _LOG_MAX_BYTES = 2 * 1024 * 1024

    def do_POST(self) -> None:  # noqa: N802 - http.server 规定的名字
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        try:
            if not self._auth_ok(route):
                self._reject_unauthorized()
                return
            if route == "/api/agent/run":
                body = self._read_json_body()
                self._send_json(api.agent_run(
                    str(body.get("path") or ""),
                    goal=str(body.get("goal") or ""),
                    provider=str(body.get("provider") or "auto"),
                    max_rounds=_optional_int(body, "max_rounds"),
                    max_tool_calls=_optional_int(body, "max_tool_calls"),
                    timeout_s=_optional_float(body, "timeout_s"),
                ))
                return
            if route == "/api/llm/config":
                self._send_json(api.llm_config(update=self._read_json_body()))
                return
            if route == "/api/llm/test":
                self._send_json(api.llm_test(self._read_json_body()))
                return
            self._send_error_json(404, "未知接口（POST）：" + route)
        except ValueError as exc:
            self._send_error_json(400, str(exc))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001
            try:
                traceback.print_exc()
            except (BrokenPipeError, OSError, ValueError):
                pass
            self._send_error_json(500, "服务端异常：{}: {}".format(type(exc).__name__, exc))

    def _read_json_body(self) -> dict[str, Any]:
        """读取请求体；空体当 {}；非法 JSON 明确报 400（不要静默当空）."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw.strip():
            return {}
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise ValueError("请求体不是合法 JSON：" + str(exc)) from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        line = "[spxh] " + (fmt % args)
        path = self.log_path
        if path is not None:
            try:
                if Path(path).exists() and Path(path).stat().st_size > self._LOG_MAX_BYTES:
                    Path(path).write_text("", encoding="utf-8")          # 简单截断，避免无限增长
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                return
            except OSError:
                pass
        if not self.quiet:
            _safe_print(line)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8760,
    model_dir: str = "models",
    quiet: bool = True,
    log_path: Optional[str] = None,
    token: Optional[str] = None,
) -> SPXHHTTPServer:
    handler = type(
        "BoundSPXHHandler",
        (SPXHRequestHandler,),
        {
            "model_dir": model_dir,
            "quiet": bool(quiet),
            "log_path": Path(log_path) if log_path else None,
            "auth": AuthConfig(token=token, host=host),
        },
    )
    server = SPXHHTTPServer((host, int(port)), handler)
    server.daemon_threads = True
    return server


def serve(
    host: str = "127.0.0.1",
    port: int = 8760,
    model_dir: str = "models",
    quiet: bool = True,
    log_path: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    try:
        server = create_server(host=host, port=port, model_dir=model_dir, quiet=quiet,
                               log_path=log_path, token=token)
    except OSError as exc:
        _safe_print("[spxh] 无法监听 {}:{}（{}）。端口可能已被占用：先关掉旧进程再启动。".format(host, port, exc))
        raise SystemExit(2)
    auth = AuthConfig(token=token, host=host)
    _safe_print("SPXH Web 工作台已启动: http://{}:{}/".format(host, port))
    if auth.required:
        if token:
            _safe_print("访问令牌已启用。令牌：" + str(token))
            _safe_print("（前端「设置」页填入该令牌；或接口带 Authorization: Bearer <token>）")
        else:
            _safe_print("警告：监听地址不是回环且未设置令牌 —— 所有接口将拒绝访问，请用 --token 设置")
    if quiet:
        if log_path:
            _safe_print("逐请求日志: " + str(log_path))
    else:
        _safe_print("静态资源目录: " + str(STATIC_DIR))
        _safe_print("接口契约: docs/前端接口约定-v1.md；Ctrl+C 退出")
    try:
        # 外层兜底：即使 serve_forever 因为意外异常退出，也继续服务而不是让平台整体不可用
        while True:
            try:
                server.serve_forever(poll_interval=0.5)
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                _safe_print("[spxh] 服务循环异常（{}: {}），1 秒后继续".format(type(exc).__name__, exc))
                time.sleep(1.0)
    except KeyboardInterrupt:
        _safe_print("\n正在关闭 ...")
    finally:
        server.server_close()

"""M10 回归：Web 服务对"客户端中途断开"必须免疫（曾经因此整体退出）."""
from __future__ import annotations

import socket
import sys
import threading
import time

import pytest

from spxh.web.server import SPXHHTTPServer, create_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_create_server_uses_hardened_class():
    port = _free_port()
    server = create_server(host="127.0.0.1", port=port)
    try:
        assert isinstance(server, SPXHHTTPServer)
        assert server.daemon_threads is True
        assert server.allow_reuse_address is True
    finally:
        server.server_close()


def test_handle_error_swallows_client_disconnects(capsys):
    """客户端 reset 不是故障：只记一次短日志，不抛异常."""
    server = SPXHHTTPServer.__new__(SPXHHTTPServer)
    server.reset_count = 0
    server.error_count = 0
    for _ in range(5):
        try:
            raise ConnectionResetError(10054, "远程主机强迫关闭了一个现有的连接")
        except ConnectionResetError:
            server.handle_error(None, ("127.0.0.1", 12345))       # 不应抛出
    assert server.reset_count == 5
    assert server.error_count == 0


def test_handle_error_counts_real_failures(capsys):
    server = SPXHHTTPServer.__new__(SPXHHTTPServer)
    server.reset_count = 0
    server.error_count = 0
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        server.handle_error(None, ("127.0.0.1", 1))
    assert server.error_count == 1


def test_server_survives_aborted_requests():
    """真实回归：连发若干"发一半就断开"的请求，服务必须还活着（这正是先前把它打死的场景）."""
    port = _free_port()
    server = create_server(host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        time.sleep(0.3)
        for i in range(12):
            sock = socket.create_connection(("127.0.0.1", port), timeout=2.0)
            sock.sendall(b"GET /api/frame?path=data/demo/qpsk_snr+10.0_000.sigmf-meta HTTP/1.1\r\nHost: x\r\n")
            time.sleep(0.01 + (i % 5) * 0.01)
            sock.close()          # 粗暴断开：正是浏览器刷新时的行为
        time.sleep(0.6)
        # 服务仍然可用
        with socket.create_connection(("127.0.0.1", port), timeout=3.0) as probe:
            probe.sendall(b"GET /api/health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            data = probe.recv(4096)
        assert b"200" in data
    finally:
        server.shutdown()
        server.server_close()

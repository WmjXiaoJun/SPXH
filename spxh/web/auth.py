"""工作台访问令牌：本机单人默认不打扰，一旦暴露就必须带令牌.

默认策略（这是经过权衡的）：
- 只监听回环地址、且没有设置令牌 -> **不鉴权**（本机单人使用，不该被令牌烦）；
- 设置了令牌，或监听非回环地址 -> 写接口与配置接口必须带 Authorization: Bearer <token>；
- /api/health 与 /api/auth/status 永远开放（探活与"我需不需要令牌"）。
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Any, Mapping, Optional

__all__ = ["AuthConfig", "resolve_token", "LOOPBACK_HOSTS"]

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0".replace("0.0.0.0", "127.0.0.1"))
_ENV_TOKEN = "SPXH_WEB_TOKEN"
#: 不需要令牌的路径（探活 / 问"我要不要令牌"）
PUBLIC_PATHS = ("/api/health", "/api/auth/status", "/favicon.ico")


def is_loopback(host: str) -> bool:
    text = str(host or "").strip().lower()
    return text in ("127.0.0.1", "localhost", "::1") or text.startswith("127.")


def resolve_token(cli_token: Optional[str] = None) -> Optional[str]:
    """决定令牌：命令行 --token 优先，其次环境变量；"auto" 表示随机生成一个."""
    raw = cli_token if cli_token is not None else os.environ.get(_ENV_TOKEN)
    if not raw:
        return None
    text = str(raw).strip()
    if text.lower() in ("auto", "random", "generate"):
        return secrets.token_urlsafe(24)
    return text


@dataclass
class AuthConfig:
    token: Optional[str] = None
    host: str = "127.0.0.1"

    @property
    def loopback_only(self) -> bool:
        return is_loopback(self.host)

    @property
    def required(self) -> bool:
        """需要令牌的条件：设了令牌，或者监听的地址不是回环."""
        return bool(self.token) or not self.loopback_only

    def allows(self, headers: Mapping[str, Any]) -> bool:
        if not self.required:
            return True
        if not self.token:
            # 监听非回环但没设令牌：拒绝（不能把配置接口裸奔在局域网上）
            return False
        presented = ""
        auth = str(headers.get("Authorization") or "")
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()
        if not presented:
            presented = str(headers.get("X-SPXH-Token") or "").strip()
        return bool(presented) and secrets.compare_digest(presented, str(self.token))

    def is_public(self, path: str) -> bool:
        return any(str(path).startswith(item) for item in PUBLIC_PATHS)

    def status(self) -> dict[str, Any]:
        if not self.required:
            hint = "当前仅监听回环地址且未设置令牌：本机单人使用不需要令牌"
        elif self.token:
            hint = "已启用访问令牌：请在「设置」页填入令牌（请求会带 Authorization: Bearer …）"
        else:
            hint = "监听的地址不是回环地址但未设置令牌：所有接口已拒绝。请用 --token 设置令牌后再暴露"
        return {
            "auth_required": bool(self.required),
            "loopback_only": bool(self.loopback_only),
            "host": self.host,
            "hint": hint,
        }

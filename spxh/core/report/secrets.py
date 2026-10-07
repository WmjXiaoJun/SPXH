"""密钥落盘保护：Windows 上用 DPAPI 加密（绑定当前用户），其它平台退回明文并如实标注.

为什么要做：配置文件就在仓库目录里，明文密钥很容易被误提交、误同步、被别的进程读走。
DPAPI（CryptProtectData）把密文绑定到**当前 Windows 用户**——换了用户或换机器都解不开，
而且不需要我们自己管密钥。非 Windows 或调用失败时退回明文，并在配置视图里标出来，
绝不假装"已加密"。
"""
from __future__ import annotations

import base64
import ctypes
import sys
from ctypes import wintypes
from typing import Optional, Tuple

__all__ = ["protect", "unprotect", "available"]


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> _DataBlob:
    buffer = ctypes.create_string_buffer(data, len(data))
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))


def available() -> bool:
    return sys.platform.startswith("win")


def protect(plain: str) -> Tuple[Optional[str], bool]:
    """返回 (密文 base64 或 None, 是否真的加密了)."""
    text = str(plain or "")
    if not text:
        return None, False
    if not available():
        return None, False
    try:
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        blob_in = _blob(text.encode("utf-8"))
        blob_out = _DataBlob()
        ok = crypt32.CryptProtectData(
            ctypes.byref(blob_in), "spxh-llm-key", None, None, None, 0, ctypes.byref(blob_out)
        )
        if not ok:
            return None, False
        try:
            raw = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            kernel32.LocalFree(blob_out.pbData)
        return base64.b64encode(raw).decode("ascii"), True
    except Exception:  # noqa: BLE001 - 加密失败不该让保存失败，退回明文并如实标注
        return None, False


def unprotect(payload: str) -> Optional[str]:
    """解密；失败返回 None（调用方按"密钥不可用"处理，并提示重新填写）."""
    if not payload or not available():
        return None
    try:
        raw = base64.b64decode(str(payload))
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        blob_in = _blob(raw)
        blob_out = _DataBlob()
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
        )
        if not ok:
            return None
        try:
            data = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            kernel32.LocalFree(blob_out.pbData)
        return data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None

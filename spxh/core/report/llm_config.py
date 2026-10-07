"""LLM 配置：本地文件 > 环境变量 > 内置默认，接口一律脱敏回显.

为什么要有"配置文件"而不是只读环境变量：在 Windows 桌面上让使用者去改系统环境变量
再重启终端，体验很差；而把密钥写进代码或前端是绝对不能接受的。折中方案就是
**服务端本地文件 + 脱敏回显 + 只提交改动字段**：
- 密钥只落在 models/llm_config.json（不进仓库、不回显、不写日志）；
- 前端只能看到"是否已配置"和尾 4 位提示；
- 每次保存都做字段级校验，非法值直接 400 并给中文原因。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from spxh.core.report.secrets import protect, unprotect

__all__ = [
    "LLMConfig",
    "PRESETS",
    "chat_completion",
    "normalize_base_url",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "config_path",
    "load_config",
    "save_config",
    "masked_config",
    "list_models",
    "test_connection",
    "describe_sources",
    "ENV_BASE",
    "ENV_KEY",
    "ENV_MODEL",
]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
ENV_BASE = "SPXH_LLM_BASE_URL"
ENV_KEY = "SPXH_LLM_API_KEY"
ENV_MODEL = "SPXH_LLM_MODEL"

_CONFIG_NAME = "llm_config.json"

#: 常见服务商预设：前端可以一键填充，避免"Base URL 到底填什么"这种低级错误
PRESETS: list[dict[str, Any]] = [
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "notes": "官方兼容 OpenAI 协议；base_url 用 https://api.deepseek.com（写 .../v1 也可以）。"
                 "模型：deepseek-chat（对话）或 deepseek-reasoner（推理，不支持 JSON 模式，平台会自动降级重试）",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "notes": "官方地址；也可填任何 OpenAI 兼容网关",
    },
    {
        "id": "ollama",
        "label": "本地 Ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b",
        "notes": "先 ollama serve 并 ollama pull 模型；本地服务通常不需要 API Key（随便填一个即可绕过校验）",
    },
    {
        "id": "custom",
        "label": "自定义（OpenAI 兼容）",
        "base_url": "",
        "model": "",
        "notes": "只要兼容 /chat/completions 与 /models 即可",
    },
]

# 字段 -> (类型, 默认值, 环境变量名)
_FIELDS: dict[str, tuple[str, Any, Optional[str]]] = {
    "enabled": ("bool", True, None),
    "base_url": ("str", DEFAULT_BASE_URL, ENV_BASE),
    "api_key": ("str", "", ENV_KEY),
    "model": ("str", DEFAULT_MODEL, ENV_MODEL),
    "temperature": ("float", 0.2, None),
    "max_tokens": ("int", 1200, None),
    "timeout_s": ("float", 40.0, None),
}


@dataclass
class LLMConfig:
    enabled: bool = True
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    #: 密钥是否以密文落盘（Windows DPAPI）；False 表示明文（或未配置）—— 会如实暴露给界面
    api_key_protected: bool = False
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout_s: float = 40.0
    source: dict[str, str] = field(default_factory=dict)
    config_path: str = ""

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self.api_key and self.base_url and self.model)

    def to_call_kwargs(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
            "timeout": float(self.timeout_s),
        }


def config_path(root: Optional[str | Path] = None) -> Path:
    if root is None:
        # 与模型目录保持一致：默认放在仓库根的 models/ 下
        root = Path(__file__).resolve().parents[3] / "models"
    path = Path(root)
    return path / _CONFIG_NAME if path.suffix != ".json" else path


def _coerce(name: str, value: Any) -> Any:
    kind = _FIELDS[name][0]
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "是")
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    return "" if value is None else str(value)


def normalize_base_url(raw: str) -> str:
    """把用户粘进来的各种写法归一化成"根地址".

    实测最容易踩的坑：很多人直接把**完整接口地址**填进 Base URL
    （例如 https://api.deepseek.com/chat/completions 或 .../v1/chat/completions），
    我们再拼一次 /chat/completions 就变成 .../chat/completions/chat/completions -> 404。
    这里统一去掉末尾的 /chat/completions、/completions、/models 与多余斜杠。
    """
    text = str(raw or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/models"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip("/")
    return text


def _validate(name: str, value: Any) -> Any:
    value = _coerce(name, value)
    if name == "base_url":
        text = normalize_base_url(value)
        if not text.startswith(("http://", "https://")):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头")
        return text
    if name == "model":
        text = str(value).strip()
        if not text:
            raise ValueError("model 不能为空")
        return text
    if name == "temperature":
        number = float(value)
        if not 0.0 <= number <= 2.0:
            raise ValueError("temperature 必须在 0 ~ 2 之间")
        return round(number, 3)
    if name == "max_tokens":
        number = int(value)
        if not 1 <= number <= 32000:
            raise ValueError("max_tokens 必须在 1 ~ 32000 之间")
        return number
    if name == "timeout_s":
        number = float(value)
        if not 1.0 <= number <= 600.0:
            raise ValueError("timeout_s 必须在 1 ~ 600 秒之间")
        return round(number, 1)
    if name == "api_key":
        return str(value).strip()
    return value


#: 配置文件里允许出现的键（api_key_enc / api_key_protected 是密文存储用的）
_ALLOWED_KEYS = set(_FIELDS) | {"api_key_enc", "api_key_protected"}


def _read_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("配置文件无法解析（{}）：先修好或删掉 {}".format(exc, path)) from exc
    if not isinstance(payload, dict):
        raise ValueError("配置文件必须是 JSON 对象")
    return {k: v for k, v in payload.items() if k in _ALLOWED_KEYS}


def load_config(root: Optional[str | Path] = None, **overrides: Any) -> LLMConfig:
    """合并三方来源；overrides 优先级最高（用于 /api/llm/test 的临时覆盖）."""
    path = config_path(root)
    stored = _read_file(path)
    config = LLMConfig(config_path=str(path))
    source: dict[str, str] = {}
    # 一次性迁移：老配置里是明文密钥，而现在 DPAPI 可用 -> 静默加密回写（明文不再落盘）
    if stored.get("api_key") and not stored.get("api_key_enc"):
        encrypted, protected = protect(str(stored["api_key"]))
        if protected and encrypted and path.exists():
            try:
                migrated = dict(stored)
                migrated.pop("api_key", None)
                migrated["api_key_enc"] = encrypted
                migrated["api_key_protected"] = True
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(tmp, path)
                stored = migrated
            except OSError:
                pass
    # 密文密钥（Windows DPAPI）优先；解不开就当作"没配置"，并让来源显示出来
    if stored.get("api_key_enc"):
        plain = unprotect(str(stored["api_key_enc"]))
        if plain:
            config.api_key = plain
            config.api_key_protected = True
            source["api_key"] = "file(dpapi)"
        else:
            source["api_key"] = "file(dpapi-无法解密)"
    for name, (_kind, default, env_name) in _FIELDS.items():
        if name == "api_key" and source.get("api_key"):
            continue
        if name in overrides and overrides[name] is not None:
            try:
                setattr(config, name, _validate(name, overrides[name]))
                source[name] = "override"
                continue
            except ValueError:
                raise
        if name in stored:
            try:
                setattr(config, name, _validate(name, stored[name]))
                source[name] = "file"
                continue
            except ValueError:
                # 文件里的坏值不能把整个服务搞挂：退回下一级，并在 source 里标出来
                source[name] = "file-invalid"
        env_value = os.environ.get(env_name) if env_name else None
        if env_value:
            try:
                setattr(config, name, _validate(name, env_value))
                source.setdefault(name, "env")
                if source.get(name) != "file-invalid":
                    source[name] = "env"
                continue
            except ValueError:
                pass
        if source.get(name) in (None, "file-invalid"):
            setattr(config, name, _validate(name, default))
            source.setdefault(name, "default")
        if source.get(name) == "file-invalid":
            source[name] = "default(文件里的值非法，已忽略)"
    config.source = source
    return config


def save_config(changes: dict[str, Any], root: Optional[str | Path] = None) -> LLMConfig:
    """原子保存：只写合法字段，先落临时文件再替换，失败不破坏原文件."""
    path = config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = _read_file(path)
    if changes.get("clear_api_key"):
        stored.pop("api_key", None)
        stored.pop("api_key_enc", None)
        stored.pop("api_key_protected", None)
    for name, value in changes.items():
        if name in ("clear_api_key",):
            continue
        if name not in _FIELDS:
            raise ValueError("未知配置项：" + str(name))
        if value is None:
            continue
        if name == "api_key":
            text = str(value).strip()
            if text == "":
                # 空密钥视为"不改动"：避免前端没改密钥却把它覆盖掉
                continue
            # 优先加密落盘；DPAPI 不可用时退回明文，并把事实记进 api_key_protected
            encrypted, protected = protect(text)
            stored.pop("api_key", None)
            if protected and encrypted:
                stored["api_key_enc"] = encrypted
                stored["api_key_protected"] = True
            else:
                stored.pop("api_key_enc", None)
                stored["api_key"] = text
                stored["api_key_protected"] = False
            continue
        stored[name] = _validate(name, value)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return load_config(root)


def _display_path(path: str) -> str:
    """尽量回相对路径（仓库根为基准），失败就回原样 —— 界面更好读."""
    try:
        root = Path(__file__).resolve().parents[3]
        return str(Path(path).resolve().relative_to(root)).replace("\\", "/")
    except (ValueError, OSError):
        return str(path)


def _hint(api_key: str) -> Optional[str]:
    if not api_key:
        return None
    tail = api_key[-4:] if len(api_key) > 4 else api_key
    return "****" + tail


def masked_config(config: Optional[LLMConfig] = None, root: Optional[str | Path] = None) -> dict[str, Any]:
    """给接口用的脱敏视图：**永远不含密钥明文**."""
    cfg = config or load_config(root)
    return {
        "enabled": bool(cfg.enabled),
        "base_url": cfg.base_url,
        # 相对路径更好读（前端直接显示），但配置在别处时老老实实给绝对路径
        "config_path_display": _display_path(cfg.config_path),
        "model": cfg.model,
        "temperature": float(cfg.temperature),
        "max_tokens": int(cfg.max_tokens),
        "timeout_s": float(cfg.timeout_s),
        "api_key_set": bool(cfg.api_key),
        "api_key_hint": _hint(cfg.api_key),
        "api_key_protected": bool(cfg.api_key_protected),
        "source": dict(cfg.source),
        "config_path": _display_path(cfg.config_path),
        "config_path_abs": str(cfg.config_path),
        "env": {
            name: ("已设置" if os.environ.get(name) else "未设置")
            for name in (ENV_BASE, ENV_KEY, ENV_MODEL)
        },
        "ready": bool(cfg.ready),
        "presets": PRESETS,
    }


def describe_sources(config: Optional[LLMConfig] = None, root: Optional[str | Path] = None) -> dict[str, str]:
    return dict((config or load_config(root)).source)


def _request_json(url: str, api_key: str, timeout: float, payload: Optional[dict] = None) -> dict[str, Any]:
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:  # noqa: S310 - 地址由使用者配置
        body = response.read().decode("utf-8", errors="replace")
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = {"raw": body[:500]}
    return parsed if isinstance(parsed, dict) else {"data": parsed}


def _one_line(text: str, limit: int = 160) -> str:
    """把多行/带大括号的错误压成一行短理由（前端要一行显示，服务商返回的 JSON 太长）."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        try:
            detail = _one_line(exc.read().decode("utf-8", errors="replace"), 120)
        except Exception:  # noqa: BLE001
            pass
        mapping = {
            400: "请求被拒（400，常见原因：模型名不对 / 不支持 JSON 输出 / 参数越界）",
            401: "密钥无效或未授权（401）",
            402: "余额不足（402）",
            403: "无权限（403）",
            404: "地址或模型不存在（404）：检查 Base URL 是否多写了 /chat/completions",
            422: "参数不合法（422）",
            429: "被限流（429）",
            500: "服务端错误（500）",
            503: "服务不可用（503）",
        }
        base = mapping.get(int(exc.code), "HTTP {}".format(exc.code))
        return _one_line(base + ("：" + detail if detail else ""))
    if isinstance(exc, urllib.error.URLError):
        return _one_line("连接失败：{}".format(getattr(exc, "reason", exc)))
    return _one_line("{}: {}".format(type(exc).__name__, exc))


def chat_completion(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    json_mode: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    **overrides: Any,
) -> dict[str, Any]:
    """调用 /chat/completions，返回 {"message": {...}, "model": ..., "usage": ...}.

    支持 function calling（工具调用代理用）；json_mode 时带 response_format，
    若服务商以 400 拒绝（例如 DeepSeek 的 reasoner）会**自动去掉该参数重试一次**。
    """
    cfg = config or load_config(**overrides)
    if not cfg.api_key:
        raise RuntimeError("未配置 API Key（可在「设置」页或 models/llm_config.json 中填写）")
    base = cfg.base_url.rstrip("/")
    url = base + "/chat/completions"
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": float(temperature if temperature is not None else cfg.temperature),
        "max_tokens": int(max_tokens if max_tokens is not None else cfg.max_tokens),
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    def _post(body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + cfg.api_key},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(timeout if timeout is not None else cfg.timeout_s)) as response:  # noqa: S310
            parsed = json.loads(response.read().decode("utf-8", errors="replace"))
        return parsed if isinstance(parsed, dict) else {"raw": parsed}

    try:
        result = _post(payload)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = _one_line(exc.read().decode("utf-8", errors="replace"), 200)
        except Exception:  # noqa: BLE001
            pass
        if int(getattr(exc, "code", 0)) == 400 and json_mode and ("response_format" in detail or "json" in detail.lower()):
            payload.pop("response_format", None)
            result = _post(payload)
        else:
            raise RuntimeError(_friendly_error(exc)) from exc
    choices = result.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError("LLM 返回里没有 choices")
    return {
        "message": choices[0].get("message") or {},
        "finish_reason": choices[0].get("finish_reason"),
        "model": str(result.get("model") or cfg.model),
        "usage": result.get("usage") or {},
    }


def list_models(config: Optional[LLMConfig] = None, timeout: float = 15.0, **overrides: Any) -> dict[str, Any]:
    cfg = config or load_config(**overrides)
    url = cfg.base_url.rstrip("/") + "/models"
    try:
        payload = _request_json(url, cfg.api_key, min(float(timeout), float(cfg.timeout_s)))
    except Exception as exc:  # noqa: BLE001 - 拉不到模型列表不算致命
        return {"models": [], "error": _friendly_error(exc), "source": url}
    rows = payload.get("data") or payload.get("models") or []
    names: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            name = row.get("id") or row.get("name")
            if name:
                names.append(str(name))
        elif isinstance(row, str):
            names.append(row)
    return {"models": names, "error": None, "source": url}


def test_connection(config: Optional[LLMConfig] = None, **overrides: Any) -> dict[str, Any]:
    """真的发一次最小请求：能不能用、多久、模型回什么.

    失败**不抛异常**（返回 ok=false + 中文原因），因为"测试连接"本身就是要显示失败原因。
    """
    cfg = config or load_config(**overrides)
    checked = {
        "base_url": cfg.base_url,
        "model": cfg.model,
        "api_key_hint": _hint(cfg.api_key),
        # 把**实际请求的地址**回给使用者：Base URL 写错是最常见的失败原因
        "chat_url": cfg.base_url.rstrip("/") + "/chat/completions",
        "models_url": cfg.base_url.rstrip("/") + "/models",
    }
    if not cfg.api_key:
        return {"ok": False, "error": "未配置 API Key", "latency_ms": 0, "checked": checked,
                "hint": "DeepSeek/OpenAI 需要 sk- 开头的密钥；本地 Ollama 可随便填一个"}
    started = time.time()
    try:
        payload = _request_json(
            cfg.base_url.rstrip("/") + "/chat/completions",
            cfg.api_key,
            float(cfg.timeout_s),
            {
                "model": cfg.model,
                "messages": [{"role": "user", "content": "只回复两个字母：ok"}],
                "max_tokens": 16,
                "temperature": 0.0,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": _friendly_error(exc),
                "latency_ms": int((time.time() - started) * 1000), "checked": checked}
    latency = int((time.time() - started) * 1000)
    choices = payload.get("choices") or []
    reply = ""
    if choices and isinstance(choices[0], dict):
        reply = str(((choices[0].get("message") or {}).get("content")) or "")[:200]
    models = list_models(cfg, timeout=10.0)
    return {
        "ok": True,
        "latency_ms": latency,
        "model": str(payload.get("model") or cfg.model),
        "reply": reply or "（模型没有返回文本）",
        "models": models.get("models") or [],
        "models_error": models.get("error"),
        "checked": checked,
        "hint": _model_hint(cfg.model, models.get("models") or []),
    }


def _model_hint(model: str, available: list[str]) -> str:
    """模型名写错时给一句有用的提示（比"HTTP 400"强得多）."""
    if available and model not in available:
        return "该服务商可用模型：{}".format("、".join(available[:12]))
    if model.startswith("deepseek") and model not in ("deepseek-chat", "deepseek-reasoner"):
        return "DeepSeek 目前可用模型名是 deepseek-chat 或 deepseek-reasoner"
    return ""

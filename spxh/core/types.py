"""SPXH 核心数据类型.

约定
----
Signal      复数基带波形 + 采样率 + 元数据，是整条链路唯一的数据载体。
Header      32 位帧头的结构化表示（帧结构定义见 core/framing/format.py）。
FrameRecord 单帧在比特域/符号域/采样域的边界与内容。
GroundTruth 合成信号真值：比特、符号、帧、生成参数，供端到端量化验证。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

MODULATIONS: tuple[str, ...] = ("bpsk", "qpsk", "8psk", "16qam", "2fsk", "4fsk")
MOD_BITS_PER_SYMBOL: dict[str, int] = {"bpsk": 1, "qpsk": 2, "8psk": 3, "16qam": 4, "2fsk": 1, "4fsk": 2}
MOD_CODES: dict[str, int] = {m: i for i, m in enumerate(MODULATIONS)}
MOD_BY_CODE: dict[int, str] = {i: m for m, i in MOD_CODES.items()}
FSK_MODULATIONS: tuple[str, ...] = ("2fsk", "4fsk")
FRAME_MAGIC: int = 0xA5
TRUTH_FORMAT_VERSION: int = 1

__all__ = [
    "MODULATIONS",
    "MOD_BITS_PER_SYMBOL",
    "MOD_CODES",
    "MOD_BY_CODE",
    "FSK_MODULATIONS",
    "FRAME_MAGIC",
    "TRUTH_FORMAT_VERSION",
    "normalize_modulation",
    "is_fsk",
    "Signal",
    "Header",
    "FrameRecord",
    "GroundTruth",
    "jsonable",
]


def normalize_modulation(name: str) -> str:
    """把各种写法归一化到标准调制名."""
    raw = str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "bpsk": "bpsk", "2psk": "bpsk", "psk2": "bpsk",
        "qpsk": "qpsk", "4psk": "qpsk", "psk4": "qpsk",
        "8psk": "8psk", "psk8": "8psk",
        "16qam": "16qam", "qam16": "16qam",
        "2fsk": "2fsk", "fsk2": "2fsk", "msk": "2fsk",
        "4fsk": "4fsk", "fsk4": "4fsk",
    }
    if raw not in aliases:
        raise ValueError(
            "unsupported modulation: " + str(name) + " (supported: " + ", ".join(MODULATIONS) + ")"
        )
    return aliases[raw]


def is_fsk(modulation: str) -> bool:
    return normalize_modulation(modulation) in FSK_MODULATIONS


def jsonable(value: Any) -> Any:
    """把 numpy 标量/数组递归转换为可 JSON 序列化的对象."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


@dataclass
class Signal:
    """复数基带信号.

    samples      shape (N,) 的复数数组
    sample_rate  采样率 Hz
    center_freq  中心频率 Hz（可选，通常来自 SigMF 的 captures 或 WAV 元数据）
    meta         任意元数据（datatype、来源文件、是否实信号等）
    """

    samples: np.ndarray
    sample_rate: float
    center_freq: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        self.samples = np.ascontiguousarray(arr, dtype=np.complex128)
        self.sample_rate = float(self.sample_rate)
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive finite number")
        if self.samples.size == 0:
            raise ValueError("samples must not be empty")
        self.meta = dict(self.meta or {})
        if self.center_freq is not None:
            self.center_freq = float(self.center_freq)

    @property
    def num_samples(self) -> int:
        return int(self.samples.size)

    @property
    def duration(self) -> float:
        return float(self.samples.size / self.sample_rate)

    @property
    def power(self) -> float:
        """平均功率 E[|x|^2]."""
        return float(np.mean(np.abs(self.samples) ** 2))

    def copy(self) -> "Signal":
        return Signal(self.samples.copy(), self.sample_rate, self.center_freq, dict(self.meta))

    def summary(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "sample_rate": self.sample_rate,
            "duration_s": self.duration,
            "power_dbfs": 10.0 * float(np.log10(max(self.power, 1e-30))),
            "center_freq": self.center_freq,
            "meta": jsonable(self.meta),
        }


@dataclass
class Header:
    """32 位帧头（4 字节，MSB-first）.

    byte0     MAGIC = 0xA5（包标识）
    byte1     bit7-4 调制码 | bit3-2 编码码 | bit1 交织标志 | bit0 保留
    byte2-3   载荷长度（uint16 大端，单位字节）
    """

    modulation: str = "qpsk"
    fec: str = "none"
    interleave: bool = False
    payload_len: int = 0
    reserved: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "magic": FRAME_MAGIC,
            "modulation": self.modulation,
            "fec": self.fec,
            "interleave": bool(self.interleave),
            "payload_len": int(self.payload_len),
            "reserved": int(self.reserved),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Header":
        return cls(
            modulation=str(data.get("modulation", "qpsk")),
            fec=str(data.get("fec", "none")),
            interleave=bool(data.get("interleave", False)),
            payload_len=int(data.get("payload_len", 0)),
            reserved=int(data.get("reserved", 0)),
        )


@dataclass
class FrameRecord:
    """单帧真值：内容 + 比特/符号/采样边界."""

    index: int
    header: Header
    payload: bytes
    crc: int
    start_bit: int
    num_bits: int
    start_symbol: int
    num_symbols: int
    start_sample: int
    num_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "header": self.header.to_dict(),
            "payload_hex": bytes(self.payload).hex(),
            "payload_len": len(self.payload),
            "crc": int(self.crc),
            "start_bit": int(self.start_bit),
            "num_bits": int(self.num_bits),
            "start_symbol": int(self.start_symbol),
            "num_symbols": int(self.num_symbols),
            "start_sample": int(self.start_sample),
            "num_samples": int(self.num_samples),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameRecord":
        return cls(
            index=int(data["index"]),
            header=Header.from_dict(data.get("header", {})),
            payload=bytes.fromhex(data.get("payload_hex", "")),
            crc=int(data.get("crc", 0)),
            start_bit=int(data.get("start_bit", 0)),
            num_bits=int(data.get("num_bits", 0)),
            start_symbol=int(data.get("start_symbol", 0)),
            num_symbols=int(data.get("num_symbols", 0)),
            start_sample=int(data.get("start_sample", 0)),
            num_samples=int(data.get("num_samples", 0)),
        )


@dataclass
class GroundTruth:
    """合成信号真值.

    标量与结构进 JSON（<prefix>.truth.json），大数组进 npz（<prefix>.truth.npz）。
    """

    modulation: str
    symbol_rate: float
    sample_rate: float
    sps: int
    snr_db: float
    cfo_hz: float
    phase_noise_rms_rad: float
    rolloff: float
    span_symbols: int
    fsk_mod_index: float
    seed: int
    num_symbols: int
    num_bits: int
    pad_bits: int
    filter_delay_samples: float = 0.0
    noise_variance: float = 0.0
    frames: list[FrameRecord] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    bits: Optional[np.ndarray] = None
    symbol_indices: Optional[np.ndarray] = None
    symbols: Optional[np.ndarray] = None
    tone_hz: Optional[np.ndarray] = None
    clean: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": TRUTH_FORMAT_VERSION,
            "modulation": self.modulation,
            "symbol_rate": self.symbol_rate,
            "sample_rate": self.sample_rate,
            "sps": self.sps,
            "snr_db": self.snr_db,
            "cfo_hz": self.cfo_hz,
            "phase_noise_rms_rad": self.phase_noise_rms_rad,
            "rolloff": self.rolloff,
            "span_symbols": int(self.span_symbols),
            "fsk_mod_index": self.fsk_mod_index,
            "seed": int(self.seed),
            "num_symbols": int(self.num_symbols),
            "num_bits": int(self.num_bits),
            "pad_bits": int(self.pad_bits),
            "filter_delay_samples": float(self.filter_delay_samples),
            "noise_variance": float(self.noise_variance),
            "frames": [f.to_dict() for f in self.frames],
            "extras": jsonable(self.extras),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], arrays: Optional[dict[str, np.ndarray]] = None) -> "GroundTruth":
        arrays = arrays or {}
        gt = cls(
            modulation=str(data["modulation"]),
            symbol_rate=float(data["symbol_rate"]),
            sample_rate=float(data["sample_rate"]),
            sps=int(data["sps"]),
            snr_db=float(data["snr_db"]),
            cfo_hz=float(data["cfo_hz"]),
            phase_noise_rms_rad=float(data.get("phase_noise_rms_rad", 0.0)),
            rolloff=float(data.get("rolloff", 0.0)),
            span_symbols=int(data.get("span_symbols", 16)),
            fsk_mod_index=float(data.get("fsk_mod_index", 0.0)),
            seed=int(data.get("seed", 0)),
            num_symbols=int(data.get("num_symbols", 0)),
            num_bits=int(data.get("num_bits", 0)),
            pad_bits=int(data.get("pad_bits", 0)),
            filter_delay_samples=float(data.get("filter_delay_samples", 0.0)),
            noise_variance=float(data.get("noise_variance", 0.0)),
            frames=[FrameRecord.from_dict(f) for f in data.get("frames", [])],
            extras=dict(data.get("extras", {})),
        )
        gt.bits = arrays.get("bits")
        gt.symbol_indices = arrays.get("symbol_indices")
        gt.symbols = arrays.get("symbols")
        gt.tone_hz = arrays.get("tone_hz")
        gt.clean = arrays.get("clean")
        return gt

    def save(self, prefix) -> dict[str, str]:
        """写出 <prefix>.truth.json 与 <prefix>.truth.npz."""
        json_path = Path(str(prefix) + ".truth.json")
        npz_path = Path(str(prefix) + ".truth.npz")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        if self.bits is not None:
            arrays["bits"] = np.asarray(self.bits, dtype=np.uint8)
        if self.symbol_indices is not None:
            arrays["symbol_indices"] = np.asarray(self.symbol_indices, dtype=np.int32)
        if self.symbols is not None:
            arrays["symbols"] = np.asarray(self.symbols, dtype=np.complex128)
        if self.tone_hz is not None:
            arrays["tone_hz"] = np.asarray(self.tone_hz, dtype=np.float64)
        if self.clean is not None:
            arrays["clean"] = np.asarray(self.clean, dtype=np.complex128)
        if arrays:
            np.savez_compressed(npz_path, **arrays)
        json_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return {"truth_json": str(json_path), "truth_npz": str(npz_path) if arrays else ""}

    @classmethod
    def load(cls, prefix) -> "GroundTruth":
        json_path = Path(str(prefix) + ".truth.json")
        if not json_path.exists():
            raise FileNotFoundError("ground truth json not found: " + str(json_path))
        data = json.loads(json_path.read_text(encoding="utf-8"))
        npz_path = Path(str(prefix) + ".truth.npz")
        arrays: dict[str, np.ndarray] = {}
        if npz_path.exists():
            with np.load(npz_path) as loaded:
                for key in loaded.files:
                    arrays[key] = loaded[key]
        return cls.from_dict(data, arrays)

    def summary(self) -> dict[str, Any]:
        return {
            "modulation": self.modulation,
            "symbol_rate": self.symbol_rate,
            "sample_rate": self.sample_rate,
            "sps": self.sps,
            "snr_db": self.snr_db,
            "cfo_hz": self.cfo_hz,
            "num_symbols": self.num_symbols,
            "num_bits": self.num_bits,
            "num_frames": len(self.frames),
            "payload_bytes": sum(len(f.payload) for f in self.frames),
            "has_arrays": {
                "bits": self.bits is not None,
                "symbol_indices": self.symbol_indices is not None,
                "symbols": self.symbols is not None,
                "tone_hz": self.tone_hz is not None,
                "clean": self.clean is not None,
            },
        }

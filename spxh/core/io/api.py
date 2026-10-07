"""统一的信号读写入口：按后缀分派到具体格式."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from spxh.core.io.bundle import bundle_path, read_bundle, write_bundle
from spxh.core.io.raw import infer_datatype, read_raw, write_raw
from spxh.core.io.sigmf import read_sigmf, sigmf_paths, write_sigmf
from spxh.core.io.wavio import read_wav, write_wav
from spxh.core.types import GroundTruth, Signal, jsonable

__all__ = ["load_signal", "save_signal", "describe_file", "DEFAULT_FORMATS"]

DEFAULT_FORMATS: tuple[str, ...] = ("sigmf", "truth")


def load_signal(path, **kwargs) -> Signal:
    """按扩展名加载信号.

    .sigmf-meta -> SigMF；.spxh.npz -> bundle；.wav -> WAV；其余按裸 I/Q 后缀推断。
    """
    target = Path(str(path))
    if not target.exists():
        raise FileNotFoundError("file not found: " + str(target))
    name = target.name.lower()
    if name.endswith(".sigmf-meta"):
        return read_sigmf(target, **kwargs)
    if name.endswith(".spxh.npz") or (target.suffix.lower() == ".npz" and "spxh" in name):
        signal, _ = read_bundle(target)
        return signal
    if target.suffix.lower() == ".wav":
        return read_wav(target, **kwargs)
    datatype = infer_datatype(target)
    if datatype is None:
        raise ValueError("unsupported signal file: " + str(target))
    return read_raw(target, datatype=datatype, **kwargs)


def save_signal(
    prefix,
    signal: Signal,
    truth: Optional[GroundTruth] = None,
    datatype: str = "cf32",
    formats: Iterable[str] = DEFAULT_FORMATS,
    description: str = "",
    annotations: Optional[list[dict[str, Any]]] = None,
) -> dict[str, str]:
    """保存信号与真值.

    prefix   路径前缀（不含扩展名）
    formats  任意组合：sigmf / truth / bundle / raw
    datatype 裸数据格式（sigmf 用 cf32_le/ci16_le 等，raw 用 cf32/ci16 等）
    """
    formats = tuple(str(f).lower() for f in formats)
    base = Path(str(prefix))
    base.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    if truth is not None and "truth" in formats:
        written.update(truth.save(base))

    if "sigmf" in formats:
        sigmf_datatype = datatype if datatype.endswith("_le") else datatype + "_le"
        paths = write_sigmf(
            base,
            signal,
            datatype=sigmf_datatype,
            annotations=annotations,
            description=description,
            truth=truth,
        )
        written["sigmf_meta"] = paths["meta"]
        written["sigmf_data"] = paths["data"]

    if "raw" in formats:
        written["raw"] = write_raw(str(base) + "." + datatype, signal, datatype=datatype)

    if "bundle" in formats:
        written["bundle"] = write_bundle(base, signal, truth)

    return written


def describe_file(path) -> dict[str, Any]:
    """加载并汇总一个信号文件（含同名真值，若存在）."""
    target = Path(str(path))
    info: dict[str, Any] = {"path": str(target), "exists": target.exists()}
    if not target.exists():
        return info

    if target.name.lower().endswith(".sigmf-meta"):
        stem = target.name[: -len(".sigmf-meta")]
        prefix = target.with_name(stem)
    else:
        prefix = target.with_suffix("") if target.suffix else target

    try:
        signal = load_signal(target)
        info["signal"] = signal.summary()
        info["ok"] = True
    except Exception as exc:  # noqa: BLE001 - 汇总接口需要把失败原因带回
        info["ok"] = False
        info["error"] = str(exc)
        return info

    truth_json = Path(str(prefix) + ".truth.json")
    if truth_json.exists():
        try:
            truth = GroundTruth.load(prefix)
            info["ground_truth"] = truth.summary()
            info["frames"] = [f.to_dict() for f in truth.frames]
        except Exception as exc:  # noqa: BLE001
            info["ground_truth_error"] = str(exc)
    return jsonable(info)

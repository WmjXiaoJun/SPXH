"""参数网格数据集生成（M0 的批量入口）."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from spxh.core.io.api import save_signal
from spxh.core.synth.generator import SynthConfig, synthesize
from spxh.core.types import MODULATIONS, jsonable, normalize_modulation

__all__ = ["generate_case", "generate_grid", "case_name"]


def case_name(modulation: str, snr_db: float, index: int = 0) -> str:
    return "{:s}_snr{:+.1f}_{:03d}".format(normalize_modulation(modulation), float(snr_db), int(index))


def generate_case(
    out_prefix,
    cfg: SynthConfig,
    formats: Sequence[str] = ("sigmf", "truth"),
    datatype: str = "cf32",
) -> dict[str, Any]:
    """生成单个案例并落盘，返回 manifest 条目."""
    signal, truth = synthesize(cfg)
    written = save_signal(
        out_prefix,
        signal,
        truth=truth,
        datatype=datatype,
        formats=formats,
        description="SPXH synthetic {}".format(truth.modulation),
    )
    return {
        "name": Path(str(out_prefix)).name,
        "prefix": str(out_prefix),
        "files": written,
        "config": cfg.to_dict(),
        "truth": truth.summary(),
        "num_samples": signal.num_samples,
        "duration_s": signal.duration,
    }


def generate_grid(
    out_dir,
    modulations: Iterable[str] = MODULATIONS,
    snrs: Iterable[float] = (-5.0, 0.0, 5.0, 10.0, 20.0, 30.0),
    base_config: Optional[dict[str, Any]] = None,
    per_combo: int = 1,
    seed0: int = 0,
    formats: Sequence[str] = ("sigmf", "truth"),
    datatype: str = "cf32",
    manifest_name: str = "manifest.json",
) -> dict[str, Any]:
    """按 调制 x SNR 网格批量生成，写出 manifest.json."""
    out = Path(str(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    base = dict(base_config or {})
    cases: list[dict[str, Any]] = []
    counter = 0
    for mod in modulations:
        for snr in snrs:
            for rep in range(int(per_combo)):
                cfg = SynthConfig(**{**base, "modulation": normalize_modulation(mod), "snr_db": float(snr), "seed": int(seed0) + counter})
                prefix = out / case_name(mod, snr, rep)
                entry = generate_case(prefix, cfg, formats=formats, datatype=datatype)
                cases.append(entry)
                counter += 1
    manifest = {
        "generator": "spxh-m0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out),
        "base_config": jsonable(base),
        "num_cases": len(cases),
        "cases": cases,
    }
    (out / manifest_name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest

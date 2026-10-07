"""SPXH 命令行：generate / dataset / info / mods."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from spxh import __version__
from spxh.core.classify import classify_signal
from spxh.core.classify.model import DEFAULT_MODEL_DIR
from spxh.core.dsp import stft_waterfall, welch_psd
from spxh.core.estimate import analyze_signal, compare_with_truth
from spxh.core.features import FEATURE_GROUPS, FeatureConfig, extract_features
from spxh.core.io.api import describe_file, load_signal
from spxh.core.synth.dataset import generate_grid
from spxh.core.synth.generator import SynthConfig, synthesize
from spxh.core.types import MODULATIONS, GroundTruth, normalize_modulation

__all__ = ["build_parser", "main"]


def _add_signal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mod", default="qpsk", help="调制方式: " + ", ".join(MODULATIONS) + " (默认 qpsk)")
    parser.add_argument("--symbol-rate", type=float, default=1000.0, help="符号速率 Hz (默认 1000)")
    parser.add_argument("--sps", type=int, default=8, help="每符号采样点数 (默认 8)")
    parser.add_argument("--sample-rate", type=float, default=None, help="采样率 Hz (默认 sps*symbol_rate)")
    parser.add_argument("--snr", type=float, default=20.0, help="Es/N0 dB (默认 20)")
    parser.add_argument("--cfo", type=float, default=0.0, help="载波频偏 Hz (默认 0)")
    parser.add_argument("--cfo-ratio", type=float, default=None, help="载波频偏按符号速率的比例设置（优先于 --cfo）")
    parser.add_argument("--phase-noise", type=float, default=0.0, help="相位噪声稳态 RMS (rad)")
    parser.add_argument("--phase-noise-bw", type=float, default=None, help="相位噪声 3dB 带宽 Hz")
    parser.add_argument("--rolloff", type=float, default=0.35, help="RRC 滚降系数 (默认 0.35)")
    parser.add_argument("--span", type=int, default=16, help="RRC 跨度（符号数）")
    parser.add_argument("--fsk-index", type=float, default=None, help="FSK 调制指数 (默认 2FSK=1.0, 4FSK=0.5)")
    parser.add_argument("--payload-len", type=int, default=64, help="每帧载荷字节数 (默认 64)")
    parser.add_argument("--frames", type=int, default=3, help="帧数 (默认 3)")
    parser.add_argument("--fec", default="none", help="信道编码: none/conv/rs/ccsds/ldpc (默认 none)")
    parser.add_argument("--interleave", action="store_true", help="对已编码载荷做交织")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--iq-gain-db", type=float, default=0.0, help="IQ 幅度不平衡 dB")
    parser.add_argument("--iq-phase-deg", type=float, default=0.0, help="IQ 相位不平衡 度")
    parser.add_argument("--dc-i", type=float, default=0.0, help="DC 偏置 I")
    parser.add_argument("--dc-q", type=float, default=0.0, help="DC 偏置 Q")
    parser.add_argument("--no-clean", action="store_true", help="真值中不保存无噪参考波形")


def _config_from_args(args: argparse.Namespace) -> SynthConfig:
    cfo = float(args.cfo)
    if args.cfo_ratio is not None:
        cfo = float(args.cfo_ratio) * float(args.symbol_rate)
    return SynthConfig(
        modulation=args.mod,
        symbol_rate=float(args.symbol_rate),
        sps=int(args.sps),
        sample_rate=args.sample_rate,
        snr_db=float(args.snr),
        cfo_hz=cfo,
        phase_noise_rms_rad=float(args.phase_noise),
        phase_noise_bw_hz=args.phase_noise_bw,
        rolloff=float(args.rolloff),
        span_symbols=int(args.span),
        fsk_mod_index=args.fsk_index,
        payload_len=int(args.payload_len),
        num_frames=int(args.frames),
        fec=str(getattr(args, "fec", "none") or "none"),
        interleave=bool(getattr(args, "interleave", False)),
        iq_gain_db=float(args.iq_gain_db),
        iq_phase_deg=float(args.iq_phase_deg),
        dc_offset=complex(float(args.dc_i), float(args.dc_q)),
        seed=int(args.seed),
        keep_clean=not bool(args.no_clean),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spxh", description="SPXH · 智能射频信号分析平台")
    parser.add_argument("--version", action="version", version="spxh " + __version__)
    sub = parser.add_subparsers(dest="command")

    p_gen = sub.add_parser("generate", help="生成单个合成信号（含真值）")
    _add_signal_args(p_gen)
    p_gen.add_argument("--out", required=True, help="输出前缀（不含扩展名）")
    p_gen.add_argument("--datatype", default="cf32", help="裸数据格式: cf32/cf64/ci16/ci8/rf32 (默认 cf32)")
    p_gen.add_argument("--formats", default="sigmf,truth", help="输出格式组合: sigmf,truth,bundle,raw")
    p_gen.add_argument("--json", action="store_true", help="以 JSON 输出结果")

    p_ds = sub.add_parser("dataset", help="按 调制 x SNR 网格批量生成数据集")
    _add_signal_args(p_ds)
    p_ds.add_argument("--out", required=True, help="输出目录")
    p_ds.add_argument("--mods", default="all", help="逗号分隔的调制列表，或 all")
    p_ds.add_argument("--snrs", default="-5,0,5,10,20,30", help="逗号分隔的 SNR 列表 (dB)")
    p_ds.add_argument("--per-combo", type=int, default=1, help="每个组合生成多少个案例")
    p_ds.add_argument("--datatype", default="cf32", help="裸数据格式")
    p_ds.add_argument("--formats", default="sigmf,truth", help="输出格式组合")
    p_ds.add_argument("--json", action="store_true", help="以 JSON 输出结果")

    p_info = sub.add_parser("info", help="查看信号文件信息（含同名真值）")
    p_info.add_argument("path", help="信号文件路径")
    p_info.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_spec = sub.add_parser("spectrum", help="功率谱与时频瀑布分析")
    p_spec.add_argument("path", help="信号文件路径")
    p_spec.add_argument("--nperseg", type=int, default=1024, help="Welch 段长 (默认 1024)")
    p_spec.add_argument("--nfft", type=int, default=None, help="FFT 点数（默认 8*nperseg，用于插值）")
    p_spec.add_argument("--out", default=None, help="输出前缀（写 .psd.csv / .waterfall.npz）")
    p_spec.add_argument("--waterfall", action="store_true", help="同时导出 STFT 瀑布数据")
    p_spec.add_argument("--plot", default=None, help="输出频谱 PNG 路径")
    p_spec.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_an = sub.add_parser("analyze", help="M1 参数估计：载频 / 符号速率 / 盲 SNR")
    p_an.add_argument("path", help="信号文件路径")
    p_an.add_argument("--nperseg", type=int, default=1024, help="Welch 段长 (默认 1024)")
    p_an.add_argument("--nfft", type=int, default=None, help="FFT 点数（默认 8*nperseg）")
    p_an.add_argument("--fmin", type=float, default=None, help="符号速率搜索下界 Hz")
    p_an.add_argument("--fmax", type=float, default=None, help="符号速率搜索上界 Hz")
    p_an.add_argument("--plot", default=None, help="频谱 PNG 输出路径")
    p_an.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_feat = sub.add_parser("features", help="M2：提取五域 23 维特征")
    p_feat.add_argument("path", help="信号文件路径")
    p_feat.add_argument("--nperseg", type=int, default=1024, help="Welch 段长")
    p_feat.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_cls = sub.add_parser("classify", help="M2：调制识别（含门控与物理回退）")
    p_cls.add_argument("path", help="信号文件路径")
    p_cls.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录 (默认 models)")
    p_cls.add_argument("--threshold", type=float, default=None, help="置信度门限（默认用模型自带）")
    p_cls.add_argument("--nperseg", type=int, default=1024, help="Welch 段长")
    p_cls.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_frame = sub.add_parser("frame", help="M5：帧同步（Barker-13）+ 载荷提取 + CRC 校验")
    p_frame.add_argument("path", help="信号文件路径")
    p_frame.add_argument("--mod", default=None, help="调制类型（默认用分类器结果）")
    p_frame.add_argument("--max-frames", type=int, default=32, help="最多解析多少帧")
    p_frame.add_argument("--blind", action="store_true", help="盲编码识别：帧头 fec 字段不可信时对编码假设做检验")
    p_frame.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录")
    p_frame.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_narrate = sub.add_parser("narrate", help="M9：把整条链路的证据写成分析报告（LLM 可选，默认离线确定性渲染）")
    p_narrate.add_argument("path", help="信号文件路径")
    p_narrate.add_argument("--provider", default="auto", choices=("auto", "offline", "llm"),
                           help="auto=有 key 才走 LLM；offline=强制离线；llm=强制调用（失败会降级）")
    p_narrate.add_argument("--max-tokens", type=int, default=1200, help="LLM 输出上限")
    p_narrate.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_narrate.add_argument("--prompt", action="store_true", help="只打印提示词（便于审计）")

    p_fec = sub.add_parser("fec", help="M4：查看信道编码目录与译码报告（数据来自 scripts/check_m4.py）")
    p_fec.add_argument("--report", default="models/check_m4_report.json", help="报告路径")
    p_fec.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_demod = sub.add_parser("demod", help="M3：定时同步 + 载波跟踪 + 解调 + 软判决 LLR")
    p_demod.add_argument("path", help="信号文件路径")
    p_demod.add_argument("--mod", default=None, help="调制类型（默认用分类器结果）")
    p_demod.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录")
    p_demod.add_argument("--rolloff", type=float, default=0.35, help="匹配滤波滚降假设")
    p_demod.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_serve = sub.add_parser("serve", help="启动本地 Web 工作台")
    p_serve.add_argument("--host", default="127.0.0.1", help="监听地址 (默认 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8760, help="端口 (默认 8760)")
    p_serve.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录")
    p_serve.add_argument("--log", default="models/web.log", help="逐请求日志文件（默认 models/web.log；后台运行时不要刷 stdout）")
    p_serve.add_argument("--token", default=None,
                         help="访问令牌：具体值或 auto（自动生成并打印）。设置后所有接口需要 Authorization: Bearer <token>")
    p_serve.add_argument("--no-auth", action="store_true", help="强制关闭鉴权（仅在确定只服务本机时使用）")
    p_serve.add_argument("--verbose", action="store_true", help="逐请求日志也打到终端（前台调试用）")

    p_agent = sub.add_parser("agent", help="M11：工具调用代理（有界循环，无 LLM 时走离线确定性计划）")
    p_agent.add_argument("path", nargs="?", default="", help="信号文件路径（--tools 时可省略）")
    p_agent.add_argument("--goal", default="", help="分析目标（自然语言，可空）")
    p_agent.add_argument("--provider", default="auto", choices=("auto", "offline", "llm"))
    p_agent.add_argument("--max-rounds", type=int, default=None, help="最大轮数（默认 6，上限 12）")
    p_agent.add_argument("--max-tool-calls", type=int, default=None, help="最大工具调用次数（默认 12，上限 24）")
    p_agent.add_argument("--timeout", type=float, default=None, help="墙钟超时秒（默认 120，上限 300）")
    p_agent.add_argument("--tools", action="store_true", help="只列出可用工具与预算")
    p_agent.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_llm = sub.add_parser("llm", help="M10：查看/设置 LLM（地址、密钥、模型、采样参数）与连通性测试")
    p_llm.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="设置配置项，可重复：--set base_url=http://... --set api_key=sk-... --set model=qwen2.5:7b")
    p_llm.add_argument("--clear-key", action="store_true", help="删除已保存的 API Key")
    p_llm.add_argument("--test", action="store_true", help="测试连通性（会真的发一次最小请求）")
    p_llm.add_argument("--models", action="store_true", help="拉取该服务商的模型列表")
    p_llm.add_argument("--json", action="store_true", help="以 JSON 输出")

    sub.add_parser("mods", help="列出支持的调制方式")
    return parser


def _print_human(payload: dict[str, Any], title: str = "") -> None:
    if title:
        print(title)
    for key, value in payload.items():
        if isinstance(value, dict):
            print("  " + str(key) + ":")
            for sub_key, sub_value in value.items():
                print("    " + str(sub_key) + ": " + str(sub_value))
        elif isinstance(value, list):
            print("  " + str(key) + ": " + str(len(value)) + " 项")
        else:
            print("  " + str(key) + ": " + str(value))


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    signal, truth = synthesize(cfg)
    from spxh.core.io.api import save_signal

    formats = tuple(f.strip() for f in str(args.formats).split(",") if f.strip())
    written = save_signal(
        args.out,
        signal,
        truth=truth,
        datatype=args.datatype,
        formats=formats,
        description="SPXH synthetic " + truth.modulation,
    )
    payload = {
        "out_prefix": str(args.out),
        "written": written,
        "signal": signal.summary(),
        "ground_truth": truth.summary(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(payload, "生成完成:")
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    mods = MODULATIONS if str(args.mods).strip().lower() == "all" else [m.strip() for m in str(args.mods).split(",") if m.strip()]
    snrs = [float(s) for s in str(args.snrs).split(",") if s.strip()]
    base = _config_from_args(args)
    base_dict = {
        "symbol_rate": base.symbol_rate,
        "sps": base.sps,
        "sample_rate": base.sample_rate,
        "cfo_hz": base.cfo_hz,
        "phase_noise_rms_rad": base.phase_noise_rms_rad,
        "phase_noise_bw_hz": base.phase_noise_bw_hz,
        "rolloff": base.rolloff,
        "span_symbols": base.span_symbols,
        "fsk_mod_index": base.fsk_mod_index,
        "payload_len": base.payload_len,
        "num_frames": base.num_frames,
        "iq_gain_db": base.iq_gain_db,
        "iq_phase_deg": base.iq_phase_deg,
        "dc_offset": base.dc_offset,
        "keep_clean": base.keep_clean,
    }
    formats = tuple(f.strip() for f in str(args.formats).split(",") if f.strip())
    manifest = generate_grid(
        args.out,
        modulations=mods,
        snrs=snrs,
        base_config=base_dict,
        per_combo=int(args.per_combo),
        seed0=int(args.seed),
        formats=formats,
        datatype=args.datatype,
    )
    if args.json:
        print(json.dumps({k: v for k, v in manifest.items() if k != "cases"}, ensure_ascii=False, indent=2))
    else:
        print("数据集生成完成: " + str(manifest["num_cases"]) + " 个案例 -> " + str(args.out))
        print("  manifest: " + str(Path(args.out) / "manifest.json"))
        for case in manifest["cases"][:10]:
            print("  - " + str(case["name"]) + " (samples=" + str(case["num_samples"]) + ")")
        if manifest["num_cases"] > 10:
            print("  ... 其余 " + str(manifest["num_cases"] - 10) + " 个见 manifest.json")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    info = describe_file(args.path)
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        _print_human(info, "文件信息:")
    return 0 if info.get("ok") else 1


def _truth_for(path) -> Optional[GroundTruth]:
    """若存在同名真值 sidecar，则加载它（用于估计误差对账）."""
    target = Path(str(path))
    name = target.name.lower()
    if name.endswith(".sigmf-meta"):
        prefix = target.with_name(target.name[: -len(".sigmf-meta")])
    elif name.endswith(".spxh.npz"):
        prefix = target.with_name(target.name[: -len(".spxh.npz")])
    else:
        prefix = target.with_suffix("") if target.suffix else target
    if Path(str(prefix) + ".truth.json").exists():
        return GroundTruth.load(prefix)
    return None


def _cmd_spectrum(args: argparse.Namespace) -> int:
    signal = load_signal(args.path)
    psd = welch_psd(signal, nperseg=args.nperseg, nfft=args.nfft)
    floor = psd.noise_floor()
    peak = psd.peak()
    band = psd.occupied_band(noise_floor=floor)
    payload: dict[str, Any] = {
        "path": str(args.path),
        "sample_rate": signal.sample_rate,
        "num_samples": signal.num_samples,
        "nperseg": psd.nperseg,
        "df_hz": psd.df,
        "total_power": psd.total_power(),
        "noise_floor_psd": floor,
        "peak": {"frequency_hz": peak["frequency"], "snr_db": peak["snr_db"]},
        "occupied_band": band,
    }
    if args.out:
        prefix = Path(str(args.out))
        prefix.parent.mkdir(parents=True, exist_ok=True)
        csv_path = Path(str(prefix) + ".psd.csv")
        np.savetxt(
            csv_path,
            np.column_stack([psd.freqs, 10.0 * np.log10(np.maximum(psd.psd, 1e-300))]),
            delimiter=",",
            header="freq_hz,psd_db_per_hz",
            comments="",
        )
        payload["psd_csv"] = str(csv_path)
        if args.waterfall:
            waterfall = stft_waterfall(signal, nperseg=max(64, int(args.nperseg) // 4))
            npz_path = Path(str(prefix) + ".waterfall.npz")
            np.savez_compressed(
                npz_path,
                times=waterfall.times,
                freqs=waterfall.freqs,
                magnitude_db=waterfall.magnitude_db,
                sample_rate=waterfall.sample_rate,
                nperseg=waterfall.nperseg,
            )
            payload["waterfall_npz"] = str(npz_path)
            payload["waterfall_shape"] = list(waterfall.shape)
    if args.plot:
        from spxh.tools.plots import plot_spectrum

        truth = _truth_for(args.path)
        payload["plot"] = plot_spectrum(
            psd,
            args.plot,
            cfo_est=None,
            cfo_true=float(truth.cfo_hz) if truth is not None else None,
            occupied=band,
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(payload, "频谱分析:")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    signal = load_signal(args.path)
    result = analyze_signal(
        signal, nperseg=args.nperseg, nfft=args.nfft, fmin=args.fmin, fmax=args.fmax
    )
    payload: dict[str, Any] = result.to_dict()
    truth = _truth_for(args.path)
    if truth is not None:
        payload["comparison"] = compare_with_truth(result, truth)
    if args.plot:
        from spxh.tools.plots import plot_spectrum

        psd = welch_psd(signal, nperseg=args.nperseg, nfft=args.nfft)
        payload["plot"] = plot_spectrum(
            psd,
            args.plot,
            cfo_est=result.cfo.value,
            cfo_true=float(truth.cfo_hz) if truth is not None else None,
            occupied=result.occupied,
            title="功率谱 + 载频估计（" + Path(str(args.path)).name + "）",
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("M1 参数估计（" + str(args.path) + "）")
        print(result.summary())
        if "comparison" in payload:
            cmp = payload["comparison"]
            print("-" * 60)
            print("与真值对账:")
            print("  载频误差      {:+.4f} Hz".format(cmp["cfo_error_hz"]))
            print("  符号速率误差  {:+.6f} Hz ({:+.2e} 相对)".format(cmp["symbol_rate_error_hz"], cmp["symbol_rate_rel_error"]))
            print("  Es/N0 误差    {:+.4f} dB".format(cmp["es_n0_error_db"]))
            print("  每符号采样点  真值 {} / 估计 {:.3f}".format(cmp["sps_true"], cmp["sps_est"]))
        if "plot" in payload:
            print("图: " + str(payload["plot"]))
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    signal = load_signal(args.path)
    feature_set = extract_features(signal, config=FeatureConfig(nperseg=int(args.nperseg)))
    payload = feature_set.as_dict()
    payload["path"] = str(args.path)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print("五域 23 维特征（" + str(args.path) + "）")
    print("  符号速率 {:.3f} Hz | 载频 {:.3f} Hz | Es/N0 {:.2f} dB".format(
        float(feature_set.evidence.get("symbol_rate", 0.0)),
        float(feature_set.evidence.get("cfo_hz", 0.0)),
        float(feature_set.evidence.get("es_n0_db", 0.0)),
    ))
    domain_names = {
        "cumulant": "原始 I/Q 累积量",
        "spectral": "频谱",
        "timefreq": "STFT 时频",
        "constellation": "星座",
        "cyclostationary": "循环平稳",
    }
    for domain, indices in FEATURE_GROUPS.items():
        print("  [" + domain_names.get(domain, domain) + "]")
        for index in indices:
            name = feature_set.names[index]
            print("    {:34s} {:14.5f}".format(name, float(feature_set.values[index])))
    for warning in feature_set.warnings:
        print("  告警: " + warning)
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    signal = load_signal(args.path)
    payload = classify_signal(
        signal,
        model_dir=args.model_dir,
        feature_config=FeatureConfig(nperseg=int(args.nperseg)),
        threshold=args.threshold,
    )
    payload["path"] = str(args.path)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print("调制识别（" + str(args.path) + "）")
    print("  判决      {}  置信度 {:.3f}  门控 {}".format(payload["modulation"], payload["confidence"], payload["gate"]))
    print("  概率      " + "  ".join("{}={:.3f}".format(k, v) for k, v in sorted(payload["probabilities"].items(), key=lambda kv: -kv[1])))
    ood = payload.get("ood") or {}
    print("  OOD       马氏距离 {:.2f}（门限 {:.2f}，{}）".format(
        float(ood.get("score", 0.0)), float(ood.get("threshold", 0.0)),
        "判定为分布外" if ood.get("is_outlier") else "分布内"))
    if payload.get("physical"):
        physical = payload["physical"]
        print("  物理规则  {}（{:.2f}）: {}".format(physical.get("modulation"), float(physical.get("confidence", 0.0)), physical.get("reason")))
    for warning in payload.get("warnings", []):
        print("  告警: " + warning)
    return 0


def _cmd_frame(args: argparse.Namespace) -> int:
    from spxh import api

    try:
        payload = api.frame_path(
            args.path, modulation=args.mod, max_frames=args.max_frames,
            model_dir=args.model_dir, blind=bool(getattr(args, "blind", False)),
        )
    except Exception as exc:  # noqa: BLE001
        print("帧提取失败：" + str(exc))
        return 1
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    sync = payload.get("sync", {})
    summary = payload.get("summary", {})
    print("M5 帧同步（" + str(args.path) + "）")
    print("  调制 {}（来源 {}）| 前导 {} | 相关峰 {:.3f} | 峰旁瓣比 {:.1f} dB | 锁定 {}".format(
        payload.get("modulation"), payload.get("modulation_source"), sync.get("preamble"),
        float(sync.get("correlation_peak", 0.0)), float(sync.get("peak_to_sidelobe_db", 0.0)),
        "是" if sync.get("locked") else "否"))
    print("  帧起点 符号 {} / 采样 {} | 前导解出的相位模糊 {:+.1f} 度（不需要真值）".format(
        sync.get("frame_start_symbol"), sync.get("frame_start_sample"),
        float(sync.get("ambiguity_rotation_deg", 0.0))))
    print("  帧数 {} | CRC 通过 {} | 载荷 {} 字节{}".format(
        summary.get("frames"), summary.get("frames_crc_ok"), summary.get("payload_bytes_total"),
        "" if summary.get("payload_bytes_exact") is None
        else " | 与真值逐字节一致 {} 字节".format(summary.get("payload_bytes_exact"))))
    print("")
    print("  {:>3s} {:>7s} {:>10s} {:>6s} {:>8s} {:>6s}  {}".format("帧", "起始符号", "载荷长度", "前导", "CRC", "误码", "载荷预览(ASCII)"))
    for frame in payload.get("frames", [])[:16]:
        header = frame.get("header", {})
        crc = frame.get("crc", {})
        pl = frame.get("payload", {})
        print("  {:>3d} {:>7.1f} {:>10d} {:>6s} {:>8s} {:>6s}  {}".format(
            int(frame.get("index", 0)), float(frame.get("start_symbol", 0.0)),
            int(header.get("payload_len", 0)), "OK" if frame.get("preamble_ok") else "NG",
            "OK" if crc.get("ok") else "NG",
            "null" if frame.get("bit_errors_vs_truth") is None else str(frame.get("bit_errors_vs_truth")),
            str(pl.get("ascii", ""))[:48]))
    for warning in payload.get("warnings", []):
        print("  告警: " + str(warning))
    return 0


def _cmd_fec(args: argparse.Namespace) -> int:
    from spxh import api

    try:
        payload = api.fec_report(args.report)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print("M4 信道编码目录（报告时间 {}）".format(payload.get("updated_at", "未知")))
    print("{:10s} {:26s} {:>7s}  {}".format("id", "名称", "码率", "参数"))
    for item in payload.get("catalogue", []):
        print("{:10s} {:26s} {:7.3f}  {}".format(
            item.get("id", ""), item.get("name", ""), float(item.get("rate", 0.0)), item.get("params", "")))
    metrics = payload.get("metrics", {})
    print("")
    print("软判决增益: {:.2f} dB（{}）".format(
        float(metrics.get("soft_gain_db", float("nan"))), metrics.get("soft_gain_note", "")))
    burst = metrics.get("rs_burst_demo", {})
    if burst:
        print("RS 突发演示: {} 符号突发，交织后每个码字最多 {} 个错（t={}）-> {}".format(
            burst.get("burst_length_symbols"), burst.get("max_errors_per_codeword"),
            burst.get("correctable"), "可纠" if burst.get("recovered") else "不可纠"))
    ldpc = metrics.get("ldpc", {})
    if ldpc:
        print("LDPC: n={} k={} m={} 码率 {:.4f} 迭代 {}".format(
            ldpc.get("n"), ldpc.get("k"), ldpc.get("m"), float(ldpc.get("rate", 0.0)), ldpc.get("iterations")))
    print("")
    print("验收清单：")
    for item in metrics.get("checks", []):
        print("  [{}] {}（实测 {}，目标 {}）".format(
            "PASS" if item.get("pass") else "FAIL", item.get("name", ""),
            item.get("value"), item.get("target")))
    return 0


def _cmd_demod(args: argparse.Namespace) -> int:
    from spxh.core.demod import demodulate
    from spxh.core.estimate import analyze_signal

    signal = load_signal(args.path)
    analysis = analyze_signal(signal)
    truth = _truth_for(args.path)
    result = demodulate(
        signal,
        analysis=analysis,
        modulation=args.mod,
        modulation_source="provided" if args.mod else "classifier",
        rolloff=float(args.rolloff),
        truth=truth,
        model_dir=args.model_dir,
    )
    payload = result.to_dict()
    payload["path"] = str(args.path)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print("M3 解调（" + str(args.path) + "）")
    print("  调制 {}（来源 {}）| 符号速率 {:.3f} Hz | 载频 {:.3f} Hz | Es/N0 {:.2f} dB".format(
        result.modulation, result.modulation_source, result.symbol_rate, result.cfo_hz, result.snr_db))
    print("  锁定  定时 {:.3f} ({}) | 载波 {:.3f} ({}) | 符号数 {} | 比特数 {}".format(
        result.timing_metric, "锁定" if result.timing_metric >= 0.5 else "未锁定",
        result.carrier_metric, "锁定" if result.carrier_metric >= 0.5 else "未锁定",
        result.num_symbols, result.num_bits))
    if result.evm_percent is not None:
        print("  EVM   {:.2f}%".format(result.evm_percent))
    print("  LLR   均值|LLR| {:.3f}（范围 {:.2f} ~ {:.2f}）".format(
        float(np.mean(np.abs(result.llr))) if result.llr.size else 0.0,
        float(np.min(result.llr)) if result.llr.size else 0.0,
        float(np.max(result.llr)) if result.llr.size else 0.0))
    if result.ber:
        print("  误码  BER {:.3e}（{}/{}）SER {:.3e} | 模糊旋转 {:+.0f} 度 | 符号滑动 {}".format(
            result.ber["bit_error_rate"], result.ber["bit_errors"], result.ber["compared_bits"],
            result.ber["symbol_error_rate"], result.ber["ambiguity_rotation_deg"], result.ber["symbol_shift"]))
    else:
        print("  误码  无真值，无法计算 BER")
    for warning in result.warnings:
        print("  告警: " + warning)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from spxh.web.server import serve

    # 默认把逐请求日志写文件：服务常驻在后台作业里时，stdout 是管道，
    # 写爆管道会把进程带走（这正是工作台"无声退出"的原因）。
    from spxh.web.auth import resolve_token

    token = None if getattr(args, "no_auth", False) else resolve_token(getattr(args, "token", None))
    serve(
        host=args.host,
        port=int(args.port),
        model_dir=args.model_dir,
        quiet=not bool(getattr(args, "verbose", False)),
        log_path=getattr(args, "log", None) or None,
        token=token,
    )
    return 0


def _cmd_mods(_args: argparse.Namespace) -> int:
    from spxh.core.synth.mapping import bits_per_symbol, num_points

    print("支持的调制:")
    for mod in MODULATIONS:
        print("  {:6s} bits/symbol={} points={}".format(mod, bits_per_symbol(mod), num_points(mod)))
    return 0


def _cmd_agent(args) -> int:
    """M11 工具调用代理：把"先看什么、再看什么"交给模型，但预算由平台硬控."""
    from spxh import api

    if getattr(args, "tools", False):
        print(json.dumps(api.agent_tools(), ensure_ascii=False, indent=2))
        return 0
    if not args.path:
        print("需要信号文件路径（或用 --tools 只看工具清单）")
        return 2
    report = api.agent_run(
        args.path, goal=args.goal, provider=args.provider,
        max_rounds=args.max_rounds, max_tool_calls=args.max_tool_calls, timeout_s=args.timeout,
    )
    if getattr(args, "json", False):
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    trace = report.get("trace_summary") or {}
    print("=" * 100)
    print("SPXH 分析代理（{} / {}，停止原因：{}）".format(report.get("mode"), report.get("model"), report.get("stop_reason")))
    print("=" * 100)
    for row in report.get("rounds") or []:
        print("")
        print("第 {} 轮{}".format(row.get("index"), ("：" + str(row["assistant_text"])[:120]) if row.get("assistant_text") else ""))
        for call in row.get("tool_calls") or []:
            mark = "OK " if call.get("ok") else "ERR"
            print("  [{}] {} ({}) {} ms".format(mark, call.get("name"), json.dumps(call.get("args"), ensure_ascii=False), call.get("duration_ms")))
            print("        " + str(call.get("summary") or call.get("error") or "")[:160])
    final = report.get("final") or {}
    print("")
    print("结论：" + str(final.get("headline") or ""))
    print("可信度：{:.2f}".format(float(final.get("confidence") or 0.0)))
    for section in final.get("sections") or []:
        print("")
        print("【{}】".format(section.get("title")))
        print("  " + str(section.get("text"))[:600])
    if final.get("uncertainties"):
        print("")
        print("不确定项：")
        for text in final["uncertainties"]:
            print("  ! " + str(text))
    grounding = report.get("grounding") or {}
    print("")
    print("运行摘要：{} 轮 / {} 次工具调用 / 用到 {} / {} ms".format(
        trace.get("rounds"), trace.get("tool_calls"), "、".join(trace.get("tools_used") or []) or "-", trace.get("elapsed_ms")))
    print("数值落地：{}（检查 {} 个数字）".format(
        "全部可溯源" if grounding.get("grounded") else "有 {} 个无法溯源".format(len(grounding.get("unsupported") or [])),
        grounding.get("checked_numbers")))
    for warning in report.get("warnings") or []:
        print("  警告: " + str(warning))
    return 0


def _cmd_llm(args) -> int:
    """查看/设置 LLM 配置；密钥只落本地文件，**输出一律脱敏**."""
    from spxh.core.report import list_models, load_config, masked_config, save_config, test_connection

    changes: dict[str, Any] = {}
    for item in getattr(args, "set", []) or []:
        if "=" not in str(item):
            print("--set 需要 KEY=VALUE 形式，收到：" + str(item))
            return 2
        key, value = str(item).split("=", 1)
        changes[key.strip()] = value.strip()
    if getattr(args, "clear_key", False):
        changes["clear_api_key"] = True
    try:
        if changes:
            save_config(changes)
            print("已保存到 " + str(load_config().config_path))
    except ValueError as exc:
        print("配置未保存：" + str(exc))
        return 2

    payload: dict[str, Any] = {"config": masked_config()}
    if getattr(args, "test", False):
        payload["test"] = test_connection()
    if getattr(args, "models", False):
        payload["models"] = list_models()
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    config = payload["config"]
    print("=" * 92)
    print("LLM 配置（来源：file=本地文件 / env=环境变量 / default=默认；密钥只回显尾 4 位）")
    print("=" * 92)
    for key in ("enabled", "base_url", "model", "temperature", "max_tokens", "timeout_s"):
        print("  {:12s} {:>28}   [{}]".format(key, str(config[key]), config["source"].get(key, "-")))
    print("  {:12s} {:>28}   [{}]".format("api_key", config["api_key_hint"] or "（未配置）",
                                          config["source"].get("api_key", "-")))
    print("  {:12s} {:>28}".format("ready", "是" if config["ready"] else "否（enabled=False 或未配置 Key）"))
    print("  配置文件: " + config["config_path"])
    for name, state in config["env"].items():
        print("  环境变量 {}: {}".format(name, state))
    if "models" in payload:
        models = payload["models"]
        print("\n可用模型（{} 个）: {}".format(len(models.get("models") or []),
                                              ", ".join(models.get("models") or []) or models.get("error")))
    if "test" in payload:
        result = payload["test"]
        print("\n连通性测试: " + ("成功" if result.get("ok") else "失败"))
        print("  延迟 {} ms | 模型 {} | 回复 {}".format(result.get("latency_ms"),
                                                      result.get("model", "-"), result.get("reply", "-")))
        if result.get("error"):
            print("  原因：" + str(result["error"]))
    return 0


def _cmd_narrate(args) -> int:
    """M9 分析报告：LLM 只做解释，所有数值来自确定性管线."""
    from spxh.core.report import build_prompt, collect_evidence, narrate as build_report

    if getattr(args, "prompt", False) and not getattr(args, "json", False):
        evidence = collect_evidence(args.path)
        print(build_prompt(evidence))
        return 0
    report = build_report(args.path, provider=args.provider, max_tokens=args.max_tokens)
    if getattr(args, "json", False):
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    mode = report.get("mode")
    print("=" * 100)
    print("SPXH 分析报告（{} / {}，可信度 {:.2f}）".format(mode, report.get("model"), float(report.get("confidence") or 0.0)))
    print("=" * 100)
    print(report.get("headline") or "")
    for section in report.get("sections") or []:
        print("")
        print("【{}】".format(section.get("title")))
        print("  " + str(section.get("text")))
        for item in section.get("evidence") or []:
            unit = (" " + str(item["unit"])) if item.get("unit") else ""
            conf = " conf={:.2f}".format(item["confidence"]) if item.get("confidence") is not None else ""
            print("    - {} = {}{}  [{}]{}".format(item["key"], item["value"], unit, item["source"], conf))
    if report.get("uncertainties"):
        print("")
        print("不确定项：")
        for text in report["uncertainties"]:
            print("  ! " + str(text))
    if report.get("next_actions"):
        print("")
        print("建议下一步：")
        for text in report["next_actions"]:
            print("  -> " + str(text))
    for warning in report.get("warnings") or []:
        print("  警告: " + str(warning))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "dataset":
        return _cmd_dataset(args)
    if args.command == "info":
        return _cmd_info(args)
    if args.command == "spectrum":
        return _cmd_spectrum(args)
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "features":
        return _cmd_features(args)
    if args.command == "classify":
        return _cmd_classify(args)
    if args.command == "demod":
        return _cmd_demod(args)
    if args.command == "narrate":
        return _cmd_narrate(args)
    if args.command == "llm":
        return _cmd_llm(args)
    if args.command == "agent":
        return _cmd_agent(args)
    if args.command == "fec":
        return _cmd_fec(args)
    if args.command == "frame":
        return _cmd_frame(args)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "mods":
        return _cmd_mods(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

<div align="center">

<img src="docs/figures/logo.png" width="140" alt="SPXH logo">

# SPXH · Smart RF Signal Analysis Platform

**智能射频信号分析平台**

From an **unknown I/Q** stream to **readable payload bits** — every step comes with its justification

Blind parameter estimation · modulation recognition · demodulation and soft information · channel coding · frame synchronization · blind FEC detection · spectral correlation · LLM reports · tool-calling agent

[![CI](https://github.com/WmjXiaoJun/SPXH/actions/workflows/ci.yml/badge.svg)](https://github.com/WmjXiaoJun/SPXH/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-3776AB)
![Tests](https://img.shields.io/badge/pytest-400%2B%20collected-brightgreen)
![Deps](https://img.shields.io/badge/deps-numpy%20%7C%20scipy-013243)
![License](https://img.shields.io/badge/license-MIT-green)

[Project on GitHub](https://github.com/WmjXiaoJun/SPXH) ·
[Quick start](#quick-start) ·
[All features](#all-features) ·
[Measured data](#measured-data) ·
[Workbench](#web-workbench) ·
[Roadmap](#roadmap) ·
[Acknowledgements](#acknowledgements)

**[中文](README.md) · English**

[Quick start](#quick-start) · [All features](#all-features) · [Measured data](#measured-data) · [Workbench](#web-workbench) · [Self-check scripts](#acceptance-scripts) · [Project layout](#project-layout) · [Honest limits](#design-trade-offs--honest-limits) · [FAQ](#faq)

</div>

<img src="docs/figures/banner.png" width="100%" alt="SPXH banner">

> Repository: [WmjXiaoJun/SPXH](https://github.com/WmjXiaoJun/SPXH). If you fork this project, remember to point the badges and this file at your own account.
> Every measured number can be replayed by `scripts/check_*.py`; the item-by-item provenance record is in [docs/开源检查清单.md](docs/开源检查清单.md).

---

## What is this

SPXH solves exactly one problem:

> You have only a chunk of I/Q data whose **waveform is unknown, whose symbol rate is unknown, and which may or may not carry error-correction coding** —
> how do you turn it into readable bits and payload, with **evidence for every step**?

It breaks that problem into a deterministic pipeline in which each stage does one thing and exposes its own intermediate quantities
(confidence, evidence, synchronization traces, CRC status, lock indicators, soft information):

```text
M0 synthesis/IO → M1 parameter estimation → M2 features+modulation recognition → M3 demodulation/LLR → M4 channel coding
          → M5 frame synchronization/payload → M6 coded link → M7 blind FEC detection and erasure decoding → M8 spectral correlation
          → M9 header CRC/blind length recovery → M10 LLM report (configurable, not allowed to invent numbers) → M11 tool-calling agent
          → M12 soft preamble (low-SNR recall) and joint decoding into the link
```

**Pure Python; the core depends only on numpy + scipy** (scikit-learn for the classifier, matplotlib for plots).
Each of the 12 milestones ships a **re-runnable acceptance report**, and every number in a report can be replayed by its corresponding script.

## Milestone overview

| Milestone | What it solves | Key capabilities (modules) | Report |
|---|---|---|---|
| **M0** | Making the whole chain verifiable | Multi-format I/O (RAW/WAV/SigMF/npz), synthesis of six modulation families, impairment injection, ground-truth sidecar (`core/io`, `core/synth`, `core/framing`) | [M0](docs/M0-验收报告.md) |
| **M1** | Unknown I/Q → parameters with confidence | Welch PSD / STFT waterfall, noise floor and occupied bandwidth, carrier frequency, symbol rate, blind SNR (`core/dsp`, `core/estimate`) | [M1](docs/M1-验收报告.md) |
| **M2** | Parameters → waveform | 23 features across five domains, SNR-binned random forest, probability calibration, confidence gating, physical-rule fallback, OOD (`core/features`, `core/classify`) | [M2](docs/M2-验收报告.md) |
| **M3** | Waveform → decided symbols and soft information | Gardner timing loop, decision-directed carrier loop, six demodulators, max-log LLR, lock indicators and sync traces (`core/demod`) | [M3](docs/M3-验收报告.md) |
| **M4** | Soft information → more reliable bits | Interleaving, Viterbi (hard/soft), RS(255,223), CCSDS concatenation, LDPC normalized min-sum (`core/fec`) | [M4](docs/M4-验收报告.md) |
| **M5** | Symbol stream → frames and payload | Barker-13 preamble sync, phase-ambiguity resolution, header parsing, CRC-16 check, payload extraction (`core/framesync`) | [M5](docs/M5-验收报告.md) |
| **M6** | Wiring the coded link into the frame structure | Framing / deinterleaving / soft-decision decoding / end-to-end CRC; CRC placed inside the code (`core/fec/codec.py`, `core/framing/format.py`) | [M6](docs/M6-验收报告.md) |
| **M7** | Smooth curves and "do not trust the header" | Multi-frame-per-point end-to-end curves, blind FEC detection (hypothesis + header rewrite + CRC check), RS erasure decoding | [M7](docs/M7-验收报告.md) |
| **M8** | Erasures + unknown errors; explainability | Joint RS error-and-erasure decoding, SSCA spectral-correlation bifrequency plane, case set grown to 32 (`core/explain`) | [M8](docs/M8-验收报告.md) |
| **M9** | Protecting the header itself | Header CRC-8 (4→5 bytes), blind payload-length recovery, LLM analysis-report layer (`core/report/narrator.py`) | [M9](docs/M9-验收报告.md) |
| **M10** | Making the LLM trustworthy and controllable | Three-level config precedence (file > env > default), masked echo, field validation, key stored locally only, per-number traceability (`core/report/llm_config.py`) | [M10](docs/M10-验收报告.md) |
| **M11** | Letting the model decide "what to look at" | Tool-calling agent (9 tools, bounded loop, dedup, offline deterministic fallback), access token, DPAPI-encrypted key (`core/agent`, `web/auth.py`) | [M11](docs/M11-验收报告.md) |
| **M12** | Low-SNR recall and joint decoding in the link | Soft-preamble decision, LLR-based joint erasure decoding, **burst detection (collapse/jam) with automatic erasure marking (CRC arbitration)**, agent batch sweeps and decode tool | [M12](docs/M12-验收报告.md) |
## Quick start

### 1. Installation

```bash
git clone https://github.com/WmjXiaoJun/SPXH.git      # or download the ZIP
cd SPXH
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Linux/mac: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,ml]"             # or: pip install -r requirements.txt
```

> The minimal runnable dependencies are **numpy + scipy**; modulation recognition needs scikit-learn; plotting needs matplotlib (when it is missing, scripts skip plots without affecting the criteria).

>
> **⚠️ The classifier model (59 MB) is not shipped with the repository.** After cloning, rebuild it once (about 1 minute) so modulation recognition can use the ML decision:
>
> ```bash
> python scripts/train_classifier.py
> ```
>
> It works without it too: `classify` automatically falls back to **physical rules** (the gate reports `fallback`) and warns explicitly that no trained model was found.
> Parameter estimation / demodulation / frame sync / decoding / spectral correlation and the full test suite are unaffected (all 411 tests pass without the model).

### 2. Four commands, from synthesis to "reading something out"

```bash
# ① Synthesize a signal with complete ground truth (M0)
python -m spxh generate --mod qpsk --snr 10 --out data/qpsk_snr10

# ② Blind parameter estimation: carrier / symbol rate / blind SNR, automatically reconciled against ground truth when available (M1)
python -m spxh analyze data/demo/qpsk_snr+10.0_000.sigmf-meta

# ③ Frame synchronization + decoding + CRC + payload extraction in a single command (M5/M6/M8/M9)
python -m spxh frame data/demo/qpsk_conv_snr+10.0_000.sigmf-meta --blind

# ④ Tool-calling agent: let the model itself decide whether to look at spectral correlation or decode frames first (M11; falls back to an offline deterministic plan without an API key)
python -m spxh agent data/demo/qpsk_snr+10.0_000.sigmf-meta --goal "What waveform is this? Can the payload be read out?"
```

`data/demo/` ships with **32 cases** (six modulation types × four SNR levels + coded-link cases),
so you can run ②③④ directly without generating anything first.

### 3. Open the graphical workbench

```bash
python -m spxh serve        # open http://127.0.0.1:8760/ in your browser
```

### 4. Run acceptance checks and tests

```bash
python -m pytest -q                       # full test suite (redirect long output: > pytest.log 2>&1)
python scripts/check_m0.py                # synthesis-chain self-check
python scripts/check_m5.py                # frame synchronization + payload recovery
python scripts/check_m12.py --part recall # low-SNR frame recall (soft preamble)
```

---

## All features

Going in the order in which data flows, each feature is explained in full: **what it does, how to use it, what it outputs, and where its limits are**.
The link at the end of each section is that item's acceptance report (with measured numbers and reproduction commands).

### 1. Data and ground truth: making the whole chain verifiable (M0)

**What it does**: reads and writes four data formats, and can **synthesize** signals with complete ground truth — without ground truth, none of the later "accuracy" or "bit error rate" claims mean anything.

- **Formats**: RAW (raw complex samples), WAV, **SigMF** (metadata + data files), npz;
- **Synthesis**: BPSK / QPSK / 8PSK / 16QAM / 2FSK / 4FSK, RRC pulse shaping (default roll-off 0.35, span 16 symbols),
  with optional injection of **AWGN, carrier frequency offset, phase noise, IQ imbalance, DC offset**;
- **Framing**: Barker-13 preamble + 5-byte frame header (containing CRC-8) + payload + CRC-16;
- **Ground-truth sidecar**: `*.truth.json` records the payload of every bit, every symbol and every frame plus all channel parameters;
  randomness is driven entirely by the seed (the same seed always yields the same waveform).

```bash
python -m spxh generate --mod 16qam --snr 8 --fec ldpc --payload-len 64 --frames 8 --out data/mine
python -m spxh dataset  --out data/grid --mods bpsk,qpsk,8psk --snrs 0,10,20   # batch dataset generation
python -m spxh info     data/demo/qpsk_snr+10.0_000.sigmf-meta                 # inspect metadata
```

**Limits**: the ground truth is "ideal receiver" ground truth; the synthesized channel contains no multipath and no nonlinearity.
See the [M0 acceptance report](docs/M0-验收报告.md).

### 2. Blind parameter estimation: first learn "how fast it is and how far it is off" (M1)

**What it does**: estimates **Es/N0, carrier frequency offset and symbol rate** with no knowledge of any parameter, and reports **confidence and method** for each.

- Welch PSD + STFT waterfall, noise floor and occupied bandwidth;
- Carrier: peak-picking is unreliable on a flat-top spectrum, so **spectral symmetry / centroid**-style criteria are used instead (this pitfall is written up in the report);
- Symbol rate: cyclostationary / spectral-line criteria;
- Blind SNR: based on the noise floor and the signal power; the largest deviation from the defined value is **< 0.06 dB**.

**Output**: every estimate is `{value, unit, confidence, method, evidence}` — which is exactly what allows the downstream LLM report to "cite only, never invent".

**Measured**: worst carrier error **10.5 Hz (0.85% Rs)**, worst symbol-rate relative error **1.3e-4**, worst Es/N0 error **0.16 dB**.

**Limits**: carrier accuracy is about 1% Rs — enough for acquisition, not enough for second-scale coherent processing — which is why M3 must have tracking loops.
See the [M1 acceptance report](docs/M1-验收报告.md).

### 3. Features and modulation recognition: saying "what waveform this is" (M2)

**What it does**: turns a signal into **23-dimensional features across five domains**, classifies with an **SNR-binned random forest**, and reports **calibrated probabilities**.

- Five domains: instantaneous statistics, spectral shape, higher-order cumulants, constellation geometry, cyclostationarity;
- **Confidence gating**: when the probability is not high enough it refuses to answer (`reject`), or falls back to **physical rules** (`fallback`), and only takes the machine-learning decision (`ml`) when it is confident;
- **OOD detection**: decides whether a sample lies outside the training distribution;
- Probabilities are **calibrated** (ECE 0.087 @SNR≥5 dB), so the "confidence" shown in the interface can be read as a probability.

**Measured**: recognition accuracy **1.000 @20 dB / 0.900 @10 dB**; among samples decided by ML the error rate is 0.057; OOD detection rate 0.594.

```bash
python -m spxh features  data/demo/qpsk_snr+10.0_000.sigmf-meta   # inspect the 23 features
python -m spxh classify  data/demo/qpsk_snr+10.0_000.sigmf-meta   # modulation + probabilities + gating
python scripts/train_classifier.py                                # run once on first use: rebuild the 59 MB classifier model (not shipped)
```

**Limits**: the classifier is trained on **synthetic data** only; below 0 dB it is essentially unusable; and it is insensitive to OOD samples that belong to "the same family but with shifted parameters".
See the [M2 acceptance report](docs/M2-验收报告.md).

### 4. Demodulation and soft information: turning symbols into "bits with confidence" (M3)

**What it does**: **Gardner timing loop + decision-directed carrier loop** for synchronization, decisions for six modulation types, and **max-log LLR** output.

- The loops expose **lock indicators** (separate metrics for timing and carrier) and **convergence traces**, so the interface shows directly whether they are actually locked;
- Soft information (LLR) is output with the convention **positive ⇒ more likely a 0**, and is carried all the way into the FEC decoder;
- LLRs go through a **calibration reconciliation**: at 5 dB the theoretically expected number of bit errors is 69.5, measured 69 (ratio 0.99).

**Measured**: for QPSK, BER against the "no-tracking baseline" improves **4.95e-1 → 9.2e-4 (537× improvement)**; for 16QAM **4.21e-1 → 1.0e-3 (406×)**.

```bash
python -m spxh demod data/demo/qpsk_conv_snr+10.0_000.sigmf-meta
```

**Limits**: 8PSK/16QAM have an **error floor** (caused by residual jitter in the synchronization loops, not by noise). See the [M3 acceptance report](docs/M3-验收报告.md).

### 5. Channel coding and decoding: trading soft information for reliability (M4)

**What it does**: encodes and decodes six schemes, and quantifies "how much was actually gained".

| Scheme | Description | Measured |
|---|---|---|
| Matrix/convolutional interleaving | Spreads bursts apart | 32-symbol burst: uncorrectable without interleaving; after interleaving at most 7 errors per codeword, all corrected |
| Convolutional code + Viterbi | Hard decision / **soft decision** | Soft decision gains **+2.92 dB @1e-4** over hard decision |
| RS(255,223) | BM + Chien + Forney | Measured correction limit t matches theory |
| **RS errors-and-erasures joint decoding** | 2f + e ≤ 2t | 8 erasures + 4 errors (=16) recovered; 4 erasures + 7 errors (18) honestly reported as uncorrectable |
| CCSDS concatenated | RS × convolutional, retaining the concatenated structure and gain magnitude | Best end-to-end threshold (see Section 7) |
| LDPC(1152,578) | Gallager regular short code + normalized min-sum | Rank 574, rate 0.5017 (reported as measured) |

**Limits**: CCSDS/LDPC are implementations with "the same class of structure and the same order of gain"; **they cannot interoperate with real equipment**.
See the [M4 acceptance report](docs/M4-验收报告.md) and the [M8 acceptance report](docs/M8-验收报告.md).

### 6. Frame synchronization and payload extraction: recognizing frames in a symbol stream (M5)

**What it does**: correlates against the **Barker-13 preamble**, resolves **phase ambiguity**, parses the frame header, verifies **CRC-16**, and extracts the payload.

- Synchronization uses no ground truth: only the preamble and the frame structure;
- **Phase ambiguity** is resolved from the preamble itself (the 2-fold symmetry of BPSK and the 4-fold symmetry of QPSK both self-correct);
- The payload can be reconciled with ground truth **byte by byte** (`payload_bytes_exact`).

**Measured**: from 15 dB up, BPSK/QPSK give **8/8 frames passing CRC with byte-exact payloads**.

```bash
python -m spxh frame data/demo/qpsk_snr+10.0_000.sigmf-meta
```

**Limits**: recall is limited at low SNR (see the soft-preamble improvement in Section 12). See the [M5 acceptance report](docs/M5-验收报告.md).

### 7. Coded link: wiring FEC into the frame structure and measuring real gain (M6/M7)

**What it does**: connects the M4 coding/decoding to the M5 frame structure, and measures the end-to-end CRC threshold **with synchronization and estimation errors present**.

- Key decision: **the CRC goes inside the coding** (the check field is protected itself), leaving only the frame header uncoded;
- **Smooth curves**: many frames per SNR point, with thresholds by linear interpolation rather than coarse 4-frame quantization;
- **Blind FEC detection**: when the `fec` field in the frame header is corrupted (lying "no coding"), the procedure "hypothesize → **rewrite the frame-header fec bits** → check CRC" recovers all four of conv/rs/ccsds/ldpc (the key point: the header must be corrected at the same time, otherwise the CRC can never match).

**Measured**: uncoded **11.20 dB** → CCSDS concatenated **6.90 dB**, a **4.30 dB threshold improvement**.

```bash
python scripts/check_m6.py                    # coded-link threshold + interleaving comparison
python scripts/check_m7.py --part curves      # smooth end-to-end curves
python scripts/check_m7.py --part blind       # blind FEC detection
```

### 8. Header self-check and "the length can be recovered blindly too" (M9)

**What it does**: extends the frame header from 4 to **5 bytes**; the 5th byte is a **CRC-8** over the first 4.

- A corrupted header becomes **immediately visible**;
- Even a corrupted **payload length field** can be recovered: hypothesize over (coding, interleaving, length), rewrite the header, filter candidates with the CRC-8, and make the final decision with the in-frame CRC-16;
- When the header is untrustworthy, sequence advancement switches to the **most recent trustworthy frame length**, so a garbage length cannot drag it off track.

**Measured**: all five coding schemes are **recovered** after the length field is corrupted; at 8 dB frame recall goes convolutional **5/16 → 11/16**, CCSDS **4/16 → 12/16**.

### 9. Spectral correlation (SSCA): a piece of evidence you can see (M8)

**What it does**: computes the **cyclic spectrum bifrequency plane** `S_x^α(f)` — communication signals are cyclostationary, the shaping pulse forms ridges at
**α = the symbol rate and its harmonics**, carrier frequency offset appears on the **f axis**, and white noise has energy only at α=0.

- **The α-axis peak = symbol rate** (measured 1000.0 Hz, ground truth 1000 Hz, resolution fs/nfft = 31.25 Hz);
- **The f position of the α≠0 ridge = carrier frequency offset** (within one resolution bin);
- Displayed **side by side** with the precise M1 estimates, so "do the two independent pieces of evidence agree" is visible at a glance.

**Pitfalls recorded**: the α axis was off by a factor of 2 (the two SSCA branches each shift by α/2); CFO cannot be read by peak-picking the α=0 row (that is the flat-top power spectrum).

### 10. LLM report layer: letting the model "explain, not estimate" (M9/M10)

**What it does**: hands the **structured evidence** of the whole chain to an LLM and produces an analyst-style report; **with no key it uses an offline deterministic renderer**.

- **Division of labour**: every number comes from the deterministic pipeline (with `source/unit/confidence`); the model only explains, integrates, spots contradictions and makes suggestions;
- **No invented numbers**: the prompt hard-requires "cite only the given fields, write null when missing", and the report must also pass a **numeric grounding check**
  (every number in the body must be traceable to the evidence table, whose whitelist is exactly the JSON the model actually saw);
- **Configurable**: priority is **local file `models/llm_config.json` > environment variables `SPXH_LLM_*` > defaults**,
  with the source of every item reported; **keys are written only locally and the interface echoes only the last 4 characters**, and on Windows they are **DPAPI-encrypted** (old plaintext configs migrate automatically);
- **Provider presets**: DeepSeek / OpenAI / local Ollama / custom, one-click fill; base_url is normalized automatically;
  for models that do not support JSON mode (such as `deepseek-reasoner`) it **automatically degrades and retries**.

```bash
python -m spxh narrate data/demo/qpsk_conv_snr+10.0_000.sigmf-meta          # offline report (deterministic)
python -m spxh llm --set base_url=https://api.deepseek.com --set model=deepseek-chat --set api_key=sk-xxx --test
```

**Measured**: all **26 numbers** in the offline report body are traceable; fabricated numbers are flagged accurately.

### 11. Tool-calling agent: letting the model decide "what to look at next" (M11/M12)

**What it does**: exposes the pipeline as **9 tools**; the model explores autonomously over multiple rounds while the platform hard-caps the budget.

| Tool | Purpose |
|---|---|
| `analyze` | M1 parameters (Es/N0, carrier, symbol rate + confidence) |
| `classify` | M2 modulation, gating, OOD |
| `spectrum` | Power spectrum and spectral peaks |
| `demod` | M3 lock state, symbol count, EVM, BER |
| `frame` | M5/M9 synchronization, frame-header CRC, payload, blind recovery |
| `decode` | Full receive chain + decoding statistics (symbols corrected, uncorrectable frames, automatic erasures) |
| `ssca` | M8 spectral correlation (α peak / ridge) |
| `batch` | Batch inspection of multiple cases, with a cross-comparison table |
| `cases` | Case inventory |

**Three hard constraints**: ① **Bounded** — threefold caps on rounds (6/12), tool calls (12/24) and wall clock (120/300 s),
so it stops on hitting a cap and honestly reports `stop_reason`; ② **The same tool with the same arguments is rejected outright** (to keep the model from spinning and burning budget);
③ **Facts may come only from tools** — conclusions must pass the numeric grounding check.

**Measured on real hardware (DeepSeek)**: 4 rounds / 7 calls / 22.5 s. The model first fetched parameters and spectral correlation on its own, then
**proactively compared frame extraction "with blind recovery vs without"** (5/6 vs 4/6 frames passing CRC), and finally gave a traceable conclusion —
which is exactly the value of an agent over a one-shot summary.

```bash
python -m spxh agent <path> --goal "..."   # LLM with a key, offline deterministic plan without
python -m spxh agent --tools               # just list the tools and the budget
```

### 12. Low-SNR frame recall: soft preamble decision (M12)

**What it does**: at low SNR the bottleneck for frame recall is not that correlation peaks are too low, but that the **parsing stage demands all 13 Barker bits to match**.
It is changed to a **soft decision**: tolerate at most 2 mismatched bits **and** a positive soft correlation, with misdetections caught in two stages by the header CRC-8 and the frame CRC-16.

**Measured** (16 frames, QPSK): convolutional at 6 dB **4/4 → 10/10**, at 10 dB **13/13 → 16/16**; CCSDS at 8 dB **12/10 → 14/11**.
The uncoded link barely changes (it has no redundancy, so failures are in the payload CRC) — recorded honestly.

**The same round also measured a counter-intuitive result**: after wiring "mark an erasure when |LLR| is low" into the link, the threshold under pure AWGN **got worse instead**
(8.78 → 9.20 dB), because most low-|LLR| symbols are **correct**, while every wrong erasure eats up half the correction capability.
After changing it to mark **only contiguous weak spans (≥3 symbols)**, it is **point-for-point identical** to no erasure marking under AWGN, with **0 automatic erasures**;
known-position bursts (9/12/16 bytes) still **all pass**. In one sentence: **what makes erasure information valuable is a trustworthy position, not a low confidence**.

### 12.1 Burst detection + automatic erasure marking (M12-C)

**What it does**: finds "this span is untrustworthy" **from the signal side** and hands the positions to joint decoding — nothing tells it where the damage is.

- **Detection**: sliding RMS + a two-part criterion (depth < -12 dB **and** below median - 4×MAD), covering two damage classes —
  **collapse** (blockage / zeroing / deep fade) and **jam** (suppression / strong interference, power raised by ≥6 dB); clean signals and pure noise give **zero false positives**;
- **Mapping**: the timing loop now exposes each symbol's sampling position (`symbol_samples`), so burst interval → symbols → in-frame bits → RS symbol indices;
- **Arbitration**: candidate erasure budgets are tried one by one and **CRC has the final say** (the budget includes 0) → **automatic erasure marking is never worse than marking nothing** (11/11 scenarios, no regression).

**Measured conclusion (honest version)**: both detection and the link wiring work, but **this configuration yields no frame-level gain** — a local burst either falls inside the RS t=8 margin,
or is strong enough to take down the synchronization loops with it (the damage spreads across the whole frame, far beyond 2t). **The bottleneck is the sync loop, not the decoder**, so the next step is burst resistance on the synchronization side.
One more note: blockage-style zeroing is actually a **very weak** impairment for QPSK (the zero point lands on a fixed decision point, so only about 8 of 20 symbols really error),
and using it for burst experiments creates the illusion of a "very robust system" — that control case is kept in the report as well.

**Sync-side burst resistance (M13 attempt)**: detected intervals are first **zeroed** (so strong interference cannot pollute the loops through the matched filter),
and the timing/carrier loops are **frozen** during the burst (coasting on the current state), re-locking afterwards.
The measured conclusion is equally honest: **the mechanism works but produces no frame-level gain** — RRC matched filtering widens a 20-symbol hole to roughly 36 symbols of effective damage,
beyond the 2t = 16 budget of RS(78,62); strong interference can also wipe out the preamble correlation entirely (only 0~2 frames of the whole record stay parseable).
That round pinned down the hard constraint: **effective damage must be ≲ 2t for erasures to be useful.**

```bash
python scripts/check_m12.py --part burst     # detection / arbitration / sync-side burst resistance / low-SNR window
python -m pytest tests/test_burst.py -q      # 9 tests (loop-freeze mechanism and the "do no harm" property)
```

See the [M12 acceptance report](docs/M12-验收报告.md).
<a id="web-workbench"></a>

### 13. Web workbench (11 tabs)

```bash
python -m spxh serve            # http://127.0.0.1:8760/ (loopback only by default)
```

| Tab | What you can see |
|---|---|
| Spectrum / waterfall | PSD, noise floor, occupied bandwidth, time-frequency waterfall (scroll to zoom, drag to pan) |
| Parameter estimation | Carrier / symbol rate / blind SNR + confidence + comparison with ground truth |
| Features | 23-dimensional feature groups, visualized |
| Recognition | Modulation, probability distribution, gating and OOD criteria |
| Demodulation | Lock indicators, synchronization traces, constellation, before/after decision comparison |
| Decoding | BER curves, soft/hard decision comparison, each coding scheme |
| Frame | Three-state frame-header CRC, payload preview, blind-recovery readout, correlation peaks |
| Spectral correlation | SSCA heat map + α/f profiles + peak table (spectral correlation vs M1 estimates side by side) |
| Report | LLM/offline analysis report: conclusion + **item-by-item evidence table** + uncertainties + suggestions + prompt review |
| Agent | Tool-call timeline (what was called each round, elapsed time, results), final conclusion, run summary, numeric grounding check |
| Settings | LLM configuration (provider presets, test connection, fetch model list, clear keys) + access token |

- **Every chart is interactive**: scroll to zoom (anchored at the mouse pointer), Shift for X only / Alt for Y only, drag to pan, double-click to reset;
- Per-request logs go to `models/web.log` by default (not to stdout); add `--verbose` for foreground debugging;
- Failure states in the page always carry explicit text and never sit silently at "loading".

**Add a token before exposing it**: `python -m spxh serve --token auto` (it prints the token; enter it on the Settings page).
Default policy: loopback only and no token → no authentication; **listening on a non-loopback address with no token → always rejected**.

### 14. Command line (15 subcommands)

```text
generate  dataset  info  spectrum  analyze  features  classify  demod
frame     fec      narrate  agent  llm      serve     mods
```

Every subcommand has `--help`, and most support `--json` (for script consumption).

---

## Measured data

The table below quotes only numbers **already written in the reports**, with their sources; rerunning the corresponding `scripts/check_*.py` reproduces them.

| Milestone | Metric | Measured | Source |
|---|---|---|---|
| M0 | EVM at 30 dB (BPSK/QPSK/8PSK/16QAM) | about 3.1% (pure-noise theoretical value 3.16%) | [M0 §4.1](docs/M0-验收报告.md) |
| M0 | BER at 10 dB: BPSK / QPSK / 8PSK / 16QAM / 2FSK / 4FSK | 0 / 0 / 3.1e-2 / 6.1e-2 / 2.3e-3 / 3.0e-2 | [M0 §4.1](docs/M0-验收报告.md) |
| M0 | Largest deviation of the measured noise variance from the Es/N0 definition | < 0.06 dB | [M0 §4.2](docs/M0-验收报告.md) |
| M1 | Worst carrier estimation error (SNR ≥ 10 dB) | 10.5 Hz (0.85% Rs) | [M1 §3.1](docs/M1-验收报告.md) |
| M1 | Worst symbol-rate relative error | 1.3e-4 (0.013%) | [M1 §3.1](docs/M1-验收报告.md) |
| M1 | Worst Es/N0 error | 0.16 dB | [M1 §3.1](docs/M1-验收报告.md) |
| M2 | Modulation recognition accuracy @20 dB / @10 dB | 1.000 / 0.900 | [M2 §3.4](docs/M2-验收报告.md) |
| M2 | Probability calibration ECE (SNR ≥ 5 dB) | 0.087 | [M2 §3.4](docs/M2-验收报告.md) |
| M2 | OOD detection rate | 0.594 | [M2 §3.4](docs/M2-验收报告.md) |
| M3 | QPSK: no-tracking baseline vs receive-chain BER | 4.95e-1 → 9.2e-4 (537× improvement) | [M3 §2.1](docs/M3-验收报告.md) |
| M3 | 16QAM: no-tracking baseline vs receive-chain BER | 4.21e-1 → 1.0e-3 (406×) | [M3 §2.1](docs/M3-验收报告.md) |
| M3 | LLR calibration (5 dB: expected/measured bit errors) | 69.5 / 69 (ratio 0.99) | [M3 §3.2](docs/M3-验收报告.md) |
| M4 | Gain of soft decision over hard decision | 2.92 dB @ BER=1e-4; 2.33 dB @ 1e-3 | [M4 §3.2](docs/M4-验收报告.md) |
| M4 | Value of interleaving for a 32-symbol burst | Uncorrectable without interleaving; with depth-5 symbol interleaving at most 7 errors per codeword, all corrected | [M4 §3.3](docs/M4-验收报告.md) |
| M5 | BPSK/QPSK 8-frame result from 15 dB up | 8/8 frames passing CRC with byte-exact payloads | [M5 §3.1](docs/M5-验收报告.md) |
| M6 | Coded link lowers the usable CRC threshold | uncoded 11.8 dB → convolutional (soft)/LDPC 7.8 dB (4.0 dB improvement) | [M6 §3.2](docs/M6-验收报告.md) |
| M6 | 32-bit burst: convolutional without vs with interleaving | 0/8 → 8/8 | [M6 §3.3](docs/M6-验收报告.md) |
| M7 | Smooth end-to-end threshold (16 frames re-measured per point) | uncoded 11.20 dB → CCSDS concatenated 6.90 dB (4.30 dB improvement) | [M7 §2](docs/M7-验收报告.md) |
| M7 | Blind FEC detection (frame-header fec field corrupted to 0) | conv / rs / ccsds / ldpc all recovered, payloads consistent | [M7 §3](docs/M7-验收报告.md) |
| M7 | RS erasure decoding (RS(76,60), t=8) | 9/12/16-byte bursts pass, 17 bytes honestly reported as uncorrectable (**re-run PASS**) | [M7 §4](docs/M7-验收报告.md) |
| M8 | RS errors-and-erasures joint decoding (2f+e ≤ 16) | 8 erasures + 4 errors recovered; 4 erasures + 7 errors (18>16) honestly reported as uncorrectable | [M8 §2](docs/M8-验收报告.md) |
| M8 | α-axis peak of SSCA spectral correlation | 1000.0 Hz (ground truth 1000 Hz; resolution 31.25 Hz) | [M8 §3](docs/M8-验收报告.md) |
| M8 | Case dataset size | 32 (six modulation types × four SNR levels + coded-link cases) | [M8 §4](docs/M8-验收报告.md) |
| M9 | Frame-header CRC-8 + blind length recovery | five coding schemes all recovered after the length field was corrupted | [M9 §2](docs/M9-验收报告.md) |
| M9 | Frame recall at 8 dB (16 frames, blind=1) | convolutional 5/16 → 11/16; CCSDS 4/16 → 12/16 | [M9 §2](docs/M9-验收报告.md) |
| M10 | Traceability of numbers in the offline report | all 26 numbers in the body traceable (`grounded: true`) | [M10 §5](docs/M10-验收报告.md) |
| M11 | Agent measured on real hardware (DeepSeek) | 3 rounds / 6 tool calls / 12.2 s, all conclusion numbers traceable | [M11 §2](docs/M11-验收报告.md) |
| M11 | Access-token policy | loopback without a token → no authentication; non-loopback without a token → always rejected | [M11 §3](docs/M11-验收报告.md) |
| M12 | Soft-preamble gain for coded-link recall (16 frames) | convolutional 6 dB 4/4 → **10/10**; 10 dB 13/13 → **16/16**; CCSDS 8 dB 12/10 → **14/11** | [M12 §2](docs/M12-验收报告.md) |
| M12 | Restrained erasure-decoding policy | under AWGN point-for-point identical to no erasure marking (threshold 8.78 dB), 0 automatic erasures; known-position bursts of 9/12/16 bytes all pass | [M12 §4](docs/M12-验收报告.md) |
| Global | Tests | 33 test files / 411 cases (`pytest --collect-only`, verified this round) | [open-source checklist](docs/开源检查清单.md) |

---

## Project layout

```text
SPXH/
├─ spxh/                         Main package (pip installable)
│  ├─ cli/main.py                15 subcommands: generate/dataset/info/spectrum/analyze/features/
│  │                             classify/demod/frame/fec/narrate/agent/llm/serve/mods
│  ├─ core/
│  │  ├─ types.py bits.py        Core types (Signal/FrameRecord/GroundTruth) and MSB-first bit packing
│  │  ├─ io/                     RAW / WAV / SigMF / npz multi-format read and write (M0)
│  │  ├─ synth/                  Six modulation types, RRC shaping, impairment injection, dataset generation (M0)
│  │  ├─ framing/                Barker-13, frame structure, CRC-16/CRC-8, frame-header hypothesis rewriting (M0/M5/M9)
│  │  ├─ dsp/                    Welch PSD, STFT waterfall (M1)
│  │  ├─ estimate/               Carrier / symbol rate / blind SNR estimation (M1)
│  │  ├─ features/               23-dimensional features across five domains (M2)
│  │  ├─ classify/               Binned random forest + probability calibration + gating + physical-rule fallback (M2)
│  │  ├─ demod/                  Timing loop + carrier loop + multi-scheme decisions + max-log LLR (M3)
│  │  ├─ fec/                    Interleaving / Viterbi / RS / CCSDS / LDPC / joint decoding (M4/M6/M8/M12)
│  │  ├─ framesync/              Preamble synchronization + phase ambiguity + payload extraction + blind recovery (M5/M6/M7/M9)
│  │  ├─ explain/                SSCA spectral correlation (M8)
│  │  ├─ report/                 LLM report layer, configurable LLM, DPAPI key encryption (M9/M10/M11)
│  │  └─ agent/                  Tool-calling agent: tool set + bounded loop (M11/M12)
│  ├─ web/                       Local web workbench (stdlib HTTP server + build-free frontend)
│  └─ tools/                     Ideal reference receiver, plotting utilities
├─ scripts/                      Acceptance self-checks: check_m0..m12.py; classifier training train_classifier.py
├─ tests/                        pytest: 33 files / 411 cases
├─ models/                       Classifier metadata + acceptance-report JSON (the 59 MB .joblib model is not shipped; llm_config.json / web.log ignored)
├─ data/demo/                    32 cases: SigMF metadata + data + ground-truth sidecar
├─ docs/                         Milestone acceptance reports, implementation plan, frontend docs, open-source checklist, figures
├─ .github/workflows/ci.yml      CI: pytest and acceptance scripts on both ubuntu and windows
├─ requirements.txt              pip install -r entry point (consistent with pyproject.toml)
└─ LICENSE (MIT) / CONTRIBUTING.md
```

## Acceptance scripts

Each script prints a human-readable table plus a `[PASS]/[FAIL]` list, and writes structured results to `models/check_*.json`;
**it returns 0 when everything passes and 1 when anything fails** (CI looks at the exit code).

| Script | What it proves | Rough runtime |
|---|---|---|
| `python scripts/check_m0.py` | Synthesis chain is correct: BER/EVM of the ideal reference receiver, frame CRC, reproducibility | seconds |
| `python scripts/check_m1.py` | Parameter-estimation accuracy + the "estimation error → EVM/BER" causal curve (with plots) | tens of seconds |
| `python scripts/check_m2.py` | Classification accuracy, gating behaviour, probability calibration (ECE), OOD detection | about 1 minute |
| `python scripts/check_m3.py` | BER for six modulation types, value of the synchronization loops over the no-tracking baseline, LLR calibration reconciliation | about 1 minute |
| `python scripts/check_m4.py` | BER curves for six coding schemes, soft-decision gain, RS correction limit, value of interleaving | about 3–5 minutes |
| `python scripts/check_m5.py` | Frame-synchronization rate, preamble phase-ambiguity resolution, byte-exact payload recovery, low-SNR CRC failure rate | tens of seconds |
| `python scripts/check_m6.py` | End-to-end coded-link CRC threshold and payload agreement rate, interleaving comparison experiment | about 2 minutes |
| `python scripts/check_m7.py --part curves` | Smooth end-to-end threshold curves (16–24 frames per point) | about 3.5 minutes |
| `python scripts/check_m7.py --part blind` | Blind FEC detection (four coding schemes, frame-header fec field lying) | seconds |
| `python scripts/check_m7.py --part erasure` | RS erasure decoding doubles the correction capability from t to 2t | seconds (**PASS**) |
| `python scripts/check_m12.py --part recall` | Soft-preamble gain for low-SNR frame recall (hard-preamble comparison) | about 3 minutes |
| `python scripts/check_m12.py --part erasure` | Joint decoding in the link: no AWGN degradation + no phantom erasures | about 1 minute |

## Design trade-offs / honest limits

This section collects the "known limitations" from the individual reports. They are not a to-do list but **the real capability boundary of the current version**:

- **8PSK / 16QAM (and QPSK around the 1e-3 level) have an error floor.** The cause is residual jitter in the synchronization loops, not noise:
  QPSK measures 9.2e-4 at 20 dB and 8PSK 5.4e-3 ([M3 §5](docs/M3-验收报告.md)). 16QAM is worse: after injecting a 0.35-symbol offset,
  30 dB still leaves 2125 bit errors and 15 dB leaves 2112, **unrelated to noise**, so it has been downgraded to a known limit on measured grounds ([M9 §3](docs/M9-验收报告.md)).
- **Low-SNR frame recall is improved by the "soft preamble", but it still has a ceiling.** After M12 changes the parsing stage from "all 13 bits must match" to a soft decision,
  of 16 frames, the convolutional link goes from 4 parseable frames to 10 at 6 dB, and reaches 16/16 at 10 dB. 5 dB is still hard (3 frames),
  because the preamble itself is only 6 symbols (QPSK) — that is simply how much information there is ([M12 §2](docs/M12-验收报告.md)).
- **"Cross-frame accumulation" is a path we tried and gave up.** The frame period is fractional in the symbol domain (581/2 = 290.5),
  and both blind estimators (|correlation| autocorrelation, peak-interval histogram) are unreliable for QPSK at low SNR,
  estimating 22/188/2385 (ground truth 581/290.5/1192.5). The conclusion is written in [M12 §3](docs/M12-验收报告.md):
  without extra priors, frame-period estimation is itself no easier than frame detection — the module was deleted rather than keeping a mechanism with no measured benefit.
- **The default erasure-decoding policy is "restrained".** With known positions the capability doubles (9–16-byte bursts all pass);
  but "mark an erasure when |LLR| is low" marks **correct** symbols under pure AWGN and makes things worse instead (8.78 → 9.20 dB).
  It now marks only **contiguous weak spans (≥3 symbols)**: under AWGN there are 0 automatic erasures and it is point-for-point identical to no erasure marking;
  what really needs it are **position-trustworthy** impairments such as shadowing and bursts ([M12 §4](docs/M12-验收报告.md)).
- **The M2 classifier is trained and tested on synthetic data only.** It is essentially unusable below 0 dB; OOD is insensitive to "the same family, just shifted parameters";
  the constellation features rely on the 0.35 roll-off assumption ([M2 §5](docs/M2-验收报告.md)).
- **M1's capability boundary**: carrier accuracy is about 1% Rs — enough for acquisition, not enough to support second-scale coherent processing (which is why M3 must have tracking loops);
  4FSK(h=0.5) gives an unreliable symbol rate at 10 dB, and the behaviour is to return low confidence rather than answer hard ([M1 §5](docs/M1-验收报告.md)).
- **The LLM only explains, never estimates.** All numbers come from the deterministic pipeline and carry `source/confidence`; reports must pass the "numeric grounding check";
  with no key it takes a structurally identical offline rendering ([M9 §4](docs/M9-验收报告.md), [M10 §5](docs/M10-验收报告.md)).
- **The settings interface listens on loopback only by default.** With loopback-only `127.0.0.1` and no token, there is no authentication (single user on the local machine);
  once it listens on a non-loopback address without a token → **always rejected**; external use requires `--token` ([M11 §3](docs/M11-验收报告.md)).
- **Key encryption is platform dependent.** On Windows it is encrypted at rest with DPAPI (bound to the current user), and old plaintext configs migrate automatically;
  other platforms fall back to plaintext and honestly label it `api_key_protected: false` instead of pretending it is encrypted ([M11 §4](docs/M11-验收报告.md)).
- **CCSDS and LDPC are not "standard codes".** CCSDS keeps the concatenated structure and the gain magnitude but does not implement the dual basis or randomization, so it cannot interoperate with real equipment;
  LDPC is a Gallager regular (1152,578) short code + normalized min-sum, with measured rank 574 and rate 0.5017 (reported as measured) ([M4 §5](docs/M4-验收报告.md)).
- **Blind detection relies on `payload_len` in the frame header.** The length field can be recovered by hypothesis search using the frame-header CRC-8 plus the frame CRC,
  but that is a "finite hypothesis space" search, not a general solution ([M7 §6](docs/M7-验收报告.md), [M9 §2](docs/M9-验收报告.md)).
- **Real channels are out of scope for validation.** All results come from synthetic signals (AWGN/frequency offset/phase noise/IQ imbalance/DC offset);
  multipath, nonlinearity and non-white-noise real recordings are not yet validated ([M2 §5](docs/M2-验收报告.md)).

> **One inconsistency that has already been fixed (kept on record so people trust that the numbers are checked)**:
> while preparing the open-source release, `check_m7.py --part erasure` failed on a re-run even though the report said it passed. It was located as
> **a stale assumption in the script itself**: it hard-coded `body_start = 13 + 32` (the 4-byte frame header from before M9),
> so once M9 changed the header to 5 bytes the injected burst landed in the wrong place. After computing it from the frame-format constants and re-running, it was **PASS**,
> and the report numbers and script behaviour agree again ([M12 §6](docs/M12-验收报告.md)).

## Roadmap

1. ~~End-to-end validation of burst detection + automatic erasure marking~~ **done (M12-C)**; ~~sync-side burst resistance~~ **also done (M13)**:
   the zeroing + loop-freeze mechanism works, but it produces no frame-level gain; the root cause is the hard constraint "matched-filter widening + 2t budget".
   **The next priority is cross-codeword interleaving + symbol-level deinterleaving**: spread the roughly 36 symbols of effective damage across 5 RS codewords,
   so each codeword carries only about 7 errors (within t=8) — currently the only direction that can genuinely raise burst capacity.
   The "interleaving × frame sync" interaction also needs investigating (in signal-level experiments the interleaved layout failed to parse at all under strong interference; not yet localized).
2. **A longer preamble as an optional wire format** (+13 bits/frame, about 3 dB of detection gain) — the soft preamble already covers the main loss,
   so this is an option "to squeeze out another 1–2 dB", not a requirement.
3. **Agent concurrency and more tools**: `batch` is currently serial (a bit slow over 12 cases); the plan is to make decoding, joint decoding
   and spectral-correlation comparison composable workflows as well.
4. **Multi-user / graded permissions for access tokens** (currently a single token, enough for one machine and small intranet use).
5. **A closed loop of LLM parameter advice and anomaly explanation**: feed low confidence, total CRC failure and gating fallback in as context,
   and output "what to tune next" that can be executed directly.

## FAQ

**Q: Why is there no classifier model after cloning?**
`models/modulation_rf_v1.joblib` is about 59 MB, above GitHub's recommended per-file size, so it is not committed. Run
`python scripts/train_classifier.py` once to rebuild it (about 1 minute). It works without it too — modulation recognition automatically falls back
to physical rules and marks the gate as `fallback`, rather than pretending to be an ML decision.

**Q: Can I run this without a GPU or a dataset?**
Yes. The core depends only on numpy + scipy and runs entirely on CPU; `data/demo/` ships with 32 cases,
and `python -m spxh generate` can create signals with arbitrary parameters and their ground truth on demand.

**Q: Does it have to be online? Does it have to have an LLM?**
Neither. Of the 14 features, only "LLM report" and "tool-calling agent" need a model; without an API key they take a
**structurally identical** offline deterministic path (the agent follows a fixed analyze→classify→frame→ssca plan).

**Q: Which modulations are supported? Why do 8PSK/16QAM perform poorly?**
BPSK / QPSK / 8PSK / 16QAM / 2FSK / 4FSK. The bottleneck for 8PSK/16QAM is an **error floor caused by residual jitter in the synchronization loops**
(not noise); it is quantified in [M3 §5](docs/M3-验收报告.md) and downgraded to a known limit. That is an honest boundary, not a bug swept under the rug.

**Q: Is my API key safe?**
It is stored only locally in `models/llm_config.json` (already in `.gitignore`), the interface **echoes only the last 4 characters**,
and on Windows it is stored as **DPAPI ciphertext** (bound to the current user), with old plaintext configs migrating automatically when loaded.
But note: no authentication by default assumes "serving this machine only"; **to expose it externally you must add `--token` first**.

**Q: Are these numbers trustworthy, or were they just written down after one good run?**
Every number is reproducible: `scripts/check_*.py` regenerates the tables and the JSON reports, and the exit code is the criterion (CI runs them too).
The numbers in the "Measured data" table are **checked automatically**: the script extracts the numbers from each row and looks each one up in the acceptance report linked from that row, with **25/25 rows hitting**;
the checking script itself ships with the repo (`python scripts/check_readme_numbers.py`, run by CI).

**Q: Can it connect to real equipment or real recordings?**
It can read all four recording formats — SigMF / WAV / RAW / npz — and the chain itself does not care where the data comes from.
But **all current validation was done on synthetic signals**; multipath, nonlinearity and non-white noise in real channels are not yet validated (see "honest limits").

## License and citation

This project is released under the **MIT License**, copyright holder `WmjXiaoJun`, year 2026; the full text is in [LICENSE](LICENSE).
You are free to use, modify and distribute it (including commercially), as long as you retain the copyright and license notice.

If SPXH helps your work, you can cite it as:

```bibtex
@misc{spxh2026,
  title        = {SPXH: Smart RF Signal Analysis Platform},
  author       = {{WmjXiaoJun}},
  year         = {2026},
  howpublished = {\url{https://github.com/WmjXiaoJun/SPXH}},
  note         = {An end-to-end RF signal analysis experimental platform from unknown I/Q to readable payload bits}
}
```

## Acknowledgements

SPXH is **an independent engineering implementation of publicly available signal-processing knowledge**; it copies no third-party project code.
It stands on methods that have long been public: Gardner timing-error detection and Farrow interpolation, decision-directed / Costas carrier loops,
Viterbi convolutional decoding, Reed-Solomon BM/Chien/Forney with errors-and-erasures joint decoding, LDPC normalized min-sum,
Barker sequences and their aperiodic autocorrelation, the CCSDS concatenated coding structure, SSCA strip spectral correlation, Welch power-spectrum estimation,
and classic machine-learning tools such as random forests and probability calibration.

Thanks to open-source projects including numpy / scipy / scikit-learn / matplotlib / pytest for the toolchain.
If you find any number in a report that **does not add up or cannot be reproduced**, please open an Issue — what this project values most is "no black boxes, no invented numbers".

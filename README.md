# SIH 26147 — Automated RF Analysis Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![DSP Engine](https://img.shields.io/badge/DSP-NumPy%20%7C%20SciPy-orange.svg)](https://scipy.org)
[![Format Standard](https://img.shields.io/badge/Format-SigMF%20%7C%20CF32%20%7C%20CI16%20%7C%20Planar-purple.svg)](https://github.com/gnuradio/SigMF)
[![Tests Passing](https://img.shields.io/badge/Tests-175%2F175%20Passing-brightgreen.svg)](#testing--verification)

> **Evidence-Backed Multi-Dataset Signal Parameter Extraction, Modulation Classification, Demodulation & Hypothesis Reasoning Engine**  
> *Ingests unknown `.IQ`, `.WAV`, SigMF, and binary archive signals, extracts physical RF parameters, evaluates candidate modulation and FEC decoding chains, aggregates findings into an ontological Evidence Graph, and outputs explainable, ranked transmission hypotheses with verifiable mathematical provenance.*

---

## 1. Executive Overview

The **SIH 26147 Automated RF Analysis Assistant** is a constrained-candidate hypothesis reasoning engine designed to bridge the gap between blind RF signal capture and explainable digital demodulation and decoding.

Unlike black-box neural networks that hallucinate modulation types on out-of-distribution signals, this engine enforces **physical domain verification** and mathematical transparency:

1. **Multi-Domain Ingestion:** Ingests raw hardware captures, astronomical telescope archives, and synthetic channel benchmarks without lossy coercion or parameter flattening.
2. **Spectral Detection & Estimation:** Computes Welch PSD, MAD noise floors, STFT spectrograms, occupied bandwidths (-3dB and 99% fractional), and multi-moment SNR (PSD integration and M2M4 ratio).
3. **Physical Characterization:** Extracts 22 scale-invariant physical features (including 4th-order cumulant $C_{42}$, phase symmetry $C_{21}$, and PAPR), estimates baud/symbol rate across 4 independent estimators, and performs carrier frequency offset (CFO) estimation.
4. **Calibrated Modulation Classification:** Evaluates candidate families (BPSK, QPSK, 16QAM, 2-FSK, 4-FSK, Noise/CW) with calibrated posterior scores and natural-language "WHY" justifications.
5. **DSP Synchronization & Slicing:** Performs CFO derotation, matched filtering, symbol timing recovery (Gardner TED / Farrow interpolator), 2nd-order Costas loop phase locking, nearest-neighbor constellation slicing, and Error Vector Magnitude (EVM) calculation.
6. **De-interleaving & FEC Decoding:** Evaluates candidate protocol decoding chains over a single canonical grid combining block de-interleaving, soft/hard Viterbi convolutional decoding ($k=7, r=1/2$), Reed-Solomon RS(255,223/239), and CRC-16 / CRC-32 integrity checks.
7. **Ontological Evidence Graph:** Assembles cross-stage measurements into a typed relational graph with support and contradiction edges, enforcing the *downstream-boost* invariant (successful decoding raises confidence in the upstream modulation hypothesis).
8. **Hypothesis Engine:** Performs bounded DAG search across valid $(\text{Modulation} \to \text{Demod} \to \text{De-interleave} \to \text{FEC} \to \text{Frame/CRC})$ chains, computing composite Bayesian confidence scores, uncertainty intervals, and explainable decision rationales.

---

## 2. Complete Data Flow Pipeline (Single View)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       SIH 26147 END-TO-END DATA FLOW PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  RAW INPUT   │  .IQ  .WAV  SigMF  .pkl  (CF32 / CI16 / CI8 / CU8 / PLANAR_FLOAT32)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: INGESTION & FORMAT AUTO-DETECTION (app/ingestion/)                                     │
│  • Auto-detect format: CF32_LE, CI16_LE, CI8, CU8, PLANAR_FLOAT32, WAV                          │
│  • Parse SigMF sidecars (.sigmf-meta) for native sample rate, center frequency, annotations      │
│  • Preserve all metadata in SignalMetadata — Zero parameter vanishing policy                     │
│  • Output: ComplexSignal (np.complex64 array + metadata + immutable provenance)                  │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: PREPROCESSING & NORMALIZATION (app/preprocessing/normalize.py)                         │
│  • DC offset removal (mean vector subtraction per channel)                                       │
│  • Unit power normalization (E[|s|^2] = 1.0) / peak scaling                                      │
│  • Optional anti-aliasing FIR low-pass filter and AGC                                             │
│  • Provenance recorded: stage="preprocessing", operations=["dc_removal", "normalize_power"]      │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: SPECTRAL DETECTION & BURST SEGMENTATION (app/detection/)                               │
│  • Welch PSD: Adaptive N_fft = min(1024, N_samples) for short bursts (e.g., RadioML 128-sample)  │
│  • STFT Spectrogram: Time-frequency matrix with adaptive temporal decimation                     │
│  • MAD Noise Floor: Robust Median Absolute Deviation baseline estimation (3 outlier iterations)  │
│  • Bandwidth Estimation: -3dB half-power bandwidth + 99% cumulative energy occupied bandwidth    │
│  • SNR Estimation: Spectral in-band integration (PSD) + time-domain moment ratio (M2M4)           │
│  • Energy Segmentation: Hysteresis energy thresholding to isolate active transmission bursts     │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: PHYSICAL FEATURE CHARACTERIZATION (app/characterization/)                              │
│  • 22 scale-invariant features: Higher-Order Cumulants (C42_norm, C21_sym, C40, C41),            │
│    PAPR, amplitude kurtosis/skewness, phase variance, Wiener spectral flatness                   │
│  • Symbol Rate Estimation: 4 methods (envelope, diff phase, squared PSD, power autocorrelation)  │
│  • Carrier Frequency Offset (CFO): 3 methods (M-th power, squaring, PSD peak)                    │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: MODULATION CLASSIFICATION & RANKING (app/modulation/classifier.py)                     │
│  • Candidates: BPSK, QPSK, 16QAM, 2-FSK, 4-FSK, Noise/CW                                        │
│  • Multi-hypothesis Bayesian evaluation using cumulant distances, constellation ring metrics     │
│  • Guardrails: Impulsive RFI detection, amplitude-variance tie discriminator (QPSK vs 16QAM)     │
│  • Output: Ranked candidates with posteriors and natural-language "WHY" evidence strings         │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: DSP SYNCHRONIZATION & CONSTELLATION SLICING (app/demodulation/)                        │
│  • Carrier recovery: CFO derotation + 2nd-order Decision-Directed phase lock (Costas loop)       │
│  • Timing recovery: Symbol strobing, cubic Farrow interpolation, Gardner TED tracking            │
│  • Slicing & Metrics: Nearest-neighbor constellation decisions, EVM (RMS % and dB), soft LLRs   │
│  • Guardrails: Short bursts (N < 256) and Noise/CW are safely GATED to prevent garbage output    │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 7: DE-INTERLEAVING & FEC PROTOCOL DECODING (app/decoding/)                                │
│  • Single canonical decode grid: build_decode_spec_grid() (shared across all consumers)          │
│  • De-interleaving: None (bypass) and Block(N, M) matrix de-interleaver                          │
│  • Viterbi FEC: Hard and soft (LLR) convolutional decoder (k=7, rate 1/2, polynomials 171/133)   │
│  • Reed-Solomon FEC: RS(255, 223) t=16 and RS(255, 239) t=8 algebraic decoder                   │
│  • Frame Sync & Checksum: Preamble cross-correlation (Barker-11/13, CCSDS), CRC-16, CRC-32      │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 8: EVIDENCE GRAPH AGGREGATION (app/evidence/)                                             │
│  • Assembles atomic EvidenceItems from detection, characterization, demodulation, and decoding   │
│  • Relational edges: SUPPORTS, REFUTES, DERIVES, CONTRADICTS with weighted scoring               │
│  • Downstream-Boost Property: Successful frame decoding directly boosts upstream modulation score │
│  • Serializes to JSON round-trip and GraphML (Gephi / yEd visualization)                         │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 9: HYPOTHESIS ENGINE & UNCERTAINTY QUANTIFICATION (app/hypothesis/)                       │
│  • Bounded DAG candidate search over (Modulation -> Demod -> Deinterleaver -> FEC -> Frame/CRC)  │
│  • SPS Seeding: Reuses the main pipeline's estimated symbol rate first to prevent search cut-off │
│  • Composite Scoring: Bayesian combination of modulation fit, demod EVM, and decoding integrity │
│  • Proximity scoring on CRC fail: Evaluates partial preamble/FEC matches, avoiding flat scoring  │
│  • Uncertainty quantification (standard deviation + 95% confidence intervals)                    │
│  • Explainability: Structured natural-language WHY reports detailing the winning hypothesis      │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 10: UNIFIED OUTPUTS & REPORTING                                                           │
│  • Machine-Readable: Complete JSON dump with full mathematical provenance                        │
│  • Human-Readable: Publication-grade vector PDF reports via Chrome Headless                      │
│  • REST API: 14 FastAPI endpoints (/api/analyze, /api/demodulate, /api/hypothesize, etc.)        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Multi-Dataset Modalities (Three-in-One Architecture)

The system is validated simultaneously across three fundamentally distinct operational RF domains:

| Dimension | Category A: `data0` (MIT RF Challenge) | Category B: `zenodo` (ATA Radio Astronomy) | Category C: `rml` (DeepSig RadioML 2016.10a) |
| :--- | :--- | :--- | :--- |
| **Physical Domain** | Hardware Over-The-Air ISM Transmissions | Continuous Deep-Space Telescope Archives | Synthetic Multipath & AWGN Channels |
| **Storage Format** | SigMF `cf32_le` Interleaved Binary | Raw Binary `ci16_le` Monolith (Sliced) | Python Pickle / NumPy Planar `float32` `(2, 128)` |
| **Sampling Rate ($f_s$)** | $25.0\text{ MHz}$ (20 MHz Channel) | $61.44\text{ MHz}$ (Observatory Wideband) | $1.0\text{ MHz}$ (Normalized Baseband) |
| **Center Freq ($f_c$)** | $2.437\text{ GHz}$ (ISM Wi-Fi Channel 6) | $622.96\text{ MHz}$ (UHF Band) / $1.4\text{ GHz}$ | $0.0\text{ Hz}$ (Analytic Complex Baseband) |
| **Slice Duration** | $40,960\text{ samples}$ ($1.638\text{ ms}$ / $327\text{ KB}$) | $40,960\text{ samples}$ ($66.7\ \mu\text{s}$ / $160\text{ KB}$) | $128\text{ samples}$ ($128\ \mu\text{s}$ / $1.02\text{ KB}$) |
| **Ground Truth** | Over-The-Air Modulation (`CommSignal2` = QPSK) | Observatory Pointing & Cosmic RFI | Explicit `(modulation, snr_db)` Key Tuples |
| **RAM Safety Policy** | Standard file-by-file loader | Memory-safe streaming slice (`slice_zenodo.py`) | Lazy-loaded streaming reader (`RMLDatasetReader`) |

---

## 4. Multi-Dataset Slice Test Benchmark (`slice-test/`)

The automated pipeline runner ([`scripts/generate_slice_reports.py`](scripts/generate_slice_reports.py)) passes singular slices from all three datasets through every layer of the pipeline, recording exact numbers and parameters, and compiling both Markdown and vector PDF reports into `slice-test/`:

| Pipeline Layer & Metric | `data0` (MIT Over-The-Air) | `rml` (RadioML Synthetic) | `zenodo` (ATA Astronomy RFI) |
| :--- | :--- | :--- | :--- |
| **Ingested Samples** | 4,096 samples | 128 samples | 4,096 samples |
| **Sampling Rate** | 25.00 MHz | 1.00 MHz | 61.44 MHz |
| **Center Frequency** | 2,437.00 MHz (ISM band) | 0.00 MHz (Baseband) | 622.96 MHz (UHF band) |
| **Slice Duration** | 163.8 $\mu$s | 128.0 $\mu$s | 66.7 $\mu$s |
| **Power Normalization** | $1.0000$ (from raw 1.3695) | $1.0000$ (from raw $7.95 \times 10^{-5}$) | $1.0000$ (from raw 0.9127) |
| **Spectral SNR (PSD)** | **19.74 dB** | **0.96 dB** (Fullband fallback) | **11.43 dB** |
| **Time SNR (M2M4)** | 2.46 dB | -0.62 dB | -8.18 dB (RFI mismatch flag) |
| **-3dB Bandwidth** | 268.55 kHz | 15.62 kHz | 120.00 kHz |
| **Occupied 99% BW** | 14.33 MHz (57.3% occupancy) | 468.75 kHz (46.9% occupancy) | 58.32 MHz (94.9% occupancy) |
| **Active Energy Bursts** | 5 bursts detected | 0 bursts (Continuous frame) | 0 bursts (Continuous RFI) |
| **Cumulant $C_{42}$** | 0.4366 (QAM/PSK region) | 0.1384 | 5.5678 (Extreme non-Gaussian) |
| **PAPR** | 6.15 dB | 7.75 dB | 15.95 dB (Impulsive spikes) |
| **Estimated Symbol Rate** | 1,562.50 kBaud ($SPS = 16.0$) | 62.99 kBaud ($SPS = 15.87$) | 5,505.00 kBaud ($SPS = 11.16$) |
| **Carrier Freq Offset** | $+67.14$ kHz | $+207.03$ kHz | $+10.38$ MHz |
| **Top Modulation** | **16QAM** (conf: 0.98, score: 0.47) | **Noise/CW** (conf: 0.98, score: 0.42) | **Noise/CW** (conf: 0.98, score: 0.45) |
| **Runner-Up Modulation** | QPSK (score: 0.32, conf: 0.71) | BPSK (score: 0.15) | BPSK (score: 0.12) |
| **Demodulation Status** | **LOCKED** (EVM: 29.91% RMS) | **GATED** (Safe, $N < 256$ samples) | **LOCKED** (EVM: 72.86%, unmodulated) |
| **Recovered Bits** | 1,012 bits (253 symbols) | 0 bits (Gated) | 730 bits (365 symbols) |
| **Frame / CRC Check** | FAILED / UNFRAMED | GATED | FAILED / UNFRAMED |
| **Evidence Graph** | 75 items, 11 hyp, 69 edges | 46 items, 6 hyp, 40 edges | 59 items, 11 hyp, 53 edges |
| **Winning Hypothesis** | `QPSK -> QPSK | sps=16.00 -> crc16` | `Noise/CW -> Noise/CW -> none` | `Noise/CW -> Noise/CW -> none` |
| **Composite Score** | 0.558 (uncertainty std: 0.500) | 0.206 (uncertainty std: 0.500) | 0.329 (uncertainty std: 0.500) |
| **Pipeline Latency** | **0.905 s** | **0.025 s** | **1.234 s** |
| **Generated Reports** | [`report.md`](slice-test/data0/report.md) & [`report.pdf`](slice-test/data0/report.pdf) | [`report.md`](slice-test/rml/report.md) & [`report.pdf`](slice-test/rml/report.pdf) | [`report.md`](slice-test/zenodo/report.md) & [`report.pdf`](slice-test/zenodo/report.pdf) |

---

## 5. Output Organization

The repository structures outputs into clean, categorized directories:

```
slice-test/
├── data0/               # MIT RF Challenge slice report (report.md, report.pdf)
├── rml/                 # RadioML 2016.10a slice report (report.md, report.pdf)
├── zenodo/              # ATA Telescope RFI slice report (report.md, report.pdf)
├── summary.md           # Cross-dataset comparative analysis markdown
└── summary.pdf          # Cross-dataset comparative vector PDF (Chrome Headless)

outputs/
├── data0/               # MIT RF Challenge artifacts (iq_sample.npz, metadata.json, psd.npz)
├── zenodo/              # ATA Telescope RFI artifacts (iq_sample.npz, metadata.json, psd.npz)
├── rml/                 # RadioML artifacts (iq_sample.npz, metadata.json, psd.npz)
├── classification/      # Batch classification JSON summaries & CSVs
└── figures/             # High-DPI (300 DPI) publication figures
```

---

## 6. Quickstart & Execution Guide

### Prerequisites
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 1. Run the Full Test Suite (175 Tests)
```bash
$env:PYTHONIOENCODING='utf-8'; pytest tests/
```

### 2. Generate Multi-Dataset Slice Reports & PDFs
Executes singular slices from all 3 datasets through every pipeline stage and compiles vector PDFs via Chrome Headless:
```bash
python scripts/generate_slice_reports.py
```

### 3. Launch the FastAPI REST Service
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### 4. Run Single-File Deep Dump Pipeline
```bash
python scripts/dump_single_file.py -f <signal_path> -o test
python scripts/make_result_pdf.py --json test/output.json -o test/result.pdf
```

---

## 7. Testing & Verification

The test suite enforces mathematical rigor and DSP correctness without external network dependencies:

```bash
pytest tests/
==================== 175 passed, 37 warnings in 25.29s ====================
```

| Test Module | Tests | Primary Focus & Verification Scope | Status |
| :--- | :---: | :--- | :---: |
| `tests/test_api_classification.py` | 10 | FastAPI REST endpoints (`/api/analyze`, `/api/modulation/classify`, `/api/demodulate`) | **PASSED** |
| `tests/test_dataset.py` | 12 | MIT RF Challenge dataset scanner, split validation, task grouping | **PASSED** |
| `tests/test_decoding.py` | 25 | Convolutional Viterbi (hard/soft), Reed-Solomon, de-interleavers, CRC check, canonical decode grid | **PASSED** |
| `tests/test_demodulation.py` | 7 | Costas loop, Gardner timing recovery, nearest-neighbor slicer, EVM, safe gating | **PASSED** |
| `tests/test_detection.py` | 18 | Welch PSD, MAD noise floor, energy segmentation, spectral SNR, M2M4 SNR, occupied bandwidth | **PASSED** |
| `tests/test_evidence.py` | 37 | EvidenceItem, HypothesisNode, SupportEdge, downstream-boost scoring, JSON/GraphML round-trip | **PASSED** |
| `tests/test_hypothesis.py` | 45 | Bounded DAG search, SPS seeding, shared decode grid, composite scoring, uncertainty quantification | **PASSED** |
| `tests/test_ingestion.py` | 12 | SigMF parser, binary auto-detection (CF32/CI16/CI8/CU8/WAV), Planar Float32 | **PASSED** |
| `tests/test_modulation_classifier.py` | 9 | Multi-candidate ranking (BPSK, QPSK, 16QAM, FSK, Noise/CW), RFI guardrail, tie-breaker | **PASSED** |
| **Total** | **175** | **Comprehensive Full-Pipeline Unit & Integration Coverage** | **100% GREEN** |

---

## 8. Implementation Status & Next Roadmap

- [x] **Multi-Dataset Ingestion:** `data0`, `zenodo`, and `rml` with zero parameter vanishing and safe streaming.
- [x] **Dynamic Spectral Detection:** Adaptive windowing, MAD noise floor, -3dB and 99% occupied bandwidth, dual-mode SNR.
- [x] **Physical Feature Characterization:** 22 scale-invariant features, 4x symbol rate estimators, 3x CFO estimators.
- [x] **Modulation Classification:** 6 candidate families, calibrated posteriors, explainable natural-language WHY generation.
- [x] **DSP Synchronization & Constellation Slicing:** CFO derotation, Costas loop phase lock, Gardner timing recovery, EVM.
- [x] **De-interleaving & FEC Decoding:** Block de-interleaving, hard/soft Viterbi, Reed-Solomon, CRC-16/32, canonical decode grid.
- [x] **Evidence Graph:** Cross-stage ontological graph, relational typed edges, downstream-boost property, GraphML export.
- [x] **Hypothesis Engine:** Bounded DAG candidate search, SPS seeding, composite Bayesian scoring, uncertainty quantification.
- [x] **Algorithmic Hardening (Phases 1 & 2):** Viterbi length protection, nested metadata parsing, RFI guardrails, F8 QPSK/16QAM tie discriminator, F9 block grid pruning, F10 measurement trust flags.
- [x] **Automated Reporting:** Per-stage multi-dataset pipeline reports and publication-ready vector PDFs in `slice-test/`.
- [ ] **[Next Priority] Phase 3 Interactive GUI / Frontend:**
  - WebGL waterfall/spectrogram viewer with zoom and energy segment overlays.
  - Interactive Welch PSD spectrum with dynamic noise floor and bandwidth markers.
  - Constellation scatter diagram with decision boundaries and EVM circles.
  - Candidate modulation leaderboard with expandable natural-language evidence trails.
  - Hypothesis Engine DAG visualization showing active transmission chains and composite scores.
  - One-click export center for JSON dumps and publication PDFs.

---

## 9. License & Citations

Developed as part of the **Smart India Hackathon (SIH 26147)**.  
Datasets utilized:
- **MIT Lincoln Laboratory AI Accelerator RF Challenge** (Single-Channel RF Signal Dataset).
- **Allen Telescope Array (ATA)** Terrestrial RFI Archives (Zenodo).
- **DeepSig Inc. RadioML 2016.10a** (GNU Radio Synthetic Channel Benchmark).

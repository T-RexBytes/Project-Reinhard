# SIH 26147 — Modulation Classification & RF Physical Analysis Showcase

**Publication-Grade RF Analysis & Hypothesis Reasoning Visualizations**  
*Generated at 300 DPI with calibrated posterior probabilities, constellation slicing EVM, and explainable "WHY" rationales.*

---

## 1. Dedicated Modulation Classification & EVM Verification Showcase

This figure provides a 4-signal deep dive into the reasoning engine across candidate families (**QPSK, 16QAM, Narrowband EMI/Tone, and ATA Deep Space RFI**):
- **Column 1:** Constellation density & recovered IQ points plotted against ideal reference lattices with EVM metrics.
- **Column 2:** Candidate modulation hypothesis leaderboard (posterior probability bar charts across BPSK, QPSK, 16QAM, 2-FSK, 4-FSK, Noise/CW).
- **Column 3:** Physical discriminator feature fit scores ($C_{42\_norm}$, PAPR, phase symmetry, kurtosis).
- **Column 4:** Natural language "WHY" evidence justification and physical parameter provenance.

![Candidate Modulation Classification & DSP Verification Showcase (QPSK, 16QAM, EMI, and Deep-Space RFI)](figures/modulation_classification_showcase.png)

---

## 2. MIT RF Challenge (`data0`) Supervised Benchmark Showcase

Visual inspection across `CommSignal2` (QPSK), `CommSignal3` (Wideband Burst), and `EMISignal1` (Narrowband Industrial EMI):
- **Column 1:** Time-domain In-Phase (I), Quadrature (Q), and $|Envelope|$ waveforms.
- **Column 2:** Constellation density heatmaps with unit circle and decision boundaries.
- **Column 3:** Welch PSD with MAD noise floor, $+10\text{ dB}$ detection threshold, and $-3\text{ dB}$ occupied bandwidth shading.
- **Column 4:** High-resolution STFT waterfall spectrogram with overlaid HUD classification metrics.

![MIT RF Challenge (data0) Multi-Domain Showcase with Classification HUD](figures/data0_rf_showcase.png)

---

## 3. Allen Telescope Array (`zenodo`) Deep-Space RFI Multi-Band Showcase

Spectra across 6 distinct radio frequency bands ($592\text{ MHz}$ to $2.33\text{ GHz}$) from the Allen Telescope Array, preserving native $61.44\text{ MHz}$ sampling rate, true physical noise floor ($-122.5\text{ dBFS}$), and RF center frequencies:
- Accurately classifies quiescent telescope sky scans as **`Noise/CW`** (honest uncertainty).
- Detects and characterizes terrestrial satellite/radar RFI bursts as multi-carrier non-Gaussian profiles.

![Allen Telescope Array (zenodo) Terrestrial RFI Showcase Across 6 RF Bands](figures/zenodo_rfi_showcase.png)

---

## 4. Executive RF Landscape & Hypothesis Distributions

A 6-panel executive cross-dataset comparative dashboard:
1. **Signal Space (SNR vs. Occupied Bandwidth):** Clustering signals across datasets.
2. **Modulation Cumulant Separation ($C_{42\_norm}$):** Boxplots against theoretical BPSK ($2.0$), QPSK ($1.0$), 16QAM ($0.68$), and FSK ($0.0$).
3. **Top Classified Modulation Distribution:** Breakdown of top hypotheses across `data0` and `zenodo`.
4. **Symbol Rate vs. -3dB Bandwidth:** Baud-rate consistency against Nyquist bounds.
5. **Classification Confidence vs. SNR:** Illustrating low-SNR uncertainty gating.
6. **Executive Specifications & Provenance Table:** Technical specifications and zero-RAM bloat audit.

![Executive RF Landscape Dashboard and Cross-Dataset Classification Comparison](figures/rf_landscape_executive.png)

---

## 5. RadioML 2016.10a Synthetic Channel Analysis & Archetype Showcase

Analysis of short-burst simulated transmissions (128 samples per burst) under multipath channel fading:
- **High-SNR Constellations:** Revealing distinct phase topologies across QPSK, BPSK, 16QAM, 8PSK, CPFSK, and WBFM.
- **Short-Burst Welch PSD:** Clamped 128-point Welch spectrum matching Nyquist roll-off and FM Carson bandwidths.
- **Higher-Order Cumulants:** Validating $C_{42}$ clustering despite short observation windows.
- **Parameter Scarcity Preservation:** Invariant tracking of `DataType.PLANAR_FLOAT32`, `dataset_origin="rml2016.10a"`, and labeled ground truth.

![RadioML 2016.10a Synthetic Channel Analysis and Parameter Showcase](figures/rml_rf_showcase.png)

---

## 6. Tri-Dataset Multi-Domain RF Intelligence Architecture & Invariant Preservation Landscape

A unified multi-domain architecture comparing `data0` (MIT hardware OTA), `zenodo` (ATA deep-space telescope archives), and `rml` (synthetic impaired channels):
- Demonstrating the SIH 26147 architecture's ability to ingest all three modalities without parameter flattening or memory spikes.
- Complete matrix comparing storage formats, sample rates, center frequencies, burst sizes, ground truth knowledge, and RAM safety policies.

![Tri-Dataset Multi-Domain RF Intelligence Architecture and Invariant Preservation Landscape](figures/tri_dataset_rf_landscape.png)


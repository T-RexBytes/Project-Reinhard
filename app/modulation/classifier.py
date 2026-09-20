"""
app/modulation/classifier.py
----------------------------
Candidate modulation classifier and hypothesis ranker.

Evaluates unknown complex IQ features against physically grounded modulation
families: BPSK, QPSK, 16QAM, 2-FSK, 4-FSK, and Noise/CW.

Produces ranked candidate hypotheses with calibrated scores, feature distance
metrics, and explainable natural-language "WHY" justifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from app.models import ComplexSignal, ProvenanceStep
from app.characterization.features import FeatureVector, extract_features
from app.characterization.symbol_rate import estimate_symbol_rate
from app.characterization.freq_offset import estimate_freq_offset
from app.detection.fft import compute_psd
from app.detection.noise_floor import estimate_noise_floor
from app.detection.snr import estimate_snr_from_psd
from app.detection.bandwidth import estimate_bandwidth


@dataclass
class ModulationCandidate:
    """A single ranked modulation hypothesis."""
    modulation: str
    score: float                # Calibrated posterior-like probability in [0, 1]
    confidence: float           # Confidence in this hypothesis in [0, 1]
    feature_fit: dict           # Per-feature fit scores
    why: str                    # Natural-language evidence justification

    def to_dict(self) -> dict:
        return {
            "modulation": self.modulation,
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "feature_fit": {k: round(float(v), 3) if isinstance(v, (int, float)) else v for k, v in self.feature_fit.items()},
            "why": self.why,
        }


@dataclass
class ModulationClassificationResult:
    """Complete modulation classification output with ranked candidates."""
    top_candidate: str
    confidence: float
    candidates: list[ModulationCandidate]
    snr_db: Optional[float] = None
    sample_rate: Optional[float] = None
    features_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "top_candidate": self.top_candidate,
            "confidence": round(float(self.confidence), 4),
            "candidates": [c.to_dict() for c in self.candidates],
            "snr_db": round(float(self.snr_db), 2) if self.snr_db is not None else None,
            "sample_rate": self.sample_rate,
            "features_summary": self.features_summary,
        }


class ModulationClassifier:
    """Multi-candidate hypothesis ranker for digital modulations.
    
    Discriminators used:
      - C42_norm (|C42| / C20^2): BPSK~2.0, QPSK~1.0, 16QAM~0.68, FSK~0.0
      - C21_symmetry (|C21| / C20): BPSK~1.0, QPSK~0.0, 16QAM~0.0
      - papr_db: Peak-to-Average Power Ratio in dB
      - amplitude_variance_norm: std(|IQ|) / mean(|IQ|)
      - kurtosis_amplitude: Sub-Gaussian (< -0.4) for single-shell PSK; >= -0.2 for 16QAM/multi-amplitude
      - phase_std: std(angle(IQ)) in [0, pi]
      - instantaneous_freq_std_norm: std(d(angle)/dt) / (2*pi) (dimensionless)
      - spectral_flatness: Wiener entropy in [0, 1]
      - bandwidth-to-symbol-rate consistency: BW / Rsym ~ 1+alpha for PSK/QAM
    """

    def __init__(self):
        pass

    def classify(
        self,
        fv: FeatureVector,
        snr_db: Optional[float] = None,
        symbol_rate_hz: Optional[float] = None,
        bandwidth_hz: Optional[float] = None,
        sample_rate: Optional[float] = None,
    ) -> ModulationClassificationResult:
        """Classify candidate modulations and return ranked hypotheses."""
        snr = snr_db if snr_db is not None else fv.snr_db_estimate
        if snr is None:
            snr = 15.0

        c42 = max(0.0, fv.C42_norm)
        papr = fv.papr_db
        amp_var = fv.amplitude_variance_norm
        kurt_amp = -1.2 if (fv.kurtosis_amplitude is None or math.isnan(fv.kurtosis_amplitude) or math.isinf(fv.kurtosis_amplitude)) else fv.kurtosis_amplitude
        phase_std = fv.phase_std
        c20 = max(fv.C20, 1e-12)
        c21_sym = fv.C21_abs / c20
        fsk_dev = fv.instantaneous_freq_std_norm
        flatness = fv.spectral_flatness

        # Effective tolerance scaling: low SNR expands tolerance bands
        snr_margin = max(0.0, (20.0 - snr) / 20.0) * 0.3
        # In low SNR (< 4 dB), digital modulations become indistinguishable from noise
        snr_weight = min(1.0, max(0.05, (snr - 1.0) / 9.0))
        # If cumulants or symmetry are strongly non-zero, the signal is physically non-Gaussian
        # (even if PSD had no out-of-band noise reference to estimate SNR accurately)
        has_digital_signature = (c42 > 0.45) or (c21_sym > 0.45)
        if has_digital_signature and snr < 4.0:
            snr_weight = max(snr_weight, 0.60)

        scores: dict[str, float] = {}
        fits: dict[str, dict] = {}
        whys: dict[str, str] = {}

        # ── Impulsive RFI / Interference Guardrail ────────────────────────────
        # When amplitude kurtosis > 3.0 or C42_norm > 3.0, the waveform is
        # physically incompatible with any digital lattice (BPSK/QPSK/16QAM).
        # Extreme statistics indicate impulsive cosmic noise, transient RFI, or
        # non-Gaussian interference.  Short-circuit all modulation scoring and
        # resolve directly to Noise/CW.
        if kurt_amp > 3.0 or c42 > 3.0:
            impulsive_why = (
                f"Excessive amplitude kurtosis ({kurt_amp:.2f}) or C42_norm ({c42:.2f}) exceeds "
                f"threshold (>3.0).  Impulsive RFI / interference rather than a digital lattice."
            )
            for mod in ("BPSK", "QPSK", "16QAM", "2-FSK", "4-FSK"):
                scores[mod] = 0.001
                fits[mod] = {}
                whys[mod] = impulsive_why
            scores["Noise/CW"] = 0.90
            fits["Noise/CW"] = {"impulsive_rfi": True, "kurt_amp": kurt_amp, "c42": c42}
            whys["Noise/CW"] = impulsive_why
        else:

            # ── 1. BPSK Scoring ──────────────────────────────────────────────────
            # BPSK: C42 ~ 2.0, 2-fold symmetry C21_sym ~ 1.0, low amp var, PAPR 1.5-5.5
            c42_fit_bpsk = math.exp(-0.5 * (abs(c42 - 2.0) / (0.6 + snr_margin)) ** 2)
            sym_fit_bpsk = math.exp(-0.5 * ((1.0 - c21_sym) / 0.4) ** 2) if c21_sym > 0.3 else 0.05
            papr_fit_bpsk = math.exp(-0.5 * (max(0.0, papr - 4.5) / 1.5) ** 2)
            amp_fit_bpsk = math.exp(-0.5 * (max(0.0, amp_var - 0.30) / 0.15) ** 2)
            kurt_fit_bpsk = 1.0 if kurt_amp < -0.3 else math.exp(-0.5 * ((kurt_amp + 0.3) / 0.4) ** 2)

            bpsk_score = (0.40 * c42_fit_bpsk + 0.30 * sym_fit_bpsk + 0.10 * papr_fit_bpsk + 0.10 * amp_fit_bpsk + 0.10 * kurt_fit_bpsk) * snr_weight
            whys["BPSK"] = (
                f"C42={c42:.2f} (BPSK target 2.0), 2-fold symmetry C21_sym={c21_sym:.2f} (target 1.0), "
                f"PAPR={papr:.1f} dB, amp_var={amp_var:.2f}."
            )
            scores["BPSK"] = max(0.001, bpsk_score)
            fits["BPSK"] = {"fit_c42": c42_fit_bpsk, "fit_sym": sym_fit_bpsk, "fit_papr": papr_fit_bpsk}

            # ── 2. QPSK Scoring ──────────────────────────────────────────────────
            # QPSK: C42 ~ 1.0 (typical 0.4-1.4 with pulse shaping & noise), C21_sym ~ 0 (4-fold symmetry),
            # single circular shell -> negative kurtosis (kurt_amp < -0.4)
            c42_fit_qpsk = math.exp(-0.5 * (abs(c42 - 0.85) / (0.50 + snr_margin)) ** 2)
            sym_fit_qpsk = math.exp(-0.5 * (c21_sym / 0.25) ** 2)  # must have near-zero C21 symmetry
            papr_fit_qpsk = 1.0 if 2.0 <= papr <= 7.0 else math.exp(-0.5 * (min(abs(papr - 2.0), abs(papr - 7.0)) / 2.0) ** 2)
            amp_fit_qpsk = 1.0 if amp_var <= 0.65 else math.exp(-0.5 * ((amp_var - 0.65) / 0.2) ** 2)
            kurt_fit_qpsk = 1.0 if kurt_amp <= -0.4 else math.exp(-0.5 * ((kurt_amp + 0.4) / 0.3) ** 2)

            qpsk_score = (0.35 * c42_fit_qpsk + 0.25 * sym_fit_qpsk + 0.15 * papr_fit_qpsk + 0.10 * amp_fit_qpsk + 0.15 * kurt_fit_qpsk) * snr_weight
            whys["QPSK"] = (
                f"C42={c42:.2f} (QPSK target ~1.0), 4-fold symmetry C21_sym={c21_sym:.3f} (<0.25), "
                f"single-shell amplitude kurtosis={kurt_amp:.2f} (< -0.40), PAPR={papr:.1f} dB."
            )
            scores["QPSK"] = max(0.001, qpsk_score)
            fits["QPSK"] = {"fit_c42": c42_fit_qpsk, "fit_sym": sym_fit_qpsk, "fit_kurt": kurt_fit_qpsk, "fit_papr": papr_fit_qpsk}

            # ── 3. 16QAM Scoring ─────────────────────────────────────────────────
            # 16QAM: C42 ~ 0.68, multi-amplitude grid (amp_var > 0.40, kurt_amp > -0.2 but <= 2.0),
            # PAPR 5.5-9.5, C21_sym ~ 0.  Upper bounds on kurtosis and C42 prevent false QAM
            # classification on impulsive interference (kurt_amp > 2.0 or C42 > 1.5 -> ruled out).
            c42_fit_qam = math.exp(-0.5 * (abs(c42 - 0.68) / (0.35 + snr_margin)) ** 2)
            sym_fit_qam = math.exp(-0.5 * (c21_sym / 0.25) ** 2)
            amp_fit_qam = 1.0 if amp_var >= 0.45 else math.exp(-0.5 * ((0.45 - amp_var) / 0.15) ** 2)
            papr_fit_qam = 1.0 if 5.5 <= papr <= 9.5 else math.exp(-0.5 * (min(abs(papr - 5.5), abs(papr - 9.5)) / 2.0) ** 2)
            kurt_fit_qam = 1.0 if -0.2 <= kurt_amp <= 2.0 else math.exp(-0.5 * (min(abs(kurt_amp + 0.2), abs(kurt_amp - 2.0)) / 0.25) ** 2)

            qam_score = (0.30 * c42_fit_qam + 0.20 * amp_fit_qam + 0.20 * papr_fit_qam + 0.15 * sym_fit_qam + 0.15 * kurt_fit_qam) * snr_weight
            whys["16QAM"] = (
                f"C42={c42:.2f} (16QAM target 0.68), multi-ring amplitude variance={amp_var:.2f}, "
                f"amplitude kurtosis={kurt_amp:.2f} (multi-level grid indicator, window -0.2..2.0), PAPR={papr:.1f} dB."
            )
            scores["16QAM"] = max(0.001, qam_score)
            fits["16QAM"] = {"fit_c42": c42_fit_qam, "fit_amp": amp_fit_qam, "fit_kurt": kurt_fit_qam, "fit_papr": papr_fit_qam}

            # ── 4. 2-FSK & 4-FSK Scoring ─────────────────────────────────────────
            # FSK: C42 must be near 0 (gated), constant envelope (PAPR < 3.2, amp_var < 0.28),
            # significant frequency deviation (fsk_dev > 0.02)
            c42_gate_fsk = math.exp(-0.5 * (c42 / 0.25) ** 2)  # drops rapidly if C42 > 0.3
            papr_fit_fsk = 1.0 if papr <= 3.0 else math.exp(-0.5 * ((papr - 3.0) / 1.5) ** 2)
            amp_fit_fsk = 1.0 if amp_var <= 0.25 else math.exp(-0.5 * ((amp_var - 0.25) / 0.15) ** 2)
            dev_fit_fsk2 = 1.0 if fsk_dev >= 0.03 else math.exp(-0.5 * ((0.03 - fsk_dev) / 0.02) ** 2)
            dev_fit_fsk4 = 1.0 if fsk_dev >= 0.05 else math.exp(-0.5 * ((0.05 - fsk_dev) / 0.025) ** 2)

            fsk2_score = c42_gate_fsk * (0.35 * papr_fit_fsk + 0.35 * amp_fit_fsk + 0.30 * dev_fit_fsk2) * snr_weight
            fsk4_score = c42_gate_fsk * (0.35 * papr_fit_fsk + 0.35 * amp_fit_fsk + 0.30 * dev_fit_fsk4) * snr_weight * 0.9

            scores["2-FSK"] = max(0.001, fsk2_score)
            scores["4-FSK"] = max(0.001, fsk4_score)
            fits["2-FSK"] = {"c42_gate": c42_gate_fsk, "fit_papr": papr_fit_fsk, "fit_dev": dev_fit_fsk2}
            fits["4-FSK"] = {"c42_gate": c42_gate_fsk, "fit_papr": papr_fit_fsk, "fit_dev": dev_fit_fsk4}
            whys["2-FSK"] = (
                f"Constant envelope PAPR={papr:.1f} dB, amp_var={amp_var:.2f}, C42={c42:.2f} (gate={c42_gate_fsk:.2f}), "
                f"frequency deviation norm={fsk_dev:.4f}."
            )
            whys["4-FSK"] = (
                f"Multi-tone FSK profile, PAPR={papr:.1f} dB, amp_var={amp_var:.2f}, C42={c42:.2f}, "
                f"frequency deviation norm={fsk_dev:.4f}."
            )

            # ── 5. Noise / CW Scoring ────────────────────────────────────────────
            is_low_snr = 1.0 if snr < 3.0 else math.exp(-0.5 * ((snr - 3.0) / 4.0) ** 2)
            if has_digital_signature:
                is_low_snr *= 0.15
            snr_noise_gate = math.exp(-0.5 * (max(0.0, snr - 3.0) / 5.0) ** 2)
            is_flat_noise = (1.0 if flatness > 0.70 else math.exp(-0.5 * ((0.70 - flatness) / 0.15) ** 2)) * snr_noise_gate
            noise_score = max(is_low_snr * 0.85, is_flat_noise * 0.70)
            scores["Noise/CW"] = max(0.001, noise_score)
            fits["Noise/CW"] = {"is_low_snr": is_low_snr, "is_flat_noise": is_flat_noise}
            whys["Noise/CW"] = (
                f"SNR={snr:.1f} dB ({'low SNR regime' if snr < 4.0 else 'moderate SNR'}), "
                f"spectral flatness={flatness:.3f}."
            )

            # ── Bandwidth & Baud Rate Consistency Boost ──────────────────────────
            if symbol_rate_hz and bandwidth_hz and symbol_rate_hz > 0:
                bw_ratio = bandwidth_hz / symbol_rate_hz
                # For linear modulations (BPSK, QPSK, 16QAM), BW/Rsym is typically 1.0 to 1.8
                if 0.8 <= bw_ratio <= 1.9:
                    scores["BPSK"] *= 1.20
                    scores["QPSK"] *= 1.35
                    scores["16QAM"] *= 1.20
                    whys["QPSK"] += f" Bandwidth/symbol_rate ratio {bw_ratio:.2f} confirms linear root-raised-cosine modulation."
                    whys["BPSK"] += f" Bandwidth/symbol_rate ratio {bw_ratio:.2f} aligns with Nyquist pulse shaping."
                elif bw_ratio > 2.5 and fsk_dev > 0.02:
                    scores["2-FSK"] *= 1.30
                    scores["4-FSK"] *= 1.35
                    whys["2-FSK"] += f" Carson bandwidth ratio {bw_ratio:.2f} confirms wideband FM deviation."

            # ── F8: QPSK vs 16-QAM Tie Discriminator ─────────────────────────────
            # When the two scores are within 0.10, use amplitude variance as a
            # single-shell vs multi-shell ring discriminator to break the tie.
            # QPSK: single constant-amplitude shell -> amp_var typically < 0.20.
            # 16QAM: multi-amplitude grid -> amp_var typically > 0.40.
            # When the two candidates are effectively tied (gap < 0.10), this
            # second-pass discriminator prevents the classifier from producing an
            # inconclusive near-tie and explicitly states the ring evidence.
            qpsk_s = scores.get("QPSK", 0.0)
            qam_s  = scores.get("16QAM", 0.0)
            if abs(qpsk_s - qam_s) < 0.10 * max(qpsk_s, qam_s, 1e-9):
                # Amplitude variance discriminator (inner vs outer shell energy)
                if amp_var < 0.20:
                    # Single-shell: boost QPSK
                    scores["QPSK"]  *= 1.25
                    scores["16QAM"] *= 0.80
                    tie_note = (
                        f" [TIE-BREAK F8: amp_var={amp_var:.3f} < 0.20 -> single-shell ring, QPSK preferred over 16QAM]"
                    )
                    whys["QPSK"]  += tie_note
                    whys["16QAM"] += f" [TIE-BREAK F8: amp_var={amp_var:.3f} < 0.20 -> single-shell ring favors QPSK]"
                elif amp_var > 0.40:
                    # Multi-shell: boost 16QAM
                    scores["16QAM"] *= 1.25
                    scores["QPSK"]  *= 0.80
                    tie_note = (
                        f" [TIE-BREAK F8: amp_var={amp_var:.3f} > 0.40 -> multi-shell ring, 16QAM preferred over QPSK]"
                    )
                    whys["16QAM"] += tie_note
                    whys["QPSK"]  += f" [TIE-BREAK F8: amp_var={amp_var:.3f} > 0.40 -> multi-shell ring favors 16QAM]"
                else:
                    # Inconclusive tie — flag explicitly in both WHY strings
                    tie_note = (
                        f" [TIE F8: QPSK/16QAM scores within 10%%; amp_var={amp_var:.3f} is ambiguous (0.20-0.40);"
                        f" C42={c42:.2f}; result is genuinely inconclusive at this SNR]"
                    )
                    whys["QPSK"]  += tie_note
                    whys["16QAM"] += tie_note

        # ── Normalize to probabilities ───────────────────────────────────────
        total_score = sum(scores.values()) + 1e-12
        probs = {k: v / total_score for k, v in scores.items()}

        candidates: list[ModulationCandidate] = []
        for mod, prob in sorted(probs.items(), key=lambda item: item[1], reverse=True):
            conf = float(np.clip(scores[mod], 0.05, 0.95))
            candidates.append(
                ModulationCandidate(
                    modulation=mod,
                    score=float(prob),
                    confidence=conf,
                    feature_fit=fits[mod],
                    why=whys[mod],
                )
            )

        top = candidates[0]
        margin = top.score - candidates[1].score if len(candidates) > 1 else top.score
        top_conf = float(np.clip(top.confidence * (1.0 + margin), 0.15, 0.98))

        return ModulationClassificationResult(
            top_candidate=top.modulation,
            confidence=top_conf,
            candidates=candidates,
            snr_db=snr,
            sample_rate=sample_rate or fv.sample_rate,
            features_summary={
                "C42_norm": round(float(c42), 3),
                "C21_sym": round(float(c21_sym), 3),
                "papr_db": round(float(papr), 2),
                "amp_var": round(float(amp_var), 3),
                "kurt_amp": round(float(kurt_amp), 3),
                "phase_std": round(float(phase_std), 3),
                "inst_freq_std_norm": round(float(fsk_dev), 5),
                "spectral_flatness": round(float(flatness), 4),
            },
        )

    def classify_signal(
        self,
        signal: ComplexSignal,
        max_samples: int = 65536,
    ) -> ModulationClassificationResult:
        """Classify modulation for a ComplexSignal object, preserving provenance.
        
        Extracts features, estimates symbol rate, carrier frequency offset,
        PSD-based SNR and bandwidth, and runs the multi-candidate ranker.
        Appends a ProvenanceStep to signal.provenance.
        """
        sr = signal.metadata.sample_rate or 1.0
        n_samples = min(len(signal.data), max_samples)
        iq_sub = signal.data[:n_samples]

        # Feature extraction
        fv = extract_features(iq_sub, sample_rate=sr)

        # Coarse symbol rate estimate
        sym_res = estimate_symbol_rate(iq_sub, sample_rate=sr)
        sym_rate = sym_res.symbol_rate_hz if sym_res.confidence > 0.3 else None

        # PSD, noise floor, SNR, and occupied bandwidth
        snr_val = fv.snr_db_estimate
        bw_val = None
        try:
            fft_n = min(1024, len(signal.data))
            if fft_n >= 64:
                psd = compute_psd(signal, fft_size=fft_n)
                nf = estimate_noise_floor(psd, method="mad")
                snr_res = estimate_snr_from_psd(psd, nf)
                snr_val = snr_res.snr_db
                bw_res = estimate_bandwidth(psd, method="3db")
                bw_val = bw_res.bandwidth_hz
        except Exception:
            pass

        # Run classification (initial pass — no mod hint yet)
        res = self.classify(
            fv=fv,
            snr_db=snr_val,
            symbol_rate_hz=sym_rate,
            bandwidth_hz=bw_val,
            sample_rate=sr,
        )

        # F2: Re-estimate carrier frequency offset with the modulation hint from
        # the classifier result so the correct algorithm is selected (fourth_power
        # for QPSK/QAM, squared for BPSK, psd_peak for FSK/CW).
        try:
            cfo_res = estimate_freq_offset(iq_sub, sample_rate=sr, mod_hint=res.top_candidate)
            signal.add_provenance(
                stage="characterization",
                operation="estimate_freq_offset",
                parameters={
                    "offset_hz": cfo_res.offset_hz,
                    "confidence": cfo_res.confidence,
                    "method": cfo_res.method,
                    "mod_hint": res.top_candidate,
                    "notes": cfo_res.notes,
                },
            )
        except Exception:
            pass  # CFO estimation is advisory; do not abort classification

        # Append provenance step
        signal.add_provenance(
            stage="modulation_classification",
            operation="classify_candidates",
            parameters={
                "top_candidate": res.top_candidate,
                "confidence": res.confidence,
                "snr_db": res.snr_db,
                "num_candidates": len(res.candidates),
            },
        )

        return res

    def classify_iq(
        self,
        iq: np.ndarray,
        sample_rate: float = 1.0,
        snr_db: Optional[float] = None,
        max_samples: int = 65536,
    ) -> ModulationClassificationResult:
        """Convenience method to classify a raw complex IQ numpy array."""
        n_samples = min(len(iq), max_samples)
        iq_sub = np.asarray(iq[:n_samples], dtype=np.complex64)
        fv = extract_features(iq_sub, sample_rate=sample_rate)
        sym_res = estimate_symbol_rate(iq_sub, sample_rate=sample_rate)
        sym_rate = sym_res.symbol_rate_hz if sym_res.confidence > 0.3 else None

        return self.classify(
            fv=fv,
            snr_db=snr_db if snr_db is not None else fv.snr_db_estimate,
            symbol_rate_hz=sym_rate,
            sample_rate=sample_rate,
        )


_DEFAULT_CLASSIFIER = ModulationClassifier()


def classify_signal(signal: ComplexSignal, max_samples: int = 65536) -> ModulationClassificationResult:
    """Classify a ComplexSignal using the default ModulationClassifier instance."""
    return _DEFAULT_CLASSIFIER.classify_signal(signal, max_samples=max_samples)


def classify_iq(
    iq: np.ndarray,
    sample_rate: float = 1.0,
    snr_db: Optional[float] = None,
    max_samples: int = 65536,
) -> ModulationClassificationResult:
    """Classify a raw complex IQ numpy array using the default ModulationClassifier instance."""
    return _DEFAULT_CLASSIFIER.classify_iq(iq, sample_rate=sample_rate, snr_db=snr_db, max_samples=max_samples)


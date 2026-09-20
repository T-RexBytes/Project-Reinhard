"""FastAPI application for RF Analysis Assistant."""

from __future__ import annotations

import tempfile
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.models import DataType, IQOrdering
from app.ingestion.parser import load_signal
from app.preprocessing.normalize import full_normalize
from app.detection.fft import compute_psd
from app.detection.spectrogram import compute_spectrogram
from app.detection.noise_floor import estimate_noise_floor
from app.detection.segmentation import build_segments
from app.detection.snr import estimate_snr_from_psd
from app.detection.bandwidth import estimate_bandwidth
from app.characterization.features import extract_features
from app.characterization.symbol_rate import estimate_symbol_rate
from app.characterization.freq_offset import estimate_freq_offset
from app.modulation.classifier import classify_signal, ModulationClassifier
from app.demodulation.slicer import demodulate_signal
from app.decoding.decoder import search_decode
from app.evidence import build_evidence_graph, graph_to_json
from app.hypothesis import (
    HypothesisEngineConfig,
    run_hypothesis_search,
    generate_chain_json,
)
from app.dataset.loader import DatasetScanner, TaskType, SignalType, SplitType
from app.api.hitl import hitl_router

app = FastAPI(title="RF Analysis Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hitl_router)


class SignalInfo(BaseModel):
    filename: str
    data_type: str
    sample_rate: Optional[float]
    center_frequency: Optional[float]
    num_samples: int
    duration_seconds: float
    provenance: list[dict]


class NormalizeRequest(BaseModel):
    remove_dc: bool = True
    normalize_to: str = "power"
    apply_lpf: bool = False
    lpf_cutoff: float = 0.9


@app.post("/api/upload", response_model=SignalInfo)
async def upload_signal(
    file: UploadFile = File(...),
    data_type: Optional[str] = Query(None, description="Override data type: cf32, ci16, ci8, cu8"),
    sample_rate: Optional[float] = Query(None, description="Sample rate in Hz"),
):
    """Upload and parse an IQ or WAV file."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".iq", ".wav", ".cf32", ".c32", ".ci16", ".i16", ".ci8", ".i8", ".cu8", ".u8"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dt = DataType(data_type) if data_type else None
        signal = load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)
        return SignalInfo(
            filename=signal.metadata.filename,
            data_type=signal.metadata.data_type.value,
            sample_rate=signal.metadata.sample_rate,
            center_frequency=signal.metadata.center_frequency,
            num_samples=signal.num_samples,
            duration_seconds=signal.duration_seconds or 0.0,
            provenance=[p.to_dict() for p in signal.provenance],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/normalize")
async def normalize_signal(
    file: UploadFile = File(...),
    remove_dc: bool = True,
    normalize_to: str = "power",
    apply_lpf: bool = False,
    lpf_cutoff: float = 0.9,
    data_type: Optional[str] = Query(None),
    sample_rate: Optional[float] = Query(None),
):
    """Upload, parse, and normalize a signal."""
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dt = DataType(data_type) if data_type else None
        signal = load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)
        signal = full_normalize(
            signal,
            remove_dc_offset=remove_dc,
            normalize_to=normalize_to,
            apply_lpf=apply_lpf,
            lpf_cutoff=lpf_cutoff,
        )

        return {
            "metadata": signal.metadata.to_dict(),
            "provenance": [p.to_dict() for p in signal.provenance],
            "stats": {
                "mean_power": float(np.mean(np.abs(signal.data) ** 2)),
                "peak_amplitude": float(np.max(np.abs(signal.data))),
                "mean_i": float(np.mean(signal.data.real)),
                "mean_q": float(np.mean(signal.data.imag)),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_signal(
    file: UploadFile = File(...),
    fft_size: int = Query(1024),
    data_type: Optional[str] = Query(None),
    sample_rate: Optional[float] = Query(None),
):
    """Run full detection pipeline on uploaded signal.

    Returns PSD info, noise floor, segments, SNR, and bandwidth.
    """
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dt = DataType(data_type) if data_type else None
        signal = load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)

        if signal.metadata.sample_rate is None:
            raise HTTPException(status_code=400, detail="sample_rate required for analysis")

        psd = compute_psd(signal, fft_size=fft_size)
        noise_floor = estimate_noise_floor(psd, method="mad")
        segmentation = build_segments(psd, noise_floor, signal)
        snr = estimate_snr_from_psd(psd, noise_floor)
        bandwidth = estimate_bandwidth(psd, method="3db")

        # Extract features and estimate baud/carrier parameters
        fv = extract_features(signal.data[:65536], sample_rate=signal.metadata.sample_rate)
        sym_res = estimate_symbol_rate(signal.data[:65536], sample_rate=signal.metadata.sample_rate)
        cfo_res = estimate_freq_offset(signal.data[:16384], sample_rate=signal.metadata.sample_rate)

        # Classify modulation candidates and rank hypotheses
        mod_result = classify_signal(signal, max_samples=65536)

        return {
            "metadata": signal.metadata.to_dict(),
            "psd": psd.to_dict(),
            "noise_floor": noise_floor.to_dict(),
            "snr": snr.to_dict(),
            "bandwidth": bandwidth.to_dict(),
            "segments": segmentation.to_dict(),
            "characterization": {
                "features": fv.to_dict(),
                "symbol_rate": sym_res.to_dict(),
                "freq_offset": cfo_res.to_dict(),
            },
            "modulation_classification": mod_result.to_dict(),
            "provenance": [p.to_dict() for p in signal.provenance],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/characterize")
async def characterize_signal(
    file: UploadFile = File(...),
    data_type: Optional[str] = Query(None),
    sample_rate: Optional[float] = Query(None),
):
    """Extract physical and dimensionless modulation features from uploaded signal."""
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dt = DataType(data_type) if data_type else None
        signal = load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)
        sr = signal.metadata.sample_rate or 1.0

        fv = extract_features(signal.data[:65536], sample_rate=sr)
        sym = estimate_symbol_rate(signal.data[:65536], sample_rate=sr)
        cfo = estimate_freq_offset(signal.data[:16384], sample_rate=sr)

        return {
            "metadata": signal.metadata.to_dict(),
            "features": fv.to_dict(),
            "symbol_rate": sym.to_dict(),
            "freq_offset": cfo.to_dict(),
            "provenance": [p.to_dict() for p in signal.provenance],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/modulation/classify")
async def classify_modulation_endpoint(
    file: UploadFile = File(...),
    data_type: Optional[str] = Query(None),
    sample_rate: Optional[float] = Query(None),
):
    """Classify candidate modulations for an uploaded signal and rank hypotheses."""
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dt = DataType(data_type) if data_type else None
        signal = load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)
        if signal.metadata.sample_rate is None:
            signal.metadata.sample_rate = 1.0

        mod_result = classify_signal(signal)
        return {
            "metadata": signal.metadata.to_dict(),
            "classification": mod_result.to_dict(),
            "provenance": [p.to_dict() for p in signal.provenance],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/demodulate")
async def demodulate_endpoint(
    file: UploadFile = File(...),
    modulation: Optional[str] = Query(None, description="Modulation: BPSK, QPSK, 16QAM, etc. If omitted, auto-classified."),
    samples_per_symbol: Optional[float] = Query(None, description="Samples per symbol override"),
    data_type: Optional[str] = Query(None),
    sample_rate: Optional[float] = Query(None),
):
    """Demodulate, synchronize, and calculate EVM for an uploaded signal."""
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dt = DataType(data_type) if data_type else None
        signal = load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)
        if signal.metadata.sample_rate is None:
            signal.metadata.sample_rate = 1.0

        demod_result = demodulate_signal(
            signal,
            modulation=modulation,
            samples_per_symbol=samples_per_symbol,
            max_symbols=1024,
        )
        return {
            "metadata": signal.metadata.to_dict(),
            "demodulation": demod_result.to_dict(),
            "provenance": [p.to_dict() for p in signal.provenance],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)



@app.post("/api/waterfall")
async def waterfall(
    file: UploadFile = File(...),
    fft_size: int = Query(512),
    max_time_bins: int = Query(512),
    data_type: Optional[str] = Query(None),
    sample_rate: Optional[float] = Query(None),
):
    """Compute spectrogram/waterfall for visualization."""
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dt = DataType(data_type) if data_type else None
        signal = load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)

        if signal.metadata.sample_rate is None:
            raise HTTPException(status_code=400, detail="sample_rate required for waterfall")

        spec = compute_spectrogram(
            signal, fft_size=fft_size, max_time_bins=max_time_bins
        )

        return {
            "metadata": spec.to_dict(),
            "spectrogram_db": spec.spectrogram_db.tolist(),
            "frequencies_hz": spec.frequencies.tolist(),
            "time_bins_seconds": spec.time_bins.tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/evidence")
async def evidence_graph_endpoint(
    file: UploadFile = File(...),
    data_type: Optional[str] = Query(None),
    sample_rate: Optional[float] = Query(None),
    run_demod: bool = Query(True, description="Include demodulation verification evidence"),
    run_decode: bool = Query(True, description="Include decode-chain evidence"),
):
    """Run the full pipeline and return the scored cross-stage Evidence Graph.

    Aggregates detection facts, characterization features (cross-linked to the
    modulation families they discriminate), classifier candidate scores,
    demodulation EVM / sync quality, and decode-chain CRC validity into a
    weighted, provenance-linked evidence graph. Downstream success (low EVM,
    CRC-valid frame) raises the upstream modulation family's score.
    """
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dt = DataType(data_type) if data_type else None
        signal = load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)
        if signal.metadata.sample_rate is None:
            raise HTTPException(status_code=400,
                                detail="sample_rate required for evidence analysis")
        sr = signal.metadata.sample_rate
        signal = full_normalize(signal, remove_dc_offset=True, normalize_to="power")

        psd = compute_psd(signal, fft_size=1024)
        noise_floor = estimate_noise_floor(psd, method="mad")
        segmentation = build_segments(psd, noise_floor, signal)
        snr = estimate_snr_from_psd(psd, noise_floor)
        bandwidth = estimate_bandwidth(psd, method="3db")

        fv = extract_features(signal.data[:65536], sample_rate=sr)
        sym_res = estimate_symbol_rate(signal.data[:65536], sample_rate=sr)
        cfo_res = estimate_freq_offset(signal.data[:16384], sample_rate=sr)

        mod_result = classify_signal(signal, max_samples=65536)

        demod_result = None
        if run_demod:
            demod_result = demodulate_signal(
                signal, modulation=mod_result.top_candidate, max_symbols=1024)

        decoding = []
        if run_decode and demod_result is not None and demod_result.bits:
            decoding = search_decode(
                demod_result.bits,
                llrs=(demod_result.llrs_sample or None),
                use_soft_viterbi=True, top_k=10,
            )

        graph = build_evidence_graph(
            snr=snr, bandwidth=bandwidth, noise_floor=noise_floor,
            segmentation=segmentation, features=fv, symbol_rate=sym_res,
            freq_offset=cfo_res, mod_classification=mod_result,
            demodulation=demod_result, decoding=decoding,
            metadata={
                "filename": file.filename,
                "sample_rate": sr,
                "center_frequency": signal.metadata.center_frequency,
            },
        )

        return {
            "metadata": signal.metadata.to_dict(),
            "evidence_graph": graph.to_dict(),
            "config": {"run_demod": run_demod, "run_decode": run_decode},
            "provenance": [p.to_dict() for p in signal.provenance],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/decode")
async def decode_endpoint(
    file: UploadFile = File(...),
    modulation: Optional[str] = Query(None, description="Modulation override; auto-classified if omitted"),
    samples_per_symbol: Optional[float] = Query(None, description="Samples-per-symbol override"),
    data_type: Optional[str] = Query(None),
    sample_rate: Optional[float] = Query(None),
):
    """Demodulate an uploaded signal and run the constrained decode-chain search.

    Returns the ranked decode-chain leaderboard (Viterbi / RS / CRC frame
    candidates) with per-chain residual statistics and provenance.
    """
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dt = DataType(data_type) if data_type else None
        signal = load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)
        if signal.metadata.sample_rate is None:
            signal.metadata.sample_rate = 1.0

        if modulation is None:
            mod_result = classify_signal(signal, max_samples=65536)
            modulation = mod_result.top_candidate

        demod_result = demodulate_signal(
            signal,
            modulation=modulation,
            samples_per_symbol=samples_per_symbol,
            max_symbols=1024,
        )

        decoding = []
        if demod_result.bits:
            decoding = search_decode(
                demod_result.bits,
                llrs=(demod_result.llrs_sample or None),
                use_soft_viterbi=True, top_k=10,
            )

        return {
            "metadata": signal.metadata.to_dict(),
            "modulation": modulation,
            "demodulation": demod_result.to_dict(),
            "decode_leaderboard": [r.to_dict() for r in decoding],
            "provenance": [p.to_dict() for p in signal.provenance],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/hypothesize")
async def hypothesize_endpoint(
    file: UploadFile = File(...),
    data_type: Optional[str] = Query(None),
    sample_rate: Optional[float] = Query(None),
    top_modulations: int = Query(3, ge=1, le=6),
    max_chains: int = Query(200, ge=1, le=2000),
):
    """Run the hypothesis engine: bounded DAG search + live scoring + WHY.

    Expands the top-N modulation candidates into a bounded (Modulation ->
    Demodulation -> Deinterleaver -> FEC -> Frame) DAG, demodulates and
    decodes each candidate group live, ranks every chain by its composite
    evidence-weighted score with quantified uncertainty, and attaches a
    natural-language WHY justification.
    """
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dt = DataType(data_type) if data_type else None
        signal = load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)
        if signal.metadata.sample_rate is None:
            raise HTTPException(status_code=400,
                                detail="sample_rate required for hypothesis search")
        sr = signal.metadata.sample_rate
        signal = full_normalize(signal, remove_dc_offset=True, normalize_to="power")

        psd = compute_psd(signal, fft_size=1024)
        noise_floor = estimate_noise_floor(psd, method="mad")
        segmentation = build_segments(psd, noise_floor, signal)
        snr = estimate_snr_from_psd(psd, noise_floor)
        bandwidth = estimate_bandwidth(psd, method="3db")

        fv = extract_features(signal.data[:65536], sample_rate=sr)
        sym_res = estimate_symbol_rate(signal.data[:65536], sample_rate=sr)
        cfo_res = estimate_freq_offset(signal.data[:16384], sample_rate=sr)

        mod_result = classify_signal(signal, max_samples=65536)
        demod_result = demodulate_signal(
            signal, modulation=mod_result.top_candidate, max_symbols=1024)

        decoding = []
        if demod_result.bits:
            decoding = search_decode(
                demod_result.bits,
                llrs=(demod_result.llrs_sample or None),
                use_soft_viterbi=True, top_k=10,
            )

        graph = build_evidence_graph(
            snr=snr, bandwidth=bandwidth, noise_floor=noise_floor,
            segmentation=segmentation, features=fv, symbol_rate=sym_res,
            freq_offset=cfo_res, mod_classification=mod_result,
            demodulation=demod_result, decoding=decoding,
            metadata={
                "filename": file.filename,
                "sample_rate": sr,
                "center_frequency": signal.metadata.center_frequency,
            },
        )

        config = HypothesisEngineConfig(
            top_modulations=top_modulations,
            max_chains=max_chains,
        )
        chains = run_hypothesis_search(signal, graph, config)

        return {
            "metadata": signal.metadata.to_dict(),
            "config": config.to_dict(),
            "evidence_graph": graph.to_dict(),
            "num_chains": len(chains),
            "chains": generate_chain_json(chains, graph),
            "top_hypothesis": generate_chain_json(chains[:1], graph)[0] if chains else None,
            "provenance": [p.to_dict() for p in signal.provenance],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Dataset integration endpoints
# ---------------------------------------------------------------------------

DEFAULT_DATASET_PATH = os.environ.get("RF_DATASET_PATH", r"D:\dataset")
_dataset_scanner: Optional[DatasetScanner] = None


def _get_scanner() -> DatasetScanner:
    global _dataset_scanner
    if _dataset_scanner is None:
        _dataset_scanner = DatasetScanner(DEFAULT_DATASET_PATH)
        _dataset_scanner.scan()
    return _dataset_scanner


@app.get("/api/dataset/scan")
async def dataset_scan():
    """Scan the dataset directory and return summary statistics."""
    try:
        scanner = _get_scanner()
        stats = scanner.get_stats()
        return {
            "root_path": stats.root_path,
            "total_files": stats.total_files,
            "total_size_bytes": stats.total_size_bytes,
            "total_size_gb": round(stats.total_size_bytes / (1024 ** 3), 2),
            "by_task": stats.by_task,
            "by_signal": stats.by_signal,
            "by_split": stats.by_split,
            "signal_types_found": stats.signal_types_found,
            "sample_rate": stats.sample_rate,
            "samples_per_file": stats.samples_per_file,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/entries")
async def dataset_entries(
    task: Optional[str] = Query(None, description="demod, sep, frame"),
    signal: Optional[str] = Query(None, description="CommSignal2, CommSignal3, EMISignal1"),
    split: Optional[str] = Query(None, description="train, val"),
    component_type: Optional[str] = Query(None, description="Interference, QPSK, Comm2"),
    max_entries: int = Query(50, ge=1, le=5000),
):
    """List dataset entries with optional filters."""
    try:
        scanner = _get_scanner()
        task_enum = TaskType(task) if task else None
        signal_enum = SignalType(signal) if signal else None
        split_enum = SplitType(split) if split else None

        entries = scanner.filter(
            task=task_enum,
            signal=signal_enum,
            split=split_enum,
            component_type=component_type,
            max_entries=max_entries,
        )

        return {
            "count": len(entries),
            "entries": [
                {
                    "index": e.index,
                    "task": e.task.value,
                    "signal": e.signal.value,
                    "split": e.split.value,
                    "component_type": e.component_type,
                    "has_ground_truth": e.has_ground_truth,
                    "data_path": str(e.data_path),
                    "meta_path": str(e.meta_path),
                    "file_size_bytes": e.data_path.stat().st_size if e.data_path.exists() else 0,
                }
                for e in entries
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/dataset/analyze")
async def dataset_analyze(
    task: str = Query(..., description="demod, sep, frame"),
    signal: str = Query(..., description="CommSignal2, CommSignal3, EMISignal1"),
    split: str = Query(..., description="train, val"),
    index: int = Query(..., description="File index number"),
    component_type: Optional[str] = Query(None),
    fft_size: int = Query(1024),
):
    """Analyze a specific file from the dataset using the full detection pipeline."""
    try:
        scanner = _get_scanner()
        task_enum = TaskType(task)
        signal_enum = SignalType(signal)
        split_enum = SplitType(split)

        entries = scanner.filter(
            task=task_enum,
            signal=signal_enum,
            split=split_enum,
            component_type=component_type,
        )

        target = None
        for e in entries:
            if e.index == index:
                target = e
                break

        if target is None:
            raise HTTPException(
                status_code=404,
                detail=f"No entry found for task={task} signal={signal} split={split} index={index}",
            )

        signal_obj = scanner.load_entry(target)

        if signal_obj.metadata.sample_rate is None:
            raise HTTPException(status_code=400, detail="Sample rate not available")

        psd = compute_psd(signal_obj, fft_size=fft_size)
        noise_floor = estimate_noise_floor(psd, method="mad")
        segmentation = build_segments(psd, noise_floor, signal_obj)
        snr = estimate_snr_from_psd(psd, noise_floor)
        bandwidth = estimate_bandwidth(psd, method="3db")

        return {
            "file_info": {
                "task": target.task.value,
                "signal": target.signal.value,
                "split": target.split.value,
                "index": target.index,
                "component_type": target.component_type,
                "data_path": str(target.data_path),
            },
            "metadata": signal_obj.metadata.to_dict(),
            "psd": psd.to_dict(),
            "noise_floor": noise_floor.to_dict(),
            "snr": snr.to_dict(),
            "bandwidth": bandwidth.to_dict(),
            "segments": segmentation.to_dict(),
            "provenance": [p.to_dict() for p in signal_obj.provenance],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/dataset/batch-analyze")
async def dataset_batch_analyze(
    task: Optional[str] = Query(None),
    signal: Optional[str] = Query(None),
    split: Optional[str] = Query(None),
    max_files: int = Query(10, ge=1, le=100),
    fft_size: int = Query(1024),
):
    """Batch-analyze multiple dataset files and return aggregate results."""
    try:
        scanner = _get_scanner()
        task_enum = TaskType(task) if task else None
        signal_enum = SignalType(signal) if signal else None
        split_enum = SplitType(split) if split else None

        entries = scanner.filter(
            task=task_enum,
            signal=signal_enum,
            split=split_enum,
            max_entries=max_files,
        )

        results = []
        for entry in entries:
            try:
                sig = scanner.load_entry(entry)
                if sig.metadata.sample_rate is None:
                    continue

                psd = compute_psd(sig, fft_size=fft_size)
                nf = estimate_noise_floor(psd, method="mad")
                snr = estimate_snr_from_psd(psd, nf)
                bw = estimate_bandwidth(psd, method="3db")
                seg = build_segments(psd, nf, sig)

                results.append({
                    "file_info": {
                        "task": entry.task.value,
                        "signal": entry.signal.value,
                        "split": entry.split.value,
                        "index": entry.index,
                        "component_type": entry.component_type,
                        "data_path": str(entry.data_path),
                    },
                    "num_samples": sig.num_samples,
                    "snr_db": snr.snr_db,
                    "bandwidth_hz": bw.bandwidth_hz,
                    "noise_floor_db": nf.noise_floor_db,
                    "num_segments": len(seg.segments) if hasattr(seg, 'segments') else 0,
                })
            except Exception as e:
                results.append({
                    "file_info": {"index": entry.index, "signal": entry.signal.value},
                    "error": str(e),
                })

        return {
            "total_processed": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Frontend static serving (frontend/ directory; optional at import time)
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


def _mount_frontend(app: FastAPI) -> None:
    if not _FRONTEND_DIR.is_dir():
        return
    try:
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
    except Exception:
        pass


_mount_frontend(app)

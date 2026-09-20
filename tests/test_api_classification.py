"""Tests for FastAPI endpoints: /api/analyze, /api/modulation/classify, and /api/demodulate."""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def _create_synthetic_wav_or_cf32(n: int = 4096, mod: str = "QPSK", sps: int = 4) -> bytes:
    """Create a raw CF32 byte buffer containing realistic oversampled complex IQ signal."""
    np.random.seed(42)
    num_symbols = max(1, n // sps)
    if mod == "BPSK":
        bits = np.random.randint(0, 2, num_symbols) * 2 - 1
        syms = bits.astype(np.complex64)
    elif mod == "16QAM":
        grid = np.array([-3, -1, 1, 3])
        i_ch = np.random.choice(grid, num_symbols)
        q_ch = np.random.choice(grid, num_symbols)
        syms = (i_ch + 1j * q_ch).astype(np.complex64) / np.sqrt(10.0)
    else:
        # QPSK
        bits = np.random.randint(0, 4, num_symbols)
        syms = np.exp(1j * (bits * np.pi / 2.0 + np.pi / 4.0)).astype(np.complex64)

    oversampled = np.repeat(syms, sps)[:n]
    noise = (np.random.randn(len(oversampled)) + 1j * np.random.randn(len(oversampled))).astype(np.complex64) * 0.04
    iq = oversampled + noise
    return iq.tobytes()


def test_api_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_api_modulation_classify():
    """POST /api/modulation/classify should return ranked candidate hypotheses."""
    raw_bytes = _create_synthetic_wav_or_cf32(4096, mod="QPSK")
    files = {"file": ("test_signal.cf32", io.BytesIO(raw_bytes), "application/octet-stream")}
    
    resp = client.post(
        "/api/modulation/classify",
        files=files,
        params={"data_type": "cf32", "sample_rate": 1000000.0},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "classification" in data
    clf = data["classification"]
    assert "top_candidate" in clf
    assert "candidates" in clf
    assert len(clf["candidates"]) >= 5
    assert clf["top_candidate"] == "QPSK"
    assert "why" in clf["candidates"][0]
    assert len(data["provenance"]) > 0


def test_api_analyze_with_classification():
    """POST /api/analyze should return detection, characterization, and modulation classification."""
    raw_bytes = _create_synthetic_wav_or_cf32(4096, mod="QPSK")
    files = {"file": ("test_signal.cf32", io.BytesIO(raw_bytes), "application/octet-stream")}

    resp = client.post(
        "/api/analyze",
        files=files,
        params={"sample_rate": 1000000.0, "data_type": "cf32"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Core detection metrics
    assert "psd" in data
    assert "noise_floor" in data
    assert "snr" in data
    assert "bandwidth" in data

    # Characterization
    assert "characterization" in data
    assert "features" in data["characterization"]
    assert "symbol_rate" in data["characterization"]
    assert "freq_offset" in data["characterization"]

    # Modulation classification
    assert "modulation_classification" in data
    mod_clf = data["modulation_classification"]
    assert mod_clf["top_candidate"] == "QPSK"
    assert len(mod_clf["candidates"]) > 0


def test_api_demodulate():
    """POST /api/demodulate should return recovered symbols and EVM metrics."""
    raw_bytes = _create_synthetic_wav_or_cf32(4096, mod="QPSK")
    files = {"file": ("test_signal.cf32", io.BytesIO(raw_bytes), "application/octet-stream")}

    resp = client.post(
        "/api/demodulate",
        files=files,
        params={"sample_rate": 1000000.0, "data_type": "cf32", "modulation": "QPSK", "samples_per_symbol": 1.0},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "demodulation" in data
    demod = data["demodulation"]
    assert demod["modulation"] == "QPSK"
    assert demod["evm"]["evm_rms_percent"] < 25.0
    assert len(demod["symbols_sample"]) > 0
    assert len(demod["bits_sample"]) > 0


def test_api_characterize():
    """POST /api/characterize should return feature vector and offsets."""
    raw_bytes = _create_synthetic_wav_or_cf32(2048, mod="BPSK")
    files = {"file": ("test_bpsk.cf32", io.BytesIO(raw_bytes), "application/octet-stream")}

    resp = client.post(
        "/api/characterize",
        files=files,
        params={"sample_rate": 2000000.0, "data_type": "cf32"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "features" in data
    assert "C42_norm" in data["features"]
    assert "papr_db" in data["features"]
    assert "symbol_rate" in data
    assert "freq_offset" in data


def test_api_evidence_graph():
    """POST /api/evidence should return a scored cross-stage evidence graph."""
    raw_bytes = _create_synthetic_wav_or_cf32(4096, mod="QPSK")
    files = {"file": ("test_signal.cf32", io.BytesIO(raw_bytes), "application/octet-stream")}

    resp = client.post(
        "/api/evidence",
        files=files,
        params={"sample_rate": 1000000.0, "data_type": "cf32"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "evidence_graph" in data
    graph = data["evidence_graph"]
    assert "items" in graph and len(graph["items"]) > 0
    assert "edges" in graph and len(graph["edges"]) > 0
    assert "hypotheses" in graph and len(graph["hypotheses"]) >= 6

    by_name = {h["name"]: h for h in graph["hypotheses"]}
    assert "QPSK" in by_name
    assert by_name["QPSK"]["score"] > 0.5  # synthetic QPSK should be supported

    top = {h["name"] for h in graph["top_hypotheses"] if h["kind"] == "modulation"}
    assert "QPSK" in top


def test_api_evidence_graph_without_decode():
    """run_decode=False should still produce a valid modulation-focused graph."""
    raw_bytes = _create_synthetic_wav_or_cf32(2048, mod="QPSK")
    files = {"file": ("test_signal.cf32", io.BytesIO(raw_bytes), "application/octet-stream")}

    resp = client.post(
        "/api/evidence",
        files=files,
        params={"sample_rate": 1000000.0, "data_type": "cf32",
                "run_demod": "false", "run_decode": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["config"]["run_demod"] is False
    graph = body["evidence_graph"]
    assert len(graph["hypotheses"]) >= 6
    by_name = {h["name"]: h for h in graph["hypotheses"]}
    assert by_name["QPSK"]["score"] > 0.5


def test_api_decode():
    """POST /api/decode should return decode-chain leaderboard + demod stats."""
    raw_bytes = _create_synthetic_wav_or_cf32(4096, mod="QPSK")
    files = {"file": ("test_signal.cf32", io.BytesIO(raw_bytes), "application/octet-stream")}

    resp = client.post(
        "/api/decode",
        files=files,
        params={"sample_rate": 1000000.0, "data_type": "cf32"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "modulation" in body and body["modulation"] == "QPSK"
    demod = body["demodulation"]
    assert demod["modulation"] == "QPSK"
    assert demod["num_symbols"] > 0
    assert "decode_leaderboard" in body
    assert len(body["decode_leaderboard"]) > 0
    print(body["decode_leaderboard"][0].keys())


def test_api_decode_with_modulation_override():
    """Passing an explicit modulation should skip classification."""
    raw_bytes = _create_synthetic_wav_or_cf32(2048, mod="BPSK")
    files = {"file": ("test_bpsk.cf32", io.BytesIO(raw_bytes), "application/octet-stream")}

    resp = client.post(
        "/api/decode",
        files=files,
        params={"sample_rate": 1000000.0, "data_type": "cf32", "modulation": "BPSK"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["modulation"] == "BPSK"
    assert body["demodulation"]["modulation"] == "BPSK"


def test_api_hypothesize():
    """POST /api/hypothesize should return ranked chains with WHY and uncertainty."""
    raw_bytes = _create_synthetic_wav_or_cf32(4096, mod="QPSK")
    files = {"file": ("test_signal.cf32", io.BytesIO(raw_bytes), "application/octet-stream")}

    resp = client.post(
        "/api/hypothesize",
        files=files,
        params={"sample_rate": 1000000.0, "data_type": "cf32",
                "top_modulations": 1, "max_chains": 60},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "evidence_graph" in body
    assert body["num_chains"] >= 1
    chains = body["chains"]
    assert isinstance(chains, list) and len(chains) >= 1
    top = chains[0]
    assert top["chain"]["modulation"] == "QPSK"
    assert "composite_score" in top["chain"]
    assert 0.0 <= top["chain"]["composite_score"] <= 1.0
    assert top["why"] and "Hypothesis" in top["why"]
    assert body["top_hypothesis"]["chain"]["modulation"] == "QPSK"

"""Tests for the HITL / Benchmark-Advisor API layer (app/api/hitl.py).

Uses the FastAPI TestClient against a synthetic QPSK burst uploaded as a raw
.cf32 file. The corpus is pointed at a tmp file and the run audit at a tmp
JSON file via env vars, so nothing on disk leaks between runs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.benchmark.generator import generate_synthetic_sample


@pytest.fixture()
def env_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("RF_BENCHMARK_CORPUS", str(tmp_path / "corpus.json"))
    monkeypatch.setenv("RF_HITL_RUNS_FILE", str(tmp_path / "runs.json"))
    yield tmp_path


def _cf32_bytes(mod: str = "QPSK", snr: float = 18.0) -> bytes:
    iq, _gt = generate_synthetic_sample(
        mod, "none", "none", "preamble+crc16", snr,
        n_symbols_target=512, payload_size=24, seed=11,
    )
    raw = np.empty(len(iq) * 2, dtype=np.float32)
    raw[0::2] = iq.real
    raw[1::2] = iq.imag
    return raw.tobytes()


@pytest.fixture(scope="module")
def client():
    from app.api.main import app
    return TestClient(app)


class TestFrontendStatic:
    def test_index_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "RF" in r.text
        assert 'app.js' in r.text or 'style.css' in r.text

    def test_assets_served(self, client):
        assert client.get("/app.js").status_code == 200
        assert client.get("/style.css").status_code == 200


class TestAdvisorConsult:
    def test_consult_returns_advisory(self, client, env_tmp):
        r = client.post(
            "/api/advisor/consult",
            params={"run_id": "run-test-1", "sample_rate": 1.0e6, "k": 8},
            files={"file": ("burst.cf32", _cf32_bytes(), "application/octet-stream")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["query_vector"]) == 14
        assert isinstance(body["advisory"]["suppressed"], bool)
        assert body["advisory"]["neighbours"] >= 0
        assert isinstance(body["corpus"], dict)

    def test_consult_requires_sample_rate(self, client, env_tmp):
        r = client.post(
            "/api/advisor/consult",
            params={"run_id": "run-test-x"},
            files={"file": ("burst.cf32", _cf32_bytes(), "application/octet-stream")},
        )
        assert r.status_code == 400


class TestGate:
    def _consult_payload(self, client):
        r = client.post(
            "/api/advisor/consult",
            params={"run_id": "run-gate-1", "sample_rate": 1.0e6, "k": 8},
            files={"file": ("burst.cf32", _cf32_bytes(), "application/octet-stream")},
        )
        assert r.status_code == 200
        b = r.json()
        return {
            "query_vector": b["query_vector"],
            "snr_db": b["physical_estimates"]["snr_db"],
        }

    def test_confirm_appends_corpus_and_audits(self, client, env_tmp):
        payload = self._consult_payload(client)
        gate = {
            "run_id": "run-gate-1",
            "gate": "G1",
            "action": "confirm",
            "to_value": "QPSK",
            "from_value": "QPSK",
            "chain": None,
            "snr_db": payload["snr_db"],
            "crc_validated": False,
            "feature_vector": payload["query_vector"],
        }
        r = client.post("/api/hitl/gate", json=gate)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["corpus"]["size"] >= 1
        assert "G1" in body["audit"]["gates_fired"]
        assert body["audit"]["human_overrides"][-1]["to"] == "QPSK"

    def test_override_records_from_to(self, client, env_tmp):
        payload = self._consult_payload(client)
        gate = {
            "run_id": "run-gate-2",
            "gate": "G2",
            "action": "override",
            "to_value": "viterbi|block(4, 8)|preamble+crc16",
            "from_value": "full grid",
            "chain": "viterbi|block(4, 8)|preamble+crc16",
            "snr_db": payload["snr_db"],
            "feature_vector": payload["query_vector"],
        }
        r = client.post("/api/hitl/gate", json=gate)
        assert r.status_code == 200
        body = r.json()
        assert body["audit"]["human_overrides"][-1]["from"] == "full grid"
        assert body["corpus"]["size"] >= 1
        run = client.get("/api/hitl/run/run-gate-2").json()
        assert run["gates_fired"] == ["G2"]

    def test_bad_feature_vector_rejected(self, client, env_tmp):
        gate = {
            "run_id": "run-bad",
            "gate": "G1",
            "action": "confirm",
            "to_value": "BPSK",
            "feature_vector": [0.0, 0.0],
        }
        r = client.post("/api/hitl/gate", json=gate)
        assert r.status_code == 400

    def test_bad_gate_name_rejected(self, client, env_tmp):
        gate = {
            "run_id": "run-bad2",
            "gate": "G9",
            "action": "confirm",
            "to_value": "BPSK",
            "feature_vector": [0.0] * 14,
        }
        r = client.post("/api/hitl/gate", json=gate)
        assert r.status_code == 400


class TestCorpusSummary:
    def test_summary_shape(self, client, env_tmp):
        r = client.get("/api/hitl/corpus")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["size"], int)
        assert isinstance(body["per_chain"], list)
        assert isinstance(body["source_mix"], dict)

    def test_audit_endpoint(self, client, env_tmp):
        r = client.get("/api/hitl/audit")
        assert r.status_code == 200
        assert isinstance(r.json(), dict)
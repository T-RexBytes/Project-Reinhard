"""HITL + Benchmark Advisor API endpoints (§9.4 backend support).

Read-only advisory (`/api/advisor/consult`) and the gate decision surface
(`/api/hitl/gate`) with run audit (§7) and corpus status (§11.4). The advisory
never enters the evidence graph and never affects a composite score — it is a
separate `AdvisoryRecord` attached to the run.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.ingestion.parser import load_signal
from app.benchmark import service
from app.benchmark.advisor import AdvisoryRecord

hitl_router = APIRouter(prefix="/api", tags=["hitl"])


class GateRequest(BaseModel):
    run_id: str
    gate: str                       # "G1" | "G2" | "G3"
    action: str                     # "confirm" | "override"
    to_value: str                   # confirmed modulation / pinned chain / etc.
    from_value: Optional[str] = None
    chain: Optional[str] = None
    snr_db: Optional[float] = None
    crc_validated: bool = False
    feature_vector: list[float]


async def _read_upload(file: UploadFile, data_type: Optional[str],
                       sample_rate: Optional[float]):
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        from app.models import DataType
        dt = DataType(data_type) if data_type else None
        return load_signal(tmp_path, data_type=dt, sample_rate=sample_rate)
    finally:
        os.unlink(tmp_path)


@hitl_router.post("/advisor/consult")
async def advisor_consult(
    file: UploadFile = File(...),
    run_id: str = Query("", description="Client-run id for the audit trail"),
    k: int = Query(12, ge=5, le=50),
    data_type: Optional[str] = Query(None),
    sample_rate: Optional[float] = Query(None),
):
    """Build the query vector and consult the verified corpus (read-only)."""
    try:
        signal = await _read_upload(file, data_type, sample_rate)
        if signal.metadata.sample_rate is None:
            raise HTTPException(status_code=400, detail="sample_rate required")
        result = service.consult_for_signal(signal, run_id=run_id, k=k)
        service.record_consult(run_id, AdvisoryRecord(**result["advisory"]))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@hitl_router.post("/hitl/gate")
async def hitl_gate(request: GateRequest):
    """Record a confirm/override at a gate; promotes to the corpus on action."""
    try:
        return service.apply_gate(
            run_id=request.run_id,
            gate=request.gate,
            action=request.action,
            to_value=request.to_value,
            from_value=request.from_value,
            chain=request.chain,
            snr_db=request.snr_db,
            crc_validated=request.crc_validated,
            feature_vector=request.feature_vector,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@hitl_router.get("/hitl/run/{run_id}")
async def hitl_run(run_id: str):
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    return run


@hitl_router.get("/hitl/corpus")
async def hitl_corpus():
    """Corpus status + §11.4 CRC/chain pass-rate rows + source mix."""
    return service.corpus_summary()


@hitl_router.get("/hitl/audit")
async def hitl_audit():
    """All recorded runs (audit trail dump)."""
    return service.all_runs()
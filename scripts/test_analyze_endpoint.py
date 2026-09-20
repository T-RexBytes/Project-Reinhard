"""Quick smoke test of /api/analyze against the generated test signal."""
import requests, json

BASE = "http://localhost:8000"

with open("test_qpsk_25MHz.cf32", "rb") as f:
    data = f.read()

r = requests.post(
    BASE + "/api/analyze",
    files={"file": ("test_qpsk_25MHz.cf32", data, "application/octet-stream")},
    params={"sample_rate": 25000000, "data_type": "cf32"},
    timeout=30,
)
print(f"Status: {r.status_code}")
if r.ok:
    j = r.json()
    mod = j.get("modulation_classification", {})
    snr = j.get("snr", {})
    bw  = j.get("bandwidth", {})
    top = mod.get("top_candidate", "?")
    snr_db = snr.get("snr_db", 0)
    bw_hz  = bw.get("bandwidth_hz", 0)
    cands  = [(c["modulation"], round(c["score"], 3)) for c in mod.get("candidates", [])]
    print(f"Top candidate : {top}")
    print(f"SNR           : {snr_db:.2f} dB")
    print(f"Bandwidth     : {bw_hz/1e3:.2f} kHz")
    print(f"Candidates    : {cands}")
else:
    print("ERROR:", r.text[:500])

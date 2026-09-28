#!/usr/bin/env python3
"""
api.py
------
FastAPI server wrapping the trained IPsec traffic-type classifier.

Loads model.joblib (produced by train_classifier.py) once at startup and
exposes a /predict endpoint: send it a single flow-window's features, get
back the predicted traffic type (icmp/web/video/voip) plus per-class
confidence scores.

Usage:
    pip install fastapi uvicorn --break-system-packages
    python3 api.py
    # or: uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Then test with:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
            "packet_count": 12, "total_bytes": 1816, "avg_packet_size": 151.3,
            "std_packet_size": 29.8, "avg_inter_arrival": 0.094,
            "std_inter_arrival": 0.293, "flow_duration": 1.037,
            "bitrate_bps": 14012.5
          }'
"""

import os
import tempfile
import threading
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from extract_features import load_esp_packets, make_windows, compute_flow_features

MODEL_PATH = "model.joblib"

app = FastAPI(
    title="IPsec Traffic Type Classifier",
    description="Predicts traffic type (icmp/web/video/voip) from encrypted IPsec ESP flow features.",
    version="1.0.0",
)

# Allows a browser-based frontend (opened as a local file or from another
# origin) to call this API. Fine for a hackathon demo on a private testbed;
# lock this down to specific origins before using this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup rather than per-request, so predictions stay fast.
_bundle = None


@app.on_event("startup")
def load_model():
    global _bundle
    try:
        _bundle = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        # Don't crash the server on startup -- /predict will report the
        # problem clearly instead, which is easier to debug during a demo.
        _bundle = None


class FlowFeatures(BaseModel):
    packet_count: int = Field(..., description="Number of packets in the flow window")
    total_bytes: int = Field(..., description="Total bytes across the window")
    avg_packet_size: float = Field(..., description="Mean packet size in bytes")
    std_packet_size: float = Field(..., description="Std deviation of packet size")
    avg_inter_arrival: float = Field(..., description="Mean time between packets, in seconds")
    std_inter_arrival: float = Field(..., description="Std deviation of inter-arrival time")
    flow_duration: float = Field(..., description="Span of the window, in seconds")
    bitrate_bps: float = Field(..., description="total_bytes * 8 / flow_duration")


class PredictionResponse(BaseModel):
    predicted_label: str
    probabilities: dict


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: FlowFeatures):
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded -- make sure {MODEL_PATH} exists next to api.py.",
        )

    clf = _bundle["model"]
    columns = _bundle["feature_columns"]

    row = pd.DataFrame([features.model_dump()], columns=columns)

    predicted_label = clf.predict(row)[0]
    proba = clf.predict_proba(row)[0]
    probabilities = {label: round(float(p), 4) for label, p in zip(clf.classes_, proba)}

    return PredictionResponse(predicted_label=predicted_label, probabilities=probabilities)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _require_model():
    if _bundle is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded -- {MODEL_PATH} missing.")


def classify(features):
    clf, cols = _bundle["model"], _bundle["feature_columns"]
    row = pd.DataFrame([features], columns=cols)
    label = clf.predict(row)[0]
    proba = clf.predict_proba(row)[0]
    return {"label": label,
            "probabilities": {l: round(float(p), 4) for l, p in zip(clf.classes_, proba)}}


# ---------------------------------------------------------------------------
# Mode 1: upload a pcap
# ---------------------------------------------------------------------------

@app.post("/upload-pcap")
async def upload_pcap(request: Request, window: float = 2.0, min_packets: int = 3):
    """Body = raw .pcap bytes. Returns one prediction per flow-window."""
    _require_model()
    data = await request.body()
    if not data:
        raise HTTPException(400, "Empty upload.")
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        records = load_esp_packets(path)
    except Exception as e:
        raise HTTPException(400, f"Could not read this capture: {e}")
    finally:
        os.remove(path)

    results = []
    for (idx, src, dst), recs in sorted(make_windows(records, window).items()):
        if len(recs) < min_packets:
            continue
        f = compute_flow_features(recs)
        results.append({"window": idx, "src": src, "dst": dst, "features": f, **classify(f)})
    return {"esp_packets": len(records), "results": results}


# ---------------------------------------------------------------------------
# Mode 2: live capture from a network device (needs sudo / admin)
# ---------------------------------------------------------------------------

_cap = {"running": False, "sniffer": None, "iface": None, "buf": [], "results": [],
        "seen": 0, "esp": 0, "lock": threading.Lock()}


@app.get("/interfaces")
def interfaces():
    from scapy.all import conf
    out = []
    for i in conf.ifaces.values():
        desc = getattr(i, "description", "") or ""
        out.append({"id": i.name, "label": f"{i.name} ({desc})" if desc and desc != i.name else i.name})
    return out


def _on_packet(pkt):
    from scapy.all import IP
    _cap["seen"] += 1
    if IP in pkt and pkt[IP].proto == 50:  # ESP
        _cap["esp"] += 1
        with _cap["lock"]:
            _cap["buf"].append((float(pkt.time), len(pkt), pkt[IP].src, pkt[IP].dst))


def _cap_loop(window, min_packets):
    while _cap["running"]:
        time.sleep(window)
        with _cap["lock"]:
            recs, _cap["buf"] = _cap["buf"], []
        flows = {}
        for ts, size, src, dst in recs:
            flows.setdefault((src, dst), []).append((ts, size))
        for (src, dst), fr in flows.items():
            if len(fr) < min_packets:
                continue
            f = compute_flow_features(fr)
            entry = {"t": time.strftime("%H:%M:%S"), "src": src, "dst": dst, "features": f, **classify(f)}
            with _cap["lock"]:
                _cap["results"].append(entry)


@app.post("/capture/start")
def capture_start(iface: str, window: float = 2.0, min_packets: int = 3):
    _require_model()
    if _cap["running"]:
        raise HTTPException(409, "A capture is already running.")
    from scapy.all import AsyncSniffer
    _cap.update(running=True, iface=iface, buf=[], results=[], seen=0, esp=0)
    try:
        sniffer = AsyncSniffer(iface=iface, prn=_on_packet, store=False)
        sniffer.start()
    except Exception as e:
        _cap["running"] = False
        raise HTTPException(500, f"Could not capture on {iface}: {e}. Run the server with sudo/admin.")
    _cap["sniffer"] = sniffer
    threading.Thread(target=_cap_loop, args=(window, min_packets), daemon=True).start()
    return {"status": "started", "iface": iface}


@app.post("/capture/stop")
def capture_stop():
    _cap["running"] = False
    if _cap["sniffer"] is not None:
        try:
            _cap["sniffer"].stop()
        except Exception:
            pass
        _cap["sniffer"] = None
    return {"status": "stopped"}


@app.get("/capture/results")
def capture_results(since: int = 0):
    with _cap["lock"]:
        return {"running": _cap["running"], "iface": _cap["iface"], "seen": _cap["seen"],
                "esp": _cap["esp"], "results": _cap["results"][since:], "next": len(_cap["results"])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

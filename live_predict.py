#!/usr/bin/env python3
"""
live_predict.py
----------------
Closes the loop: capture -> extract features -> predict, calling the
running FastAPI server for each flow-window instead of writing a CSV.

Two modes:

  Replay an existing pcap (fastest way to demo -- no live traffic needed):
    python3 live_predict.py --pcap video_1.pcap

  Live-capture on an interface for N seconds, then classify what it saw:
    sudo python3 live_predict.py --iface enp0s8 --duration 15

Requires the API server to already be running (python3 api.py), and reuses
the feature-extraction logic from extract_features.py so both stay in sync.

Requires: requests, scapy (for --iface mode)
    pip3 install requests --break-system-packages
"""

import argparse
import sys
import tempfile

import requests

from extract_features import load_esp_packets, make_windows, compute_flow_features

API_URL_DEFAULT = "http://localhost:8000/predict"


def capture_live(iface, duration, out_path):
    """Capture ESP traffic on an interface for `duration` seconds using
    tcpdump (more reliable under sudo than scapy's sniff for this)."""
    import subprocess

    print(f"Capturing on {iface} for {duration}s ...")
    proc = subprocess.Popen(["tcpdump", "-i", iface, "-w", out_path])
    try:
        proc.wait(timeout=duration)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait()
    print("Capture finished.")


def classify_windows(pcap_path, window_size, min_packets, api_url):
    records = load_esp_packets(pcap_path)
    print(f"{len(records)} ESP packets found in {pcap_path}")

    if not records:
        print("No ESP packets found -- nothing to classify. "
              "Check the IPsec tunnel is up and traffic is actually flowing through it.")
        return

    windows = make_windows(records, window_size)
    classified = 0

    for (window_idx, src, dst), flow_records in sorted(windows.items()):
        if len(flow_records) < min_packets:
            continue

        features = compute_flow_features(flow_records)

        try:
            response = requests.post(api_url, json=features, timeout=5)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            print(f"  [window {window_idx}] API call failed: {e}")
            continue

        label = result["predicted_label"]
        confidence = result["probabilities"].get(label, 0)
        print(f"  [window {window_idx}] {src} -> {dst}: "
              f"predicted = {label}  (confidence {confidence:.0%}, "
              f"{features['packet_count']} pkts, {features['bitrate_bps']:.0f} bps)")
        classified += 1

    if classified == 0:
        print("No windows met the minimum packet threshold -- try --window or --min-packets.")
    else:
        print(f"\nClassified {classified} flow-window(s).")


def main():
    parser = argparse.ArgumentParser(description="Capture/replay traffic and classify it live via the API.")
    parser.add_argument("--pcap", help="Existing pcap file to replay and classify")
    parser.add_argument("--iface", help="Network interface to capture live from (requires sudo)")
    parser.add_argument("--duration", type=float, default=15.0,
                         help="Seconds to capture when using --iface (default: 15)")
    parser.add_argument("--window", type=float, default=2.0, help="Window size in seconds (default: 2.0)")
    parser.add_argument("--min-packets", type=int, default=3,
                         help="Skip windows with fewer than this many packets (default: 3)")
    parser.add_argument("--api-url", default=API_URL_DEFAULT,
                         help=f"Predict endpoint URL (default: {API_URL_DEFAULT})")
    args = parser.parse_args()

    if not args.pcap and not args.iface:
        parser.error("Provide either --pcap FILE (replay) or --iface IFACE (live capture)")

    if args.iface:
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
            pcap_path = tmp.name
        capture_live(args.iface, args.duration, pcap_path)
    else:
        pcap_path = args.pcap

    classify_windows(pcap_path, args.window, args.min_packets, args.api_url)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

#!/usr/bin/env python3
"""
extract_features.py
--------------------
Turns a labeled .pcap capture of ESP-encrypted IPsec traffic into rows of
flow-level statistical features for training a traffic-type classifier.

Since ESP encrypts everything above the IP layer, we CANNOT use TCP/UDP
ports or flags. Instead we treat all ESP packets between the same
(src IP, dst IP) pair as one "flow", and cut that flow into fixed-size
time windows. Each window becomes one training row -- this is what turns
a single 30-second capture into dozens of labeled samples instead of just one.

Usage (run once per pcap, appending to the same CSV):

    python3 extract_features.py --pcap icmp_1.pcap  --label icmp  --output dataset.csv
    python3 extract_features.py --pcap web_1.pcap   --label web   --output dataset.csv
    python3 extract_features.py --pcap video_1.pcap --label video --output dataset.csv
    python3 extract_features.py --pcap voip_1.pcap  --label voip  --output dataset.csv

Requires: scapy   (pip install scapy --break-system-packages)
"""

import argparse
import csv
import os
import statistics as stats

from scapy.all import IP
from scapy.utils import PcapReader


ESP_PROTO_NUMBER = 50  # IP protocol number for ESP


def load_esp_packets(pcap_path):
    """Read the pcap and return a sorted list of (timestamp, size, src, dst)
    tuples for ESP packets only. Non-IP or non-ESP packets (ARP, etc.) are
    skipped since they aren't part of the tunnel traffic we care about.

    Uses PcapReader to stream through the file one packet at a time instead
    of loading the whole capture into memory at once (rdpcap does that, and
    it will get OOM-killed on large captures like a 200+ MB video test)."""
    records = []
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if IP in pkt and pkt[IP].proto == ESP_PROTO_NUMBER:
                records.append((float(pkt.time), len(pkt), pkt[IP].src, pkt[IP].dst))
    records.sort(key=lambda r: r[0])
    return records


def make_windows(records, window_size):
    """Group (timestamp, size, src, dst) records into fixed-size time
    windows, keyed by (window_index, src, dst) so traffic in each direction
    is treated as its own flow within that window."""
    if not records:
        return {}

    start_time = records[0][0]
    windows = {}
    for ts, size, src, dst in records:
        window_idx = int((ts - start_time) // window_size)
        key = (window_idx, src, dst)
        windows.setdefault(key, []).append((ts, size))
    return windows


def compute_flow_features(flow_records):
    """Given a list of (timestamp, size) tuples for one flow-window,
    compute the statistical features used for classification."""
    sizes = [size for _, size in flow_records]
    timestamps = sorted(ts for ts, _ in flow_records)

    packet_count = len(flow_records)
    total_bytes = sum(sizes)
    avg_size = stats.mean(sizes)
    std_size = stats.pstdev(sizes) if packet_count > 1 else 0.0

    if packet_count > 1:
        inter_arrivals = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
        avg_iat = stats.mean(inter_arrivals)
        std_iat = stats.pstdev(inter_arrivals) if len(inter_arrivals) > 1 else 0.0
        duration = timestamps[-1] - timestamps[0]
    else:
        avg_iat = 0.0
        std_iat = 0.0
        duration = 0.0

    # Bitrate is the strongest single signal for telling apart traffic types
    # that differ mainly in throughput -- e.g. a ~50 Mbit/s video stream vs
    # a ~64 Kbit/s voip call look similar in packet timing but wildly
    # different here. Guard against divide-by-zero for single-packet windows.
    bitrate_bps = (total_bytes * 8 / duration) if duration > 0 else 0.0

    return {
        "packet_count": packet_count,
        "total_bytes": total_bytes,
        "avg_packet_size": round(avg_size, 3),
        "std_packet_size": round(std_size, 3),
        "avg_inter_arrival": round(avg_iat, 6),
        "std_inter_arrival": round(std_iat, 6),
        "flow_duration": round(duration, 3),
        "bitrate_bps": round(bitrate_bps, 3),
    }


FIELDNAMES = [
    "label",
    "src_ip",
    "dst_ip",
    "window_index",
    "packet_count",
    "total_bytes",
    "avg_packet_size",
    "std_packet_size",
    "avg_inter_arrival",
    "std_inter_arrival",
    "flow_duration",
    "bitrate_bps",
]


def main():
    parser = argparse.ArgumentParser(description="Extract flow features from a labeled pcap.")
    parser.add_argument("--pcap", required=True, help="Path to the .pcap file")
    parser.add_argument("--label", required=True, help="Traffic type label, e.g. web/video/voip/icmp/email")
    parser.add_argument("--output", default="dataset.csv", help="CSV file to append rows to")
    parser.add_argument("--window", type=float, default=2.0, help="Window size in seconds (default: 2.0)")
    parser.add_argument("--min-packets", type=int, default=3,
                         help="Skip windows with fewer than this many packets (default: 3)")
    args = parser.parse_args()

    print(f"Reading {args.pcap} ...")
    records = load_esp_packets(args.pcap)
    print(f"  {len(records)} ESP packets found")

    windows = make_windows(records, args.window)
    print(f"  {len(windows)} (window, src, dst) flow-windows built at {args.window}s each")

    rows = []
    for (window_idx, src, dst), flow_records in sorted(windows.items()):
        if len(flow_records) < args.min_packets:
            continue
        features = compute_flow_features(flow_records)
        row = {
            "label": args.label,
            "src_ip": src,
            "dst_ip": dst,
            "window_index": window_idx,
            **features,
        }
        rows.append(row)

    if not rows:
        print("  WARNING: no windows met the minimum packet threshold -- nothing written.")
        return

    file_exists = os.path.isfile(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {len(rows)} rows labeled '{args.label}' to {args.output}")


if __name__ == "__main__":
    main()

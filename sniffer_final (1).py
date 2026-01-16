from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import numpy as np
import time

# ===============================
# 71 Features (exact order)
# ===============================
FEATURE_ORDER = [
    "src_port","dst_port","protocol","ip_version","vlan_id","tunnel_id",
    "bidirectional_first_seen_ms","bidirectional_last_seen_ms","bidirectional_duration_ms",
    "bidirectional_packets","bidirectional_bytes",
    "src2dst_first_seen_ms","src2dst_last_seen_ms","src2dst_duration_ms",
    "src2dst_packets","src2dst_bytes",
    "dst2src_first_seen_ms","dst2src_last_seen_ms","dst2src_duration_ms",
    "dst2src_packets","dst2src_bytes",
    "bidirectional_min_ps","bidirectional_mean_ps","bidirectional_stddev_ps","bidirectional_max_ps",
    "src2dst_min_ps","src2dst_mean_ps","src2dst_stddev_ps","src2dst_max_ps",
    "dst2src_min_ps","dst2src_mean_ps","dst2src_stddev_ps","dst2src_max_ps",
    "bidirectional_min_piat_ms","bidirectional_mean_piat_ms","bidirectional_stddev_piat_ms","bidirectional_max_piat_ms",
    "src2dst_min_piat_ms","src2dst_mean_piat_ms","src2dst_stddev_piat_ms","src2dst_max_piat_ms",
    "dst2src_min_piat_ms","dst2src_mean_piat_ms","dst2src_stddev_piat_ms","dst2src_max_piat_ms",
    "bidirectional_syn_packets","bidirectional_cwr_packets","bidirectional_ece_packets",
    "bidirectional_urg_packets","bidirectional_ack_packets","bidirectional_psh_packets",
    "bidirectional_rst_packets","bidirectional_fin_packets",
    "src2dst_syn_packets","src2dst_cwr_packets","src2dst_ece_packets","src2dst_urg_packets",
    "src2dst_ack_packets","src2dst_psh_packets","src2dst_rst_packets","src2dst_fin_packets",
    "dst2src_syn_packets","dst2src_cwr_packets","dst2src_ece_packets","dst2src_urg_packets",
    "dst2src_ack_packets","dst2src_psh_packets","dst2src_rst_packets","dst2src_fin_packets"
]

# =================================
# Flow state to compute features
# =================================
flows = {}

def extract_flags(pkt):
    if TCP not in pkt:
        return dict(syn=0,cwr=0,ece=0,urg=0,ack=0,psh=0,rst=0,fin=0)
    t = pkt[TCP]
    return dict(
        syn=int(t.flags & 0x02 > 0),
        cwr=int(t.flags & 0x80 > 0),
        ece=int(t.flags & 0x40 > 0),
        urg=int(t.flags & 0x20 > 0),
        ack=int(t.flags & 0x10 > 0),
        psh=int(t.flags & 0x08 > 0),
        rst=int(t.flags & 0x04 > 0),
        fin=int(t.flags & 0x01 > 0),
    )

def update_flow(pkt):
    if IP not in pkt:
        return None

    proto = 6 if TCP in pkt else (17 if UDP in pkt else None)
    if proto is None:
        return None

    key = (pkt[IP].src, pkt[IP].dst, pkt.sport, pkt.dport, proto)

    now = time.time() * 1000
    size = len(pkt)
    flags = extract_flags(pkt)

    if key not in flows:
        flows[key] = dict(
            timestamps=[],
            sizes=[],
            flags=[],
            s2d_ts=[], s2d_sizes=[], s2d_flags=[],
            d2s_ts=[], d2s_sizes=[], d2s_flags=[],
            first=now, last=now,
            src=pkt[IP].src, dst=pkt[IP].dst,
            sport=pkt.sport, dport=pkt.dport, proto=proto
        )

    f = flows[key]
    f["last"] = now

    # direction
    if pkt[IP].src == f["src"]:
        f["s2d_ts"].append(now)
        f["s2d_sizes"].append(size)
        f["s2d_flags"].append(flags)
    else:
        f["d2s_ts"].append(now)
        f["d2s_sizes"].append(size)
        f["d2s_flags"].append(flags)

    f["timestamps"].append(now)
    f["sizes"].append(size)
    f["flags"].append(flags)

    return key


# ============================================
# Compute final flow vector (71 features)
# ============================================
def compute_feature_vector(f):

    def stats(arr):
        return [
            np.min(arr) if len(arr)>0 else 0,
            np.mean(arr) if len(arr)>0 else 0,
            np.std(arr) if len(arr)>0 else 0,
            np.max(arr) if len(arr)>0 else 0,
        ]

    def piat(ts):
        if len(ts) <= 1:
            return [0,0,0,0]
        diffs = np.diff(ts)
        return stats(diffs)

    # packet sizes
    bid_ps = stats(f["sizes"])
    s2d_ps = stats(f["s2d_sizes"])
    d2s_ps = stats(f["d2s_sizes"])

    # piats
    bid_piat = piat(f["timestamps"])
    s2d_piat = piat(f["s2d_ts"])
    d2s_piat = piat(f["d2s_ts"])

    # flags
    def sumflags(lst):
        return [
            sum(x[k] for x in lst) for k in
            ["syn","cwr","ece","urg","ack","psh","rst","fin"]
        ]

    bid_flags = sumflags(f["flags"])
    s2d_flags = sumflags(f["s2d_flags"])
    d2s_flags = sumflags(f["d2s_flags"])

    duration = f["last"] - f["first"]

    row = [
        f["sport"], f["dport"], f["proto"], 4, 0, 0,
        f["first"], f["last"], duration,
        len(f["timestamps"]), sum(f["sizes"]),
        f["s2d_ts"][0] if f["s2d_ts"] else 0,
        f["s2d_ts"][-1] if f["s2d_ts"] else 0,
        (f["s2d_ts"][-1] - f["s2d_ts"][0]) if len(f["s2d_ts"])>1 else 0,
        len(f["s2d_ts"]), sum(f["s2d_sizes"]),
        f["d2s_ts"][0] if f["d2s_ts"] else 0,
        f["d2s_ts"][-1] if f["d2s_ts"] else 0,
        (f["d2s_ts"][-1] - f["d2s_ts"][0]) if len(f["d2s_ts"])>1 else 0,
        len(f["d2s_ts"]), sum(f["d2s_sizes"]),
        *bid_ps, *s2d_ps, *d2s_ps,
        *bid_piat, *s2d_piat, *d2s_piat,
        *bid_flags, *s2d_flags, *d2s_flags
    ]

    return row


# ============================================
# Main Sniffer Loop (returns stream of vectors)
# ============================================
def start_sniffer(callback):
    def handle(pkt):
        key = update_flow(pkt)
        if key is None:
            return
        f = flows[key]
        vector = compute_feature_vector(f)
        callback(vector)

    sniff(prn=handle, store=False)

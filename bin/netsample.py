#!/usr/bin/env python3
"""1 Hz network trace, flushed and fsynced every line.

    bin/netsample.py <out.tsv> [stop-file]

Stops when the stop file appears, or after PYG_NET_MAX_HOURS (24 by default) so a
sampler whose run died cannot outlive it by days.

Why fsync every sample: two runs ended with the node being power-cut to get the
house back online, and anything sitting in a buffer went with it. A line on disk is
the only kind that survives.

What it answers that the Spark event log cannot:
  - real bytes on the wire (the event log's Bytes Read is not wire bytes -- it
    reported 8.4 TB for a 2.8 h run, which no home line can carry)
  - how many sockets are open to the internet at once, and how fast new ones are
    created, which is what a consumer router's state table runs out of

Interfaces come from the environment so this file names no hardware:
  PYG_NET_WAN_IFACE      the uplink. Default: whichever interface holds the
                         default route, read from /proc/net/route.
  PYG_NET_FABRIC_IFACES  comma-separated node-to-node interfaces, counted
                         together and reported apart from the uplink. Optional.
"""
import os
import sys
import time

PORT443 = "01BB"
MAX_HOURS = float(os.environ.get("PYG_NET_MAX_HOURS", "24"))


def default_iface(path="/proc/net/route"):
    """The interface holding the default route, or "" if there is none."""
    try:
        with open(path) as fh:
            next(fh, None)
            for line in fh:
                f = line.split()
                # destination 0.0.0.0, and the route is up (RTF_UP = 0x1)
                if len(f) > 3 and f[1] == "00000000" and int(f[3], 16) & 1:
                    return f[0]
    except OSError:
        pass
    return ""


def netdev(path="/proc/net/dev"):
    """iface -> (rx_bytes, rx_pkts, tx_bytes, tx_pkts)"""
    out = {}
    with open(path) as fh:
        for line in fh:
            if ":" not in line:
                continue
            name, _, rest = line.partition(":")
            f = rest.split()
            if len(f) >= 10:
                out[name.strip()] = (int(f[0]), int(f[1]), int(f[8]), int(f[9]))
    return out


def tcpstates(paths=("/proc/net/tcp", "/proc/net/tcp6")):
    """(established to :443, syn_sent, time_wait, total)"""
    est443 = syn = tw = total = 0
    for path in paths:
        try:
            with open(path) as fh:
                next(fh, None)
                for line in fh:
                    f = line.split()
                    if len(f) < 4:
                        continue
                    total += 1
                    st = f[3]
                    if st == "01":
                        if f[2].rsplit(":", 1)[-1].upper() == PORT443:
                            est443 += 1
                    elif st == "02":
                        syn += 1
                    elif st == "06":
                        tw += 1
        except OSError:
            pass
    return est443, syn, tw, total


def conntrack(path="/proc/sys/net/netfilter/nf_conntrack_count"):
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except OSError:
        return -1


def main():
    if not 2 <= len(sys.argv) <= 3:
        print("usage: %s <out.tsv> [stop-file]" % os.path.basename(sys.argv[0]),
              file=sys.stderr)
        return 2
    out = sys.argv[1]
    # The stop file is an argument because it must be per-run. A shared one in the
    # run directory let a killed attempt's cleanup stop the live run's samplers:
    # node2's trace then ended before the run began, and the end-of-run copy
    # brought that file back as if it described this run.
    stop = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(out), "STOP")

    wan = os.environ.get("PYG_NET_WAN_IFACE") or default_iface()
    fabrics = [s for s in os.environ.get("PYG_NET_FABRIC_IFACES", "").split(",") if s.strip()]

    cols = ["ts", "iso", "wan_rx", "wan_tx", "wan_rxp", "wan_txp",
            "fab_rx", "fab_tx", "est443", "syn_sent", "timewait", "tcp_all", "ct"]

    fh = open(out, "a", buffering=1)
    if fh.tell() == 0:
        fh.write("\t".join(cols) + "\n")

    deadline = time.time() + MAX_HOURS * 3600
    while not os.path.exists(stop) and time.time() < deadline:
        nd = netdev()
        w = nd.get(wan, (0, 0, 0, 0))
        fr = ft = 0
        for f in fabrics:
            v = nd.get(f.strip())
            if v:
                fr += v[0]
                ft += v[2]
        e, s, t, a = tcpstates()
        row = [f"{time.time():.3f}", time.strftime("%H:%M:%S"),
               w[0], w[2], w[1], w[3], fr, ft, e, s, t, a, conntrack()]
        fh.write("\t".join(str(x) for x in row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        time.sleep(1)

    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

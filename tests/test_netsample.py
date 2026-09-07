"""Pin the parsing in `bin/netsample.py`, and the stop file that has to be per-run.

WHY THIS IS WORTH A TEST
------------------------
This trace answers the one question the Spark event log cannot: how many bytes really
crossed the uplink, and how many sockets were open at once. Its numbers went into the
decision to stage sources locally, and into two write-ups about a network collapse.

Everything it reports is column arithmetic over /proc. A field read one position off
does not raise -- it produces a plausible number that is wrong, and there is nothing
to compare it against afterwards. So the offsets are pinned against real /proc text.

The stop file is here for a different reason. It used to be a fixed name inside the
run directory, shared by every run that used that directory, and a killed attempt's
cleanup would stop the live run's sampler on the other node. The end-of-run copy then
brought the dead attempt's trace back, and it was summarised as this run's. Taking the
path as an argument is what makes that impossible; this pins that it is honoured even
when the old shared name is sitting right there.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "bin" / "netsample.py"

# One real line each, so the offsets are pinned against the format as it ships.
PROC_NET_DEV = """\
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets
    lo: 1000      10    0    0    0     0          0         0     2000      20    0    0    0     0       0          0
  eth0: 553491108  9812  0    0    0     0          0         0  814791147  7311   0    0    0     0       0          0
  fab0: 111       1     0    0    0     0          0         0      222       2    0    0    0     0       0          0
"""

# Destination 00000000 is the default route; Flags bit 0x1 is RTF_UP.
PROC_NET_ROUTE = """\
Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT
down0\t00000000\t0100A8C0\t0000\t0\t0\t0\t00000000\t0\t0\t0
eth1\t00C0A8C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0
eth0\t00000000\t0100A8C0\t0003\t0\t0\t100\t00000000\t0\t0\t0
"""

# sl  local_address rem_address st ...   st 01 = established, 02 = syn_sent, 06 = time_wait
PROC_NET_TCP = """\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid
   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000
   1: 0100007F:B3C6 5DB8D822:01BB 01 00000000:00000000 00:00000000 00000000  1000
   2: 0100007F:B3C7 5DB8D822:01BB 01 00000000:00000000 00:00000000 00000000  1000
   3: 0100007F:01BB 5DB8D822:C001 01 00000000:00000000 00:00000000 00000000  1000
   4: 0100007F:B3C8 5DB8D822:0050 02 00000000:00000000 00:00000000 00000000  1000
   5: 0100007F:B3C9 5DB8D822:0050 06 00000000:00000000 00:00000000 00000000  1000
"""


def _load_tool():
    """Import by path -- bin/ is scripts, not an importable package."""
    spec = importlib.util.spec_from_file_location("netsample_tool", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# --------------------------------------------------------------------------- #
# Which interface is the uplink
# --------------------------------------------------------------------------- #

def test_default_iface_reads_the_up_default_route(tool, tmp_path):
    """Two rows here are traps: a default route that is DOWN, and an up route that
    is not the default. Taking either gives a trace of the wrong interface."""
    assert tool.default_iface(_write(tmp_path, "route", PROC_NET_ROUTE)) == "eth0"


def test_default_iface_is_empty_when_there_is_no_default_route(tool, tmp_path):
    """No uplink is a state to report zeroes for, not to crash the run's tracing on."""
    lines = [ln for ln in PROC_NET_ROUTE.splitlines(True) if "\t00000000\t" not in ln]
    assert tool.default_iface(_write(tmp_path, "route", "".join(lines))) == ""


def test_default_iface_survives_a_missing_file(tool, tmp_path):
    assert tool.default_iface(str(tmp_path / "not-there")) == ""


# --------------------------------------------------------------------------- #
# Byte counters
# --------------------------------------------------------------------------- #

def test_netdev_columns_are_rx_bytes_rx_packets_tx_bytes_tx_packets(tool, tmp_path):
    """/proc/net/dev puts transmit bytes at field 8, after eight receive fields.

    Off by one here reads errs or drop as the uplink's traffic, which is a small
    plausible number rather than an error.
    """
    nd = tool.netdev(_write(tmp_path, "dev", PROC_NET_DEV))
    assert nd["eth0"] == (553491108, 9812, 814791147, 7311)
    assert set(nd) == {"lo", "eth0", "fab0"}


# --------------------------------------------------------------------------- #
# Socket states
# --------------------------------------------------------------------------- #

def test_tcpstates_counts_the_remote_port_not_the_local_one(tool, tmp_path):
    """Sockets TO :443 are the internet-facing ones. A local :443 is somebody
    connecting to us, and counting it inflates the number the router's state table
    is being judged by."""
    tcp = _write(tmp_path, "tcp", PROC_NET_TCP)
    est443, syn, tw, total = tool.tcpstates((tcp,))
    assert est443 == 2          # rows 1 and 2; row 3 is a LOCAL :443
    assert syn == 1
    assert tw == 1
    assert total == 6


def test_tcpstates_tolerates_a_missing_address_family(tool, tmp_path):
    """A host with IPv6 disabled has no /proc/net/tcp6, which is not a failure."""
    tcp = _write(tmp_path, "tcp", PROC_NET_TCP)
    assert tool.tcpstates((tcp, str(tmp_path / "tcp6")))[0] == 2


def test_conntrack_reports_minus_one_when_the_counter_is_absent(tool, tmp_path):
    """Distinguishable from a real zero, which is what a load-bearing column needs."""
    assert tool.conntrack(str(tmp_path / "not-there")) == -1


# --------------------------------------------------------------------------- #
# The stop file
# --------------------------------------------------------------------------- #

def test_the_stop_file_argument_wins_over_the_old_shared_name(tmp_path):
    """A shared STOP in the run directory must not stop a sampler that was given its
    own. That is the whole defect: one run's cleanup ending another run's trace."""
    out = tmp_path / "net-node-a-20260101T000000Z.tsv"
    (tmp_path / "STOP").touch()                      # the old shared name, set
    mine = tmp_path / "STOP-20260101T000000Z"        # this run's, not set

    r = subprocess.run(
        [sys.executable, str(TOOL), str(out), str(mine)],
        capture_output=True, text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin", "PYG_NET_MAX_HOURS": "0.0001"},
    )
    assert r.returncode == 0, r.stderr
    lines = out.read_text().splitlines()
    assert lines[0].startswith("ts\tiso\twan_rx")
    assert len(lines) >= 2, "the shared STOP stopped a sampler that was given its own"


def test_its_own_stop_file_ends_it(tmp_path):
    out = tmp_path / "net.tsv"
    stop = tmp_path / "STOP-20260101T000000Z"
    stop.touch()

    r = subprocess.run([sys.executable, str(TOOL), str(out), str(stop)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert out.read_text().splitlines() == [
        "ts\tiso\twan_rx\twan_tx\twan_rxp\twan_txp\tfab_rx\tfab_tx"
        "\test443\tsyn_sent\ttimewait\ttcp_all\tct"
    ]


def test_usage_is_rejected_rather_than_guessed(tmp_path):
    r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 2
    assert "usage:" in r.stderr

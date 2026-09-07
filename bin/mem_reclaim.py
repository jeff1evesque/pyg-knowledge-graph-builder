#!/usr/bin/env python3
"""Push the file cache out so the next executor can size a real RMM pool.

    bin/mem_reclaim.py <target_free_gb>

RAPIDS opens its pool as (free - reserve) x allocFraction, and on a unified-memory
host "free" is MemFree from /proc/meminfo. A finished run leaves its file cache
behind and the kernel holds on to it until something asks for the memory, so MemFree
sits near zero for hours while MemAvailable still reads over 100 G. That gap is how
the 2026-09-03 run launched into ~1.3 G free and came up with a 337 MB pool instead
of 24 GB.

Waiting does not close the gap. This asks for the memory instead: allocate until
MemFree plus what we are holding reaches the target, then exit. Handing our pages
back returns everything the kernel had to reclaim to give them to us, so MemFree
lands near the target and stays there.

Exit 0 if MemFree reached the target, 1 if it did not, so a launcher can gate on it.
"""

import os
import sys
import time

CHUNK = 512 << 20      # one allocation step
PAGE = 4096            # touch this often, so every page is really ours
FLOOR_GB = 8           # stop if MemAvailable gets this low; we are past cache
DEADLINE = 600         # seconds


def meminfo(path="/proc/meminfo"):
    out = {}
    with open(path) as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            out[key] = int(rest.split()[0]) / (1000 * 1000)   # kB -> GB
    return out


def main():
    if len(sys.argv) != 2:
        print("usage: %s <target_free_gb>" % os.path.basename(sys.argv[0]), file=sys.stderr)
        return 2
    target = float(sys.argv[1])

    # If we ask for too much, the kernel should pick us and nothing else.
    try:
        with open("/proc/self/oom_score_adj", "w") as fh:
            fh.write("1000")
    except OSError:
        pass

    start = meminfo()
    if start["MemFree"] >= target:
        print("MemFree %.1fG, already at or above target %.0fG -- nothing to do"
              % (start["MemFree"], target))
        return 0

    # Write back dirty pages first; a dirty page cannot be dropped.
    os.sync()

    # Aim past the target. Handing back N GB does not raise MemFree by a full N:
    # page tables and the allocator keep a little, and measured on 2026-09-03 the
    # shortfall was 2.2 G on a 99 G ask. Overshoot so the gate is not lost to it.
    aim = target + 4

    held = 0.0
    chunks = []
    began = time.time()
    stopped = ""
    while True:
        mem = meminfo()
        if mem["MemFree"] + held >= aim:
            break
        if mem["MemAvailable"] < FLOOR_GB:
            stopped = "MemAvailable fell to %.1fG" % mem["MemAvailable"]
            break
        if time.time() - began > DEADLINE:
            stopped = "gave up after %ds" % DEADLINE
            break
        block = bytearray(CHUNK)
        for off in range(0, CHUNK, PAGE):
            block[off] = 1
        chunks.append(block)
        held += CHUNK / (1000 * 1000 * 1000)

    del chunks
    time.sleep(2)

    end = meminfo()
    note = " (%s)" % stopped if stopped else ""
    print("MemFree %.1fG -> %.1fG, target %.0fG, asked for %.0fG%s"
          % (start["MemFree"], end["MemFree"], target, held, note))
    return 0 if end["MemFree"] >= target else 1


if __name__ == "__main__":
    sys.exit(main())

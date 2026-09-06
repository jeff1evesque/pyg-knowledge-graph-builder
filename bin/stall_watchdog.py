#!/usr/bin/env python3
"""
Watch a running Spark job for a stalled stage and capture the evidence that
proves what stalled it, without anyone having to be at the keyboard.

WHY THIS EXISTS
---------------
On 2026-09-06 the seed leg stalled at 168 of 169 tasks in stage 13 seven times,
and the one measurement that would have explained it -- a thread dump of the
executor holding the stuck task -- was never taken. Not because it was hard.
The stall is silent, so it looks like a dead run; three of those runs were
killed early on that assumption and every conclusion drawn from the resulting
logs was wrong. By the time anyone decided it was worth a dump, the job was
gone.

The window is the problem, not the command. A stall can sit for hours, but
somebody has to notice it while it is sitting. This watches instead.

HOW IT TELLS A STALL FROM A SLOW TASK
-------------------------------------
Quiet is ambiguous, and reading it as "hung" is what cost the runs above. A
stage near the end of its task set has few tasks left, so gaps between
completions are normal and get longer as it drains.

The measurement that is not ambiguous is how long the unfinished task has been
running, against how long tasks in that same stage actually take. On the runs
above, a stuck task had been running 165s where the median was 47s and the
longest that finished took 76s; another had been running 738s against a
median of 48s. Two of those runs were misread as healthy on quiet alone.

So the trigger is two-part:

  1. a stage has running tasks and its completed count has not moved for
     --stall-seconds, and
  2. at least one running task is older than --straggler-factor times the
     longest task that finished in that same stage

Part 2 is what makes it safe to run unattended. Part 1 on its own fires on any
stage that is merely slow.

A stage where nothing has finished has no such baseline. Staying quiet there
would miss a stage whose very first task hangs, so those still qualify -- but
only once a running task has itself been going for --stall-seconds. Without that
floor, any stage that has not finished its first task inside the threshold is
reported stalled, naming tasks that started moments earlier. That happened twice
on 2026-09-06, on a stage with one task and on a 200-task ``toPandas``; both
went on to finish, and both captures named tasks under a second old. It cost a
run that had actually succeeded its outcome report, because the harness around
this tool reads a capture as proof of a stall.

WHAT IT CANNOT TELL YOU
-----------------------
It captures, it does not diagnose. A dump showing threads parked in
``PrioritySemaphore.acquire`` is not by itself a finding -- healthy runs park
there too, because that is how the plugin queues work onto the GPU. What makes
a dump evidence is that the *named stuck tasks* are in it, that they are still
in it 30 seconds later, and what holds the permits they are waiting for. The
capture records the stuck task ids alongside the dumps so that comparison can
actually be made.

It also cannot see a driver that is wedged before any stage is submitted, since
it works from the stage list.

USAGE
-----
Point it at the driver and leave it running beside the job::

    bin/stall_watchdog.py --host <driver-host> --out ~/pyg-runs/<issue>/stalls

It re-resolves the application between legs, so one invocation covers a whole
notebook run. Everything it does is a read: HTTP GETs against the driver UI and
two local files. It never touches the job.
"""
import argparse
import html
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

TIMEOUT = 20


def log(msg):
    """One line per event, flushed, so this can be tailed or monitored."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get(url, as_json=True):
    """GET a URL. Returns None on any failure -- a blip must not kill the loop."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as fh:
            raw = fh.read()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not as_json:
        return raw.decode("utf-8", "replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def parse_launch_time(value):
    """Spark serialises launchTime as '2026-09-06T10:00:47.244GMT'. Give back ms."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace("GMT", "+0000").replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt).timestamp() * 1000.0
        except ValueError:
            continue
    return None


def dump_to_text(page):
    """Turn the thread dump page into the stacks, one frame per line."""
    if not page:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", page)
    text = re.sub(r"</(tr|div|p|li)>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)


class Api:
    def __init__(self, host, port):
        self.base = f"http://{host}:{port}"

    def app_id(self):
        apps = get(f"{self.base}/api/v1/applications")
        if not apps:
            return None
        return apps[0].get("id")

    def active_stages(self, app):
        stages = get(f"{self.base}/api/v1/applications/{app}/stages?status=ACTIVE")
        return stages or []

    def tasks(self, app, stage_id, attempt):
        url = (
            f"{self.base}/api/v1/applications/{app}"
            f"/stages/{stage_id}/{attempt}/taskList?length=100000"
        )
        return get(url) or []

    def executor_ids(self, app):
        execs = get(f"{self.base}/api/v1/applications/{app}/executors")
        if not execs:
            return ["driver", "0", "1"]
        return [e.get("id") for e in execs if e.get("id")]

    def thread_dump(self, executor_id):
        url = f"{self.base}/executors/threadDump/?executorId={executor_id}"
        return get(url, as_json=False)


def classify(tasks, straggler_factor, no_baseline_floor_ms=0.0):
    """Split a stage's tasks and decide whether any running one is past the max.

    Returns (running, finished_ms, stuck) where stuck is the running tasks
    older than straggler_factor x the longest task that finished.

    When nothing has finished there is no maximum to compare against. Those
    stages still qualify, because a stage whose first task hangs never completes
    one -- but only for tasks that have themselves been running longer than
    no_baseline_floor_ms. The default of 0 names every running task, which is the
    eager reading and what the pure decision means on its own; main passes
    --stall-seconds so an unattended run cannot report a task that started
    seconds ago.
    """
    now_ms = time.time() * 1000.0
    running, finished_ms = [], []
    for t in tasks:
        status = (t.get("status") or "").upper()
        if status == "RUNNING":
            launch = parse_launch_time(t.get("launchTime"))
            if launch is not None:
                running.append((t, now_ms - launch))
        elif status == "SUCCESS" and t.get("duration"):
            finished_ms.append(float(t["duration"]))

    if not finished_ms:
        # No completed task to compare against, so the task's own age is all
        # there is to go on.
        return running, finished_ms, [
            (t, age) for t, age in running if age >= no_baseline_floor_ms
        ]

    ceiling = max(finished_ms) * straggler_factor
    return running, finished_ms, [(t, age) for t, age in running if age > ceiling]


def local_facts():
    """The two host readings worth having next to a dump."""
    facts = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(("MemFree:", "MemAvailable:", "SwapFree:")):
                    key, value = line.split(":", 1)
                    facts[key] = value.strip()
    except OSError:
        pass
    try:
        facts["nvidia-smi"] = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        facts["nvidia-smi"] = "(unavailable)"
    return facts


def capture(api, app, stage, tasks, stuck, out_dir, rounds, gap):
    """Write the whole bundle: dumps twice, the stuck task list, host state."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = os.path.join(out_dir, f"stall-{stamp}-stage{stage['stageId']}")
    os.makedirs(target, exist_ok=True)

    summary = {
        "captured_utc": stamp,
        "app": app,
        "stage_id": stage.get("stageId"),
        "attempt_id": stage.get("attemptId"),
        "stage_name": stage.get("name"),
        "num_tasks": stage.get("numTasks"),
        "num_complete": stage.get("numCompleteTasks"),
        "num_active": stage.get("numActiveTasks"),
        "stuck_tasks": [
            {
                "task_id": t.get("taskId"),
                "index": t.get("index"),
                "executor_id": t.get("executorId"),
                "host": t.get("host"),
                "running_for_s": round(age / 1000.0, 1),
            }
            for t, age in stuck
        ],
    }

    durations = [float(t["duration"]) / 1000.0 for t in tasks
                 if (t.get("status") or "").upper() == "SUCCESS" and t.get("duration")]
    if durations:
        summary["finished_task_seconds"] = {
            "count": len(durations),
            "median": round(statistics.median(durations), 1),
            "max": round(max(durations), 1),
        }

    executors = api.executor_ids(app)
    log(f"capturing {rounds} round(s) of dumps from {len(executors)} executors -> {target}")

    for round_no in range(1, rounds + 1):
        for executor_id in executors:
            page = api.thread_dump(executor_id)
            if not page:
                log(f"  round {round_no} executor {executor_id}: no dump returned")
                continue
            stem = os.path.join(target, f"round{round_no}-exec-{executor_id}")
            with open(stem + ".html", "w") as fh:
                fh.write(page)
            with open(stem + ".txt", "w") as fh:
                fh.write(dump_to_text(page))
            log(f"  round {round_no} executor {executor_id}: {len(page)} bytes")
        if round_no < rounds:
            time.sleep(gap)

    with open(os.path.join(target, "tasks.json"), "w") as fh:
        json.dump(tasks, fh, indent=2)
    with open(os.path.join(target, "host.json"), "w") as fh:
        json.dump(local_facts(), fh, indent=2)
    with open(os.path.join(target, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    log(f"capture complete -> {target}")
    for entry in summary["stuck_tasks"]:
        log(
            f"  stuck: partition {entry['index']} on executor {entry['executor_id']}"
            f", running {entry['running_for_s']}s"
        )
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--host", default=os.environ.get("SPARK_DRIVER_HOST", "127.0.0.1"),
                    help="driver host (default $SPARK_DRIVER_HOST, else 127.0.0.1)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SPARK_UI_PORT", "4040")))
    ap.add_argument("--out", default="./stall-dumps", help="where to write captures")
    ap.add_argument("--poll", type=int, default=15, help="seconds between checks")
    ap.add_argument("--stall-seconds", type=int, default=300,
                    help="a stage must go this long with no completion to qualify")
    ap.add_argument("--straggler-factor", type=float, default=1.5,
                    help="a running task must exceed this x the stage's longest finished task")
    ap.add_argument("--dump-rounds", type=int, default=2, help="dumps per capture")
    ap.add_argument("--dump-gap", type=int, default=30, help="seconds between rounds")
    ap.add_argument("--max-captures", type=int, default=3, help="stop after this many")
    ap.add_argument("--exit-after-idle", type=int, default=0,
                    help="exit after this many seconds with no application (0 = never)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    api = Api(args.host, args.port)
    log(f"watching {api.base}, stall threshold {args.stall_seconds}s, writing to {args.out}")

    # (stage_id, attempt) -> [completed count, when it last changed]
    seen = {}
    captured = set()
    captures = 0
    idle_since = None

    while True:
        app = api.app_id()
        if not app:
            if idle_since is None:
                idle_since = time.time()
                log("no application up (between legs, or the run is over)")
            elif args.exit_after_idle and time.time() - idle_since > args.exit_after_idle:
                log("no application for too long, exiting")
                return 0
            seen.clear()
            time.sleep(args.poll)
            continue

        if idle_since is not None:
            log(f"application {app} is up")
            idle_since = None

        now = time.time()
        for stage in api.active_stages(app):
            key = (app, stage.get("stageId"), stage.get("attemptId"))
            done = stage.get("numCompleteTasks", 0)
            active = stage.get("numActiveTasks", 0)

            if key not in seen or seen[key][0] != done:
                seen[key] = [done, now]
                continue
            if active <= 0 or key in captured:
                continue

            frozen_for = now - seen[key][1]
            if frozen_for < args.stall_seconds:
                continue

            tasks = api.tasks(app, stage["stageId"], stage["attemptId"])
            running, finished, stuck = classify(
                tasks, args.straggler_factor, args.stall_seconds * 1000.0
            )
            # Which of the two rules applied has to be in the log. Reading these
            # lines wrong is how the stall they exist for was misdiagnosed.
            if not stuck:
                if finished:
                    why = (f"no task is past {args.straggler_factor}x the "
                           f"{max(finished) / 1000.0:.0f}s max")
                else:
                    why = ("nothing has finished, and no task has itself been "
                           f"running {args.stall_seconds}s")
                log(
                    f"stage {stage['stageId']}: {done} done, {active} running, no completion "
                    f"for {frozen_for:.0f}s -- but {why}. Slow, not stuck."
                )
                continue

            basis = ("past the stage maximum" if finished else
                     f"running past {args.stall_seconds}s with nothing finished")
            log(
                f"STALL: stage {stage['stageId']} ({stage.get('name', '')[:60]}) "
                f"{done}/{stage.get('numTasks')} done, no completion for {frozen_for:.0f}s, "
                f"{len(stuck)} task(s) {basis}"
            )
            capture(api, app, stage, tasks, stuck, args.out,
                    args.dump_rounds, args.dump_gap)
            captured.add(key)
            captures += 1
            if captures >= args.max_captures:
                log(f"reached --max-captures ({args.max_captures}), exiting")
                return 0

        time.sleep(args.poll)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("stopped")
        sys.exit(130)

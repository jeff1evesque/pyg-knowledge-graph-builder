#!/usr/bin/env python3
"""
Generate a combined test report (JSON + matching HTML) from pytest JUnit XML.

Reads one or more JUnit XML files, each tagged with a human label + execution
mode, and writes ``report.json`` and ``report.html`` (same content, two
formats) into the output directory, OVERWRITING any previous pair.

This is a LOCAL-only tool: the end-to-end smoke suite it summarizes is too
heavy for the GitHub Actions runners (see the Testing section of README.md), so
the report is produced on a developer machine, not in CI. ``bin/generate_report.sh``
is the usual entry point; this script is the underlying renderer.

Usage:
    generate_test_report.py OUT_DIR LABEL=MODE=PATH [LABEL=MODE=PATH ...]

Example:
    generate_test_report.py reports/tests \\
        "Fast (unit) suite=CPU=fast.xml" \\
        "End-to-end pipeline smoke=CPU=e2e_cpu.xml" \\
        "End-to-end pipeline smoke=GPU (RAPIDS Accelerator)=e2e_gpu.xml"
"""
import html
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _rapids_version() -> str:
    """RAPIDS Accelerator version parsed from the jar name, or ''."""
    jars = sorted(Path("/opt/spark/jars").glob("rapids-4-spark_*.jar"))
    if not jars:
        return ""
    # rapids-4-spark_2.12-25.02.0-cuda12-<arch>.jar -> "25.02.0 (cuda12)".
    # The architecture suffix is intentionally dropped.
    parts = jars[-1].stem.split("-")
    ver = parts[3] if len(parts) > 3 else "?"
    cuda = next((p for p in parts if p.startswith("cuda")), "")
    return f"{ver} ({cuda})" if cuda else ver


def _env(include_rapids: bool) -> dict:
    import platform
    info = {
        "python": platform.python_version(),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git("rev-parse", "--short", "HEAD"),
    }
    try:
        import pyspark
        info["pyspark"] = pyspark.__version__
    except Exception:
        info["pyspark"] = "unknown"
    if include_rapids:
        rv = _rapids_version()
        if rv:
            info["rapids"] = rv
    return info


def _parse_suite(path: str) -> dict:
    root = ET.parse(path).getroot()
    ts = root if root.tag == "testsuite" else root.find("testsuite")
    cases = []
    for tc in ts.findall("testcase"):
        outcome = "passed"
        if tc.find("failure") is not None or tc.find("error") is not None:
            outcome = "failed"
        elif tc.find("skipped") is not None:
            outcome = "skipped"
        cls = tc.attrib.get("classname", "").split(".")[-1]
        cases.append({
            "name": (f"{cls}::" if cls else "") + tc.attrib["name"],
            "outcome": outcome,
            "duration_s": round(float(tc.attrib.get("time", 0.0)), 3),
        })
    a = ts.attrib
    passed = sum(1 for c in cases if c["outcome"] == "passed")
    return {
        "total": int(a.get("tests", len(cases))),
        "passed": passed,
        "failed": int(a.get("failures", 0)) + int(a.get("errors", 0)),
        "skipped": int(a.get("skipped", 0)),
        "duration_s": round(float(a.get("time", 0.0)), 3),
        "tests": cases,
    }


def _build_name(artifact_dir: Path, path: Path) -> str:
    """Identify the run a slot_mapping.json came from.

    Artifacts are laid out as ``<run>/pyg/year=/month=/metadata/<file>``, so
    everything ABOVE the ``pyg/`` output tree is the run's identity. That is
    one component for the per-test runs (``test_full_ntriples0``) and two for
    the reproducibility twins (``reproducibility0/run1``).

    Previously this searched the ancestors for a ``test_``-prefixed directory
    and fell back to the parent directory name. The twin runs have no such
    ancestor -- their basetemp is ``reproducibility0``, named by
    tmp_path_factory rather than by a test function -- so both fell back to
    ``metadata``, the directory every slot mapping happens to sit in. The
    report showed two identically-labelled rows and lost which twin was which.
    Their NUMBERS matching is correct and is the point of
    test_output_is_reproducible; only the label was wrong.
    """
    try:
        parts = path.relative_to(artifact_dir).parts
    except ValueError:
        return path.parent.name
    if "pyg" in parts:
        head = parts[:parts.index("pyg")]
        if head:
            return "/".join(head)
    return path.parent.name


def _encoding_capacity(artifact_dir: str) -> list[dict]:
    """class_identity capacity from every slot_mapping.json a run produced.

    Tracked in the report rather than only asserted in a test because it
    DEGRADES rather than breaks: the class count grows with every source
    added while the segment width does not, and past the segment width class
    identity stops being linearly recoverable although every code stays
    distinct. A number that drifts toward a cliff needs to be visible on each
    run, not discovered by a test failing after the cliff.
    """
    rows = []
    for path in sorted(Path(artifact_dir).rglob("slot_mapping.json")):
        try:
            ci = json.loads(path.read_text())["collision_report"][
                "class_identity"
            ]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        total = ci.get("total_classes")
        dim = ci.get("segment_dim")
        if not dim:
            continue   # slot mapping < 1.1 cannot report capacity
        rows.append({
            "build": _build_name(Path(artifact_dir), path),
            "total_classes": total,
            "segment_dim": dim,
            "headroom_classes": ci.get("headroom_classes"),
            "used_pct": round(100.0 * total / dim, 1) if total else 0.0,
            "distinct_codes": ci.get("distinct_codes"),
            "classes_sharing_a_code": len(
                ci.get("classes_sharing_a_code") or []
            ),
            "linearly_separable": ci.get("linearly_separable"),
        })
    return rows


def build_report(
    specs: list[tuple[str, str, str]], artifact_dir: str = "",
) -> dict:
    suites = []
    for label, mode, path in specs:
        s = _parse_suite(path)
        s["name"] = label
        s["mode"] = mode
        suites.append(s)
    has_gpu = any(
        "GPU" in s["mode"] or "RAPIDS" in s["mode"] for s in suites
    )
    totals = {
        k: sum(s[k] for s in suites)
        for k in ("total", "passed", "failed", "skipped")
    }
    totals["duration_s"] = round(sum(s["duration_s"] for s in suites), 3)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": "pyg-knowledge-graph-builder",
        "environment": _env(include_rapids=has_gpu),
        "has_gpu_suite": has_gpu,
        "totals": totals,
        "suites": suites,
        "encoding_capacity": (
            _encoding_capacity(artifact_dir) if artifact_dir else []
        ),
    }


# ------------------------------------------------------------------ HTML ----

def _fmt(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m)}m {s:04.1f}s" if m else f"{s:.2f}s"


def render_html(rep: dict) -> str:
    env = rep["environment"]
    t = rep["totals"]
    status_ok = t["failed"] == 0
    badge = "PASSING" if status_ok else "FAILING"

    rows = []
    for s in rep["suites"]:
        ok = s["failed"] == 0
        rows.append(f"""
      <tr class="suite {'ok' if ok else 'bad'}">
        <td class="s-name">{html.escape(s['name'])}</td>
        <td><span class="mode">{html.escape(s['mode'])}</span></td>
        <td class="num">{s['total']}</td>
        <td class="num pass">{s['passed']}</td>
        <td class="num fail">{s['failed']}</td>
        <td class="num skip">{s['skipped']}</td>
        <td class="num">{_fmt(s['duration_s'])}</td>
      </tr>""")

    detail = []
    for s in rep["suites"]:
        slow = sorted(s["tests"], key=lambda c: -c["duration_s"])[:10]
        items = "".join(
            f"<li><span class='dot {c['outcome']}'></span>"
            f"<code>{html.escape(c['name'])}</code>"
            f"<span class='t'>{_fmt(c['duration_s'])}</span></li>"
            for c in slow
        )
        show_all = s["total"] <= 10
        detail.append(f"""
      <details {'open' if s['total'] <= 5 else ''}>
        <summary>{html.escape(s['name'])} &mdash; <span class="mode">{html.escape(s['mode'])}</span>
          <span class="chip">{s['passed']}/{s['total']} passed</span></summary>
        <p class="hint">{'All ' + str(s['total']) + ' tests' if show_all else 'Top 10 by duration (of ' + str(s['total']) + ')'}:</p>
        <ul class="tests">{items}</ul>
      </details>""")

    # Encoding capacity — a number that drifts toward a cliff, so it belongs on
    # every report rather than only in a test that fails after the cliff.
    capacity_section = ""
    if rep.get("encoding_capacity"):
        cap_rows = []
        for c in rep["encoding_capacity"]:
            ok = c["linearly_separable"] and not c["classes_sharing_a_code"]
            cap_rows.append(f"""
      <tr class="suite {'ok' if ok else 'bad'}">
        <td class="s-name">{html.escape(str(c['build']))}</td>
        <td class="num">{c['total_classes']}</td>
        <td class="num">{c['segment_dim']}</td>
        <td class="num">{c['used_pct']}%</td>
        <td class="num">{c['headroom_classes']}</td>
        <td class="num {'pass' if ok else 'fail'}">
          {'separable' if ok else 'NOT separable'}</td>
      </tr>""")
        capacity_section = f"""
  <h2>Encoding capacity &mdash; class_identity</h2>
  <div class="card"><table>
    <thead><tr><th>Build</th><th class="num">Classes</th>
      <th class="num">Segment dims</th><th class="num">Used</th>
      <th class="num">Headroom</th><th class="num">Verdict</th></tr></thead>
    <tbody>{''.join(cap_rows)}</tbody>
  </table></div>
  <p class="hint">A d-dimensional segment separates at most d classes. The class
    count grows with every source added while the segment width does not, so
    headroom shrinking toward zero is the signal to re-tune
    <code>_SEG1_CLASS_IDENTITY_FRAC</code> or <code>vector_dim</code>.</p>"""

    envline = (
        f"branch <code>{html.escape(env['branch'])}</code> @ "
        f"<code>{html.escape(env['commit'])}</code> &middot; "
        f"Python {env['python']} &middot; PySpark {env['pyspark']}"
    )
    if env.get("rapids"):
        envline += f" &middot; RAPIDS {html.escape(env['rapids'])}"

    gpu_note = ""
    if rep.get("has_gpu_suite"):
        gpu_note = (
            "The GPU (RAPIDS) e2e run is a correctness sanity check, not a "
            "benchmark: on tiny fixtures GPU wall-clock exceeds CPU due to "
            "one-time kernel JIT compilation and per-operator launch/transfer "
            "overhead. Both modes assert identical results. &middot; "
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test Report &mdash; {html.escape(rep['repository'])}</title>
<style>
  :root {{ color-scheme: light dark;
    --bg:#f6f7f9; --card:#fff; --ink:#1b1f24; --mut:#5b6570; --line:#e3e7ec;
    --ok:#137a3d; --okbg:#e6f4ea; --bad:#c02636; --skip:#8a6d00; --accent:#2c5cff; }}
  @media (prefers-color-scheme: dark) {{ :root {{
    --bg:#0d1117; --card:#161b22; --ink:#e6edf3; --mut:#93a1b1; --line:#28303a;
    --ok:#3fb950; --okbg:#0f2c1a; --bad:#f85149; --skip:#d29922; --accent:#589bff; }} }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .sub {{ color:var(--mut); font-size:13px; margin-bottom:24px; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.92em; }}
  .hero {{ display:flex; align-items:center; gap:16px; background:var(--card);
    border:1px solid var(--line); border-radius:14px; padding:20px 24px; margin-bottom:20px; flex-wrap:wrap; }}
  .badge {{ font-weight:700; letter-spacing:.05em; font-size:13px; padding:6px 14px;
    border-radius:999px; background:var(--okbg); color:var(--ok); }}
  .badge.bad {{ background:transparent; color:var(--bad); border:1px solid var(--bad); }}
  .tiles {{ display:flex; gap:14px; flex-wrap:wrap; margin-left:auto; }}
  .tile {{ text-align:center; min-width:66px; }}
  .tile .v {{ font-size:26px; font-weight:700; line-height:1; }}
  .tile .k {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); margin-top:4px; }}
  .v.pass {{ color:var(--ok); }} .v.fail {{ color:var(--bad); }} .v.skip {{ color:var(--skip); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:8px 4px; margin-bottom:20px; overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; min-width:560px; }}
  th,td {{ text-align:left; padding:11px 14px; border-bottom:1px solid var(--line); }}
  th {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--mut); font-weight:600; }}
  tr:last-child td {{ border-bottom:none; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .s-name {{ font-weight:600; }}
  .num.pass {{ color:var(--ok); font-weight:600; }}
  .num.fail {{ color:var(--bad); }} .num.skip {{ color:var(--skip); }}
  .mode {{ font-size:12px; color:var(--mut); background:var(--bg);
    border:1px solid var(--line); border-radius:6px; padding:2px 8px; white-space:nowrap; }}
  details {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:6px 16px; margin-bottom:12px; }}
  summary {{ cursor:pointer; font-weight:600; padding:8px 0; }}
  .chip {{ float:right; font-weight:600; font-size:12px; color:var(--ok);
    background:var(--okbg); border-radius:999px; padding:2px 10px; }}
  .hint {{ color:var(--mut); font-size:12px; margin:4px 0 8px; }}
  ul.tests {{ list-style:none; margin:0 0 8px; padding:0; }}
  ul.tests li {{ display:flex; align-items:center; gap:10px; padding:5px 0;
    border-top:1px solid var(--line); font-size:13px; }}
  ul.tests li code {{ flex:1; word-break:break-all; }}
  ul.tests .t {{ color:var(--mut); font-variant-numeric:tabular-nums; }}
  .dot {{ width:9px; height:9px; border-radius:50%; flex:none; background:var(--ok); }}
  .dot.failed {{ background:var(--bad); }} .dot.skipped {{ background:var(--skip); }}
  h2 {{ font-size:15px; margin:26px 0 12px; }}
  footer {{ color:var(--mut); font-size:12px; margin-top:32px; }}
</style></head><body><div class="wrap">
  <h1>Test Report &mdash; {html.escape(rep['repository'])}</h1>
  <div class="sub">{envline}</div>

  <div class="hero">
    <span class="badge {'' if status_ok else 'bad'}">{badge}</span>
    <div class="tiles">
      <div class="tile"><div class="v">{t['total']}</div><div class="k">Total</div></div>
      <div class="tile"><div class="v pass">{t['passed']}</div><div class="k">Passed</div></div>
      <div class="tile"><div class="v fail">{t['failed']}</div><div class="k">Failed</div></div>
      <div class="tile"><div class="v skip">{t['skipped']}</div><div class="k">Skipped</div></div>
      <div class="tile"><div class="v">{_fmt(t['duration_s'])}</div><div class="k">Duration</div></div>
    </div>
  </div>

  <div class="card"><table>
    <thead><tr><th>Suite</th><th>Mode</th><th class="num">Tests</th>
      <th class="num">Pass</th><th class="num">Fail</th><th class="num">Skip</th>
      <th class="num">Time</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>

  {capacity_section}

  <h2>Per-suite detail</h2>
  {''.join(detail)}

  <footer>
    {gpu_note}Durations vary run-to-run; see <code>report.json</code> for the machine-readable form.
    Generated locally &mdash; the e2e suite is too heavy for CI.
  </footer>
</div></body></html>"""


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_dir = Path(sys.argv[1])
    specs = []
    artifact_dir = ""
    for arg in sys.argv[2:]:
        # Optional, and kept out of the spec grammar so callers that do not
        # retain e2e artifacts are unaffected.
        if arg.startswith("--artifacts="):
            artifact_dir = arg.split("=", 1)[1]
            continue
        label, mode, path = arg.split("=", 2)
        specs.append((label, mode, path))
    rep = build_report(specs, artifact_dir=artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    (out_dir / "report.html").write_text(render_html(rep))
    print(f"Wrote {out_dir/'report.json'} and {out_dir/'report.html'}")
    print(f"Totals: {rep['totals']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Execute a notebook to completion and keep the report readable no matter how it ends.

    bin/execute_notebook.py <source.ipynb> <executed.ipynb> <kernel-name>

Two things `jupyter execute` does not give us, both of which matter when the run is
hours long and nobody is watching:

1. The executed notebook is written after EVERY cell, not once at the end, so a run
   that dies in the seed cell still leaves a report showing how far it got.
2. A failing cell does not abort the file. `allow_errors` lets the later read-back
   and comparison cells run against whatever did succeed, and the traceback is
   captured in the notebook rather than only in the log.

Needs nbformat and nbclient, which the job's own venv does not carry -- run it with
the runner venv.
"""
import os
import re
import sys
from pathlib import Path


def find_config_source(nb):
    """The notebook's own redaction cell, found by content rather than by index.

    It used to be read as cells[2], which silently picks up the wrong cell the first
    time anyone inserts one above it.
    """
    for cell in nb.cells:
        text = "".join(cell["source"])
        if "REDACT_OUTPUT =" in text and "def show(" in text:
            return text
    raise LookupError("no config cell defining REDACT_OUTPUT and show()")


def load_redactor(nb):
    """Reuse the notebook's own redact(), rather than a second copy of the patterns.

    TRACEBACKS BYPASS THE NOTEBOOK'S REDACTION. show() masks identity on the way out
    of a print, but a cell that RAISES has its traceback written straight into the
    report by nbclient -- filenames, source lines and repr'd arguments included. So
    every output is scrubbed again at save time, with the same function.
    """
    text = find_config_source(nb)
    body = text[text.index("REDACT_OUTPUT ="):text.index("def show(")]
    ns = {"os": os, "re": re}
    exec(compile(body, "cfg", "exec"), ns)
    return ns["redact"]


def scrub(obj, redact):
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, list):
        return [scrub(v, redact) for v in obj]
    if isinstance(obj, dict):
        return {k: (v if k in ("output_type", "name", "ename") else scrub(v, redact))
                for k, v in obj.items()}
    return obj


def save_notebook(nb, dst, redact):
    import nbformat

    for cell in nb.cells:
        for out in cell.get("outputs", []) or []:
            for key in ("text", "traceback", "evalue", "data"):
                if key in out:
                    out[key] = scrub(out[key], redact)
    tmp = dst.with_suffix(".tmp")
    nbformat.write(nb, tmp)
    tmp.replace(dst)          # atomic: never leave a half-written report


def _client_class():
    from nbclient import NotebookClient

    class Saving(NotebookClient):
        # NBCLIENT CAPTURES A CELL'S OUTPUT INTO THE NOTEBOOK AND NOWHERE ELSE.
        # Without the tee below, the run log held only the "[runner] cell N done"
        # lines -- so every watcher that read it searched for submission markers that
        # were never written there, matched nothing, and concluded the run was fine.
        # That is how a run whose seed died after 67 minutes reported "SUCCEEDED".
        def __init__(self, nb, save, redact, **kwargs):
            super().__init__(nb, **kwargs)
            self._save = save
            self._redact = redact

        def output(self, outs, msg, display_id, cell_index):
            out = super().output(outs, msg, display_id, cell_index)
            self._tee(msg)
            return out

        def _tee(self, msg):
            content = msg.get("content") or {}
            if msg.get("msg_type") == "stream":
                text = content.get("text", "")
            elif msg.get("msg_type") == "error":
                # A cell that RAISES never reaches show(), so this text is unmasked.
                text = "\n".join(("", *content.get("traceback", []), ""))
            else:
                return
            if text:
                # show() already masked whatever was printed through it; bare print()s
                # and the tracebacks above were not. redact() maps its own placeholders
                # back to themselves, so applying it to all of it is safe.
                sys.stdout.write(self._redact(text))
                sys.stdout.flush()

        def on_cell_executed(self, cell_index=None, cell=None, execute_reply=None):
            self._save()
            print(f"[runner] cell {cell_index + 1}/{len(self.nb.cells)} done", flush=True)

    return Saving


def main(argv):
    import nbformat

    if len(argv) != 4:
        print(f"usage: {Path(argv[0]).name} <source.ipynb> <executed.ipynb> <kernel>",
              file=sys.stderr)
        return 2

    src, dst, kernel = Path(argv[1]), Path(argv[2]), argv[3]
    nb = nbformat.read(src, as_version=4)
    try:
        redact = load_redactor(nb)
    except LookupError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    def save():
        save_notebook(nb, dst, redact)

    save()   # a report exists from the first second, before any cell has run

    client = _client_class()(
        nb,
        save,
        redact,
        kernel_name=kernel,
        timeout=None,          # the notebook enforces its own per-submission timeouts
        allow_errors=True,     # capture failures in the report instead of aborting it
        resources={"metadata": {"path": str(src.parent)}},
    )

    try:
        client.execute()
    finally:
        save()

    # Exit non-zero if any cell raised, so the caller records an accurate verdict.
    failed = [i for i, c in enumerate(nb.cells)
              if any(o.get("output_type") == "error" for o in c.get("outputs", []) or [])]
    if failed:
        print(f"[runner] cells with errors: {[i + 1 for i in failed]}", flush=True)
        return 1
    print("[runner] all cells clean", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

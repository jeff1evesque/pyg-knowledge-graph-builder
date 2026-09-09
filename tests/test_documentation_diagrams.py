"""An inline SVG in the documentation has to survive the MkDocs build.

Python-Markdown does not treat `svg` as block-level but does treat `style` as
one, so an SVG written straight into a page is split across two paragraphs at
its `<style>` child and a `<br>` is inserted. `<br>` is a breakout tag in SVG
foreign content: the browser closes the `<svg>` at the start tag and every
shape after it is parsed as HTML, outside the drawing. The published page keeps
a correctly sized but empty box. Wrapping the SVG in a `<div>` is what keeps it
one raw block.

Needs MkDocs, which `requirements-test.txt` layers in from
`requirements-docs.txt`. The skip covers a virtualenv that predates that.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mkdocs", reason="needs requirements-docs.txt")

_ROOT = Path(__file__).resolve().parent.parent
_DOCS = _ROOT / "documentation"
_SVG = re.compile(r"<svg\b.*?</svg>", re.S)

_UNWRAPPED = (
    '<svg viewBox="0 0 100 40" xmlns="http://www.w3.org/2000/svg">\n'
    "  <style>\n"
    "    .a { fill: red; }\n"
    "  </style>\n"
    '  <rect class="a" x="0" y="0" width="50" height="20"/>\n'
    "</svg>"
)


def _build(project, into):
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "-d", str(into)],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    return into


def _page(md):
    """Where mkdocs puts a source page, under its default directory URLs."""
    rel = md.relative_to(_DOCS).with_suffix("")
    return rel.parent / "index.html" if rel.name == "index" else rel / "index.html"


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return _build(_ROOT, tmp_path_factory.mktemp("site"))


def test_every_documentation_svg_survives_the_build(site):
    checked = 0
    for md in sorted(_DOCS.rglob("*.md")):
        svgs = _SVG.findall(md.read_text())
        if not svgs:
            continue
        built = (site / _page(md)).read_text()
        for svg in svgs:
            assert svg in built, (
                f"{md.relative_to(_ROOT)}: Markdown rewrote this SVG instead of "
                f"emitting it verbatim -- {svg.splitlines()[0][:70]}"
            )
            checked += 1
    assert checked, "no inline SVG found to check -- has the guard lost its subject?"


def test_the_guard_catches_an_unwrapped_svg(tmp_path):
    """Without the `<div>` the same SVG is torn apart; with it, it is not."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "mkdocs.yml").write_text("site_name: guard\n")
    page = docs / "index.md"
    out = tmp_path / "site"

    page.write_text(f"# t\n\n{_UNWRAPPED}\n")
    built = (_build(tmp_path, out) / "index.html").read_text()
    assert _UNWRAPPED not in built
    assert "<br" in built

    page.write_text(f"# t\n\n<div>\n{_UNWRAPPED}\n</div>\n")
    built = (_build(tmp_path, out) / "index.html").read_text()
    assert _UNWRAPPED in built

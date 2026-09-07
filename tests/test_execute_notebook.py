"""Pin how `bin/execute_notebook.py` finds the notebook's redaction and applies it.

WHY THIS IS WORTH A TEST
------------------------
The notebook masks identity on the way out of show(). A cell that RAISES never reaches
show(): nbclient writes the traceback straight into the report, with file paths, source
lines and repr'd arguments in it. The runner's job is to scrub every output again at
save time using the notebook's OWN redact(), so there is one set of patterns rather
than two that drift.

That makes the lookup of the notebook's config cell load-bearing. It used to be read as
cells[2] -- a fixed index that silently picks up the wrong cell the first time anyone
inserts one above it, and whose failure mode is a report published with nothing masked.
Finding the cell by content, and failing loudly when it is absent, is what these pin.

No notebook is executed here: the functions under test are pure, and the module keeps
its nbformat and nbclient imports inside the functions that need them so it can be
imported without either.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "bin" / "execute_notebook.py"

CONFIG_CELL = '''
REDACT_OUTPUT = True

_PATTERNS = [("SECRET-HOST", "<host>")]


def redact(text):
    for needle, mask in _PATTERNS:
        text = text.replace(needle, mask)
    return text


def show(text):
    print(redact(text))
'''


def _load_tool():
    """Import by path -- bin/ is scripts, not an importable package."""
    spec = importlib.util.spec_from_file_location("execute_notebook_tool", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def _nb(*sources):
    return SimpleNamespace(cells=[{"source": s} for s in sources])


# --------------------------------------------------------------------------- #
# Finding the notebook's own redaction
# --------------------------------------------------------------------------- #

def test_config_cell_is_found_by_content_not_by_position(tool):
    """Inserting a cell above the config used to silently redirect the lookup."""
    nb = _nb("# title", "import os", "x = 1", "y = 2", CONFIG_CELL)
    assert tool.find_config_source(nb) == nb.cells[4]["source"]


def test_a_cell_holding_only_one_of_the_two_markers_is_not_the_config(tool):
    """A cell that mentions REDACT_OUTPUT in passing is not the definition."""
    nb = _nb("# REDACT_OUTPUT = see below", "def show(x): pass", CONFIG_CELL)
    assert tool.find_config_source(nb) == nb.cells[2]["source"]


def test_a_notebook_without_a_config_cell_fails_loudly(tool):
    """The alternative is exec'ing whatever happened to be there, then publishing a
    report with nothing masked."""
    with pytest.raises(LookupError):
        tool.find_config_source(_nb("import os", "x = 1"))


def test_the_redactor_is_the_notebook_s_own(tool):
    """Loaded out of the notebook rather than reimplemented, so the two cannot drift."""
    redact = tool.load_redactor(_nb("x = 1", CONFIG_CELL))
    assert redact("ran on SECRET-HOST today") == "ran on <host> today"


def test_loading_the_redactor_does_not_run_the_rest_of_the_cell(tool):
    """Only the patterns are needed. The cell also defines show(), which prints."""
    cell = CONFIG_CELL + "\nraise RuntimeError('the tail of the cell ran')\n"
    assert tool.load_redactor(_nb(cell))("SECRET-HOST") == "<host>"


# --------------------------------------------------------------------------- #
# Scrubbing what nbclient captured
# --------------------------------------------------------------------------- #

def test_scrub_reaches_into_tracebacks_and_nested_output(tool):
    """A traceback is a list of strings, and data is a dict of them. Both are written
    into the report by nbclient without ever passing through show()."""
    redact = tool.load_redactor(_nb(CONFIG_CELL))
    out = {
        "output_type": "error",
        "ename": "ValueError",
        "evalue": "bad path on SECRET-HOST",
        "traceback": ["Traceback:", '  File "/tmp/x.py" on SECRET-HOST'],
        "data": {"text/plain": "SECRET-HOST"},
    }
    scrubbed = tool.scrub(out, redact)

    assert "SECRET-HOST" not in repr(scrubbed)
    assert scrubbed["traceback"][1].endswith("on <host>")
    assert scrubbed["data"]["text/plain"] == "<host>"


def test_scrub_leaves_the_structural_keys_alone(tool):
    """output_type, name and ename are read by nbformat, not shown to a reader."""
    redact = tool.load_redactor(_nb(CONFIG_CELL))
    scrubbed = tool.scrub(
        {"output_type": "stream", "name": "stdout", "text": "SECRET-HOST"}, redact)
    assert scrubbed["output_type"] == "stream"
    assert scrubbed["name"] == "stdout"
    assert scrubbed["text"] == "<host>"

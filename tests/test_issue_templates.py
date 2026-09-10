"""A broken issue template does not fail anything -- it stops appearing.

GitHub reads the YAML front matter of every file in `.github/ISSUE_TEMPLATE/` to
build the chooser behind "New issue". Front matter that does not parse, or that
omits `name`, drops that template out of the list silently: nothing errors,
nothing 404s, and the only symptom is a chooser one entry short. Templates load
exclusively from the DEFAULT branch, so the first chance to catch it by hand is
after the merge that shipped it -- which is why it is caught here instead.

Needs PyYAML, which `requirements-test.txt` layers in from
`requirements-docs.txt`. The skip covers a virtualenv that predates that.
"""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="needs requirements-docs.txt")

_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATES = _ROOT / ".github" / "ISSUE_TEMPLATE"

# `name` is what GitHub labels the chooser entry with; `about` is the line under
# it. The rest are optional to GitHub and conventional here, so they are checked
# for presence, not for content -- `labels: ''` is the deliberate value while the
# repository's label set is undecided.
_CHOOSER_KEYS = ("name", "about")
_CONVENTION_KEYS = ("title", "labels", "assignees")


def _front_matter(text):
    """Parse the front matter, or say why GitHub would skip the file."""
    lines = text.split("\n")
    if lines[0] != "---":
        raise ValueError("does not open with a --- front matter fence")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise ValueError("front matter fence is never closed") from None

    meta = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(meta, dict):
        raise ValueError(f"front matter is {type(meta).__name__}, not a mapping")
    return meta, "\n".join(lines[end + 1:])


def test_every_template_is_one_github_will_offer():
    names = []
    for path in sorted(_TEMPLATES.glob("*.md")):
        where = path.relative_to(_ROOT)
        try:
            meta, body = _front_matter(path.read_text())
        except ValueError as exc:
            pytest.fail(f"{where}: {exc}")

        for key in _CHOOSER_KEYS:
            assert meta.get(key), f"{where}: {key} is empty, so the chooser cannot label it"
        missing = [key for key in _CONVENTION_KEYS if key not in meta]
        assert not missing, f"{where}: front matter omits {missing}"
        assert body.strip(), f"{where}: nothing below the front matter to prefill with"
        names.append(meta["name"])

    assert names, "no issue templates found -- has the guard lost its subject?"
    assert len(set(names)) == len(names), f"two templates share a chooser name: {names}"


def test_config_keeps_blank_issues_available():
    """Issues drafted in full elsewhere are pasted whole; the chooser must not
    be the only way in."""
    config = yaml.safe_load((_TEMPLATES / "config.yml").read_text())
    assert config["blank_issues_enabled"] is True


@pytest.mark.parametrize(
    "text",
    [
        "name: Change\nabout: no fence at all\n",
        "---\nname: Change\nabout: fence never closed\n",
        "---\njust a string, not a mapping\n---\n\n## Problem\n",
    ],
    ids=["no-fence", "unclosed-fence", "not-a-mapping"],
)
def test_the_guard_catches_a_template_github_would_skip(text):
    """Each of these renders as a plain file GitHub declines to offer."""
    with pytest.raises(ValueError):
        _front_matter(text)

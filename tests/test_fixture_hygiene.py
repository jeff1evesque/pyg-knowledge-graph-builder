"""Nothing in a committed fixture may describe the deployment that produced it.

The fixtures are generated from private buckets and committed to a public
repository. The data itself is fine -- public filings, public quotes, public
statistics -- but the objects they were sampled from are not, and neither is
anything naming the accounts, hosts or third parties involved.

WHY A STRUCTURAL RULE RATHER THAN A LIST OF BANNED WORDS
--------------------------------------------------------
A test that greps committed files for a sensitive string has to contain that
string, which publishes it in the course of protecting it. So the rules here
are shaped the other way round: an ALLOWLIST of what may appear, and generic
patterns for deployment detail that are safe to write down because they are
shapes rather than secrets.

THE CASE THAT PROMPTED IT
-------------------------
A Turtle prefix label. `@prefix mq: <.../ontology/market-quotes/>` and
`@prefix XXXX: <.../ontology/market-quotes/>` produce identical triples -- the
label is cosmetic, expands to the same namespace, and nothing downstream reads
it -- but it travels with the document from upstream, where it is named after
the data vendor. It survived a regeneration on 2026-09-12 and would have been
committed, while the namespace it binds says only `market-quotes`.

The rule that catches it: a prefix label may not say more than the namespace it
binds. Enforced as an allowlist, so a new vendor-named label fails without this
file ever naming one.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"

# Prefix labels a committed fixture may use.
#
#   ns1..ns9   what rdflib invents when the source bound no label. Most of the
#              fixtures serialize this way, and it says nothing at all.
#   the rest   standard vocabularies, plus short neutral names for this
#              project's own namespaces.
#
# Add to this list only a label that could be derived from the namespace it
# binds. If a regeneration introduces one that could not, that is the point of
# the guard -- fix the generator, do not widen this.
ALLOWED_PREFIX_LABELS = {
    "alert", "cap", "geosparql", "mq", "owl", "rdf", "rdfs", "sec",
    "sec-common", "sec-filings", "unified", "weather", "xsd",
} | {f"ns{n}" for n in range(1, 10)}

# Deployment detail. Shapes, not secrets, so naming them here publishes nothing.
FORBIDDEN_PATTERNS = {
    "an S3 URI": re.compile(rb"s3a?://", re.I),
    "an AWS access key id": re.compile(rb"AKIA[0-9A-Z]{12,}"),
    "an IAM ARN": re.compile(rb"arn:aws:"),
    "a POSIX home directory": re.compile(rb"/home/[a-z0-9_.-]+/", re.I),
    "a private IPv4 address": re.compile(
        rb"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"
    ),
    "an AWS account id": re.compile(rb"\b\d{12}\b"),
}

PREFIX_DECLARATION = re.compile(rb"@prefix\s+([^\s:]*):")


def _fixture_files():
    return sorted(p for p in FIXTURES.rglob("*") if p.is_file())


def test_there_are_fixtures_to_check():
    """A guard over an empty set passes and proves nothing."""
    assert _fixture_files(), f"no fixtures found under {FIXTURES}"


@pytest.mark.parametrize(
    "fixture", _fixture_files(), ids=lambda p: p.relative_to(FIXTURES).as_posix()
)
def test_no_fixture_carries_deployment_detail(fixture):
    """Bucket names, hosts, account ids and paths describe where the data came
    from, not what it says. Checked as BYTES so a Parquet fixture is covered --
    the interesting strings live inside the compressed column, where a text
    search over the repository never looks."""
    body = fixture.read_bytes()
    found = sorted(
        name for name, pattern in FORBIDDEN_PATTERNS.items()
        if pattern.search(body)
    )
    assert not found, (
        f"{fixture.relative_to(REPO_ROOT)} contains {', '.join(found)}. "
        f"Fixtures are committed to a public repository; regenerate without it."
    )


@pytest.mark.parametrize(
    "fixture", _fixture_files(), ids=lambda p: p.relative_to(FIXTURES).as_posix()
)
def test_prefix_labels_say_no_more_than_their_namespace(fixture):
    """A Turtle prefix label is cosmetic and expands to the namespace it binds,
    so it is the one part of a fixture that can carry a name the data does not.
    Upstream's market Turtle binds one named after the data vendor."""
    labels = {
        match.group(1).decode("utf-8", "replace").lower()
        for match in PREFIX_DECLARATION.finditer(fixture.read_bytes())
    }
    unknown = sorted(labels - ALLOWED_PREFIX_LABELS)
    assert not unknown, (
        f"{fixture.relative_to(REPO_ROOT)} binds prefix label(s) {unknown} that "
        f"are not known-neutral. A label must not say more than the namespace "
        f"it binds — fix the generator's PREFERRED_PREFIXES rather than "
        f"widening ALLOWED_PREFIX_LABELS."
    )

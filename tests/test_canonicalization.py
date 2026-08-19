"""
Tests for source-shape canonicalization (spark_jobs/utils/canonicalization.py).

The defect these pin is an identity split, not a dead join: upstream states one
filer's CIK two ways in the SAME filing -- ``Issuer_0001729997`` with
``hasIssuerCik "0001729997"`` from its EDGAR-metadata path, and
``Issuer_1729997`` with ``hasIssuerCik 1729997.0`` from its document path -- so
the filer becomes two nodes and the join that would unify them sees two
different CIKs. Nothing errors; the company's edges simply partition.

The fixtures in tests/fixtures/e2e carry only the padded spelling (every one of
their 110 issuer references is 10 digits), so the unpadded half is written out
here from the measured production shape rather than sampled.
"""
from decimal import Decimal

from rdflib import Graph

from spark_jobs.utils.canonicalization import (
    CIK_DIGITS,
    canonicalize_sec_identifiers,
    canonicalize_source_triples,
)
from spark_jobs.utils.rdf_utils import SEC_FILINGS, identifier_namespace

_ID = identifier_namespace(str(SEC_FILINGS))
_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_HAS_ISSUER = str(SEC_FILINGS.hasIssuer)
_HAS_ISSUER_CIK = str(SEC_FILINGS.hasIssuerCik)
_HAS_ISSUER_NAME = str(SEC_FILINGS.hasIssuerName)
_HAS_OWNER_CIK = str(SEC_FILINGS.hasReportingOwnerCik)
_ISSUER_TYPE = str(SEC_FILINGS.Issuer)

_XSD_DECIMAL = "http://www.w3.org/2001/XMLSchema#decimal"

_FILING = f"{_ID}0001977454-26-000239_Filing"
_PADDED = f"{_ID}Issuer_0001729997"
_UNPADDED = f"{_ID}Issuer_1729997"

# One real Form 144 row, reduced to the statements that carry the split. Both
# hasIssuer objects, both Issuer subjects and both CIK spellings are verbatim
# from s3://<archive>/raw/source=sec/feed=filings/year=2026/month=08/06.
_SPLIT_ROWS = [
    (_FILING, _TYPE, str(SEC_FILINGS.SECFiling)),
    (_FILING, _HAS_ISSUER, _PADDED),
    (_FILING, _HAS_ISSUER, _UNPADDED),
    (_PADDED, _TYPE, _ISSUER_TYPE),
    (_PADDED, _HAS_ISSUER_CIK, "0001729997"),
    (_PADDED, _HAS_ISSUER_NAME, "Grayscale CoinDesk Crypto 5 ETF  (GDLC)"),
    (_UNPADDED, _TYPE, _ISSUER_TYPE),
    # The document path ships the CIK unquoted, so rdflib types it xsd:decimal
    # and the loader stores str(Decimal("1729997.0")) -- see the round-trip
    # test below, which pins that this is really the string the frame carries.
    (_UNPADDED, _HAS_ISSUER_CIK, "1729997.0"),
    (_UNPADDED, _HAS_ISSUER_NAME, "Grayscale CoinDesk Crypto 5 ETF"),
]


def _triples(result):
    return {(r["subject"], r["predicate"], r["object"]) for r in result.collect()}


# ======================================================================
# The split, and its repair
# ======================================================================

def test_unpadded_and_padded_issuers_resolve_to_one_node(spark, make_triples):
    """The acceptance criterion: one filer, one issuer URI, one CIK."""
    result = _triples(canonicalize_sec_identifiers(make_triples(_SPLIT_ROWS)))

    issuers = {s for s, p, _o in result if p == _HAS_ISSUER_CIK}
    assert issuers == {_PADDED}, (
        f"the two spellings did not merge: {sorted(issuers)}"
    )

    ciks = {o for _s, p, o in result if p == _HAS_ISSUER_CIK}
    assert ciks == {"0001729997"}, f"CIK literals did not merge: {sorted(ciks)}"

    # The filing stated hasIssuer twice, once at each spelling. Both now name
    # the same node, so the company no longer partitions its filings.
    targets = {o for _s, p, o in result if p == _HAS_ISSUER}
    assert targets == {_PADDED}


def test_canonicalization_adds_and_drops_no_triples(spark, make_triples):
    """Value-preserving: the row count is untouched.

    Dedup is the enrichment pipeline's job and it already runs one; collapsing
    rows here would make this step's output depend on what else was in the
    frame.
    """
    source = make_triples(_SPLIT_ROWS)
    assert canonicalize_sec_identifiers(source).count() == source.count()


def test_both_names_survive_the_merge(spark, make_triples):
    """Merging the identity must not merge the facts.

    The two paths spell the filer's name differently. Both are statements the
    source made about one company, and both stay -- exactly as they already do
    for an issuer upstream happens to pad on both paths.
    """
    result = _triples(canonicalize_sec_identifiers(make_triples(_SPLIT_ROWS)))
    names = {o for s, p, o in result if p == _HAS_ISSUER_NAME and s == _PADDED}
    assert len(names) == 2, names


# ======================================================================
# What it must leave alone
# ======================================================================

def test_positional_issuer_fallback_is_left_alone(spark, make_triples):
    """``0001-26-1_Issuer_0`` keys on position, not on a CIK.

    Upstream mints it for a document that states no issuer CIK -- a 13F names a
    security, not an issuer -- so its trailing digit is an ordinal. Padding it
    to Issuer_0000000000 would invent a filer and merge every such document
    onto it.
    """
    positional = f"{_ID}0001-26-1_Issuer_0"
    rows = [(positional, _TYPE, _ISSUER_TYPE)]
    assert _triples(canonicalize_sec_identifiers(make_triples(rows))) == set(rows)


def test_already_canonical_input_is_unchanged(spark, make_triples):
    """Idempotence, which is what makes it safe to run on every load."""
    rows = [
        (_PADDED, _HAS_ISSUER_CIK, "0001729997"),
        (_FILING, _HAS_ISSUER, _PADDED),
    ]
    once = canonicalize_sec_identifiers(make_triples(rows))
    assert _triples(once) == set(rows)
    assert _triples(canonicalize_sec_identifiers(once)) == set(rows)


def test_overlong_identifier_is_not_truncated(spark, make_triples):
    """Spark's lpad truncates from the RIGHT when the input is too long.

    ``lpad("00017299970", 10, "0")`` is "0001729997" -- a different filer. The
    guard is an explicit length test, not lpad's own behaviour, and this is
    what pins it: an 11-digit local name must come back untouched rather than
    silently renamed.
    """
    overlong = f"{_ID}Issuer_00017299970"
    rows = [
        (overlong, _TYPE, _ISSUER_TYPE),
        (overlong, _HAS_ISSUER_CIK, "00017299970"),
    ]
    assert _triples(canonicalize_sec_identifiers(make_triples(rows))) == set(rows)


def test_non_cik_values_and_uris_are_untouched(spark, make_triples):
    """Only CIK-keyed locals and CIK predicates are in scope."""
    date_node = f"{_ID}Date_2026-08-05"
    rows = [
        # A date intermediate node: digits in the local name, not a CIK.
        (_FILING, str(SEC_FILINGS.hasFilingDate), date_node),
        (date_node, _TYPE, str(SEC_FILINGS.Date)),
        # A numeric literal under a predicate that is not a CIK.
        (_FILING, str(SEC_FILINGS.hasSequence), "1"),
        # A CIK predicate whose object is not numeric at all.
        (_PADDED, _HAS_ISSUER_CIK, "unknown"),
    ]
    assert _triples(canonicalize_sec_identifiers(make_triples(rows))) == set(rows)


def test_reporting_owner_ciks_are_canonicalized_too(spark, make_triples):
    """Same minting path, same hazard -- confirmed upstream, not assumed.

    The document walker keys TWO entity classes on a CIK read out of the
    filing: ``issuerCik`` under ``<issuer>``/``<issuerInfo>`` mints ``Issuer``,
    and ``rptOwnerCik`` under ``<reportingOwner>`` mints ``ReportingOwner``.
    Both go through the same leading-zero heuristic. Every reporting-owner CIK
    in the sampled data happens to be padded today, so this covers the half
    that has not yet been hit rather than a half that cannot be.
    """
    unpadded = f"{_ID}ReportingOwner_1729997"
    padded = f"{_ID}ReportingOwner_0001729997"
    rows = [
        (unpadded, _HAS_OWNER_CIK, "1729997"),
        (_FILING, str(SEC_FILINGS.hasReportingOwner), unpadded),
    ]
    result = _triples(canonicalize_sec_identifiers(make_triples(rows)))
    assert result == {
        (padded, _HAS_OWNER_CIK, "0001729997"),
        (_FILING, str(SEC_FILINGS.hasReportingOwner), padded),
    }


# ======================================================================
# The shape the loader actually hands over
# ======================================================================

def test_decimal_cik_reaches_the_frame_as_a_dot_zero_string():
    """Pins the assumption the ``.0`` branch is written against.

    The ``.0`` is rdflib, not a float anywhere: upstream types the digit string
    "1729997" as xsd:decimal, and Turtle cannot write that as a bare 1729997
    because it would re-parse as xsd:integer. Our loader then stores
    ``str(literal.toPython())``, and Decimal keeps the fractional part -- so
    what the frame carries is "1729997.0". If rdflib ever normalised that away
    this test fails, rather than the padding rule quietly stopping at a shape
    that no longer occurs.
    """
    graph = Graph()
    graph.parse(
        data=f"""
        @prefix f: <{SEC_FILINGS}> .
        <{_UNPADDED}> f:hasIssuerCik 1729997.0 .
        """,
        format="turtle",
    )
    (_s, _p, obj), = graph
    assert str(obj.datatype) == _XSD_DECIMAL
    assert isinstance(obj.toPython(), Decimal)
    assert str(obj.toPython()) == "1729997.0"


# ======================================================================
# The rest of the class: every all-digit identifier without a leading zero
# ======================================================================

def test_cusip_is_padded_to_nine(spark, make_triples):
    """The largest instance of the class, not the CIK.

    A CUSIP is 9 characters and only about one in ten starts with a zero, so
    the leading-zero heuristic upstream types most all-digit CUSIPs as decimal.
    Both spellings below are verbatim from the committed e2e fixtures.
    """
    subject = f"{_ID}Holding_1"
    rows = [
        (subject, str(SEC_FILINGS.hasCusip), "824348106.0"),
        (subject, str(SEC_FILINGS.hasCusip), "002824100"),
        # Alphanumeric CUSIPs never looked numeric, so they were never mis-typed.
        (subject, str(SEC_FILINGS.hasCusip), "G1151C101"),
    ]
    assert _triples(canonicalize_sec_identifiers(make_triples(rows))) == {
        (subject, str(SEC_FILINGS.hasCusip), "824348106"),
        (subject, str(SEC_FILINGS.hasCusip), "002824100"),
        (subject, str(SEC_FILINGS.hasCusip), "G1151C101"),
    }


def test_zip_code_recovers_its_leading_zero(spark, make_triples):
    """Padding here repairs real loss rather than tidying a spelling.

    ``6375.0`` is ZIP 06375. The zero is gone from the source -- the agent
    wrote it without one, so upstream's identifier test did not fire -- and
    zero-padding to 5 is what puts it back. Contrast ``"06510"``, which kept
    its zero and was therefore already a string.
    """
    owner = f"{_ID}ReportingOwner_0001729997"
    rows = [
        (owner, str(SEC_FILINGS.hasRptOwnerZipCode), "6375.0"),
        (owner, str(SEC_FILINGS.hasRptOwnerZipCode), "06510"),
        (owner, str(SEC_FILINGS.hasRptOwnerZipCode), "60607.0"),
    ]
    assert _triples(canonicalize_sec_identifiers(make_triples(rows))) == {
        (owner, str(SEC_FILINGS.hasRptOwnerZipCode), "06375"),
        (owner, str(SEC_FILINGS.hasRptOwnerZipCode), "06510"),
        (owner, str(SEC_FILINGS.hasRptOwnerZipCode), "60607"),
    }


def test_form_codes_lose_the_decimal_but_are_never_padded(spark, make_triples):
    """A form type is "4", never "0000000004".

    This is the half of the repair that must NOT pad. It also shows the damage
    reaching something that is not a join: the metadata path states
    ``"4"^^xsd:string`` and the document path ``4.0`` for the same filing, so
    without this one form type is two category labels in the feature encoder.
    """
    rows = [
        (_FILING, str(SEC_FILINGS.hasDocumentType), "4.0"),
        (_FILING, str(SEC_FILINGS.hasDocumentType), "4"),
        (_FILING, str(SEC_FILINGS.hasDocumentType), "10-K"),
        (_FILING, str(SEC_FILINGS.hasTransactionFormType), "5.0"),
    ]
    assert _triples(canonicalize_sec_identifiers(make_triples(rows))) == {
        (_FILING, str(SEC_FILINGS.hasDocumentType), "4"),
        (_FILING, str(SEC_FILINGS.hasDocumentType), "10-K"),
        (_FILING, str(SEC_FILINGS.hasTransactionFormType), "5"),
    }


def test_quantities_and_flags_are_left_alone(spark, make_triples):
    """Only identifiers and codes are in scope.

    The share counts are genuine magnitudes and their decimals mean something.
    The ownership flags arrive as 0.0/1.0 and are mis-typed too, but a flag has
    no identity to destroy -- rewriting them would change feature values
    without fixing a join or a node, so they are left for a separate decision
    about booleans.
    """
    subject = f"{_ID}Transaction_1"
    rows = [
        (subject, str(SEC_FILINGS.hasTransactionShares), "1439.0"),
        (subject, str(SEC_FILINGS.hasTransactionPricePerShare), "307.75"),
        (subject, str(SEC_FILINGS.isDirector), "1.0"),
        (subject, str(SEC_FILINGS.hasSequence), "1"),
    ]
    assert _triples(canonicalize_sec_identifiers(make_triples(rows))) == set(rows)


def test_dotted_uri_fragment_cannot_mint_a_third_node(spark, make_triples):
    """A guard, not an observation.

    Upstream builds the URI fragment from the raw text before any datatype is
    decided, so it cannot carry a ".0" today. But its URI encoder passes "."
    through untouched, so a value that ever did arrive that way would mint
    ``Issuer_1729997.0`` alongside the two spellings already known. Nothing
    upstream prevents it.
    """
    rows = [(f"{_ID}Issuer_1729997.0", _TYPE, _ISSUER_TYPE)]
    assert _triples(canonicalize_sec_identifiers(make_triples(rows))) == {
        (_PADDED, _TYPE, _ISSUER_TYPE)
    }


def test_entry_point_applies_the_sec_rule(spark, make_triples):
    """canonicalize_source_triples is what the loader calls."""
    result = _triples(canonicalize_source_triples(make_triples(_SPLIT_ROWS)))
    assert _UNPADDED not in {s for s, _p, _o in result}


def test_padding_width_is_the_canonical_cik_width():
    """Ten digits, zero-padded -- the form the SEC states everywhere."""
    assert CIK_DIGITS == 10

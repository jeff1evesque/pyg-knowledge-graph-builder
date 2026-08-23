"""
Source-shape canonicalization — the same fact, spelled one way.

WHAT THIS IS NOT
----------------
Not enrichment. Every function here is value-preserving: it asserts no new
fact, mints no new relationship, and adds no row. It rewrites an identifier
that upstream spelled two ways into the one spelling both halves of a join can
see. Enrichment lives one layer up and runs after this.

It sits in the LOADER rather than in a pipeline phase because the split it
repairs is an identity split: the two spellings are different URIs, so they are
different nodes, and every consumer downstream -- enrichment, the enriched
Parquet, node_mapper -- has to agree on which one is the entity. Repairing it
once, before anything reads the frame, is the only placement where they cannot
disagree.

WHY IT EXISTS
-------------
A join keyed on a term the data no longer emits matches nothing and is
detectable by diffing the vocabulary (see bin/check_vocabulary_drift.py). This
is the adjacent failure: the term is live, the join is correct, and the DATA is
shaped two ways, so the join matches some of what it should and silently
under-covers.

Measured instance, from 23,983 filing rows across four dates:

    <.../id/sec-filings/Issuer_0001729997>  ns1:hasIssuerCik "0001729997"
    <.../id/sec-filings/Issuer_1729997>     ns1:hasIssuerCik 1729997.0

Both are stated by the SAME Form 144 filing, which carries two ``hasIssuer``
objects as a result. 84 issuer references in 12 distinct filers arrived
unpadded; every one of the 12 also appears padded in the same sample, so all 12
are live splits rather than one-sided oddities.

Nothing errors. The filer becomes two ``filings_Issuer`` nodes whose filings,
edges and message passing partition between them, and
``_unify_company_entities`` cannot join them either, because it groups on the
CIK literal and sees "0001729997" and "1729997.0" as two different companies.

WHERE THE TWO SPELLINGS COME FROM
---------------------------------
Confirmed upstream rather than guessed, because the guess was wrong. There is
no float and no null: the ``.0`` is rdflib round-tripping an ``xsd:decimal``,
which Turtle cannot write as a bare ``1729997`` without it re-parsing as
``xsd:integer``.

Upstream mints the issuer on two paths that meet in one merged document:

  * the EDGAR search metadata, declaratively mapped. Its CIK is a JSON string
    already padded to 10, and every literal it emits defaults to xsd:string.
    This path is never affected.
  * the filing's ``primary_doc.xml``, walked structurally. Its CIK is whatever
    text the FILING AGENT put in ``<issuerCik>``, and agents disagree -- for
    one issuer, three filings from agent 0001977454 carry ``1729997`` while a
    filing from agent 0001976415 carries ``0001729997``.

The document path then decides a literal's datatype with a leading-zero
heuristic: a digit string counts as an identifier only if it starts with "0",
so an unpadded CIK falls through to the numeric branch and is typed decimal.
The URI fragment is built from the same text BEFORE that typing, which is why
the node keys on ``1729997`` while the literal reads ``1729997.0``.

Form 144 is where this concentrates, and not because of anything about Form
144's schema: it is the highest-volume form whose only attachment IS
``primary_doc.xml``, so it is the form on which the agent-formatted file is the
one that gets walked. Richer forms have a real data document (``form4.xml``, a
13F info table) and skip ``primary_doc.xml`` entirely.

Upstream has no fix planned, and a fix there would still leave every row
already written wrong -- so this repair is worth having either way.

THE REST OF THE CLASS
---------------------
The same heuristic mis-types every all-digit identifier that does not happen to
begin with a zero, so the CIK is not the largest instance -- CUSIPs are, being
9 characters with only about one in ten starting with "0". Present in the
committed fixtures already:

    hasCusip             824348106.0     an identifier, not a quantity
    hasDocumentType      4.0             a form code, not a number
    hasTransactionFormType 5.0           the same
    hasRptOwnerZipCode   6375.0          ZIP 06375, its leading zero gone

The ZIP is the one that shows padding does real repair rather than tidying:
``06375`` reaches us as ``6375.0`` and only zero-padding to 5 recovers it.

``hasDocumentType`` shows the damage need not involve a join at all. The
metadata path states ``"4"^^xsd:string`` and the document path ``4.0`` for the
same filing, so ONE form type becomes TWO category labels in the feature
encoder -- see the numeric/categorical split note in
pyg_builder/feature_extractor.py, which already had to defend against bare-digit
form types being z-scored as magnitudes.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Column

from spark_jobs.utils.rdf_utils import SEC_FILINGS, identifier_namespace

import logging
import re

logger = logging.getLogger(__name__)

_SEC_FILINGS_ID = identifier_namespace(str(SEC_FILINGS))

# Identifier locals keyed by a CIK: "<Word>_<cik>" under the sec-filings id
# namespace. Issuer is the shape measured splitting; the other two are the same
# minting path with the same hazard, and padding an already-padded CIK is a
# no-op, so covering them costs nothing and closes the case where the next
# document parser drops a leading zero on an owner instead of an issuer.
#
# The POSITIONAL fallback shape -- ".../0001-26-1_Issuer_0", which upstream
# mints for a document stating no CIK -- is deliberately NOT matched. Its
# trailing digit is an ordinal, not a CIK, and padding it would invent a filer.
# The anchors below are what keep them apart: the local name must BEGIN with
# the word, so a local name beginning with an accession number cannot match.
CIK_KEYED_LOCALS = ("Issuer", "ReportingOwner", "Owner")

# Fixed-width identifiers, and the width each is padded back to. Every one is
# an identifier whose leading zeros are part of it, so a lost zero is a
# different entity rather than a cosmetic difference.
#
#   CIK    10  the SEC's own form everywhere it controls the spelling. Also
#              imported by market/patterns.py, which pads the constituents
#              CSV's unpadded CIK to this width so the market side of the
#              company bridge keys on the same spelling the filings state.
#   CUSIP   9  fixed-length by definition; the largest instance of the class,
#              since only ~1 in 10 begins with a zero and the rest are typed
#              numeric upstream
#   ZIP     5  measured repairing real loss: 06375 arrives as 6375.0
CIK_DIGITS = 10
CUSIP_DIGITS = 9
ZIP_DIGITS = 5

PADDED_IDENTIFIER_WIDTHS = {
    str(SEC_FILINGS.hasIssuerCik): CIK_DIGITS,
    str(SEC_FILINGS.hasReportingOwnerCik): CIK_DIGITS,
    str(SEC_FILINGS.hasCusip): CUSIP_DIGITS,
    str(SEC_FILINGS.hasRptOwnerZipCode): ZIP_DIGITS,
}

# Kept for callers and tests that ask specifically about the CIK half.
CIK_PREDICATES = tuple(
    predicate
    for predicate, width in PADDED_IDENTIFIER_WIDTHS.items()
    if width == CIK_DIGITS
)

# Codes: opaque labels that merely look numeric. The ".0" comes off and NOTHING
# is padded -- a form type is "4", never "0000000004". Padding these would be
# the same class of damage in the other direction, inventing a width the code
# does not have.
OPAQUE_CODE_PREDICATES = (
    str(SEC_FILINGS.hasDocumentType),
    str(SEC_FILINGS.hasTransactionFormType),
    str(SEC_FILINGS.hasOtherManager),
)

# Rates for the five terms this module keys on, from the same 23,983-row census
# as above, so a future reader can tell a rare term from a dead one:
#
#   hasDocumentType         100.00%   every filing
#   hasRptOwnerZipCode       24.14%   98% of Form 4
#   hasTransactionFormType   21.23%   97% of Form 4
#   hasCusip                  6.85%   86% of 13F-HR, 98% of NPORT-P
#   hasOtherManager           1.22%   19% of 13F-HR
#
# hasOtherManager used to be the reason conf/vocabulary_baseline.json carried a
# SEC_FILINGS entry: the e2e fixtures held 13F-HR rows but none that happened to
# carry it, so referencing it here registered as drift against the fixtures
# while being demonstrably live in production. Re-sampling the fixture from a
# day upstream had backfilled (see generate_sec_e2e_fixtures.py, WHICH DAY TO
# SAMPLE) picked up a 13F-HR that states it, and the entry left the baseline.
# The lesson is about the baseline rather than about this term -- an entry there
# is a claim about the FIXTURES, and a better sample can retire one.

# Deliberately NOT repaired: the ownership flags (isDirector, isOfficer,
# isTenPercentOwner, isOther, hasEquitySwapInvolved) also arrive as 0.0 / 1.0.
# They are mis-typed too -- xsd:boolean would be honest -- but a flag has no
# identity to destroy, so rewriting them changes feature values without fixing
# a join or a node. Left for a decision about booleans rather than folded into
# an identifier repair.

# A numeric identifier as it may arrive: bare digits, optionally with the
# trailing ".0" rdflib writes for an xsd:decimal. Anchored at both ends so a
# value that merely CONTAINS digits is left alone.
_NUMERIC_IDENTIFIER = r"^(\d+)(?:\.0+)?$"


def _padded(digits: Column, width: int) -> Column:
    """Zero-pad a captured digit string to a fixed width.

    ``lpad`` is only ever reached under an explicit ``length <= width`` guard,
    because Spark's lpad TRUNCATES -- from the right -- when the input is
    longer than the target width. ``lpad("00012345678", 10, "0")`` is
    "0001234567", a different filer. The same trap cost a wrong-state bug in
    cross_source_linker's FIPS matching; see the note there.
    """
    return F.lpad(digits, width, "0")


def _canonical_cik_uri(column: Column) -> Column:
    """Rewrite ``.../id/sec-filings/<Word>_<cik>`` to its padded form."""
    result = column
    for local in CIK_KEYED_LOCALS:
        prefix = f"{_SEC_FILINGS_ID}{local}_"
        # Anchored at both ends against the ORIGINAL column, not against the
        # partly-rewritten one: each local is a distinct prefix, so at most one
        # branch can fire, and reading the original keeps the three tests
        # independent of the order they are applied in.
        #
        # The optional ".0" is a guard rather than an observation: upstream
        # builds this fragment from the raw text before any datatype is
        # decided, so today it cannot contain one. But its URI-encoder passes
        # "." through untouched, so a value that ever did arrive as "1729997.0"
        # would mint Issuer_1729997.0 -- a THIRD node for the same company.
        # Nothing upstream prevents that; two characters here do.
        #
        # regexp_extract returns "" when the pattern does not match, which is
        # how a non-matching URI falls through to .otherwise() below.
        digits = F.regexp_extract(
            column, f"^{re.escape(prefix)}" + r"(\d+)(?:\.0+)?$", 1
        )
        result = F.when(
            (F.length(digits) > 0) & (F.length(digits) <= CIK_DIGITS),
            F.concat(F.lit(prefix), _padded(digits, CIK_DIGITS)),
        ).otherwise(result)
    return result


def _canonical_identifier_literal(predicate: Column, obj: Column) -> Column:
    """Rewrite a numeric-typed identifier literal to its canonical string."""
    digits = F.regexp_extract(obj, _NUMERIC_IDENTIFIER, 1)
    numeric = F.length(digits) > 0

    # Start from the URI rule, so an object that is neither a padded
    # identifier nor a code still gets the CIK-keyed URI repair.
    result = _canonical_cik_uri(obj)

    # Codes first, then the padded identifiers. The two predicate sets are
    # disjoint, so the order of these branches carries no meaning -- but the
    # widths do, which is why each is its own condition rather than one branch
    # with a computed width.
    result = F.when(
        predicate.isin(list(OPAQUE_CODE_PREDICATES)) & numeric, digits
    ).otherwise(result)

    for width in sorted(set(PADDED_IDENTIFIER_WIDTHS.values())):
        predicates = [
            p for p, w in PADDED_IDENTIFIER_WIDTHS.items() if w == width
        ]
        result = F.when(
            predicate.isin(predicates) & numeric & (F.length(digits) <= width),
            _padded(digits, width),
        ).otherwise(result)

    return result


def canonicalize_sec_identifiers(triples_df: DataFrame) -> DataFrame:
    """Collapse the several spellings of a SEC identifier onto one.

    Rewrites, in place and without changing the row count:

      * subject and object URIs of the form
        ``.../id/sec-filings/Issuer_<cik>`` (and the other CIK_KEYED_LOCALS),
        zero-padded to CIK_DIGITS;
      * the object of every PADDED_IDENTIFIER_WIDTHS statement, with a trailing
        ``.0`` dropped and the digits zero-padded to that identifier's width;
      * the object of every OPAQUE_CODE_PREDICATES statement, with the trailing
        ``.0`` dropped and no padding.

    For the CIK, both the URI and the literal halves are required and neither
    is sufficient. Padding only the URI merges the nodes and leaves
    ``_unify_company_entities`` grouping on two different literals; padding
    only the literal unifies two nodes that stay two nodes.

    Args:
        triples_df: canonical (subject, predicate, object) frame.

    Returns:
        The same frame with SEC identifiers in canonical form.
    """
    return triples_df.select(
        _canonical_cik_uri(F.col("subject")).alias("subject"),
        F.col("predicate"),
        _canonical_identifier_literal(
            F.col("predicate"), F.col("object")
        ).alias("object"),
    )


def canonicalize_source_triples(triples_df: DataFrame) -> DataFrame:
    """Apply every source-shape repair to a freshly loaded triples frame.

    One entry point so the loader does not grow a rule list, and so a new
    source's repair lands beside the existing ones rather than in whichever
    caller noticed the problem.
    """
    logger.info("Canonicalizing source identifier shapes")
    return canonicalize_sec_identifiers(triples_df)

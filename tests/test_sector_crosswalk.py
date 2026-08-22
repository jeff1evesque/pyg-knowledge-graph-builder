"""
Tests for the curated sector crosswalks (enrichment/sector_crosswalk.py).

Pure Python, no SparkSession — these pin a hand-maintained decision table, and
the decisions are the thing worth pinning.

The unusual one here is test_the_deliberate_gaps_stay_gaps. Most tests fail when
someone breaks the code; that one fails when someone IMPROVES the table by
filling in three sectors that were left empty on purpose. That is intended. The
gaps encode a judgment — that the equity and economic sector vocabularies are
not parallel taxonomies — and a contributor completing the table by symmetry
would silently assert relationships that do not exist, with no error anywhere.
The test is the only place that judgment is enforceable.
"""
import pytest

from spark_jobs.enrichment import sector_crosswalk as xwalk
from spark_jobs.enrichment.intra_source.market.patterns import (
    DEFAULT_MARKET_SECTOR_PATTERNS, _gics_sector_to_pascal,
)
from spark_jobs.utils.rdf_utils import BLS_ENRICHMENT, NAMESPACE_PREFIXES


def _relation_name(predicate_uri):
    """The relation name node_mapper would shorten this predicate to.

    Relation GROUPS in graph_schema.json are keyed on exactly this string; see
    metadata_writer._build_relation_groups.
    """
    for namespace, prefix in NAMESPACE_PREFIXES:
        if predicate_uri.startswith(namespace):
            return f"{prefix}_{predicate_uri[len(namespace):]}"
    raise AssertionError(f"{predicate_uri} is in no registered namespace")


# ======================================================================
# The gaps
# ======================================================================

def test_the_deliberate_gaps_stay_gaps():
    """Three equity sectors map to NOTHING, and must keep mapping to nothing.

    Information Technology and Communication Services are unmapped because the
    economic `Information` sector covers publishing, telecom and data services
    while the equity `Information Technology` sector covers semiconductors,
    hardware and software. The names align; the contents do not. Linking them
    ties a chip maker to the price index for telephone service.

    Utilities is unmapped because no counterpart exists: the economic `Energy`
    sector is fuel prices, not regulated utilities.

    If this test is failing for you because you added one of these rows, the
    row is the bug.
    """
    for gics in xwalk.DELIBERATELY_UNMAPPED_EQUITY_SECTORS:
        assert gics not in xwalk.EQUITY_TO_ECONOMIC_SECTORS, (
            f"{gics!r} is mapped to an economic sector. It is in "
            "DELIBERATELY_UNMAPPED_EQUITY_SECTORS because the two "
            "vocabularies are not parallel taxonomies — read the module "
            "docstring before removing it from that tuple."
        )


def test_the_unmapped_three_are_real_gics_sectors():
    """A gap only means something if the sector it names actually exists.

    Guards the case where a GICS rename turns a deliberate gap into a dead
    string, quietly re-opening the mapping it was there to prevent.
    """
    known = {
        pattern["description"].split(" companies")[0]
        for pattern in DEFAULT_MARKET_SECTOR_PATTERNS.values()
    }
    # The default table describes a couple of them loosely ("Financial
    # services companies"), so match on the derived URI instead of the prose.
    known_pascal = {
        str(pattern["sector_uri"]).rsplit("/", 1)[-1]
        for pattern in DEFAULT_MARKET_SECTOR_PATTERNS.values()
    }

    for gics in xwalk.DELIBERATELY_UNMAPPED_EQUITY_SECTORS:
        assert f"{_gics_sector_to_pascal(gics)}Sector" in known_pascal, (
            f"{gics!r} is not a GICS sector this pipeline classifies into"
        )
    assert known  # the loose-prose set is unused but proves the table is populated


# ======================================================================
# The mapped rows
# ======================================================================

def test_every_mapped_row_is_a_real_economic_sector():
    for gics, mapped in xwalk.EQUITY_TO_ECONOMIC_SECTORS.items():
        assert mapped, f"{gics} maps to an empty tuple; use the unmapped tuple"
        for sector_uri, confidence in mapped:
            assert sector_uri.startswith(str(BLS_ENRICHMENT)), (
                f"{gics} -> {sector_uri} is not an economic sector URI"
            )
            assert confidence in (
                xwalk.CONFIDENCE_STRONG, xwalk.CONFIDENCE_MODERATE
            ), f"{gics} -> {sector_uri} has confidence {confidence!r}"


def test_no_equity_sector_is_both_mapped_and_declared_unmapped():
    overlap = set(xwalk.EQUITY_TO_ECONOMIC_SECTORS) & set(
        xwalk.DELIBERATELY_UNMAPPED_EQUITY_SECTORS
    )
    assert not overlap, overlap


def test_mapped_targets_are_distinct_within_a_row():
    """A row listing one economic sector twice would double-weight it."""
    for gics, mapped in xwalk.EQUITY_TO_ECONOMIC_SECTORS.items():
        targets = [uri for uri, _confidence in mapped]
        assert len(targets) == len(set(targets)), gics


# ======================================================================
# Membership and similarity stay different claims
# ======================================================================

def test_related_and_belongs_are_distinct_relations():
    """They must not collapse into one relation group.

    A constituent BELONGS TO its GICS sector. That GICS sector is merely
    RELATED TO an economic sector, through a table someone wrote by hand.
    Sharing one predicate would let a GNN weight a curated resemblance exactly
    as heavily as a stated membership, which is the whole reason the weaker
    predicate exists.

    Relation groups are keyed on the shortened relation name, so this asserts
    at the level graph_schema.json actually groups on rather than on the URIs.
    """
    from spark_jobs.utils.rdf_utils import MARKET_ENRICHMENT

    related = xwalk.RELATED_TO_ECONOMIC_SECTOR
    market_belongs = str(MARKET_ENRICHMENT.belongsToSector)
    bls_belongs = str(BLS_ENRICHMENT.belongsToSector)

    assert related != market_belongs
    assert related != bls_belongs

    names = {
        _relation_name(related),
        _relation_name(market_belongs),
        _relation_name(bls_belongs),
    }
    assert len(names) == 3, (
        f"expected three distinct relation groups, got {sorted(names)}"
    )


def test_the_equity_sector_type_is_not_the_economic_sector_type():
    """Typing a GICS sector as an economic sector asserts the equivalence
    this module exists to deny."""
    assert xwalk.EQUITY_SECTOR_TYPE != str(BLS_ENRICHMENT.EconomicSector)


# ======================================================================
# SIC divisions
# ======================================================================

@pytest.mark.parametrize("sic,expected", [
    ("2836", str(BLS_ENRICHMENT.ManufacturingSector)),      # biological products
    ("6021", str(BLS_ENRICHMENT.FinancialSector)),          # national banks
    ("6035", str(BLS_ENRICHMENT.FinancialSector)),          # savings institutions
    ("3669", str(BLS_ENRICHMENT.ManufacturingSector)),      # communications equipment
    ("1311", str(BLS_ENRICHMENT.NaturalResourcesSector)),   # crude petroleum
    ("1531", str(BLS_ENRICHMENT.ConstructionTradesSector)),  # operative builders
    ("4512", str(BLS_ENRICHMENT.TransportationSector)),     # air transport
    ("5812", str(BLS_ENRICHMENT.RetailSector)),             # eating places
    ("7372", str(BLS_ENRICHMENT.ProfessionalServicesSector)),  # prepackaged software
    ("9111", str(BLS_ENRICHMENT.GovernmentSector)),         # executive offices
])
def test_sic_codes_resolve_to_their_division_sector(sic, expected):
    _division, sector = xwalk.economic_sector_for_sic(sic)
    assert sector == expected


def test_a_sic_code_with_a_dropped_leading_zero_still_resolves():
    """A numeric-typed literal loses the leading zero: 0100 arrives as 100.

    Slicing the raw text at two characters reads that as major group "10"
    (Mining) instead of "01" (Agriculture) — right only by accident here, and
    wrong for a code like 0755, which would read as "75" (Auto Services).
    """
    assert xwalk.economic_sector_for_sic("100")[1] == str(
        BLS_ENRICHMENT.NaturalResourcesSector
    )
    assert xwalk.economic_sector_for_sic("0100")[1] == str(
        BLS_ENRICHMENT.NaturalResourcesSector
    )

    assert xwalk.economic_sector_for_sic("755")[1] == str(
        BLS_ENRICHMENT.NaturalResourcesSector
    ), "0755 is Agricultural Services, not Auto Services"


@pytest.mark.parametrize("sic", ["", "  ", "abc", "12a4", "12345", "1800", "6800", "9000"])
def test_an_unclassifiable_sic_gets_no_sector_rather_than_a_wrong_one(sic):
    """SIC reserves gaps between divisions (18-19, 68-69, 90).

    A filing landing in one should get no sector. Snapping it to the nearest
    division would emit a confident, wrong membership claim.
    """
    assert xwalk.economic_sector_for_sic(sic) == ("", "")


def test_sic_divisions_do_not_overlap():
    """Two divisions claiming one major group would give a company two sectors
    with no way to tell which is meant."""
    seen = {}
    for low, high, division, _sector in xwalk.SIC_DIVISIONS:
        assert low <= high, division
        for group in range(low, high + 1):
            assert group not in seen, (
                f"major group {group:02d} claimed by both "
                f"{seen[group]!r} and {division!r}"
            )
            seen[group] = division

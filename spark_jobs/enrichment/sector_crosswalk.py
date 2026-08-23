"""
Equity sector -> economic sector, and SIC division -> economic sector.

TWO CROSSWALKS, ONE REASON
--------------------------
The graph carries an economic-sector vocabulary (BLS_ENRICHMENT.*Sector, built
from bls/patterns.py) that is its best-connected hub: every economic-indicator
type attaches to it. Two other source families classify companies by industry
and reach it not at all -- the market feed by GICS sector, the filings by SIC
code -- so each mints or implies its own parallel taxonomy and the hub stays
one-source.

Both tables here are CURATED BY HAND and both are deliberately incomplete. That
is the design, not an unfinished state.

WHY NOT MECHANICAL
------------------
These are not parallel taxonomies with a missing join key. The GICS
classification sorts COMPANIES BY REVENUE SOURCE. The economic sectors sort
PRICE AND EMPLOYMENT SERIES BY CONSUMPTION OR INDUSTRY CATEGORY. A 1:1 mapping
between them would assert relationships that do not exist, and it would do so
with the same confidence as the rows that are real.

Two GICS sectors are therefore mapped to NOTHING, on purpose:

  * Information Technology. The economic `Information` sector covers
    publishing, telecommunications and data services; the equity `Information
    Technology` sector covers semiconductors, hardware and software. The names
    align and the contents do not. Linking them ties a chip maker to the price
    index for telephone service.
  * Communication Services. The mirror of the same problem from the other side.

And one is mapped to nothing because there is no counterpart at all:

  * Utilities. The economic `Energy` sector is fuel PRICES. A regulated utility
    is not a fuel price, and the economic vocabulary has no sector for one.

If you are here to "finish" the table, read the test that guards these three
first (test_cross_source_linker: the deliberately-unmapped test). It fails when
they are filled in, because the gaps ARE the decision -- the same way the
ontology mapper emits NO rdfs:domain for a predicate with more than one
candidate class, rather than asserting both.

WHY A WEAKER PREDICATE
----------------------
Emitted under `relatedToEconomicSector`, never `belongsToSector`. Membership and
similarity are different claims. A constituent BELONGS TO its GICS sector; that
GICS sector is merely RELATED TO an economic sector, at a confidence this table
states row by row. Collapsing the two into one relation denies a GNN the ability
to weight them differently, which is the only reason the distinction is worth
carrying into the graph at all.
"""
from typing import Dict, Tuple

from spark_jobs.utils.rdf_utils import BLS_ENRICHMENT, MARKET_ENRICHMENT

# ============================================
# Confidence
# ============================================
#
# Carried as an edge property rather than folded into the predicate, so the
# relation stays one type and the strength stays a feature. Splitting it into
# relatedToEconomicSectorStrongly / ...Moderately would double the edge types
# for a distinction a scalar states better.
CONFIDENCE_STRONG = "strong"
CONFIDENCE_MODERATE = "moderate"

RELATED_TO_ECONOMIC_SECTOR = str(MARKET_ENRICHMENT.relatedToEconomicSector)
RELATION_CONFIDENCE = str(MARKET_ENRICHMENT.relationConfidence)

# The class the equity sector nodes carry. They need one: market_linker mints
# these URIs and states belongsToSector against them, but never typed them, and
# node_mapper only makes a node out of a typed URI -- so every equity sector was
# a dangling object and every market belongsToSector edge was dropped during
# edge resolution. "The market side has zero edges to the economic hub" was
# partly this: it had no sector nodes of its own either.
#
# A distinct class from bls:EconomicSector, deliberately. Typing a GICS sector
# as an economic sector would assert the equivalence this whole module exists to
# avoid.
EQUITY_SECTOR_TYPE = str(MARKET_ENRICHMENT.EquitySector)


# ============================================
# GICS sector -> economic sector(s)
# ============================================
#
# Keyed on the GICS sector name exactly as the constituents CSV spells it, so
# the market sector URI is derived with the same _gics_sector_to_pascal the
# enricher uses rather than spelled a second time here.
#
# Many-to-many by design: one revenue-source category maps onto several
# consumption categories, and vice versa.
EQUITY_TO_ECONOMIC_SECTORS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    # Same thing under two names. Oil and gas companies ARE the energy prices.
    "Energy": (
        (str(BLS_ENRICHMENT.EnergySector), CONFIDENCE_STRONG),
    ),
    # Likewise: healthcare providers and the healthcare price/employment series
    # describe one industry.
    "Health Care": (
        (str(BLS_ENRICHMENT.HealthcareSector), CONFIDENCE_STRONG),
    ),
    # Staples splits cleanly across three consumption categories, and covers
    # essentially all of each.
    "Consumer Staples": (
        (str(BLS_ENRICHMENT.FoodSector), CONFIDENCE_STRONG),
        (str(BLS_ENRICHMENT.TobaccoSector), CONFIDENCE_STRONG),
        (str(BLS_ENRICHMENT.PersonalCareSector), CONFIDENCE_STRONG),
    ),
    # Moderate, and worth stating why it is not strong: REIT income and shelter
    # costs are genuinely linked, but they are opposite sides of the same
    # payment. The relationship is real; the direction is not what the name
    # suggests.
    "Real Estate": (
        (str(BLS_ENRICHMENT.HousingSector), CONFIDENCE_MODERATE),
    ),
    # Banks and the financial-services price series overlap substantially, but
    # the equity sector includes insurers and exchanges the series does not
    # track.
    "Financials": (
        (str(BLS_ENRICHMENT.FinancialSector), CONFIDENCE_MODERATE),
    ),
    # A wide equity sector spanning three narrower economic ones; no single
    # pairing is strong, and omitting the other two would misrepresent it.
    "Industrials": (
        (str(BLS_ENRICHMENT.ManufacturingSector), CONFIDENCE_MODERATE),
        (str(BLS_ENRICHMENT.TransportationSector), CONFIDENCE_MODERATE),
        (str(BLS_ENRICHMENT.ConstructionTradesSector), CONFIDENCE_MODERATE),
    ),
    # The widest of them. Retailers, restaurants, hotels and clothing sit in one
    # equity bucket and four economic ones.
    "Consumer Discretionary": (
        (str(BLS_ENRICHMENT.RetailSector), CONFIDENCE_MODERATE),
        (str(BLS_ENRICHMENT.RecreationSector), CONFIDENCE_MODERATE),
        (str(BLS_ENRICHMENT.LeisureHospitalitySector), CONFIDENCE_MODERATE),
        (str(BLS_ENRICHMENT.ApparelSector), CONFIDENCE_MODERATE),
    ),
    # Extraction and processing: the raw input and the industry that works it.
    "Materials": (
        (str(BLS_ENRICHMENT.NaturalResourcesSector), CONFIDENCE_MODERATE),
        (str(BLS_ENRICHMENT.ManufacturingSector), CONFIDENCE_MODERATE),
    ),
}

# The three that map to nothing, named rather than merely absent.
#
# Stated explicitly so the omission is legible as a decision and so a test can
# assert it. An empty tuple in the table above would read as an oversight; a
# missing key would read as one too.
DELIBERATELY_UNMAPPED_EQUITY_SECTORS: Tuple[str, ...] = (
    "Information Technology",
    "Communication Services",
    "Utilities",
)


# ============================================
# SIC division -> economic sector
# ============================================
#
# filings:hasSic is stated ON THE FILING, and is declared
# `rdfs:domain filings:SECFiling` by the mapper's ontology. Consuming it is what
# lets a filing reach a sector through real data rather than through a keyword
# match on its URI -- see CrossSourceLinker._link_by_sector for what that
# keyword classifier was doing and why the filings half of it is now gone.
#
# Coverage is PARTIAL and that is EDGAR's doing, not this table's: the term maps
# from `_source.sics`, an array EDGAR routinely returns empty, and the mapper
# omits the triple rather than asserting a blank literal. 38% of filings on the
# e2e fixtures.
#
# There is no plan to move this term to the issuer node. An earlier draft of
# this comment said there was, sourced from an issue that also claimed the term
# was on every filing; both were checked against the mapper and neither held.
# SIC arguably describes the company rather than the document, so moving it is a
# defensible design opinion -- but it is nobody's work in progress, and this
# reader must not be written as though it were.
#
# Keyed on the SIC MAJOR GROUP (the leading two digits), grouped into the
# standard SIC divisions, because a 4-digit industry code is finer than the
# ~19-sector economic vocabulary can express. The division boundaries are the
# federal ones and have not moved since 1987.
#
# This reaches the DERIVED sector vocabulary only. It deliberately does NOT
# join the raw industry terms (empsit:hasIndustry, jolts:hasIndustry), which
# are keyed on a different classification system and would need a real
# crosswalk file. That is out of scope, not overlooked.
SIC_DIVISIONS: Tuple[Tuple[int, int, str, str], ...] = (
    # (first major group, last major group, division name, economic sector)
    (1, 9, "Agriculture, Forestry, Fishing", str(BLS_ENRICHMENT.NaturalResourcesSector)),
    (10, 14, "Mining", str(BLS_ENRICHMENT.NaturalResourcesSector)),
    (15, 17, "Construction", str(BLS_ENRICHMENT.ConstructionTradesSector)),
    (20, 39, "Manufacturing", str(BLS_ENRICHMENT.ManufacturingSector)),
    (40, 49, "Transportation and Public Utilities", str(BLS_ENRICHMENT.TransportationSector)),
    # Wholesale is mapped to Retail with open eyes: the economic vocabulary has
    # no wholesale category, and goods distribution is the nearest true
    # neighbour. It is the weakest row in this table.
    (50, 51, "Wholesale Trade", str(BLS_ENRICHMENT.RetailSector)),
    (52, 59, "Retail Trade", str(BLS_ENRICHMENT.RetailSector)),
    (60, 67, "Finance, Insurance, Real Estate", str(BLS_ENRICHMENT.FinancialSector)),
    (70, 89, "Services", str(BLS_ENRICHMENT.ProfessionalServicesSector)),
    (91, 99, "Public Administration", str(BLS_ENRICHMENT.GovernmentSector)),
)


def economic_sector_for_sic(sic: str) -> Tuple[str, str]:
    """(division name, economic sector URI) for a SIC code, or ("", "").

    Accepts the code in any width upstream states it -- SIC codes are 3 or 4
    digits and a leading zero is routinely dropped by a numeric-typed literal,
    so "100" and "0100" are the same mining code. Only the major group (the
    leading two digits of the four-digit form) is read.

    Returns empty strings rather than raising for an unrecognised code: SIC
    reserves gaps between divisions (18-19, 68-69, 90) and a filing landing in
    one should get no sector, not a wrong one.
    """
    digits = (sic or "").strip()
    if not digits.isdigit():
        return ("", "")

    # Normalise to the 4-digit form before slicing, so a dropped leading zero
    # cannot shift the major group -- "100" read as major group "10" (Mining)
    # is right only by accident, and "755" read as "75" is wrong.
    if len(digits) > 4:
        return ("", "")
    major = int(digits.zfill(4)[:2])

    for low, high, division, sector in SIC_DIVISIONS:
        if low <= major <= high:
            return (division, sector)

    return ("", "")

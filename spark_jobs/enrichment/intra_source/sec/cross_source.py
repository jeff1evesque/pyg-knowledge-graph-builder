"""SEC's side of cross-source linking: its companies, and their SIC sectors.

The SEC source spec (spark_jobs/sources/sec.py) hands these functions to the
cross-source linker, which builds the company hub and the sector links from
what every source in the run gives it.
"""
import logging
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from rdflib.namespace import RDF

from spark_jobs.enrichment.cross_source_linker import (
    UNIFIED_COMPANY_PREFIX,
    CrossSourceContext,
)
from spark_jobs.enrichment.intra_source.market.symbols import equity_symbol
from spark_jobs.enrichment.sector_crosswalk import SIC_DIVISIONS
from spark_jobs.sources.spec import CompanyKeys
from spark_jobs.utils.rdf_utils import BLS_ENRICHMENT, SEC_FILINGS
from spark_jobs.utils.spark_rdf_utils import extract_property

logger = logging.getLogger(__name__)

_RDF_TYPE = str(RDF.type)

_FILINGS_HAS_ISSUER_CIK = str(SEC_FILINGS.hasIssuerCik)
_FILINGS_HAS_ISSUER_TRADING_SYMBOL = str(SEC_FILINGS.hasIssuerTradingSymbol)
_FILINGS_HAS_ISSUER = str(SEC_FILINGS.hasIssuer)
_FILINGS_HAS_SIC = str(SEC_FILINGS.hasSic)


def issuer_ciks(triples_df: DataFrame) -> DataFrame:
    """(issuer_uri, cik) for every SEC issuer that states a CIK.

    This is the SEC half of the company bridge, and it replaces a half
    keyed on ``hasIssuerTradingSymbol``. The ticker was never the issuer's
    identifier here -- it is emitted by exactly one upstream parser, the
    ownership-form document walker, so it requires a filing that (a) had
    its document fetched and stored, (b) stored it as XML, and (c) is a
    Form 3/4/5, the only forms whose schema carries <issuerTradingSymbol>.
    The EDGAR search API never returns a ticker at all.

    MEASURED, from a census of 23,983 filing rows across 2026-08-07,
    2026-08-06, 2026-08-04 and 2025-10-02 (bin/census_sec_terms.py):

        rows stating hasIssuerTradingSymbol   5,893   24.57% of all filings
          of which root_form 4                5,229   100% of the 5,229
          of which root_form 3                  655   100% of the 655
          of which root_form 5                    9   100% of the 9
        rows on any other form                    0

    So the OLD bridge's ceiling was exactly the ownership-form share --
    about a quarter of filings -- and no amount of market-side work could
    lift it. hasIssuerCik has no such ceiling: the issuer states it on
    every filing, from both the metadata path and the document path, which
    is what moves this from a quarter to all of them.

    The CIK arrives already zero-padded to ten, because
    utils/sec_identifiers.canonicalize_sec_identifiers runs in the LOADER,
    before any enricher sees the frame. Nothing is re-padded here on
    purpose: if that ever stops being true the join under-covers visibly
    rather than being silently patched in two places that can disagree.
    """
    return extract_property(
        triples_df, _FILINGS_HAS_ISSUER_CIK, "raw_cik"
    ).select(
        F.col("subject").alias("issuer_uri"),
        F.trim(F.col("raw_cik")).alias("cik"),
    ).filter(F.length(F.col("cik")) > 0).dropDuplicates()


def company_keys(context: CrossSourceContext) -> CompanyKeys:
    """The issuers, by the CIK each states, and the ticker pairings the filings
    carry.

    An ownership-form issuer states both its ticker and its CIK, so the graph
    already carries the pairing for every issuer whose filings were ingested.
    It needs no external file, and it cannot disagree with the issuers' own
    CIKs, because it IS those issuers read a second way. That is why these
    pairings rank first (priority 0), ahead of the constituents CSV's.
    """
    issuers = issuer_ciks(context.triples_df)

    symbol_ciks = extract_property(
        context.triples_df, _FILINGS_HAS_ISSUER_TRADING_SYMBOL, "raw_symbol"
    ).select(
        F.col("subject").alias("issuer_uri"),
        equity_symbol(F.col("raw_symbol")).alias("symbol"),
    ).filter(F.length(F.col("symbol")) > 0).join(
        issuers, "issuer_uri", "inner"
    ).select(
        F.col("symbol"),
        F.col("cik"),
        F.lit(0).alias("priority"),
    )

    return CompanyKeys(
        entities=issuers.select(F.col("issuer_uri").alias("entity"), F.col("cik")),
        symbol_ciks=symbol_ciks,
    )


def sector_keys(context: CrossSourceContext) -> Optional[DataFrame]:
    """The filings' own industry code, instead of a keyword match.

    filings:hasSic is emitted by the mapper and this repo read it nowhere,
    so the filings side reached the economic sector hub only through
    the sector keyword step, which matches substrings of a URI's local name
    and for a filing is matching an accession number.

    IT IS NOT ON EVERY FILING, and do not read a partial result here as a
    defect. The term comes from EDGAR's ``_source.sics`` array, which is
    routinely empty, and the mapper omits a triple rather than asserting a
    blank literal -- so a filing EDGAR left unclassified says nothing about
    its SIC. Measured on the e2e fixtures: 15 of 40 filings, 38%, giving
    14 of 39 companies a sector.

    That ceiling is EDGAR's, not this join's. Widening it means finding a
    second source for the classification, not fixing anything here.

    The sector lands on the COMPANY, not on the filing, and that is what
    makes it worth doing here rather than as a SEC intra-source step: a
    filing is a document, its industry is a fact about the issuer, and
    every filing by that issuer then inherits it through the company node
    Part 1 unified. The chain is

        Filing -hasSic-> code
        Filing -hasIssuer-> Issuer -hasIssuerCik-> CIK -> Company

    so this depends on the CIK-keyed company node existing, which is why
    it lands with Part 1 rather than before it. It needs nothing from any
    other source, so a run without market still gets it.

    belongsToSector, not relatedToEconomicSector: this IS membership. The
    issuer's registered industry classification is a claim about what the
    company does, unlike the GICS-to-economic mapping, which is a claim about
    two taxonomies resembling each other.
    """
    sic = extract_property(
        context.triples_df, _FILINGS_HAS_SIC, "raw_sic"
    ).select(
        F.col("subject").alias("filing"),
        F.trim(F.col("raw_sic")).alias("sic"),
    ).filter(F.length(F.col("sic")) > 0)

    if sic.head(1) == []:
        return None

    filing_issuers = context.triples_df.filter(
        F.col("predicate") == _FILINGS_HAS_ISSUER
    ).select(
        F.col("subject").alias("filing"),
        F.col("object").alias("issuer_uri"),
    )

    # SIC major group -> economic sector, expanded to one row per group so
    # the join is an equality rather than a range. Ninety-nine rows at
    # most, and it keeps the division boundaries stated once, in the
    # crosswalk module.
    division_rows = [
        (f"{group:02d}", division, sector)
        for low, high, division, sector in SIC_DIVISIONS
        for group in range(low, high + 1)
    ]
    divisions = context.spark.createDataFrame(
        division_rows, schema=["major_group", "division", "economic_sector"]
    )

    companies = (
        sic
        .join(filing_issuers, "filing", "inner")
        .join(issuer_ciks(context.triples_df), "issuer_uri", "inner")
        # Normalise to the 4-digit form before slicing. A SIC code typed
        # numeric upstream loses its leading zero, and "100" sliced at two
        # is "10" -- right only by accident -- while "755" sliced at two is
        # "75", which is a different division entirely.
        .withColumn(
            "major_group",
            F.substring(F.lpad(F.col("sic"), 4, "0"), 1, 2),
        )
        .filter(F.length(F.col("sic")) <= 4)
        .join(F.broadcast(divisions), "major_group", "inner")
        .select("cik", "economic_sector")
        .distinct()
    )

    company_uri = F.concat(F.lit(UNIFIED_COMPANY_PREFIX), F.col("cik"))

    belongs = companies.select(
        company_uri.alias("subject"),
        F.lit(str(BLS_ENRICHMENT.belongsToSector)).alias("predicate"),
        F.col("economic_sector").alias("object"),
    )

    sector_types = companies.select("economic_sector").distinct().select(
        F.col("economic_sector").alias("subject"),
        F.lit(_RDF_TYPE).alias("predicate"),
        F.lit(str(BLS_ENRICHMENT.EconomicSector)).alias("object"),
    )

    logger.info("  SIC sector triples prepared (lazy)")
    return belongs.unionByName(sector_types)

"""
Market Sector Patterns - Industry sectors and market segments.

Sector definitions are loaded dynamically from the S&P 500 tickers CSV
at runtime (preferred), falling back to hardcoded defaults if the CSV
is unavailable.

The tickers CSV is produced by the market-morning pipeline and stored at:
    s3://{bucket}/{tickers_latest_key}

Expected CSV format (GitHub S&P 500 constituents):
    Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,Date added,CIK,Founded
    AAPL,Apple Inc.,Information Technology,...
    MSFT,Microsoft Corporation,Information Technology,...
    JPM,JPMorgan Chase & Co.,Financials,...

Tickers are grouped by GICS Sector to build sector patterns. The GICS
sector name is converted to a snake_case key and PascalCase URI suffix:
    "Information Technology" → key: "information_technology_sector"
                            → URI: MARKET_ENRICHMENT.InformationTechnologySector
                            → relationship: MARKET_ENRICHMENT.informationTechnologySectorCorrelation

Aligned with the flat snapshot model. Sector classification is
based on the equity ticker symbol (the `symbol` property on
EquitySnapshot nodes).

Option snapshots inherit sector classification from their
underlying equity via the `underlyingSymbol` property.
"""
import csv
import io
import logging
from typing import Dict, Any, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from spark_jobs.utils.canonicalization import CIK_DIGITS
from spark_jobs.utils.rdf_utils import MARKET_ENRICHMENT

logger = logging.getLogger(__name__)

# ============================================
# Constituent CSV columns
# ============================================
#
# Named rather than spelled at each call site because three readers now share
# this file and each wants a different subset. The CSV is the GitHub S&P 500
# constituents table; see the module docstring for the full header.
SYMBOL_COLUMN = "Symbol"
SECTOR_COLUMN = "GICS Sector"
SUB_INDUSTRY_COLUMN = "GICS Sub-Industry"
CIK_COLUMN = "CIK"

# ============================================
# GICS Sector → enrichment naming conventions
# ============================================

# Maps GICS Sector names to (snake_case_key, relationship_suffix)
# If a GICS sector is not in this map, it's derived automatically.
_GICS_SECTOR_OVERRIDES: Dict[str, Dict[str, str]] = {
    # Override where the auto-derived name would be awkward
}


def _gics_sector_to_key(gics_sector: str) -> str:
    """
    Convert GICS Sector name to a snake_case pattern key.

    "Information Technology" → "information_technology_sector"
    "Health Care" → "health_care_sector"
    "Consumer Discretionary" → "consumer_discretionary_sector"
    """
    normalized = gics_sector.strip().lower()
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("-", "_")
    return f"{normalized}_sector"


def _gics_sector_to_pascal(gics_sector: str) -> str:
    """
    Convert GICS Sector name to PascalCase for URI construction.

    "Information Technology" → "InformationTechnology"
    "Health Care" → "HealthCare"
    "Consumer Discretionary" → "ConsumerDiscretionary"
    """
    return "".join(word.capitalize() for word in gics_sector.strip().split())


# ============================================
# CIK padding
# ============================================


def padded_cik(raw: str) -> Optional[str]:
    """The constituents CSV's CIK, in the 10-digit form the filings state.

    THE WHOLE POINT OF THIS FUNCTION. The CSV states the CIK unpadded --
    ``320193`` -- and every SEC filing states it zero-padded to ten --
    ``0000320193``. Compared as strings those are different keys, so a join
    between the market side and the filings side on the raw CSV value matches
    NOTHING and reports no error: the bridge comes back empty and reads as
    "no overlap in this data" rather than as a defect. That is the failure
    mode this repairs, and it is why the value is padded at the point it is
    read rather than at the point it is joined.

    ``zfill`` rather than Spark's ``lpad``: lpad TRUNCATES, from the right,
    when the input is longer than the target width, so a malformed 11-digit
    value would silently become a different, plausible filer. zfill leaves an
    over-long value alone, which lets the length guard below reject it
    instead. The same trap is documented in utils/canonicalization._padded and
    cost a wrong-state bug in cross_source_linker's FIPS matching.

    Returns None -- with a warning -- for anything that is not a CIK: an empty
    cell, a value carrying non-digits, or one already longer than CIK_DIGITS.
    None rather than a raise, because one malformed row in a 500-row index
    table should drop that constituent, not fail the build.
    """
    value = (raw or "").strip()
    if not value:
        return None

    if not value.isdigit():
        logger.warning(f"Constituent CIK is not all digits, skipping: {raw!r}")
        return None

    if len(value) > CIK_DIGITS:
        logger.warning(
            f"Constituent CIK is longer than {CIK_DIGITS} digits, "
            f"skipping: {raw!r}"
        )
        return None

    return value.zfill(CIK_DIGITS)


def assert_ciks_are_padded(ticker_cik_map: Dict[str, str]) -> None:
    """Fail loudly on a ticker->CIK map that would join against nothing.

    The empty bridge is the dangerous outcome here, not the exception. A map
    carrying unpadded CIKs produces a join that matches zero filings, emits
    zero edges, raises nothing, and looks exactly like a fixture with no
    overlap -- so it survives review and ships. Any future path that builds
    this map (a different CSV, a hand-supplied override, the universal
    registry planned upstream) has to come through here, and a wrong-width key
    stops the build with the offending entries named.

    Raises:
        ValueError: if any value is not exactly CIK_DIGITS digits.
    """
    offenders = {
        ticker: cik
        for ticker, cik in ticker_cik_map.items()
        if not (isinstance(cik, str) and cik.isdigit() and len(cik) == CIK_DIGITS)
    }
    if offenders:
        raise ValueError(
            f"ticker->CIK map carries {len(offenders)} key(s) that are not "
            f"{CIK_DIGITS}-digit zero-padded CIKs and would join against no "
            f"filing: {dict(sorted(offenders.items())[:10])}"
        )


# ============================================
# S3 Loader — reads the tickers CSV
# ============================================


def _read_constituents(
    bucket: str,
    key: str,
    required_columns: set,
    s3_client=None,
) -> Optional[List[Dict[str, str]]]:
    """Fetch and parse the constituents CSV, or None if it is unusable.

    Shared by the three readers below so the S3 error taxonomy, the empty-body
    check and the missing-column check are stated once. Each reader passes the
    columns IT needs: the sector reader must not fail because a CSV vintage
    predates the CIK column, and the CIK reader must not fail because the
    sector column moved. Every failure path returns None and logs, because
    every caller has a defined behaviour without this file.
    """
    if not bucket or not key:
        logger.debug("No S3 bucket/key provided for market sector definitions")
        return None

    client = s3_client or boto3.client("s3")

    try:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read().decode("utf-8")

        if not body.strip():
            logger.warning(
                f"Empty file at s3://{bucket}/{key} — "
                f"falling back to defaults"
            )
            return None

        reader = csv.DictReader(io.StringIO(body))

        if reader.fieldnames is None:
            logger.warning(
                f"No CSV headers in s3://{bucket}/{key} — "
                f"falling back to defaults"
            )
            return None

        missing = set(required_columns) - set(reader.fieldnames)
        if missing:
            logger.warning(
                f"CSV at s3://{bucket}/{key} missing columns: "
                f"{missing}. Available: {reader.fieldnames} — "
                f"falling back to defaults"
            )
            return None

        return list(reader)

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchKey":
            logger.info(
                f"Tickers file not found at s3://{bucket}/{key} — "
                f"using defaults"
            )
        elif error_code == "NoSuchBucket":
            logger.warning(
                f"Bucket does not exist: {bucket} — using defaults"
            )
        else:
            logger.warning(
                f"S3 error loading tickers from "
                f"s3://{bucket}/{key}: {e} — using defaults"
            )
        return None

    except (csv.Error, ValueError, TypeError, UnicodeDecodeError) as e:
        logger.warning(
            f"Failed to parse tickers CSV from "
            f"s3://{bucket}/{key}: {e} — using defaults"
        )
        return None

    except Exception as e:
        logger.warning(
            f"Unexpected error loading tickers from "
            f"s3://{bucket}/{key}: {e} — using defaults"
        )
        return None


def load_sector_patterns_from_s3(
    bucket: str,
    key: str,
    s3_client=None,
) -> Optional[Dict[str, Any]]:
    """
    Load market sector patterns from the S&P 500 tickers CSV in S3.

    Reads the CSV produced by the market-morning pipeline's
    tickers handler (same file as latest.json). Groups tickers
    by GICS Sector column to build sector patterns.

    Args:
        bucket: S3 bucket name (e.g., "market-data-bucket")
        key: S3 object key (e.g., "market/sp500/tickers/latest.json")
        s3_client: Optional pre-built S3 client (for testability)

    Returns:
        Dict matching MARKET_SECTOR_PATTERNS structure, or None on failure.
    """
    rows = _read_constituents(
        bucket, key, {SYMBOL_COLUMN, SECTOR_COLUMN}, s3_client
    )
    if rows is None:
        return None

    # Group tickers by GICS Sector
    sector_tickers: Dict[str, List[str]] = {}
    for row in rows:
        symbol = row.get(SYMBOL_COLUMN, "").strip()
        gics_sector = row.get(SECTOR_COLUMN, "").strip()

        if not symbol or not gics_sector:
            continue

        if gics_sector not in sector_tickers:
            sector_tickers[gics_sector] = []
        sector_tickers[gics_sector].append(symbol)

    if not sector_tickers:
        logger.warning(
            f"No valid (Symbol, GICS Sector) pairs in "
            f"s3://{bucket}/{key} — falling back to defaults"
        )
        return None

    # Build patterns from grouped tickers
    patterns: Dict[str, Any] = {}
    for gics_sector, tickers in sorted(sector_tickers.items()):
        sector_key = _gics_sector_to_key(gics_sector)
        pascal_name = _gics_sector_to_pascal(gics_sector)

        sector_uri = MARKET_ENRICHMENT[f"{pascal_name}Sector"]
        relationship = MARKET_ENRICHMENT[
            f"{pascal_name[0].lower()}{pascal_name[1:]}SectorCorrelation"
        ]

        patterns[sector_key] = {
            "description": f"{gics_sector} companies (from S&P 500)",
            "sector_uri": sector_uri,
            "tickers": sorted(set(tickers)),
            "relationship": relationship,
        }

    logger.info(
        f"Loaded {len(patterns)} market sector patterns from "
        f"s3://{bucket}/{key} — "
        f"{sum(len(p['tickers']) for p in patterns.values())} "
        f"total tickers across {len(patterns)} GICS sectors"
    )
    return patterns


def load_ticker_cik_map_from_s3(
    bucket: str,
    key: str,
    s3_client=None,
) -> Optional[Dict[str, str]]:
    """Ticker -> 10-digit padded CIK, from the constituents CSV already loaded.

    The CSV carries the regulator's company ID beside the symbol and this file
    used to discard it, which is why the market half of the company bridge was
    keyed on the TICKER: the only company identifier the market side appeared
    to have. Keying on the CIK instead is what lets a quote and a filing land
    on ONE unified company node rather than two, and it lifts the bridge's
    ceiling off ``hasIssuerTradingSymbol`` -- a term emitted on ownership forms
    only, a measured 24.57% of filings -- onto ``hasIssuerCik``, which every
    filing states.

    Reach is limited to index constituents for now. A universal
    company-to-ticker registry covering all listed issuers is planned upstream;
    it widens this map without changing the join key, so nothing here has to
    move when it lands.

    Returns None if the CSV is unusable, and an empty dict if it is readable
    but carries no usable pair -- the caller treats both as "no bridge", but
    only the first means the file was the problem.
    """
    rows = _read_constituents(
        bucket, key, {SYMBOL_COLUMN, CIK_COLUMN}, s3_client
    )
    if rows is None:
        return None

    ticker_cik: Dict[str, str] = {}
    for row in rows:
        symbol = row.get(SYMBOL_COLUMN, "").strip().upper()
        cik = padded_cik(row.get(CIK_COLUMN, ""))

        if not symbol or cik is None:
            continue

        ticker_cik[symbol] = cik

    logger.info(
        f"Loaded {len(ticker_cik)} ticker->CIK pairs from "
        f"s3://{bucket}/{key}"
    )
    return ticker_cik


def load_sub_industries_from_s3(
    bucket: str,
    key: str,
    s3_client=None,
) -> Optional[List[Tuple[str, str]]]:
    """(ticker, GICS sub-industry) pairs, for the constituent peer edges.

    The finer half of the classification the sector reader above throws away.
    Sector alone is too coarse to be a useful similarity signal -- the largest
    of the eleven holds 73 of 503 constituents, so "same sector" separates
    almost nothing. Sub-industry is roughly 160 buckets of 3-5 constituents,
    where "same sub-industry" is a sharp claim about two companies competing
    in one market.

    Returned as pairs rather than grouped, so the caller decides what to group
    on -- the peer step groups on the sub-industry after resolving each ticker
    to its company node, and a ticker with no company node in the graph drops
    out before the grouping rather than after.
    """
    rows = _read_constituents(
        bucket, key, {SYMBOL_COLUMN, SUB_INDUSTRY_COLUMN}, s3_client
    )
    if rows is None:
        return None

    pairs: List[Tuple[str, str]] = []
    for row in rows:
        symbol = row.get(SYMBOL_COLUMN, "").strip().upper()
        sub_industry = row.get(SUB_INDUSTRY_COLUMN, "").strip()

        if not symbol or not sub_industry:
            continue

        pairs.append((symbol, sub_industry))

    logger.info(
        f"Loaded {len(pairs)} ticker->sub-industry pairs from "
        f"s3://{bucket}/{key}"
    )
    return pairs


def get_sector_patterns(
    bucket: str = "",
    key: str = "",
    s3_client=None,
) -> Dict[str, Any]:
    """
    Get market sector patterns, attempting S3 first with fallback to defaults.

    Args:
        bucket: S3 bucket for tickers CSV (empty = skip S3)
        key: S3 key for tickers CSV (empty = skip S3)
        s3_client: Optional pre-built S3 client

    Returns:
        Dict matching MARKET_SECTOR_PATTERNS structure (always non-empty)
    """
    if bucket and key:
        s3_patterns = load_sector_patterns_from_s3(
            bucket=bucket, key=key, s3_client=s3_client
        )
        if s3_patterns is not None:
            return s3_patterns

    logger.info("Using default hardcoded market sector patterns")
    return DEFAULT_MARKET_SECTOR_PATTERNS


def get_ticker_cik_map(
    bucket: str = "",
    key: str = "",
    s3_client=None,
) -> Dict[str, str]:
    """Ticker -> padded CIK, or an empty map when the CSV is unavailable.

    NO hardcoded fallback, deliberately -- unlike get_sector_patterns above.
    A sector assignment that is a little stale is still roughly true; a CIK is
    a regulator's primary key, and a wrong one silently merges two unrelated
    companies into one node. There is no honest way to hardcode 500 of them,
    so the absence of the file is reported as an absence.

    An empty map does not mean no company bridge. The linker also derives
    ticker -> CIK from the filings themselves, where an issuer states both, and
    the SEC half of the bridge keys on the CIK directly and needs no map at
    all. This widens the market half to constituents whose filings were not
    ingested; it is not what makes the bridge exist.
    """
    if bucket and key:
        loaded = load_ticker_cik_map_from_s3(
            bucket=bucket, key=key, s3_client=s3_client
        )
        if loaded is not None:
            return loaded

    logger.info(
        "No constituents CSV available — the market half of the company "
        "bridge will key only on tickers stated by ingested filings"
    )
    return {}


def get_sub_industries(
    bucket: str = "",
    key: str = "",
    s3_client=None,
) -> List[Tuple[str, str]]:
    """(ticker, sub-industry) pairs, or empty when the CSV is unavailable.

    No hardcoded fallback, for the same reason as get_ticker_cik_map: the
    DEFAULT patterns above carry sector only, and inventing a sub-industry
    split of them would assert competitor relationships nobody checked.
    """
    if bucket and key:
        loaded = load_sub_industries_from_s3(
            bucket=bucket, key=key, s3_client=s3_client
        )
        if loaded is not None:
            return loaded

    logger.info(
        "No constituents CSV available — no sub-industry peer edges"
    )
    return []


# ============================================
# DEFAULT HARDCODED PATTERNS (fallback)
# ============================================

DEFAULT_MARKET_SECTOR_PATTERNS: Dict[str, Any] = {
    'information_technology_sector': {
        'description': 'Information Technology companies',
        'sector_uri': MARKET_ENRICHMENT.InformationTechnologySector,
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMZN',
                    'AVGO', 'ORCL', 'CRM', 'ADBE', 'AMD', 'INTC', 'QCOM'],
        'relationship': MARKET_ENRICHMENT.informationTechnologySectorCorrelation,
    },

    'financials_sector': {
        'description': 'Financial services companies',
        'sector_uri': MARKET_ENRICHMENT.FinancialsSector,
        'tickers': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW',
                    'AXP', 'V', 'MA', 'COF'],
        'relationship': MARKET_ENRICHMENT.financialsSectorCorrelation,
    },

    'health_care_sector': {
        'description': 'Health Care companies',
        'sector_uri': MARKET_ENRICHMENT.HealthCareSector,
        'tickers': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'MRK',
                    'LLY', 'BMY', 'AMGN', 'GILD'],
        'relationship': MARKET_ENRICHMENT.healthCareSectorCorrelation,
    },

    'energy_sector': {
        'description': 'Energy companies',
        'sector_uri': MARKET_ENRICHMENT.EnergySector,
        'tickers': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
                    'OXY', 'HAL', 'DVN', 'FANG'],
        'relationship': MARKET_ENRICHMENT.energySectorCorrelation,
    },

    'consumer_discretionary_sector': {
        'description': 'Consumer Discretionary companies',
        'sector_uri': MARKET_ENRICHMENT.ConsumerDiscretionarySector,
        'tickers': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
                    'BKNG', 'TJX', 'CMG', 'LULU'],
        'relationship': MARKET_ENRICHMENT.consumerDiscretionarySectorCorrelation,
    },

    'consumer_staples_sector': {
        'description': 'Consumer Staples companies',
        'sector_uri': MARKET_ENRICHMENT.ConsumerStaplesSector,
        'tickers': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'CL',
                    'MDLZ', 'KHC', 'GIS', 'SYY'],
        'relationship': MARKET_ENRICHMENT.consumerStaplesSectorCorrelation,
    },

    'industrials_sector': {
        'description': 'Industrial companies',
        'sector_uri': MARKET_ENRICHMENT.IndustrialsSector,
        'tickers': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT',
                    'DE', 'UNP', 'FDX', 'WM'],
        'relationship': MARKET_ENRICHMENT.industrialsSectorCorrelation,
    },

    'real_estate_sector': {
        'description': 'Real Estate companies',
        'sector_uri': MARKET_ENRICHMENT.RealEstateSector,
        'tickers': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL',
                    'DLR', 'AVB', 'EQR', 'VTR'],
        'relationship': MARKET_ENRICHMENT.realEstateSectorCorrelation,
    },

    'utilities_sector': {
        'description': 'Utility companies',
        'sector_uri': MARKET_ENRICHMENT.UtilitiesSector,
        'tickers': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL',
                    'WEC', 'ED', 'ES', 'AWK'],
        'relationship': MARKET_ENRICHMENT.utilitiesSectorCorrelation,
    },

    'materials_sector': {
        'description': 'Materials companies',
        'sector_uri': MARKET_ENRICHMENT.MaterialsSector,
        'tickers': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'ECL', 'DD', 'DOW',
                    'NUE', 'VMC', 'MLM', 'PPG'],
        'relationship': MARKET_ENRICHMENT.materialsSectorCorrelation,
    },

    'communication_services_sector': {
        'description': 'Communication Services companies',
        'sector_uri': MARKET_ENRICHMENT.CommunicationServicesSector,
        'tickers': ['META', 'GOOGL', 'GOOG', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T',
                    'TMUS', 'CHTR', 'EA', 'TTWO'],
        'relationship': MARKET_ENRICHMENT.communicationServicesSectorCorrelation,
    },
}

# Legacy alias for backward compatibility
MARKET_SECTOR_PATTERNS = DEFAULT_MARKET_SECTOR_PATTERNS


# ============================================
# Option strategy patterns (unchanged — not externalized)
# ============================================

MARKET_OPTION_STRATEGY_PATTERNS = {
    'straddle': {
        'description': 'Long straddle (call and put at same strike/expiration)',
        'pattern_uri': MARKET_ENRICHMENT.StraddlePattern,
        'relationship': MARKET_ENRICHMENT.straddleWith,
    },

    'call_spread': {
        'description': 'Vertical call spread (adjacent strikes, same expiration)',
        'pattern_uri': MARKET_ENRICHMENT.CallSpreadPattern,
        'relationship': MARKET_ENRICHMENT.callSpreadWith,
    },

    'put_spread': {
        'description': 'Vertical put spread (adjacent strikes, same expiration)',
        'pattern_uri': MARKET_ENRICHMENT.PutSpreadPattern,
        'relationship': MARKET_ENRICHMENT.putSpreadWith,
    },

    'strangle': {
        'description': 'Long strangle (OTM call and OTM put, same expiration)',
        'pattern_uri': MARKET_ENRICHMENT.StranglePattern,
        'relationship': MARKET_ENRICHMENT.strangleWith,
    },
}
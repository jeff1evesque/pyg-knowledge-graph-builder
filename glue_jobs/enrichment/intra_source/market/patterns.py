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
from typing import Dict, Any, List, Optional

import boto3
from botocore.exceptions import ClientError

from glue_jobs.utils.rdf_utils import MARKET_ENRICHMENT

logger = logging.getLogger(__name__)

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
# S3 Loader — reads the tickers CSV
# ============================================


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
    if not bucket or not key:
        logger.debug(
            "No S3 bucket/key provided for market sector definitions"
        )
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

        # Parse CSV
        reader = csv.DictReader(io.StringIO(body))

        if reader.fieldnames is None:
            logger.warning(
                f"No CSV headers in s3://{bucket}/{key} — "
                f"falling back to defaults"
            )
            return None

        # Validate required columns
        required_columns = {"Symbol", "GICS Sector"}
        available_columns = set(reader.fieldnames)
        missing = required_columns - available_columns
        if missing:
            logger.warning(
                f"CSV at s3://{bucket}/{key} missing columns: "
                f"{missing}. Available: {reader.fieldnames} — "
                f"falling back to defaults"
            )
            return None

        # Group tickers by GICS Sector
        sector_tickers: Dict[str, List[str]] = {}
        for row in reader:
            symbol = row.get("Symbol", "").strip()
            gics_sector = row.get("GICS Sector", "").strip()

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
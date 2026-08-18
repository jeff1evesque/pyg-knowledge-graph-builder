"""
OCC option-symbol parsing, as Spark column expressions.

The quote-snapshot API identifies an option by its OCC symbol and states
nothing else about the contract's identity:

    "A     260717C00065000"
     |     |     ||
     |     |     |+-- strike x 1000, 8 digits    (65.000)
     |     |     +--- C(all) or P(ut)
     |     +--------- expiration YYMMDD          (2026-07-17)
     +--------------- root, left-justified in a 6-character field

Everything a consumer needs about that contract is therefore IN the symbol,
and three separate call sites were reading it from predicates the quotes
vocabulary does not emit -- `underlyingSymbol` and `expirationDate`, which
belong to the FEEDS model. Each filter matched nothing, each join dropped every
row, and each step logged "no option snapshots found" and returned None on data
full of option snapshots.

Parsed here once rather than at each site, so the field widths are stated in
one place. Every function returns NULL when the symbol is not OCC-shaped, which
makes "this is not an option symbol" a value callers can filter on instead of a
silently wrong answer: an equity symbol run through occ_underlying() yields
NULL, not itself.

Deliberately NOT a UDF -- these are regexp_extract/concat over a string column,
so they stay on the JVM (and on the GPU under RAPIDS) rather than paying a
per-row Python round trip.
"""
from pyspark.sql import functions as F

# Six characters of root, then YYMMDD, then C/P, then the 8-digit strike.
# Anchored at both ends: a partial match is not an OCC symbol, and accepting
# one would silently truncate a longer identifier down to a plausible ticker.
OCC_OPTION_SYMBOL = r"^(.{6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$"

_ROOT, _YY, _MM, _DD, _RIGHT, _STRIKE = 1, 2, 3, 4, 5, 6


def _group(col: F.Column, index: int) -> F.Column:
    """One OCC field, or NULL where the symbol does not parse."""
    extracted = F.regexp_extract(col, OCC_OPTION_SYMBOL, index)
    return F.when(F.length(extracted) > 0, extracted)


def is_occ_option_symbol(col: F.Column) -> F.Column:
    """Whether the symbol is an OCC option identifier."""
    return col.rlike(OCC_OPTION_SYMBOL)


def occ_underlying(col: F.Column) -> F.Column:
    """The equity ticker the contract is written against, or NULL.

    The root is padded to six characters with spaces, so it is trimmed. Case is
    left alone: tickers are uppercase at the source and forcing it here would
    hide a source that stopped being.
    """
    return F.trim(_group(col, _ROOT))


def occ_expiration_date(col: F.Column) -> F.Column:
    """The expiration as an ISO date string, or NULL.

    OCC writes a two-digit year. It is read as 20YY, which is correct for every
    listed contract: OCC symbology postdates 2000 and no exchange lists options
    expiring beyond 2099.
    """
    return F.concat(
        F.lit("20"), _group(col, _YY),
        F.lit("-"), _group(col, _MM),
        F.lit("-"), _group(col, _DD),
    )


def occ_contract_type(col: F.Column) -> F.Column:
    """CALL or PUT from the right indicator, or NULL.

    Spelled out rather than left as C/P so it matches what
    market_linker._canonical_contract_type produces, which is what every
    comparison downstream is written against.
    """
    right = _group(col, _RIGHT)
    return F.when(right == "C", F.lit("CALL")).when(right == "P", F.lit("PUT"))


def occ_strike_price(col: F.Column) -> F.Column:
    """The strike as a double, or NULL. Encoded in thousandths."""
    return _group(col, _STRIKE).cast("double") / F.lit(1000.0)


def equity_symbol(col: F.Column) -> F.Column:
    """The equity ticker a market symbol refers to.

    An OCC option symbol resolves to its underlying; anything else is passed
    through trimmed and uppercased. Unlike occ_underlying() this never returns
    NULL for a well-formed input, because its callers are asking "which company
    is this quote about" of BOTH equity and option snapshots.

    Falling through to the original rather than dropping it means a vocabulary
    using some other option encoding fails to join, instead of being mangled
    into the wrong ticker.
    """
    root = occ_underlying(col)
    return F.upper(F.when(root.isNotNull(), root).otherwise(F.trim(col)))

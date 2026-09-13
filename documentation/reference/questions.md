# What the Graph Answers

Tables answer lookups; edges answer relationships. This page is the second
kind — the traversals the [query tables](tables.md) exist to make possible, and,
just as carefully, the ones they do not.

Every path below is measured from `graph_schema.json` on run
`20260910T214553Z`. Counts are one day's.

## Everything about one ticker

```
OptionSnapshot --refersToCompany 9,214,506--> UnifiedCompany (2,408 nodes)
filings_Issuer --refersToCompany     2,005--> UnifiedCompany
SECFiling      --hasIssuer           3,550--> filings_Issuer
SECFiling      --hasXbrlFact         7,185--> XbrlFact
UnifiedCompany --belongsToSector     1,505--> EconomicSector
OptionSnapshot --belongsToSector 9,214,506--> EquitySector
OptionSnapshot --hasUnderlyingEquity 9,214,506--> EquitySnapshot
```

One traversal reaches a ticker's option chain and greeks, its underlying equity,
its GICS sector, its BLS economic sector, its filings, the XBRL facts inside
them, and its insider transactions.

This is the capability that justifies the pipeline. Without it a consumer needs
four separate APIs **and a ticker↔CIK mapping it does not have** — that mapping
*is* the enrichment, and `refersToCompany` is where it lives.

## Peers

`UnifiedCompany --sharesSubIndustryWith 1,424--> UnifiedCompany` is a company
peer network: *which company in this sub-industry has option activity unlike its
peers?*

## Option structure, already computed

The pipeline derives spread and straddle pairings as edges —
`callSpreadWith` 4,551,393, `putSpreadWith` 4,551,393, `straddleWith` 4,658,496 —
plus `hasMoneyness` 9,317,030 into 3 `Moneyness` nodes.

*At-the-money straddles on energy names where implied vol sits above the sector*
is a filter over edges, not a computation at query time.

## Filings to market — cross-sectional only

The join works:

```
SECFiling --hasIssuer--> Issuer --refersToCompany--> UnifiedCompany
                                <--refersToCompany-- OptionSnapshot
```

Two measured limits bound it hard.

**Timing depends on when a filing was ingested.** Filings already in the graph
carry no clock time — `hasFilingDate` has exactly one distinct value for a whole
day. The upstream acceptance timestamp fixes that going forward. So "what did
options do after the filing landed" works on filings ingested after that change
and never on the ones already here.

**The joinable universe is 93 companies today.** Market references 496
companies, filings reference 2,005, and 93 appear in both. The upstream ticker
registry covers 8,020 issuers, which lifts this by more than an order of
magnitude without changing the join key.

What that leaves, answerable on a single day: *did the companies that filed
today show different option volume or implied vol than the optionable companies
that did not?* Today that compares 93 filers against 403 non-filers.

## Weather against regional economics

`WeatherAlert --affectsRegion--> Region <--hasRegion-- BLS measurement` is the
two-hop traversal the [triple store](tables.md#graph) exists for, and the one
`edges/` answers badly.

## What it does not answer

Stated plainly so nobody builds a product promise on it.

- **Intraday questions about filings already in the graph.** Nothing ingested so
  far carries a clock time, and none of it is being backfilled. This is a
  permanent split rather than a gap that closes.
- **Macro to market at company grain.** BLS reaches market through 570
  `leadsTo` edges and the shared calendar; NOAA reaches it through **no edge
  type at all**. The fix is upstream connective tissue, not storage.
- **Anything geographic about a ticker.** Companies carry no location, so
  regional weather cannot be tied to regionally-exposed companies.
- **Natural-language search.** Only 5 of 155 node types carry any text — 22,096
  nodes, 0.010% of triples.

## How far back each question reaches

| What answers it | Window | Why |
|---|---|---|
| SEC and market today | 1 day | one published run |
| BLS today | 8 months | BLS ships its own history in each response |
| a run's `.pt`, `graph_schema.json`, indices | 21 days | run-scoped expiration |
| **the query tables** | **365 days** | their own lifecycle, their own prefix |
| rebuilt from the raw archive | as deep as the archive | filings are rebuildable now; market depth is unconfirmed |

Multi-hop traversal through `graph/` is retained for a year but queryable one
day at a time — 30 days is 5.8 GB and a year is 71 GB, against 194 MB for a
single day.

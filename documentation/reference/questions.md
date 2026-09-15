# What the Graph Answers

Tables answer lookups; edges answer relationships. This page walks through the
questions the [query tables](tables.md) make possible: the edges each one uses,
what one published day actually holds, and how far back it reaches. It also
lists what the data cannot answer.

Every number here is measured on the published tables for **2026-09-09** (run
`20260913T150428Z`): edge counts from `edge_types/`, everything else from
`nodes/`, `edges/`, `facts/`, `snapshots/` and `graph/`. That run's `.pt` leaves
the weather types out; the tables keep them.

## Everything about one ticker

```
OptionSnapshot --refersToCompany     9,214,506--> UnifiedCompany (2,408 nodes)
OptionSnapshot --hasUnderlyingEquity 9,214,506--> EquitySnapshot
OptionSnapshot --belongsToSector     9,214,506--> EquitySector
UnifiedCompany --belongsToSector         1,508--> EconomicSector
filings_Issuer --refersToCompany         2,005--> UnifiedCompany
SECFiling      --hasIssuer               3,547--> filings_Issuer
SECFiling      --hasXbrlFact             7,185--> XbrlFact
SECFiling      --hasNonDerivativeTransaction 42--> NonDerivativeTransaction
```

From one ticker this reaches its option chain, its underlying equity, its GICS
sector, its BLS economic sector, its filings, the XBRL facts inside them and any
insider trades they report.

Market is not in `graph/`, so this is a chain of one-hop joins over `edges/` and
`nodes/` for the same day, with `snapshots/` for the quotes and `facts/` for the
filings' values. The ticker-to-company join is the part a consumer would
otherwise have to build, and `refersToCompany` is that join, already done.

## Insider disclosure against the peer group

*An officer of a chip maker reports a trade on a Form 4. Did the option chains
of its sub-industry peers move differently from its own?*

```
Form 4         --hasIssuer--> Issuer --refersToCompany--> UnifiedCompany
Form 4         --hasReportingOwner--> ReportingOwner
Form 4         --hasNonDerivativeTransaction--> NonDerivativeTransaction
UnifiedCompany --sharesSubIndustryWith--> UnifiedCompany
               <--refersToCompany-- OptionSnapshot
```

The trade itself is in `facts/`: `hasTransactionAcquiredDisposedCode` (A or D),
`hasTransactionShares`, `hasTransactionPricePerShare` and `hasTransactionDate`.

What the day holds:

- **The peer group is real and fully quoted.** NVDA reaches 13 peers through
  `sharesSubIndustryWith`, every one with option snapshots: ADI, AMD, AVGO,
  FSLR, INTC, MCHP, MPWR, MRVL, NXPI, ON, QCOM, SWKS and TXN. The grouping is
  GICS's and is taken as given, which is why First Solar is in it. Equipment
  makers are a separate sub-industry (AMAT, KLAC, LRCX, Q and TER), so they are
  not peers under this edge.
- **Each pair is stored once.** None of the 1,424 `sharesSubIndustryWith` edges
  has a reverse edge, so a query follows it in both directions.
- **The issuer resolves from the form itself.** 537 of the day's 553 ownership
  forms (Forms 3, 4 and 5 and their amendments) name an issuer with a trading
  symbol. On the 58 forms that carry a reporting owner, the owner's own CIK is
  also a `hasIssuer` target, so join through the issuer that has the symbol.
- **Trade details are the limit.** Only 58 of the 553 forms carry a reporting
  owner, and 44 filings carry 63 transactions between them. Of the 60 ownership
  forms about a quoted company, one carries a trade: a sale of 2,000 KEYS shares
  at $326.10, dated 2026-09-04.

**Window:** one published day. Filings can be rebuilt from the archive. Watching
the peers' chains over the days after a trade also needs those days of market
data, and how far back the market archive goes has not been measured.

## One director, several boards

*Did one director trade at two of their own companies in the same week?*

```
Form 4       --hasReportingOwner--> ReportingOwner      (hasReportingOwnerCik)
Form 4       --hasIssuer--> Issuer --refersToCompany--> UnifiedCompany
Person_{CIK} --owl:sameAs--> ReportingOwner
```

This one needs no market data, so it is not limited to the 93 companies both
sides share. It is also a question no single-source API answers, because it
needs one person's identity across filings. The SEC linker
(`enrichment/intra_source/sec_linker.py`) mints `Person_{CIK}` when the same CIK
appears on more than one reporting-owner node.

What the day holds: 54 reporting owners, each on one company's filings, so no
multi-board case and no `Person_{CIK}` node. Owner coverage bounds it too: 58 of
553 ownership forms carry an owner at all.

**Window:** "the same week" needs several days of filings. The published tables
hold one; a rebuild from the archive can supply more.

## The insider price against the option surface

*Did the insider sell into strikes the market had already crowded?*

```
NonDerivativeTransaction   hasTransactionPricePerShare, hasTransactionDate   (facts/)
Form 4 --hasIssuer--> Issuer --refersToCompany--> UnifiedCompany
                              <--refersToCompany-- OptionSnapshot
OptionSnapshot             strikePrice, mark, openInterest, volatility       (snapshots/)
OptionSnapshot --hasMoneyness--> AtTheMoney | InTheMoney | OutOfTheMoney     (edges/)
```

What the day holds: one priced trade on a quoted company, the KEYS sale above.

**The two sides are not the same day.** A Form 4 reports a trade already made.
The day's 63 transactions are dated 2026-08-27 to 2026-09-08 and none is dated
2026-09-09, so every trade predates the snapshots published with it. Against the
same day's chain, the KEYS sale is compared with the market five days later.

**Window:** the chain has to come from the trade's own date, so this needs the
market day of each trade, not only the day the form was filed.

## Weather against regional economics

*Which BLS series cover the states a severe weather alert hit?*

```
WeatherAlert --affectsRegion--> GeographicRegion <--hasRegion-- BLS measurement
WeatherAlert --hasInfo--> Info --hasSeverity--> Minor | Moderate | Severe | Extreme
```

What the day holds: 902 alerts. 361 of them reach 43 of the 50 state regions
through 454 `affectsRegion` edges, and those states carry 3,620 BLS
measurements, mostly state labor-force and employment series. Severity splits
into Minor 527, Moderate 254, Severe 106, Extreme 3 and unknown 12. The 92
severe or extreme alerts that reach a state cover 2,203 measurements.

**Severity is in `graph/` only.** It points at a URI no source types, so it is
neither an edge nor a fact and appears in no table. The filtered question is a
query against the store:

```python
import pyoxigraph

store = pyoxigraph.Store.read_only("graph/day=2026-09-09")
for row in store.query("""
    PREFIX cap: <https://jefflevesque.com/ontology/cap-model/>
    PREFIX bls: <https://jefflevesque.com/ontology/bls/>
    SELECT DISTINCT ?measurement WHERE {
      ?alert cap:hasInfo ?info .
      ?info  cap:hasSeverity ?severity .
      FILTER(?severity IN (cap:Severe, cap:Extreme))
      ?alert       bls:affectsRegion ?region .
      ?measurement bls:hasRegion     ?region
    }
"""):
    print(row["measurement"].value)
```

Counted, that query returns 2,203 measurements in 8 ms. Without the severity
filter the same two hops give 38,352 pairs, in 10 ms from `graph/` and in
0.05 s from a local copy of `edges/`.

The tables answer this and the published `.pt` cannot, because the run left the
weather types out of the model.

**Window:** one day. Alerts share no nodes between days, so weather over time
needs accumulated days. Each BLS series carries its own history.

## Filings against the market clock

```
SECFiling --hasIssuer--> Issuer --refersToCompany--> UnifiedCompany
                                <--refersToCompany-- OptionSnapshot
```

**Both sides carry a clock time.** 2,402 of the day's 2,403 filings carry
`hasAcceptanceDateTime`, to the second, in `facts/`. Snapshots carry
`captureTime`: 19 a day, every 20 minutes from 9:40 AM to 3:40 PM ET, with
`quoteTime` and `tradeTime` in epoch milliseconds, in `snapshots/`.

`hasFilingDate` is not the clock. It has one value for the whole day, and EDGAR
dates a late filing to the next business day, so the filings dated 2026-09-09
were accepted between 5:31 PM ET on 2026-09-08 and 9:55 PM ET on 2026-09-09.

Upstream backfilled the acceptance time into its stored filing history on
2026-09-12, so a rebuilt day carries it too. The run published on 2026-09-10
read this same day before the backfill and found no clock time on any filing.
The instant is kept out of the `.pt` feature vector on purpose; it is published
for ordering, not encoded.

When the day's filings were accepted (ET):

| window | all filings | filings on quoted companies |
|---|---:|---:|
| before the 9:30 open, including the evening before | 534 | 54 |
| 9:30 to the first snapshot | 19 | 9 |
| between the first and last snapshot | 739 | 237 |
| last snapshot to the 4:00 close | 65 | 14 |
| 4:00 to 5:30 PM | 805 | 144 |
| after 5:30 PM | 240 | 29 |

What that makes askable on one day:

- *What did the option chain do in the snapshot after a filing, against the
  snapshot before it?* 237 filings from 23 quoted companies sit between two
  snapshots.
- *Did a filing land before or after the close?* 1,045 of the day's filings
  were accepted after 4:00 PM.
- *Which filings land while EDGAR is open and the market is not?* A filing
  outside the snapshot window lacks a snapshot before or after it that day, so
  it has to be left out rather than matched to the nearest one.

What the clock does not buy:

- **Anything finer than the snapshots.** They are 20 minutes apart, so two
  filings inside one interval cannot be told apart.
- **Cause.** "The chain moved in the next snapshot" fits the filing moving it
  and both reacting to something else equally well.
- **BLS.** BLS carries no clock time, so this is a filings-and-market question.

**The overlap is 93 companies, and that is a ceiling.** Market quotes 499 equity
symbols, which resolve to 496 companies (FOX and FOXA, GOOG and GOOGL, NWS and
NWSA each share one). The filings name 2,005 `UnifiedCompany` nodes, insiders
named as issuers on ownership forms among them, and 93 are also quoted. Every
quoted symbol already resolves, so no ticker map raises the overlap; only the
list of quoted symbols bounds it.

The cross-sectional form needs no clock: *did the 93 companies that filed show
different option volume or implied vol than the 403 quoted companies that did
not?*

## Option structure, already computed

The pipeline derives spread and straddle pairings as edges — `callSpreadWith`
4,551,393, `putSpreadWith` 4,551,393, `straddleWith` 4,658,496 — plus
`hasMoneyness` 9,317,030 into 3 `Moneyness` nodes.

*At-the-money straddles on energy names where implied vol sits above the
sector:* the pairing, the moneyness bucket and the sector are edges. Comparing
implied vol with the sector's is still an aggregate over `snapshots/`
(`volatility`) at query time.

## Sectors where the three sources meet

Company joins stop at 93. Sector joins do not, because all three sources reach
the BLS `EconomicSector` nodes, each by its own route:

| source | route | sectors reached |
|---|---|---:|
| BLS | series `--belongsToSector-->` `EconomicSector` | 25 |
| filings | `UnifiedCompany --belongsToSector-->` `EconomicSector`, from the SIC code | 7 |
| market | `EquitySector --relatedToEconomicSector-->` `EconomicSector` | 15 |

All three reach six: ConstructionTrades, Financial, Manufacturing,
NaturalResources, Retail and Transportation. The market route is a similarity
link rather than membership, so a rollup that unions the routes mixes two kinds
of claim. A three-source question resolves at BLS's monthly grain.

## What it does not answer

Stated plainly so nobody builds a product promise on it.

- **Macro or weather data against one company's price.** BLS reaches market
  through 570 `leadsTo` edges into equity sectors, and through the calendar.
  NOAA has no edge to market at all and shares only the calendar day. Sector
  grain works (above), but a sector-level association is not a company-level
  one.
- **Where a company's assets are.** Filings carry `hasBusinessState` (2,221 of
  2,403), `hasBusinessLocation` (1,872) and `hasEntityIncorporationState`
  (1,753) in `facts/`, and no enrichment step reads them yet. They are stated
  per filing, and ownership forms name the insider as well as the company, so a
  headquarters should come from a company's own filings; that gives 38 of the
  496 quoted companies on this day. No source here carries facility locations,
  so regional exposure can be approximated by headquarters but not measured.
- **Search over prose.** `entities/` holds text for 12,607 nodes across 49 node
  types, mostly names and labels. Of the 990 nodes with 300 or more characters,
  901 are weather alert descriptions.

## How far back each question reaches

| What answers it | Window | Why |
|---|---|---|
| SEC and market | 1 day | one published day of tables |
| weather | 1 day | alerts share no nodes between days |
| BLS | up to 8 months | each response carries its own history; most series run from January 2026 to July or August |
| a run's `.pt`, `graph_schema.json`, node index | 21 days | run retention |
| the query tables and `graph/` | 365 days | their own retention, under their own prefix |
| rebuilt from the raw archive | as deep as the archive | filings carry the backfilled acceptance time; market and weather depth are not measured |

`graph/` is still queried one day at a time. The 2026-09-09 store is 185 MB
(1,413,546 triples), so a 30-day window would be about 5.6 GB and a year about
68 GB if days stay that size.

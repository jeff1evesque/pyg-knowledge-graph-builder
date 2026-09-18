# Questions the Tables Answer

Each question below can be answered from the [query tables](tables.md). For
each one, this page gives the tables it reads, the path it follows from one kind
of node to the next, what the tables held on one real day, and how many days of
tables it needs. Every number is measured on the tables published for
**2026-09-09**, built by run `20260913T150428Z`: edge counts from
`edge_types/`, everything else from `nodes/`, `edges/`, `facts/`, `snapshots/`
and `graph/`.

None of these questions uses the `.pt` file.

## Before you start

### The tables, not the `.pt`

A run publishes two different things. This page uses only the first.

| | Query tables | PyG graph (`.pt`) |
|---|---|---|
| What it is | Six Parquet tables and a triple store, `graph/`: the nodes, the edges between them, and their values | One PyTorch Geometric `HeteroData` file: a matrix of numbers per node type, and each edge as a pair of row numbers |
| What it is for | Looking things up and answering questions: SQL, SPARQL, or retrieval for an LLM | Training and running a graph neural network |
| Names, text and dates | Yes. Every row carries a URI, and `facts/`, `entities/` and `snapshots/` carry the values | No. A row is a position, and `node_index/` says which entity it is |
| Where | `<PYG_TABLES_ROOT>/<dataset>/<table>/day=YYYY-MM-DD/` | In the run's folder, as `<variant>/hetero_data_<variant>.pt` |
| Kept for | 365 days | 21 days, with its run |
| Covers | Every source the run read: BLS, SEC, market and NOAA weather | Every source except NOAA weather — each build names the sources it holds in `graph_schema.json`'s `sources_in_graph` |

Weather is left out of the `.pt` on purpose and kept in the tables. NOAA is
0.05% of the nodes on 2026-09-09, shares nothing between days and has no edge to
market, so it earns little in a graph neural network, and it still answers
weather questions from a table. Which node types a `.pt` leaves out is set per
run (`exclude_node_types` in its PyG config), and the tables ignore that
setting. A build says which sources it ended up with in
[`graph_schema.json`](outputs.md#what-the-graph-is-made-of)'s `sources_in_graph`
— read that rather than `sources` beside it, which is what the run read, weather
included. [Query tables](tables.md) describes the tables, and
[Metadata files](outputs.md) describes the `.pt`.

### How far back you can ask

**A day's tables are kept for 365 days.** Nothing expires after one day. Each
run publishes the tables for the day its data was cut from, under
`day=YYYY-MM-DD`, and a [scheduled run](../operations/running-a-job.md#scheduled-runs)
publishes one day every day, so the history you can query grows by a day each
day, up to a year.

What one day's tables hold depends on the source:

| Source | One day's tables hold |
|---|---|
| SEC filings | the filings dated that day |
| Market | that day's 19 snapshots, every 20 minutes from 9:40 AM to 3:40 PM ET |
| NOAA | that day's weather alerts |
| BLS | every series with its own history; most run from January 2026 to July or August |

So a question about one day reads one `day=` partition, and a question about a
week reads seven. Across days, follow an entity by its URI: node ids are
renumbered every day, so they only join within a day. `graph/` is one store per
day, so a multi-hop question over several days opens several stores. Each
question below ends with the days it needs.

Days before the first published day can be built by running the pipeline for
them, as far back as each source's raw archive goes. Filings built that way
carry acceptance times; how far back the market and weather archives go has not
been measured.

### Asking with an LLM

The tables can be the retrieval side of retrieval-augmented generation (RAG)
with any LLM. Nothing in this repository calls an LLM, builds embeddings or
serves the tables. Your code queries the tables, writes the rows it gets back as
text, and gives that text and the question to the LLM you choose. Each
question on this page is the query half of that.
[Using the tables with an LLM](llm.md) works one through end to end.

The `.pt` is no use to an LLM: it holds numbers without names.

### Reading a path

Each question's path is a table of edge types, and each row is stored in
`edges/` exactly as written: **From** is `src_type`, **Edge** is `relation` and
**To** is `dst_type`. Every edge points from From to To. A path can still walk
an edge backwards. To go from a company to its option snapshots, take the
`OptionSnapshot`, `refersToCompany`, `UnifiedCompany` row and join on its To
side.

Names on this page are short. In the tables, each one carries the prefix of the
vocabulary it comes from:

| Prefix | Short names on this page |
|---|---|
| `market_quotes_` | `OptionSnapshot`, `EquitySnapshot`, and the `snapshots/` columns `strikePrice`, `mark`, `openInterest`, `volatility`, `captureTime`, `quoteTime` and `tradeTime` |
| `market_enrichment_` | `EquitySector`, `Moneyness`, `hasUnderlyingEquity`, `hasMoneyness`, `callSpreadWith`, `putSpreadWith`, `straddleWith`, `sharesSubIndustryWith`, `relatedToEconomicSector`, and `belongsToSector` from a snapshot |
| `bls_enrichment_` | `EconomicSector`, `GeographicRegion`, `refersToCompany`, `affectsRegion`, `hasRegion`, `leadsTo`, and `belongsToSector` from a company or a BLS series |
| `sec_enrichment_` | `UnifiedCompany`, `UnifiedPerson` |
| `filings_` | `SECFiling`, `Issuer`, `ReportingOwner`, `NonDerivativeTransaction`, `XbrlFact`, `hasIssuer`, `hasReportingOwner`, `hasNonDerivativeTransaction`, `hasXbrlFact`, and the `facts/` values `hasTransaction...`, `hasAcceptanceDateTime`, `hasFilingDate`, `hasReportingOwnerCik` and `hasBusiness...` |
| `weather_` | `WeatherAlert` |
| `cap_` | `Info`, `hasInfo` |
| `owl_` | `sameAs` |

So `OptionSnapshot` is `market_quotes_OptionSnapshot` in `nodes/` and `edges/`,
and `strikePrice` is the column `market_quotes_strikePrice`. A day's
`edge_types/` lists the full name of every edge type.

## The questions

### Everything about one ticker

*What do the tables hold about one company, starting from its ticker?*

Reads `edges/` and `nodes/`, with `snapshots/` for the quotes and `facts/` for
the filings' values.

| From | Edge | To | Edges on 2026-09-09 |
|---|---|---|---:|
| `OptionSnapshot` | `refersToCompany` | `UnifiedCompany` | 9,214,506 |
| `OptionSnapshot` | `hasUnderlyingEquity` | `EquitySnapshot` | 9,214,506 |
| `OptionSnapshot` | `belongsToSector` | `EquitySector` | 9,214,506 |
| `UnifiedCompany` | `belongsToSector` | `EconomicSector` | 1,508 |
| `Issuer` | `refersToCompany` | `UnifiedCompany` | 2,005 |
| `SECFiling` | `hasIssuer` | `Issuer` | 3,547 |
| `SECFiling` | `hasXbrlFact` | `XbrlFact` | 7,185 |
| `SECFiling` | `hasNonDerivativeTransaction` | `NonDerivativeTransaction` | 42 |

From one ticker this reaches the company's option chain, its underlying equity,
its GICS sector, its BLS economic sector, its filings, the XBRL facts inside
them and any insider trades they report. The day held 2,408 `UnifiedCompany`
nodes.

The ticker-to-company join is the part a consumer would otherwise have to
build, and `refersToCompany` is that join, already done. Market is not in
`graph/`, so this is a chain of one-hop joins over the same day's `edges/` and
`nodes/`.

**Days needed:** one.

### Insider disclosure against the peer group

*An officer of a chip maker reports a trade on a Form 4. Did the option chains
of its sub-industry peers move differently from its own?*

Reads `edges/`, `nodes/`, `facts/` and `snapshots/`.

| From | Edge | To |
|---|---|---|
| `SECFiling` (the Form 4) | `hasIssuer` | `Issuer` |
| `Issuer` | `refersToCompany` | `UnifiedCompany` |
| `SECFiling` (the Form 4) | `hasReportingOwner` | `ReportingOwner` |
| `SECFiling` (the Form 4) | `hasNonDerivativeTransaction` | `NonDerivativeTransaction` |
| `UnifiedCompany` | `sharesSubIndustryWith` | `UnifiedCompany` |
| `OptionSnapshot` | `refersToCompany` | `UnifiedCompany` |

The form names its issuer, the issuer is a company, the company has peers, and
every company's option snapshots point at it. The trade itself is in `facts/`,
on the `NonDerivativeTransaction`: `hasTransactionAcquiredDisposedCode` (A or
D), `hasTransactionShares`, `hasTransactionPricePerShare` and
`hasTransactionDate`.

What 2026-09-09 held:

- **The peer group is real and fully quoted.** NVDA reaches 13 peers through
  `sharesSubIndustryWith`, every one with option snapshots: ADI, AMD, AVGO,
  FSLR, INTC, MCHP, MPWR, MRVL, NXPI, ON, QCOM, SWKS and TXN. The grouping is
  GICS's and is taken as given, which is why First Solar is in it. Equipment
  makers are a separate sub-industry (AMAT, KLAC, LRCX, Q and TER), so they are
  not peers under this edge.
- **Each pair is stored once.** None of the 1,424 `sharesSubIndustryWith` edges
  has a reverse edge, so a company's peers are on both sides of it: look for
  the company as From and as To.
- **The issuer resolves from the form itself.** 537 of the day's 553 ownership
  forms (Forms 3, 4 and 5 and their amendments) name an issuer with a trading
  symbol. On the 58 forms that carry a reporting owner, the owner's own CIK is
  also a `hasIssuer` target, so join through the issuer that has the symbol.
- **Trade details are the limit.** Only 58 of the 553 forms carry a reporting
  owner, and 44 filings carry 63 transactions between them. Of the 60 ownership
  forms about a quoted company, one carries a trade: a sale of 2,000 KEYS shares
  at $326.10, dated 2026-09-04.

**Days needed:** one, for the question as asked. Following the peers' chains
over the days after the trade needs those days' tables too.

### One director, several boards

*Did one director trade at two of their own companies in the same week?*

Reads `edges/`, `nodes/` and `facts/`.

| From | Edge | To |
|---|---|---|
| `SECFiling` (a Form 4) | `hasReportingOwner` | `ReportingOwner` |
| `SECFiling` (a Form 4) | `hasIssuer` | `Issuer` |
| `Issuer` | `refersToCompany` | `UnifiedCompany` |
| `UnifiedPerson` | `sameAs` | `ReportingOwner` |

The owner's CIK is `hasReportingOwnerCik`, in `facts/`. A `UnifiedPerson` is one
person, with the URI `Person_{CIK}`: the SEC linker
(`enrichment/intra_source/sec_linker.py`) mints one when the same CIK appears on
more than one reporting-owner node, with a `sameAs` edge to each of them.

This one needs no market data, so it is not limited to the 93 companies both
sides share. It is also a question no single-source API answers, because it
needs one person's identity across filings.

What 2026-09-09 held: 54 reporting owners, each on one company's filings, so no
multi-board case and no `Person_{CIK}` node. Owner coverage bounds it too: 58 of
553 ownership forms carry an owner at all.

**Days needed:** seven, for "the same week", since each day's tables hold that
day's filings.

### The insider price against the option surface

*Did the insider sell into strikes the market had already crowded?*

Reads `edges/`, `nodes/`, `facts/` and `snapshots/`.

| From | Edge | To |
|---|---|---|
| `SECFiling` (a Form 4) | `hasNonDerivativeTransaction` | `NonDerivativeTransaction` |
| `SECFiling` (a Form 4) | `hasIssuer` | `Issuer` |
| `Issuer` | `refersToCompany` | `UnifiedCompany` |
| `OptionSnapshot` | `refersToCompany` | `UnifiedCompany` |
| `OptionSnapshot` | `hasMoneyness` | `Moneyness`: `AtTheMoney`, `InTheMoney` or `OutOfTheMoney` |

The trade's `hasTransactionPricePerShare` and `hasTransactionDate` are in
`facts/`. Each snapshot's `strikePrice`, `mark`, `openInterest` and `volatility`
are columns of `snapshots/`.

What 2026-09-09 held: one priced trade on a quoted company, the KEYS sale above.

**The two sides are not the same day.** A Form 4 reports a trade already made.
The day's 63 transactions are dated 2026-08-27 to 2026-09-08 and none is dated
2026-09-09, so every trade predates the snapshots published with it. Against the
same day's chain, the KEYS sale is compared with the market five days later.

**Days needed:** the day the form was filed, for the trade, and the day of the
trade, for the option chain it should be compared with.

### Weather against regional economics

*Which BLS series cover the states a severe weather alert hit?*

Reads `graph/`, because the severity is only there. Without the severity, the
same path runs over `edges/` and `nodes/`.

| From | Edge | To |
|---|---|---|
| `WeatherAlert` | `affectsRegion` | `GeographicRegion` |
| a BLS series, such as a state's labor force | `hasRegion` | `GeographicRegion` |
| `WeatherAlert` | `hasInfo` | `Info` |

The severity is one more hop, from the `Info` through `hasSeverity` to `Minor`,
`Moderate`, `Severe` or `Extreme`, and it is in `graph/` only.

What 2026-09-09 held: 902 alerts. 361 of them reach 43 of the 50 state regions
through 454 `affectsRegion` edges, and those states carry 3,620 BLS
measurements, mostly state labor-force and employment series. Severity splits
into Minor 527, Moderate 254, Severe 106, Extreme 3 and unknown 12. The 92
severe or extreme alerts that reach a state cover 2,203 measurements.

**Severity is in `graph/` only.** It points at a URI no source types, so it is
neither an edge nor a fact and appears in no table. The filtered question is a
query against the store. `graph/` is a directory rather than Parquet, so copy
the day to local disk first:

```bash
aws s3 sync <PYG_TABLES_ROOT>/<dataset>/graph/day=2026-09-09/ graph/day=2026-09-09/
```

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

**Days needed:** one. Alerts share no nodes between days, so weather over time
needs one day's tables per day. Each BLS series carries its own history.

### Filings against the market clock

*Where does each filing fall against the day's option snapshots?*

Reads `edges/`, `nodes/`, `facts/` and `snapshots/`.

| From | Edge | To |
|---|---|---|
| `SECFiling` | `hasIssuer` | `Issuer` |
| `Issuer` | `refersToCompany` | `UnifiedCompany` |
| `OptionSnapshot` | `refersToCompany` | `UnifiedCompany` |

**Both sides carry a clock time.** 2,402 of the day's 2,403 filings carry
`hasAcceptanceDateTime`, to the second, in `facts/`. Snapshots carry
`captureTime`: 19 a day, every 20 minutes from 9:40 AM to 3:40 PM ET, with
`quoteTime` and `tradeTime` in epoch milliseconds, in `snapshots/`.

`hasFilingDate` is not the clock. It has one value for the whole day, and EDGAR
dates a late filing to the next business day, so the filings dated 2026-09-09
were accepted between 5:31 PM ET on 2026-09-08 and 9:55 PM ET on 2026-09-09.

Acceptance times are not limited to new days. Upstream added them to its whole
stored filing history on 2026-09-12, so any day built after that carries them,
however old the day. Run `20260910T214553Z` read this same day before then and
found no clock time on any filing.

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

**Days needed:** one.

### Option structure, already computed

*Which at-the-money straddles on energy names have implied vol above their
sector's?*

Reads `edges/`, `nodes/` and `snapshots/`.

| From | Edge | To | Edges on 2026-09-09 |
|---|---|---|---:|
| `OptionSnapshot` | `callSpreadWith` | `OptionSnapshot` | 4,551,393 |
| `OptionSnapshot` | `putSpreadWith` | `OptionSnapshot` | 4,551,393 |
| `OptionSnapshot` | `straddleWith` | `OptionSnapshot` | 4,658,496 |
| `OptionSnapshot` | `hasMoneyness` | `Moneyness` | 9,317,030 |
| `OptionSnapshot` | `belongsToSector` | `EquitySector` | 9,214,506 |

The pipeline derives the spread and straddle pairings as edges, and
`hasMoneyness` points into 3 `Moneyness` nodes. The pairing, the moneyness
bucket and the sector are all edges. Comparing implied vol with the sector's is
still an aggregate over `snapshots/` (`volatility`) at query time.

**Days needed:** one.

### Sectors where the three sources meet

*Which economic sectors can one question about BLS, filings and market data
cover?*

Reads `edges/` and `nodes/`.

Company joins stop at 93. Sector joins do not, because all three sources reach
the BLS `EconomicSector` nodes, each by its own route:

| Source | From | Edge | To | Sectors reached |
|---|---|---|---|---:|
| BLS | a BLS series | `belongsToSector` | `EconomicSector` | 25 |
| filings | `UnifiedCompany`, from its SIC code | `belongsToSector` | `EconomicSector` | 7 |
| market | `EquitySector` | `relatedToEconomicSector` | `EconomicSector` | 15 |

A snapshot reaches its `EquitySector` through `belongsToSector` first.

All three reach six: ConstructionTrades, Financial, Manufacturing,
NaturalResources, Retail and Transportation. The market route is a similarity
link rather than membership, so a rollup that unions the routes mixes two kinds
of claim.

**Days needed:** one, for one month. BLS is monthly, so a three-source question
resolves at the month, and a year of days gives about twelve points.

## What the tables do not answer

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
  901 are weather alert descriptions. An embedding index over it finds a node by
  name, not an answer to a question.

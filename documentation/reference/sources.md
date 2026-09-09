# Data Sources

The pipeline ingests RDF from four sources: BLS economic series, SEC filings,
intraday market snapshots, and NOAA weather alerts. One mapper is written per
published table, and the chart below is all one hundred of them — hover a bar
for the category behind it.

<div>
<svg viewBox="0 0 760 300" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-labelledby="srcTitle srcDesc"
     style="width:100%;height:auto;display:block;margin:1.2rem auto">
  <title id="srcTitle">Mappers per source</title>
  <desc id="srcDesc">One hundred mappers across four sources. BLS is ten
    categories holding 97 published tables between them: EMPSIT 27, JOLTS 15,
    ECI 14, XIMPIM 11, CPI 8, PPI 7, WKYENG 6, METRO 4, LAUS 3, REALER 2. SEC
    filings, market quotes and NOAA alerts contribute one mapper each.</desc>

  <style>
    .cap  { font: 600 11px var(--md-text-font-family, system-ui, sans-serif);
            fill: var(--md-default-fg-color--light, #5a5a5a);
            letter-spacing: .07em; }
    .lbl  { font: 600 11px var(--md-text-font-family, system-ui, sans-serif);
            fill: var(--md-default-fg-color, #1a1a1a); }
    .sub  { font: 400 10.5px var(--md-text-font-family, system-ui, sans-serif);
            fill: var(--md-default-fg-color--light, #5a5a5a); }
    .val  { font: 600 10.5px var(--md-text-font-family, system-ui, sans-serif);
            fill: var(--md-default-fg-color, #1a1a1a); text-anchor: middle; }
    .tick { font: 600 10px var(--md-text-font-family, system-ui, sans-serif);
            fill: var(--md-default-fg-color--light, #5a5a5a);
            text-anchor: middle; }
    .band { fill: none; stroke: var(--md-default-fg-color--lightest, #d8d8dc);
            stroke-dasharray: 4 4; }
    .axis { stroke: var(--md-default-fg-color--lightest, #d8d8dc); }
    .bar  { fill: var(--md-primary-fg-color--light, #5d6cc0);
            transition: fill 120ms; }
    .hit  { fill: transparent; pointer-events: all; }
    .lead { stroke: var(--md-accent-fg-color, #526cfe); stroke-dasharray: 3 3;
            opacity: 0; transition: opacity 120ms; }
    .tip  { opacity: 0; pointer-events: none; transition: opacity 120ms; }
    .tipbox { fill: var(--md-default-bg-color, #fff);
              stroke: var(--md-accent-fg-color, #526cfe); }
    .col:hover .bar  { fill: var(--md-accent-fg-color, #526cfe); }
    .col:hover .lead { opacity: .6; }
    .col:hover .tip  { opacity: 1; }
  </style>

  <text class="cap" x="12" y="22">MAPPERS PER SOURCE &#183; 100 TOTAL</text>

  <rect class="band" x="12" y="82" width="560" height="210" rx="8"/>
  <rect class="band" x="576" y="82" width="172" height="210" rx="8"/>
  <text class="cap" x="24" y="284">BLS &#183; 10 CATEGORIES &#183; 97 TABLES</text>
  <text class="cap" x="588" y="284">ONE MAPPER EACH</text>
  <line class="axis" x1="12" y1="248" x2="748" y2="248"/>

  <g class="col">
    <rect class="hit" x="16" y="80" width="56" height="190"/>
    <line class="lead" x1="44" y1="74" x2="44" y2="204"/>
    <rect class="bar" x="27" y="204" width="34" height="44" rx="2"/>
    <text class="val" x="44" y="198">8</text>
    <text class="tick" x="44" y="264">CPI</text>
    <g class="tip">
      <rect class="tipbox" x="8" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="20" y="51">Consumer Price Index</text>
      <text class="sub" x="20" y="66">8 tables &#183; 8 mappers</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="72" y="80" width="56" height="190"/>
    <line class="lead" x1="100" y1="74" x2="100" y2="209.5"/>
    <rect class="bar" x="83" y="209.5" width="34" height="38.5" rx="2"/>
    <text class="val" x="100" y="203.5">7</text>
    <text class="tick" x="100" y="264">PPI</text>
    <g class="tip">
      <rect class="tipbox" x="8" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="20" y="51">Producer Price Index</text>
      <text class="sub" x="20" y="66">7 tables &#183; 7 mappers</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="128" y="80" width="56" height="190"/>
    <line class="lead" x1="156" y1="74" x2="156" y2="171"/>
    <rect class="bar" x="139" y="171" width="34" height="77" rx="2"/>
    <text class="val" x="156" y="165">14</text>
    <text class="tick" x="156" y="264">ECI</text>
    <g class="tip">
      <rect class="tipbox" x="24" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="36" y="51">Employment Cost Index</text>
      <text class="sub" x="36" y="66">14 tables &#183; 14 mappers</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="184" y="80" width="56" height="190"/>
    <line class="lead" x1="212" y1="74" x2="212" y2="99.5"/>
    <rect class="bar" x="195" y="99.5" width="34" height="148.5" rx="2"/>
    <text class="val" x="212" y="93.5">27</text>
    <text class="tick" x="212" y="264">EMPSIT</text>
    <g class="tip">
      <rect class="tipbox" x="80" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="92" y="51">Employment Situation</text>
      <text class="sub" x="92" y="66">27 tables &#183; 27 mappers</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="240" y="80" width="56" height="190"/>
    <line class="lead" x1="268" y1="74" x2="268" y2="165.5"/>
    <rect class="bar" x="251" y="165.5" width="34" height="82.5" rx="2"/>
    <text class="val" x="268" y="159.5">15</text>
    <text class="tick" x="268" y="264">JOLTS</text>
    <g class="tip">
      <rect class="tipbox" x="136" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="148" y="51">Job Openings and Labor Turnover</text>
      <text class="sub" x="148" y="66">15 tables &#183; 15 mappers</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="296" y="80" width="56" height="190"/>
    <line class="lead" x1="324" y1="74" x2="324" y2="231.5"/>
    <rect class="bar" x="307" y="231.5" width="34" height="16.5" rx="2"/>
    <text class="val" x="324" y="225.5">3</text>
    <text class="tick" x="324" y="264">LAUS</text>
    <g class="tip">
      <rect class="tipbox" x="192" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="204" y="51">Local Area Unemployment Statistics</text>
      <text class="sub" x="204" y="66">3 tables &#183; 3 mappers</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="352" y="80" width="56" height="190"/>
    <line class="lead" x1="380" y1="74" x2="380" y2="226"/>
    <rect class="bar" x="363" y="226" width="34" height="22" rx="2"/>
    <text class="val" x="380" y="220">4</text>
    <text class="tick" x="380" y="264">METRO</text>
    <g class="tip">
      <rect class="tipbox" x="248" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="260" y="51">Metropolitan Area Statistics</text>
      <text class="sub" x="260" y="66">4 tables &#183; 4 mappers</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="408" y="80" width="56" height="190"/>
    <line class="lead" x1="436" y1="74" x2="436" y2="237"/>
    <rect class="bar" x="419" y="237" width="34" height="11" rx="2"/>
    <text class="val" x="436" y="231">2</text>
    <text class="tick" x="436" y="264">REALER</text>
    <g class="tip">
      <rect class="tipbox" x="304" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="316" y="51">Real Earnings</text>
      <text class="sub" x="316" y="66">2 tables &#183; 2 mappers</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="464" y="80" width="56" height="190"/>
    <line class="lead" x1="492" y1="74" x2="492" y2="215"/>
    <rect class="bar" x="475" y="215" width="34" height="33" rx="2"/>
    <text class="val" x="492" y="209">6</text>
    <text class="tick" x="492" y="264">WKYENG</text>
    <g class="tip">
      <rect class="tipbox" x="360" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="372" y="51">Weekly Earnings</text>
      <text class="sub" x="372" y="66">6 tables &#183; 6 mappers</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="520" y="80" width="56" height="190"/>
    <line class="lead" x1="548" y1="74" x2="548" y2="187.5"/>
    <rect class="bar" x="531" y="187.5" width="34" height="60.5" rx="2"/>
    <text class="val" x="548" y="181.5">11</text>
    <text class="tick" x="548" y="264">XIMPIM</text>
    <g class="tip">
      <rect class="tipbox" x="416" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="428" y="51">Import/Export Price Indexes</text>
      <text class="sub" x="428" y="66">11 tables &#183; 11 mappers</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="576" y="80" width="56" height="190"/>
    <line class="lead" x1="604" y1="74" x2="604" y2="242.5"/>
    <rect class="bar" x="587" y="242.5" width="34" height="5.5" rx="2"/>
    <text class="val" x="604" y="236.5">1</text>
    <text class="tick" x="604" y="264">SEC</text>
    <g class="tip">
      <rect class="tipbox" x="472" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="484" y="51">Filings &#183; 10-K, 10-Q, 8-K, Forms 3/4/5</text>
      <text class="sub" x="484" y="66">1 of 8 SEC feeds &#183; 1 mapper</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="632" y="80" width="56" height="190"/>
    <line class="lead" x1="660" y1="74" x2="660" y2="242.5"/>
    <rect class="bar" x="643" y="242.5" width="34" height="5.5" rx="2"/>
    <text class="val" x="660" y="236.5">1</text>
    <text class="tick" x="660" y="264">Market</text>
    <g class="tip">
      <rect class="tipbox" x="488" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="500" y="51">Intraday equity and option snapshots</text>
      <text class="sub" x="500" y="66">1 mapper</text>
    </g>
  </g>

  <g class="col">
    <rect class="hit" x="688" y="80" width="56" height="190"/>
    <line class="lead" x1="716" y1="74" x2="716" y2="242.5"/>
    <rect class="bar" x="699" y="242.5" width="34" height="5.5" rx="2"/>
    <text class="val" x="716" y="236.5">1</text>
    <text class="tick" x="716" y="264">NOAA</text>
    <g class="tip">
      <rect class="tipbox" x="488" y="34" width="264" height="38" rx="4"/>
      <text class="lbl" x="500" y="51">US weather alerts, CAP format</text>
      <text class="sub" x="500" y="66">1 mapper</text>
    </g>
  </g>
</svg>
</div>

**BLS** is the only source split into categories — ten of them, 97 published
tables between them. **SEC** is one feed of eight: `feed=filings` is the only one
carrying RDF, and a source path naming any of the other seven is rejected before
the job starts. **Market** is a single flat vocabulary, `EquitySnapshot` and
`OptionSnapshot` with every field a direct property, covering ~500+ tickers with
full options chains (~500K+ symbols per snapshot) at ~39 snapshots a day on
20-minute intervals during market hours. **NOAA** is US weather alerts in CAP
format.

> **Measured volume:** one four-source day loads **322.7M triples** and enriches
> to **421.4M**. Market is 99.5% of that; BLS 1.3M, SEC 198K, NOAA 143K.

> **Note:** Raw RDF data is generated by separate Lambda scraper functions (not
> part of this repository). This pipeline assumes RDF data is already available
> in S3 in N-Triples format conforming to 100+ domain-specific ontologies.

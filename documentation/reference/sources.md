# Data Sources

The pipeline ingests RDF from four sources: BLS economic series, SEC filings,
intraday market snapshots, and NOAA weather alerts. One mapper is written per
published table, and the chart below is all one hundred of them — hover a bar
for the category behind it.

<div>
<svg viewBox="0 0 780 296" xmlns="http://www.w3.org/2000/svg" role="img"
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
    .frame { fill: transparent; pointer-events: all;
             stroke: var(--md-default-fg-color--lightest, #d8d8dc);
             transition: stroke 120ms; }
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
    .grp:hover .frame { stroke: #3aa76d; }
  </style>

  <text class="cap" x="12" y="22">MAPPERS PER SOURCE &#183; 100 TOTAL</text>

  <g class="grp">
    <rect class="frame" x="12" y="44" width="548" height="240" rx="8"/>
    <line class="axis" x1="26" y1="234" x2="546" y2="234"/>
    <text class="cap" x="26" y="270">BLS &#183; 10 CATEGORIES &#183; 97 TABLES</text>

    <g class="col">
      <rect class="hit" x="26" y="46" width="52" height="212"/>
      <line class="lead" x1="52" y1="44" x2="52" y2="186"/>
      <rect class="bar" x="40" y="186" width="24" height="48" rx="2"/>
      <text class="val" x="52" y="180">8</text>
      <text class="tick" x="52" y="250">CPI</text>
      <g class="tip">
        <rect class="tipbox" x="12" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="24" y="21">Consumer Price Index</text>
        <text class="sub" x="24" y="36">8 tables &#183; 8 mappers</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="78" y="46" width="52" height="212"/>
      <line class="lead" x1="104" y1="44" x2="104" y2="192"/>
      <rect class="bar" x="92" y="192" width="24" height="42" rx="2"/>
      <text class="val" x="104" y="186">7</text>
      <text class="tick" x="104" y="250">PPI</text>
      <g class="tip">
        <rect class="tipbox" x="12" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="24" y="21">Producer Price Index</text>
        <text class="sub" x="24" y="36">7 tables &#183; 7 mappers</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="130" y="46" width="52" height="212"/>
      <line class="lead" x1="156" y1="44" x2="156" y2="150"/>
      <rect class="bar" x="144" y="150" width="24" height="84" rx="2"/>
      <text class="val" x="156" y="144">14</text>
      <text class="tick" x="156" y="250">ECI</text>
      <g class="tip">
        <rect class="tipbox" x="24" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="36" y="21">Employment Cost Index</text>
        <text class="sub" x="36" y="36">14 tables &#183; 14 mappers</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="182" y="46" width="52" height="212"/>
      <line class="lead" x1="208" y1="44" x2="208" y2="72"/>
      <rect class="bar" x="196" y="72" width="24" height="162" rx="2"/>
      <text class="val" x="208" y="66">27</text>
      <text class="tick" x="208" y="250">EMPSIT</text>
      <g class="tip">
        <rect class="tipbox" x="76" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="88" y="21">Employment Situation</text>
        <text class="sub" x="88" y="36">27 tables &#183; 27 mappers</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="234" y="46" width="52" height="212"/>
      <line class="lead" x1="260" y1="44" x2="260" y2="144"/>
      <rect class="bar" x="248" y="144" width="24" height="90" rx="2"/>
      <text class="val" x="260" y="138">15</text>
      <text class="tick" x="260" y="250">JOLTS</text>
      <g class="tip">
        <rect class="tipbox" x="128" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="140" y="21">Job Openings and Labor Turnover</text>
        <text class="sub" x="140" y="36">15 tables &#183; 15 mappers</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="286" y="46" width="52" height="212"/>
      <line class="lead" x1="312" y1="44" x2="312" y2="216"/>
      <rect class="bar" x="300" y="216" width="24" height="18" rx="2"/>
      <text class="val" x="312" y="210">3</text>
      <text class="tick" x="312" y="250">LAUS</text>
      <g class="tip">
        <rect class="tipbox" x="180" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="192" y="21">Local Area Unemployment Statistics</text>
        <text class="sub" x="192" y="36">3 tables &#183; 3 mappers</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="338" y="46" width="52" height="212"/>
      <line class="lead" x1="364" y1="44" x2="364" y2="210"/>
      <rect class="bar" x="352" y="210" width="24" height="24" rx="2"/>
      <text class="val" x="364" y="204">4</text>
      <text class="tick" x="364" y="250">METRO</text>
      <g class="tip">
        <rect class="tipbox" x="232" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="244" y="21">Metropolitan Area Statistics</text>
        <text class="sub" x="244" y="36">4 tables &#183; 4 mappers</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="390" y="46" width="52" height="212"/>
      <line class="lead" x1="416" y1="44" x2="416" y2="222"/>
      <rect class="bar" x="404" y="222" width="24" height="12" rx="2"/>
      <text class="val" x="416" y="216">2</text>
      <text class="tick" x="416" y="250">REALER</text>
      <g class="tip">
        <rect class="tipbox" x="284" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="296" y="21">Real Earnings</text>
        <text class="sub" x="296" y="36">2 tables &#183; 2 mappers</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="442" y="46" width="52" height="212"/>
      <line class="lead" x1="468" y1="44" x2="468" y2="198"/>
      <rect class="bar" x="456" y="198" width="24" height="36" rx="2"/>
      <text class="val" x="468" y="192">6</text>
      <text class="tick" x="468" y="250">WKYENG</text>
      <g class="tip">
        <rect class="tipbox" x="336" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="348" y="21">Weekly Earnings</text>
        <text class="sub" x="348" y="36">6 tables &#183; 6 mappers</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="494" y="46" width="52" height="212"/>
      <line class="lead" x1="520" y1="44" x2="520" y2="168"/>
      <rect class="bar" x="508" y="168" width="24" height="66" rx="2"/>
      <text class="val" x="520" y="162">11</text>
      <text class="tick" x="520" y="250">XIMPIM</text>
      <g class="tip">
        <rect class="tipbox" x="388" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="400" y="21">Import/Export Price Indexes</text>
        <text class="sub" x="400" y="36">11 tables &#183; 11 mappers</text>
      </g>
    </g>
  </g>

  <g class="grp">
    <rect class="frame" x="584" y="44" width="184" height="240" rx="8"/>
    <line class="axis" x1="598" y1="234" x2="754" y2="234"/>
    <text class="cap" x="598" y="270">ONE MAPPER EACH</text>

    <g class="col">
      <rect class="hit" x="598" y="46" width="52" height="212"/>
      <line class="lead" x1="624" y1="44" x2="624" y2="228"/>
      <rect class="bar" x="612" y="228" width="24" height="6" rx="2"/>
      <text class="val" x="624" y="222">1</text>
      <text class="tick" x="624" y="250">SEC</text>
      <g class="tip">
        <rect class="tipbox" x="492" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="504" y="21">Filings &#183; 10-K, 10-Q, 8-K, Forms 3/4/5</text>
        <text class="sub" x="504" y="36">1 of 8 SEC feeds &#183; 1 mapper</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="650" y="46" width="52" height="212"/>
      <line class="lead" x1="676" y1="44" x2="676" y2="228"/>
      <rect class="bar" x="664" y="228" width="24" height="6" rx="2"/>
      <text class="val" x="676" y="222">1</text>
      <text class="tick" x="676" y="250">Market</text>
      <g class="tip">
        <rect class="tipbox" x="504" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="516" y="21">Intraday equity and option snapshots</text>
        <text class="sub" x="516" y="36">1 mapper</text>
      </g>
    </g>

    <g class="col">
      <rect class="hit" x="702" y="46" width="52" height="212"/>
      <line class="lead" x1="728" y1="44" x2="728" y2="228"/>
      <rect class="bar" x="716" y="228" width="24" height="6" rx="2"/>
      <text class="val" x="728" y="222">1</text>
      <text class="tick" x="728" y="250">NOAA</text>
      <g class="tip">
        <rect class="tipbox" x="504" y="4" width="264" height="38" rx="4"/>
        <text class="lbl" x="516" y="21">US weather alerts, CAP format</text>
        <text class="sub" x="516" y="36">1 mapper</text>
      </g>
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

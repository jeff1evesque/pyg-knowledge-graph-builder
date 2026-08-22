"""
One region vocabulary, and the tables that get every source into it.

THE SPLIT THIS CLOSES
---------------------
Two region vocabularies existed and nothing joined them:

    bls_enrichment_GeographicRegion   <- the weather feed attaches here
    jolts_Region                      <- regional economic series attach here

So `WeatherAlert -> Region -> regional indicator` did not exist even though both
halves were built. Unifying the region vocabulary is the only data-grounded
route the weather feed has to the rest of the graph -- everything else it could
be joined on is a coincidence of timing.

Note this is a GEOGRAPHIC axis and only that. A sector is not located in a
region, so the two must never be mapped onto each other: weather reaches
economic activity through geography, and reaches an industry only transitively,
through the activity. The `geographic_regions_sector` entry removed from
bls/patterns.py was that error made concrete -- it sorted states and countries
into a pseudo-industry.

THE TWO GRAINS, AND WHY THEY ARE DIFFERENT
------------------------------------------
1. STATE. The local-area series are state-level only -- 53 areas, no counties --
   so a weather county FIPS reaches them by taking its first two digits. That
   needs no crosswalk file at all, which makes it the fastest path to a
   connected graph and the reason the state region is the primary node here.

2. CENSUS REGION. The job-openings series are NOT state-level: they are the four
   census regions. Reaching those needs the state -> census-region table below,
   which is maintained here because neither side emits it. It is Census-defined
   and has been stable for decades, which is what makes a hand-maintained table
   acceptable for it and not for, say, metro delineation.

Metro-to-county resolution needs a federal delineation file neither side holds,
and is deliberately out of scope.

ON THE UPSTREAM HALF
--------------------
The weather side works today. The economic side does not yet: the regional
series emit a name and a slug subject and NO CODE AT ALL, and the codes exist in
the upstream catalogs unmapped. They are to be emitted as `laus:hasStateFIPS`,
`metro:hasStateFIPS` and `metro:hasCBSACode`, all zero-padded `xsd:string`.

The readers keyed on those terms are written and tested here anyway, against
synthetic triples, and they match nothing until upstream lands. That is
deliberate: the alternative is discovering the shape mismatch after the fact.
THE DATATYPE MATTERS MORE THAN THE NAME -- a numeric 1 never matches a string
"01", and that failure is silent.

The name-keyed path for the job-openings regions works today, because those
entities already state a label and a slug, so the census-region bridge does not
have to wait for a code it will merely be confirmed by.
"""
from typing import Dict, Tuple

from pyspark.sql import Column
from pyspark.sql import functions as F

# ============================================
# States
# ============================================
#
# DC is included throughout. It is a FIPS state (11), the weather service issues
# alerts for it, and the census assigns it to the South -- so leaving it out
# would have put a hole in a 51-row table for no reason. The previous state list
# in cross_source_linker omitted it and silently dropped every DC alert.
STATE_ABBREVIATIONS: Dict[str, str] = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT',
    'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL',
    'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY',
    'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY',
}

US_STATES: Tuple[str, ...] = tuple(sorted(STATE_ABBREVIATIONS))

# 2-digit state FIPS. The gaps are real -- FIPS skips 03, 07, 14, 43 and 52 --
# so this is a lookup rather than a range.
STATE_FIPS_TO_NAME: Dict[str, str] = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
    '06': 'California', '08': 'Colorado', '09': 'Connecticut',
    '10': 'Delaware', '11': 'District of Columbia', '12': 'Florida',
    '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois',
    '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky',
    '22': 'Louisiana', '23': 'Maine', '24': 'Maryland',
    '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana',
    '31': 'Nebraska', '32': 'Nevada', '33': 'New Hampshire',
    '34': 'New Jersey', '35': 'New Mexico', '36': 'New York',
    '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
    '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania',
    '44': 'Rhode Island', '45': 'South Carolina', '46': 'South Dakota',
    '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont',
    '51': 'Virginia', '53': 'Washington', '54': 'West Virginia',
    '55': 'Wisconsin', '56': 'Wyoming',
}


def state_key(state_name: str) -> str:
    """The URI-safe form of a state name, as the region URI uses it."""
    return state_name.replace(' ', '').replace('.', '')


# ============================================
# Census regions
# ============================================
#
# The four Census Bureau regions, and their numeric codes -- which is how
# upstream will key jolts:hasCensusRegionCode. Zero-padded strings, not ints:
# a numeric 1 never matches a string "01", and that mismatch is silent.
CENSUS_REGION_CODES: Dict[str, str] = {
    "Northeast": "01",
    "Midwest": "02",
    "South": "03",
    "West": "04",
}

CENSUS_REGIONS: Tuple[str, ...] = tuple(CENSUS_REGION_CODES)

# The 51 rows: 50 states plus DC, each in exactly one census region.
#
# Census-defined and unchanged for decades, which is what makes maintaining it
# here reasonable. Written as region -> members so the grouping is checkable by
# eye against the Census definition; inverted below for lookups.
_CENSUS_REGION_MEMBERS: Dict[str, Tuple[str, ...]] = {
    # New England + Middle Atlantic
    "Northeast": (
        'Connecticut', 'Maine', 'Massachusetts', 'New Hampshire',
        'Rhode Island', 'Vermont',
        'New Jersey', 'New York', 'Pennsylvania',
    ),
    # East North Central + West North Central
    "Midwest": (
        'Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin',
        'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska',
        'North Dakota', 'South Dakota',
    ),
    # South Atlantic + East South Central + West South Central
    "South": (
        'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Maryland',
        'North Carolina', 'South Carolina', 'Virginia', 'West Virginia',
        'Alabama', 'Kentucky', 'Mississippi', 'Tennessee',
        'Arkansas', 'Louisiana', 'Oklahoma', 'Texas',
    ),
    # Mountain + Pacific
    "West": (
        'Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico',
        'Utah', 'Wyoming',
        'Alaska', 'California', 'Hawaii', 'Oregon', 'Washington',
    ),
}

STATE_TO_CENSUS_REGION: Dict[str, str] = {
    state: region
    for region, states in _CENSUS_REGION_MEMBERS.items()
    for state in states
}


# ============================================
# FIPS normalisation
# ============================================


def normalized_state_fips(column: Column) -> Column:
    """The 2-digit state FIPS, from whatever width upstream states.

    TOLERANT ON PURPOSE, AND PERMANENTLY SO. The weather geocodes currently emit
    nws:hasStateFIPS as the 3-character head of a SAME code -- "020", "024" --
    because upstream slices it as value[:3] and keeps the leading part-digit.
    That is being corrected upstream to a bare 2-digit FIPS. Reading the LAST
    two digits is correct for BOTH the current and the fixed form, so the
    upstream fix is non-breaking and needs no backfill, and this reader should
    stay tolerant permanently rather than being tightened once the fix lands.

    Taking the last two digits rather than stripping a leading zero: the
    part-digit is not always 0 -- it marks a partial-county code -- so trimming
    zeros would mangle those.

    substring(-2, 2) alone, NOT lpad(..., 2, "0") first. Spark's lpad TRUNCATES
    when the input is longer than the target width, and it truncates from the
    RIGHT: lpad("040", 2, "0") is "04", not "40". That destroyed the state digit
    and produced confidently WRONG states -- Oklahoma (040) and Texas (048) both
    resolved to Arizona (04). A negative start already reads from the end and is
    safe on short input, so no padding is needed at all.
    """
    return F.substring(F.trim(column), -2, 2)

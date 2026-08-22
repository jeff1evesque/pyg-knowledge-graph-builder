"""
Tests for the region tables (enrichment/region_crosswalk.py).

Pure Python, no SparkSession. These pin hand-maintained reference data whose
errors are silent: a state assigned to the wrong census region does not fail,
it gives the weather feed a confident path to economic series for a part of the
country the weather never touched.
"""
import pytest

from spark_jobs.enrichment import region_crosswalk as regions


# ======================================================================
# The 51-row census table
# ======================================================================

def test_the_census_table_has_a_row_per_state_and_dc():
    """51 = 50 states + District of Columbia.

    DC is included on purpose: it is a FIPS state (11), the weather service
    issues alerts for it, and the Census assigns it to the South. The state
    list this replaced omitted it and silently dropped every DC alert.
    """
    assert len(regions.STATE_TO_CENSUS_REGION) == 51
    assert regions.STATE_TO_CENSUS_REGION["District of Columbia"] == "South"


def test_every_state_is_in_exactly_one_census_region():
    """A state in two regions gives the weather feed a path to series for a
    part of the country it has nothing to do with."""
    seen = {}
    for region, members in regions._CENSUS_REGION_MEMBERS.items():
        for state in members:
            assert state not in seen, (
                f"{state} is in both {seen[state]} and {region}"
            )
            seen[state] = region

    assert set(seen) == set(regions.STATE_ABBREVIATIONS)


def test_the_census_regions_are_the_four_the_bureau_defines():
    assert regions.CENSUS_REGIONS == ("Northeast", "Midwest", "South", "West")


@pytest.mark.parametrize("state,region", [
    ("Kansas", "Midwest"),
    ("Texas", "South"),
    ("New York", "Northeast"),
    ("California", "West"),
    # The two that are geographically confusable and censally is not.
    ("Maryland", "South"),
    ("Delaware", "South"),
])
def test_states_land_in_their_census_region(state, region):
    assert regions.STATE_TO_CENSUS_REGION[state] == region


def test_census_region_codes_are_zero_padded_strings():
    """THE DATATYPE MATTERS MORE THAN THE NAME.

    Upstream will key jolts:hasCensusRegionCode as a zero-padded xsd:string. A
    numeric 1 never matches a string "01", and the join comes back empty with
    no error — which is indistinguishable from data that does not overlap.
    """
    for name, code in regions.CENSUS_REGION_CODES.items():
        assert isinstance(code, str), name
        assert len(code) == 2 and code.isdigit(), (name, code)


# ======================================================================
# States, abbreviations, FIPS
# ======================================================================

def test_every_state_has_an_abbreviation_and_a_fips_code():
    assert len(regions.US_STATES) == 51
    assert set(regions.STATE_FIPS_TO_NAME.values()) == set(regions.US_STATES)


def test_fips_codes_are_two_digit_and_unique():
    codes = list(regions.STATE_FIPS_TO_NAME)
    assert len(codes) == len(set(codes))
    for code in codes:
        assert len(code) == 2 and code.isdigit(), code


def test_abbreviations_are_unique():
    values = list(regions.STATE_ABBREVIATIONS.values())
    assert len(values) == len(set(values))


@pytest.mark.parametrize("name,key", [
    ("Kansas", "Kansas"),
    ("New York", "NewYork"),
    ("West Virginia", "WestVirginia"),
    ("District of Columbia", "DistrictofColumbia"),
])
def test_state_key_matches_the_region_uri_form(name, key):
    assert regions.state_key(name) == key


def test_state_keys_are_unique():
    """Two states sharing a key would share a region node."""
    keys = [regions.state_key(name) for name in regions.US_STATES]
    assert len(keys) == len(set(keys))


def test_exactly_one_state_name_contains_another():
    """Pins the population the delimited-head match has to separate.

    Only ONE pair of state names is confusable by substring: "Virginia" sits
    inside "WestVirginia". That is the pair that broke the old
    `subject.contains(state_key)` match, and it is the whole population -- not
    a sample of a larger problem.

    Arkansas/Kansas is NOT such a pair, which is worth stating because it looks
    like one. The comparison is case-sensitive and the state key is
    "Arkansas": the 'k' is lowercase, so "Kansas" is not a substring of it and
    never was. Assuming otherwise is how a comment ends up describing a bug the
    code does not have.

    No key is a PREFIX of another either, which is what makes anchoring the
    match at the head of the local name sufficient. If a future name broke that
    -- a "Dakota" alongside "DakotaNorth", say -- the head anchor would stop
    being enough, and this fails rather than silently under-separating.
    """
    keys = sorted(regions.state_key(name) for name in regions.US_STATES)

    prefixed = {(a, b) for a in keys for b in keys if a != b and b.startswith(a)}
    assert prefixed == set(), (
        f"a state key is a prefix of another: {prefixed}. The head-anchored "
        "match in cross_source_linker._delimited_local_name cannot separate "
        "these."
    )

    substrings = {(a, b) for a in keys for b in keys if a != b and a in b}
    assert substrings == {("Virginia", "WestVirginia")}, substrings

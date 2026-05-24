"""
NOAA Weather Event Patterns - Weather event types and geographic regions

Aligned with CAP 1.2 ontology v3.0 and NWS RML mapper.
Event types match the literal strings produced by the mapper's
cap:hasEvent property (on Info subjects).

Categories align with the CAP enumeration classes (cap:Category)
and the NWS SKOS event type vocabulary (nws:EventTypeScheme).
"""
from spark_jobs.utils.rdf_utils import NOAA_ENRICHMENT

# ============================================
# EVENT TYPE PATTERNS
# ============================================
# Keys are category names used by the enricher.
# 'event_types' lists match the literal string values of cap:hasEvent
# as produced by the RML mapper (e.g., "Tornado Warning").
# 'relationship' is the enrichment predicate used to classify alerts.

NOAA_EVENT_PATTERNS = {
    'severe_weather': {
        'description': 'Severe weather events including tornadoes and thunderstorms',
        'event_types': [
            'Severe Thunderstorm Warning',
            'Severe Thunderstorm Watch',
            'Tornado Warning',
            'Tornado Watch',
            'Extreme Wind Warning',
        ],
        'cap_category': 'Met',
        'relationship': str(NOAA_ENRICHMENT.severeWeatherEvent),
    },
    'flood_events': {
        'description': 'Flooding and water-related events',
        'event_types': [
            'Flood Warning',
            'Flood Watch',
            'Flood Advisory',
            'Flash Flood Warning',
            'Flash Flood Watch',
            'Coastal Flood Warning',
            'Coastal Flood Watch',
            'Coastal Flood Advisory',
        ],
        'cap_category': 'Met',
        'relationship': str(NOAA_ENRICHMENT.floodEvent),
    },
    'winter_weather': {
        'description': 'Winter weather events',
        'event_types': [
            'Winter Storm Warning',
            'Winter Storm Watch',
            'Winter Weather Advisory',
            'Blizzard Warning',
            'Blizzard Watch',
            'Ice Storm Warning',
            'Freezing Rain Advisory',
            'Wind Chill Warning',
            'Wind Chill Advisory',
        ],
        'cap_category': 'Met',
        'relationship': str(NOAA_ENRICHMENT.winterWeatherEvent),
    },
    'heat_events': {
        'description': 'Heat-related events',
        'event_types': [
            'Excessive Heat Warning',
            'Excessive Heat Watch',
            'Heat Advisory',
        ],
        'cap_category': 'Met',
        'relationship': str(NOAA_ENRICHMENT.heatEvent),
    },
    'tropical_events': {
        'description': 'Tropical weather events including hurricanes',
        'event_types': [
            'Hurricane Warning',
            'Hurricane Watch',
            'Tropical Storm Warning',
            'Tropical Storm Watch',
            'Storm Surge Warning',
            'Storm Surge Watch',
        ],
        'cap_category': 'Met',
        'relationship': str(NOAA_ENRICHMENT.tropicalEvent),
    },
    'fire_events': {
        'description': 'Fire weather events',
        'event_types': [
            'Red Flag Warning',
            'Fire Weather Watch',
            'Fire Warning',
        ],
        'cap_category': 'Fire',
        'relationship': str(NOAA_ENRICHMENT.fireEvent),
    },
    'marine_events': {
        'description': 'Marine and coastal weather events',
        'event_types': [
            'Small Craft Advisory',
            'Gale Warning',
            'Storm Warning',
            'Hurricane Force Wind Warning',
            'Special Marine Warning',
            'Marine Weather Statement',
        ],
        'cap_category': 'Met',
        'relationship': str(NOAA_ENRICHMENT.marineEvent),
    },
}

# ============================================
# SEVERITY HIERARCHY
# ============================================
# Maps CAP severity named individual URIs to numeric levels.
# Higher number = more severe. Used by escalation detection.
# URIs match the enum mapping in the RML mapper:
#   Extreme → cap:Extreme, Severe → cap:Severe, etc.

SEVERITY_HIERARCHY = {
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Minor": 1,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Moderate": 2,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Severe": 3,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Extreme": 4,
}

# ============================================
# URGENCY HIERARCHY
# ============================================
# Maps CAP urgency named individual URIs to numeric levels.
# Higher number = more urgent. Used by urgency escalation detection.

URGENCY_HIERARCHY = {
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Past": 1,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Future": 2,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Expected": 3,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Immediate": 4,
}

# ============================================
# CERTAINTY HIERARCHY
# ============================================
# Maps CAP certainty named individual URIs to numeric levels.

CERTAINTY_HIERARCHY = {
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Unlikely": 1,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Possible": 2,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Likely": 3,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Observed": 4,
}

# ============================================
# RESPONSE TYPE SEVERITY
# ============================================
# Maps CAP response type named individual URIs to severity weight.
# Used to augment severity escalation detection.

RESPONSE_TYPE_SEVERITY = {
    "http://www.oasis-open.org/committees/emergency/cap/1.2/None": 0,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/AllClear": 0,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Monitor": 1,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Assess": 1,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Prepare": 2,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Avoid": 3,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Execute": 3,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Shelter": 4,
    "http://www.oasis-open.org/committees/emergency/cap/1.2/Evacuate": 5,
}
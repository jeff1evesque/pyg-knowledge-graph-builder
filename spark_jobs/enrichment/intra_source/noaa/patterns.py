"""
NOAA Weather Event Patterns - Weather event types and geographic regions

Aligned with CAP 1.2 ontology v3.0 and NWS RML mapper.
Event types match the literal strings produced by the mapper's
cap:hasEvent property (on Info subjects).

Categories align with the CAP enumeration classes (cap:Category)
and the NWS SKOS event type vocabulary (nws:EventTypeScheme).
"""
from spark_jobs.utils.rdf_utils import NOAA_ENRICHMENT, CAP

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
    str(CAP.Minor): 1,
    str(CAP.Moderate): 2,
    str(CAP.Severe): 3,
    str(CAP.Extreme): 4,
}

# ============================================
# URGENCY HIERARCHY
# ============================================
# Maps CAP urgency named individual URIs to numeric levels.
# Higher number = more urgent. Used by urgency escalation detection.

URGENCY_HIERARCHY = {
    str(CAP.Past): 1,
    str(CAP.Future): 2,
    str(CAP.Expected): 3,
    str(CAP.Immediate): 4,
}

# ============================================
# CERTAINTY HIERARCHY
# ============================================
# Maps CAP certainty named individual URIs to numeric levels.

CERTAINTY_HIERARCHY = {
    str(CAP.Unlikely): 1,
    str(CAP.Possible): 2,
    str(CAP.Likely): 3,
    str(CAP.Observed): 4,
}

# ============================================
# RESPONSE TYPE SEVERITY
# ============================================
# Maps CAP response type named individual URIs to severity weight.
# Used to augment severity escalation detection.

RESPONSE_TYPE_SEVERITY = {
    str(CAP["None"]): 0,
    str(CAP.AllClear): 0,
    str(CAP.Monitor): 1,
    str(CAP.Assess): 1,
    str(CAP.Prepare): 2,
    str(CAP.Avoid): 3,
    str(CAP.Execute): 3,
    str(CAP.Shelter): 4,
    str(CAP.Evacuate): 5,
}
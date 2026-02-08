"""
NOAA Weather Event Patterns - Weather event types and geographic regions
"""
from glue_jobs.utils.rdf_utils import BLS_ENRICHMENT

NOAA_EVENT_PATTERNS = {
    'severe_weather': {
        'description': 'Severe weather events',
        'event_types': [
            'Severe Thunderstorm Warning',
            'Severe Thunderstorm Watch',
            'Tornado Warning',
            'Tornado Watch',
        ],
        'relationship': BLS_ENRICHMENT.severeWeatherEvent
    },
    'flood_events': {
        'description': 'Flooding and water-related events',
        'event_types': [
            'Flood Warning',
            'Flood Advisory',
            'Flash Flood Warning',
            'Coastal Flood Warning',
        ],
        'relationship': BLS_ENRICHMENT.floodEvent
    },
    'winter_weather': {
        'description': 'Winter weather events',
        'event_types': [
            'Winter Storm Warning',
            'Winter Weather Advisory',
            'Blizzard Warning',
            'Ice Storm Warning',
        ],
        'relationship': BLS_ENRICHMENT.winterWeatherEvent
    },
    'heat_events': {
        'description': 'Heat-related events',
        'event_types': [
            'Excessive Heat Warning',
            'Heat Advisory',
        ],
        'relationship': BLS_ENRICHMENT.heatEvent
    },
}
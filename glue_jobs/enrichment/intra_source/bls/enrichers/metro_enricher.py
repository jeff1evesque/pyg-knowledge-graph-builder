"""
METRO-specific enrichment logic

Metropolitan Area Statistics - Employment and unemployment data for US metropolitan areas
"""
from rdflib import Graph
from typing import Dict, List
from glue_jobs.utils.rdf_utils import METRO, BLS_ENRICHMENT, get_month_name, get_year_value
from glue_jobs.enrichment.intra_source.base import DatasetEnricher
from glue_jobs.enrichment.intra_source.bls.patterns import BLS_SECTOR_PATTERNS
from glue_jobs.enrichment.intra_source.bls.measurements import MEASUREMENT_TYPES
import logging

logger = logging.getLogger(__name__)


class METROEnricher(DatasetEnricher):
    """
    METRO-specific enrichment

    Metropolitan Area Statistics provides employment and unemployment data
    for US metropolitan statistical areas (MSAs), including:
    - Civilian labor force by metro area
    - Unemployment levels and rates by metro area
    - Metropolitan divisions for large metro areas
    - Both seasonally adjusted and not seasonally adjusted data

    Data Structure:
    - Table 1: States and Selected Areas (Not Seasonally Adjusted)
    - Table 2: Large Metropolitan Divisions (Not Seasonally Adjusted)
    - Table 3: States and Selected Areas (Seasonally Adjusted)
    - Table 4: Large Metropolitan Divisions (Seasonally Adjusted)
    """

    def __init__(self, graph: Graph):
        super().__init__(graph, METRO)
        self.month_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        # Metropolitan areas from the provided data
        self.metro_areas = self._get_metro_areas()
        self.metro_divisions = self._get_metro_divisions()

    def _get_metro_areas(self) -> List[str]:
        """Return list of metropolitan areas from tables 1 and 3"""
        return [
            # Alabama
            'Anniston-Oxford', 'Auburn-Opelika', 'Birmingham', 'Daphne-Fairhope-Foley',
            'Decatur', 'Dothan', 'Florence-Muscle Shoals', 'Gadsden', 'Huntsville',
            'Mobile', 'Montgomery', 'Tuscaloosa',

            # Alaska
            'Anchorage', 'Fairbanks-College',

            # Arizona
            'Flagstaff', 'Lake Havasu City-Kingman', 'Phoenix-Mesa-Chandler',
            'Prescott Valley-Prescott', 'Sierra Vista-Douglas', 'Tucson', 'Yuma',

            # Arkansas
            'Fayetteville-Springdale-Rogers', 'Fort Smith', 'Hot Springs', 'Jonesboro',
            'Little Rock-North Little Rock-Conway',

            # California
            'Bakersfield-Delano', 'Chico', 'El Centro', 'Fresno', 'Hanford-Corcoran',
            'Los Angeles-Long Beach-Anaheim', 'Merced', 'Modesto', 'Napa',
            'Oxnard-Thousand Oaks-Ventura', 'Redding', 'Riverside-San Bernardino-Ontario',
            'Sacramento-Roseville-Folsom', 'Salinas', 'San Diego-Chula Vista-Carlsbad',
            'San Francisco-Oakland-Fremont', 'San Jose-Sunnyvale-Santa Clara',
            'San Luis Obispo-Paso Robles', 'Santa Cruz-Watsonville', 'Santa Maria-Santa Barbara',
            'Santa Rosa-Petaluma', 'Stockton-Lodi', 'Vallejo', 'Visalia', 'Yuba City',

            # Colorado
            'Boulder', 'Colorado Springs', 'Denver-Aurora-Centennial', 'Fort Collins-Loveland',
            'Grand Junction', 'Greeley', 'Pueblo',

            # Connecticut
            'Bridgeport-Stamford-Danbury', 'Hartford-West Hartford-East Hartford',
            'New Haven', 'Norwich-New London-Willimantic', 'Waterbury-Shelton',

            # Delaware
            'Dover',

            # District of Columbia
            'Washington-Arlington-Alexandria',

            # Florida
            'Cape Coral-Fort Myers', 'Crestview-Fort Walton Beach-Destin',
            'Deltona-Daytona Beach-Ormond Beach', 'Gainesville', 'Homosassa Springs',
            'Jacksonville', 'Lakeland-Winter Haven', 'Miami-Fort Lauderdale-West Palm Beach',
            'Naples-Marco Island', 'North Port-Bradenton-Sarasota', 'Ocala',
            'Orlando-Kissimmee-Sanford', 'Palm Bay-Melbourne-Titusville',
            'Panama City-Panama City Beach', 'Pensacola-Ferry Pass-Brent', 'Port St. Lucie',
            'Punta Gorda', 'Sebastian-Vero Beach-West Vero Corridor', 'Sebring',
            'Tallahassee', 'Tampa-St. Petersburg-Clearwater', 'Wildwood-The Villages',

            # Georgia
            'Albany', 'Athens-Clarke County', 'Atlanta-Sandy Springs-Roswell',
            'Augusta-Richmond County', 'Brunswick-St. Simons', 'Columbus', 'Dalton',
            'Gainesville', 'Hinesville', 'Macon-Bibb County', 'Rome', 'Savannah',
            'Valdosta', 'Warner Robins',

            # Hawaii
            'Kahului-Wailuku', 'Urban Honolulu',

            # Idaho
            'Boise City', "Coeur d'Alene", 'Idaho Falls', 'Lewiston', 'Pocatello', 'Twin Falls',

            # Illinois
            'Bloomington', 'Champaign-Urbana', 'Chicago-Naperville-Elgin', 'Decatur',
            'Kankakee', 'Peoria', 'Rockford', 'Springfield',

            # Indiana
            'Bloomington', 'Columbus', 'Elkhart-Goshen', 'Evansville', 'Fort Wayne',
            'Indianapolis-Carmel-Greenwood', 'Kokomo', 'Lafayette-West Lafayette',
            'Michigan City-La Porte', 'Muncie', 'South Bend-Mishawaka', 'Terre Haute',

            # Iowa
            'Ames', 'Cedar Rapids', 'Davenport-Moline-Rock Island', 'Des Moines-West Des Moines',
            'Dubuque', 'Iowa City', 'Sioux City', 'Waterloo-Cedar Falls',

            # Kansas
            'Lawrence', 'Manhattan', 'Topeka', 'Wichita',

            # Kentucky
            'Bowling Green', 'Elizabethtown', 'Lexington-Fayette', 'Louisville/Jefferson County',
            'Owensboro', 'Paducah',

            # Louisiana
            'Alexandria', 'Baton Rouge', 'Hammond', 'Houma-Bayou Cane-Thibodaux',
            'Lafayette', 'Lake Charles', 'Monroe', 'New Orleans-Metairie',
            'Shreveport-Bossier City', 'Slidell-Mandeville-Covington',

            # Maine
            'Bangor', 'Lewiston-Auburn', 'Portland-South Portland',

            # Maryland
            'Baltimore-Columbia-Towson', 'Hagerstown-Martinsburg', 'Lexington Park', 'Salisbury',

            # Massachusetts
            'Amherst Town-Northampton', 'Barnstable Town', 'Boston-Cambridge-Newton',
            'Pittsfield', 'Springfield', 'Worcester',

            # Michigan
            'Ann Arbor', 'Battle Creek', 'Bay City', 'Detroit-Warren-Dearborn', 'Flint',
            'Grand Rapids-Wyoming-Kentwood', 'Jackson', 'Kalamazoo-Portage',
            'Lansing-East Lansing', 'Midland', 'Monroe', 'Muskegon-Norton Shores',
            'Niles', 'Saginaw', 'Traverse City',

            # Minnesota
            'Duluth', 'Mankato', 'Minneapolis-St. Paul-Bloomington', 'Rochester', 'St. Cloud',

            # Mississippi
            'Gulfport-Biloxi', 'Hattiesburg', 'Jackson',

            # Missouri
            'Cape Girardeau', 'Columbia', 'Jefferson City', 'Joplin', 'Kansas City',
            'St. Joseph', 'St. Louis', 'Springfield',

            # Montana
            'Billings', 'Bozeman', 'Great Falls', 'Helena', 'Missoula',

            # Nebraska
            'Grand Island', 'Lincoln', 'Omaha',

            # Nevada
            'Carson City', 'Las Vegas-Henderson-North Las Vegas', 'Reno',

            # New Hampshire
            'Manchester-Nashua',

            # New Jersey
            'Atlantic City-Hammonton', 'Trenton-Princeton', 'Vineland',

            # New Mexico
            'Albuquerque', 'Farmington', 'Las Cruces', 'Santa Fe',

            # New York
            'Albany-Schenectady-Troy', 'Binghamton', 'Buffalo-Cheektowaga', 'Elmira',
            'Glens Falls', 'Ithaca', 'Kingston', 'Kiryas Joel-Poughkeepsie-Newburgh',
            'New York-Newark-Jersey City', 'Rochester', 'Syracuse', 'Utica-Rome',
            'Watertown-Fort Drum',

            # North Carolina
            'Asheville', 'Burlington', 'Charlotte-Concord-Gastonia', 'Durham-Chapel Hill',
            'Fayetteville', 'Goldsboro', 'Greensboro-High Point', 'Greenville',
            'Hickory-Lenoir-Morganton', 'Jacksonville', 'Pinehurst-Southern Pines',
            'Raleigh-Cary', 'Rocky Mount', 'Wilmington', 'Winston-Salem',

            # North Dakota
            'Bismarck', 'Fargo', 'Grand Forks', 'Minot',

            # Ohio
            'Akron', 'Canton-Massillon', 'Cincinnati', 'Cleveland', 'Columbus',
            'Dayton-Kettering-Beavercreek', 'Lima', 'Mansfield', 'Sandusky',
            'Springfield', 'Toledo', 'Weirton-Steubenville', 'Youngstown-Warren',

            # Oklahoma
            'Enid', 'Lawton', 'Oklahoma City', 'Tulsa',

            # Oregon
            'Albany', 'Bend', 'Corvallis', 'Eugene-Springfield', 'Grants Pass',
            'Medford', 'Portland-Vancouver-Hillsboro', 'Salem',

            # Pennsylvania
            'Allentown-Bethlehem-Easton', 'Altoona', 'Chambersburg', 'Erie', 'Gettysburg',
            'Harrisburg-Carlisle', 'Johnstown', 'Lancaster', 'Lebanon',
            'Philadelphia-Camden-Wilmington', 'Pittsburgh', 'Reading',
            'Scranton--Wilkes-Barre', 'State College', 'Williamsport', 'York-Hanover',

            # Rhode Island
            'Providence-Warwick',

            # South Carolina
            'Charleston-North Charleston', 'Columbia', 'Florence', 'Greenville-Anderson-Greer',
            'Hilton Head Island-Bluffton-Port Royal', 'Myrtle Beach-Conway-North Myrtle Beach',
            'Spartanburg', 'Sumter',

            # South Dakota
            'Rapid City', 'Sioux Falls',

            # Tennessee
            'Chattanooga', 'Clarksville', 'Cleveland', 'Jackson', 'Johnson City',
            'Kingsport-Bristol', 'Knoxville', 'Memphis', 'Morristown',
            'Nashville-Davidson--Murfreesboro--Franklin',

            # Texas
            'Abilene', 'Amarillo', 'Austin-Round Rock-San Marcos', 'Beaumont-Port Arthur',
            'Brownsville-Harlingen', 'College Station-Bryan', 'Corpus Christi',
            'Dallas-Fort Worth-Arlington', 'Eagle Pass', 'El Paso',
            'Houston-Pasadena-The Woodlands', 'Killeen-Temple', 'Laredo', 'Longview',
            'Lubbock', 'McAllen-Edinburg-Mission', 'Midland', 'Odessa', 'San Angelo',
            'San Antonio-New Braunfels', 'Sherman-Denison', 'Texarkana', 'Tyler',
            'Victoria', 'Waco', 'Wichita Falls',

            # Utah
            'Logan', 'Ogden', 'Provo-Orem-Lehi', 'St. George', 'Salt Lake City-Murray',

            # Vermont
            'Burlington-South Burlington',

            # Virginia
            'Blacksburg-Christiansburg-Radford', 'Charlottesville', 'Harrisonburg',
            'Lynchburg', 'Richmond', 'Roanoke', 'Staunton-Stuarts Draft',
            'Virginia Beach-Chesapeake-Norfolk', 'Winchester',

            # Washington
            'Bellingham', 'Bremerton-Silverdale-Port Orchard', 'Kennewick-Richland',
            'Longview-Kelso', 'Mount Vernon-Anacortes', 'Olympia-Lacey-Tumwater',
            'Seattle-Tacoma-Bellevue', 'Spokane-Spokane Valley', 'Walla Walla',
            'Wenatchee-East Wenatchee', 'Yakima',

            # West Virginia
            'Beckley', 'Charleston', 'Huntington-Ashland', 'Morgantown',
            'Parkersburg-Vienna', 'Wheeling',

            # Wisconsin
            'Appleton', 'Eau Claire', 'Fond du Lac', 'Green Bay', 'Janesville-Beloit',
            'Kenosha', 'La Crosse-Onalaska', 'Madison', 'Milwaukee-Waukesha',
            'Oshkosh-Neenah', 'Racine-Mount Pleasant', 'Sheboygan', 'Wausau',

            # Wyoming
            'Casper', 'Cheyenne',

            # Puerto Rico
            'Aguadilla', 'Arecibo', 'Guayama', 'Mayaguez', 'Ponce', 'San Juan-Bayamon-Caguas',

            # Virgin Islands
            'Virgin Islands',
        ]

    def _get_metro_divisions(self) -> List[str]:
        """Return list of metropolitan divisions from tables 2 and 4"""
        return [
            # California - Los Angeles
            'Anaheim-Santa Ana-Irvine',
            'Los Angeles-Long Beach-Glendale',

            # California - San Francisco
            'Oakland-Fremont-Berkeley',
            'San Francisco-San Mateo-Redwood City',
            'San Rafael',

            # District of Columbia - Washington
            'Arlington-Alexandria-Reston',
            'Frederick-Gaithersburg-Bethesda',
            'Washington',

            # Florida - Miami
            'Fort Lauderdale-Pompano Beach-Sunrise',
            'Miami-Miami Beach-Kendall',
            'West Palm Beach-Boca Raton-Delray Beach',

            # Florida - Tampa
            'St. Petersburg-Clearwater-Largo',
            'Tampa',

            # Georgia - Atlanta
            'Atlanta-Sandy Springs-Roswell',
            'Marietta',

            # Illinois - Chicago
            'Chicago-Naperville-Schaumburg',
            'Elgin',
            'Lake County',
            'Lake County-Porter County-Jasper County',

            # Massachusetts - Boston
            'Boston',
            'Cambridge-Newton-Framingham',
            'Rockingham County-Strafford County',

            # Michigan - Detroit
            'Detroit-Dearborn-Livonia',
            'Warren-Troy-Farmington Hills',

            # New York - New York
            'Lakewood-New Brunswick',
            'Nassau County-Suffolk County',
            'Newark',
            'New York-Jersey City-White Plains',

            # Pennsylvania - Philadelphia
            'Camden',
            'Montgomery County-Bucks County-Chester County',
            'Philadelphia',
            'Wilmington',

            # Texas - Dallas
            'Dallas-Plano-Irving',
            'Fort Worth-Arlington-Grapevine',

            # Washington - Seattle
            'Everett',
            'Seattle-Bellevue-Kent',
            'Tacoma-Lakewood',
        ]

    def get_sector_keywords(self) -> Dict[str, List[str]]:
        """Extract METRO keywords from sector patterns"""
        keywords = {}
        for sector_name, pattern in BLS_SECTOR_PATTERNS.items():
            if 'metro' in pattern['keywords']:
                keywords[sector_name] = pattern['keywords']['metro']
        return keywords

    def get_measurement_types(self) -> Dict[str, Dict]:
        """Return METRO measurement type configurations"""
        return MEASUREMENT_TYPES.get('metro', {})

    def link_temporal_sequences(self) -> int:
        """Link temporal sequences for METRO measurements"""
        total_links = 0

        for measurement_name, config in self.get_measurement_types().items():
            links = self._link_sequences_for_measurement(
                measurement_type=config['class'],
                category_property=config['category_property'],
                month_property=config['month_property'],
                year_property=config['year_property'],
                measurement_name=measurement_name
            )
            total_links += links
            self.stats['temporal_sequences'] += links

        return total_links

    def _link_sequences_for_measurement(self, measurement_type, category_property,
                                        month_property, year_property, measurement_name) -> int:
        """Helper to link temporal sequences for a specific measurement type"""
        query = f"""
        SELECT ?measurement ?category ?month ?year WHERE {{
            ?measurement a <{measurement_type}> ;
                        <{category_property}> ?category ;
                        <{month_property}> ?month ;
                        <{year_property}> ?year .
        }}
        """

        results = list(self.graph.query(query))
        if not results:
            return 0

        # Group by category (metropolitan area or division)
        by_category = {}
        for row in results:
            category = row.category
            if category not in by_category:
                by_category[category] = []

            by_category[category].append({
                'measurement': row.measurement,
                'month': get_month_name(row.month),
                'year': get_year_value(row.year)
            })

        # Sort and link
        links_added = 0
        for category, measurements in by_category.items():
            measurements.sort(key=lambda x: (
                int(x['year']),
                self.month_order.index(x['month'])
            ))

            for i in range(len(measurements) - 1):
                current = measurements[i]['measurement']
                next_measurement = measurements[i + 1]['measurement']

                self.graph.add((
                    current,
                    BLS_ENRICHMENT.precedes,
                    next_measurement
                ))
                links_added += 1

        if links_added > 0:
            logger.info(f"  METRO.{measurement_name}: {links_added} sequence links")

        return links_added
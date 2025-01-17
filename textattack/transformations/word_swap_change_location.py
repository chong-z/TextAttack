import numpy as np
from textattack.transformations import Transformation
from flair.data import Sentence
from flair.models import SequenceTagger


def cluster_idx(idx_ls):
    """Given a list of idx, return a list that contains sub-lists of adjacent
    idx."""

    if len(idx_ls) < 2:
        return [[i] for i in idx_ls]
    else:
        output = [[idx_ls[0]]]
        prev = idx_ls[0]
        list_pos = 0

        for idx in idx_ls[1:]:
            if idx - 1 == prev:
                output[list_pos].append(idx)
            else:
                output.append([idx])
                list_pos += 1
            prev = idx
        return output


def idx_to_words(ls, words):
    """Given a list generated from cluster_idx, return a list that contains
    sub-list (the first element being the idx, and the second element being the
    words corresponding to the idx)"""

    output = []
    for sub_ls in ls:
        word = words[sub_ls[0]]
        for idx in sub_ls[1:]:
            word = " ".join([word, words[idx]])
        output.append([sub_ls, word])
    return output


class WordSwapChangeLocation(Transformation):
    def __init__(self, n=3, confidence_score=0.7, **kwargs):
        """Transformation that changes recognized locations of a sentence to
        another location that is given in the location map.

        :param n: Number of new locations to generate
        :param confidence_score: Location will only be changed if it's above the confidence score
        """
        super().__init__(**kwargs)
        self.n = n
        self.confidence_score = confidence_score

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        # TODO: move ner recognition to AttackedText
        # really want to silent this line:
        tagger = SequenceTagger.load("ner")
        sentence = Sentence(current_text.text)
        tagger.predict(sentence)
        location_idx = []

        # pre-screen for actual locations, using flair
        # summarize location idx into a list (location_idx)
        for token in sentence:
            tag = token.get_tag("ner")
            if (
                "LOC" in tag.value
                and tag.score > self.confidence_score
                and (token.idx - 1) in indices_to_modify
            ):
                location_idx.append(token.idx - 1)

        # Combine location idx and words to a list ([0] is idx, [1] is location name)
        # For example, [1,2] to [ [1,2] , ["New York"] ]
        location_idx = cluster_idx(location_idx)
        location_words = idx_to_words(location_idx, words)

        transformed_texts = []
        for location in location_words:
            idx = location[0]
            word = location[1]
            replacement_words = self._get_new_location(word)
            for r in replacement_words:
                if r == word:
                    continue
                text = current_text

                # if original location is more than a single word, remain only the starting word
                if len(idx) > 1:
                    index = idx[1]
                    for i in idx[1:]:
                        text = text.delete_word_at_index(index)

                # replace the starting word with new location
                text = text.replace_word_at_index(idx[0], r)

                transformed_texts.append(text)
        return transformed_texts

    def _get_new_location(self, word):
        """Return a list of new locations, with the choice of country,
        nationality, and city."""
        if word in LOCATION["country"]:
            return np.random.choice(LOCATION["country"], self.n)
        elif word in LOCATION["nationality"]:
            return np.random.choice(LOCATION["nationality"], self.n)
        elif word in LOCATION["city"]:
            return np.random.choice(LOCATION["city"], self.n)
        return []


LOCATION = {
    "country": [
        "China",
        "India",
        "United States",
        "Indonesia",
        "Pakistan",
        "Brazil",
        "Nigeria",
        "Bangladesh",
        "Russian Federation",
        "Japan",
        "Mexico",
        "Ethiopia",
        "Philippines",
        "Egypt",
        "Vietnam",
        "Germany",
        "Turkey",
        "Iran",
        "Thailand",
        "France",
        "United Kingdom",
        "Italy",
        "South Africa",
        "Tanzania",
        "Myanmar",
        "Kenya",
        "Colombia",
        "Spain",
        "Ukraine",
        "Argentina",
        "Uganda",
        "Algeria",
        "Sudan",
        "Iraq",
        "Poland",
        "Afghanistan",
        "Canada",
        "Morocco",
        "Saudi Arabia",
        "Uzbekistan",
        "Peru",
        "Malaysia",
        "Angola",
        "Ghana",
        "Mozambique",
        "Venezuela",
        "Yemen",
        "Nepal",
        "Madagascar",
        "Korea",
        "Cameroon",
        "Australia",
        "Niger",
        "Sri Lanka",
        "Burkina Faso",
        "Romania",
        "Mali",
        "Chile",
        "Kazakhstan",
        "Malawi",
        "Zambia",
        "Guatemala",
        "Netherlands",
        "Ecuador",
        "Syrian Arab Republic",
        "Cambodia",
        "Senegal",
        "Chad",
        "Somalia",
        "Zimbabwe",
        "Guinea",
        "Rwanda",
        "Tunisia",
        "Benin",
        "Belgium",
        "Bolivia",
        "Cuba",
        "Burundi",
        "Haiti",
        "South Sudan",
        "Greece",
        "Dominican Republic",
        "Czech Republic",
        "Portugal",
        "Sweden",
        "Jordan",
        "Azerbaijan",
        "Hungary",
        "United Arab Emirates",
        "Honduras",
        "Belarus",
        "Tajikistan",
        "Israel",
        "Austria",
        "Papua New Guinea",
        "Switzerland",
        "Togo",
        "Sierra Leone",
        "Hong Kong SAR",
        "Lao PDR",
        "Bulgaria",
        "Serbia",
        "Paraguay",
        "Lebanon",
        "Libya",
        "Nicaragua",
        "El Salvador",
        "Kyrgyz Republic",
        "Turkmenistan",
        "Denmark",
        "Singapore",
        "Finland",
        "Slovak Republic",
        "Norway",
        "Congo",
        "Costa Rica",
        "New Zealand",
        "Ireland",
        "Oman",
        "Liberia",
        "Central African Republic",
        "West Bank and Gaza",
        "Mauritania",
        "Panama",
        "Kuwait",
        "Croatia",
        "Georgia",
        "Moldova",
        "Uruguay",
        "Bosnia and Herzegovina",
        "Eritrea",
        "Puerto Rico",
        "Mongolia",
        "Armenia",
        "Jamaica",
        "Albania",
        "Lithuania",
        "Qatar",
        "Namibia",
        "Gambia",
        "Botswana",
        "Gabon",
        "Lesotho",
        "North Macedonia",
        "Slovenia",
        "Latvia",
        "Guinea-Bissau",
        "Kosovo",
        "Bahrain",
        "Trinidad and Tobago",
        "Estonia",
        "Equatorial Guinea",
        "Timor-Leste",
        "Mauritius",
        "Cyprus",
        "Eswatini",
        "Djibouti",
        "Fiji",
        "Comoros",
        "Guyana",
        "Bhutan",
        "Solomon Islands",
        "Macao SAR",
        "Montenegro",
        "Luxembourg",
        "Suriname",
        "Cabo Verde",
        "Maldives",
        "Malta",
        "Brunei Darussalam",
        "Bahamas",
        "Belize",
        "Iceland",
        "Vanuatu",
        "Barbados",
        "New Caledonia",
        "French Polynesia",
        "Samoa",
        "St. Lucia",
        "Channel Islands",
        "Guam",
        "Kiribati",
        "Micronesia",
        "Grenada",
        "St. Vincent and the Grenadines",
        "Virgin Islands (U.S.)",
        "Aruba",
        "Tonga",
        "Seychelles",
        "Antigua and Barbuda",
        "Isle of Man",
        "Andorra",
        "Dominica",
        "Cayman Islands",
        "Bermuda",
        "Marshall Islands",
        "Northern Mariana Islands",
        "Greenland",
        "American Samoa",
        "St. Kitts and Nevis",
        "Faroe Islands",
        "Sint Maarten (Dutch part)",
        "Monaco",
        "Liechtenstein",
        "Turks and Caicos Islands",
        "St. Martin (French part)",
        "San Marino",
        "Gibraltar",
        "British Virgin Islands",
        "Palau",
        "Nauru",
        "Tuvalu",
        "C\u00f4te d'Ivoire",
        "Cura\u00e7ao",
        "S\u00e3o Tom\u00e9 and Principe",
    ],
    "nationality": [
        "Chinese",
        "Indian",
        "American",
        "Indonesian",
        "Pakistani",
        "Brazilian",
        "Nigerian",
        "Bangladeshi",
        "Russian",
        "Japanese",
        "Mexican",
        "Ethiopian",
        "Philippine",
        "Egyptian",
        "Vietnamese",
        "German",
        "Turkish",
        "Iranian",
        "Thai",
        "French",
        "British",
        "Italian",
        "South African",
        "Tanzanian",
        "Burmese",
        "Kenyan",
        "Colombian",
        "Spanish",
        "Ukrainian",
        "Argentine",
        "Ugandan",
        "Algerian",
        "Sudanese",
        "Iraqi",
        "Polish",
        "Afghan",
        "Canadian",
        "Moroccan",
        "Saudi",
        "Uzbekistani",
        "Peruvian",
        "Malaysian",
        "Angolan",
        "Ghanaian",
        "Mozambican",
        "Venezuelan",
        "Yemeni",
        "Nepali",
        "Malagasy",
        "South Korean",
        "Cameroonian",
        "Australian",
        "Nigerien",
        "Sri Lankan",
        "Burkinab\u00e9",
        "Romanian",
        "Malian",
        "Chilean",
        "Kazakhstani",
        "Malawian",
        "Zambian",
        "Guatemalan",
        "Dutch",
        "Ecuadorian",
        "Syrian",
        "Cambodian",
        "Senegalese",
        "Chadian",
        "Somali",
        "Zimbabwean",
        "Guinean",
        "Rwandan",
        "Tunisian",
        "Beninese",
        "Belgian",
        "Bolivian",
        "Cuban",
        "Burundian",
        "Haitian",
        "South Sudanese",
        "Greek",
        "Dominican",
        "Czech",
        "Portuguese",
        "Swedish",
        "Jordanian",
        "Azerbaijani",
        "Hungarian",
        "Emirati",
        "Honduran",
        "Belarusian",
        "Tajikistani",
        "Israeli",
        "Austrian",
        "Papua New Guinean",
        "Swiss",
        "Togolese",
        "Sierra Leonean",
        "Hong Kong",
        "Lao",
        "Bulgarian",
        "Serbian",
        "Paraguayan",
        "Lebanese",
        "Libyan",
        "Nicaraguan",
        "Salvadoran",
        "Kyrgyzstani",
        "Turkmen",
        "Danish",
        "Singaporean",
        "Finnish",
        "Slovak",
        "Norwegian",
        "Congolese",
        "Costa Rican",
        "New Zealand",
        "Irish",
        "Omani",
        "Liberian",
        "Central African",
        "Palestinian",
        "Mauritanian",
        "Panamanian",
        "Kuwaiti",
        "Croatian",
        "Georgian",
        "Moldovan",
        "Uruguayan",
        "Bosnian or Herzegovinian",
        "Eritrean",
        "Puerto Rican",
        "Mongolian",
        "Armenian",
        "Jamaican",
        "Albanian",
        "Lithuanian",
        "Qatari",
        "Namibian",
        "Gambian",
        "Motswana",
        "Gabonese",
        "Basotho",
        "Macedonian",
        "Slovenian",
        "Latvian",
        "Bissau-Guinean",
        "from Kosovo",
        "Bahraini",
        "Trinidadian or Tobagonian",
        "Estonian",
        "Equatorial Guinean",
        "Timorese",
        "Mauritian",
        "Cypriot",
        "Swazi",
        "Djiboutian",
        "Fijian",
        "Comoran",
        "Guyanese",
        "Bhutanese",
        "Solomon Island",
        "Macanese",
        "Montenegrin",
        "Luxembourg",
        "Surinamese",
        "Cabo Verdean",
        "Maldivian",
        "Maltese",
        "Bruneian",
        "Bahamian",
        "Belizean",
        "Icelandic",
        "Ni-Vanuatu",
        "Barbadian",
        "New Caledonian",
        "French Polynesian",
        "Samoan",
        "Saint Lucian",
        "from Channel Islands",
        "Guamanian",
        "I-Kiribati",
        "Micronesian",
        "Grenadian",
        "Saint Vincentian",
        "U.S. Virgin Island",
        "Aruban",
        "Tongan",
        "Seychellois",
        "Antiguan or Barbudan",
        "Manx",
        "Andorran",
        "Dominican",
        "Caymanian",
        "Bermudian",
        "Marshallese",
        "Northern Marianan",
        "Greenlandic",
        "American Samoan",
        "Kittitian or Nevisian",
        "Faroese",
        "Sint Maarten",
        "Mon\u00e9gasque",
        "Liechtenstein",
        "Turks and Caicos Island",
        "Saint-Martinoise",
        "Sammarinese",
        "Gibraltar",
        "British Virgin Island",
        "Palauan",
        "Nauruan",
        "Tuvaluan",
        "Ivorian",
        "Cura\u00e7aoan",
        "S\u00e3o Tom\u00e9an",
    ],
    "city": [
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Philadelphia",
        "Phoenix",
        "San Antonio",
        "San Diego",
        "Dallas",
        "San Jose",
        "Austin",
        "Indianapolis",
        "Jacksonville",
        "San Francisco",
        "Columbus",
        "Charlotte",
        "Fort Worth",
        "Detroit",
        "El Paso",
        "Memphis",
        "Seattle",
        "Denver",
        "Washington",
        "Boston",
        "Nashville-Davidson",
        "Baltimore",
        "Oklahoma City",
        "Louisville/Jefferson County",
        "Portland",
        "Las Vegas",
        "Milwaukee",
        "Albuquerque",
        "Tucson",
        "Fresno",
        "Sacramento",
        "Long Beach",
        "Kansas City",
        "Mesa",
        "Virginia Beach",
        "Atlanta",
        "Colorado Springs",
        "Omaha",
        "Raleigh",
        "Miami",
        "Oakland",
        "Minneapolis",
        "Tulsa",
        "Cleveland",
        "Wichita",
        "Arlington",
        "New Orleans",
        "Bakersfield",
        "Tampa",
        "Honolulu",
        "Aurora",
        "Anaheim",
        "Santa Ana",
        "St. Louis",
        "Riverside",
        "Corpus Christi",
        "Lexington-Fayette",
        "Pittsburgh",
        "Anchorage",
        "Stockton",
        "Cincinnati",
        "St. Paul",
        "Toledo",
        "Greensboro",
        "Newark",
        "Plano",
        "Henderson",
        "Lincoln",
        "Buffalo",
        "Jersey City",
        "Chula Vista",
        "Fort Wayne",
        "Orlando",
        "St. Petersburg",
        "Chandler",
        "Laredo",
        "Norfolk",
        "Durham",
        "Madison",
        "Lubbock",
        "Irvine",
        "Winston-Salem",
        "Glendale",
        "Garland",
        "Hialeah",
        "Reno",
        "Chesapeake",
        "Gilbert",
        "Baton Rouge",
        "Irving",
        "Scottsdale",
        "North Las Vegas",
        "Fremont",
        "Boise City",
        "Richmond",
        "San Bernardino",
        "Birmingham",
        "Spokane",
        "Rochester",
        "Des Moines",
        "Modesto",
        "Fayetteville",
        "Tacoma",
        "Oxnard",
        "Fontana",
        "Columbus",
        "Montgomery",
        "Moreno Valley",
        "Shreveport",
        "Aurora",
        "Yonkers",
        "Akron",
        "Huntington Beach",
        "Little Rock",
        "Augusta-Richmond County",
        "Amarillo",
        "Glendale",
        "Mobile",
        "Grand Rapids",
        "Salt Lake City",
        "Tallahassee",
        "Huntsville",
        "Grand Prairie",
        "Knoxville",
        "Worcester",
        "Newport News",
        "Brownsville",
        "Overland Park",
        "Santa Clarita",
        "Providence",
        "Garden Grove",
        "Chattanooga",
        "Oceanside",
        "Jackson",
        "Fort Lauderdale",
        "Santa Rosa",
        "Rancho Cucamonga",
        "Port St. Lucie",
        "Tempe",
        "Ontario",
        "Vancouver",
        "Cape Coral",
        "Sioux Falls",
        "Springfield",
        "Peoria",
        "Pembroke Pines",
        "Elk Grove",
        "Salem",
        "Lancaster",
        "Corona",
        "Eugene",
        "Palmdale",
        "Salinas",
        "Springfield",
        "Pasadena",
        "Fort Collins",
        "Hayward",
        "Pomona",
        "Cary",
        "Rockford",
        "Alexandria",
        "Escondido",
        "McKinney",
        "Kansas City",
        "Joliet",
        "Sunnyvale",
        "Torrance",
        "Bridgeport",
        "Lakewood",
        "Hollywood",
        "Paterson",
        "Naperville",
        "Syracuse",
        "Mesquite",
        "Dayton",
        "Savannah",
        "Clarksville",
        "Orange",
        "Pasadena",
        "Fullerton",
        "Killeen",
        "Frisco",
        "Hampton",
        "McAllen",
        "Warren",
        "Bellevue",
        "West Valley City",
        "Columbia",
        "Olathe",
        "Sterling Heights",
        "New Haven",
        "Miramar",
        "Waco",
        "Thousand Oaks",
        "Cedar Rapids",
        "Charleston",
        "Visalia",
        "Topeka",
        "Elizabeth",
        "Gainesville",
        "Thornton",
        "Roseville",
        "Carrollton",
        "Coral Springs",
        "Stamford",
        "Simi Valley",
        "Concord",
        "Hartford",
        "Kent",
        "Lafayette",
        "Midland",
        "Surprise",
        "Denton",
        "Victorville",
        "Evansville",
        "Santa Clara",
        "Abilene",
        "Athens-Clarke County",
        "Vallejo",
        "Allentown",
        "Norman",
        "Beaumont",
        "Independence",
        "Murfreesboro",
        "Ann Arbor",
        "Springfield",
        "Berkeley",
        "Peoria",
        "Provo",
        "El Monte",
        "Columbia",
        "Lansing",
        "Fargo",
        "Downey",
        "Costa Mesa",
        "Wilmington",
        "Arvada",
        "Inglewood",
        "Miami Gardens",
        "Carlsbad",
        "Westminster",
        "Rochester",
        "Odessa",
        "Manchester",
        "Elgin",
        "West Jordan",
        "Round Rock",
        "Clearwater",
        "Waterbury",
        "Gresham",
        "Fairfield",
        "Billings",
        "Lowell",
        "San Buenaventura (Ventura)",
        "Pueblo",
        "High Point",
        "West Covina",
        "Richmond",
        "Murrieta",
        "Cambridge",
        "Antioch",
        "Temecula",
        "Norwalk",
        "Centennial",
        "Everett",
        "Palm Bay",
        "Wichita Falls",
        "Green Bay",
        "Daly City",
        "Burbank",
        "Richardson",
        "Pompano Beach",
        "North Charleston",
        "Broken Arrow",
        "Boulder",
        "West Palm Beach",
        "Santa Maria",
        "El Cajon",
        "Davenport",
        "Rialto",
        "Las Cruces",
        "San Mateo",
        "Lewisville",
        "South Bend",
        "Lakeland",
        "Erie",
        "Tyler",
        "Pearland",
        "College Station",
        "Kenosha",
        "Sandy Springs",
        "Clovis",
        "Flint",
        "Roanoke",
        "Albany",
        "Jurupa Valley",
        "Compton",
        "San Angelo",
        "Hillsboro",
        "Lawton",
        "Renton",
        "Vista",
        "Davie",
        "Greeley",
        "Mission Viejo",
        "Portsmouth",
        "Dearborn",
        "South Gate",
        "Tuscaloosa",
        "Livonia",
        "New Bedford",
        "Vacaville",
        "Brockton",
        "Roswell",
        "Beaverton",
        "Quincy",
        "Sparks",
        "Yakima",
        "Lee's Summit",
        "Federal Way",
        "Carson",
        "Santa Monica",
        "Hesperia",
        "Allen",
        "Rio Rancho",
        "Yuma",
        "Westminster",
        "Orem",
        "Lynn",
        "Redding",
        "Spokane Valley",
        "Miami Beach",
        "League City",
        "Lawrence",
        "Santa Barbara",
        "Plantation",
        "Sandy",
        "Sunrise",
        "Macon",
        "Longmont",
        "Boca Raton",
        "San Marcos",
        "Greenville",
        "Waukegan",
        "Fall River",
        "Chico",
        "Newton",
        "San Leandro",
        "Reading",
        "Norwalk",
        "Fort Smith",
        "Newport Beach",
        "Asheville",
        "Nashua",
        "Edmond",
        "Whittier",
        "Nampa",
        "Bloomington",
        "Deltona",
        "Hawthorne",
        "Duluth",
        "Carmel",
        "Suffolk",
        "Clifton",
        "Citrus Heights",
        "Livermore",
        "Tracy",
        "Alhambra",
        "Kirkland",
        "Trenton",
        "Ogden",
        "Hoover",
        "Cicero",
        "Fishers",
        "Sugar Land",
        "Danbury",
        "Meridian",
        "Indio",
        "Concord",
        "Menifee",
        "Champaign",
        "Buena Park",
        "Troy",
        "O'Fallon",
        "Johns Creek",
        "Bellingham",
        "Westland",
        "Bloomington",
        "Sioux City",
        "Warwick",
        "Hemet",
        "Longview",
        "Farmington Hills",
        "Bend",
        "Lakewood",
        "Merced",
        "Mission",
        "Chino",
        "Redwood City",
        "Edinburg",
        "Cranston",
        "Parma",
        "New Rochelle",
        "Lake Forest",
        "Napa",
        "Hammond",
        "Fayetteville",
        "Bloomington",
        "Avondale",
        "Somerville",
        "Palm Coast",
        "Bryan",
        "Gary",
        "Largo",
        "Brooklyn Park",
        "Tustin",
        "Racine",
        "Deerfield Beach",
        "Lynchburg",
        "Mountain View",
        "Medford",
        "Lawrence",
        "Bellflower",
        "Melbourne",
        "St. Joseph",
        "Camden",
        "St. George",
        "Kennewick",
        "Baldwin Park",
        "Chino Hills",
        "Alameda",
        "Albany",
        "Arlington Heights",
        "Scranton",
        "Evanston",
        "Kalamazoo",
        "Baytown",
        "Upland",
        "Springdale",
        "Bethlehem",
        "Schaumburg",
        "Mount Pleasant",
        "Auburn",
        "Decatur",
        "San Ramon",
        "Pleasanton",
        "Wyoming",
        "Lake Charles",
        "Plymouth",
        "Bolingbrook",
        "Pharr",
        "Appleton",
        "Gastonia",
        "Folsom",
        "Southfield",
        "Rochester Hills",
        "New Britain",
        "Goodyear",
        "Canton",
        "Warner Robins",
        "Union City",
        "Perris",
        "Manteca",
        "Iowa City",
        "Jonesboro",
        "Wilmington",
        "Lynwood",
        "Loveland",
        "Pawtucket",
        "Boynton Beach",
        "Waukesha",
        "Gulfport",
        "Apple Valley",
        "Passaic",
        "Rapid City",
        "Layton",
        "Lafayette",
        "Turlock",
        "Muncie",
        "Temple",
        "Missouri City",
        "Redlands",
        "Santa Fe",
        "Lauderhill",
        "Milpitas",
        "Palatine",
        "Missoula",
        "Rock Hill",
        "Jacksonville",
        "Franklin",
        "Flagstaff",
        "Flower Mound",
        "Weston",
        "Waterloo",
        "Union City",
        "Mount Vernon",
        "Fort Myers",
        "Dothan",
        "Rancho Cordova",
        "Redondo Beach",
        "Jackson",
        "Pasco",
        "St. Charles",
        "Eau Claire",
        "North Richland Hills",
        "Bismarck",
        "Yorba Linda",
        "Kenner",
        "Walnut Creek",
        "Frederick",
        "Oshkosh",
        "Pittsburg",
        "Palo Alto",
        "Bossier City",
        "Portland",
        "St. Cloud",
        "Davis",
        "South San Francisco",
        "Camarillo",
        "North Little Rock",
        "Schenectady",
        "Gaithersburg",
        "Harlingen",
        "Woodbury",
        "Eagan",
        "Yuba City",
        "Maple Grove",
        "Youngstown",
        "Skokie",
        "Kissimmee",
        "Johnson City",
        "Victoria",
        "San Clemente",
        "Bayonne",
        "Laguna Niguel",
        "East Orange",
        "Shawnee",
        "Homestead",
        "Rockville",
        "Delray Beach",
        "Janesville",
        "Conway",
        "Pico Rivera",
        "Lorain",
        "Montebello",
        "Lodi",
        "New Braunfels",
        "Marysville",
        "Tamarac",
        "Madera",
        "Conroe",
        "Santa Cruz",
        "Eden Prairie",
        "Cheyenne",
        "Daytona Beach",
        "Alpharetta",
        "Hamilton",
        "Waltham",
        "Coon Rapids",
        "Haverhill",
        "Council Bluffs",
        "Taylor",
        "Utica",
        "Ames",
        "La Habra",
        "Encinitas",
        "Bowling Green",
        "Burnsville",
        "Greenville",
        "West Des Moines",
        "Cedar Park",
        "Tulare",
        "Monterey Park",
        "Vineland",
        "Terre Haute",
        "North Miami",
        "Mansfield",
        "West Allis",
        "Bristol",
        "Taylorsville",
        "Malden",
        "Meriden",
        "Blaine",
        "Wellington",
        "Cupertino",
        "Springfield",
        "Rogers",
        "St. Clair Shores",
        "Gardena",
        "Pontiac",
        "National City",
        "Grand Junction",
        "Rocklin",
        "Chapel Hill",
        "Casper",
        "Broomfield",
        "Petaluma",
        "South Jordan",
        "Springfield",
        "Great Falls",
        "Lancaster",
        "North Port",
        "Lakewood",
        "Marietta",
        "San Rafael",
        "Royal Oak",
        "Des Plaines",
        "Huntington Park",
        "La Mesa",
        "Orland Park",
        "Auburn",
        "Lakeville",
        "Owensboro",
        "Moore",
        "Jupiter",
        "Idaho Falls",
        "Dubuque",
        "Bartlett",
        "Rowlett",
        "Novi",
        "White Plains",
        "Arcadia",
        "Redmond",
        "Lake Elsinore",
        "Ocala",
        "Tinley Park",
        "Port Orange",
        "Medford",
        "Oak Lawn",
        "Rocky Mount",
        "Kokomo",
        "Coconut Creek",
        "Bowie",
        "Berwyn",
        "Midwest City",
        "Fountain Valley",
        "Buckeye",
        "Dearborn Heights",
        "Woodland",
        "Noblesville",
        "Valdosta",
        "Diamond Bar",
        "Manhattan",
        "Santee",
        "Taunton",
        "Sanford",
        "Kettering",
        "New Brunswick",
        "Decatur",
        "Chicopee",
        "Anderson",
        "Margate",
        "Weymouth Town",
        "Hempstead",
        "Corvallis",
        "Eastvale",
        "Porterville",
        "West Haven",
        "Brentwood",
        "Paramount",
        "Grand Forks",
        "Georgetown",
        "St. Peters",
        "Shoreline",
        "Mount Prospect",
        "Hanford",
        "Normal",
        "Rosemead",
        "Lehi",
        "Pocatello",
        "Highland",
        "Novato",
        "Port Arthur",
        "Carson City",
        "San Marcos",
        "Hendersonville",
        "Elyria",
        "Revere",
        "Pflugerville",
        "Greenwood",
        "Bellevue",
        "Wheaton",
        "Smyrna",
        "Sarasota",
        "Blue Springs",
        "Colton",
        "Euless",
        "Castle Rock",
        "Cathedral City",
        "Kingsport",
        "Lake Havasu City",
        "Pensacola",
        "Hoboken",
        "Yucaipa",
        "Watsonville",
        "Richland",
        "Delano",
        "Hoffman Estates",
        "Florissant",
        "Placentia",
        "West New York",
        "Dublin",
        "Oak Park",
        "Peabody",
        "Perth Amboy",
        "Battle Creek",
        "Bradenton",
        "Gilroy",
        "Milford",
        "Albany",
        "Ankeny",
        "La Crosse",
        "Burlington",
        "DeSoto",
        "Harrisonburg",
        "Minnetonka",
        "Elkhart",
        "Lakewood",
        "Glendora",
        "Southaven",
        "Charleston",
        "Joplin",
        "Enid",
        "Palm Beach Gardens",
        "Brookhaven",
        "Plainfield",
        "Grand Island",
        "Palm Desert",
        "Huntersville",
        "Tigard",
        "Lenexa",
        "Saginaw",
        "Kentwood",
        "Doral",
        "Apple Valley",
        "Grapevine",
        "Aliso Viejo",
        "Sammamish",
        "Casa Grande",
        "Pinellas Park",
        "Troy",
        "West Sacramento",
        "Burien",
        "Commerce City",
        "Monroe",
        "Cerritos",
        "Downers Grove",
        "Coral Gables",
        "Wilson",
        "Niagara Falls",
        "Poway",
        "Edina",
        "Cuyahoga Falls",
        "Rancho Santa Margarita",
        "Harrisburg",
        "Huntington",
        "La Mirada",
        "Cypress",
        "Caldwell",
        "Logan",
        "Galveston",
        "Sheboygan",
        "Middletown",
        "Murray",
        "Roswell",
        "Parker",
        "Bedford",
        "East Lansing",
        "Methuen",
        "Covina",
        "Alexandria",
        "Olympia",
        "Euclid",
        "Mishawaka",
        "Salina",
        "Azusa",
        "Newark",
        "Chesterfield",
        "Leesburg",
        "Dunwoody",
        "Hattiesburg",
        "Roseville",
        "Bonita Springs",
        "Portage",
        "St. Louis Park",
        "Collierville",
        "Middletown",
        "Stillwater",
        "East Providence",
        "Lawrence",
        "Wauwatosa",
        "Mentor",
        "Ceres",
        "Cedar Hill",
        "Mansfield",
        "Binghamton",
        "Coeur d'Alene",
        "San Luis Obispo",
        "Minot",
        "Palm Springs",
        "Pine Bluff",
        "Texas City",
        "Summerville",
        "Twin Falls",
        "Jeffersonville",
        "San Jacinto",
        "Madison",
        "Altoona",
        "Columbus",
        "Beavercreek",
        "Apopka",
        "Elmhurst",
        "Maricopa",
        "Farmington",
        "Glenview",
        "Cleveland Heights",
        "Draper",
        "Lincoln",
        "Sierra Vista",
        "Lacey",
        "Biloxi",
        "Strongsville",
        "Barnstable Town",
        "Wylie",
        "Sayreville",
        "Kannapolis",
        "Charlottesville",
        "Littleton",
        "Titusville",
        "Hackensack",
        "Newark",
        "Pittsfield",
        "York",
        "Lombard",
        "Attleboro",
        "DeKalb",
        "Blacksburg",
        "Dublin",
        "Haltom City",
        "Lompoc",
        "El Centro",
        "Danville",
        "Jefferson City",
        "Cutler Bay",
        "Oakland Park",
        "North Miami Beach",
        "Freeport",
        "Moline",
        "Coachella",
        "Fort Pierce",
        "Smyrna",
        "Bountiful",
        "Fond du Lac",
        "Everett",
        "Danville",
        "Keller",
        "Belleville",
        "Bell Gardens",
        "Cleveland",
        "North Lauderdale",
        "Fairfield",
        "Salem",
        "Rancho Palos Verdes",
        "San Bruno",
        "Concord",
        "Burlington",
        "Apex",
        "Midland",
        "Altamonte Springs",
        "Hutchinson",
        "Buffalo Grove",
        "Urbandale",
        "State College",
        "Urbana",
        "Plainfield",
        "Manassas",
        "Bartlett",
        "Kearny",
        "Oro Valley",
        "Findlay",
        "Rohnert Park",
        "Westfield",
        "Linden",
        "Sumter",
        "Wilkes-Barre",
        "Woonsocket",
        "Leominster",
        "Shelton",
        "Brea",
        "Covington",
        "Rockwall",
        "Meridian",
        "Riverton",
        "St. Cloud",
        "Quincy",
        "Morgan Hill",
        "Warren",
        "Edmonds",
        "Burleson",
        "Beverly",
        "Mankato",
        "Hagerstown",
        "Prescott",
        "Campbell",
        "Cedar Falls",
        "Beaumont",
        "La Puente",
        "Crystal Lake",
        "Fitchburg",
        "Carol Stream",
        "Hickory",
        "Streamwood",
        "Norwich",
        "Coppell",
        "San Gabriel",
        "Holyoke",
        "Bentonville",
        "Florence",
        "Peachtree Corners",
        "Brentwood",
        "Bozeman",
        "New Berlin",
        "Goose Creek",
        "Huntsville",
        "Prescott Valley",
        "Maplewood",
        "Romeoville",
        "Duncanville",
        "Atlantic City",
        "Clovis",
        "The Colony",
        "Culver City",
        "Marlborough",
        "Hilton Head Island",
        "Moorhead",
        "Calexico",
        "Bullhead City",
        "Germantown",
        "La Quinta",
        "Lancaster",
        "Wausau",
        "Sherman",
        "Ocoee",
        "Shakopee",
        "Woburn",
        "Bremerton",
        "Rock Island",
        "Muskogee",
        "Cape Girardeau",
        "Annapolis",
        "Greenacres",
        "Ormond Beach",
        "Hallandale Beach",
        "Stanton",
        "Puyallup",
        "Pacifica",
        "Hanover Park",
        "Hurst",
        "Lima",
        "Marana",
        "Carpentersville",
        "Oakley",
        "Huber Heights",
        "Lancaster",
        "Montclair",
        "Wheeling",
        "Brookfield",
        "Park Ridge",
        "Florence",
        "Roy",
        "Winter Garden",
        "Chelsea",
        "Valley Stream",
        "Spartanburg",
        "Lake Oswego",
        "Friendswood",
        "Westerville",
        "Northglenn",
        "Phenix City",
        "Grove City",
        "Texarkana",
        "Addison",
        "Dover",
        "Lincoln Park",
        "Calumet City",
        "Muskegon",
        "Aventura",
        "Martinez",
        "Greenfield",
        "Apache Junction",
        "Monrovia",
        "Weslaco",
        "Keizer",
        "Spanish Fork",
        "Beloit",
        "Panama City",
    ],
}

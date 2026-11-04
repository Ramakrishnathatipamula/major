import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Administrative Hierarchy Dictionary
ADMIN_HIERARCHY: Dict[str, Dict[str, List[str]]] = {
    "India": {
        "Telangana": [
            "Jagtial",
            "Medak",
            "Rangareddy",
            "Hyderabad",
            "Nizamabad",
            "Karimnagar",
            "Nalgonda",
            "Khammam",
            "Warangal",
            "Sangareddy",
            "Siddipet",
            "Kamareddy",
            "Peddapalli",
            "Suryapet",
            "Mahabubnagar",
            "Adilabad",
            "Mancherial",
            "Bhadradri Kothagudem",
            "Custom / Other District..."
        ],
        "Andhra Pradesh": [
            "Visakhapatnam",
            "Vijayawada",
            "Guntur",
            "Chittoor",
            "Kurnool",
            "Anantapur",
            "Nellore",
            "Kakinada",
            "Tirupati",
            "Prakasam",
            "Custom / Other District..."
        ],
        "Maharashtra": [
            "Mumbai",
            "Pune",
            "Nagpur",
            "Nashik",
            "Aurangabad",
            "Satara",
            "Solapur",
            "Kolhapur",
            "Ahmednagar",
            "Jalgaon",
            "Custom / Other District..."
        ],
        "Karnataka": [
            "Bengaluru Urban",
            "Mysuru",
            "Belagavi",
            "Dharwad",
            "Shimoga",
            "Hassan",
            "Tumakuru",
            "Mandya",
            "Kalaburagi",
            "Custom / Other District..."
        ],
        "Punjab": [
            "Amritsar",
            "Ludhiana",
            "Jalandhar",
            "Patiala",
            "Bathinda",
            "Hoshiarpur",
            "Firozpur",
            "Kapurthala",
            "Gurdaspur",
            "Custom / Other District..."
        ],
        "Custom / Other State...": ["Custom / Other District..."]
    },
    "United States": {
        "Minnesota": [
            "Hennepin",
            "Ramsey",
            "Dakota",
            "Stearns",
            "Clay",
            "Olmsted",
            "St. Louis",
            "Anoka",
            "Scott",
            "Washington",
            "Custom / Other District..."
        ],
        "Idaho": [
            "Ada",
            "Canyon",
            "Twin Falls",
            "Bingham",
            "Bonneville",
            "Bannock",
            "Kootenai",
            "Elmore",
            "Cassia",
            "Jerome",
            "Custom / Other District..."
        ],
        "California": [
            "Fresno",
            "Monterey",
            "Sacramento",
            "Kern",
            "San Joaquin",
            "Tulare",
            "Imperial",
            "Santa Barbara",
            "Ventura",
            "Riverside",
            "Custom / Other District..."
        ],
        "Custom / Other State...": ["Custom / Other District..."]
    },
    "Germany": {
        "Berlin": [
            "Berlin",
            "Mitte",
            "Pankow",
            "Spandau",
            "Neukölln",
            "Charlottenburg",
            "Tempelhof",
            "Steglitz",
            "Friedrichshain",
            "Custom / Other District..."
        ],
        "Bavaria": [
            "Munich",
            "Nuremberg",
            "Augsburg",
            "Würzburg",
            "Regensburg",
            "Ingolstadt",
            "Fürth",
            "Erlangen",
            "Passau",
            "Landshut",
            "Custom / Other District..."
        ],
        "Custom / Other State...": ["Custom / Other District..."]
    },
    "Kenya": {
        "Rift Valley": [
            "Nakuru",
            "Uasin Gishu",
            "Trans Nzoia",
            "Narok",
            "Kericho",
            "Nandi",
            "Bomet",
            "Baringo",
            "Laikipia",
            "Turkana",
            "Custom / Other District..."
        ],
        "Central": [
            "Kiambu",
            "Nyeri",
            "Murang'a",
            "Kirinyaga",
            "Nyandarua",
            "Custom / Other District..."
        ],
        "Custom / Other State...": ["Custom / Other District..."]
    },
    "Custom / Other Country...": {
        "Custom / Other State...": ["Custom / Other District..."]
    }
}


class LocationHierarchy:
    """
    Manages multi-country administrative location hierarchies (Country -> State -> District).
    Provides picklist options for dependent UI selectors.
    """

    @staticmethod
    def get_countries() -> List[str]:
        """Return list of supported countries."""
        return list(ADMIN_HIERARCHY.keys())

    @staticmethod
    def get_states(country: str) -> List[str]:
        """Return list of states/regions for the given country."""
        country_data = ADMIN_HIERARCHY.get(country, ADMIN_HIERARCHY.get("India", {}))
        return list(country_data.keys())

    @staticmethod
    def get_districts(country: str, state: str) -> List[str]:
        """Return list of districts for the given country and state."""
        country_data = ADMIN_HIERARCHY.get(country, {})
        state_districts = country_data.get(state, [])
        if not state_districts:
            # Fallback to general district list if state not found directly
            for s_name, d_list in country_data.items():
                if s_name.lower() == str(state).lower():
                    return d_list
            return ["Custom / Other District..."]
        return state_districts

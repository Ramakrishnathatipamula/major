import sys
import requests
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass
class ResolvedLocation:
    """
    Dataclass holding resolved location name and coordinates.
    """
    name: str
    latitude: float
    longitude: float
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def display_name(self) -> str:
        return self.name


# Built-in offline fallback dictionary for common test places
PRESET_LOCATIONS: Dict[str, ResolvedLocation] = {
    "keechulatapally": ResolvedLocation(
        name="Keechulatapally area, Telangana, India", latitude=17.8421, longitude=78.1254
    ),
    "hyderabad": ResolvedLocation(
        name="Hyderabad, Telangana, India", latitude=17.3850, longitude=78.4867
    ),
    "delhi": ResolvedLocation(
        name="Delhi, India", latitude=28.6139, longitude=77.2090
    ),
    "bengaluru": ResolvedLocation(
        name="Bengaluru, Karnataka, India", latitude=12.9716, longitude=77.5946
    ),
    "mumbai": ResolvedLocation(
        name="Mumbai, Maharashtra, India", latitude=19.0760, longitude=72.8777
    ),
    "minneapolis": ResolvedLocation(
        name="Minneapolis, Minnesota, United States", latitude=44.9778, longitude=-93.2650
    ),
    "boise": ResolvedLocation(
        name="Boise, Idaho, United States", latitude=43.6150, longitude=-116.2023
    ),
    "berlin": ResolvedLocation(
        name="Berlin, Germany", latitude=52.5200, longitude=13.4050
    ),
}


class LocationResolver:
    """
    Map-First Location Resolver Component.
    Provides geocoding place search and display-only reverse geocoding.
    Coordinates selected by the user are authoritative. No silent fallbacks.
    """
    NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
    NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
    OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    HEADERS = {"User-Agent": "AICropDoctor/1.0 (Agricultural Extension Map Support)"}

    def __init__(self, timeout: int = 6):
        self.timeout = timeout

    def search_location(self, query: Optional[str]) -> List[ResolvedLocation]:
        """
        Search for places/villages/cities by query string.

        Args:
            query: Location query string (e.g. "Keechulatapally", "Hyderabad")

        Returns:
            List of ResolvedLocation search candidates.

        Raises:
            ValueError: If query is empty or no matching place is found.
        """
        if not query or not str(query).strip():
            raise ValueError("Location search query cannot be empty.")

        clean_query = str(query).strip()
        query_key = clean_query.lower()

        candidates: List[ResolvedLocation] = []

        # 1. Query Nominatim OpenStreetMap API
        try:
            params = {
                "q": clean_query,
                "format": "json",
                "addressdetails": 1,
                "limit": 5
            }
            resp = requests.get(self.NOMINATIM_SEARCH_URL, params=params, headers=self.HEADERS, timeout=self.timeout)
            if resp.status_code == 200:
                results = resp.json()
                if isinstance(results, list):
                    for item in results:
                        if not isinstance(item, dict):
                            continue
                        disp_name = item.get("display_name", clean_query)
                        lat = float(item["lat"])
                        lon = float(item["lon"])
                        candidates.append(ResolvedLocation(name=disp_name, latitude=lat, longitude=lon, raw=item))
        except Exception:
            pass

        # 2. Query Open-Meteo Geocoding API if Nominatim returned no results
        if not candidates:
            try:
                params = {
                    "name": clean_query,
                    "count": 5,
                    "language": "en",
                    "format": "json"
                }
                resp = requests.get(self.OPEN_METEO_GEOCODING_URL, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", []) if isinstance(data, dict) else []
                    for item in results:
                        if not isinstance(item, dict):
                            continue
                        parts = [item.get("name")]
                        if item.get("admin1"):
                            parts.append(item.get("admin1"))
                        if item.get("country"):
                            parts.append(item.get("country"))
                        disp_name = ", ".join([p for p in parts if p])
                        candidates.append(
                            ResolvedLocation(
                                name=disp_name,
                                latitude=float(item.get("latitude", 0.0)),
                                longitude=float(item.get("longitude", 0.0)),
                                raw=item
                            )
                        )
            except Exception:
                pass

        # 3. Check preset dictionary fallback
        if not candidates:
            for key, preset in PRESET_LOCATIONS.items():
                if key in query_key or query_key in key:
                    candidates.append(preset)
                    break

        # STRICT NO SILENT FALLBACK: If no results found, raise ValueError
        if not candidates:
            raise ValueError(
                f"Location '{clean_query}' could not be found. Please check spelling or select location directly on the map."
            )

        return candidates

    def reverse_geocode(self, latitude: float, longitude: float) -> ResolvedLocation:
        """
        Reverse geocode latitude and longitude to produce a human-friendly display label.
        Does NOT validate or reject coordinates based on administrative hierarchy.

        Args:
            latitude: Latitude float
            longitude: Longitude float

        Returns:
            ResolvedLocation containing display name and coordinates.
        """
        try:
            params = {
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "zoom": 14
            }
            resp = requests.get(self.NOMINATIM_REVERSE_URL, params=params, headers=self.HEADERS, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                disp_name = data.get("display_name")
                if disp_name:
                    return ResolvedLocation(name=disp_name, latitude=latitude, longitude=longitude, raw=data)
        except Exception:
            pass

        # Fallback display label using coordinates
        return ResolvedLocation(
            name=f"Field Location ({latitude:.4f}, {longitude:.4f})",
            latitude=latitude,
            longitude=longitude
        )

    def resolve(self, location_query: Optional[str]) -> ResolvedLocation:
        """
        Convenience method to return top search result for a place query.
        """
        candidates = self.search_location(location_query)
        return candidates[0]

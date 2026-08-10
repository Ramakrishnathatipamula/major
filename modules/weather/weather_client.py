import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import requests

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings

@dataclass
class CurrentWeather:
    """Dataclass holding current weather conditions."""
    temperature_c: float
    relative_humidity_pct: float
    precipitation_mm: float
    wind_speed_kmh: float
    weather_code: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HourlyForecast:
    """Dataclass holding hourly forecast time-series data."""
    timestamps: List[str] = field(default_factory=list)
    temperatures_c: List[float] = field(default_factory=list)
    relative_humidities_pct: List[float] = field(default_factory=list)
    precipitations_mm: List[float] = field(default_factory=list)
    wind_speeds_kmh: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WeatherData:
    """
    Standardized Weather Data structure for Disease Risk Prediction engine.
    """
    latitude: float
    longitude: float
    current: CurrentWeather
    hourly: HourlyForecast
    summary_24h: Dict[str, Any] = field(default_factory=dict)
    provider: str = "open-meteo"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "provider": self.provider,
            "current": self.current.to_dict(),
            "hourly": self.hourly.to_dict(),
            "summary_24h": self.summary_24h,
        }


class WeatherClient:
    """
    Reusable Weather API Client.
    Fetches microclimate data (temperature, humidity, rain, wind, hourly forecast)
    for crop disease risk analysis.
    """
    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
    OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/onecall"

    def __init__(
        self,
        provider: str = "open-meteo",
        api_key: Optional[str] = None,
        timeout: int = 10
    ):
        self.provider = provider.lower()
        self.api_key = api_key if api_key is not None else settings.WEATHER_API_KEY
        self.timeout = timeout

    def fetch_weather(self, latitude: float, longitude: float) -> WeatherData:
        """
        Fetch current weather and forecast for given lat/lon coordinates.

        Args:
            latitude: Latitude (-90.0 to 90.0)
            longitude: Longitude (-180.0 to 180.0)

        Returns:
            WeatherData object containing normalized current and forecast metrics.
        """
        self._validate_coordinates(latitude, longitude)

        if self.provider == "open-meteo":
            return self._fetch_open_meteo(latitude, longitude)
        elif self.provider in ("openweather", "open-weather"):
            return self._fetch_openweather(latitude, longitude)
        else:
            raise ValueError(f"Unsupported weather provider: {self.provider}")

    def _validate_coordinates(self, latitude: float, longitude: float) -> None:
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"Latitude must be between -90.0 and 90.0, got: {latitude}")
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Longitude must be between -180.0 and 180.0, got: {longitude}")

    def _fetch_open_meteo(self, latitude: float, longitude: float) -> WeatherData:
        """Fetch and parse data from Open-Meteo REST API."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "forecast_days": 7,
            "timezone": "auto"
        }

        try:
            response = requests.get(self.OPEN_METEO_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Open-Meteo API request failed: {exc}") from exc

        # Parse current weather
        curr_raw = data.get("current", {})
        current = CurrentWeather(
            temperature_c=float(curr_raw.get("temperature_2m", 0.0)),
            relative_humidity_pct=float(curr_raw.get("relative_humidity_2m", 0.0)),
            precipitation_mm=float(curr_raw.get("precipitation", 0.0)),
            wind_speed_kmh=float(curr_raw.get("wind_speed_10m", 0.0)),
            weather_code=int(curr_raw.get("weather_code", 0)),
            timestamp=str(curr_raw.get("time", ""))
        )

        # Parse hourly forecast
        hourly_raw = data.get("hourly", {})
        hourly = HourlyForecast(
            timestamps=[str(t) for t in hourly_raw.get("time", [])],
            temperatures_c=[float(v) for v in hourly_raw.get("temperature_2m", [])],
            relative_humidities_pct=[float(v) for v in hourly_raw.get("relative_humidity_2m", [])],
            precipitations_mm=[float(v) for v in hourly_raw.get("precipitation", [])],
            wind_speeds_kmh=[float(v) for v in hourly_raw.get("wind_speed_10m", [])]
        )

        # Compute 24-hour summary metrics
        summary_24h = self._compute_24h_summary(hourly)

        return WeatherData(
            latitude=latitude,
            longitude=longitude,
            current=current,
            hourly=hourly,
            summary_24h=summary_24h,
            provider="open-meteo"
        )

    def _fetch_openweather(self, latitude: float, longitude: float) -> WeatherData:
        """Fetch and parse data from OpenWeatherMap API if configured."""
        if not self.api_key or self.api_key == "mock_key_for_testing":
            # Fallback to Open-Meteo if no valid OpenWeather key is provided
            return self._fetch_open_meteo(latitude, longitude)

        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": self.api_key,
            "units": "metric",
            "exclude": "minutely,alerts"
        }

        try:
            response = requests.get(self.OPENWEATHER_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenWeather API request failed: {exc}") from exc

        curr_raw = data.get("current", {})
        current = CurrentWeather(
            temperature_c=float(curr_raw.get("temp", 0.0)),
            relative_humidity_pct=float(curr_raw.get("humidity", 0.0)),
            precipitation_mm=float(curr_raw.get("rain", {}).get("1h", 0.0)),
            wind_speed_kmh=float(curr_raw.get("wind_speed", 0.0) * 3.6),  # convert m/s to km/h
            weather_code=int(curr_raw.get("weather", [{}])[0].get("id", 0)),
            timestamp=str(curr_raw.get("dt", ""))
        )

        hourly_list = data.get("hourly", [])
        hourly = HourlyForecast(
            timestamps=[str(h.get("dt", "")) for h in hourly_list],
            temperatures_c=[float(h.get("temp", 0.0)) for h in hourly_list],
            relative_humidities_pct=[float(h.get("humidity", 0.0)) for h in hourly_list],
            precipitations_mm=[float(h.get("rain", {}).get("1h", 0.0)) for h in hourly_list],
            wind_speeds_kmh=[float(h.get("wind_speed", 0.0) * 3.6) for h in hourly_list]
        )

        summary_24h = self._compute_24h_summary(hourly)

        return WeatherData(
            latitude=latitude,
            longitude=longitude,
            current=current,
            hourly=hourly,
            summary_24h=summary_24h,
            provider="openweather"
        )

    def _compute_24h_summary(self, hourly: HourlyForecast) -> Dict[str, Any]:
        """Compute aggregate summary stats for the first 24 hours of forecast."""
        if not hourly.temperatures_c:
            return {}

        temps_24 = hourly.temperatures_c[:24]
        hum_24 = hourly.relative_humidities_pct[:24]
        rain_24 = hourly.precipitations_mm[:24]
        wind_24 = hourly.wind_speeds_kmh[:24]

        hours_high_humidity = sum(1 for h in hum_24 if h >= 90.0)

        return {
            "max_temp_c": max(temps_24),
            "min_temp_c": min(temps_24),
            "avg_temp_c": round(sum(temps_24) / len(temps_24), 2),
            "max_humidity_pct": max(hum_24),
            "avg_humidity_pct": round(sum(hum_24) / len(hum_24), 2),
            "total_precipitation_mm": round(sum(rain_24), 2),
            "max_wind_speed_kmh": max(wind_24),
            "hours_high_humidity_gte_90pct": hours_high_humidity,
        }

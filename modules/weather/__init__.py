"""
Weather Data Layer for AI Crop Doctor.
Provides weather API integration and microclimate data structures for crop disease risk assessment.
"""

from .weather_client import (
    WeatherClient,
    WeatherData,
    CurrentWeather,
    HourlyForecast,
)

__all__ = [
    "WeatherClient",
    "WeatherData",
    "CurrentWeather",
    "HourlyForecast",
]

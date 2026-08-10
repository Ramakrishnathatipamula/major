import math
from typing import List, Dict, Tuple, Any

class LateBlightCalculator:
    """
    Calculates Late Blight (Phytophthora infestans) disease risk using Wallin's 
    Disease Severity Value (DSV) model (Wallin, 1962; Krause et al., 1975).
    """

    @staticmethod
    def calculate_daily_dsv(temp_c_series: List[float], rh_pct_series: List[float]) -> Tuple[int, float, int]:
        """
        Calculate Wallin DSV (0 to 4) for a single 24-hour window.

        Args:
            temp_c_series: 24 hourly temperature readings in °C.
            rh_pct_series: 24 hourly relative humidity readings in %.

        Returns:
            Tuple of (dsv: int, avg_temp_during_high_rh: float, hours_high_rh: int)
        """
        if len(temp_c_series) == 0 or len(rh_pct_series) == 0:
            return (0, 0.0, 0)

        # Count hours where relative humidity >= 90%
        high_rh_indices = [i for i, rh in enumerate(rh_pct_series) if rh >= 90.0]
        hours_high_rh = len(high_rh_indices)

        if hours_high_rh < 9:
            return (0, 0.0, hours_high_rh)

        # Average temperature during high humidity hours
        high_rh_temps = [temp_c_series[i] for i in high_rh_indices]
        avg_temp = sum(high_rh_temps) / len(high_rh_temps)

        dsv = 0
        if 7.2 <= avg_temp <= 11.6:
            if 9 <= hours_high_rh <= 14:
                dsv = 1
            elif 15 <= hours_high_rh <= 18:
                dsv = 2
            elif hours_high_rh >= 19:
                dsv = 3
        elif 11.7 <= avg_temp <= 15.0:
            if 9 <= hours_high_rh <= 11:
                dsv = 1
            elif 12 <= hours_high_rh <= 14:
                dsv = 2
            elif 15 <= hours_high_rh <= 18:
                dsv = 3
            elif hours_high_rh >= 19:
                dsv = 4
        elif 15.1 <= avg_temp <= 26.6:
            if 9 <= hours_high_rh <= 11:
                dsv = 2
            elif 12 <= hours_high_rh <= 14:
                dsv = 3
            elif hours_high_rh >= 15:
                dsv = 4

        return (dsv, round(avg_temp, 2), hours_high_rh)

    @classmethod
    def evaluate_7day_risk(
        cls, 
        hourly_temps: List[float], 
        hourly_rh: List[float]
    ) -> Dict[str, Any]:
        """
        Evaluate 7-day cumulative Wallin DSV risk score.
        """
        total_hours = min(len(hourly_temps), len(hourly_rh))
        days = total_hours // 24
        if days == 0:
            days = 1  # evaluate available window

        daily_dsvs = []
        factors = []
        cumulative_dsv = 0

        for d in range(days):
            t_sub = hourly_temps[d*24 : (d+1)*24]
            rh_sub = hourly_rh[d*24 : (d+1)*24]
            dsv, avg_temp, high_rh_hrs = cls.calculate_daily_dsv(t_sub, rh_sub)
            daily_dsvs.append(dsv)
            cumulative_dsv += dsv

            if dsv > 0:
                factors.append(
                    f"Day {d+1}: {high_rh_hrs} hrs RH >= 90% at avg temp {avg_temp}°C -> {dsv} DSV(s)"
                )

        # Scale cumulative DSV (28 DSVs max over 7 days = 100%)
        # Normalizing cap: max cumulative DSV target for 7 days is 28
        max_possible_dsv = max(1, days * 4)
        risk_score = round(min(100.0, (cumulative_dsv / max_possible_dsv) * 100.0), 1)

        if cumulative_dsv >= 14 or risk_score >= 50.0:
            risk_level = "HIGH"
        elif cumulative_dsv >= 7 or risk_score >= 25.0:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        factors.append(f"Cumulative DSV: {cumulative_dsv} over {days} day(s)")

        return {
            "cumulative_dsv": cumulative_dsv,
            "daily_dsvs": daily_dsvs,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "factors": factors
        }


class EarlyBlightCalculator:
    """
    Calculates Early Blight (Alternaria solani) risk using P-Day thermal accumulation 
    model (Sands et al., 1979) paired with FAST humidity duration index (Madden et al., 1978).
    """

    @staticmethod
    def _pday_growth_function(t: float) -> float:
        """P-Day temperature response curve f(T)."""
        if t <= 7.0 or t >= 30.0:
            return 0.0
        elif 7.0 < t <= 21.0:
            return 10.0 * (1.0 - math.pow((t - 21.0) / (21.0 - 7.0), 2))
        else:  # 21.0 < t < 30.0
            return 10.0 * (1.0 - math.pow((t - 21.0) / (30.0 - 21.0), 2))

    @classmethod
    def calculate_daily_pday(cls, t_min: float, t_max: float) -> float:
        """Calculate single-day Physiological Day (P-Day) rating."""
        t1 = t_min
        t2 = (2.0 * t_min + t_max) / 3.0
        t3 = (t_min + 2.0 * t_max) / 3.0
        t4 = t_max

        f1 = cls._pday_growth_function(t1)
        f2 = cls._pday_growth_function(t2)
        f3 = cls._pday_growth_function(t3)
        f4 = cls._pday_growth_function(t4)

        pday = (1.0 / 24.0) * (5.0 * f1 + 8.0 * f2 + 8.0 * f3 + 3.0 * f4)
        return round(pday, 2)

    @classmethod
    def evaluate_risk(
        cls, 
        t_min: float, 
        t_max: float, 
        hourly_rh: List[float]
    ) -> Dict[str, Any]:
        """
        Evaluate Early Blight risk score based on daily P-Day and high RH duration (>= 85%).
        """
        pday = cls.calculate_daily_pday(t_min, t_max)

        # Count hours with RH >= 85%
        high_rh_hrs = sum(1 for rh in hourly_rh[:24] if rh >= 85.0)

        # Combine P-Day and humidity into risk score (0 - 100%)
        # Max expected daily P-Day is ~10.0; max high RH hours is 24
        pday_component = min(1.0, pday / 8.0) * 50.0
        rh_component = min(1.0, high_rh_hrs / 12.0) * 50.0

        risk_score = round(min(100.0, pday_component + rh_component), 1)

        if risk_score >= 60.0 or (pday >= 6.0 and high_rh_hrs >= 8):
            risk_level = "HIGH"
        elif risk_score >= 30.0 or (pday >= 3.0 and high_rh_hrs >= 4):
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        factors = [
            f"Daily P-Day thermal rating: {pday}",
            f"Hours with relative humidity >= 85%: {high_rh_hrs} hrs",
            f"Temperature bounds: Min {t_min}°C, Max {t_max}°C"
        ]

        return {
            "pday": pday,
            "high_rh_hours": high_rh_hrs,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "factors": factors
        }


class HealthyBaselineCalculator:
    """
    Adjusts risk assessment for healthy crops (no active disease symptoms in sample).
    """

    @staticmethod
    def adjust_risk_for_healthy(
        env_risk_score: float, 
        env_factors: List[str]
    ) -> Dict[str, Any]:
        """
        Applies inoculum scaling factor (0.5x) for healthy leaf sample diagnosis.
        """
        adjusted_score = round(env_risk_score * 0.5, 1)

        if adjusted_score >= 60.0:
            risk_level = "HIGH"
        elif adjusted_score >= 30.0:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        adjusted_factors = [
            "Leaf sample visual diagnosis: Healthy (No active lesion present in sample)",
            "Base environmental vulnerability factor scaled by 0.5x (Inoculum discount)",
        ] + env_factors

        return {
            "risk_score": adjusted_score,
            "risk_level": risk_level,
            "factors": adjusted_factors
        }

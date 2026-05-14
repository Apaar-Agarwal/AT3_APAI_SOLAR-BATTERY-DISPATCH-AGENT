"""
tariff_client.py — time-of-use tariff schedule.

Indicative AU residential ToU tariff (editable). Schedule:
  Peak     : weekday 16:00 - 21:00
  Shoulder : weekday 07:00 - 16:00 and 21:00 - 22:00
  Off-peak : everything else (incl. all weekend)

Tariff dictionary keys:
  off_peak, shoulder, peak, feed_in   (AUD/kWh)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np
import pandas as pd

from .. import config as cfg


@dataclass
class TariffSchedule:
    rates: dict = field(default_factory=lambda: dict(cfg.TARIFF))

    # ─── period classification ──────────────────────────────────────────────
    def period_of(self, ts: pd.Timestamp) -> str:
        """Return one of {'peak', 'shoulder', 'off_peak'}."""
        h = ts.hour
        is_weekday = ts.dayofweek < 5
        if is_weekday and 16 <= h < 21:
            return "peak"
        if is_weekday and (7 <= h < 16 or 21 <= h < 22):
            return "shoulder"
        return "off_peak"

    # ─── lookups ────────────────────────────────────────────────────────────
    def import_price(self, ts: pd.Timestamp) -> float:
        return float(self.rates[self.period_of(ts)])

    def export_price(self, ts: pd.Timestamp) -> float:
        return float(self.rates["feed_in"])

    def schedule(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return tariff DataFrame: import_price, export_price, period."""
        rows = []
        for ts in index:
            p = self.period_of(ts)
            rows.append({
                "period": p,
                "import_price": self.rates[p],
                "export_price": self.rates["feed_in"],
            })
        return pd.DataFrame(rows, index=index)


# ─── module-level convenience ───────────────────────────────────────────────
def default_schedule() -> TariffSchedule:
    return TariffSchedule()


if __name__ == "__main__":
    idx = pd.date_range("2025-01-15 00:00", periods=24, freq="h")
    sched = default_schedule().schedule(idx)
    print(sched.head(10))
    print("\nUnique periods over 24h:", sched["period"].value_counts().to_dict())

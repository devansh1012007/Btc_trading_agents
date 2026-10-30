"""
Market-agnostic profile (`MarketProfile`, Phase 15 T15.1): defines session
hours, trading days, lot/tick rounding, contract multipliers, settlement
currency, shorting rules, and overnight-gap policies for:
* `crypto_spot` (24/7, no leverage/funding)
* `crypto_perp` (24/7, perpetual swap with funding interval)
* `equity` (Mon-Fri 09:30-16:00 America/New_York / 14:30-21:00 UTC regular session)
* `option` (equity/index or crypto derivative contract with multiplier & expiry).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class MarketProfile:
    profile_name: str  # "crypto_spot" | "crypto_perp" | "equity" | "option"
    asset_class: str
    is_24_7: bool = True
    session_open_minute_utc: int = 0       # e.g. 14*60 + 30 = 870 for 14:30 UTC (09:30 ET)
    session_close_minute_utc: int = 1440   # e.g. 21*60 = 1260 for 21:00 UTC (16:00 ET)
    trading_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    contract_multiplier: float = 1.0
    lot_size: float = 1e-6
    tick_size: float = 0.01
    settlement_currency: str = "USDT"
    allow_short: bool = True
    overnight_gap_policy: str = "allow"    # "allow" | "flatten_before_close"

    @classmethod
    def preset(cls, name: str) -> "MarketProfile":
        key = str(name).strip().lower()
        if key == "crypto_spot":
            return cls(
                profile_name="crypto_spot",
                asset_class="spot",
                is_24_7=True,
                contract_multiplier=1.0,
                lot_size=1e-5,
                tick_size=0.01,
                settlement_currency="USDT",
                allow_short=False,
            )
        if key in ("crypto_perp", "perp"):
            return cls(
                profile_name="crypto_perp",
                asset_class="perp",
                is_24_7=True,
                contract_multiplier=1.0,
                lot_size=1e-4,
                tick_size=0.01,
                settlement_currency="USDT",
                allow_short=True,
            )
        if key == "equity":
            return cls(
                profile_name="equity",
                asset_class="equity",
                is_24_7=False,
                session_open_minute_utc=14 * 60 + 30,   # 09:30 ET (standard UTC offset)
                session_close_minute_utc=21 * 60,       # 16:00 ET
                trading_weekdays=(0, 1, 2, 3, 4),
                contract_multiplier=1.0,
                lot_size=1.0,
                tick_size=0.01,
                settlement_currency="USD",
                allow_short=True,
                overnight_gap_policy="allow",
            )
        if key == "option":
            return cls(
                profile_name="option",
                asset_class="option",
                is_24_7=False,
                session_open_minute_utc=14 * 60 + 30,
                session_close_minute_utc=21 * 60,
                trading_weekdays=(0, 1, 2, 3, 4),
                contract_multiplier=100.0,
                lot_size=1.0,
                tick_size=0.01,
                settlement_currency="USD",
                allow_short=False,
                overnight_gap_policy="flatten_before_close",
            )
        raise ValueError(f"Unknown MarketProfile preset {name!r}")

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "MarketProfile":
        if not config:
            return cls.preset("crypto_perp")
        m_cfg = config.get("market") or {}
        base = cls.preset(str(m_cfg.get("profile", "crypto_perp")))
        if not m_cfg:
            return base
        return cls(
            profile_name=base.profile_name,
            asset_class=str(m_cfg.get("asset_class", base.asset_class)),
            is_24_7=bool(m_cfg.get("is_24_7", base.is_24_7)),
            session_open_minute_utc=int(m_cfg.get("session_open_minute_utc", base.session_open_minute_utc)),
            session_close_minute_utc=int(m_cfg.get("session_close_minute_utc", base.session_close_minute_utc)),
            trading_weekdays=tuple(m_cfg.get("trading_weekdays", base.trading_weekdays)),
            contract_multiplier=float(m_cfg.get("contract_multiplier", base.contract_multiplier)),
            lot_size=float(m_cfg.get("lot_size", base.lot_size)),
            tick_size=float(m_cfg.get("tick_size", base.tick_size)),
            settlement_currency=str(m_cfg.get("settlement_currency", base.settlement_currency)),
            allow_short=bool(m_cfg.get("allow_short", base.allow_short)),
            overnight_gap_policy=str(m_cfg.get("overnight_gap_policy", base.overnight_gap_policy)),
        )

    def is_session_open(self, timestamp: int) -> bool:
        if self.is_24_7:
            return True
        dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        if dt.weekday() not in self.trading_weekdays:
            return False
        minute_of_day = dt.hour * 60 + dt.minute
        return self.session_open_minute_utc <= minute_of_day < self.session_close_minute_utc

    def round_price(self, price: float) -> float:
        if self.tick_size <= 0:
            return float(price)
        return round(round(float(price) / self.tick_size) * self.tick_size, 8)

    def round_quantity(self, quantity: float) -> float:
        if self.lot_size <= 0:
            return float(quantity)
        lots = math.floor((float(quantity) + 1e-12) / self.lot_size)
        return round(max(0.0, lots * self.lot_size), 8)

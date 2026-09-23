"""
Hard-coded map of single-stock leveraged / inverse ETFs → their underlying stock.

These funds are designed to deliver a multiple (e.g. 2x) of the *daily* return of
a single stock, and they reset that exposure **every day**. In practice the
exposure is obtained synthetically via total-return swaps, and the swap
counterparty hedges its delta by trading the underlying stock. That daily
re-hedge is concentrated around the close, which is the trade this strategy
attempts to detect and front-run.

Scope (v1): the **underlying** only. This map defines the strategy's symbol
universe and provides a leverage/AUM-weighted prior for which direction the
dealer hedge must flow. Fund-level flow signals (fetching the ETFs' own bars)
are deliberately out of scope for v1.

Data provenance
---------------
The curated entries below were compiled from issuer fund pages and the ETF
Database single-stock theme listing (https://etfdb.com/themes/single-stock-etfs/).
Tickers, leverage factors, and AUM change over time, and funds are launched and
closed frequently. AUM figures are a point-in-time snapshot (``aum_asof``) and
are used only for *relative* weighting — never as an absolute truth. Run
:func:`validate_available` against the live Alpaca asset list before relying on
this universe in production.

Excluded on purpose:
- Option-income / buy-write funds (YieldMax ``*Y``, ``TSLY``, ``NVDY`` …). They
  are not daily-leveraged trackers, so they do not generate the same daily
  delta hedge.
- Broad-index leveraged ETFs (TQQQ, SPXL, SOXL …). Those hedge baskets of
  constituents, not a single underlying stock.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = [
    "LeveragedEtf",
    "LEVERAGED_SINGLE_STOCK_ETFS",
    "underlyings",
    "etfs_for",
    "net_flow_direction",
    "available_etfs",
    "validate_available",
]

# Sentinel date for the AUM snapshot below.
_AUM_ASOF = "2026-09"


@dataclass(frozen=True)
class LeveragedEtf:
    """A single-stock leveraged/inverse ETF and its underlying.

    Attributes:
        etf: The ETF ticker (registry key of the map).
        underlying: The single stock the ETF tracks.
        leverage: Signed daily leverage. Positive = long (dealer buys the
            underlying when it rises), negative = inverse (dealer sells the
            underlying when it rises). e.g. ``2.0`` for a 2x bull fund,
            ``-2.0`` for a 2x bear fund, ``-1.0`` for a 1x inverse fund.
        family: Issuer / fund family (for grouping and diagnostics).
        name: Human-readable fund name.
        aum_usd: Point-in-time assets under management in USD, or ``None``.
            Used only for relative flow weighting; never as ground truth.
        aum_asof: Date label for ``aum_usd``.
    """

    etf: str
    underlying: str
    leverage: float
    family: str
    name: str
    aum_usd: Optional[float] = None
    aum_asof: Optional[str] = _AUM_ASOF

    @property
    def is_inverse(self) -> bool:
        """True when the fund shorts its underlying (leverage < 0)."""
        return self.leverage < 0


# ---------------------------------------------------------------------------
# The map (keyed by ETF ticker).
#
# Ordering groups by underlying for readability. Keep entries alphabetical by
# ETF ticker within an underlying.
# ---------------------------------------------------------------------------
LEVERAGED_SINGLE_STOCK_ETFS: Dict[str, LeveragedEtf] = {
    # --- AAPL -------------------------------------------------------------
    "AAPU": LeveragedEtf("AAPU", "AAPL", 2.0, "Direxion",
                         "Direxion Daily AAPL Bull 2X ETF"),
    "AAPB": LeveragedEtf("AAPB", "AAPL", 2.0, "GraniteShares",
                         "GraniteShares 2x Long AAPL Daily ETF"),
    "AAPD": LeveragedEtf("AAPD", "AAPL", -1.0, "Direxion",
                         "Direxion Daily AAPL Bear 1X ETF"),

    # --- AMD --------------------------------------------------------------
    "AMDL": LeveragedEtf("AMDL", "AMD", 2.0, "GraniteShares",
                         "GraniteShares 2x Long AMD Daily ETF", 924_100_000.0),
    "AMDU": LeveragedEtf("AMDU", "AMD", 2.0, "Direxion",
                         "Direxion Daily AMD Bull 2X ETF"),
    "AMDD": LeveragedEtf("AMDD", "AMD", -2.0, "Direxion",
                         "Direxion Daily AMD Bear 2X ETF"),

    # --- AMZN -------------------------------------------------------------
    "AMZU": LeveragedEtf("AMZU", "AMZN", 2.0, "Direxion",
                         "Direxion Daily AMZN Bull 2X ETF"),
    "AMZZ": LeveragedEtf("AMZZ", "AMZN", 2.0, "GraniteShares",
                         "GraniteShares 2x Long AMZN Daily ETF"),
    "AMZD": LeveragedEtf("AMZD", "AMZN", -1.0, "Direxion",
                         "Direxion Daily AMZN Bear 1X ETF"),

    # --- AVGO -------------------------------------------------------------
    "AVL": LeveragedEtf("AVL", "AVGO", 2.0, "Direxion",
                        "Direxion Daily AVGO Bull 2X ETF"),

    # --- COIN -------------------------------------------------------------
    "CONL": LeveragedEtf("CONL", "COIN", 2.0, "GraniteShares",
                         "GraniteShares 2x Long COIN Daily ETF", 462_490_000.0),
    "CONI": LeveragedEtf("CONI", "COIN", -2.0, "GraniteShares",
                         "GraniteShares 2x Short COIN Daily ETF"),

    # --- GOOGL ------------------------------------------------------------
    "GGLL": LeveragedEtf("GGLL", "GOOGL", 2.0, "Direxion",
                         "Direxion Daily GOOGL Bull 2X ETF", 997_770_000.0),
    "GGLS": LeveragedEtf("GGLS", "GOOGL", -2.0, "Direxion",
                         "Direxion Daily GOOGL Bear 2X ETF"),

    # --- META -------------------------------------------------------------
    "METU": LeveragedEtf("METU", "META", 2.0, "Direxion",
                         "Direxion Daily META Bull 2X ETF", 476_450_000.0),
    "FBL": LeveragedEtf("FBL", "META", 2.0, "GraniteShares",
                        "GraniteShares 2x Long META Daily ETF"),
    "METD": LeveragedEtf("METD", "META", -2.0, "Direxion",
                         "Direxion Daily META Bear 2X ETF"),

    # --- MRVL -------------------------------------------------------------
    "MVLL": LeveragedEtf("MVLL", "MRVL", 2.0, "GraniteShares",
                         "GraniteShares 2x Long MRVL Daily ETF", 397_240_000.0),

    # --- MSFT -------------------------------------------------------------
    "MSFU": LeveragedEtf("MSFU", "MSFT", 2.0, "Direxion",
                         "Direxion Daily MSFT Bull 2X ETF", 506_420_000.0),
    "MSFL": LeveragedEtf("MSFL", "MSFT", 2.0, "GraniteShares",
                         "GraniteShares 2x Long MSFT Daily ETF"),
    "MSFD": LeveragedEtf("MSFD", "MSFT", -2.0, "Direxion",
                         "Direxion Daily MSFT Bear 2X ETF"),

    # --- MSTR -------------------------------------------------------------
    "MSTU": LeveragedEtf("MSTU", "MSTR", 2.0, "T-Rex",
                         "T-Rex 2X Long MSTR Daily Target ETF", 500_120_000.0),
    "MSTX": LeveragedEtf("MSTX", "MSTR", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long MSTR ETF"),
    "MSTZ": LeveragedEtf("MSTZ", "MSTR", -2.0, "T-Rex",
                         "T-Rex 2X Inverse MSTR Daily Target ETF"),
    "SMST": LeveragedEtf("SMST", "MSTR", -2.0, "Defiance",
                         "Defiance Daily Target 2X Short MSTR ETF"),

    # --- MU ---------------------------------------------------------------
    "MUU": LeveragedEtf("MUU", "MU", 2.0, "Direxion",
                        "Direxion Daily MU Bull 2X ETF", 3_391_620_000.0),
    "MULL": LeveragedEtf("MULL", "MU", 2.0, "GraniteShares",
                         "GraniteShares 2x Long MU Daily ETF", 624_360_000.0),
    "MUD": LeveragedEtf("MUD", "MU", -2.0, "Direxion",
                        "Direxion Daily MU Bear 2X ETF"),

    # --- NFLX -------------------------------------------------------------
    "NFXL": LeveragedEtf("NFXL", "NFLX", 2.0, "Direxion",
                         "Direxion Daily NFLX Bull 2X ETF"),

    # --- NVDA -------------------------------------------------------------
    "NVDL": LeveragedEtf("NVDL", "NVDA", 2.0, "GraniteShares",
                         "GraniteShares 2x Long NVDA Daily ETF", 3_477_490_000.0),
    "NVDU": LeveragedEtf("NVDU", "NVDA", 2.0, "Direxion",
                         "Direxion Daily NVDA Bull 2X ETF", 490_630_000.0),
    "NVDX": LeveragedEtf("NVDX", "NVDA", 2.0, "T-Rex",
                         "T-Rex 2X Long NVIDIA Daily Target ETF", 460_510_000.0),
    "NVD": LeveragedEtf("NVD", "NVDA", -2.0, "GraniteShares",
                        "GraniteShares 2x Short NVDA Daily ETF"),
    "NVDD": LeveragedEtf("NVDD", "NVDA", -2.0, "Direxion",
                         "Direxion Daily NVDA Bear 2X ETF"),
    "NVDQ": LeveragedEtf("NVDQ", "NVDA", -2.0, "T-Rex",
                         "T-Rex 2X Inverse NVIDIA Daily Target ETF"),

    # --- PLTR -------------------------------------------------------------
    "PLTU": LeveragedEtf("PLTU", "PLTR", 2.0, "Direxion",
                         "Direxion Daily PLTR Bull 2X ETF", 423_590_000.0),
    "PTIR": LeveragedEtf("PTIR", "PLTR", 2.0, "GraniteShares",
                         "GraniteShares 2x Long PLTR Daily ETF", 319_790_000.0),

    # --- SNDK -------------------------------------------------------------
    "SNXX": LeveragedEtf("SNXX", "SNDK", 2.0, "Tradr",
                         "Tradr 2X Long SNDK Daily ETF", 1_533_580_000.0),

    # --- TSLA -------------------------------------------------------------
    "TSLL": LeveragedEtf("TSLL", "TSLA", 2.0, "Direxion",
                         "Direxion Daily TSLA Bull 2X ETF", 3_675_750_000.0),
    "TSLQ": LeveragedEtf("TSLQ", "TSLA", -1.0, "Direxion",
                         "Direxion Daily TSLA Bear 1X ETF"),

    # --- TSM --------------------------------------------------------------
    "TSMX": LeveragedEtf("TSMX", "TSM", 2.0, "Direxion",
                         "Direxion Daily TSM Bull 2X ETF", 536_530_000.0),
}


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def underlyings() -> List[str]:
    """Sorted, deduplicated list of underlying stock symbols.

    This is the strategy's symbol universe.
    """
    return sorted({etf.underlying for etf in LEVERAGED_SINGLE_STOCK_ETFS.values()})


def etfs_for(underlying: str) -> List[LeveragedEtf]:
    """All mapped ETFs tracking ``underlying`` (case-insensitive)."""
    target = underlying.strip().upper()
    return [e for e in LEVERAGED_SINGLE_STOCK_ETFS.values()
            if e.underlying == target]


def available_etfs(available_symbols) -> List[LeveragedEtf]:
    """Mapped ETFs whose ticker is present in ``available_symbols``."""
    available = {s.strip().upper() for s in available_symbols}
    return [e for e in LEVERAGED_SINGLE_STOCK_ETFS.values()
            if e.etf in available]


def validate_available(available_symbols) -> Dict[str, List[str]]:
    """Check the map against a live set of tradable symbols.

    Args:
        available_symbols: Iterable of tradable symbols (e.g. Alpaca assets).

    Returns:
        ``{"missing": [...], "underlying_missing": [...]}`` — ETF tickers in
        the map that are not tradable, and underlying symbols that are not
        tradable. Both lists are sorted.
    """
    available = {s.strip().upper() for s in available_symbols}
    missing_etfs = sorted(
        e.etf for e in LEVERAGED_SINGLE_STOCK_ETFS.values() if e.etf not in available)
    missing_underlyings = sorted(
        u for u in underlyings() if u not in available)
    return {"missing": missing_etfs, "underlying_missing": missing_underlyings}


def net_flow_direction(
    underlying: str,
    underlying_return: float,
    use_aum: bool = True,
) -> float:
    """Signed estimate of the dealer hedge notional for one underlying.

    Each fund's daily rebalance implies a hedge of roughly
    ``leverage × AUM × underlying_return`` in the underlying (positive =
    the dealer must BUY, negative = SELL). Summing over every mapped fund
    gives the net flow the strategy is trying to front-run.

    Args:
        underlying: Underlying stock symbol.
        underlying_return: The underlying's move over the reference window
            (e.g. today's return so far), as a decimal (0.01 = +1%).
        use_aum: When True, weight each fund by its AUM (falling back to an
            equal weight of 1.0 for funds with no AUM recorded). When False,
            weight every fund equally by 1.0 — useful when AUM is stale.

    Returns:
        Signed notional estimate. Sign is the actionable direction.
    """
    total = 0.0
    for etf in etfs_for(underlying):
        weight = 1.0
        if use_aum and etf.aum_usd:
            weight = float(etf.aum_usd)
        total += etf.leverage * weight * underlying_return
    return total
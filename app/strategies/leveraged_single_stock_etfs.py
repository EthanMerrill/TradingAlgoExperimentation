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
The entries below were generated from the ETF Database single-stock theme
listing (https://etfdb.com/themes/single-stock-etfs/). etfdb.com blocks scripted
access, so the table is copy-pasted by hand, parsed with
``app.research.parse_etfdb_paste``, and archived under
``app/research/data/etfdb_single_stock_etfs_<date>.*``. The map itself is
regenerated from that snapshot with ``app.research.build_etf_map``. Tickers,
leverage factors, and AUM change over time, and funds are launched and closed
frequently. AUM figures are a point-in-time snapshot (``aum_asof``) and are used
only for *relative* weighting — never as an absolute truth. Run
:func:`validate_available` against the live Alpaca asset list before relying on
this universe in production.

Notes from the 2026-10 refresh:
- The universe grew from 41 to 158 funds, adding the Leverage Shares, Tradr,
  ProShares (``Ultra``), KraneShares, Corgi and T-Rex families alongside the
  original Direxion, GraniteShares and Defiance names.
- ``AMDU`` no longer appears in the listing; Direxion's AMD Bull 2X fund is
  listed as ``AMUU`` and is recorded under that ticker.
- Direxion's single-stock Bear funds are **1X** (e.g. ``NVDD``, ``METD``), so
  several leverage factors previously recorded as ``-2.0`` are now ``-1.0``.
- ``TSLQ`` is now "Tradr 2X Short TSLA Daily ETF" (a -2.0 fund), not Direxion's
  1X bear fund.
- ``BRKU`` tracks Berkshire Hathaway Class B, recorded as ``BRK.B``.

Excluded on purpose:
- Option-income / buy-write / structured funds (YieldMax/YieldBOOST ``*Y``,
  Kurv "Yield Premium", ``*W`` WeeklyPay, ``*Autocallable``, "Option Income",
  "BuyWrite", "Growth & Income", Defiance "Leveraged Long Income"). They are not
  daily-leveraged trackers, so they do not generate the same daily delta hedge.
- Currency-hedged "ADRhedged" single-stock funds (they are not leveraged).
- Funds whose underlying has no tradable US symbol (e.g. the SpaceX trackers
  ``SPCU``/``SPAL``/``SNK``); the strategy needs a hedgeable stock.
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
_AUM_ASOF = "2026-10"


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
    # --- AAL ------------------------------------------------------------
    "AALG": LeveragedEtf("AALG", "AAL", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long AAL Daily ETF", 4_150_000.0),
    # --- AAPL -----------------------------------------------------------
    "AAPB": LeveragedEtf("AAPB", "AAPL", 2.0, "GraniteShares",
                         "GraniteShares 2x Long AAPL Daily ETF", 17_330_000.0),
    "AAPD": LeveragedEtf("AAPD", "AAPL", -1.0, "Direxion",
                         "Direxion Daily AAPL Bear 1X ETF", 20_350_000.0),
    "AAPU": LeveragedEtf("AAPU", "AAPL", 2.0, "Direxion",
                         "Direxion Daily AAPL Bull 2X ETF", 164_310_000.0),
    "AAPX": LeveragedEtf("AAPX", "AAPL", 2.0, "T-Rex",
                         "T-Rex 2X Long Apple Daily Target ETF", 8_770_000.0),
    # --- ACHR -----------------------------------------------------------
    "ARCX": LeveragedEtf("ARCX", "ACHR", 2.0, "Tradr",
                         "Tradr 2X Long ACHR Daily ETF", 5_310_000.0),
    # --- ADBE -----------------------------------------------------------
    "ADBG": LeveragedEtf("ADBG", "ADBE", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long ADBE Daily ETF", 33_940_000.0),
    # --- ALAB -----------------------------------------------------------
    "LABX": LeveragedEtf("LABX", "ALAB", 2.0, "Tradr",
                         "Tradr 2X Long ALAB Daily ETF", 80_470_000.0),
    # --- AMD ------------------------------------------------------------
    "AMDD": LeveragedEtf("AMDD", "AMD", -1.0, "Direxion",
                         "Direxion Daily AMD Bear 1X ETF", 22_650_000.0),
    "AMDG": LeveragedEtf("AMDG", "AMD", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long AMD Daily ETF", 77_830_000.0),
    "AMDL": LeveragedEtf("AMDL", "AMD", 2.0, "GraniteShares",
                         "GraniteShares 2x Long AMD Daily ETF", 1_486_000_000.0),
    "AMUU": LeveragedEtf("AMUU", "AMD", 2.0, "Direxion",
                         "Direxion Daily AMD Bull 2X ETF", 149_010_000.0),
    # --- AMZN -----------------------------------------------------------
    "AMZD": LeveragedEtf("AMZD", "AMZN", -1.0, "Direxion",
                         "Direxion Daily AMZN Bear 1X ETF", 12_590_000.0),
    "AMZU": LeveragedEtf("AMZU", "AMZN", 2.0, "Direxion",
                         "Direxion Daily AMZN Bull 2X ETF", 343_850_000.0),
    "AMZZ": LeveragedEtf("AMZZ", "AMZN", 2.0, "GraniteShares",
                         "GraniteShares 2x Long AMZN Daily ETF", 45_680_000.0),
    # --- APP ------------------------------------------------------------
    "APPX": LeveragedEtf("APPX", "APP", 2.0, "Tradr",
                         "Tradr 2X Long APP Daily ETF", 76_660_000.0),
    # --- ARM ------------------------------------------------------------
    "ARMG": LeveragedEtf("ARMG", "ARM", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long ARM Daily ETF", 114_590_000.0),
    # --- ASML -----------------------------------------------------------
    "ASMG": LeveragedEtf("ASMG", "ASML", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long ASML Daily ETF", 70_340_000.0),
    # --- ASTS -----------------------------------------------------------
    "ASTX": LeveragedEtf("ASTX", "ASTS", 2.0, "Tradr",
                         "Tradr 2X Long ASTS Daily ETF", 238_080_000.0),
    # --- AVGO -----------------------------------------------------------
    "AVGG": LeveragedEtf("AVGG", "AVGO", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long AVGO Daily ETF", 56_150_000.0),
    "AVGU": LeveragedEtf("AVGU", "AVGO", 2.0, "GraniteShares",
                         "GraniteShares 2x Long AVGO Daily ETF", 39_910_000.0),
    "AVGX": LeveragedEtf("AVGX", "AVGO", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long AVGO ETF", 230_380_000.0),
    "AVL": LeveragedEtf("AVL", "AVGO", 2.0, "Direxion",
                        "Direxion Daily AVGO Bull 2X ETF", 306_400_000.0),
    "AVS": LeveragedEtf("AVS", "AVGO", -1.0, "Direxion",
                        "Direxion Daily AVGO Bear 1X ETF", 9_370_000.0),
    # --- BA -------------------------------------------------------------
    "BOEG": LeveragedEtf("BOEG", "BA", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long BA Daily ETF", 10_150_000.0),
    "BOEU": LeveragedEtf("BOEU", "BA", 2.0, "Direxion",
                         "Direxion Daily BA Bull 2X ETF", 23_900_000.0),
    # --- BABA -----------------------------------------------------------
    "BABX": LeveragedEtf("BABX", "BABA", 2.0, "GraniteShares",
                         "GraniteShares 2x Long BABA Daily ETF", 116_720_000.0),
    "KBAB": LeveragedEtf("KBAB", "BABA", 2.0, "KraneShares",
                         "KraneShares 2x Long BABA Daily ETF", 2_680_000.0),
    # --- BB -------------------------------------------------------------
    "BBUL": LeveragedEtf("BBUL", "BB", 2.0, "GraniteShares",
                         "GraniteShares 2x Long BB Daily ETF", 1_540_000.0),
    # --- BRK.B ----------------------------------------------------------
    "BRKU": LeveragedEtf("BRKU", "BRK.B", 2.0, "Direxion",
                         "Direxion Daily BRKB Bull 2X ETF", 39_510_000.0),
    # --- BULL -----------------------------------------------------------
    "BULG": LeveragedEtf("BULG", "BULL", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long BULL Daily ETF", 5_090_000.0),
    # --- CEG ------------------------------------------------------------
    "CEGX": LeveragedEtf("CEGX", "CEG", 2.0, "Tradr",
                         "Tradr 2X Long CEG Daily ETF", 15_560_000.0),
    # --- COIN -----------------------------------------------------------
    "COIG": LeveragedEtf("COIG", "COIN", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long COIN Daily ETF", 11_050_000.0),
    "CONI": LeveragedEtf("CONI", "COIN", -2.0, "GraniteShares",
                         "GraniteShares 2x Short COIN Daily ETF", 12_030_000.0),
    "CONL": LeveragedEtf("CONL", "COIN", 2.0, "GraniteShares",
                         "GraniteShares 2x Long COIN Daily ETF", 590_400_000.0),
    # --- CRCL -----------------------------------------------------------
    "CCUP": LeveragedEtf("CCUP", "CRCL", 2.0, "T-Rex",
                         "T-Rex 2X Long CRCL Daily Target ETF", 33_610_000.0),
    "CRCA": LeveragedEtf("CRCA", "CRCL", 2.0, "ProShares",
                         "ProShares Ultra CRCL", 102_320_000.0),
    "CRCG": LeveragedEtf("CRCG", "CRCL", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long CRCL Daily ETF", 115_870_000.0),
    # --- CRM ------------------------------------------------------------
    "CRMG": LeveragedEtf("CRMG", "CRM", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long CRM Daily ETF", 93_860_000.0),
    # --- CRWD -----------------------------------------------------------
    "CRWL": LeveragedEtf("CRWL", "CRWD", 2.0, "GraniteShares",
                         "GraniteShares 2x Long CRWD Daily ETF", 84_120_000.0),
    # --- CRWV -----------------------------------------------------------
    "CRWG": LeveragedEtf("CRWG", "CRWV", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long CRWV Daily ETF", 154_620_000.0),
    "CRWU": LeveragedEtf("CRWU", "CRWV", 2.0, "T-Rex",
                         "T-Rex 2X Long CRWV Daily Target ETF", 30_590_000.0),
    "CWVX": LeveragedEtf("CWVX", "CRWV", 2.0, "Tradr",
                         "Tradr 2X Long CRWV Daily ETF", 79_650_000.0),
    # --- CSCO -----------------------------------------------------------
    "CSCL": LeveragedEtf("CSCL", "CSCO", 2.0, "Direxion",
                         "Direxion Daily CSCO Bull 2X ETF", 16_340_000.0),
    "CSCS": LeveragedEtf("CSCS", "CSCO", -1.0, "Direxion",
                         "Direxion Daily CSCO Bear 1X ETF", 2_010_000.0),
    # --- DELL -----------------------------------------------------------
    "DLLL": LeveragedEtf("DLLL", "DELL", 2.0, "GraniteShares",
                         "GraniteShares 2x Long DELL Daily ETF", 216_490_000.0),
    # --- DJT ------------------------------------------------------------
    "DJTU": LeveragedEtf("DJTU", "DJT", 2.0, "T-Rex",
                         "T-Rex 2X Long DJT Daily Target ETF", 9_330_000.0),
    # --- GEV ------------------------------------------------------------
    "GEVX": LeveragedEtf("GEVX", "GEV", 2.0, "Tradr",
                         "Tradr 2X Long GEV Daily ETF", 40_570_000.0),
    # --- GME ------------------------------------------------------------
    "GMEU": LeveragedEtf("GMEU", "GME", 2.0, "T-Rex",
                         "T-Rex 2X Long GME Daily Target ETF", 25_920_000.0),
    # --- GOOGL ----------------------------------------------------------
    "GGLL": LeveragedEtf("GGLL", "GOOGL", 2.0, "Direxion",
                         "Direxion Daily GOOGL Bull 2X ETF", 1_128_180_000.0),
    "GGLS": LeveragedEtf("GGLS", "GOOGL", -1.0, "Direxion",
                         "Direxion Daily GOOGL Bear 1X ETF", 13_140_000.0),
    "GOOX": LeveragedEtf("GOOX", "GOOGL", 2.0, "T-Rex",
                         "T-Rex 2X Long Alphabet Daily Target ETF", 64_690_000.0),
    "GOU": LeveragedEtf("GOU", "GOOGL", 2.0, "GraniteShares",
                        "GraniteShares 2x Long GOOGL Daily ETF", 18_070_000.0),
    # --- HIMS -----------------------------------------------------------
    "HIMZ": LeveragedEtf("HIMZ", "HIMS", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long HIMS ETF", 57_690_000.0),
    # --- HOOD -----------------------------------------------------------
    "HOOG": LeveragedEtf("HOOG", "HOOD", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long HOOD Daily ETF", 108_980_000.0),
    "HOOX": LeveragedEtf("HOOX", "HOOD", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long HOOD ETF", 31_210_000.0),
    "ROBN": LeveragedEtf("ROBN", "HOOD", 2.0, "T-Rex",
                         "T-Rex 2X Long HOOD Daily Target ETF", 124_350_000.0),
    # --- INTC -----------------------------------------------------------
    "INTW": LeveragedEtf("INTW", "INTC", 2.0, "GraniteShares",
                         "GraniteShares 2x Long INTC Daily ETF", 531_150_000.0),
    # --- IONQ -----------------------------------------------------------
    "IONL": LeveragedEtf("IONL", "IONQ", 2.0, "GraniteShares",
                         "GraniteShares 2x Long IONQ Daily ETF", 81_780_000.0),
    "IONX": LeveragedEtf("IONX", "IONQ", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long IONQ ETF", 174_270_000.0),
    "IONZ": LeveragedEtf("IONZ", "IONQ", -2.0, "Defiance",
                         "Defiance Daily Target 2x Short IONQ ETF", 6_970_000.0),
    # --- ISRG -----------------------------------------------------------
    "ISUL": LeveragedEtf("ISUL", "ISRG", 2.0, "GraniteShares",
                         "GraniteShares 2x Long ISRG Daily ETF", 10_430_000.0),
    # --- LLY ------------------------------------------------------------
    "ELIL": LeveragedEtf("ELIL", "LLY", 2.0, "Direxion",
                         "Direxion Daily LLY Bull 2X ETF", 21_500_000.0),
    "LLYX": LeveragedEtf("LLYX", "LLY", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long LLY ETF", 82_440_000.0),
    # --- LMT ------------------------------------------------------------
    "LMTL": LeveragedEtf("LMTL", "LMT", 2.0, "Direxion",
                         "Direxion Daily LMT Bull 2X ETF", 5_770_000.0),
    # --- LRCX -----------------------------------------------------------
    "LRCU": LeveragedEtf("LRCU", "LRCX", 2.0, "Tradr",
                         "Tradr 2X Long LRCX Daily ETF", 49_380_000.0),
    # --- MARA -----------------------------------------------------------
    "MRAL": LeveragedEtf("MRAL", "MARA", 2.0, "GraniteShares",
                         "GraniteShares 2x Long MARA Daily ETF", 40_930_000.0),
    # --- MELI -----------------------------------------------------------
    "KMLI": LeveragedEtf("KMLI", "MELI", 2.0, "KraneShares",
                         "KraneShares 2x Long MELI Daily ETF", 7_680_000.0),
    # --- META -----------------------------------------------------------
    "FBL": LeveragedEtf("FBL", "META", 2.0, "GraniteShares",
                        "GraniteShares 2x Long META Daily ETF", 186_750_000.0),
    "METD": LeveragedEtf("METD", "META", -1.0, "Direxion",
                         "Direxion Daily META Bear 1X ETF", 14_110_000.0),
    "METU": LeveragedEtf("METU", "META", 2.0, "Direxion",
                         "Direxion Daily META Bull 2X ETF", 553_480_000.0),
    # --- MRVL -----------------------------------------------------------
    "MVLL": LeveragedEtf("MVLL", "MRVL", 2.0, "GraniteShares",
                         "GraniteShares 2x Long MRVL Daily ETF", 414_150_000.0),
    # --- MSFT -----------------------------------------------------------
    "MSFD": LeveragedEtf("MSFD", "MSFT", -1.0, "Direxion",
                         "Direxion Daily MSFT Bear 1X ETF", 8_770_000.0),
    "MSFL": LeveragedEtf("MSFL", "MSFT", 2.0, "GraniteShares",
                         "GraniteShares 2x Long MSFT Daily ETF", 66_170_000.0),
    "MSFU": LeveragedEtf("MSFU", "MSFT", 2.0, "Direxion",
                         "Direxion Daily MSFT Bull 2X ETF", 551_580_000.0),
    "MSFX": LeveragedEtf("MSFX", "MSFT", 2.0, "T-Rex",
                         "T-Rex 2X Long Microsoft Daily Target ETF", 19_690_000.0),
    # --- MSTR -----------------------------------------------------------
    "MSTP": LeveragedEtf("MSTP", "MSTR", 2.0, "GraniteShares",
                         "GraniteShares 2x Long MSTR Daily ETF", 20_010_000.0),
    "MSTU": LeveragedEtf("MSTU", "MSTR", 2.0, "T-Rex",
                         "T-Rex 2X Long MSTR Daily Target ETF", 894_660_000.0),
    "MSTX": LeveragedEtf("MSTX", "MSTR", 2.0, "Defiance",
                         "Defiance Daily Target 2x Long MSTR ETF", 390_220_000.0),
    "MSTZ": LeveragedEtf("MSTZ", "MSTR", -2.0, "T-Rex",
                         "T-Rex 2X Inverse MSTR Daily Target ETF", 69_960_000.0),
    "SMST": LeveragedEtf("SMST", "MSTR", -2.0, "Defiance",
                         "Defiance Daily Target 2x Short MSTR ETF", 20_440_000.0),
    # --- MU -------------------------------------------------------------
    "MUD": LeveragedEtf("MUD", "MU", -1.0, "Direxion",
                        "Direxion Daily MU Bear 1X ETF", 31_250_000.0),
    "MULL": LeveragedEtf("MULL", "MU", 2.0, "GraniteShares",
                         "GraniteShares 2x Long MU Daily ETF", 713_380_000.0),
    "MUU": LeveragedEtf("MUU", "MU", 2.0, "Direxion",
                        "Direxion Daily MU Bull 2X ETF", 4_454_360_000.0),
    # --- NBIS -----------------------------------------------------------
    "NBIL": LeveragedEtf("NBIL", "NBIS", 2.0, "GraniteShares",
                         "GraniteShares 2x Long NBIS Daily ETF", 167_480_000.0),
    # --- NFLX -----------------------------------------------------------
    "NFLU": LeveragedEtf("NFLU", "NFLX", 2.0, "T-Rex",
                         "T-Rex 2X Long NFLX Daily Target ETF", 23_200_000.0),
    "NFXL": LeveragedEtf("NFXL", "NFLX", 2.0, "Direxion",
                         "Direxion Daily NFLX Bull 2X ETF", 128_240_000.0),
    "NFXS": LeveragedEtf("NFXS", "NFLX", -1.0, "Direxion",
                         "Direxion Daily NFLX Bear 1X ETF", 3_910_000.0),
    # --- NOW ------------------------------------------------------------
    "NOWL": LeveragedEtf("NOWL", "NOW", 2.0, "GraniteShares",
                         "GraniteShares 2x Long NOW Daily ETF", 190_960_000.0),
    # --- NVDA -----------------------------------------------------------
    "NVD": LeveragedEtf("NVD", "NVDA", -2.0, "GraniteShares",
                        "GraniteShares 2x Short NVDA Daily ETF", 63_910_000.0),
    "NVDD": LeveragedEtf("NVDD", "NVDA", -1.0, "Direxion",
                         "Direxion Daily NVDA Bear 1X ETF", 13_690_000.0),
    "NVDG": LeveragedEtf("NVDG", "NVDA", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long NVDA Daily ETF", 90_730_000.0),
    "NVDL": LeveragedEtf("NVDL", "NVDA", 2.0, "GraniteShares",
                         "GraniteShares 2x Long NVDA Daily ETF", 3_820_080_000.0),
    "NVDQ": LeveragedEtf("NVDQ", "NVDA", -2.0, "T-Rex",
                         "T-Rex 2X Inverse NVIDIA Daily Target ETF", 13_980_000.0),
    "NVDS": LeveragedEtf("NVDS", "NVDA", -1.5, "Tradr",
                         "Tradr 1.5X Short NVDA Daily ETF", 10_030_000.0),
    "NVDU": LeveragedEtf("NVDU", "NVDA", 2.0, "Direxion",
                         "Direxion Daily NVDA Bull 2X ETF", 576_670_000.0),
    "NVDX": LeveragedEtf("NVDX", "NVDA", 2.0, "T-Rex",
                         "T-Rex 2X Long NVIDIA Daily Target ETF", 512_950_000.0),
    # --- NVO ------------------------------------------------------------
    "NVOX": LeveragedEtf("NVOX", "NVO", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long NVO ETF", 38_570_000.0),
    # --- OKLO -----------------------------------------------------------
    "OKLL": LeveragedEtf("OKLL", "OKLO", 2.0, "Defiance",
                         "Defiance Daily Target 2x Long OKLO ETF", 106_660_000.0),
    # --- ORCL -----------------------------------------------------------
    "ORCX": LeveragedEtf("ORCX", "ORCL", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long ORCL ETF", 241_710_000.0),
    # --- P --------------------------------------------------------------
    "PUL": LeveragedEtf("PUL", "P", 2.0, "GraniteShares",
                        "GraniteShares 2x Long P Daily ETF", 2_210_000.0),
    # --- PANW -----------------------------------------------------------
    "PALD": LeveragedEtf("PALD", "PANW", -1.0, "Direxion",
                         "Direxion Daily PANW Bear 1X ETF", 2_490_000.0),
    "PALU": LeveragedEtf("PALU", "PANW", 2.0, "Direxion",
                         "Direxion Daily PANW Bull 2X ETF", 52_360_000.0),
    "PANG": LeveragedEtf("PANG", "PANW", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long PANW Daily ETF", 21_740_000.0),
    # --- PDD ------------------------------------------------------------
    "KPDD": LeveragedEtf("KPDD", "PDD", 2.0, "KraneShares",
                         "KraneShares 2x Long PDD Daily ETF", 14_680_000.0),
    "PDDL": LeveragedEtf("PDDL", "PDD", 2.0, "GraniteShares",
                         "GraniteShares 2x Long PDD Daily ETF", 6_520_000.0),
    # --- PLTR -----------------------------------------------------------
    "PLTD": LeveragedEtf("PLTD", "PLTR", -1.0, "Direxion",
                         "Direxion Daily PLTR Bear 1X ETF", 26_850_000.0),
    "PLTG": LeveragedEtf("PLTG", "PLTR", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long PLTR Daily ETF", 53_310_000.0),
    "PLTU": LeveragedEtf("PLTU", "PLTR", 2.0, "Direxion",
                         "Direxion Daily PLTR Bull 2X ETF", 444_140_000.0),
    "PLTZ": LeveragedEtf("PLTZ", "PLTR", -2.0, "Defiance",
                         "Defiance Daily Target 2x Short PLTR ETF", 34_050_000.0),
    "PTIR": LeveragedEtf("PTIR", "PLTR", 2.0, "GraniteShares",
                         "GraniteShares 2x Long PLTR Daily ETF", 361_180_000.0),
    # --- PYPL -----------------------------------------------------------
    "PYPG": LeveragedEtf("PYPG", "PYPL", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long PYPL Daily ETF", 15_110_000.0),
    # --- QBTS -----------------------------------------------------------
    "QBTX": LeveragedEtf("QBTX", "QBTS", 2.0, "Tradr",
                         "Tradr 2X Long QBTS Daily ETF", 56_530_000.0),
    # --- QCOM -----------------------------------------------------------
    "QCMD": LeveragedEtf("QCMD", "QCOM", -1.0, "Direxion",
                         "Direxion Daily QCOM Bear 1X ETF", 2_800_000.0),
    "QCML": LeveragedEtf("QCML", "QCOM", 2.0, "GraniteShares",
                         "GraniteShares 2x Long QCOM Daily ETF", 54_370_000.0),
    "QCMU": LeveragedEtf("QCMU", "QCOM", 2.0, "Direxion",
                         "Direxion Daily QCOM Bull 2X ETF", 23_350_000.0),
    # --- QUBT -----------------------------------------------------------
    "QUBX": LeveragedEtf("QUBX", "QUBT", 2.0, "Tradr",
                         "Tradr 2X Long QUBT Daily ETF", 18_400_000.0),
    # --- RBLX -----------------------------------------------------------
    "RBLU": LeveragedEtf("RBLU", "RBLX", 2.0, "T-Rex",
                         "T-Rex 2X Long RBLX Daily Target ETF", 11_100_000.0),
    # --- RDDT -----------------------------------------------------------
    "RDTL": LeveragedEtf("RDTL", "RDDT", 2.0, "GraniteShares",
                         "GraniteShares 2x Long RDDT Daily ETF", 63_380_000.0),
    # --- RGTI -----------------------------------------------------------
    "RGTU": LeveragedEtf("RGTU", "RGTI", 2.0, "Tradr",
                         "Tradr 2X Long RGTI Daily ETF", 8_220_000.0),
    "RGTX": LeveragedEtf("RGTX", "RGTI", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long RGTI ETF", 31_870_000.0),
    # --- RIOT -----------------------------------------------------------
    "RIOX": LeveragedEtf("RIOX", "RIOT", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long RIOT ETF", 27_360_000.0),
    # --- RIVN -----------------------------------------------------------
    "RVNL": LeveragedEtf("RVNL", "RIVN", 2.0, "GraniteShares",
                         "GraniteShares 2x Long RIVN Daily ETF", 12_230_000.0),
    # --- RKLB -----------------------------------------------------------
    "RKLX": LeveragedEtf("RKLX", "RKLB", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long RKLB ETF", 206_480_000.0),
    # --- RTX ------------------------------------------------------------
    "RTXG": LeveragedEtf("RTXG", "RTX", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long RTX Daily ETF", 2_550_000.0),
    # --- SHOP -----------------------------------------------------------
    "SHPU": LeveragedEtf("SHPU", "SHOP", 2.0, "Direxion",
                         "Direxion Daily SHOP Bull 2X ETF", 14_000_000.0),
    # --- SMCI -----------------------------------------------------------
    "SMCC": LeveragedEtf("SMCC", "SMCI", 2.0, "Corgi",
                         "Corgi SMCI 2x Daily ETF", 460_000.0),
    "SMCL": LeveragedEtf("SMCL", "SMCI", 2.0, "GraniteShares",
                         "GraniteShares 2x Long SMCI Daily ETF", 52_280_000.0),
    "SMCX": LeveragedEtf("SMCX", "SMCI", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long SMCI ETF", 187_680_000.0),
    "SMCZ": LeveragedEtf("SMCZ", "SMCI", -2.0, "Defiance",
                         "Defiance Daily Target 2X Short SMCI ETF", 1_050_000.0),
    # --- SMR ------------------------------------------------------------
    "SMU": LeveragedEtf("SMU", "SMR", 2.0, "Tradr",
                        "Tradr 2X Long SMR Daily ETF", 42_710_000.0),
    "SMUP": LeveragedEtf("SMUP", "SMR", 2.0, "T-Rex",
                         "T-Rex 2X Long SMR Daily Target ETF", 5_820_000.0),
    # --- SNDK -----------------------------------------------------------
    "SNXX": LeveragedEtf("SNXX", "SNDK", 2.0, "Tradr",
                         "Tradr 2X Long SNDK Daily ETF", 1_943_880_000.0),
    # --- SNOW -----------------------------------------------------------
    "SNOU": LeveragedEtf("SNOU", "SNOW", 2.0, "T-Rex",
                         "T-Rex 2X Long SNOW Daily Target ETF", 25_520_000.0),
    # --- SOFI -----------------------------------------------------------
    "SOFX": LeveragedEtf("SOFX", "SOFI", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long SOFI ETF", 53_910_000.0),
    # --- SOUN -----------------------------------------------------------
    "SOUX": LeveragedEtf("SOUX", "SOUN", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long SOUN ETF", 7_050_000.0),
    # --- TEM ------------------------------------------------------------
    "TEMT": LeveragedEtf("TEMT", "TEM", 2.0, "Tradr",
                         "Tradr 2X Long TEM Daily ETF", 48_680_000.0),
    # --- TSLA -----------------------------------------------------------
    "TSDD": LeveragedEtf("TSDD", "TSLA", -2.0, "GraniteShares",
                         "GraniteShares 2x Short TSLA Daily ETF", 27_990_000.0),
    "TSL": LeveragedEtf("TSL", "TSLA", 1.25, "GraniteShares",
                        "GraniteShares 1.25x Long TSLA Daily ETF", 12_910_000.0),
    "TSLG": LeveragedEtf("TSLG", "TSLA", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long TSLA Daily ETF", 55_790_000.0),
    "TSLI": LeveragedEtf("TSLI", "TSLA", 2.0, "ProShares",
                         "ProShares Ultra TSLA ETF", 3_810_000.0),
    "TSLL": LeveragedEtf("TSLL", "TSLA", 2.0, "Direxion",
                         "Direxion Daily TSLA Bull 2X ETF", 4_122_130_000.0),
    "TSLQ": LeveragedEtf("TSLQ", "TSLA", -2.0, "Tradr",
                         "Tradr 2X Short TSLA Daily ETF", 88_970_000.0),
    "TSLR": LeveragedEtf("TSLR", "TSLA", 2.0, "GraniteShares",
                         "GraniteShares 2x Long TSLA Daily ETF", 80_140_000.0),
    "TSLS": LeveragedEtf("TSLS", "TSLA", -1.0, "Direxion",
                         "Direxion Daily TSLA Bear 1X ETF", 44_700_000.0),
    "TSLT": LeveragedEtf("TSLT", "TSLA", 2.0, "T-Rex",
                         "T-Rex 2X Long Tesla Daily Target ETF", 163_790_000.0),
    "TSLZ": LeveragedEtf("TSLZ", "TSLA", -2.0, "T-Rex",
                         "T-Rex 2X Inverse Tesla Daily Target ETF", 25_900_000.0),
    # --- TSM ------------------------------------------------------------
    "TSMG": LeveragedEtf("TSMG", "TSM", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long TSM Daily ETF", 38_030_000.0),
    "TSMU": LeveragedEtf("TSMU", "TSM", 2.0, "GraniteShares",
                         "GraniteShares 2x Long TSM Daily ETF", 52_130_000.0),
    "TSMX": LeveragedEtf("TSMX", "TSM", 2.0, "Direxion",
                         "Direxion Daily TSM Bull 2X ETF", 612_100_000.0),
    "TSMZ": LeveragedEtf("TSMZ", "TSM", -1.0, "Direxion",
                         "Direxion Daily TSM Bear 1X ETF", 2_360_000.0),
    # --- UBER -----------------------------------------------------------
    "UBRL": LeveragedEtf("UBRL", "UBER", 2.0, "GraniteShares",
                         "GraniteShares 2x Long UBER Daily ETF", 29_610_000.0),
    # --- UNH ------------------------------------------------------------
    "UNHG": LeveragedEtf("UNHG", "UNH", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long UNH Daily ETF", 63_690_000.0),
    # --- UPST -----------------------------------------------------------
    "UPSX": LeveragedEtf("UPSX", "UPST", 2.0, "Tradr",
                         "Tradr 2X Long UPST Daily ETF", 16_770_000.0),
    # --- VRT ------------------------------------------------------------
    "VRTL": LeveragedEtf("VRTL", "VRT", 2.0, "GraniteShares",
                         "GraniteShares 2x Long VRT Daily ETF", 43_100_000.0),
    # --- VST ------------------------------------------------------------
    "VSTL": LeveragedEtf("VSTL", "VST", 2.0, "Defiance",
                         "Defiance Daily Target 2X Long VST ETF", 17_920_000.0),
    # --- XOM ------------------------------------------------------------
    "XOMX": LeveragedEtf("XOMX", "XOM", 2.0, "Direxion",
                         "Direxion Daily XOM Bull 2X ETF", 6_040_000.0),
    # --- XYZ ------------------------------------------------------------
    "XYZG": LeveragedEtf("XYZG", "XYZ", 2.0, "Leverage Shares",
                         "Leverage Shares 2X Long XYZ Daily ETF", 1_230_000.0),
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

"""Build the ``LEVERAGED_SINGLE_STOCK_ETFS`` map from a scraped ETFdb listing.

One-off research helper paired with :mod:`app.research.parse_etfdb_paste`. It reads
the newest ``app/research/data/etfdb_single_stock_etfs_*.json``, keeps the
daily-leveraged single-stock ETFs, derives ``(underlying, leverage, family)``
from each fund's name and issuer, prints a review table, and can emit a ready
``LEVERAGED_SINGLE_STOCK_ETFS = {...}`` literal for
``app/strategies/leveraged_single_stock_etfs.py``.

Funds that are *not* daily-leveraged trackers are dropped: option-income /
buy-write ("YieldBOOST", "WeeklyPay", "Autocallable", "Option Income",
"BuyWrite", "Growth & Income", "Yield Premium") and currency-hedged
("ADRhedged") products. Funds whose underlying has no tradable symbol (private
companies such as SpaceX) are also dropped.

Usage::

    python -m app.research.build_etf_map                 # review table
    python -m app.research.build_etf_map --emit block.py # write the dict literal
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = ["build_map", "render_block", "main"]

DATA_DIR = Path(__file__).parent / "data"
STRATEGY = (Path(__file__).resolve().parent.parent
            / "strategies" / "leveraged_single_stock_etfs.py")

# A fund is a daily-leveraged tracker if its name advertises leverage.
_LEVERAGED_RE = re.compile(
    r"(?i)(\b[0-9](?:\.[0-9]+)?\s*x\b|\bbull\b|\bbear\b|\binverse\b"
    r"|\bshort\b|\blong\b|\bdaily target\b|\bultra\b)")
# ...and is not one of the excluded product types.
_EXCLUDE_RE = re.compile(
    r"(?i)(option income|yieldmax|yield ?boost|covered call|premium income"
    r"|weeklypay|autocallable|adrhedged|buywrite|buy-write|growth & income"
    r"|yield premium|revolution|leveraged long income)")

# Names that spell the underlying out in words instead of using its ticker.
_COMPANY_TICKERS = {
    "alphabet": "GOOGL", "apple": "AAPL", "microsoft": "MSFT",
    "tesla": "TSLA", "nvidia": "NVDA",
}
# Underlyings whose fund-style symbol differs from the tradable one.
_UNDERLYING_OVERRIDES = {"BRKB": "BRK.B"}
# No tradable underlying -> cannot produce a dealer-hedge signal.
_PRIVATE_COMPANIES = {"spacex"}

# Issuer (regex) -> family label used in the map.
_FAMILIES: List[Tuple[str, str]] = [
    (r"leverage\s*shares", "Leverage Shares"),
    (r"graniteshares", "GraniteShares"),
    (r"direxion", "Direxion"),
    (r"defiance", "Defiance"),
    (r"tradr", "Tradr"),
    (r"t[\s-]*rex", "T-Rex"),
    (r"proshares", "ProShares"),
    (r"kraneshares", "KraneShares"),
    (r"corgi", "Corgi"),
]

_UNDERLYING_PATTERNS = [
    # "2x Long INTC", "2X Short COIN", "1.5X Inverse NVDA"
    r"(?i)\b[0-9](?:\.[0-9]+)?\s*x\s*(?:long|bull|short|bear|inverse)\s+([A-Za-z][A-Za-z0-9.]*)",
    # "AAPL Bull 2X", "PLTR Bear 1X"
    r"(?i)([A-Za-z][A-Za-z0-9.]*)\s+(?:bull|bear)\s+[0-9](?:\.[0-9]+)?\s*x",
    # "ProShares Ultra CRCL"
    r"(?i)\bultra\s+([A-Za-z][A-Za-z0-9.]*)",
    # "Corgi SMCI 2x Daily ETF"
    r"(?i)^[A-Za-z]+\s+([A-Za-z][A-Za-z0-9.]*)\s+[0-9](?:\.[0-9]+)?\s*x\b",
    # generic fallback: "<lev>x <UND>"
    r"(?i)\b[0-9](?:\.[0-9]+)?\s*x\s+([A-Za-z][A-Za-z0-9.]*)",
]


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    name = name.replace("Dailly", "Daily")          # etfdb typo
    name = re.sub(r"(?i)\bT\s*-?\s*REX\b", "T-Rex", name)
    name = re.sub(r"(?i)\s+New\s+[A-Za-z]+\s+\d{4}$", "", name)
    return name


def _parse_underlying(name: str) -> Optional[str]:
    for pattern in _UNDERLYING_PATTERNS:
        match = re.search(pattern, name)
        if match:
            return match.group(1)
    return None


def _parse_leverage(name: str) -> Optional[float]:
    match = re.search(r"([0-9](?:\.[0-9]+)?)\s*[xX]\b", name)
    if match:
        magnitude = float(match.group(1))
    elif re.search(r"(?i)\bultra\b", name):
        magnitude = 2.0
    else:
        return None
    if re.search(r"(?i)\b(short|bear|inverse)\b", name):
        magnitude = -magnitude
    return magnitude


def _family_of(name: str) -> str:
    for pattern, family in _FAMILIES:
        if re.search(pattern, name, re.I):
            return family
    return "Unknown"


def _resolve_underlying(token: str) -> Optional[str]:
    key = token.lower()
    if key in _PRIVATE_COMPANIES:
        return None
    underlying = _COMPANY_TICKERS.get(key) or token.upper()
    return _UNDERLYING_OVERRIDES.get(underlying, underlying)


def build_map(rows: List[Dict]) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Return (entries, dropped) for the scraped rows."""
    entries: List[Dict] = []
    dropped: Dict[str, List[Dict]] = {"income": [], "private": [], "unresolved": []}
    for row in rows:
        name = _clean_name(row["name"])
        if not _LEVERAGED_RE.search(name):
            dropped["income"].append(row)
            continue
        if _EXCLUDE_RE.search(name):
            dropped["income"].append(row)
            continue
        token = _parse_underlying(name)
        leverage = _parse_leverage(name) if token else None
        underlying = _resolve_underlying(token) if token else None
        if token is None or leverage is None:
            dropped["unresolved"].append(row)
            continue
        if underlying is None:
            dropped["private"].append(row)
            continue
        entries.append({
            "etf": row["symbol"],
            "underlying": underlying,
            "leverage": leverage,
            "family": _family_of(name),
            "name": name,
            "aum_usd": row.get("aum_usd"),
        })
    entries.sort(key=lambda e: (e["underlying"], e["etf"]))
    return entries, dropped


def _leverage_literal(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") + ("" if value != int(value) else ".0")


def _aum_literal(aum_usd: Optional[float]) -> str:
    if not aum_usd:
        return ""
    return f", {int(round(aum_usd)):_.1f}"


def render_block(entries: List[Dict]) -> str:
    """Render the ``LEVERAGED_SINGLE_STOCK_ETFS`` dict literal."""
    lines = ["LEVERAGED_SINGLE_STOCK_ETFS: Dict[str, LeveragedEtf] = {"]
    current_underlying = None
    for entry in entries:
        if entry["underlying"] != current_underlying:
            current_underlying = entry["underlying"]
            header = f"    # --- {current_underlying} "
            lines.append(header.ljust(74, "-"))
        prefix = f'    "{entry["etf"]}": LeveragedEtf('
        indent = " " * len(prefix)
        lines.append(
            prefix
            + f'"{entry["etf"]}", "{entry["underlying"]}", '
            + f'{_leverage_literal(entry["leverage"])}, "{entry["family"]}",'
        )
        lines.append(
            indent
            + f'"{entry["name"]}"{_aum_literal(entry["aum_usd"])}),'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _load_scrape(path: Optional[Path]) -> Dict:
    if path is None:
        candidates = sorted(DATA_DIR.glob("etfdb_single_stock_etfs_*.json"))
        if not candidates:
            raise SystemExit(f"no scrape found in {DATA_DIR}")
        path = candidates[-1]
    print(f"scrape: {path}")
    return json.loads(path.read_text())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scrape", type=Path, default=None,
                        help="scraped JSON (default: newest in app/research/data)")
    parser.add_argument("--emit", type=Path, default=None,
                        help="write the dict literal here instead of printing the table")
    args = parser.parse_args(argv)

    payload = _load_scrape(args.scrape)
    entries, dropped = build_map(payload["rows"])

    if args.emit:
        args.emit.write_text(render_block(entries), encoding="utf-8")
        print(f"wrote {len(entries)} entries -> {args.emit}")
        return 0

    for entry in entries:
        aum = f'{entry["aum_usd"] / 1e6:9.2f}' if entry["aum_usd"] else "     n/a"
        print(f'{entry["etf"]:6s} {entry["underlying"]:6s} '
              f'{_leverage_literal(entry["leverage"]):>5s} {entry["family"]:15s} '
              f'{aum}  {entry["name"]}')
    print(f"\nentries: {len(entries)}  dropped(income/other): {len(dropped['income'])}"
          f"  dropped(private underlying): {len(dropped['private'])}"
          f"  unresolved: {len(dropped['unresolved'])}")
    for row in dropped["unresolved"]:
        print(f"  UNRESOLVED {row['symbol']:6s} {row['name']}")
    for row in dropped["private"]:
        print(f"  PRIVATE    {row['symbol']:6s} {row['name']}")
    unknown = [e for e in entries if e["family"] == "Unknown"]
    for entry in unknown:
        print(f"  UNKNOWN FAMILY {entry['etf']} {entry['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Parse copy-pasted ETF Database single-stock-ETF table text into rows.

etfdb.com blocks scripted access, so instead of scraping, paste the theme table
(https://etfdb.com/themes/single-stock-etfs/) text by hand and let this module
turn it into the same structured rows the map builder consumes.

Copy the table from page 1 through the last page and save it to
``app/research/data/etfdb_paste.txt`` (or pipe it on stdin). Each row arrives as
concatenated Markdown links and values, for example::

    [MUU](https://etfdb.com/etf/MUU/)[Direxion Daily MU Bull 2X ETF](...)
    Equity$4,454.36604.04%34,781,472.0$35.14-3.38%[](/members/join/)

which decodes to Symbol=MUU, Name="Direxion Daily MU Bull 2X ETF",
Asset Class=Equity, AUM=$4,454.36MM, YTD=604.04%, Avg Volume=34,781,472.0,
Prev Close=$35.14, 1-Day Change=-3.38%. The trailing empty link is the
"Overall Rating" cell and is ignored.

Usage::

    python -m app.research.parse_etfdb_paste                 # reads data/etfdb_paste.txt
    python -m app.research.parse_etfdb_paste my_paste.txt
    pbpaste | python -m app.research.parse_etfdb_paste -

Writes ``etfdb_single_stock_etfs_<YYYY-MM-DD>.json`` and ``.csv`` next to the
paste file, overwriting any earlier snapshot for the same day, then run
``python -m app.research.build_etf_map`` to regenerate the strategy map.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["parse_rows", "write_outputs", "main"]

DATA_DIR = Path(__file__).parent / "data"

_CSV_FIELDS = ["symbol", "name", "asset_class", "aum_mm", "aum_usd",
               "ytd_return_pct", "avg_volume_3m", "prev_close",
               "one_day_change_pct"]

# One table row. Fields are glued together in the copied text, so each is
# matched by shape rather than by a separator.
#   [SYM](url)[NAME](url) <AssetClass> <$AUM> <YTD%> <AvgVolume> <$Price> <Chg%>
_ROW_RE = re.compile(
    r"\[(?P<symbol>[A-Z][A-Z0-9.]*)\]\([^)]*\)"
    r"\[(?P<name>[^\]]+)\]\([^)]*\)"
    r"(?P<asset_class>[A-Za-z][A-Za-z /&-]*?)"
    r"(?P<aum>\$[\d,]+(?:\.\d+)?|N/A)"
    r"(?P<ytd>[+-]?[\d.]+%|N/A)"
    r"(?P<volume>[\d,]+(?:\.\d+)?|N/A)"
    r"(?P<price>\$[\d,]+(?:\.\d+)?|N/A)"
    r"(?P<change>[+-]?[\d.]+%|N/A)"
)


def _to_float(value: Any) -> Optional[float]:
    """Parse ``"$4,454.36"`` / ``"604.04%"`` / ``"N/A"`` into a float or None."""
    if value is None:
        return None
    text = str(value).replace("$", "").replace(",", "").replace("%", "").strip()
    text = text.lstrip("+")
    if text in ("", "-", "--", "N/A"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_rows(text: str) -> List[Dict[str, Any]]:
    """Extract one dict per fund from pasted table text, de-duplicated by ticker."""
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for match in _ROW_RE.finditer(text):
        group = match.groupdict()
        symbol = group["symbol"].strip().upper()
        if symbol in seen:
            continue
        seen.add(symbol)
        aum_mm = _to_float(group["aum"])
        rows.append({
            "symbol": symbol,
            "name": re.sub(r"\s+", " ", group["name"]).strip(),
            "asset_class": group["asset_class"].strip(),
            "aum_mm": aum_mm,
            "aum_usd": aum_mm * 1_000_000 if aum_mm is not None else None,
            "ytd_return_pct": _to_float(group["ytd"]),
            "avg_volume_3m": _to_float(group["volume"]),
            "prev_close": _to_float(group["price"]),
            "one_day_change_pct": _to_float(group["change"]),
        })
    return rows


def write_outputs(rows: List[Dict[str, Any]], outdir: Path,
                  source: str) -> Dict[str, Path]:
    """Write the parsed rows as JSON + CSV in the shared snapshot format."""
    outdir.mkdir(parents=True, exist_ok=True)
    asof = date.today().isoformat()
    json_path = outdir / f"etfdb_single_stock_etfs_{asof}.json"
    csv_path = outdir / f"etfdb_single_stock_etfs_{asof}.csv"

    payload = {
        "source": source,
        "method": "pasted table text",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(rows),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {"json": json_path, "csv": csv_path}


def _read_input(path: Optional[str]) -> str:
    if path in (None, "-"):
        if path is None:
            default = DATA_DIR / "etfdb_paste.txt"
            if default.exists():
                print(f"reading {default}")
                return default.read_text(encoding="utf-8")
        print("reading stdin")
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=None,
                        help="paste file (default: data/etfdb_paste.txt); '-' for stdin")
    parser.add_argument("--outdir", type=Path, default=DATA_DIR,
                        help="where to write the JSON/CSV snapshot")
    args = parser.parse_args(argv)

    text = _read_input(args.input)
    rows = parse_rows(text)
    if not rows:
        print("No rows parsed. Is the pasted text in the expected format (see module docstring)?",
              file=sys.stderr)
        return 1

    paths = write_outputs(rows, args.outdir, source=str(args.input or "stdin"))
    print(f"parsed {len(rows)} rows")
    for row in rows[:5]:
        print(f'  {row["symbol"]:6s} {row["aum_mm"]:>10.2f}  {row["name"]}')
    print(f"  ... (+{max(0, len(rows) - 5)} more)")
    print("wrote:")
    for kind, path in paths.items():
        print(f"  {kind:4s} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

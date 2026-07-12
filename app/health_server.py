"""
Flask-based health-check and dashboard server for keep-alive mode.

Exposes:
  /health         — JSON health check (no auth, for Docker HEALTHCHECK)
  /api/positions  — JSON positions data (basic auth)
  /               — Dashboard HTML frontend (basic auth)
  /static/        — Static assets: CSS, JS (basic auth)

Uses Waitress (production WSGI) so it runs safely in a daemon thread
without the signal-handler limitations of Flask's dev server.
"""

import logging
import os
from datetime import datetime
from functools import wraps
from typing import Any, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from waitress import serve

logger = logging.getLogger(__name__)

# Resolve the frontend directory relative to this file (app/ → ../frontend/)
_FRONTEND_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'frontend')
)
_STATIC_DIR = os.path.join(_FRONTEND_DIR, 'static')


# ---------------------------------------------------------------------------
# Basic Auth helpers
# ---------------------------------------------------------------------------

def _check_auth(username: str, password: str) -> bool:
    """Validate credentials against DASHBOARD_PASSWORD env var."""
    dashboard_password = os.getenv('DASHBOARD_PASSWORD', '')
    if not dashboard_password:
        return False
    return username == 'admin' and password == dashboard_password


def _auth_required(f):
    """Decorator that enforces HTTP Basic Auth on a Flask route.

    Returns 503 if DASHBOARD_PASSWORD is not configured (fail-closed).
    Returns 401 with WWW-Authenticate header if credentials are missing or wrong.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        dashboard_password = os.getenv('DASHBOARD_PASSWORD', '')
        if not dashboard_password:
            return jsonify({
                'error': 'DASHBOARD_PASSWORD environment variable is not set. '
                         'Set it to enable dashboard access.'
            }), 503
        auth = request.authorization
        if not auth or not _check_auth(auth.username, auth.password):
            return jsonify({'error': 'Unauthorized'}), 401, {
                'WWW-Authenticate': 'Basic realm="Trading Dashboard"'
            }
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _df_row_to_dict(row) -> dict:
    """Convert a pandas DataFrame row to a JSON-safe dict.

    Converts pandas Timestamps to ISO 8601 strings.
    Converts numpy numeric types to native Python types.
    Normalizes storage column names to frontend field names:
      - shares → quantity
      - Derives side from quantity if missing (negative = short)
    """
    d = {}
    for key, value in row.items() if hasattr(row, 'items') else row._asdict().items():
        # Normalize NaN / NA → None
        if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            value = None
        elif hasattr(value, 'item'):  # numpy scalar → native Python
            value = value.item()
        d[key] = value

    # Normalize column names: storage uses 'shares', frontend expects 'quantity'
    if 'shares' in d and 'quantity' not in d:
        d['quantity'] = d.pop('shares')

    # Normalize 'closed' to a real boolean (CSV stores strings "True"/"False",
    # and JavaScript treats non-empty strings as truthy, breaking filters).
    if 'closed' in d:
        raw = d['closed']
        if isinstance(raw, str):
            d['closed'] = raw.strip().lower() in ('true', '1', 'yes')
        elif isinstance(raw, (int, float, np.integer, np.floating)):
            d['closed'] = bool(raw) and not pd.isna(raw)
        elif raw is None or raw is pd.NA:
            d['closed'] = False

    # Derive side if not present (GCS backend doesn't store it)
    if 'side' not in d or d['side'] is None:
        qty = d.get('quantity', 0) or 0
        d['side'] = 'short' if qty < 0 else 'long'

    # Convert datetime columns to ISO strings
    for dt_col in ('entry_date', 'exit_date'):
        val = d.get(dt_col)
        if isinstance(val, (pd.Timestamp, datetime)):
            d[dt_col] = val.isoformat()

    # Ensure exit_reason is present (GCS doesn't store it)
    if 'exit_reason' not in d:
        d['exit_reason'] = None

    return d


def _fetch_positions_from_storage(storage_backend) -> list[dict]:
    """Fetch the latest position snapshot from storage and return JSON-safe dicts.

    Reads the most recent position file (via the abstract StorageBackend
    interface).  Does NOT filter by open/closed — the frontend handles that.
    """
    # Step 1: list all files to verify "latest" is what we think it is
    all_files = storage_backend.list_position_files()
    all_files.sort()
    logger.info("Position files in storage (%d total): %s ... %s",
                len(all_files),
                all_files[:3] if len(all_files) >= 3 else all_files,
                all_files[-3:] if len(all_files) >= 3 else [])

    latest_file = storage_backend.get_latest_position_file()
    if not latest_file:
        logger.warning("No position files found in storage")
        return []
    logger.info("Latest position file selected: %s", latest_file)

    # Step 2: load raw dataframe
    df = storage_backend.load_position_entries(latest_file)
    if df is None or df.empty:
        logger.warning(
            "Position file %s loaded but is empty/None", latest_file)
        return []

    logger.info("Raw DataFrame: %d rows × %d cols — columns: %s",
                len(df), len(df.columns), list(df.columns))
    logger.info("dtypes sample: %s",
                {c: str(dt) for c, dt in zip(df.columns[:8], df.dtypes[:8])})

    # Step 3: inspect the 'closed' column raw values
    if 'closed' in df.columns:
        closed_series = df['closed']
        logger.info(
            "'closed' column — unique values: %s, "
            "counts: True-ish=%d, False-ish=%d, NaN/None=%d",
            closed_series.dropna().unique().tolist(),
            int(closed_series.fillna(False).astype(bool).sum()),
            int((~closed_series.fillna(False).astype(bool)).sum()),
            int(closed_series.isna().sum()),
        )
    else:
        logger.warning(
            "No 'closed' column in DataFrame — all rows treated as-is")

    # Step 4: log before/after serialization
    logger.info("Serializing %d rows (NO FILTERING) ...", len(df))
    rows = [_df_row_to_dict(row) for _, row in df.iterrows()]

    # Quick sanity: what fraction of the output has closed=True?
    n_closed_out = sum(1 for r in rows if r.get('closed'))
    n_open_out = len(rows) - n_closed_out
    logger.info("Serialized %d total → %d open, %d closed",
                len(rows), n_open_out, n_closed_out)

    if rows:
        logger.info("First row keys: %s, sample: %s",
                    list(rows[0].keys()),
                    {k: rows[0][k] for k in ['symbol', 'closed', 'quantity', 'entry_price'] if k in rows[0]})

    return rows


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(storage_backend=None, shared_state: Optional[dict[str, Any]] = None, data_provider=None):
    """Create and configure the Flask application.

    Args:
        storage_backend: StorageBackend instance for reading position snapshots.
        shared_state: Mutable dict with a 'last_result' key that gets updated
                      after the trading cycle completes.  If None, the server
                      reports 'running' indefinitely.
        data_provider: DataProvider instance for live Alpaca data (open orders).
    """
    app = Flask(__name__, static_folder=None)

    # ---------- /health (no auth — Docker HEALTHCHECK) ----------

    @app.route('/health')
    def health():
        result = shared_state.get('last_result') if shared_state else None
        env = os.getenv('ENVIRONMENT', 'dev')
        paper = os.getenv('PAPER_TRADE', 'true').lower() in (
            'true', '1', 'yes')

        if result is None:
            return jsonify({
                'status': 'running',
                'last_run_status': 'running',
                'last_run_summary': {},
                'last_run_backtest_count': 0,
                'last_run_duration_seconds': 0,
                'environment': env,
                'paper_trade': paper,
            })
        return jsonify({
            'status': 'idle',
            'last_run_status': result.get('status', 'unknown'),
            'last_run_summary': result.get('trading_summary', {}),
            'last_run_backtest_count': result.get('backtest_count', 0),
            'last_run_duration_seconds': result.get('duration', 0),
            'environment': env,
            'paper_trade': paper,
        })

    # ---------- /api/positions (auth required) ----------

    @app.route('/api/positions')
    @_auth_required
    def api_positions():
        if storage_backend is None:
            return jsonify({'error': 'Storage backend not available'}), 503

        try:
            rows = _fetch_positions_from_storage(storage_backend)
            return jsonify(rows)
        except Exception as e:
            logger.error("Error fetching positions from storage: %s", e)
            return jsonify({'error': 'Failed to fetch positions from storage'}), 500

    # ---------- /api/open-orders (auth required) ----------

    @app.route('/api/open-orders')
    @_auth_required
    def api_open_orders():
        if data_provider is None:
            return jsonify({'error': 'Data provider not available'}), 503

        try:
            df = data_provider.get_open_orders()
            if df.empty:
                return jsonify([])

            # Convert to JSON-safe dicts
            rows = []
            for _, row in df.iterrows():
                d = {}
                for key, value in row.items():
                    if hasattr(value, 'item'):
                        value = value.item()
                    elif pd.isna(value):
                        value = None
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        value = value.isoformat()
                    d[key] = value
                rows.append(d)
            logger.info("Returning %d open orders", len(rows))
            return jsonify(rows)
        except Exception as e:
            logger.error("Error fetching open orders: %s", e)
            return jsonify({'error': 'Failed to fetch open orders'}), 500

    # ---------- /api/live-alpaca (auth required) ----------
    #
    # Returns per-symbol live data pulled directly from the Alpaca API:
    #   - current_price    (from snapshots)
    #   - stop_loss_order  (bracket leg: status, stop_price, order_id)
    #   - take_profit_order(bracket leg: status, limit_price, order_id)
    #
    # Symbols with no open bracket legs still get current_price if they
    # have an open position stored.

    @app.route('/api/live-alpaca')
    @_auth_required
    def api_live_alpaca():
        if data_provider is None:
            return jsonify({'error': 'Data provider not available'}), 503

        try:
            # 1. Get all open bracket-leg orders from Alpaca
            orders_df = data_provider.get_open_orders()

            # 2. Collect symbols from ALL sources:
            #    (a) open bracket-leg orders from Alpaca
            #    (b) open positions from storage (so every position row gets
            #        a snapshot price even if no bracket orders exist)
            symbol_set: set = set()
            if not orders_df.empty:
                symbol_set.update(
                    orders_df['symbol'].dropna().astype(str).tolist())

            if storage_backend is not None:
                try:
                    pos_df = storage_backend.get_latest_positions_df(
                        openPosition=True)
                    if not pos_df.empty and 'symbol' in pos_df.columns:
                        symbol_set.update(
                            pos_df['symbol'].dropna().astype(str).tolist())
                except Exception:
                    pass  # best-effort; don't fail the whole endpoint

            # 3. Build per-symbol result dict
            result: dict = {}

            for symbol in sorted(symbol_set):
                entry: dict = {
                    'current_price': None,
                    'stop_loss_order': None,
                    'take_profit_order': None,
                }

                # — current price via snapshot —
                try:
                    snapshot = data_provider.get_current_snapshot(symbol)
                    if snapshot:
                        price = snapshot.get('price')
                        if price is not None:
                            entry['current_price'] = round(float(price), 2)
                except Exception:
                    logger.exception("Snapshot failed for %s", symbol)
                    # Leave current_price = None; frontend falls back to stored value

                # — stop-loss leg —
                sl_rows = pd.DataFrame()
                if not orders_df.empty:
                    sl_rows = orders_df[
                        (orders_df['symbol'] == symbol) &
                        (orders_df['leg_type'] == 'stop_loss')
                    ]
                if not sl_rows.empty:
                    sl = sl_rows.iloc[0]
                    entry['stop_loss_order'] = {
                        'order_id': (
                            str(sl['order_id']) if pd.notna(
                                sl['order_id']) and sl['order_id'] is not None else None
                        ),
                        'status': (
                            str(sl['status']) if pd.notna(sl['status']
                                                          ) and sl['status'] is not None else None
                        ),
                        'stop_price': (
                            round(float(sl['stop_price']), 2)
                            if pd.notna(sl['stop_price']) and sl['stop_price'] is not None
                            else None
                        ),
                        'created_at': (
                            sl['created_at'] if pd.notna(
                                sl.get('created_at')) else None
                        ),
                    }

                # — take-profit leg —
                tp_rows = pd.DataFrame()
                if not orders_df.empty:
                    tp_rows = orders_df[
                        (orders_df['symbol'] == symbol) &
                        (orders_df['leg_type'] == 'take_profit')
                    ]
                if not tp_rows.empty:
                    tp = tp_rows.iloc[0]
                    entry['take_profit_order'] = {
                        'order_id': (
                            str(tp['order_id']) if pd.notna(
                                tp['order_id']) and tp['order_id'] is not None else None
                        ),
                        'status': (
                            str(tp['status']) if pd.notna(tp['status']
                                                          ) and tp['status'] is not None else None
                        ),
                        'limit_price': (
                            round(float(tp['limit_price']), 2)
                            if pd.notna(tp['limit_price']) and tp['limit_price'] is not None
                            else None
                        ),
                        'created_at': (
                            tp['created_at'] if pd.notna(
                                tp.get('created_at')) else None
                        ),
                    }

                result[symbol] = entry

            logger.info(
                "Returning live Alpaca data for %d symbols", len(result))
            return jsonify(result)

        except Exception as e:
            logger.error("Error fetching live Alpaca data: %s", e)
            return jsonify({'error': 'Failed to fetch live Alpaca data'}), 500

    # ---------- / (dashboard HTML, auth required) ----------

    @app.route('/')
    @_auth_required
    def index():
        return send_from_directory(_FRONTEND_DIR, 'index.html')

    # ---------- /static/<path> (auth required) ----------

    @app.route('/static/<path:filename>')
    @_auth_required
    def static_files(filename):
        return send_from_directory(_STATIC_DIR, filename)

    return app


# ---------------------------------------------------------------------------
# Server launcher
# ---------------------------------------------------------------------------

def start_health_server(port: int, shared_state: dict[str, Any], storage_backend=None, data_provider=None):
    """Start the dashboard server via Waitress. Blocks until stopped.

    Designed to run in a daemon thread — Waitress does not register signal
    handlers, so it works safely outside the main thread.

    Called from main.py when KEEP_ALIVE is enabled.

    Args:
        port: TCP port to listen on (defaults to 8080).
        shared_state: Mutable dict.  Its 'last_result' key is read by
                      /health and updated in-place after the cycle finishes.
        storage_backend: StorageBackend instance for /api/positions.
        data_provider: DataProvider instance for live Alpaca data.
    """
    app = create_app(
        storage_backend=storage_backend,
        shared_state=shared_state,
        data_provider=data_provider,
    )

    env = os.getenv('ENVIRONMENT', 'dev')
    logger.info(
        "🏥 Health + dashboard server listening on 0.0.0.0:%d (env=%s)",
        port, env,
    )
    # Waitress is a production WSGI server — no reloader, no signals, thread-safe
    serve(app, host='0.0.0.0', port=port, _quiet=True)

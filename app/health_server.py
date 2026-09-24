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
        # Normalize NaN / NA / NaT / ±Infinity → None.
        # NOTE: pd.NaT must be checked explicitly — it is *not* pd.NA, is not a
        # float, and (critically) IS an instance of datetime, so without this
        # guard it would reach the isoformat() step below and serialize as the
        # literal string "NaT" instead of null.
        if value is pd.NA or value is pd.NaT or (
                isinstance(value, float) and not np.isfinite(value)):
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

    # Derive side if not present (legacy snapshots don't store it)
    if 'side' not in d or d['side'] is None:
        qty = d.get('quantity', 0) or 0
        d['side'] = 'short' if qty < 0 else 'long'

    # Convert datetime columns to ISO strings
    for dt_col in ('entry_date', 'exit_date'):
        val = d.get(dt_col)
        if isinstance(val, (pd.Timestamp, datetime)) and not pd.isna(val):
            d[dt_col] = val.isoformat()

    # Ensure exit_reason is present (legacy snapshots don't store it)
    if 'exit_reason' not in d:
        d['exit_reason'] = None

    # Ensure strategy_name is present (legacy rows predate the multi-strategy
    # column); default to the legacy RSI strategy so the UI always has a tag.
    if 'strategy_name' not in d or d['strategy_name'] is None:
        d['strategy_name'] = 'rsi_mean_reversion'

    return d


def _fetch_positions_from_storage(storage_backend) -> list[dict]:
    """Fetch the latest position snapshot from storage and return JSON-safe dicts.

    Delegates the "find latest snapshot + load rows" logic to the storage
    backend (``get_latest_positions_df(openPosition=None)`` = all rows,
    unfiltered — the frontend handles open/closed filtering).
    """
    df = storage_backend.get_latest_positions_df(openPosition=None)
    if df is None or df.empty:
        logger.warning("No position snapshots found in storage")
        return []

    logger.info("Latest snapshot: %d rows × %d cols", len(df), len(df.columns))
    rows = [_df_row_to_dict(row) for _, row in df.iterrows()]

    n_closed = sum(1 for r in rows if r.get('closed'))
    logger.info("Serialized %d total → %d open, %d closed",
                len(rows), len(rows) - n_closed, n_closed)
    return rows


def _json_safe(value):
    """Recursively convert a value into a JSON-serializable form.

    The trading summary now carries a ``datetime`` top-level timestamp plus a
    ``orders`` list of plain dicts (shares/price/timestamp).  This sanitizer
    normalises datetime/timestamp, NaN/NA/NaT, and numpy scalars so Flask can
    serialise the whole summary without a custom encoder at every call site.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _enabled_strategies() -> list[str]:
    """Return the configured strategy keys (best-effort, never raises).

    Used by /health so the dashboard header can show the real number of enabled
    strategies rather than a backtest-row count.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from config import globalConfig  # type: ignore
        return [str(s) for s in (getattr(globalConfig, 'STRATEGIES_ENABLED', None) or [])]
    except Exception:  # pylint: disable=broad-exception-caught
        return []


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

        cycle_running = shared_state.get(
            'cycle_running', False) if shared_state else False
        overall_status = 'running' if cycle_running else (
            'idle' if result else 'running')

        if result is None and not cycle_running:
            # First cycle hasn't started yet
            overall_status = 'running'

        strategies_enabled = _enabled_strategies()

        if result is None:
            return jsonify({
                'status': overall_status,
                'last_run_status': 'running',
                'last_run_summary': {},
                'last_run_backtest_count': 0,
                'last_run_duration_seconds': 0,
                'environment': env,
                'paper_trade': paper,
                'strategies_enabled': strategies_enabled,
            })
        return jsonify({
            'status': overall_status,
            'last_run_status': result.get('status', 'unknown'),
            'last_run_summary': _json_safe(result.get('trading_summary', {})),
            'last_run_backtest_count': result.get('backtest_count', 0),
            'last_run_duration_seconds': result.get('duration', 0),
            'environment': env,
            'paper_trade': paper,
            'strategies_enabled': strategies_enabled,
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

    # ---------- /api/db/tables (auth required, read-only) ----------
    # Backs the dashboard "Database" tab: lists browsable tables.

    @app.route('/api/db/tables')
    @_auth_required
    def api_db_tables():
        if storage_backend is None or not storage_backend.db_browse_enabled():
            return jsonify({
                'enabled': False,
                'error': 'Table browsing is not supported by the active storage '
                         'backend (requires STORAGE_BACKEND=postgres).',
            }), 501

        try:
            tables = storage_backend.db_list_tables()
            return jsonify({'enabled': True, 'tables': tables})
        except Exception as e:
            logger.error("Error listing database tables: %s", e)
            return jsonify({'error': 'Failed to list database tables'}), 500

    # ---------- /api/db/table/<name> (auth required, read-only) ----------
    # Returns a page of rows from a browsable table.

    @app.route('/api/db/table/<path:name>')
    @_auth_required
    def api_db_table(name):
        if storage_backend is None or not storage_backend.db_browse_enabled():
            return jsonify({
                'enabled': False,
                'error': 'Table browsing is not supported by the active storage '
                         'backend (requires STORAGE_BACKEND=postgres).',
            }), 501

        try:
            limit = min(int(request.args.get('limit', 100)), 500)
            offset = max(int(request.args.get('offset', 0)), 0)
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid limit/offset'}), 400

        try:
            return jsonify(storage_backend.db_fetch_table(name, limit, offset))
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error("Error fetching table %s: %s", name, e)
            return jsonify({'error': f'Failed to fetch table {name}'}), 500

    # ---------- /api/strategy-performance (auth required) ----------
    # Backs the dashboard "Performance" tab: per-strategy historic daily
    # snapshots (one row per strategy per session) with derived cumulative
    # P&L/return and per-strategy summary stats.

    @app.route('/api/strategy-performance')
    @_auth_required
    def api_strategy_performance():
        if storage_backend is None:
            return jsonify({'error': 'Storage backend not available'}), 503

        try:
            rows = storage_backend.load_strategy_performance()
        except Exception as e:
            logger.error("Error loading strategy performance: %s", e)
            return jsonify({'error': 'Failed to load strategy performance'}), 500

        rows = [_json_safe(r) for r in rows]
        strategies: dict[str, dict] = {}
        for row in rows:
            name = row.get('strategy_name') or 'rsi_mean_reversion'
            bucket = strategies.setdefault(name, {
                'strategy_name': name,
                'n_days': 0,
                'first_date': None,
                'last_date': None,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'open_positions': 0,
                'open_market_value': 0.0,
                'series': [],
            })
            bucket['n_days'] += 1
            bucket['first_date'] = bucket['first_date'] or row.get(
                'snapshot_date')
            bucket['last_date'] = row.get('snapshot_date')
            # Each row's realized_pnl is the cumulative closed-position P&L
            # already in the position book, so the latest row IS the total —
            # do not sum across rows.
            bucket['realized_pnl'] = row.get('realized_pnl')
            bucket['open_positions'] = int(row.get('open_positions') or 0)
            bucket['open_market_value'] = row.get('open_market_value')
            bucket['unrealized_pnl'] = row.get('unrealized_pnl')
            bucket['budget_notional'] = row.get('budget_notional')
            bucket['equity'] = row.get('equity')
            bucket['series'].append({
                'date': row.get('snapshot_date'),
                'cumulative_pnl': (
                    float(row.get('realized_pnl') or 0.0)
                    + float(row.get('unrealized_pnl') or 0.0)),
                'unrealized_pnl': row.get('unrealized_pnl'),
                'realized_pnl': row.get('realized_pnl'),
                'open_market_value': row.get('open_market_value'),
                'open_positions': row.get('open_positions'),
            })
        return jsonify({'strategies': list(strategies.values())})

    # (The old /api/open-orders endpoint was removed — the frontend never
    # called it; /api/live-alpaca covers the live-order use case.)

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

    # ---------- /api/run-cycle (auth required) ----------
    #
    # Triggers a new trading cycle in the main thread via a threading.Event.
    # The cron job should POST to this endpoint instead of spawning a new
    # process, which avoids "Address already in use" errors from Waitress.
    #
    # Phase 4: cycle execution is tracked as a background job. The response
    # returns 202 with a job_id; poll /api/jobs/<job_id> for progress.
    #
    # Query params:
    #   force_backtest  (bool)  — skip cached backtest results
    #   dry_run         (bool)  — analyze without placing orders
    #   test_mode       (bool)  — limited universe for fast validation

    @app.route('/api/run-cycle', methods=['POST'])
    @_auth_required
    def api_run_cycle():
        if shared_state is None:
            return jsonify({
                'error': 'Server not configured for cycle execution'
            }), 503

        if shared_state.get('cycle_running', False):
            return jsonify({
                'status': 'already_running',
                'message': 'A trading cycle is already in progress',
            }), 409

        trigger = shared_state.get('trigger_event')
        if trigger is None:
            return jsonify({
                'error': 'Trigger mechanism not available'
            }), 503

        # Register the background job first so the main loop can pick it up.
        # pylint: disable=import-outside-toplevel
        from jobs import job_manager
        job = job_manager.start_job(kind="full_cycle")
        if job is None:
            return jsonify({
                'status': 'already_running',
                'message': 'A background job is already in progress',
            }), 409

        # Capture optional flags from query params
        flags = {}
        for flag in ('force_backtest', 'dry_run', 'test_mode'):
            val = request.args.get(flag, '').lower()
            if val in ('true', '1', 'yes'):
                flags[flag] = True
        flags['job_id'] = job.job_id
        shared_state['cycle_flags'] = flags

        trigger.set()
        logger.info("Trading cycle triggered via API (job=%s flags=%s)",
                    job.job_id, flags)
        return jsonify({
            'status': 'queued',
            'job_id': job.job_id,
            'message': 'Trading cycle queued',
            'flags': flags,
        }), 202

    # ---------- /api/run-session (auth required) ----------
    #
    # Runs a trading session using the LATEST cached backtest results without
    # re-running the (slow) optimization/backtest pass.  Mirrors /api/run-cycle
    # but forces run_session_only=True so the UI button can trade on existing
    # analysis on demand.
    #
    # Query params:
    #   dry_run  (bool) — analyze without placing orders

    @app.route('/api/run-session', methods=['POST'])
    @_auth_required
    def api_run_session():
        if shared_state is None:
            return jsonify({
                'error': 'Server not configured for cycle execution'
            }), 503

        if shared_state.get('cycle_running', False):
            return jsonify({
                'status': 'already_running',
                'message': 'A trading cycle is already in progress',
            }), 409

        trigger = shared_state.get('trigger_event')
        if trigger is None:
            return jsonify({
                'error': 'Trigger mechanism not available'
            }), 503

        # Register the background job (same mechanism as /api/run-cycle).
        # pylint: disable=import-outside-toplevel
        from jobs import job_manager
        job = job_manager.start_job(kind="run_session")
        if job is None:
            return jsonify({
                'status': 'already_running',
                'message': 'A background job is already in progress',
            }), 409

        # Session-only run: reuse the latest cached backtest, skip optimization.
        flags = {'run_session_only': True, 'job_id': job.job_id}

        dry_run = request.args.get('dry_run', '').lower()
        if dry_run in ('true', '1', 'yes'):
            flags['dry_run'] = True

        shared_state['cycle_flags'] = flags

        trigger.set()
        logger.info(
            "Trading session triggered via API (job=%s reuse-latest flags=%s)",
            job.job_id, flags)
        return jsonify({
            'status': 'queued',
            'job_id': job.job_id,
            'message': 'Trading session queued using latest backtest data',
            'flags': flags,
        }), 202

    # ---------- /api/jobs (auth required) ----------
    # Background-job status for the dashboard Jobs panel (Phase 4).

    @app.route('/api/jobs')
    @_auth_required
    def api_jobs():
        # pylint: disable=import-outside-toplevel
        from jobs import job_manager
        return jsonify({'jobs': job_manager.list_jobs()})

    @app.route('/api/jobs/<job_id>')
    @_auth_required
    def api_job_detail(job_id: str):
        # pylint: disable=import-outside-toplevel
        from jobs import job_manager
        job = job_manager.get_job(job_id)
        if job is None:
            return jsonify({'error': 'Job not found'}), 404
        return jsonify(job.to_dict())

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

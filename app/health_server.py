"""
Lightweight HTTP health-check server for keep-alive mode.

Exposes /health with JSON containing the last run's status and timing.
Designed to be extended with live performance visualizations in the future.
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger(__name__)


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler that serves a /health endpoint."""

    result_json: bytes = b""

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(self.result_json)))
            self.end_headers()
            self.wfile.write(self.result_json)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress access logs


def start_health_server(port: int, last_result: dict):
    """Start a minimal HTTP health-check server on the given port.

    Runs in a daemon thread so it doesn't block shutdown.
    Called from main.py when KEEP_ALIVE is enabled.

    Args:
        port: TCP port to listen on (defaults to 8080).
        last_result: The dict returned by TradingAlgorithm.run_full_cycle().
    """
    result_json = json.dumps({
        'status': 'idle',
        'last_run_status': last_result.get('status', 'unknown'),
        'last_run_summary': last_result.get('trading_summary', {}),
        'last_run_backtest_count': last_result.get('backtest_count', 0),
        'last_run_duration_seconds': last_result.get('duration', 0),
    }, default=str).encode('utf-8')

    # Attach the serialized payload to the handler class so each request
    # can serve it without re-serializing.
    HealthHandler.result_json = result_json

    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    logger.info("🏥 Health server listening on 0.0.0.0:%d", port)
    server.serve_forever()

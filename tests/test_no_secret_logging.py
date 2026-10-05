"""Guard: fail if a secret-named environment value is written to output.

Background
----------
``app/config.py`` used to echo the full Alpaca secret (including the LIVE key
in production) from an f-string.  ``detect-secrets`` scans *files* for
hardcoded secrets and cannot see a value that is only read at runtime — so this
check fills that gap statically.

What is flagged
---------------
A ``print(...)`` or ``logger.<level>(...)`` call any of whose arguments read a
secret-looking environment variable *by literal name*, e.g.::

    print(f"... {os.getenv('ALPACA_LIVE_SECRET')}")
    logger.info("key=%s", os.environ['API_KEY'])

Reading such a variable to decide *whether* it is set (for example
``'loaded' if os.getenv(secret_env) else 'Not Set'``) is not flagged: the key is
passed as a variable, and the value is never emitted.
"""
import ast
import re
from pathlib import Path
from typing import List, Optional, Tuple

# Variable names that indicate a credential value.
_SECRET_NAME = re.compile(
    r"(?i)(secret|token|password|passwd|api[_-]?key|access[_-]?key"
    r"|private[_-]?key|credential)"
)

_LOG_METHODS = frozenset({
    "debug", "info", "warning", "warn", "error", "exception", "critical",
    "fatal",
})

_APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _is_os_environ(node: ast.AST) -> bool:
    """True for the ``os.environ`` attribute access."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _literal_secret_key(expr: ast.AST) -> Optional[str]:
    """Return the literal secret-named key read by *expr*, else None.

    Matches ``os.environ['X']``, ``os.getenv('X')`` and
    ``os.environ.get('X')`` where ``X`` looks like a credential name.
    """
    if isinstance(expr, ast.Subscript) and _is_os_environ(expr.value):
        key = expr.slice
        if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                and _SECRET_NAME.search(key.value)):
            return key.value
        return None

    if isinstance(expr, ast.Call):
        func = expr.func
        is_env_get = (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and _is_os_environ(func.value)
        )
        is_os_getenv = (
            isinstance(func, ast.Attribute)
            and func.attr == "getenv"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        )
        if (is_env_get or is_os_getenv) and expr.args:
            key = expr.args[0]
            if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                    and _SECRET_NAME.search(key.value)):
                return key.value
    return None


def _secret_reads(expr: ast.AST) -> List[str]:
    """Collect literal secret-named env keys read anywhere under *expr*."""
    return [key for node in ast.walk(expr)
            if (key := _literal_secret_key(node))]


def _is_output_call(node: ast.AST) -> bool:
    """True for a ``print`` or ``logger.<level>`` call."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "print":
        return True
    return isinstance(func, ast.Attribute) and func.attr in _LOG_METHODS


def find_secret_logging_in_source(
    source: str, filename: str = "<inline>"
) -> List[Tuple[str, int, List[str]]]:
    """Return ``(filename, lineno, [secret_keys])`` for each offending call."""
    tree = ast.parse(source, filename=filename)
    findings: List[Tuple[str, int, List[str]]] = []
    for node in ast.walk(tree):
        if not _is_output_call(node):
            continue
        args = list(node.args) + [kw.value for kw in node.keywords]
        keys: List[str] = []
        for arg in args:
            keys.extend(_secret_reads(arg))
        if keys:
            findings.append((filename, node.lineno, sorted(set(keys))))
    return findings


def _find_in_app() -> List[Tuple[str, int, List[str]]]:
    findings: List[Tuple[str, int, List[str]]] = []
    for path in sorted(_APP_DIR.rglob("*.py")):
        findings.extend(
            find_secret_logging_in_source(
                path.read_text(encoding="utf-8"), str(path))
        )
    return findings


def test_no_secret_values_are_logged_in_app():
    """None of the app's print/log calls may emit a secret-named env value."""
    findings = _find_in_app()
    assert not findings, "\n".join(
        f"{path}:{line} emits secret-named env var(s): {', '.join(keys)}"
        for path, line, keys in findings
    )


def test_guard_detects_the_original_incident_pattern():
    """The guard must actually catch the f-string secret leak it exists for."""
    bad = (
        "print(f\"Using Alpaca secret: "
        "{os.getenv('ALPACA_DEV_PAPER_SECRET', 'Not Set')}\")\n"
    )
    findings = find_secret_logging_in_source(bad, "bad.py")
    assert findings, "guard failed to flag an f-string secret leak"
    assert findings[0][2] == ["ALPACA_DEV_PAPER_SECRET"]


def test_guard_allows_presence_check():
    """Checking *whether* a secret is set (value not emitted) is allowed."""
    ok = (
        "secret_env = 'ALPACA_LIVE_SECRET'\n"
        "print(f\"secret ({secret_env}): "
        "{'loaded' if os.getenv(secret_env) else 'Not Set'}\")\n"
    )
    assert find_secret_logging_in_source(ok, "ok.py") == []

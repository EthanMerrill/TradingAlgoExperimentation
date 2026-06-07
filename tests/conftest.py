"""pytest configuration: add the app directory to the Python path."""
import os
import sys

# Ensure the app directory is on the path so test files can import app modules.
_app_dir = os.path.join(os.path.dirname(__file__), '..', 'app')
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

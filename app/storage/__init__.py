"""
Shared storage singleton.
Import this module to get the configured storage backend instance.

Usage:
    from storage import storage
    storage.save_positions(...)
"""
from storage.backend import StorageBackend
from config import globalConfig  # type: ignore

storage: StorageBackend = StorageBackend.create(globalConfig)

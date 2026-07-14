"""CLI progress indicator for long-running operations."""
import sys
import time
from typing import Optional


class ProgressIndicator:
    """Simple CLI progress indicator for long-running operations."""

    def __init__(self, total: int, description: str = "Processing", width: int = 50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        self.last_update = 0

    def update(self, count: int = 1, item_name: Optional[str] = None) -> None:
        """Update progress counter and display."""
        self.current += count
        current_time = time.time()

        # Only update display every 0.1 seconds to avoid spam
        if current_time - self.last_update >= 0.1:
            self._display_progress(item_name)
            self.last_update = current_time

    def _display_progress(self, item_name: Optional[str] = None) -> None:
        """Display progress bar."""
        if self.total == 0:
            percent = 100
        else:
            percent = min(100, (self.current / self.total) * 100)

        # Calculate filled and empty portions
        filled = int(self.width * percent / 100)
        empty = self.width - filled

        # Create progress bar
        bar = '█' * filled + '░' * empty

        # Calculate time estimates
        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed_time / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta/60:.1f}m" if eta > 60 else f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"

        # Build progress line
        progress_line = f"\r{self.description}: [{bar}] {percent:.1f}% ({self.current}/{self.total}) {eta_str}"

        if item_name:
            progress_line += f" | {item_name}"

        # Print without newline and flush
        sys.stdout.write(progress_line)
        sys.stdout.flush()

    def finish(self, message: str = "Complete!") -> None:
        """Finish progress indicator with final message."""
        elapsed_time = time.time() - self.start_time
        final_msg = f"\r{self.description}: [{'█' * self.width}] 100.0% ({self.total}/{self.total}) "
        final_msg += f"Completed in {elapsed_time/60:.1f}m - {message}\n"
        sys.stdout.write(final_msg)
        sys.stdout.flush()

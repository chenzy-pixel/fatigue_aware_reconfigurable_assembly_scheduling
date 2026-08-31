from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class _TeeTextStream:
    """Mirror one text stream to its original destination and a shared log."""

    def __init__(self, primary: TextIO, log: TextIO, lock: threading.RLock):
        self._primary = primary
        self._log = log
        self._lock = lock

    @property
    def encoding(self) -> str | None:
        return self._primary.encoding

    @property
    def errors(self) -> str | None:
        return self._primary.errors

    def write(self, value: str) -> int:
        with self._lock:
            written = self._primary.write(value)
            self._log.write(value)
            self._log.flush()
        return written

    def flush(self) -> None:
        with self._lock:
            self._primary.flush()
            self._log.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def fileno(self) -> int:
        return self._primary.fileno()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


@contextmanager
def capture_terminal_output(path: str | Path) -> Iterator[Path]:
    """Tee Python stdout/stderr to a new UTF-8, line-buffered log file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    lock = threading.RLock()
    with output.open("x", encoding="utf-8", buffering=1) as log:
        sys.stdout = _TeeTextStream(original_stdout, log, lock)
        sys.stderr = _TeeTextStream(original_stderr, log, lock)
        try:
            yield output
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

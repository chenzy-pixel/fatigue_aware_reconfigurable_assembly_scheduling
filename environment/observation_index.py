"""Static operation-index reuse confined to one observation query."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.models import AssemblyInstance


@dataclass
class _IndexCache:
    entry: tuple[AssemblyInstance, dict[str, int]] | None


_operation_index_cache: ContextVar[_IndexCache | None] = ContextVar(
    "observation_operation_index", default=None
)


@contextmanager
def observation_index_scope(instance: AssemblyInstance):
    """Reuse one immutable instance's index, including its hypothetical WAIT state."""
    current = _operation_index_cache.get()
    entry = None if current is None else current.entry
    if entry is not None and entry[0] is instance:
        yield
        return
    cache = _IndexCache((instance, instance.operation_index))
    token = _operation_index_cache.set(cache)
    try:
        yield
    finally:
        # Copied/inherited contexts must not extend this query's cache lifetime.
        cache.entry = None
        _operation_index_cache.reset(token)


def observation_operation_index(instance: AssemblyInstance) -> dict[str, int]:
    """Return a caller-owned dict; queries outside observe use the original getter."""
    current = _operation_index_cache.get()
    entry = None if current is None else current.entry
    if entry is not None and entry[0] is instance:
        return entry[1].copy()
    return instance.operation_index

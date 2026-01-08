"""
Generic registry pattern for managing named components (algorithms, operators, etc.).
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """
    A simple thread-safe registry for managing named items.

    Allows registering items by name and retrieving them.
    Supports usage as a decorator.
    """

    def __init__(self, name: str = "Registry") -> None:
        self._name = name
        self._items: dict[str, T] = {}

    def register(self, key: str, item: T | None = None, *, override: bool = False) -> Callable[[T], T] | T:
        """
        Register an item with the given key.

        Can be used as a function call or a decorator.

        Args:
            key: The unique name for the item.
            item: The item to register. If None, returns a decorator.
            override: If True, overwrite existing key. If False, raising ValueError on duplicate.

        Returns:
            The registered item (if passed) or a decorator (if item is None).
        """

        def _do_register(obj: T) -> T:
            if key in self._items and not override:
                raise ValueError(f"Key '{key}' already exists in registry '{self._name}'")
            self._items[key] = obj
            return obj

        if item is None:
            return _do_register
        return _do_register(item)

    def get(self, key: str, default: Any = ...) -> T:
        """
        Retrieve an item by key.

        Args:
            key: Identify of item to retrieve.
            default: Value to return if key missing. If not provided, raises KeyError.

        Returns:
            The item.
        """
        if key not in self._items:
            if default is not ...:
                return default
            raise KeyError(f"Key '{key}' not found in registry '{self._name}'")
        return self._items[key]

    def list(self) -> list[str]:
        """Return a sorted list of registered keys."""
        return sorted(self._items.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __getitem__(self, key: str) -> T:
        return self.get(key)

    def __iter__(self) -> Iterable[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def values(self) -> Iterable[T]:
        return self._items.values()

    def items(self) -> Iterable[tuple[str, T]]:
        return self._items.items()

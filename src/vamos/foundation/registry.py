"""
Generic registry pattern for managing named components (algorithms, operators, etc.).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Generic, TypeVar, overload

T = TypeVar("T")
U = TypeVar("U")


class Registry(Generic[T]):
    """
    A simple registry for managing named items.

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

        Parameters
        ----------
        key : str
            Unique name for the item.
        item : T | None, optional
            Item to register. When omitted, returns a decorator.
        override : bool, default ``False``
            Whether to overwrite an existing key.

        Returns
        -------
        Callable[[T], T] | T
            The registered item, or a decorator when ``item`` is omitted.
        """

        def _do_register(obj: T) -> T:
            if key in self._items and not override:
                raise ValueError(f"Key '{key}' already exists in registry '{self._name}'")
            self._items[key] = obj
            return obj

        if item is None:
            return _do_register
        return _do_register(item)

    @overload
    def get(self, key: str) -> T: ...

    @overload
    def get(self, key: str, default: U) -> T | U: ...

    def get(self, key: str, default: object = ...) -> object:
        """
        Retrieve an item by key.

        Parameters
        ----------
        key : str
            Identifier of the item to retrieve.
        default : object, optional
            Value returned when the key is missing. If omitted, a ``KeyError``
            is raised instead.

        Returns
        -------
        object
            The registered item or ``default``.
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

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def values(self) -> Iterable[T]:
        return self._items.values()

    def items(self) -> Iterable[tuple[str, T]]:
        return self._items.items()

"""A tiny typed in-memory database for τ³-Bench domains.

τ³-Bench verifies agent behavior by inspecting the final state of a
domain-specific database. This module provides a minimal relational store with
column typing and a few mutation helpers sufficient for the Retail / Airline /
Telecom domains.
"""

from __future__ import annotations

from typing import Any


class DBError(Exception):
    pass


class Database:
    def __init__(self, tables: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = tables or {}

    def rows(self, table: str) -> list[dict[str, Any]]:
        if table not in self.tables:
            raise DBError(f"unknown table {table!r}")
        return self.tables[table]

    def get(self, table: str, key: str, value: Any) -> dict[str, Any] | None:
        for row in self.rows(table):
            if row.get(key) == value:
                return row
        return None

    def insert(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        self.rows(table).append(row)
        return row

    def update(self, table: str, key: str, value: Any, updates: dict[str, Any]) -> dict[str, Any] | None:
        row = self.get(table, key, value)
        if row is None:
            raise DBError(f"no {table} row with {key}={value!r}")
        row.update(updates)
        return row

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        import copy

        return copy.deepcopy(self.tables)

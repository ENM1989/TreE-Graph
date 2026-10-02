"""Pytest configuration and fixtures."""

from __future__ import annotations

from pytest import Config

from tests.constants import PytestConstants


def pytest_configure(config: Config) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line(
        PytestConstants.MARKER_NAME,
        PytestConstants.SLOW_MARKER_DESCRIPTION,
    )

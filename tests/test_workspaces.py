"""Tests de la validation des noms de workspace (dependances legeres)."""

import pytest

from app.workspaces import normalize_workspace


def test_empty_or_blank_falls_back_to_default():
    assert normalize_workspace(None) == "default"
    assert normalize_workspace("") == "default"
    assert normalize_workspace("   ") == "default"


def test_valid_names_are_trimmed_and_kept():
    assert normalize_workspace(" projet-alpha ") == "projet-alpha"
    assert normalize_workspace("a.b_c-1") == "a.b_c-1"


@pytest.mark.parametrize(
    "bad",
    ["a/b", "espace interdit", "x" * 65, "workspace!", "../etc", "a\tb"],
)
def test_invalid_names_are_rejected(bad):
    with pytest.raises(ValueError, match="workspace invalide"):
        normalize_workspace(bad)

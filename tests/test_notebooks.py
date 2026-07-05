"""Tests du slugify des titres de notebook (fonction pure, sans dependance)."""

from app.notebooks import slugify


def test_slugify_spaces_and_case():
    assert slugify("Warhammer paint") == "warhammer-paint"


def test_slugify_strips_accents():
    assert slugify("Café Crème") == "cafe-creme"


def test_slugify_collapses_and_trims():
    assert slugify("  --Hello!!!  World--  ") == "hello-world"


def test_slugify_keeps_safe_characters():
    assert slugify("projet_alpha-1.0") == "projet_alpha-1.0"


def test_slugify_empty_falls_back():
    assert slugify("   ") == "notebook"
    assert slugify("!!!") == "notebook"


def test_slugify_max_length():
    assert len(slugify("a" * 200)) <= 64

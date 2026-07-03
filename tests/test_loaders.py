"""Tests de l'extraction de texte (txt/md + PDF non scanne)."""

from pathlib import Path

import pytest

from app.rag.loaders import extract_text, is_supported

FIXTURES = Path(__file__).parent / "fixtures"


def test_text_file_is_decoded():
    content = "Bonjour le monde.\nDeuxieme ligne.".encode("utf-8")
    assert extract_text("note.txt", content) == "Bonjour le monde.\nDeuxieme ligne."


def test_markdown_is_supported():
    assert is_supported("README.md")
    assert extract_text("README.md", b"# Titre") == "# Titre"


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Format non pris en charge"):
        extract_text("image.png", b"\x89PNG")


def test_non_utf8_text_raises():
    with pytest.raises(ValueError, match="UTF-8"):
        extract_text("note.txt", b"\xff\xfe invalide")


def test_pdf_text_layer_is_extracted():
    pytest.importorskip("pypdf")
    content = (FIXTURES / "sample_text.pdf").read_bytes()
    text = extract_text("rapport.pdf", content)
    assert "canape" in text
    # Le PDF fixture comporte deux pages : les deux doivent etre extraites.
    assert "jardin" in text


def test_scanned_pdf_raises_explicit_error():
    pytest.importorskip("pypdf")
    # PDF minimal valide, une page vide (aucune couche texte) -> assimile a un scan.
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    with pytest.raises(ValueError, match="scanne"):
        extract_text("scan.pdf", buf.getvalue())

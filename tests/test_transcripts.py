"""Tests du traitement des transcripts (parsing, nettoyage, chunking, video_id).

Pur Python, sans dependance lourde : execute en CI legere comme en local.
"""

import pytest

from app.rag.transcripts import (
    Cue,
    chunk_cues,
    clean_text,
    cues_to_bracketed_text,
    extract_video_id,
    format_timestamp,
    is_transcript,
    parse_bracketed,
    parse_cue_format,
    parse_transcript,
)

# --- Nettoyage ---------------------------------------------------------------


def test_clean_text_removes_annotations_and_tags():
    assert clean_text("Bonjour [rires] a  tous") == "Bonjour a tous"
    assert clean_text("Second <c>line</c> ici") == "Second line ici"
    assert clean_text("  espaces\n\tmultiples ") == "espaces multiples"


# --- Format horodate [HH:MM:SS] ---------------------------------------------


def test_parse_bracketed_hms_and_boundaries():
    text = "[00:00:00] Bonjour [rires] a tous.\n[00:00:05] Deuxieme.\n[00:01:00] Fin."
    cues = parse_bracketed(text)
    assert len(cues) == 3
    assert cues[0].start == 0 and cues[0].end == 5
    assert cues[0].text == "Bonjour a tous."  # annotation retiree
    assert cues[1].start == 5 and cues[1].end == 60
    assert cues[2].start == 60 and cues[2].end == 60  # derniere cue : fin ouverte


def test_parse_bracketed_mm_ss():
    cues = parse_bracketed("[00:30] a\n[01:00] b")
    assert cues[0].start == 30  # MM:SS
    assert cues[1].start == 60


# --- Formats SRT / VTT -------------------------------------------------------


def test_parse_srt():
    srt = (
        "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n"
        "2\n00:00:04,000 --> 00:00:06,500\nSecond <c>line</c>\n"
    )
    cues = parse_cue_format(srt)
    assert cues[0].start == 1.0 and cues[0].end == 4.0
    assert cues[0].text == "Hello world"
    assert cues[1].end == 6.5 and cues[1].text == "Second line"


def test_parse_vtt_skips_nonverbal_and_header():
    vtt = "WEBVTT\n\n00:00.000 --> 00:02.000\n[Music]\n\n00:02.000 --> 00:05.000\nBonjour\n"
    cues = parse_cue_format(vtt)
    assert len(cues) == 1  # le bloc [Music] devient vide -> ignore
    assert cues[0].start == 2.0 and cues[0].end == 5.0
    assert cues[0].text == "Bonjour"


def test_parse_transcript_dispatch_and_empty():
    assert parse_transcript("a.srt", b"1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    with pytest.raises(ValueError, match="horodate"):
        parse_transcript("a.txt", b"aucun horodatage ici")


# --- Detection ---------------------------------------------------------------


def test_is_transcript_detection():
    assert is_transcript("a.srt", None) is True
    assert is_transcript("a.vtt", None) is True
    assert is_transcript("a.txt", "[00:00:00] x\n[00:00:02] y\n[00:00:04] z") is True
    assert is_transcript("a.txt", "Un paragraphe normal sans horodatage.") is False
    assert is_transcript("a.txt", None) is False
    assert is_transcript("a.pdf", None) is False


# --- Chunking temporel -------------------------------------------------------


def _sample_cues(n=10):
    return [Cue(start=i * 2.0, end=i * 2.0 + 2, text=f"phrase numero {i}") for i in range(n)]


def test_chunk_cues_respects_size_and_keeps_times():
    cues = _sample_cues()
    chunks = chunk_cues(cues, chunk_size=40, chunk_overlap=0)

    assert [c.passage_id for c in chunks] == list(range(len(chunks)))
    assert chunks[0].start == 0.0
    assert chunks[-1].end == cues[-1].end
    for chunk in chunks:
        assert len(chunk.text) <= 40  # jamais au-dela de la taille cible

    joined = " ".join(c.text for c in chunks)
    for i in range(10):
        assert f"phrase numero {i}" in joined  # aucun segment perdu


def test_chunk_cues_overlap_reintroduces_text():
    cues = _sample_cues()
    without = chunk_cues(cues, chunk_size=40, chunk_overlap=0)
    with_ov = chunk_cues(cues, chunk_size=40, chunk_overlap=15)
    total_without = sum(len(c.text) for c in without)
    total_with = sum(len(c.text) for c in with_ov)
    assert total_with > total_without  # le recouvrement duplique du contexte


def test_chunk_cues_empty():
    assert chunk_cues([], chunk_size=40, chunk_overlap=0) == []


def test_chunk_cues_breaks_on_speaker_change():
    cues = [
        Cue(0, 2, "bonjour tout le monde", speaker="SPEAKER_00"),
        Cue(2, 4, "salut a tous", speaker="SPEAKER_00"),
        Cue(4, 6, "merci de m'accueillir", speaker="SPEAKER_01"),
    ]
    # Meme si tout tiendrait dans un seul passage, le changement de locuteur coupe.
    chunks = chunk_cues(cues, chunk_size=1000, chunk_overlap=0)
    assert len(chunks) == 2
    assert chunks[0].speaker == "SPEAKER_00"
    assert chunks[1].speaker == "SPEAKER_01"
    assert "merci" in chunks[1].text and "merci" not in chunks[0].text


# --- Utilitaires -------------------------------------------------------------


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(3661) == "01:01:01"


def test_bracketed_roundtrip():
    cues = [Cue(0, 2, "bonjour"), Cue(2, 4, "monde")]
    reparsed = parse_bracketed(cues_to_bracketed_text(cues))
    assert [c.text for c in reparsed] == ["bonjour", "monde"]
    assert reparsed[0].start == 0


# --- Identifiant video -------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=30",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ],
)
def test_extract_video_id_variants(url):
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_rejects_unknown():
    with pytest.raises(ValueError, match="non reconnue"):
        extract_video_id("https://example.com/watch?x=1")

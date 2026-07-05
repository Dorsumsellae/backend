"""Tests de la construction du prompt."""

from app.rag.prompt import build_chat_prompt, build_prompt, cited_indices


def test_prompt_contains_question_and_context():
    prompt = build_prompt("Que dit le texte ?", ["Passage A", "Passage B"])
    assert "Que dit le texte ?" in prompt
    assert "Passage A" in prompt
    assert "Passage B" in prompt


def test_prompt_contains_fallback_instruction():
    prompt = build_prompt("Question", ["contexte"])
    assert "Je ne trouve pas cette information dans le document fourni." in prompt


def test_prompt_numbers_context_and_requests_citation():
    prompt = build_prompt("Question", ["premier passage", "second passage"])
    assert "[1]" in prompt and "[2]" in prompt  # contexte numerote = ancres [n]
    assert "cite" in prompt.lower() and "crochets" in prompt.lower()  # consigne


def test_build_chat_prompt_includes_recent_history():
    messages = [
        {"role": "user", "content": "Quelle est la capitale ?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "Et sa population ?"},
    ]
    prompt = build_chat_prompt(messages, ["contexte demographique"])
    assert "Quelle est la capitale ?" in prompt  # tour precedent conserve
    assert "Et sa population ?" in prompt  # question courante
    assert "Utilisateur:" in prompt and "Assistant:" in prompt


def test_cited_indices_filters_out_of_range_and_dedupes():
    assert cited_indices("Reponse [1], encore [1], puis [3].", n_sources=2) == [1]
    assert cited_indices("Aucune citation ici.", n_sources=3) == []
    assert cited_indices("[2][1][2]", n_sources=2) == [2, 1]  # ordre d'apparition

"""Tests de la construction du prompt."""

from app.rag.prompt import (
    build_chat_prompt,
    build_prompt,
    build_router_prompt,
    build_summary_prompt,
    cited_indices,
)


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


def test_prompt_forbids_citation_only_answers():
    # Garde-fou : le prompt doit interdire de repondre uniquement par des numeros de
    # citation (bug observe : le petit modele cite [1][2] sans rediger de reponse).
    prompt = build_prompt("Question", ["contexte"]).lower()
    assert "jamais uniquement" in prompt
    assert "redigee" in prompt  # exige une reponse formulee en toutes lettres


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


def test_build_summary_prompt_includes_passages_and_question():
    prompt = build_summary_prompt(["extrait un", "extrait deux"], "de quoi parle ce doc ?")
    assert "extrait un" in prompt and "extrait deux" in prompt
    assert "de quoi parle ce doc ?" in prompt  # question rappelee pour orienter
    assert "[1]" in prompt and "[2]" in prompt  # extraits numerotes


def test_build_summary_prompt_without_question_asks_general_summary():
    prompt = build_summary_prompt(["seul extrait"])
    assert "seul extrait" in prompt
    # Sans question : consigne de synthese generale (pas de phrase de refus imposee).
    assert "synthese generale" in prompt.lower()
    assert "Je ne trouve pas cette information" not in prompt


def test_build_router_prompt_asks_for_single_label():
    prompt = build_router_prompt("resume-moi la video")
    assert "resume-moi la video" in prompt
    assert "RESUME" in prompt and "FACTUEL" in prompt  # les deux etiquettes proposees


def test_cited_indices_filters_out_of_range_and_dedupes():
    assert cited_indices("Reponse [1], encore [1], puis [3].", n_sources=2) == [1]
    assert cited_indices("Aucune citation ici.", n_sources=3) == []
    assert cited_indices("[2][1][2]", n_sources=2) == [2, 1]  # ordre d'apparition

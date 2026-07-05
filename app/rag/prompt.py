"""Construction des prompts enrichis envoyes au LLM (RAG mono-tour et chat)."""

import re

# Consigne commune : reponse ancree dans le contexte + citations numerotees.
# Le numero cite `[n]` correspond exactement au numero du passage dans le contexte
# (voir `_format_context`), ce qui maximise les chances qu'un petit modele cite bien.
# La clause de refus est volontairement conditionnelle (« si, et seulement si... ») :
# le modele est autorise a SYNTHETISER et relier les passages, et ne doit sortir la
# phrase de repli que si aucun passage ne permet reellement de repondre.
_CITATION_RULES = """Tu reponds en francais, de facon claire et structuree.
Formule d'abord une reponse REDIGEE en toutes lettres, en t'appuyant uniquement sur
le contexte numerote ci-dessous : tu peux synthetiser, reformuler et relier les
passages, mais n'ajoute aucun fait absent du contexte et n'invente aucune source.
Les marqueurs entre crochets [1] ou [2][3] ne font que COMPLETER tes phrases pour
signaler les passages utilises : ne reponds JAMAIS uniquement par des numeros de
citation. Apres chaque affirmation, cite le ou les passages correspondants.
Si, et seulement si, aucun passage ne permet de repondre, reponds exactement :
"Je ne trouve pas cette information dans le document fourni." """

# Regles specifiques au chat : le modele doit s'appuyer sur la conversation pour
# lever les references (« il », « ce projet », « et son budget ? ») avant de repondre
# a partir du contexte documentaire. Sans cela, un petit modele ignore l'historique.
_CHAT_RULES = """Tu reponds en francais a la DERNIERE question de l'utilisateur.
Appuie-toi sur la conversation pour comprendre les references implicites (« il »,
« ce projet », etc.). Formule d'abord une reponse REDIGEE en toutes lettres a partir
du contexte numerote ci-dessous : tu peux synthetiser et reformuler, mais n'ajoute
aucun fait absent du contexte et n'invente aucune source. Les marqueurs entre
crochets [1] ou [2][3] ne font que COMPLETER tes phrases : ne reponds JAMAIS
uniquement par des numeros de citation. Apres chaque affirmation, cite le ou les
passages correspondants.
Si, et seulement si, aucun passage ne permet de repondre, reponds exactement :
"Je ne trouve pas cette information dans le document fourni." """

PROMPT_TEMPLATE = """Tu es un assistant documentaire.
{rules}

Contexte :
{context}

Question :
{question}

Reponse :"""

CHAT_PROMPT_TEMPLATE = """Tu es un assistant documentaire conversationnel.
{rules}

Contexte :
{context}

Conversation :
{history}"""

# Nombre de tours de conversation conserves dans le prompt de chat (fenetre glissante).
_HISTORY_TURNS = 6


def _format_context(passages: list[str]) -> str:
    """Numerote les passages `[1] ...` (le numero sert d'ancre de citation)."""
    return "\n\n".join(f"[{i + 1}] {p.strip()}" for i, p in enumerate(passages))


def build_prompt(question: str, passages: list[str]) -> str:
    """Assemble le prompt RAG mono-tour (consigne + contexte numerote + question)."""
    return PROMPT_TEMPLATE.format(
        rules=_CITATION_RULES,
        context=_format_context(passages),
        question=question.strip(),
    )


def build_chat_prompt(messages: list[dict], passages: list[str]) -> str:
    """Assemble le prompt de chat multi-tours.

    `messages` est l'historique complet `[{role, content}]` (le dernier etant la
    question courante de l'utilisateur) ; seuls les derniers tours sont conserves
    pour rester dans une fenetre de contexte raisonnable. On termine par
    `Assistant:` pour amorcer la generation de la reponse.
    """
    recent = messages[-_HISTORY_TURNS:]
    lines = []
    for message in recent:
        role = "Utilisateur" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {(message.get('content') or '').strip()}")
    lines.append("Assistant:")
    return CHAT_PROMPT_TEMPLATE.format(
        rules=_CHAT_RULES,
        context=_format_context(passages),
        history="\n".join(lines),
    )


# --- Mode resume (synthese globale) et routage d'intention -------------------

# Prompt de synthese : utilise pour les questions globales (« de quoi parle... »,
# « resume »). Contrairement au prompt Q/R, il n'impose ni citations `[n]` ni phrase
# de refus : il s'agit de degager les themes d'ensemble a partir d'un echantillon de
# passages couvrant tout le document (cf. pipeline.summarize).
SUMMARY_PROMPT_TEMPLATE = """Tu es un assistant documentaire.
Voici des extraits representatifs d'un meme document, donnes dans l'ordre, du debut
a la fin. Redige en francais une synthese claire et structuree de ce dont parle le
document : les themes principaux, les points saillants et, si pertinent, leur
enchainement. Reste fidele aux extraits, sans inventer d'informations absentes.

Extraits :
{context}

{task}

Synthese :"""

# Prompt de routage : quand l'heuristique par mots-cles ne tranche pas, le LLM classe
# lui-meme l'intention (synthese globale vs reponse factuelle). Reponse en un seul mot
# pour rester fiable et rapide, meme avec un petit modele.
ROUTER_PROMPT_TEMPLATE = """Classe l'intention de la question posee sur un document.
Reponds par UN SEUL mot, sans ponctuation ni explication :
- RESUME : la question demande une vue d'ensemble, un resume, le sujet global ou les
  themes du document (ex. « de quoi parle ce document », « resume-moi la video »).
- FACTUEL : la question porte sur une information precise contenue dans le document.

Question : {question}
Reponse :"""


def build_summary_prompt(passages: list[str], question: str | None = None) -> str:
    """Assemble le prompt de synthese globale (echantillon large -> resume).

    Si `question` est fournie, elle est rappelee au modele pour orienter la synthese ;
    sinon on demande une synthese generale du document.
    """
    task = (
        f"Question de l'utilisateur : {question.strip()}"
        if question and question.strip()
        else "Redige la synthese generale du document."
    )
    return SUMMARY_PROMPT_TEMPLATE.format(
        context=_format_context(passages),
        task=task,
    )


def build_router_prompt(question: str) -> str:
    """Assemble le prompt de classification d'intention (resume vs factuel)."""
    return ROUTER_PROMPT_TEMPLATE.format(question=question.strip())


_CITATION_RE = re.compile(r"\[(\d+)\]")


def cited_indices(answer: str, n_sources: int) -> list[int]:
    """Retourne les numeros de passages `[n]` reellement cites dans la reponse.

    Filtre sur `1..n_sources` (ignore les marqueurs hors borne, frequents avec un
    petit modele) et dedoublonne en conservant l'ordre d'apparition.
    """
    seen: list[int] = []
    for match in _CITATION_RE.findall(answer or ""):
        idx = int(match)
        if 1 <= idx <= n_sources and idx not in seen:
            seen.append(idx)
    return seen

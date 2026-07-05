"""Configuration centralisee, lue depuis les variables d'environnement.

Les valeurs par defaut correspondent au parametrage conseille dans le sujet :
    chunk_size    : 700 a 900 caracteres
    chunk_overlap : 100 a 150 caracteres
    top_k         : 3 a 4 passages
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- MinIO ---
    minio_endpoint: str = "minio:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # --- ChromaDB ---
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "rag_documents"

    # --- Ollama ---
    ollama_base_url: str = "http://ollama:11434"
    # Modele par defaut : Gemma 3 (~4B). Plus capable que qwen2.5:0.5b (meilleures
    # citations, coreference, grande fenetre de contexte).
    ollama_model: str = "gemma3:latest"
    # Fenetre de contexte cote Ollama (num_ctx). Ollama plafonne par defaut a ~4096
    # tokens QUELLE QUE SOIT la capacite du modele : sans ce reglage, un prompt a fort
    # top_k ou une synthese globale serait tronque en silence. Gemma 3 accepte jusqu'a
    # 131072 tokens ; 8192 est un compromis qualite/vitesse/RAM raisonnable.
    ollama_num_ctx: int = 8192
    # Temperature de generation. Le Modelfile de Gemma 3 fixe 1.0 par defaut, ce qui
    # est trop eleve pour du RAG : reponses instables, parfois degenerees (le modele
    # « des fois » ne cite que les passages [1][2] sans rediger de reponse). Une valeur
    # basse ancre la reponse au contexte et fiabilise la sortie.
    ollama_temperature: float = 0.2

    # --- Embeddings ---
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # --- Parametres RAG ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    # Nombre de passages transmis au LLM (releve : gemma3 gere une grande fenetre,
    # cf. ollama_num_ctx). Pilotable par l'env (TOP_K).
    top_k: int = 10

    # --- Mode resume (questions globales : « de quoi parle... », « resume ») -------
    # Le retrieval top-k (pertinence locale) est inadapte aux questions de synthese :
    # on echantillonne plutot des passages repartis sur TOUT le document. Ce reglage
    # fixe le nombre de passages echantillonnes (couverture debut -> fin).
    summary_sample_size: int = 24

    # --- Retrieval avance (voir app/rag/retrieval.py) ------------------------
    # Nombre de candidats recuperes AVANT reranking (« retrieve large, rerank »).
    retrieval_top_n: int = 20
    # Reranking par cross-encoder : reordonne finement (question, passage), garde top_k.
    # Modele multilingue leger (~120 Mo, telecharge une fois puis mis en cache).
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Recherche hybride : fusion dense (embeddings) + BM25 (mots-cles) par RRF.
    use_hybrid: bool = True
    rrf_k: int = 60
    # MMR : diversifie les candidats denses (penalise les passages redondants).
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    # Reorder anti « lost-in-the-middle » : meilleurs passages en debut ET fin du prompt.
    reorder_lost_in_middle: bool = True

    # Strategie de decoupage par defaut (surchargeable par requete) :
    #   "fixed"     -> taille fixe avec recouvrement (autonome, sans LangChain).
    #   "recursive" -> RecursiveCharacterTextSplitter (paragraphes -> phrases -> mots).
    chunk_strategy: str = "fixed"

    # Workspace utilise quand aucun n'est precise (cloisonnement logique des documents).
    default_workspace: str = "default"

    # --- Base de donnees (metadonnees applicatives : notebooks, chat, notes) ---
    # Postgres par defaut (service `postgres` du docker-compose). La persistance est
    # best-effort : si la base est injoignable, le RAG (upload/index/chat) fonctionne
    # quand meme, seule la sauvegarde des notebooks/conversations/notes est perdue.
    database_url: str = "postgresql+psycopg2://rag:rag@postgres:5432/rag"

    # Service ASR (transcription audio) appele en fallback quand une video YouTube
    # n'a pas de sous-titres. Vide => fallback desactive (on renvoie une erreur claire).
    asr_service_url: str = ""
    # Duree max d'attente de la transcription ASR (secondes). L'ASR est lent (CPU).
    asr_timeout: int = 900


settings = Settings()

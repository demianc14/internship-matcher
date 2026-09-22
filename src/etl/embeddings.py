"""Adaptador opcional a sentence-transformers (pip install -e ".[semantic]").

Import perezoso: el resto del pipeline funciona sin torch instalado; en ese caso
el match corre solo con el componente de skills y lo dice en la explicación.
"""

from collections.abc import Sequence

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # local, es/en/de/fr


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError(
                'sentence-transformers no está instalado: pip install -e ".[semantic]"'
            ) from exc
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]

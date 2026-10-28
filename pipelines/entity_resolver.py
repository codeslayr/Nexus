"""
Entity Resolution & Canonicalization — Section 2, Step 4.

Prevents duplicate / near-identical Neo4j nodes by:
1. Exact normalisation (lowercase, strip whitespace).
2. Alias lookup (known aliases collapsed to canonical name).
3. Cosine embedding similarity for fuzzy deduplication above a threshold.
4. In-memory canonical registry with thread-safe upsert.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import get_settings
from src.models.schemas import Entity
from src.utils.llm_factory import get_embeddings
from src.utils.logging import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Canonical entity record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CanonicalEntity:
    name: str
    type: str
    description: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    def add_alias(self, alias: str) -> None:
        normalised = _normalise(alias)
        if normalised not in [_normalise(a) for a in self.aliases]:
            self.aliases.append(alias)


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc\.?|corp\.?|ltd\.?|llc\.?|llp\.?|plc\.?|co\.?|group|holding|holdings)\b",
    re.IGNORECASE,
)


def _normalise(name: str) -> str:
    """Strip punctuation, legal suffixes, and lowercase for comparison."""
    name = name.strip().lower()
    name = _LEGAL_SUFFIXES.sub("", name)
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ──────────────────────────────────────────────────────────────────────────────
# Resolver
# ──────────────────────────────────────────────────────────────────────────────

class EntityResolver:
    """
    Maintains a session-scoped canonical entity registry.

    Resolution flow:
      1. Exact normalised name match → return existing canonical.
      2. Alias match → return the canonical that owns the alias.
      3. Embedding cosine similarity ≥ threshold → merge into best match.
      4. No match → register as new canonical entity.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._threshold = settings.entity_similarity_threshold
        self._registry: dict[str, CanonicalEntity] = {}  # normalised_name → canonical
        self._alias_map: dict[str, str] = {}             # normalised_alias → canonical_name
        self._embeddings = get_embeddings()

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(self, entity: Entity) -> str:
        """
        Resolve an extracted Entity to a canonical name.
        Synchronous path — uses cached embeddings only.
        Returns the canonical name.
        """
        key = _normalise(entity.name)

        # 1. Exact match
        if key in self._registry:
            self._registry[key].add_alias(entity.name)
            return self._registry[key].name

        # 2. Alias match
        if key in self._alias_map:
            canonical_key = _normalise(self._alias_map[key])
            canonical = self._registry[canonical_key]
            canonical.add_alias(entity.name)
            return canonical.name

        # Check all aliases from the new entity
        for alias in entity.aliases:
            alias_key = _normalise(alias)
            if alias_key in self._alias_map:
                canonical_key = _normalise(self._alias_map[alias_key])
                canonical = self._registry[canonical_key]
                canonical.add_alias(entity.name)
                canonical.add_alias(alias)
                return canonical.name

        # 3. Embedding similarity (if embeddings are available)
        if entity.embedding is not None:
            best_name, best_score = self._find_best_embedding_match(
                np.array(entity.embedding)
            )
            if best_score >= self._threshold:
                canonical = self._registry[_normalise(best_name)]
                canonical.add_alias(entity.name)
                self._alias_map[key] = canonical.name
                log.info(
                    "entity_merged",
                    new=entity.name,
                    canonical=canonical.name,
                    score=round(best_score, 4),
                )
                return canonical.name

        # 4. Register as new canonical entity
        canonical = CanonicalEntity(
            name=entity.name,
            type=entity.type,
            description=entity.description,
            aliases=list(entity.aliases),
            embedding=np.array(entity.embedding) if entity.embedding else None,
        )
        self._registry[key] = canonical
        for alias in entity.aliases:
            self._alias_map[_normalise(alias)] = entity.name

        log.debug("entity_registered", name=entity.name, type=entity.type)
        return entity.name

    async def resolve_with_embedding(self, entity: Entity) -> str:
        """
        Full resolution path including on-demand embedding lookup for new entities.
        Use this during ingestion when the entity has no pre-computed embedding.
        """
        # Fetch embedding only if no existing similar canonical found by text
        key = _normalise(entity.name)
        if key not in self._registry and key not in self._alias_map:
            vec = await self._embeddings.aembed_query(entity.name)
            entity_with_emb = entity.model_copy()
            object.__setattr__(entity_with_emb, "embedding", vec)
            return self.resolve(entity_with_emb)
        return self.resolve(entity)

    def get_canonical_entities(self) -> list[CanonicalEntity]:
        return list(self._registry.values())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_best_embedding_match(
        self, query_vec: np.ndarray
    ) -> tuple[str, float]:
        best_name = ""
        best_score = -1.0
        for canonical in self._registry.values():
            if canonical.embedding is None:
                continue
            score = _cosine(query_vec, canonical.embedding)
            if score > best_score:
                best_score = score
                best_name = canonical.name
        return best_name, best_score

"""
KG Extraction Pipeline — Section 2, Steps 3–5.

Extracts (Entity, Relationship) triples from document chunks using a
structured LLM call with strict Pydantic output parsing.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import get_settings
from src.utils.llm_factory import get_chat_llm
from src.models.schemas import (
    Entity,
    KGExtractionResult,
    Relationship,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Intermediate schema for LLM structured output
# ──────────────────────────────────────────────────────────────────────────────

class _ExtractionPayload(BaseModel):
    """Raw LLM extraction output — validated before domain mapping."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a knowledge graph extraction specialist for an enterprise AI system.

Your task is to extract structured entities and relationships from a document chunk.

## Entity Types (use exactly these strings):
PERSON | ORG | PRODUCT | LOCATION | CONCEPT | EVENT | METRIC | DEPARTMENT | REGULATION | TECHNOLOGY | SECURITY

## Relationship Types (SCREAMING_SNAKE_CASE):
Examples: ACQUIRED, INVESTED_IN, PARTNERED_WITH, CEO_OF, LOCATED_IN,
REPORTED_METRIC, COMPETES_WITH, PART_OF, OWNS, FOUNDED_BY, WORKED_AT

## Rules:
1. Extract ONLY entities and relationships that are EXPLICITLY stated — no inferences.
2. Canonical names: use full formal names (e.g., "Apple Inc." not "Apple" or "AAPL").
3. List alternative names in the `aliases` field.
4. Every relationship MUST reference entities you extracted.
5. Set `context` to the verbatim sentence supporting the relationship.
6. If the chunk contains no entities or relationships, return empty lists.
7. Respond ONLY with valid JSON matching the schema — no prose.

{format_instructions}
"""

_HUMAN_PROMPT = """Document chunk (chunk_id={chunk_id}):

---
{chunk_text}
---

Extract all entities and relationships. Set context_chunk_id = "{chunk_id}" for every relationship.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Extractor class
# ──────────────────────────────────────────────────────────────────────────────

class KGExtractor:
    """
    Uses a structured LLM call to extract Knowledge Graph triples from text.

    Design decisions:
    - Uses `.with_structured_output()` when available, falls back to
      PydanticOutputParser for maximum compatibility.
    - Processes chunks in async batches to respect rate limits.
    - Returns KGExtractionResult with provenance chunk_id.
    """

    def __init__(self, concurrency: int = 5) -> None:
        settings = get_settings()
        self._llm = get_chat_llm(temperature=0)
        self._max_entities = settings.max_entities_per_chunk
        self._semaphore = asyncio.Semaphore(concurrency)
        self._parser = PydanticOutputParser(pydantic_object=_ExtractionPayload)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ]).partial(format_instructions=self._parser.get_format_instructions())

    async def extract_from_chunk(
        self,
        chunk_id: str,
        chunk_text: str,
    ) -> KGExtractionResult:
        """Extract triples from a single chunk with rate-limit concurrency guard."""
        async with self._semaphore:
            try:
                chain = self._prompt | self._llm | self._parser
                payload: _ExtractionPayload = await chain.ainvoke(
                    {"chunk_id": chunk_id, "chunk_text": chunk_text}
                )

                # Enforce entity cap
                entities = payload.entities[: self._max_entities]
                entity_names = {e.name for e in entities}

                # Filter relationships to only those referencing extracted entities
                relationships = [
                    r for r in payload.relationships
                    if r.source in entity_names and r.target in entity_names
                ]

                log.info(
                    "kg_extracted",
                    chunk_id=chunk_id,
                    entities=len(entities),
                    relationships=len(relationships),
                )
                return KGExtractionResult(
                    chunk_id=chunk_id,
                    entities=entities,
                    relationships=relationships,
                )

            except Exception as exc:
                log.error("kg_extraction_failed", chunk_id=chunk_id, error=str(exc))
                return KGExtractionResult(chunk_id=chunk_id)

    async def extract_batch(
        self,
        chunks: list[tuple[str, str]],  # [(chunk_id, chunk_text), ...]
    ) -> list[KGExtractionResult]:
        """
        Process many chunks concurrently, respecting the semaphore limit.
        """
        tasks = [
            self.extract_from_chunk(chunk_id, chunk_text)
            for chunk_id, chunk_text in chunks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

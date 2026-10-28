"""
Text-to-Cypher Engine — Section 3, Step 4.

Generates syntactically-valid Cypher from a natural-language query,
then validates it against a whitelist of safe operations before execution.
"""
from __future__ import annotations

import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import get_settings
from src.utils.llm_factory import get_chat_llm
from src.utils.logging import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Safety validation
# ──────────────────────────────────────────────────────────────────────────────

_DANGEROUS_KEYWORDS = re.compile(
    r"\b(DELETE|DETACH|CREATE|MERGE|SET|REMOVE|DROP|CALL\s+apoc\.schema|"
    r"CALL\s+apoc\.periodic\.commit|LOAD\s+CSV)\b",
    re.IGNORECASE,
)

_MAX_CYPHER_LENGTH = 2000


class CypherValidationError(ValueError):
    pass


def validate_cypher(cypher: str) -> str:
    """
    Validate generated Cypher for safety and length.
    Only MATCH/RETURN/WITH/WHERE/OPTIONAL MATCH/LIMIT/ORDER BY are permitted.
    """
    stripped = cypher.strip()

    if len(stripped) > _MAX_CYPHER_LENGTH:
        raise CypherValidationError(
            f"Generated Cypher exceeds max length ({_MAX_CYPHER_LENGTH})"
        )

    if _DANGEROUS_KEYWORDS.search(stripped):
        raise CypherValidationError(
            "Generated Cypher contains disallowed write/delete operations"
        )

    # Must begin with MATCH or WITH or OPTIONAL
    if not re.match(r"^\s*(MATCH|WITH|OPTIONAL\s+MATCH)\b", stripped, re.IGNORECASE):
        raise CypherValidationError(
            "Generated Cypher must start with MATCH, WITH, or OPTIONAL MATCH"
        )

    return stripped


# ──────────────────────────────────────────────────────────────────────────────
# Schema context (injected into prompt so LLM knows the graph shape)
# ──────────────────────────────────────────────────────────────────────────────

_GRAPH_SCHEMA = """
Node labels: Entity
Entity properties: name (string), type (string), description (string), aliases (list)
Entity type values: PERSON | ORG | PRODUCT | LOCATION | CONCEPT | EVENT | METRIC

Relationship type: RELATES (with property `type` containing the semantic relation)
Relationship properties: type (string), chunk_ids (list), context (string), confidence (float)

Example query pattern:
MATCH (a:Entity {{name: "Apple Inc."}})-[r:RELATES]->(b:Entity)
WHERE r.type = "ACQUIRED"
RETURN a.name, r.type, b.name, r.context
LIMIT 10
"""

_SYSTEM_PROMPT = f"""You are a Neo4j Cypher query generator for a Knowledge Graph.

## Graph Schema:
{_GRAPH_SCHEMA}

## Rules:
1. Output ONLY the Cypher query — no explanation, no markdown fences, no comments.
2. NEVER use DELETE, CREATE, MERGE, SET, REMOVE, or any write operation.
3. Always add LIMIT (max 25) to prevent runaway queries.
4. Use fuzzy matching with toLower() when appropriate.
5. Return ONLY: entity names, relationship types, relationship context, chunk_ids.
6. If the question cannot be answered with a graph query, output: CANNOT_GENERATE
"""

_HUMAN_PROMPT = """User question: {question}

Detected entities in question: {entities}

Generate a Cypher query to retrieve relevant information from the knowledge graph.
"""


# ──────────────────────────────────────────────────────────────────────────────
# LLM-generated Cypher result
# ──────────────────────────────────────────────────────────────────────────────

class CypherQueryResult(BaseModel):
    cypher: Optional[str] = None
    raw_response: str = ""
    is_valid: bool = False
    validation_error: Optional[str] = None


class TextToCypherEngine:
    """
    Converts natural-language questions to validated read-only Cypher.

    Returns a CypherQueryResult — callers should check `is_valid`
    before executing.
    """

    def __init__(self) -> None:
        self._llm = get_chat_llm(temperature=0)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])
        self._chain = self._prompt | self._llm

    async def generate(
        self,
        question: str,
        entities: list[str],
    ) -> CypherQueryResult:
        """Generate and validate Cypher for a question + detected entity list."""
        try:
            response = await self._chain.ainvoke({
                "question": question,
                "entities": ", ".join(entities) if entities else "none detected",
            })
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                parts = [
                    p if isinstance(p, str) else (p.get("text", "") if isinstance(p, dict) else str(p))
                    for p in content
                ]
                raw = "".join(parts).strip()
            else:
                raw = str(content).strip()

            if raw == "CANNOT_GENERATE":
                log.info("cypher_cannot_generate", question=question[:80])
                return CypherQueryResult(raw_response=raw, is_valid=False)

            validated = validate_cypher(raw)
            log.info("cypher_generated", cypher=validated[:120])
            return CypherQueryResult(cypher=validated, raw_response=raw, is_valid=True)

        except CypherValidationError as exc:
            log.warning("cypher_validation_failed", error=str(exc))
            return CypherQueryResult(
                raw_response="",
                is_valid=False,
                validation_error=str(exc),
            )
        except Exception as exc:
            log.error("cypher_generation_error", error=str(exc))
            return CypherQueryResult(is_valid=False, validation_error=str(exc))

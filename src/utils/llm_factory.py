"""
LLM and Embeddings factory module.

Dynamically provisions Chat LLM and Embeddings instances based on the
configured provider (Google Gemini or OpenAI).
"""
from __future__ import annotations

import os
from typing import Any, Optional

from src.config import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


def get_chat_llm(
    temperature: float = 0.0,
    model: Optional[str] = None,
    streaming: bool = False,
    max_output_tokens: Optional[int] = None,
) -> Any:
    """
    Return a LangChain BaseChatModel instance configured per settings.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
            )

        chat_model = model or settings.gemini_chat_model
        log.debug("init_gemini_chat", model=chat_model, temperature=temperature)
        kwargs: dict[str, Any] = {
            "model": chat_model,
            "google_api_key": api_key,
            "temperature": temperature,
            "streaming": streaming,
        }
        if max_output_tokens:
            kwargs["max_output_tokens"] = max_output_tokens

        return ChatGoogleGenerativeAI(**kwargs)

    else:
        from langchain_openai import ChatOpenAI

        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        chat_model = model or settings.openai_chat_model
        log.debug("init_openai_chat", model=chat_model, temperature=temperature)
        kwargs: dict[str, Any] = {
            "model": chat_model,
            "api_key": api_key,
            "temperature": temperature,
            "streaming": streaming,
        }
        if max_output_tokens:
            kwargs["max_tokens"] = max_output_tokens

        return ChatOpenAI(**kwargs)


def get_embeddings() -> Any:
    """
    Return a LangChain Embeddings instance configured per settings.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
            )

        emb_model = settings.gemini_embedding_model
        log.debug("init_gemini_embeddings", model=emb_model, dimensions=settings.gemini_embedding_dimensions)
        return GoogleGenerativeAIEmbeddings(
            model=emb_model,
            google_api_key=api_key,
            output_dimensionality=settings.gemini_embedding_dimensions,
        )

    else:
        from langchain_openai import OpenAIEmbeddings

        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        emb_model = settings.openai_embedding_model
        log.debug("init_openai_embeddings", model=emb_model)
        return OpenAIEmbeddings(
            model=emb_model,
            api_key=api_key,
            dimensions=settings.openai_embedding_dimensions,
        )

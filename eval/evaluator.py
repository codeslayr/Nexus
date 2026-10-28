"""
Automated Evaluation Harness — Section 5.

Uses Ragas to measure:
  - Faithfulness (hallucination audit)
  - Answer Relevance
  - Context Precision / Recall
  - Multi-hop Retrieval Recall vs. vector-only baseline

Run: python -m eval.evaluator --dataset eval/data/eval_dataset.json
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd
import typer
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from rich.console import Console
from rich.table import Table

from src.config import get_settings
from src.core.retriever import HybridGraphRAGRetriever
from src.utils.llm_factory import get_chat_llm, get_embeddings
from src.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)
console = Console()
app = typer.Typer()


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation dataset schema
# ──────────────────────────────────────────────────────────────────────────────

# Expected JSON structure:
# [
#   {
#     "question": "Who acquired Twitter?",
#     "ground_truth": "Elon Musk acquired Twitter in October 2022.",
#     "reference_chunk_ids": ["uuid-1", "uuid-2"]  # optional
#   },
#   ...
# ]


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval runners
# ──────────────────────────────────────────────────────────────────────────────

async def _run_hybrid_retrieval(questions: list[str]) -> list[dict]:
    """Run HybridGraphRAGRetriever on all questions, return structured results."""
    retriever = HybridGraphRAGRetriever(use_cross_encoder=False)
    results = []

    for question in questions:
        t0 = time.perf_counter()
        context = await retriever.retrieve(question)
        latency_ms = (time.perf_counter() - t0) * 1000

        contexts = [c.chunk_text for c in context.vector_chunks]
        contexts += [p.summary for p in context.graph_paths]

        results.append({
            "question": question,
            "contexts": contexts,
            "latency_ms": round(latency_ms, 2),
            "source": "hybrid",
        })

    return results


async def _run_vector_only_retrieval(questions: list[str]) -> list[dict]:
    """Baseline: vector-only retrieval without graph traversal."""
    vector_store = VectorStore()
    llm = get_chat_llm(temperature=0)
    results = []

    for question in questions:
        t0 = time.perf_counter()
        chunks = await vector_store.similarity_search(question, top_k=10)
        context_text = "\n\n".join(c.chunk_text for c in chunks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using only the context provided."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        chain = prompt | llm | StrOutputParser()
        answer = await chain.ainvoke({"context": context_text, "question": question})
        latency_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "question": question,
            "contexts": [c.chunk_text for c in chunks],
            "answer": answer,
            "latency_ms": round(latency_ms, 2),
            "source": "vector_only",
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# LLM answer synthesis (for hybrid path)
# ──────────────────────────────────────────────────────────────────────────────

async def _synthesize_answers(
    retrieval_results: list[dict],
) -> list[dict]:
    """Run LLM synthesis over retrieved contexts to produce final answers."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = get_chat_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context. Be concise and precise."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    for result in retrieval_results:
        if "answer" not in result:
            context_text = "\n\n---\n\n".join(result["contexts"][:8])
            result["answer"] = await chain.ainvoke({
                "context": context_text,
                "question": result["question"],
            })
    return retrieval_results


# ──────────────────────────────────────────────────────────────────────────────
# Ragas evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _build_ragas_dataset(
    eval_items: list[dict],
    retrieval_results: list[dict],
) -> Dataset:
    """Build a HuggingFace Dataset in the format Ragas expects."""
    rows = []
    result_map = {r["question"]: r for r in retrieval_results}

    for item in eval_items:
        q = item["question"]
        if q not in result_map:
            continue
        r = result_map[q]
        rows.append({
            "question": q,
            "answer": r.get("answer", ""),
            "contexts": r.get("contexts", []),
            "ground_truth": item.get("ground_truth", ""),
        })

    return Dataset.from_list(rows)


def _run_ragas_evaluation(dataset: Dataset, label: str) -> pd.DataFrame:
    """Run Ragas evaluation and return a DataFrame of scores."""
    llm = LangchainLLMWrapper(get_chat_llm(temperature=0))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    console.print(f"\n[bold cyan]Running Ragas evaluation: {label}[/bold cyan]")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    df = result.to_pandas()
    df["retrieval_source"] = label
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def run_evaluation(
    dataset_path: Path = typer.Argument(
        ..., help="Path to eval_dataset.json"
    ),
    output_path: Optional[Path] = typer.Option(
        None, help="Path to save results CSV (default: eval/results/)"
    ),
    skip_baseline: bool = typer.Option(
        False, help="Skip vector-only baseline comparison"
    ),
) -> None:
    """
    Run Ragas evaluation comparing Hybrid GraphRAG vs. vector-only baseline.
    """
    # Load eval dataset
    with open(dataset_path) as f:
        eval_items: list[dict] = json.load(f)

    questions = [item["question"] for item in eval_items]
    console.print(f"[bold]Loaded {len(questions)} evaluation questions[/bold]")

    # ── Run retrievals ──────────────────────────────────────────────────────
    console.print("\n[yellow]-> Running Hybrid GraphRAG retrieval...[/yellow]")
    hybrid_results = asyncio.run(_run_hybrid_retrieval(questions))
    hybrid_results = asyncio.run(_synthesize_answers(hybrid_results))

    all_dfs = []

    hybrid_dataset = _build_ragas_dataset(eval_items, hybrid_results)
    hybrid_df = _run_ragas_evaluation(hybrid_dataset, "Hybrid GraphRAG")
    all_dfs.append(hybrid_df)

    if not skip_baseline:
        console.print("\n[yellow]-> Running Vector-Only baseline retrieval...[/yellow]")
        baseline_results = asyncio.run(_run_vector_only_retrieval(questions))
        baseline_dataset = _build_ragas_dataset(eval_items, baseline_results)
        baseline_df = _run_ragas_evaluation(baseline_dataset, "Vector-Only Baseline")
        all_dfs.append(baseline_df)

    # ── Aggregate & display ─────────────────────────────────────────────────
    combined_df = pd.concat(all_dfs, ignore_index=True)

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = (
        combined_df
        .groupby("retrieval_source")[metric_cols]
        .mean()
        .round(4)
    )

    # Rich table output
    table = Table(title="GraphRAG Evaluation Results", show_header=True)
    table.add_column("Retrieval Method", style="bold cyan")
    for col in metric_cols:
        table.add_column(col.replace("_", " ").title(), justify="right")

    for source, row in summary.iterrows():
        table.add_row(source, *[f"{v:.4f}" for v in row])

    console.print("\n")
    console.print(table)

    # ── Save outputs ────────────────────────────────────────────────────────
    out_dir = output_path or Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_dir / "eval_results_full.csv", index=False)
    summary.to_csv(out_dir / "eval_summary.csv")
    console.print(f"\n[green]Results saved to {out_dir}[/green]")

    # Print latency comparison
    hybrid_latency = pd.Series([r["latency_ms"] for r in hybrid_results])
    console.print(f"\n[bold]Hybrid latency:[/bold] mean={hybrid_latency.mean():.0f}ms  "
                  f"p95={hybrid_latency.quantile(0.95):.0f}ms")

    if not skip_baseline:
        baseline_latency = pd.Series([r["latency_ms"] for r in baseline_results])
        console.print(f"[bold]Baseline latency:[/bold] mean={baseline_latency.mean():.0f}ms  "
                      f"p95={baseline_latency.quantile(0.95):.0f}ms")


if __name__ == "__main__":
    app()

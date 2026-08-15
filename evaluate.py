"""
Reproducible evaluation suite for the AI Investment Chatbot.

This script intentionally reports measured results only. It does not modify the
Streamlit application or manufacture benchmark numbers.

Run:
    python evaluate.py --all
    python evaluate.py --nlp
    python evaluate.py --latency
    python evaluate.py --forecast

Requirements are the same as the application plus scikit-learn.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ---------------------------------------------------------------------------
# NLP benchmark
# ---------------------------------------------------------------------------

SECTOR_KEYWORDS = [
    "technology", "finance", "healthcare", "consumer goods", "energy",
    "utilities", "materials", "industrials", "telecommunications",
    "real estate", "consumer services", "transportation", "agriculture",
    "media and entertainment", "government",
]

INVESTMENT_GOALS = {
    "retirement": ["retirement", "retirement fund", "retirement savings"],
    "education": ["education", "college", "higher education", "education funding", "saving for education"],
    "home purchase": ["buying a home", "down payment", "house purchase", "real estate investment"],
    "wealth accumulation": ["building wealth", "asset accumulation", "wealth growth", "capital appreciation"],
    "emergency savings": ["emergency fund", "emergency savings", "financial safety net", "unexpected expenses"],
    "major purchases": ["major purchases", "large purchases", "buying a car", "home renovations"],
    "vacation": ["vacation", "travel", "holiday planning", "saving for vacation"],
    "debt repayment": ["debt repayment", "paying off debt", "debt reduction", "loan repayment"],
    "healthcare": ["healthcare", "medical expenses", "health savings", "wellness"],
    "charitable giving": ["charitable giving", "donations", "charity", "philanthropy"],
    "estate planning": ["estate planning", "inheritance planning", "financial security for heirs"],
    "business investment": ["business investment", "starting a business", "entrepreneurship", "business growth"],
    "tax planning": ["tax planning", "tax minimization", "tax liability management"],
    "legacy building": ["legacy building", "impact investing", "supporting causes", "creating a lasting impact"],
    "growth": ["investment growth", "capital appreciation", "value increase", "wealth growth"],
}

# Fixed, human-readable evaluation set. Labels are intentionally explicit so
# the resulting score can be audited rather than inferred from model output.
NLP_TEST_CASES: List[Tuple[str, str, str]] = [
    ("I want to save for retirement with low risk", "retirement", "low"),
    ("I need a retirement fund and can accept medium risk", "retirement", "medium"),
    ("Help me build wealth aggressively", "wealth accumulation", "high"),
    ("I am saving for my child's college education", "education", "medium"),
    ("I need money for higher education in five years", "education", "medium"),
    ("I want to buy a home with low volatility", "home purchase", "low"),
    ("I need a down payment for a house", "home purchase", "medium"),
    ("I want to create an emergency fund", "emergency savings", "low"),
    ("I need a financial safety net for unexpected expenses", "emergency savings", "low"),
    ("I want to save for a vacation", "vacation", "low"),
    ("Help me plan my holiday investments", "vacation", "medium"),
    ("I want to pay off my loan", "debt repayment", "low"),
    ("I need debt reduction over the next few years", "debt repayment", "medium"),
    ("I want to invest in healthcare", "healthcare", "medium"),
    ("I want to save for medical expenses", "healthcare", "low"),
    ("I want to make charitable donations", "charitable giving", "low"),
    ("I want to plan my estate", "estate planning", "low"),
    ("I want to start a business", "business investment", "high"),
    ("Help me invest for business growth", "business investment", "high"),
    ("I need tax minimization advice", "tax planning", "medium"),
    ("I want to reduce my tax liability", "tax planning", "low"),
    ("I want to build a legacy through investing", "legacy building", "medium"),
    ("I want long term investment growth", "growth", "medium"),
    ("I want capital appreciation and high risk", "growth", "high"),
    ("Show me technology stocks", "technology", "medium"),
    ("I prefer finance companies with low risk", "finance", "low"),
    ("Find healthcare stocks for a high risk investor", "healthcare", "high"),
    ("I want energy companies with medium risk", "energy", "medium"),
    ("Show me industrial companies", "industrials", "medium"),
    ("I want real estate stocks with low volatility", "real estate", "low"),
]


def load_nlp():
    import spacy
    from sentence_transformers import SentenceTransformer, util

    nlp = spacy.load("en_core_web_md")
    sentence_model = SentenceTransformer("all-mpnet-base-v2")
    return nlp, sentence_model, util


def classify_sector(text: str, nlp) -> str:
    doc = nlp(text)
    max_similarity = 0.0
    best = ""
    for token in doc:
        for sector in SECTOR_KEYWORDS:
            similarity = nlp(sector).similarity(token)
            if similarity > max_similarity:
                max_similarity = similarity
                best = sector
    return best if max_similarity > 0.6 else "others"


def classify_goal(text: str, model, util) -> str:
    text_embedding = model.encode(text, convert_to_tensor=True)
    max_similarity = 0.0
    best = "others"
    for goal, keywords in INVESTMENT_GOALS.items():
        for keyword in keywords:
            keyword_embedding = model.encode(keyword, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(text_embedding, keyword_embedding).item()
            if similarity > max_similarity:
                max_similarity = similarity
                best = goal
    return best if max_similarity > 0.5 else "others"


def classify_risk(text: str) -> str:
    lowered = text.lower()
    # Mirrors the app's final risk-level resolution: explicit high wins,
    # followed by medium, then low; no explicit risk defaults to medium.
    if "high" in lowered:
        return "high"
    if "medium" in lowered:
        return "medium"
    if "low" in lowered:
        return "low"
    if any(word in lowered for word in ("risk", "risky", "volatility", "volatile")):
        return "medium"
    return "medium"


def run_nlp_benchmark() -> Dict:
    nlp, model, util = load_nlp()
    rows = []
    for text, expected_goal, expected_risk in NLP_TEST_CASES:
        predicted_goal = classify_goal(text, model, util)
        predicted_risk = classify_risk(text)
        rows.append({
            "text": text,
            "expected_goal": expected_goal,
            "predicted_goal": predicted_goal,
            "expected_risk": expected_risk,
            "predicted_risk": predicted_risk,
        })

    goal_true = [r["expected_goal"] for r in rows]
    goal_pred = [r["predicted_goal"] for r in rows]
    risk_true = [r["expected_risk"] for r in rows]
    risk_pred = [r["predicted_risk"] for r in rows]

    goal_p, goal_r, goal_f1, _ = precision_recall_fscore_support(
        goal_true, goal_pred, average="weighted", zero_division=0
    )
    risk_p, risk_r, risk_f1, _ = precision_recall_fscore_support(
        risk_true, risk_pred, average="weighted", zero_division=0
    )

    result = {
        "test_cases": len(rows),
        "investment_goal": {
            "accuracy": accuracy_score(goal_true, goal_pred),
            "weighted_precision": goal_p,
            "weighted_recall": goal_r,
            "weighted_f1": goal_f1,
        },
        "risk": {
            "accuracy": accuracy_score(risk_true, risk_pred),
            "weighted_precision": risk_p,
            "weighted_recall": risk_r,
            "weighted_f1": risk_f1,
        },
        "cases": rows,
    }
    print("\nNLP benchmark")
    print(f"  Labeled queries: {result['test_cases']}")
    print(f"  Goal accuracy:   {result['investment_goal']['accuracy']:.4f}")
    print(f"  Goal F1:         {result['investment_goal']['weighted_f1']:.4f}")
    print(f"  Risk accuracy:   {result['risk']['accuracy']:.4f}")
    print(f"  Risk F1:         {result['risk']['weighted_f1']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

LATENCY_QUERIES = [
    "I want to save for retirement with low risk",
    "Show me technology stocks with medium risk",
    "I want high risk healthcare investments",
    "Help me buy a home in five years",
    "I want to build wealth aggressively",
    "I need an emergency fund with low volatility",
    "I want to invest in energy companies",
    "Help me plan my college education savings",
    "I want tax planning advice",
    "Find finance companies for medium risk",
]


def run_latency_benchmark(iterations: int = 5) -> Dict:
    nlp, model, util = load_nlp()

    # Warm-up removes one-time model initialization from the measured path.
    for query in LATENCY_QUERIES[:2]:
        classify_sector(query, nlp)
        classify_goal(query, model, util)
        classify_risk(query)

    timings_ms = []
    for _ in range(iterations):
        for query in LATENCY_QUERIES:
            start = time.perf_counter()
            classify_sector(query, nlp)
            classify_goal(query, model, util)
            classify_risk(query)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        "queries": len(LATENCY_QUERIES),
        "iterations": iterations,
        "measurements": len(timings_ms),
        "mean_ms": statistics.mean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": float(np.percentile(timings_ms, 95)),
    }
    print("\nNLP query-understanding latency")
    print(f"  Queries:         {result['queries']}")
    print(f"  Measurements:    {result['measurements']}")
    print(f"  Mean:             {result['mean_ms']:.2f} ms")
    print(f"  Median:           {result['median_ms']:.2f} ms")
    print(f"  P95:              {result['p95_ms']:.2f} ms")
    return result


# ---------------------------------------------------------------------------
# Forecast evaluation
# ---------------------------------------------------------------------------


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def run_forecast_benchmark(test_fraction: float = 0.20) -> Dict:
    # Same period used by model.ipynb, making the benchmark reproducible.
    data = yf.download(
        "^NSEI",
        start="2010-01-01",
        end="2024-06-01",
        auto_adjust=False,
        progress=False,
    )
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)

    split = int(len(close) * (1.0 - test_fraction))
    train = close.iloc[:split]
    test = close.iloc[split:]

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
    ).fit()
    forecast = model.forecast(len(test))

    # Naive baseline: predict each test point using the immediately preceding
    # observed price. This is a genuine out-of-sample benchmark.
    naive = train.iloc[-1]
    naive_forecast = np.full(len(test), naive, dtype=float)

    model_mae = float(np.mean(np.abs(test.values - forecast.values)))
    model_rmse = float(np.sqrt(np.mean((test.values - forecast.values) ** 2)))
    model_mape = mean_absolute_percentage_error(test.values, forecast.values)
    naive_mae = float(np.mean(np.abs(test.values - naive_forecast)))
    naive_rmse = float(np.sqrt(np.mean((test.values - naive_forecast) ** 2)))
    naive_mape = mean_absolute_percentage_error(test.values, naive_forecast)

    result = {
        "ticker": "^NSEI",
        "start": "2010-01-01",
        "end": "2024-06-01",
        "rows": len(close),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_fraction": test_fraction,
        "exponential_smoothing": {
            "mae": model_mae,
            "rmse": model_rmse,
            "mape_percent": model_mape,
        },
        "naive_baseline": {
            "mae": naive_mae,
            "rmse": naive_rmse,
            "mape_percent": naive_mape,
        },
    }
    print("\nForecast benchmark")
    print(f"  Dataset rows:       {result['rows']}")
    print(f"  Train/test rows:    {result['train_rows']}/{result['test_rows']}")
    print(f"  ES MAE:             {model_mae:.4f}")
    print(f"  ES RMSE:            {model_rmse:.4f}")
    print(f"  ES MAPE:             {model_mape:.4f}%")
    print(f"  Naive MAE:          {naive_mae:.4f}")
    print(f"  Naive RMSE:         {naive_rmse:.4f}")
    print(f"  Naive MAPE:          {naive_mape:.4f}%")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AI Investment Chatbot")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--nlp", action="store_true", help="Run NLP classification benchmark")
    parser.add_argument("--latency", action="store_true", help="Run NLP latency benchmark")
    parser.add_argument("--forecast", action="store_true", help="Run forecasting benchmark")
    parser.add_argument("--latency-iterations", type=int, default=5)
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()

    if not any((args.all, args.nlp, args.latency, args.forecast)):
        parser.error("Choose at least one of --all, --nlp, --latency, --forecast")

    results = {}
    if args.all or args.nlp:
        results["nlp"] = run_nlp_benchmark()
    if args.all or args.latency:
        results["latency"] = run_latency_benchmark(args.latency_iterations)
    if args.all or args.forecast:
        results["forecast"] = run_forecast_benchmark()

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved reproducible results to {args.output}")


if __name__ == "__main__":
    main()

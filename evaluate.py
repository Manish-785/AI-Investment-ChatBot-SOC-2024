"""Reproducible evaluation suite for the AI Investment Chatbot.

The NLP tests evaluate three separate tasks: investment goal, sector, and
explicit risk extraction. The benchmark set deliberately does not infer risk
from a user's financial objective. All reported values are measured outputs.

Run:
    python evaluate.py --all
    python evaluate.py --nlp
    python evaluate.py --latency
    python evaluate.py --forecast
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SECTORS = [
    "technology", "finance", "healthcare", "consumer goods", "energy",
    "utilities", "materials", "industrials", "telecommunications",
    "real estate", "consumer services", "transportation", "agriculture",
    "media and entertainment", "government",
]

# Objective-oriented phrases. Sector names are intentionally excluded as goals.
GOALS = {
    "retirement": ["retirement", "retirement fund", "retirement savings", "save for retirement"],
    "education": ["education funding", "college savings", "higher education", "save for college", "education expenses"],
    "home purchase": ["buying a home", "down payment", "house purchase", "buy a house", "home purchase"],
    "wealth accumulation": ["building wealth", "asset accumulation", "wealth growth", "capital appreciation", "grow my wealth"],
    "emergency savings": ["emergency fund", "emergency savings", "financial safety net", "unexpected expenses", "rainy day fund"],
    "major purchases": ["major purchases", "large purchases", "buying a car", "home renovations", "large expense"],
    "vacation": ["vacation", "travel savings", "holiday planning", "saving for vacation", "travel fund"],
    "debt repayment": ["debt repayment", "paying off debt", "debt reduction", "loan repayment", "pay off my loan"],
    "healthcare": ["medical expenses", "health savings", "healthcare expenses", "medical costs", "health expenses"],
    "charitable giving": ["charitable giving", "donations", "charity", "philanthropy", "donate"],
    "estate planning": ["estate planning", "inheritance planning", "financial security for heirs", "plan my estate"],
    "business investment": ["business investment", "starting a business", "entrepreneurship", "business growth", "fund my business"],
    "tax planning": ["tax planning", "tax minimization", "tax liability management", "reduce my taxes", "tax savings"],
    "legacy building": ["legacy building", "impact investing", "supporting causes", "creating a lasting impact", "leave a legacy"],
    "growth": ["investment growth", "capital appreciation", "value increase", "long term growth", "grow my investment"],
}

# Paraphrased, independently written evaluation cases; labels are explicit.
GOAL_TESTS: List[Tuple[str, str]] = [
    ("I am building a nest egg for when I stop working", "retirement"),
    ("How should I invest for my retirement years", "retirement"),
    ("I need to accumulate money for university fees", "education"),
    ("Help me prepare financially for my daughter's college", "education"),
    ("I need to save enough for a house deposit", "home purchase"),
    ("My main objective is purchasing a property", "home purchase"),
    ("I want my portfolio to increase my overall wealth", "wealth accumulation"),
    ("I am focused on growing my assets over time", "wealth accumulation"),
    ("I want cash available for unexpected emergencies", "emergency savings"),
    ("Help me build a rainy day reserve", "emergency savings"),
    ("I need to save for a new car", "major purchases"),
    ("I am planning a large home renovation", "major purchases"),
    ("I want to set aside money for my next trip", "vacation"),
    ("Help me build a travel fund", "vacation"),
    ("I want to eliminate my outstanding loans", "debt repayment"),
    ("My priority is becoming debt free", "debt repayment"),
    ("I need to prepare for future medical bills", "healthcare"),
    ("I want savings specifically for medical treatment", "healthcare"),
    ("I want to donate part of my investment gains", "charitable giving"),
    ("My goal is supporting charities financially", "charitable giving"),
    ("I need to organize my finances for my heirs", "estate planning"),
    ("Help me prepare an inheritance strategy", "estate planning"),
    ("I want capital to launch a new company", "business investment"),
    ("My objective is funding my startup", "business investment"),
    ("I want to legally reduce my future tax burden", "tax planning"),
    ("Help me optimize my taxes through investing", "tax planning"),
    ("I want my investments to create a lasting impact", "legacy building"),
    ("I want to leave financial support for future generations", "legacy building"),
    ("I am primarily looking for long-term investment appreciation", "growth"),
    ("I want my capital to grow substantially over time", "growth"),
]

SECTOR_TESTS: List[Tuple[str, str]] = [
    ("Find me software and technology companies", "technology"),
    ("I want to research banks and financial institutions", "finance"),
    ("Show me pharmaceutical and medical companies", "healthcare"),
    ("I am interested in consumer packaged goods", "consumer goods"),
    ("Find oil and gas companies", "energy"),
    ("Show me electricity and power utilities", "utilities"),
    ("I want chemical and materials stocks", "materials"),
    ("Find engineering and manufacturing companies", "industrials"),
    ("I am looking for telecom companies", "telecommunications"),
    ("Show me REITs and property businesses", "real estate"),
    ("I want retail and consumer service companies", "consumer services"),
    ("Find airlines and logistics businesses", "transportation"),
    ("I want companies involved in farming and agriculture", "agriculture"),
    ("Show me entertainment and media companies", "media and entertainment"),
    ("I want defense contractors and government suppliers", "government"),
]

RISK_TESTS: List[Tuple[str, str]] = [
    ("I can tolerate low risk", "low"),
    ("Please keep my portfolio low risk", "low"),
    ("I have a low risk appetite", "low"),
    ("I prefer conservative low risk investments", "low"),
    ("I want medium risk exposure", "medium"),
    ("A medium level of risk is acceptable", "medium"),
    ("I have a medium risk tolerance", "medium"),
    ("Please target medium volatility", "medium"),
    ("I am comfortable with high risk", "high"),
    ("I want aggressive high risk investments", "high"),
    ("My risk appetite is high", "high"),
    ("I can accept high volatility", "high"),
    ("Use a low risk strategy for me", "low"),
    ("Give me a high risk portfolio", "high"),
    ("I am comfortable taking medium risk", "medium"),
]


def load_nlp():
    import re
    import spacy
    from sentence_transformers import SentenceTransformer, util

    nlp = spacy.load("en_core_web_md")
    model = SentenceTransformer("all-mpnet-base-v2")
    goal_phrases = [(goal, phrase) for goal, phrases in GOALS.items() for phrase in phrases]
    goal_embeddings = model.encode(
        [phrase for _, phrase in goal_phrases],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    sector_docs = {sector: nlp(sector) for sector in SECTORS}
    return re, nlp, model, util, goal_phrases, goal_embeddings, sector_docs


def classify_goal(text, re, model, util, goal_phrases, goal_embeddings):
    lowered = text.lower()
    for goal, phrase in goal_phrases:
        if re.search(rf"\b{re.escape(phrase.lower())}\b", lowered):
            return goal
    embedding = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    similarities = util.cos_sim(embedding, goal_embeddings)[0]
    index = int(similarities.argmax().item())
    return goal_phrases[index][0] if float(similarities[index]) > 0.5 else "others"


def classify_sector(text, re, nlp, sector_docs):
    lowered = text.lower()
    for sector in sorted(SECTORS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(sector)}\b", lowered):
            return sector
    # Phrase-level semantic fallback handles natural paraphrases.
    doc = nlp(text)
    best, score = "others", 0.0
    for sector, sector_doc in sector_docs.items():
        similarity = doc.similarity(sector_doc)
        if similarity > score:
            score, best = similarity, sector
    return best if score > 0.55 else "others"


def classify_risk(text):
    import re
    matches = re.findall(r"\b(low|medium|high)\b", text.lower())
    if "high" in matches:
        return "high"
    if "medium" in matches:
        return "medium"
    if "low" in matches:
        return "low"
    return "medium"


def score_task(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
    }


def run_nlp_benchmark() -> Dict:
    re, nlp, model, util, goal_phrases, goal_embeddings, sector_docs = load_nlp()

    goal_pred = [classify_goal(q, re, model, util, goal_phrases, goal_embeddings) for q, _ in GOAL_TESTS]
    sector_pred = [classify_sector(q, re, nlp, sector_docs) for q, _ in SECTOR_TESTS]
    risk_pred = [classify_risk(q) for q, _ in RISK_TESTS]

    result = {
        "investment_goal": {"test_cases": len(GOAL_TESTS), **score_task([x[1] for x in GOAL_TESTS], goal_pred)},
        "sector": {"test_cases": len(SECTOR_TESTS), **score_task([x[1] for x in SECTOR_TESTS], sector_pred)},
        "risk": {"test_cases": len(RISK_TESTS), **score_task([x[1] for x in RISK_TESTS], risk_pred)},
        "misclassified": {
            "goal": [(q, y, p) for (q, y), p in zip(GOAL_TESTS, goal_pred) if y != p],
            "sector": [(q, y, p) for (q, y), p in zip(SECTOR_TESTS, sector_pred) if y != p],
            "risk": [(q, y, p) for (q, y), p in zip(RISK_TESTS, risk_pred) if y != p],
        },
    }

    print("\nNLP benchmark")
    for name, metrics in result.items():
        if name == "misclassified":
            continue
        print(f"  {name.replace('_', ' ').title():18s}: {metrics['test_cases']} queries | accuracy {metrics['accuracy']:.4f} | F1 {metrics['weighted_f1']:.4f}")
    return result


LATENCY_QUERIES = [q for q, _ in GOAL_TESTS[:10]] + [q for q, _ in SECTOR_TESTS[:10]] + [q for q, _ in RISK_TESTS[:10]]


def run_latency_benchmark(iterations: int = 5) -> Dict:
    re, nlp, model, util, goal_phrases, goal_embeddings, sector_docs = load_nlp()
    for query in LATENCY_QUERIES[:3]:
        classify_goal(query, re, model, util, goal_phrases, goal_embeddings)
        classify_sector(query, re, nlp, sector_docs)
        classify_risk(query)

    timings_ms = []
    for _ in range(iterations):
        for query in LATENCY_QUERIES:
            start = time.perf_counter()
            classify_goal(query, re, model, util, goal_phrases, goal_embeddings)
            classify_sector(query, re, nlp, sector_docs)
            classify_risk(query)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        "queries": len(LATENCY_QUERIES),
        "iterations": iterations,
        "measurements": len(timings_ms),
        "mean_ms": float(statistics.mean(timings_ms)),
        "median_ms": float(statistics.median(timings_ms)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
    }
    print("\nNLP query-understanding latency")
    print(f"  Queries:      {result['queries']}")
    print(f"  Measurements: {result['measurements']}")
    print(f"  Mean:         {result['mean_ms']:.2f} ms")
    print(f"  Median:       {result['median_ms']:.2f} ms")
    print(f"  P95:          {result['p95_ms']:.2f} ms")
    return result


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def run_forecast_benchmark(test_fraction: float = 0.20) -> Dict:
    data = yf.download("^NSEI", start="2010-01-01", end="2024-06-01", auto_adjust=False, progress=False)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)

    split = int(len(close) * (1.0 - test_fraction))
    train, test = close.iloc[:split], close.iloc[split:]
    model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
    forecast = model.forecast(len(test))

    naive_forecast = np.full(len(test), train.iloc[-1], dtype=float)
    model_mae = float(np.mean(np.abs(test.values - forecast.values)))
    model_rmse = float(np.sqrt(np.mean((test.values - forecast.values) ** 2)))
    model_mape = mean_absolute_percentage_error(test.values, forecast.values)
    naive_mae = float(np.mean(np.abs(test.values - naive_forecast)))
    naive_rmse = float(np.sqrt(np.mean((test.values - naive_forecast) ** 2)))
    naive_mape = mean_absolute_percentage_error(test.values, naive_forecast)

    result = {
        "ticker": "^NSEI", "start": "2010-01-01", "end": "2024-06-01",
        "rows": len(close), "train_rows": len(train), "test_rows": len(test),
        "exponential_smoothing": {"mae": model_mae, "rmse": model_rmse, "mape_percent": model_mape},
        "naive_baseline": {"mae": naive_mae, "rmse": naive_rmse, "mape_percent": naive_mape},
        "relative_mape_reduction_percent": float((1 - model_mape / naive_mape) * 100),
    }
    print("\nForecast benchmark")
    print(f"  Dataset rows:       {result['rows']}")
    print(f"  Train/test rows:    {result['train_rows']}/{result['test_rows']}")
    print(f"  ES MAE:             {model_mae:.4f}")
    print(f"  ES RMSE:            {model_rmse:.4f}")
    print(f"  ES MAPE:            {model_mape:.4f}%")
    print(f"  Naive MAPE:         {naive_mape:.4f}%")
    print(f"  MAPE reduction:     {result['relative_mape_reduction_percent']:.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate the AI Investment Chatbot")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--nlp", action="store_true")
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--forecast", action="store_true")
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
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

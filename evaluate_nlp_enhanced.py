"""Benchmark the domain-aware NLP candidate against the fixed evaluation set."""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluate import NLP_TEST_CASES, LATENCY_QUERIES
from nlp_enhanced import InvestmentNLP


def metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return accuracy_score(y_true, y_pred), f1


def main():
    clf = InvestmentNLP()

    goal_true, goal_pred = [], []
    sector_true, sector_pred = [], []
    risk_true, risk_pred = [], []

    for text, expected, expected_risk in NLP_TEST_CASES:
        result = clf.classify(text)
        # The fixed NLP set contains both goal and sector cases. Sector labels
        # are recognized by membership in the production sector taxonomy.
        if expected in clf.sectors:
            sector_true.append(expected)
            sector_pred.append(result["sector"])
        else:
            goal_true.append(expected)
            goal_pred.append(result["goal"])
        risk_true.append(expected_risk)
        risk_pred.append(result["risk"])

    print("Enhanced NLP benchmark")
    print(f"  Goal   : {len(goal_true)} queries | accuracy {metrics(goal_true, goal_pred)[0]:.4f} | F1 {metrics(goal_true, goal_pred)[1]:.4f}")
    print(f"  Sector : {len(sector_true)} queries | accuracy {metrics(sector_true, sector_pred)[0]:.4f} | F1 {metrics(sector_true, sector_pred)[1]:.4f}")
    print(f"  Risk   : {len(risk_true)} queries | accuracy {metrics(risk_true, risk_pred)[0]:.4f} | F1 {metrics(risk_true, risk_pred)[1]:.4f}")

    print("\nMisclassifications")
    for text, expected, expected_risk in NLP_TEST_CASES:
        result = clf.classify(text)
        if expected in clf.sectors:
            predicted = result["sector"]
        else:
            predicted = result["goal"]
        if predicted != expected:
            print(f"  {text!r}: expected={expected!r}, predicted={predicted!r}")


if __name__ == "__main__":
    main()

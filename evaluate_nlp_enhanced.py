"""Benchmark the domain-aware NLP candidate against the fixed evaluation set."""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluate import GOAL_TESTS, SECTOR_TESTS, RISK_TESTS
from nlp_enhanced import InvestmentNLP


def metrics(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return accuracy_score(y_true, y_pred), f1


def main():
    clf = InvestmentNLP()

    goal_true, goal_pred = zip(*[(expected, clf.classify(text)["goal"]) for text, expected in GOAL_TESTS])
    sector_true, sector_pred = zip(*[(expected, clf.classify(text)["sector"]) for text, expected in SECTOR_TESTS])
    risk_true, risk_pred = zip(*[(expected, clf.classify(text)["risk"]) for text, expected in RISK_TESTS])

    print("Enhanced NLP benchmark")
    print(f"  Goal   : {len(goal_true)} queries | accuracy {metrics(goal_true, goal_pred)[0]:.4f} | F1 {metrics(goal_true, goal_pred)[1]:.4f}")
    print(f"  Sector : {len(sector_true)} queries | accuracy {metrics(sector_true, sector_pred)[0]:.4f} | F1 {metrics(sector_true, sector_pred)[1]:.4f}")
    print(f"  Risk   : {len(risk_true)} queries | accuracy {metrics(risk_true, risk_pred)[0]:.4f} | F1 {metrics(risk_true, risk_pred)[1]:.4f}")

    print("\nMisclassifications")
    for cases, field in [(GOAL_TESTS, "goal"), (SECTOR_TESTS, "sector"), (RISK_TESTS, "risk")]:
        for text, expected in cases:
            predicted = clf.classify(text)[field]
            if predicted != expected:
                print(f"  [{field}] {text!r}: expected={expected!r}, predicted={predicted!r}")


if __name__ == "__main__":
    main()

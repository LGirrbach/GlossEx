import pandas as pd

from typing import List
from typing import Dict
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from textbook_shortcuts import textbook_renaming


def get_textbook_lemmas(
    parsed_textbooks: dict, labels: Dict[str, bool]
) -> Dict[str, set]:
    textbook_lemmas = defaultdict(lambda: defaultdict(int))

    for title, content in parsed_textbooks.items():
        if title.endswith(".txt"):
            title = title[:-4]

        for section in content:
            for token in section["text"]["tokens"]:
                lemma = token[1]
                if lemma in labels:
                    textbook_lemmas[title][lemma] += 1

    textbook_lemmas = {
        title: set([lemma for lemma, count in counts.items() if count >= 3])
        for title, counts in textbook_lemmas.items()
    }

    return textbook_lemmas


def evaluate(
    predictions: Dict[str, List[str]],
    textbook_lemmas: Dict[str, set],
    labels: Dict[str, bool],
) -> pd.DataFrame:
    results = []

    for textbook_title, lemmas in textbook_lemmas.items():
        y_true = [bool(labels[lemma]) for lemma in lemmas]
        y_pred = [lemma in predictions[textbook_title] for lemma in lemmas]

        scores = {
            "title": textbook_renaming[textbook_title],
            "precision": precision_score(
                y_true=y_true, y_pred=y_pred, zero_division=0.0
            ),
            "recall": recall_score(y_true=y_true, y_pred=y_pred, zero_division=0.0),
            "f1": f1_score(y_true=y_true, y_pred=y_pred, zero_division=0.0),
        }
        results.append(scores)

    results_df = pd.DataFrame.from_records(results)
    return results_df

import numpy as np
import pandas as pd

from typing import Dict
from typing import List
from rake_nltk import Rake
from stopwords import stopwords
from collections import defaultdict
from sklearn.metrics import f1_score
from glossary_mapping import glossary_mapping
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict


# Initialize glossary predictions
glossaries = pd.read_csv("./data/glossaries.csv")
glossary_predictions = defaultdict(set)

for _, row in glossaries.iterrows():
    glossary_predictions[row["title"]].add(row["entry"])


def seedlist_baseline(economics_seedlist: List[str]) -> List[str]:
    return list(sorted(set(economics_seedlist)))


def glossary_baseline(textbook_title: str) -> List[str]:
    if textbook_title.endswith(".txt"):
        textbook_title = textbook_title[:-4]

    if textbook_title in glossary_mapping:
        textbook_title = glossary_mapping[textbook_title]
        return list(sorted(glossary_predictions[textbook_title]))
    else:
        return []


def tfidf_baseline(
    parsed_textbooks, textbook_title: str, labels: Dict[str, bool], textbook_lemmas
) -> List[str]:
    content = parsed_textbooks[textbook_title]

    if textbook_title.endswith(".txt"):
        textbook_title = textbook_title[:-4]
    lemmas = textbook_lemmas[textbook_title]

    term_frequency = defaultdict(int)
    inverse_document_frequency = defaultdict(int)
    num_documents = len(content)

    for section in content:
        local_terms = set()
        for token in section["text"]["tokens"]:
            term_frequency[token[1]] += 1
            local_terms.add(token[1])

        for term in local_terms:
            inverse_document_frequency[term] += 1

    # Normalize term frequencies
    total_term_frequency = sum(term_frequency.values())
    term_frequency = {
        term: count / total_term_frequency for term, count in term_frequency.items()
    }

    # Normalize inverse document frequencies
    inverse_document_frequency = {
        term: np.log(num_documents / count)
        for term, count in inverse_document_frequency.items()
    }

    # Compute TF-IDF scores
    tfidf_scores = {
        term: term_frequency[term] * inverse_document_frequency[term]
        for term in term_frequency.keys()
    }

    # Find optimal threshold (uses label information -> make baseline stronger)
    ordered_terms, tfidf_scores = zip(
        *sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    )
    ordered_terms = list(ordered_terms)
    tfidf_scores = list(tfidf_scores)

    best_f1 = 0.0
    best_predictions = []
    for index in range(0, len(ordered_terms)):
        predictions = set(ordered_terms[:index])

        y_true = [bool(labels[lemma]) for lemma in lemmas]
        y_pred = [lemma in predictions for lemma in lemmas]
        f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0.0)

        if f1 > best_f1:
            best_f1 = f1
            best_predictions = list(sorted(predictions))

    return best_predictions


def rake_baseline(parsed_textbooks, textbook_title: str) -> List[str]:
    content = parsed_textbooks[textbook_title]
    all_lemmas = set()

    # Collect sentences
    sentences = []
    for section in content:
        for sentence in section["text"]["sentences"]:
            _, sentence_lemmas, _ = zip(*sentence)
            merged_sentence = " ".join(sentence_lemmas)
            merged_sentence = merged_sentence.replace("„", '"')
            sentences.append(merged_sentence)

            all_lemmas.update(sentence_lemmas)

    # Instantiate RAKE
    r = Rake(
        stopwords=stopwords,
        language="german",
        max_length=1,
        min_length=1,
        include_repeated_phrases=False,
        word_tokenizer=lambda text: text.split(" "),
    )

    # Extract keywords
    r.extract_keywords_from_sentences(sentences)
    keywords = r.get_ranked_phrases()
    keywords = list(sorted(set(keywords)))

    truecase_keywords = []
    for keyword in keywords:
        if keyword in all_lemmas:
            truecase_keywords.append(keyword)
        else:
            truecase_keywords.append(keyword.title())

    # Return
    return truecase_keywords


def supervised_baseline(
    textbook_title: str, fasttext_model, labels: Dict[str, bool], textbook_lemmas
) -> List[str]:
    # Collect lemmas and embeddings
    if textbook_title.endswith(".txt"):
        textbook_title = textbook_title[:-4]

    lemmas = textbook_lemmas[textbook_title]
    embeddings = np.stack([fasttext_model[lemma] for lemma in lemmas])

    # Collect labels
    y_true = [bool(labels[lemma]) for lemma in lemmas]

    # Get cross-validation predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(
        MLPClassifier(
            hidden_layer_sizes=[512, 256],
            batch_size=32,
            shuffle=True,
            random_state=42,
        ),
        embeddings,
        y_true,
        cv=cv,
        n_jobs=-1,
    )

    predictions = [lemma for lemma, label in zip(lemmas, y_pred) if label]
    predictions = list(sorted(set(predictions)))
    return predictions

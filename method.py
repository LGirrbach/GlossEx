import re
import numpy as np
import pandas as pd

from stopwords import stopwords
from collections import defaultdict
from background_corpora import Corpus
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering


def print_textbook_summary(parsed_textbooks) -> None:
    for title in sorted(parsed_textbooks.keys()):
        content = parsed_textbooks[title]
        num_tokens = 0
        unique_lemmas = set()

        for section in content:
            for token in section["text"]["tokens"]:
                num_tokens += 1
                unique_lemmas.add(token[1])

        textbook_summary.append(
            {
                "title": title,
                "#Tokens": num_tokens,
                "#Unique Lemmas": len(unique_lemmas),
            }
        )

    textbook_summary = pd.DataFrame.from_records(textbook_summary)
    print(textbook_summary)


def compute_lemma_scores(parsed_textbooks: dict, textbook_title: str, corpus: Corpus):
    possible_tags = defaultdict(list)
    textbook_vocab_counts = defaultdict(int)
    all_textbook_tokens = []

    content = parsed_textbooks[textbook_title]

    for section in content:
        for token in section["text"]["tokens"]:
            all_textbook_tokens.append(token[1])
            textbook_vocab_counts[token[1]] += 1
            possible_tags[token[1]].append(token[2])

    textbook_vocab_scores = corpus.get_vocabulary_by_specificity(all_textbook_tokens)

    return textbook_vocab_scores, textbook_vocab_counts, possible_tags


def method(
    parsed_textbooks,
    textbook_title,
    corpus,
    fasttext_model,
    education_seeds,
    economics_seeds,
    lemma_embeddings,
):
    # Initialize result dictionaries
    discarded_vocab = []

    # Load static seed list vocab embeddings
    education_embeddings_fasttext = np.stack(
        [fasttext_model[seed] for seed in education_seeds]
    )
    economics_embeddings_fasttext = np.stack(
        [fasttext_model[seed] for seed in economics_seeds]
    )

    # Load lemma scores
    textbook_vocab, textbook_vocab_counts, possible_tags = compute_lemma_scores(
        parsed_textbooks, textbook_title, corpus
    )

    # Filter Lemmas
    textbook_lemmas = set(textbook_vocab.keys())
    textbook_vocab_filtered = {
        lemma: score
        for lemma, score in textbook_vocab.items()
        if lemma not in stopwords
        and len(lemma) > 3
        and re.match(r"\w+$", lemma)
        and not re.match(r"\d+$", lemma)
        and textbook_vocab_counts[lemma] >= 3
        and any([tag == "NN" or tag.startswith("VV") for tag in possible_tags[lemma]])
    }

    textbook_vocab_filtered = {
        lemma: score for lemma, score in textbook_vocab_filtered.items() if score > 3.2
    }

    discarded_vocab = [
        lemma for lemma in textbook_lemmas if lemma not in textbook_vocab_filtered
    ]
    discarded_vocab = list(sorted(set(discarded_vocab)))

    textbook_vocab = list(sorted(textbook_vocab_filtered.keys()))

    if not textbook_vocab:
        return [], [], discarded_vocab

    # Cluster Lemma Embeddings
    textbook_embeddings = np.stack(
        [lemma_embeddings[lemma] for lemma in textbook_vocab]
    )
    cluster_labels = AgglomerativeClustering(
        n_clusters=max(1, len(textbook_vocab) // 4)
    )
    cluster_labels = cluster_labels.fit(textbook_embeddings)
    cluster_labels = cluster_labels.labels_
    clusters = {int(label): list() for label in set(cluster_labels)}

    for key, label in zip(textbook_vocab, cluster_labels):
        clusters[int(label)].append(key)

    # Extract Economics Words by Labelling Clusters
    economics_words = set()
    education_example_words = set()

    for label, items in clusters.items():
        cluster_words_embeddings = np.stack([fasttext_model[word] for word in items])
        education_dists = cdist(
            cluster_words_embeddings, education_embeddings_fasttext, metric="cosine"
        )
        education_dists = education_dists.reshape((-1,))
        education_dists = education_dists[np.argsort(education_dists)[:10]]
        economics_dists = cdist(
            cluster_words_embeddings, economics_embeddings_fasttext, metric="cosine"
        )
        economics_dists = economics_dists.reshape((-1,))
        economics_dists = economics_dists[np.argsort(economics_dists)[:10]]

        education_score = np.mean(education_dists).item()
        economics_score = np.mean(economics_dists)
        # print(economics_score, education_score)
        # print()

        if economics_score <= education_score and (
            economics_score < 0.3 or education_score - economics_score > 0.03
        ):
            for word in items:
                economics_words.add(word)
        else:
            for word in items:
                education_example_words.add(word)

    economics_words = list(sorted(economics_words))
    education_example_words = list(sorted(education_example_words))

    return economics_words, education_example_words, discarded_vocab

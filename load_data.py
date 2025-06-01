import os
import json
import nltk
import pickle
import pandas as pd

from typing import List
from typing import Dict
from typing import Tuple
from HanTa.HanoverTagger import HanoverTagger
from compress_fasttext.models import CompressedFastTextKeyedVectors

tagger = HanoverTagger("morphmodel_ger.pgz")


def load_textbook_data(
    parsed_textbooks_path: str = "./data/parsed_textbooks.json",
    lemma_embeddings_path: str = "./data/embeddings.pickle",
) -> dict:
    # Load parsed textbooks
    with open(parsed_textbooks_path, "r") as f:
        parsed_textbooks = json.load(f)

    # Load lemma embeddings
    with open(lemma_embeddings_path, "rb") as f:
        lemma_embeddings = pickle.load(f)

    # Return
    return parsed_textbooks, lemma_embeddings


def load_fasttext_model():
    # Load fastText model
    if not os.path.exists("data/models/fasttext-de-mini"):
        raise FileNotFoundError(
            "fastText model not found. Place the fastText model in data/models/"
        )

    fasttext_model = CompressedFastTextKeyedVectors.load("data/models/fasttext-de-mini")
    return fasttext_model


def load_seedlists(seedlist: str = "custom") -> Tuple[List[str], List[str]]:
    # Load custom seed lists
    if seedlist == "custom":
        with open("./data/seed_lists/wirtschaft_vokabel.txt") as esf:
            economics_seeds = [line.strip() for line in esf]
        with open("./data/seed_lists/schule_vokabeln.txt") as esf:
            education_seeds = [line.strip() for line in esf]

    elif seedlist == "wikipedia":
        with open("./data/seed_lists/wirtschafts_vokabeln_wikipedia.txt") as esf:
            economics_seeds = set()
            for line in esf:
                line = line.strip()
                parsed_line = tagger.tag_sent(
                    nltk.word_tokenize(line, language="german")
                )
                lemmas = [lemma for _, lemma, _ in parsed_line]
                economics_seeds.update(lemmas)

        with open("./data/seed_lists/schule_vokabeln_wikipedia.txt") as esf:
            education_seeds = set()
            for line in esf:
                line = line.strip()
                parsed_line = tagger.tag_sent(
                    nltk.word_tokenize(line, language="german")
                )
                lemmas = [lemma for _, lemma, _ in parsed_line]
                education_seeds.update(lemmas)
    else:
        raise ValueError(f"Unknown economics seedlist: {seedlist}")

    # Return
    return list(sorted(education_seeds)), list(sorted(economics_seeds))


def load_labels(
    label_csv_path: str = "./data/annotations/lemma_counts_sufficiently_frequent_Felix.csv",
) -> Dict[str, bool]:
    all_labels = pd.read_csv(label_csv_path, index_col=0)
    all_labels = all_labels.dropna(axis=0)
    all_labels = dict(
        list(zip(all_labels["Lemma"].to_list(), all_labels["Fachvokabular"].to_list()))
    )
    all_labels = {lemma: bool(int(label)) for lemma, label in all_labels.items()}

    # Return
    return all_labels

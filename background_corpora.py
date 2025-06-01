import numpy as np
import pandas as pd

from typing import List
from typing import Union
from stopwords import stopwords
from collections import Counter
from collections import defaultdict


class DeReKoCorpus:
    def __init__(
        self,
        dereko_file: str = "data/models/DeReKo-2014-II-MainArchive-STT.100000.freq",
    ):
        known_tokens = set()
        known_lemmas = set()
        lemma_frequencies = defaultdict(lambda: 1e7)
        corpus_frequencies = defaultdict(float)

        with open(dereko_file) as freq_file:
            for i, line in enumerate(freq_file):
                try:
                    token, lemma, _, count = line.strip().split("\t")
                except ValueError:
                    continue

                known_tokens.add(token.strip())
                known_lemmas.add(lemma)
                if lemma in stopwords or not lemma.isalpha():
                    continue

                else:
                    count = float(count)
                    corpus_frequencies[lemma] += count

                    if i < lemma_frequencies[lemma]:
                        lemma_frequencies[lemma] = i

        self.corpus_frequencies_dict = {
            word: score for word, score in corpus_frequencies.items()
        }
        self.C = sum(corpus_frequencies.values())
        self.corpus_frequencies = corpus_frequencies

    def get_vocabulary_by_specificity(self, tokens: List[str]):
        """
        Implements the specificity score from
        Two methods for extracting 'specific' single-word terms from specialized corpora
        (Lemay, L'homme, and Drouin, 2005)
        """
        token_counts = Counter(tokens)
        d = len(tokens)
        vocabulary = dict()

        for token in token_counts:
            a = self.corpus_frequencies.get(token, 0)
            b = token_counts[token]

            # Calculate parameters of normal approximation to binomial distribution
            mu = d * ((a + b) / (self.C + d))
            sigma = np.sqrt(mu * (1 - ((a + b) / (self.C + d))))

            # Calculate Z-score of frequency of token in analysis corpus
            z = (b - mu) / sigma

            vocabulary[token] = z

        return vocabulary


class SubtlexCorpus(DeReKoCorpus):
    def __init__(
        self, subtlex_file: str = "data/models/SUBTLEX-DE_cleaned_with_Google00.txt"
    ):
        subtlex = pd.read_csv(subtlex_file, sep="\t")
        self.corpus_frequencies = {
            word: count
            for word, count in zip(
                subtlex["Word"].tolist(), subtlex["WFfreqcount"].tolist()
            )
        }
        self.C = sum(self.corpus_frequencies.values())


Corpus = Union[DeReKoCorpus, SubtlexCorpus]


def load_background_corpus(corpus: str) -> Corpus:
    if corpus == "dereko":
        return DeReKoCorpus()
    elif corpus == "subtlex":
        return SubtlexCorpus()
    else:
        raise ValueError(f"Unknown corpus: {corpus}")

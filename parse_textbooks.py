import os
import nltk
import json
import pickle
import argparse
import numpy as np


from tqdm.auto import tqdm
from tqdm.auto import trange
from transformers import pipeline
from collections import defaultdict
from HanTa.HanoverTagger import HanoverTagger

tagger = HanoverTagger("morphmodel_ger.pgz")


def parse(text: str):
    parsed_sentences = []
    parsed_tokens = []

    for sentence in nltk.sent_tokenize(text, language="german"):
        parsed_sentence = tagger.tag_sent(
            nltk.word_tokenize(sentence, language="german")
        )
        parsed_sentences.append(parsed_sentence)
        parsed_tokens.extend(parsed_sentence)

    return {"sentences": parsed_sentences, "tokens": parsed_tokens}


def get_textbook_sections(path: str):
    with open(path) as tf:
        textbook_lines = tf.readlines()

    sections = []
    for line in textbook_lines:
        if line.startswith("#"):
            sections.append({"title": "", "text": "", "page": "", "category": ""})
            section_metadata = line.strip().split(",", maxsplit=2)

            if len(section_metadata) == 3:
                _, page, category = section_metadata
                page = page.strip()
                category = category.strip()
            elif len(section_metadata) == 2:
                _, category = section_metadata
                category = category.strip()
                page = None
            else:
                print(path)
                raise ValueError(f"Unknown header: {line}")

            sections[-1]["category"] = category
            sections[-1]["page"] = page

        elif not sections:
            continue

        elif line.startswith("@"):
            sections[-1]["title"] += f"\n{line.strip().lstrip('@')}"
            sections[-1]["title"] = sections[-1]["title"].strip()

        else:
            sections[-1]["text"] += line

    for section in sections:
        section["text"] = parse(section["text"])

    return sections


def get_bert_embeddings(parsed_textbooks):
    # Load BERT Model
    embedder = pipeline("feature-extraction", model="bert-base-german-cased")

    # Split sentences
    all_sentences = []
    all_sentence_tokens = []
    all_sentence_lemmas = []

    for textbook in parsed_textbooks.values():
        for section in textbook:
            for sentence in section["text"]["sentences"]:
                sentence_tokens, sentence_lemmas, _ = zip(*sentence)
                all_sentence_tokens.append(sentence_tokens)
                all_sentence_lemmas.append(sentence_lemmas)

                merged_sentence = " ".join(sentence_tokens)
                merged_sentence = merged_sentence.replace("„", '"')
                all_sentences.append(merged_sentence)

    batch_size = 64
    lemma_embeddings = defaultdict(list)

    for k in trange(0, len(all_sentences), batch_size):
        batch = all_sentences[k : k + batch_size]
        embeddings = [np.array(embedding)[0, 1:-1, :] for embedding in embedder(batch)]
        batch_tokens_bert = [
            embedder.tokenizer.tokenize(sentence) for sentence in batch
        ]

        for idx in range(batch_size):
            if k + idx >= len(all_sentences):
                continue

            original_tokens = all_sentence_tokens[k + idx]
            sentence_bert_token_mapping = []

            for token_idx, token in enumerate(original_tokens):
                for bert_token in embedder.tokenizer.tokenize(token):
                    sentence_bert_token_mapping.append(token_idx)

            sentence_embeddings = embeddings[idx]
            bert_tokens = batch_tokens_bert[idx]
            original_lemmas = all_sentence_lemmas[k + idx]
            sentence_bert_token_mapping

            recombined_embeddings = [[] for _ in original_lemmas]

            for bert_token_idx, embedding in enumerate(sentence_embeddings):
                recombined_embeddings[
                    sentence_bert_token_mapping[bert_token_idx]
                ].append(embedding)

            recombined_embeddings = [
                np.mean(subword_embeddings, axis=0)
                for subword_embeddings in recombined_embeddings
            ]

            for lemma, embedding in zip(original_lemmas, recombined_embeddings):
                lemma_embeddings[lemma].append(embedding)

            assert len(recombined_embeddings) == len(original_lemmas)

    lemma_embeddings = {
        lemma: np.mean(embeddings, axis=0)
        for lemma, embeddings in lemma_embeddings.items()
    }
    return lemma_embeddings


def parse_cmd_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--textbook-path", type=str, default="./data/schulbuecher")
    parser.add_argument("--out-path", type=str, default="./data/")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmd_arguments()
    textbook_path = args.textbook_path

    parsed_textbooks = {
        textbook_title: get_textbook_sections(
            os.path.join(textbook_path, textbook_title)
        )
        for textbook_title in tqdm(list(sorted(os.listdir(textbook_path))))
    }

    with open(os.path.join(args.out_path, "parsed_textbooks.json"), "w") as tsf:
        json.dump(parsed_textbooks, tsf)

    lemma_embeddings = get_bert_embeddings(parsed_textbooks)

    with open(os.path.join(args.out_path, "embeddings.pickle"), "wb") as ef:
        pickle.dump(lemma_embeddings, ef)

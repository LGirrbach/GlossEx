import os
import json
import random
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from tqdm import trange
from evaluation import evaluate
from load_data import load_labels
from baselines import rake_baseline
from collections import defaultdict
from load_data import load_seedlists
from baselines import tfidf_baseline
from baselines import seedlist_baseline
from baselines import glossary_baseline
from method import method as clustering
from load_data import load_textbook_data
from load_data import load_fasttext_model
from baselines import supervised_baseline
from evaluation import get_textbook_lemmas
from background_corpora import load_background_corpus


def parse_cmd_arguments() -> argparse.Namespace:
    # Parse command line arguments
    parser = argparse.ArgumentParser("Textbook vocabulary extraction")
    parser.add_argument_group("Data paths")
    parser.add_argument(
        "--textbook-path", type=str, default="./data/parsed_textbooks.json"
    )
    parser.add_argument(
        "--lemma-embeddings-path", type=str, default="./data/embeddings.pickle"
    )
    parser.add_argument(
        "--corpus", type=str, default="dereko", choices=["dereko", "subtlex"]
    )
    parser.add_argument(
        "--seedlist",
        type=str,
        default="custom",
        choices=["custom", "wikipedia"],
    )
    parser.add_argument(
        "--seedlist-size", type=int, default=-1, help="Number of seed words to use"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations"
    )

    parser.add_argument(
        "--label-csv-path",
        type=str,
        default="./data/annotations/lemma_counts_sufficiently_frequent_Felix.csv",
    )

    parser.add_argument_group("Method")
    parser.add_argument(
        "--method",
        type=str,
        default="clustering",
        choices=["clustering", "glossary", "seedlist", "tfidf", "rake", "supervised"],
    )

    return parser.parse_args()


def make_results_filename(args: argparse.Namespace) -> str:
    if args.method == "clustering":
        seedlist_size = args.seedlist_size if args.seedlist_size > 0 else "all"
        return f"{args.method}_{args.corpus}_{args.seedlist}_{seedlist_size}"

    elif args.method == "glossary":
        return f"{args.method}"

    elif args.method == "seedlist":
        return f"{args.method}_{args.seedlist}_{args.seedlist_size}"

    elif args.method == "tfidf":
        return f"{args.method}"

    elif args.method == "rake":
        return f"{args.method}"

    elif args.method == "supervised":
        return f"{args.method}"

    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    # Set global random seed
    random.seed(42)
    np.random.seed(42)

    # Parse command line arguments
    args = parse_cmd_arguments()

    # Load parsed textbooks
    print("\nLoading parsed textbooks...")
    parsed_textbooks, lemma_embeddings = load_textbook_data(
        parsed_textbooks_path=args.textbook_path,
        lemma_embeddings_path=args.lemma_embeddings_path,
    )

    # Load fastText model
    print("\nLoading fastText model...")
    fasttext_model = load_fasttext_model()

    # Load seed lists
    print(f"\nUsing seed lists: {args.seedlist}")
    education_seeds, economics_seeds = load_seedlists(
        seedlist=args.seedlist,
    )

    # Load background corpus
    print(f"\nLoading background corpus {args.corpus}...")
    corpus = load_background_corpus(args.corpus)

    # Load labels
    print("\nLoading labels...")
    labels = load_labels(
        label_csv_path=args.label_csv_path,
    )

    # Extract lemmas from textbooks
    print("\nExtracting lemmas from textbooks...")
    textbook_lemmas = get_textbook_lemmas(
        parsed_textbooks=parsed_textbooks,
        labels=labels,
    )

    # Initialise predictions
    print("\nExtracting vocabulary...")
    print(f"Method: {args.method}")

    predictions = defaultdict(dict)
    results = dict()

    for iteration in trange(args.iterations):
        # Sample seed lists
        if args.seedlist_size > 0:
            sampled_education_seeds = random.sample(education_seeds, args.seedlist_size)
            sampled_economics_seeds = random.sample(economics_seeds, args.seedlist_size)
        else:
            sampled_education_seeds = education_seeds
            sampled_economics_seeds = economics_seeds

        # Run method for each textbook
        for textbook_title in tqdm(
            parsed_textbooks.keys(), total=len(parsed_textbooks), leave=False
        ):
            if args.method == "clustering":
                economics_words, _, _ = clustering(
                    parsed_textbooks,
                    textbook_title,
                    corpus,
                    fasttext_model,
                    sampled_education_seeds,
                    sampled_economics_seeds,
                    lemma_embeddings,
                )

            elif args.method == "glossary":
                economics_words = glossary_baseline(textbook_title)

            elif args.method == "seedlist":
                economics_words = seedlist_baseline(sampled_economics_seeds)

            elif args.method == "tfidf":
                economics_words = tfidf_baseline(
                    parsed_textbooks, textbook_title, labels, textbook_lemmas
                )

            elif args.method == "rake":
                economics_words = rake_baseline(parsed_textbooks, textbook_title)

            elif args.method == "supervised":
                economics_words = supervised_baseline(
                    textbook_title, fasttext_model, labels, textbook_lemmas
                )

            else:
                raise NotImplementedError(f"Method {args.method} not implemented")

            # Store predictions
            if textbook_title.endswith(".txt"):
                textbook_title = textbook_title[:-4]

            predictions[iteration][textbook_title] = economics_words

        # Compute metrics
        results_df = evaluate(
            predictions=predictions[iteration],
            textbook_lemmas=textbook_lemmas,
            labels=labels,
        )
        results_df["iteration"] = iteration
        results[iteration] = results_df

    # Combine results
    results_df = pd.concat(results.values(), ignore_index=True)
    # print(results_df)

    # Get result filename
    results_filename = make_results_filename(args)

    # Save predictions
    predictions_file_name = f"{results_filename}.json"

    os.makedirs("./predictions", exist_ok=True)
    save_path = os.path.join("./predictions", predictions_file_name)
    with open(save_path, "w") as f:
        json.dump(predictions, f)
    print(f"\nPredictions saved to {save_path}")

    # Save evaluation results
    results_filename = f"{results_filename}.csv"

    os.makedirs("./results", exist_ok=True)
    save_path = os.path.join("./results", results_filename)
    results_df.to_csv(save_path, index=True)
    print(f"\nEvaluation results saved to {save_path}")

    print("Done.")

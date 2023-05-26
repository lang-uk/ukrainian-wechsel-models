from typing import Dict, List
import os
from pathlib import Path
import glob
import multiprocessing
import argparse
import subprocess
import shlex

from tqdm import tqdm
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH


def load_bruk_corpus(path: str, max_bytes: int) -> List[str]:
    """
    Loads the bruk corpus from the given path, until the maximum number of bytes is reached
    Args:
        path: glob path to the bruk corpus
        max_bytes: Maximum number of bytes to load
    Returns:
        A list of articles
    """

    total_bytes = 0
    result: List[str] = []

    # At first we load the good and so-so part of the bruk corpus, to create a validation dataset
    for path in tqdm(sorted(glob.glob(path))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            total_bytes += len(text)

            # That won't happen, but just in case
            if total_bytes >= max_bytes:
                break

            result.append(text)

        if total_bytes >= max_bytes:
            break

    return result


def n_overlap(example: Dict[str, str]) -> int:
    """
    Returns the number of articles from the validation dataset that have at least some tokens in common
    """
    mh = MinHash()
    for word in example["text"].split():
        mh.update(word.encode("utf8"))

    return len(lsh.query(mh))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Prepare data for training using the Bruk corpus as validation dataset"
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=100_000_000,
        help="Maximum number of bytes to use for validation dataset",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for the minhash index"
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of permutations for the minhash index",
    )
    parser.add_argument(
        "--dataset", default="oscar", help="Dataset to use for training"
    )
    parser.add_argument(
        "--dataset-name", default=None, help="Name of the dataset to use for training"
    )
    parser.add_argument(
        "--disable-filter",
        action="store_true",
        help="Disable filtering of the dataset against validation dataset",
        default=False,
    )
    parser.add_argument(
        "--bruk-repo",
        default="https://github.com/brown-uk/corpus",
        help="Url of the bruk repo",
    )
    parser.add_argument("output", help="Output directory")
    args = parser.parse_args()

    bruk_corpus_destination = Path("bruk_corpus")
    data_destination = Path("data")
    data_destination.mkdir(exist_ok=True)

    if not bruk_corpus_destination.exists():
        # Let's clone the bruk repo
        subprocess.Popen(
            shlex.split(f"git clone {args.bruk_repo} {bruk_corpus_destination}"),
            stdout=subprocess.PIPE,
        )

    articles = load_bruk_corpus(
        path=str(bruk_corpus_destination / "data/good/*.txt"), max_bytes=args.max_bytes
    )

    # Here we are creating the minhash index from the validation dataset so later one we can
    # filter out the train dataset
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)

    for i, article in enumerate(tqdm(articles)):
        h = MinHash(num_perm=args.num_perm)
        for token in article.split():
            h.update(token.encode("utf8"))

        lsh.insert(str(i), h)

    # Let's load oscar dataset and filter out the articles that have at least
    # some overlap with the validation dataset
    train_dataset = load_dataset(
        path=args.dataset, name=args.dataset_name, split="train"
    )
    if not args.disable_filter:
        train_dataset = train_dataset.filter(
            lambda example: n_overlap(example) == 0, num_proc=multiprocessing.cpu_count()
        )

    # Save the train dataset to disk
    train_dataset.save_to_disk(data_destination / args.output)

    # Save the validation dataset to disk
    with open(
        data_destination / "bruk_valid_data.txt", "w", encoding="utf-8"
    ) as fp_out:
        fp_out.write("\n".join(articles))

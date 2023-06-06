"""
Script to export given dataset from the huggingface datasets, applying some transformations 
required for the merging process.
"""

import json
import hashlib
import argparse
from typing import Dict
from pathlib import Path

import smart_open
import ftfy
from tqdm import tqdm
import html2text
from datasets import load_dataset

h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_images = True
h.used = 0


def remove_tags(s: str) -> str:
    """
    Turn html into markdown format
    """
    global h

    if h.used > 1000:
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.used = 0
    else:
        h.used += 1

    try:
        return h.handle(s).strip()
    except AssertionError:
        print(f"AssertionError: cannot handle the '{s}'")
        return s


def process_doc(doc: Dict, args: argparse.Namespace) -> Dict:
    """
    Render doc into the jsonl format suitable for Wechsel
    Args:
        doc: doc dict from the dataset
        args: command line arguments
    Returns:
        dict with the doc in the format suitable for Wechsel
    """

    _id: str = str(doc.get("id"))
    compound_id: str = f"{args.path}.{args.name}.{args.split}.{_id}"

    return {
        "id": hashlib.sha1(compound_id.encode("utf-8")).hexdigest(),
        "compound_id": compound_id,
        "_id": str(doc.get("id")),
        "text": ftfy.fix_text(remove_tags(doc.get("text", "") or "")),
        "title": ftfy.fix_text(doc.get("title", "") or ""),
        "date_of_publish": doc.get("datetime", ""),
        "tags": [ftfy.fix_text(doc.get("owner", "") or "")],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export news dataset in the format requested by Volodymyr"
    )
    parser.add_argument(
        "--total",
        help="Total number of records in the dataset for the nice counter",
        default=22_567_099,
        type=int,
    )
    parser.add_argument("--path", help="Path to the dataset")
    parser.add_argument("--name", help="Optional name of the dataset", nargs="?", default=None)
    parser.add_argument("--split", help="Split of the dataset", default="train")
    parser.add_argument("output_file", help="path to input JSONL file", type=Path)
    args = parser.parse_args()

    dataset = load_dataset(args.path, name=args.name, split=args.split, streaming=True)

    with smart_open.open(args.output_file, "wt", encoding="utf-8") as writer:
        for doc in tqdm(dataset, total=args.total):
            writer.write(
                json.dumps(process_doc(doc, args), ensure_ascii=False, sort_keys=True) + "\n"
            )

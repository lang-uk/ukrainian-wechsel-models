"""
Combine corpora from jsonlines into an arrow file.
"""
from typing import Dict
import argparse
import lzma
import glob
import json

from tqdm import tqdm
import datasets
import smart_open


def _handle_xz(file_obj, mode):
    return lzma.LZMAFile(filename=file_obj, mode=mode, format=lzma.FORMAT_XZ)


def transform_doc(fname: str, doc: Dict) -> Dict:
    """
    Render doc into the jsonl format suitable for Wechsel
    Args:
        fname: filename
        doc: doc dict from the dataset
    Returns:
        dict with the doc in the format suitable for Wechsel
    """

    if "id" not in doc:
        doc["id"] = doc["_id"]

    if "compound_id" not in doc:
        doc["compound_id"] = f"{fname}.{doc['id']}"

    doc["text"] = doc["text"].strip()

    if "title" in doc:
        title: str = doc["title"].strip()
        if title and not doc["text"].startswith(title):
            doc["text"] = title + "\n\n" + doc["text"]

    return {f: doc[f] for f in ["id", "compound_id", "text"]}


smart_open.register_compressor(".xz", _handle_xz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine corpora from jsonlines into an arrow file."
    )
    parser.add_argument(
        "input",
        help="input glob pattern for jsonlines files",
    )
    parser.add_argument("output", help="output arrow file")

    args = parser.parse_args()

    dataset = datasets.Dataset.from_dict({"id": [], "compound_id": [], "text": []})

    for filename in tqdm(glob.glob(args.input), desc="Processing files"):
        with smart_open.open(filename, "rt", encoding="utf-8") as reader:
            for doc in map(
                json.loads,
                tqdm(reader, desc=f"Processing docs from {filename}", leave=False),
            ):
                dataset.add_item(transform_doc(filename, doc))

    dataset.save_to_disk(args.output)

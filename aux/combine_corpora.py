"""
Combine corpora from jsonlines into an arrow file.
"""
from typing import List, Iterator, TypeVar, Dict
import argparse
import lzma
import glob
import json
import multiprocessing
from functools import partial
from itertools import islice


from tqdm import tqdm
import smart_open

T = TypeVar("T")


def _handle_xz(file_obj, mode):
    return lzma.LZMAFile(filename=file_obj, mode=mode, format=lzma.FORMAT_XZ)


def batch_iterator(iterator: Iterator[T], batch_size: int = 50) -> Iterator[List[T]]:
    """
    Iterates over the given iterator in batches.
    iterator: the iterator to iterate over
    batch_size: the size of the batch
    returns an iterator over batches
    """
    iterator = iter(iterator)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def transform_doc(doc: str, fname: str) -> Dict:
    """
    Render doc into the jsonl format suitable for Wechsel
    Args:
        fname: filename
        doc: doc dict from the dataset
    Returns:
        dict with the doc in the format suitable for Wechsel
    """
    doc = json.loads(doc)
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


if __name__ == "__main__":
    smart_open.register_compressor(".xz", _handle_xz)

    parser = argparse.ArgumentParser(
        description="Combine corpora from jsonlines into an arrow file."
    )
    parser.add_argument(
        "input",
        help="input glob pattern for jsonlines files",
    )
    parser.add_argument("output", help="output jsonlines file")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="number of processes to use for parallel processing",
    )

    args = parser.parse_args()

    with smart_open.open(args.output, "wt", encoding="utf-8") as writer:
        for filename in tqdm(glob.glob(args.input), desc="Processing files"):
            with smart_open.open(filename, "rt", encoding="utf-8") as reader:
                with multiprocessing.Pool(
                    processes=args.num_processes,
                ) as pool:
                    for chunk in batch_iterator(
                        tqdm(reader, desc=f"Processing docs from {filename}"),
                        batch_size=10000,
                    ):
                        if not chunk:
                            break

                        for record in pool.imap(
                            partial(
                                transform_doc,
                                fname=filename,
                            ),
                            chunk,
                        ):
                            writer.write(
                                json.dumps(record, ensure_ascii=False, sort_keys=True)
                                + "\n"
                            )

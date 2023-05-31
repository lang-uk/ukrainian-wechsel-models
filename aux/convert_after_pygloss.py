import argparse
from tqdm import tqdm
from w3lib.html import remove_tags

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert txt dictionary produced by pyglossary to a format, suitable for WECHSEL")
    parser.add_argument("input", help="Input txt dictionary", type=argparse.FileType("r"))
    parser.add_argument("output", help="Output txt dictionary", type=argparse.FileType("w"))

    args = parser.parse_args()

    for l in map(str.strip, tqdm(args.input)):
        if l.startswith("#") or l.startswith("About dictionary"):
            continue
        
        w, t = l.split("\t")
        t = remove_tags(t).strip()

        if w and t:
            args.output.write(f"{w}\t{t}\n")

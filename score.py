import argparse
from pathlib import Path
from hashlib import sha1
import math

import torch

parser = argparse.ArgumentParser(description='compute nll/bpc/bpb on a text dataset')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('ckpt_path')
parser.add_argument('sentences', nargs='*', type=Path)
args = parser.parse_args()

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
model = AutoModelForCausalLM.from_pretrained(args.ckpt_path)
model.to(args.device)

print('id', 'sentence', 'num_tokens', 'nll', 'bpc', 'bpb', sep='\t')

for i, sentence in enumerate(s.strip() for f in args.sentences or [] for s in f.read_text().split('\n')):
    if not sentence:
        continue

    sentence_bytes = sentence.encode('utf-8')

    x = tokenizer(sentence, add_special_tokens=False)['input_ids']
    x = torch.LongTensor([tokenizer.eos_token_id] + x)
    x = x.to(args.device).long()

    with torch.inference_mode():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            y = model(input_ids=x[None, :]).logits

    x = x[1:]
    y = y[0, :x.size(-1), :] 

    log_prob_per_token = torch.nn.functional.cross_entropy(y, x)
    log_prob = log_prob_per_token.item() * x.size(-1)
    bpc = log_prob / math.log(2) / len(sentence)
    bpb = log_prob / math.log(2) / len(sentence_bytes)

    id_ = sha1(sentence_bytes).hexdigest()
    print(id_, sentence, x.size(-1), log_prob, bpc, bpb, sep='\t', flush=True)

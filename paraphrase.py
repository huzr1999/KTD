from lib.utils import get_input_col_name
from lib.gen_knockoffs import generate_knockoffs
from datasets import load_from_disk

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--tmp", default=1.9, type=float)
parser.add_argument("--topk", default=50, type=int)
parser.add_argument("--topp", default=0.95, type=float)
parser.add_argument("--early_stopping", default=False, type=bool)
parser.add_argument("--num_beams", default=1, type=int)
parser.add_argument("--do_sample", default=True, type=bool)
parser.add_argument("--num_knockoffs", default=10, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--length_penalty", default=1, type=float)

args = parser.parse_args()


topk = args.topk
topp = args.topp
temperature = args.tmp
num_knockoffs = args.num_knockoffs
early_stopping = args.early_stopping
num_beams = args.num_beams
do_sample = args.do_sample
dataset_name = args.dataset
batch_size = args.batch_size
length_penalty = args.length_penalty

input_column_name = get_input_col_name(dataset_name)

dataset = load_from_disk(f"./datasets/{dataset_name}")
dataset = generate_knockoffs(dataset, input_column_name, topk=topk, topp=topp, 
                             temperature=temperature, m=num_knockoffs, early_stopping=early_stopping, num_beams=num_beams, do_sample=do_sample, batch_size=batch_size, length_penalty=length_penalty)

dataset.save_to_disk(f"./datasets/{dataset_name}-knockoffs-{num_knockoffs}")




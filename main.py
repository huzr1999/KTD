import os
from lib.utils import load_model, load_dataset
from lib.run import run_loss, run_grad, run_topk_prob
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True, choices=["gpt2", "pythia", "gpt-neo", "pythia-410m", "pythia-1b", "pythia-2.8b", "pythia-6.9b", "mistralai"])
parser.add_argument("--method", required=True, choices=["loss", "grad", "topk_prob"])
parser.add_argument("--num_knockoffs", default=10, type=int)
parser.add_argument("--finetuned", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--k", default=0.2, type=float)
args = parser.parse_args()

dataset_name = args.dataset
model_name = args.model
method = args.method
num_knockoffs = args.num_knockoffs
finetuned = args.finetuned
overwrite = args.overwrite
k = args.k

device = "cuda:0"
batch_size = 16


dataset, input_column_name = load_dataset(dataset_name, num_knockoffs)
model, tokenizer = load_model(model_name, dataset_name, finetuned, device)

if method == "loss":
    saved_file_name = f"./saved_tensors/losses/{model_name}-{dataset_name}-{num_knockoffs}-{finetuned}.pt"
    if os.path.exists(saved_file_name) and not overwrite:
        ft_origin_loss, ft_nkf_loss = torch.load(saved_file_name)
    else:
        os.makedirs(os.path.dirname(saved_file_name), exist_ok=True)
        ft_origin_loss, ft_nkf_loss = run_loss(
            dataset, input_column_name, model, tokenizer, batch_size)

        torch.save([ft_origin_loss, ft_nkf_loss], saved_file_name)

elif method == "topk_prob":
    saved_file_name = f"./saved_tensors/topk_prob/{model_name}-{dataset_name}-{num_knockoffs}-{finetuned}.pt"
    if os.path.exists(saved_file_name) and not overwrite:
        ft_origin_topk_prob, ft_nkf_topk_prob = torch.load(saved_file_name)
    else:
        os.makedirs(os.path.dirname(saved_file_name), exist_ok=True)
        ft_origin_topk_prob, ft_nkf_topk_prob = run_topk_prob(
            dataset, input_column_name, model, tokenizer, k)

        torch.save([ft_origin_topk_prob, ft_nkf_topk_prob], saved_file_name)


elif method == "grad":
    saved_file_name = f"./saved_tensors/grad_norms/{model_name}-{dataset_name}-{num_knockoffs}-{finetuned}.pt"
    if os.path.exists(saved_file_name) and not overwrite:
        ft_origin_grad, ft_nkf_grad = torch.load(saved_file_name)
    else:
        os.makedirs(os.path.dirname(saved_file_name), exist_ok=True)
        ft_origin_grad, ft_nkf_grad = run_grad(
            dataset, input_column_name, model, tokenizer, batch_size)

        torch.save([ft_origin_grad, ft_nkf_grad], saved_file_name)
                                
else:
    raise ValueError("Invalid method!")
                            
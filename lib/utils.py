from itertools import chain
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
from lib.calc_prob import calculate_log_probabilities_batched
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from lib.knockoff import evalute_knockoff
import json
import os

def get_original_checkpoint(model_name):
    
    if model_name == "gpt2":
        checkpoint = "openai-community/gpt2"
    elif model_name == "pythia":
        checkpoint = "EleutherAI/pythia-1.4b-v0"
    elif model_name == "pythia-410m":
        checkpoint = "EleutherAI/pythia-410m-v0"
    elif model_name == "pythia-1b":
        checkpoint = "EleutherAI/pythia-1b-v0"
    elif model_name == "pythia-2.8b":
        checkpoint = "EleutherAI/pythia-2.8b-v0"
    elif model_name == "pythia-6.9b":
        checkpoint = "EleutherAI/pythia-2.8b-v0"
    elif model_name == "gpt-neo":
        checkpoint = "EleutherAI/gpt-neo-1.3B"
    elif model_name == "mistralai":
        checkpoint = "mistralai/Mistral-7B-v0.1"
    else:
        raise ValueError("Invalid model name!")
    
    return checkpoint

def get_input_col_name(dataset_name):

    if dataset_name == "wiki":
        input_column_name = "input"
    elif dataset_name == "xsum":
        input_column_name = "summary"
    elif dataset_name == "bbc":
        input_column_name = "text"
    else:
        raise ValueError("Invalide dataset name")

    return input_column_name


def load_model(model_name, dataset_name, finetunded, device):

    if finetunded:
        model_checkpoint = f"/path/to/finetuned_models/{model_name}-finetuned/{dataset_name}-finetuned/final_model"
        tokenizer_checkpoint = model_checkpoint
    else:
        model_checkpoint = get_original_checkpoint(model_name)
        tokenizer_checkpoint = model_checkpoint

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

    # model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    return model, tokenizer

def load_dataset(dataset_name, num_knockoffs):

    input_column_name = get_input_col_name(dataset_name)

    dataset_dir = os.path.abspath("./datasets")

    dataset = load_from_disk(os.path.join(dataset_dir, f"{dataset_name}-knockoffs-{num_knockoffs}"))

    return dataset, input_column_name


def run(dataset, model, tokenizer, batch_size, input_column_name, num_knockoffs):
    input_sentences = dataset[input_column_name]
    origin_log_prob = calculate_log_probabilities_batched(input_sentences, model, tokenizer, batch_size)

    input_sentences = list(chain(*dataset["knockoffs"]))
    nkf_log_prob = calculate_log_probabilities_batched(input_sentences, model, tokenizer, batch_size)

    return (origin_log_prob - nkf_log_prob.reshape(-1, num_knockoffs).mean(dim=1)).tolist()

def plot_hist(arr, x_range=None):
    if x_range == None:
        plt.hist(arr, bins=30, edgecolor="black", range=(0.8 * min(arr), 1.2 * max(arr)))
    else:
        plt.hist(arr, bins=30, edgecolor="black", range=x_range)

def filter_label(probs, dataset, label, label_name="label"):
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if label is None:
        return probs
    else:
        return probs[np.array(dataset[label_name]) == label]

def get_results(origin_metric, nkf_metric):
    num_knockoffs = len(nkf_metric) // len(origin_metric)
    return (origin_metric - 
                nkf_metric.reshape(-1, num_knockoffs).mean(dim=1)).cpu().numpy()

def eval_and_save(results, model_name, dataset_name, dataset, method, num_knockoffs):

    q_range = 0.05 * np.arange(1, 20)
    auc = evalute_knockoff(results, dataset["label"], 0.1)[0]
    fdr_wrt_q = [1 - evalute_knockoff(results, dataset["label"], q)[1] for q in q_range]
    recall_wrt_q = [evalute_knockoff(results, dataset["label"], q)[2] for q in q_range]

    exp_results = {
        "dataset": dataset_name,
        "model": model_name,
        "AUC": auc,
        "FDR": fdr_wrt_q,
        "Power": recall_wrt_q
    }

    with open(f"./results/{model_name}-{dataset_name}-{method}-{num_knockoffs}-results.json", "w") as f:
        json.dump(exp_results, f)

def calc_grad_l2_norm(grad):
    return torch.sqrt(grad.sum(dim=-1))
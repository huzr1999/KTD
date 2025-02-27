from itertools import chain
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
from calc_prob import calculate_log_probabilities_batched
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from datasets import load_from_disk
from knockoff import evalute_knockoff
import json
import os

def get_original_checkpoint(model_name):
    
    if model_name == "gpt2":
        checkpoint = "openai-community/gpt2"
    elif model_name == "pythia":
        checkpoint = "EleutherAI/pythia-1.4b-v0"
    elif model_name == "pythia-410m":
        checkpoint = "/root/autodl-tmp/.cache/hub/pythia-410m"
    elif model_name == "pythia-1b":
        checkpoint = "/root/autodl-tmp/.cache/hub/pythia-1b"
    elif model_name == "pythia-2.8b":
        checkpoint = "/root/autodl-tmp/.cache/hub/pythia-2.8b"
    elif model_name == "pythia-6.9b":
        checkpoint = "/root/autodl-tmp/.cache/hub/pythia-6.9b"
    elif model_name == "gpt-neo":
        checkpoint = "EleutherAI/gpt-neo-1.3B"
    elif model_name == "mistralai":
        # checkpoint = "/root/autodl-tmp/.cache/mistralai-7b"
        checkpoint = "/root/autodl-tmp/mistralai-finetuned/bbc-fine-tuned"
    else:
        raise ValueError("Invalid model name!")
    
    return checkpoint

def get_input_col_name(dataset_name):

    if dataset_name.endswith("pretraining"):
        dataset_name = dataset_name.replace("-pretraining", "")

    if dataset_name == "tofu":
        input_column_name = "question"
    elif dataset_name == "wiki":
        input_column_name = "input"
    elif dataset_name == "xsum":
        input_column_name = "summary"
    elif dataset_name == "bbc":
        input_column_name = "text"
    elif dataset_name == "tinystory":
        input_column_name = "text"
    elif dataset_name == "pile_cc":
        input_column_name = "text"
    elif dataset_name == "agnews":
        input_column_name = "text"
    elif dataset_name == "hackernews":
        input_column_name = "text"
    elif dataset_name == "arxiv":
        input_column_name = "text"
    elif dataset_name == "emotion":
        input_column_name = "text"

    else:
        raise ValueError("Invalide dataset name")

    return input_column_name


def load_model(model_name, dataset_name, finetunded, device):

    if finetunded:
        model_checkpoint = f"/root/autodl-tmp/{model_name}-finetuned/{dataset_name}-fine-tuned"
        tokenizer_checkpoint = model_checkpoint
    else:
        model_checkpoint = get_original_checkpoint(model_name)
        tokenizer_checkpoint = model_checkpoint

    # if use_lora:
    #     if model_name == "gpt2":
    #         base_model_checkpoint = "openai-community/gpt2"
    #     elif model_name == "pythia":
    #         base_model_checkpoint = "EleutherAI/pythia-1.4b-v0"
    #     elif model_name == "gpt-neo":
    #         base_model_checkpoint = "EleutherAI/gpt-neo-1.3B"

    #     base_model = AutoModelForCausalLM.from_pretrained(base_model_checkpoint)
    #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    #     base_model.resize_token_embeddings(len(tokenizer))

    #     model = PeftModel.from_pretrained(base_model, model_checkpoint)
    #     model.merge_and
        
    # else:
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

    model.resize_token_embeddings(len(tokenizer))
    # model.to(device)
    model.eval()
    print(model.device)

    return model, tokenizer

def load_dataset(dataset_name, num_knockoffs, alt=False):

    input_column_name = get_input_col_name(dataset_name)

    dataset_dir = os.path.abspath("../datasets")

    if alt:
        dataset = load_from_disk(os.path.join(dataset_dir, f"{dataset_name}-knockoffs_alt-{num_knockoffs}"))
    else:
        dataset = load_from_disk(os.path.join(dataset_dir, f"/root/Code/ICLR_code/datasets/{dataset_name}-knockoffs-{num_knockoffs}"))

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

def get_results_alt(origin_metric, nkf_metric, n):
    num_knockoffs = len(nkf_metric) // len(origin_metric)
    return (origin_metric - 
                nkf_metric.reshape(-1, num_knockoffs)[:, :n].mean(dim=1)).cpu().numpy()


def plot_two_hist(results, dataset):

    _, ax = plt.subplots(1)
    ax.hist(filter_label(results, dataset, 1), bins=30, edgecolor="black", alpha=0.5, label="1")
    ax.hist(filter_label(results, dataset, 0), bins=30, edgecolor="black", alpha=0.5, label="0")
    ax.legend()

def kde_find_peak(results, ax, prominence=0.05):
    pdf = gaussian_kde(results, bw_method="silverman")

    x = np.linspace(min(results), max(results), 1000)
    y = pdf(x)

    peaks, _ = find_peaks(y, prominence=0.05)
    offset = x[peaks[0]]

    ax.plot(x, y)
    return offset

def directly_find_peak(results, ax, prominence=0.05):

    if len(results) > 5000:
        nbins = len(results) // 100
    else:
        nbins = 50

    freq, val = np.histogram(results, nbins, density=True)
    peaks, _ = find_peaks(freq, prominence=prominence)

    offset = (val[peaks[0]] + val[peaks[0] + 1]) / 2
    ax.hist(results, bins=nbins, edgecolor="black")
    return offset

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

def truncate_dataset(dataset, input_column_name, max_length):
    def truncate_text(example):
        example[input_column_name] = example[input_column_name][:max_length]
        return example

    return dataset.map(truncate_text)

def plot_dataset_token_length_dist(dataset, input_column_name):
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    fig, ax = plt.subplots(1)
    ax.hist([len(t) for t in tokenizer(dataset[input_column_name]).input_ids], bins=30, edgecolor="black")
    ax.hist([len(t) for t in tokenizer(list(chain(*dataset["knockoffs"]))).input_ids], bins=30, edgecolor="black")

    return ax


# def plot_two_hist(arr1, arr2, dataset, label):

#     num_knockoffs = len(arr2) // len(arr1)

#     arr1 = filter_label(arr1, dataset, label)
#     arr2 = filter_label(arr2.reshape(-1, num_knockoffs).mean(dim=1), dataset, label)

#     fig, ax = plt.subplots(1)
#     ax.hist(arr1, bins=30, edgecolor="black", alpha=0.5, label="origin")
#     ax.hist(arr2, bins=30, edgecolor="black", alpha=0.5, label="knockoffs")
#     ax.legend()
from lib.calc_prob import calculate_log_probabilities_batched, calculate_topk_log_probability
from itertools import chain
from tqdm import tqdm
import torch
from torch import linalg as LA

def run_loss(dataset, input_column_name, model, tokenizer, batch_size):

    
    input_sentences = dataset[input_column_name]
    ft_origin_log_prob = calculate_log_probabilities_batched(input_sentences, model, tokenizer, batch_size)

    input_sentences = list(chain(*dataset["knockoffs"]))
    ft_nkf_log_prob = calculate_log_probabilities_batched(input_sentences, model, tokenizer, batch_size)

    return ft_origin_log_prob, ft_nkf_log_prob


def run_topk_prob(dataset, input_column_name, model, tokenizer, k):

    result_list = []
    for text in tqdm(dataset[input_column_name]):
        topk_log_prob = calculate_topk_log_probability(text, model, tokenizer, k)
        result_list.append(topk_log_prob)
    ft_origin_log_prob = torch.stack(result_list)

    result_list = []
    for text in tqdm(list(chain(*dataset["knockoffs"]))):
        topk_log_prob = calculate_topk_log_probability(text, model, tokenizer, k)
        result_list.append(topk_log_prob)
    ft_nkf_log_prob = torch.stack(result_list)

    return ft_origin_log_prob, ft_nkf_log_prob


def run_grad(dataset, input_column_name, model, tokenizer, batch_size):

    origin_grad_norms = []
    for text in tqdm(dataset[input_column_name]):

        model.zero_grad()
        tokenized_text = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        tokenized_text = {k: v.to(model.device) for k, v in tokenized_text.items()}
        loss = model(**tokenized_text, labels=tokenized_text["input_ids"].to(model.device)).loss
        loss.backward()

        origin_grad_norms.append(
            torch.stack([(LA.vector_norm(p.grad) ** 2).cpu() for p in model.parameters() if p.grad is not None])
        )

    nkf_grad_norms = []
    for text in tqdm(list(chain(*dataset["knockoffs"]))):
        model.zero_grad()
        tokenized_text = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        tokenized_text = {k: v.to(model.device) for k, v in tokenized_text.items()}
        loss = model(**tokenized_text, labels=tokenized_text["input_ids"]).loss
        loss.backward()

        nkf_grad_norms.append(
            torch.stack([(LA.vector_norm(p.grad) ** 2).cpu() for p in model.parameters() if p.grad is not None])
        )

    origin_grad_norms = torch.stack(origin_grad_norms)
    nkf_grad_norms = torch.stack(nkf_grad_norms)

    return origin_grad_norms, nkf_grad_norms

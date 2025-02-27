import torch
import torch.nn.functional as F
from tqdm import tqdm

def calculate_log_probability(sentence, model, tokenizer):

    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=1024, return_tensors='pt')

    # print(inputs)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}

    labels = inputs["input_ids"][:, 1:]
    attention_mask = inputs["attention_mask"][:, 1:]

    # print(labels.shape)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])


    probabilities = F.softmax(outputs.logits, dim=-1)

    log_probabilities = torch.log(probabilities[:, :-1, :].gather(2, labels.unsqueeze(-1)).squeeze(-1))

    log_probabilities = torch.where(labels != tokenizer.pad_token_id, log_probabilities, torch.zeros_like(log_probabilities))

    log_probabilities = torch.where(log_probabilities != float('-inf'), log_probabilities, torch.zeros_like(log_probabilities))

    total_log_probabilities = torch.sum(log_probabilities, dim=1) / torch.sum(attention_mask, dim=1)
    # total_log_probabilities = torch.sum(log_probabilities, dim=1)

    return total_log_probabilities

def calculate_topk_log_probability(sentence, model, tokenizer, k):

    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=1024, return_tensors='pt')

    # print(inputs)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}

    labels = inputs["input_ids"][:, 1:]
    attention_mask = inputs["attention_mask"][:, 1:]

    # print(labels.shape)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])


    probabilities = F.softmax(outputs.logits, dim=-1)

    log_probabilities = torch.log(probabilities[:, :-1, :].gather(-1, labels.unsqueeze(-1)).squeeze(-1))

    # log_probabilities = torch.where(labels != tokenizer.pad_token_id, log_probabilities, torch.zeros_like(log_probabilities))

    log_probabilities = torch.where(log_probabilities != float('-inf'), log_probabilities, torch.zeros_like(log_probabilities))

    non_zero_mask = log_probabilities != 0
    sorted_log_prob = torch.sort(log_probabilities[non_zero_mask], descending=True)[0]

    return sorted_log_prob[:int(k * non_zero_mask.sum())].mean()


def calculate_log_probabilities_batched(input_sentences, model, tokenizer, batch_size):
    tensor_list = []

    for i in tqdm(range(0, len(input_sentences), batch_size)):


        batched_sentence = input_sentences[i:i+batch_size]
        batched_log_probability = calculate_log_probability(batched_sentence, model, tokenizer)
        tensor_list.append(batched_log_probability)

    # print(tensor_list)
    return torch.cat(tensor_list)
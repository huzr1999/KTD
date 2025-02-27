from transformers.pipelines.pt_utils import KeyDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from tqdm import tqdm


def generate_knockoffs(dataset, input_column_name, topk=10, topp=0.95, temperature=1.9, 
                        m=10, early_stopping=True, num_beams=1, do_sample=True, batch_size=4, length_penalty=1):

    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    model = T5ForConditionalGeneration.from_pretrained(model_name, )
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

    # generator = pipeline("text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base", device=0)
    # generator = pipeline("text2text-generation", model="prithivida/parrot_paraphraser_on_T5", device=0)


    output_list = []
    for out in tqdm(generator(
        KeyDataset(dataset, input_column_name),
        max_length=128,  # Set this to a higher value to allow longer text
        do_sample=do_sample,   # Optional: Use sampling to introduce diversity
        top_k=topk,         # Optional: Top-k sampling for diversity
        top_p=topp,       # Optional: Nucleus sampling for diversity
        temperature=temperature,  # Optional: Controls the randomness of the output
        num_return_sequences=m,  # Number of different outputs to return
        batch_size=batch_size,
        early_stopping=early_stopping,
        num_beams=num_beams,
        length_penalty=length_penalty
        # num_beam_groups=num_beams,
        # diversity_penalty=1.0
        )):
        output_list.append(list(map(lambda x: x['generated_text'], out)))

    if "knockoffs" in dataset.column_names:
        dataset = dataset.remove_columns(["knockoffs"])

    dataset = dataset.add_column("knockoffs", output_list)

    return dataset

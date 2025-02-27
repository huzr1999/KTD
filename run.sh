#!/bin/bash
datasets=("wiki" "xsum" "bbc")
models=("gpt2" "pythia" "gpt-neo")

for dataset in "${datasets[@]}"; do

    # python paraphrase.py --num_knockoffs 10 --dataset "$dataset"

    for model in "${models[@]}"; do
        if [ "$model" == "gpt2" ]; then
            epochs=10
        else
            epochs=3 
        fi

        echo "Running experiments for dataset: $dataset, model: $model, epochs: $epochs"


        python finetuning.py --dataset "$dataset" --model "$model" --epochs "$epochs" --batch_size 8

        python main.py --dataset "$dataset" --model "$model" --method grad --num_knockoffs 10 --overwrite
    done
done

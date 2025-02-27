# A Statistical Approach for Controlled Training Data Detection  

This repository contains the code necessary to reproduce the results from the paper *A Statistical Approach for Controlled Training Data Detection*, published at ICLR 2025. This document provides instructions for setting up the required environment, running experiments, and reproducing the results.  

Our proposed method, **KTD**, leverages knockoff inference to detect training data in large language models. For a detailed explanation of the methodology, please refer to our paper.  

## Installation  

To set up the required environment, run the following commands:  

```bash
conda create -n torch python=3.10 -y
conda activate torch
pip install -r requirements.txt
```  

## Running Experiments  

You can execute all experiments by running the provided script:  

```bash
bash run.sh
```  

Alternatively, you can run customized experiments by specifying the model and dataset manually. For example:  

```bash
python finetuning.py --dataset bbc --model pythia --epochs 3
python main.py --dataset bbc --model pythia --method grad --num_knockoffs 10
```  

The computed gradient results will be stored in `./saved_tensors/grad_norm`. To visualize the FDR control results, run the Jupyter notebook:  

```bash
jupyter notebook plot.ipynb
```  

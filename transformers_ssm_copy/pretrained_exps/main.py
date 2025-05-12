import itertools
import os
import argparse
from datasets import load_dataset, DatasetDict

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

import transformers

from collections import Counter 
from pathlib import Path

import string
from model_utils import get_model, get_tokenizer
from test_utils import copy_c4_evaluation, phone_book_evaluation, squad_evaluation 

def parse_args():
    parser = argparse.ArgumentParser()

    ##model
    parser.add_argument('--model', choices=["state-spaces/mamba-370m","state-spaces/mamba-1.4b","state-spaces/mamba-2.8b","EleutherAI/pythia-410m","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b"], type=str, required=True)
    
    ##evaluation
    parser.add_argument('--eval_task', choices=["c4_copy","squad","phone_book"], type=str, required=True, help="evaluation task")
    
    parser.add_argument('--text_order', choices=["standard","random"], type=str, help="only applies when eval_task = c4_copy. Order of the text to copy. When text_order = random, we randomly change the order of the text. Otherwise, keeps the same order.", default="standard")
    
    parser.add_argument('--eval_batch_size', default=32, type=int, help="Size of the batch size for evaluation. Only applies when eval_task == ''c4_copy'' or eval_task == 'phone_book'")
    parser.add_argument('--eval_num_batches', default=3, type=int, help="Number of batches for the evaluation to compute mean + std. Only applies when eval_task == ''c4_copy'' or eval_task == 'phone_book'")
    
    parser.add_argument('--min_eval_len', default=20, type=int, help="Minimum length of the text to copy (when eval_task == ''c4_copy'') or minimum number of entries in the phone book (when eval_task == ''phone_book''). Only applies when eval_task == ''c4_copy'' or eval_task == 'phone_book")
    
    parser.add_argument('--max_eval_len', default=20, type=int, help="Maximum length of the text to copy (when eval_task == ''c4_copy'') or maximum number of entries in the phone book (when eval_task == ''phone_book''). Only applies when eval_task == ''c4_copy'' or eval_task == 'phone_book")

    
    return parser.parse_args()

args = parse_args()
if not hasattr(args, "text_order"):
    args.text_order = "natural"


print(args)



## Get tokenizer & model
tokenizer = get_tokenizer()
model = get_model(args)
print("^"*100)
print(model)
print("^"*100)
model.eval()



### Evaluation
if args.eval_task == "c4_copy":
    str_acc_mean_list, str_acc_std_list = copy_c4_evaluation(args,model,tokenizer)
    import csv
    import os

    csv_path = "c4_results.csv"
    write_header = not os.path.exists(csv_path)  # Only write header if file doesn't exist

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "length", "accuracy", "std"])

        lengths = np.arange(args.min_eval_len, args.max_eval_len + 1)
        for length, acc, std in zip(lengths, str_acc_mean_list, str_acc_std_list):
            writer.writerow([args.model, length, acc, std])

elif args.eval_task == "phone_book":
    str_acc_mean_list, str_acc_std_list = phone_book_evaluation(args,model,tokenizer)
elif args.eval_task == "squad":
    import csv
    import torch
    from transformers import AutoTokenizer
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
    from test_utils import squad_evaluation

    # Load tokenizer manually (from EleutherAI tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

    # Load Mamba model + config using Mamba's utils
    from mamba_ssm.models.config_mamba import MambaConfig
    raw_config = load_config_hf("state-spaces/mamba-370m")
    config = MambaConfig(**raw_config)  # Convert dict to MambaConfig
    model = MambaLMHeadModel(config)

    model.load_state_dict(load_state_dict_hf("state-spaces/mamba-370m"))
    model = model.cuda().eval()

    # Patch pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run evaluation
    em_list, f1_list, std_em_list, std_f1_list, lengths = squad_evaluation(args, model, tokenizer)

    # Save results
    with open("squad_results.csv", "a") as f:
        writer = csv.writer(f)
        for length, em, f1, em_std, f1_std in zip(lengths, em_list, f1_list, std_em_list, std_f1_list):
            writer.writerow([args.model, length, em, em_std, f1, f1_std])




else:
    raise ValueError(f"Non-valid evaluation task {args.eval_task}")



print("DONE")




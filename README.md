# Transformers vs SSMs: Replication Project

This repository contains the replication of two key experiments from the paper:

**"Repeat After Me: Transformers are Better than State Space Models at Copying"**  
[arXiv link](https://arxiv.org/abs/2306.11695)

---

## 🧪 Tasks Replicated

### 1️⃣ Synthetic Copying Task

We compare the performance of **Hard-ALiBi Transformers** and **Mamba** on a synthetic string copying task, as described in the paper.

#### ▶️ How to Run:

Navigate to the synthetic experiment directory:
```bash
cd transformers_ssm_copy/synthetic_exps

To run both the model for various training steps, use 

python run_experiments.py

To generate final accuracy plot, run 

python plot_results.py

This will create:

string_accuracy_vs_train_steps.png and char_accuracy_vs_train_steps.png

### 1️⃣ Question Answering (SQuAD v2)

We evaluate Mamba and Pythia (a Transformer) on the SQuAD v2 dataset to compare their performance on real-world QA.

#### ▶️ How to Run:

Navigate to the pretrained QA experiment directory:

cd transformers_ssm_copy/pretrained_exps

Run evaluation for each model:

# For Mamba
python3 main.py --model_name mamba --eval_dataset squad_v2

# For Pythia
python3 main.py --model_name pythia --eval_dataset squad_v2

Evaluation result are stored in :

squad_results.csv

To generate the F1 score vs. context length graph:

python plot_graph.py

This produces 
f1_vs_context_length.png


import subprocess

# Settings
models = ["mamba", "T_hard_alibi"]
steps_list = [250, 500, 1000, 2000, 4000]
hidden_size = 384
layers = 6
heads = 6
masked_heads = 3
state_dim = 64

for model in models:
    print(f"\nRunning {model} across training sizes...")
    for steps in steps_list:
        print(f"\n🚀 Training {model} for {steps} steps (~{steps * 8} samples)")

        cmd = [
            "python3", "main.py",
            "--model", model,
            "--train_task", "copy",
            "--eval_task", "copy",
            "--min_train_len", "5",
            "--max_train_len", "20",
            "--min_eval_len", "20",
            "--max_eval_len", "20",
            "--steps", str(steps),
            "--hidden_size", str(hidden_size),
            "--layers", str(layers)
        ]

        if model == "mamba":
            cmd += ["--state_dim", str(state_dim)]
        elif model == "T_hard_alibi":
            cmd += ["--heads", str(heads), "--num_masked_heads", str(masked_heads)]

        subprocess.run(cmd)

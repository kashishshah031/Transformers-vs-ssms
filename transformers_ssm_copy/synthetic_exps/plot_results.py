import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results.csv")

# Group by model
mamba = df[df["model"] == "mamba"]
alibi = df[df["model"] == "T_hard_alibi"]

# String-level accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(mamba["steps"], mamba["acc_str"], marker='o', label="Mamba")
plt.plot(alibi["steps"], alibi["acc_str"], marker='o', label="T_hard_alibi")
plt.xlabel("Training steps")
plt.ylabel("String Accuracy (%)")
plt.title("String Accuracy vs Training steps")
plt.grid(True)
plt.legend()
plt.savefig("string_accuracy_plot.png")
plt.show()

# Character-level accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(mamba["steps"], mamba["acc_char"], marker='o', label="Mamba")
plt.plot(alibi["steps"], alibi["acc_char"], marker='o', label="T_hard_alibi")
plt.xlabel("Training steps")
plt.ylabel("Character Accuracy (%)")
plt.title("Character Accuracy vs Training steps")
plt.grid(True)
plt.legend()
plt.savefig("char_accuracy_plot.png")
plt.show()

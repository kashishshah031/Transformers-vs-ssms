import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("squad_results.csv", skiprows=1, header=None, names=["model", "length", "em", "em_std", "f1", "f1_std"])

# Convert relevant columns to numeric types
df["length"] = pd.to_numeric(df["length"])
df["em"] = pd.to_numeric(df["em"])
df["em_std"] = pd.to_numeric(df["em_std"])
df["f1"] = pd.to_numeric(df["f1"])
df["f1_std"] = pd.to_numeric(df["f1_std"])

# Filter models
pythia = df[df["model"] == "EleutherAI/pythia-410m"]
mamba = df[df["model"] == "state-spaces/mamba-370m"]

# Plot F1 graph
plt.figure(figsize=(10, 6))
plt.plot(pythia["length"], pythia["f1"] * 100, label="Pythia", marker="o")
#plt.fill_between(
#    pythia["length"],
#    (pythia["f1"] - pythia["f1_std"]) * 100,
#    (pythia["f1"] + pythia["f1_std"]) * 100,
#    alpha=0.2
#)
plt.plot(mamba["length"], mamba["f1"] * 100, label="Mamba", marker="s")
#plt.fill_between(
#    mamba["length"],
#    (mamba["f1"] - mamba["f1_std"]) * 100,
#    (mamba["f1"] + mamba["f1_std"]) * 100,
#    alpha=0.2
#)
plt.xlabel("Number of Context Tokens")
plt.ylabel("F1 Accuracy (%)")
plt.title("F1 Accuracy vs Context Length (SQuAD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("f1_vs_context_length.png")
plt.show()

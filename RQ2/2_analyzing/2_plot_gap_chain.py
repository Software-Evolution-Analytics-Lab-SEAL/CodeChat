import pandas as pd
import csv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


CATEGORY_MAP = {
    "1": "Missing Specifications",
    "2": "Different Use Cases",
    "3": "Incremental Problem Solving",
    "4": "Exploring Alternative Approaches",
    "5": "Wordy Response",
    "6": "Additional Functionality",
    "7": "Erroneous Response",
    "8": "Missing Context",
    "9": "Clarity of Generated Response"
    # Add more if needed
}

# ----------------------------
# Step 1: Generate gap_chain.csv
# ----------------------------
def generate_gap_chains(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    # Drop missing labels
    df = df[pd.notna(df["predicted_category_number"])]
    df["predicted_category_number"] = df["predicted_category_number"].astype(int)

    # Sort and group by conversation_id
    chain_df = (
        df.sort_values(["conversation_id", "turn1_number"])
          .groupby("conversation_id")["predicted_category_number"]
          .apply(lambda x: "-".join(map(str, x)))
          .reset_index(name="gap_chain")
    )

    chain_df.to_csv(output_csv_path, index=False)
    print(f"[✓] Saved gap chains to: {output_csv_path}")

# ----------------------------
# Step 2: Plot top 3-grams from gap chains
# ----------------------------
def label_ngram(ngram_str):
    nums = ngram_str.split("-")
    labels = [CATEGORY_MAP.get(n, f"Unknown({n})") for n in nums]
    return " → ".join(labels)

def load_gap_chains(csv_path):
    chains = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["gap_chain"].strip()
            if raw:
                chain = raw.split("-")
                chains.append(chain)
    return chains

def plot_gap_3grams_named_percentage(csv_path, top_n=10):
    chains = load_gap_chains(csv_path)

    ngram_counter = Counter()
    for chain in chains:
        if len(chain) >= 3:
            for i in range(len(chain) - 2):
                ngram = chain[i:i+3]
                ngram_str = "-".join(ngram)
                ngram_counter[ngram_str] += 1

    total_ngrams = sum(ngram_counter.values())
    print(f"✅ Total 3-grams: {total_ngrams}\n")
    top_ngrams = ngram_counter.most_common(top_n)

    labels = [label_ngram(ng) for ng, _ in top_ngrams]
    proportions = [count / total_ngrams * 100 for _, count in top_ngrams]

    # --- Print table ---
    print("Top 3-gram Gap Chains:")
    print(f"{'Rank':<5} {'Chain (numbers)':<20} {'%':<8} {'Labels'}")
    print("-" * 80)
    for idx, ((ngram_str, count), pct, lbl) in enumerate(zip(top_ngrams, proportions, labels), start=1):
        print(f"{idx:<5} {ngram_str:<20} {pct:>6.2f}%   {lbl}")
    print()
    
    # Plot
    plt.figure(figsize=(12, 6))
    bar_height = 0.55
    y_positions = np.arange(len(labels)) * (bar_height * 1.3)

    bars = plt.barh(y_positions, proportions, height=bar_height, color="#5f5f5f")
    plt.gca().invert_yaxis()

    max_prop = max(proportions)
    label_offset = max_prop * 0.04 + 20.0
    label_position = max_prop + label_offset

    for bar, pct in zip(bars, proportions):
        plt.text(label_position,
                 bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%",
                 va='center',
                 ha='right',
                 fontsize=20)

    # Grid and styling
    plt.xlim(0, label_position + max_prop * 0.05)
    for x in np.linspace(0, max_prop, 4):
        plt.axvline(x, ymin=0.05, ymax=0.95, color="gray", linestyle="--", alpha=0.7)

    plt.yticks(y_positions, labels, fontsize=17, weight="bold")
    plt.gca().spines[["top", "right", "left", "bottom"]].set_visible(False)
    plt.gca().xaxis.set_visible(False)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # --- Config ---
    input_csv = "../rq2_results/1_topic7_gap_labeled.csv"
    gap_chain_csv = "../rq2_results/2_gap_chains.csv"
    top_n = 10
    generate_gap_chains(input_csv, gap_chain_csv)
    plot_gap_3grams_named_percentage(gap_chain_csv, top_n=top_n)

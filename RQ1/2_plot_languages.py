import os
import json
import re
import csv
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from collections import Counter 
import ast
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import stats
import sys
sys.path.append('..')

from utilities.rq1_utilities import SnippetProcessor
import utilities.cliffdelta as cliffdelta

LANGUAGE_ALIASES = {
        "js": "javascript", 
        "py": "python",
        "python3": "python",
        "c++": "cpp",
        "c#" : "csharp",
        "rb": "ruby", 
        "ts": "typescript",
        "pl" : "perl",
        "kt" : "kotlin",
        "md" : "markdown",
        "yml" : "yaml",
        "htm" : "html",
        "visualbasic":"vba",
        "visual-basic":"vba"
    }

def plot_lan_distri_vertical(input_csv, num=20, filename="total_distribution_vertical.png"):
    # Load CSV
    df = pd.read_csv(input_csv)
    total_df = df[df["type"] == "total"].iloc[1:].copy()  # skip "all" row

    # Sort and take top num
    top_data = total_df.sort_values(by="percentage", ascending=False).head(num).copy()
    top_data["percentage"] = np.ceil(top_data["percentage"]).astype(int)

    # Calculate "Others"
    others_percentage = total_df.loc[~total_df.index.isin(top_data.index), "percentage"].sum()
    others_count = total_df.loc[~total_df.index.isin(top_data.index), "count"].sum()
    others_row = pd.DataFrame([{"language": "Others", "percentage": others_percentage, "count": others_count}])
    top_data = pd.concat([top_data, others_row], ignore_index=True)

    # Capitalize + fix language names
    top_data["language"] = top_data["language"].str.capitalize().replace({
        "Javascript": "JavaScript",
        "Html": "HTML",
        "Sql": "SQL",
        "Json": "JSON",
        "Css": "CSS",
        "Php": "PHP",
        "Matlab": "MATLAB",
        "Xml": "XML",
        "Typescript": "TypeScript",
        "Jsx": "JSX",
        "Cpp": "C++",
        "Csharp": "C#"
    })

    # Colors (same dimgray style)
    cmap = LinearSegmentedColormap.from_list("custom", ["dimgray", "dimgray"])
    colors = cmap(np.linspace(0, 1, len(top_data))).tolist()
    colors[-1] = colors[-2]  # keep "Others" same as previous

    # Plot
    plt.figure(figsize=(12, 4))
    bars = plt.bar(top_data["language"], top_data["percentage"], color=colors)

    # Font sizes same as original function
    plt.xticks(fontsize=16, fontweight='bold', rotation=35, ha="right")
    plt.yticks(fontsize=16, fontweight='bold')

    # Remove top/right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Percentage labels on top of bars
    for i, bar in enumerate(bars):
        percentage_value = top_data["percentage"][i]
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{percentage_value:.0f}%",
            ha='center',
            va='bottom',
            fontsize=18
        )

    # Y-axis gridlines (dashed, like original)
    max_value = top_data["percentage"].max()
    for y in np.linspace(0, max_value, 5):
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.ylabel("Percentage (%)", fontsize=16, fontweight='bold')
    # plt.xlabel("Language", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # # step 2.2.1 plot the bar chart with top numbers
    input_csv = "./rq1_results/2_2taglan_distri.csv"
    # plot_lan_distri(input_csv,num=20)
    plot_lan_distri_vertical(input_csv,num=20)
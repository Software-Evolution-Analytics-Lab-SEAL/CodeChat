from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def extract_topic_name(openai_text: str | None) -> str:
    """Extracts <topic_label>...</topic_label> from OpenAI text."""
    match = re.search(r'<topic_label>\s*(.*?)\s*</topic_label>', str(openai_text))
    return match.group(1) if match else "Unknown"


def enhance_cluster_map_with_labels(
    cluster_map_csv: str | Path,
    topics_info_csv: str | Path,
    output_csv: str | Path,
) -> Dict[int, str]:
    """
    Enhances the topic map CSV with OpenAI topic names and returns
    a label_map for final_cluster_id → final_OpenAIname.
    """
    cluster_df = pd.read_csv(cluster_map_csv)
    topics_df = pd.read_csv(topics_info_csv)

    topics_df["label"] = topics_df["OpenAI"].apply(extract_topic_name)
    topic_label_map = dict(zip(topics_df["Topic"], topics_df["label"]))

    cluster_df["Openai_name"] = cluster_df["original_cluster"].map(topic_label_map)
    cluster_df["aftermerged_OpenAIname"] = cluster_df["merged_cluster"].map(topic_label_map)

    ordered_cols = [
        "original_cluster", "original_count", "Openai_name",
        "merged_cluster", "merged_total_count",
        "final_cluster_id", "aftermerged_OpenAIname",
    ]
    cluster_df = cluster_df[ordered_cols]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(output_csv, index=False)
    print(f"✅ Enhanced topic map saved to {output_csv}")

    # Build robust label_map (skip NaNs, skip negatives)
    label_map: Dict[int, str] = {}
    for _, row in cluster_df.iterrows():
        fc = row.get("final_cluster_id", None)
        if pd.notna(fc):
            fc_int = int(fc)
            if fc_int >= 0:
                label_map[fc_int] = row.get("aftermerged_OpenAIname", "Unknown")
    return label_map


lang_stats: Dict[int, List[Tuple[str, float]]] = {
    0: [("HTML", 39.0), ("Python", 32.3), ("JavaScript", 26.6)],   # Topic 1
    1: [("Python", 72.9), ("Bash", 9.8), ("JSON", 8.1)],           # Topic 2
    2: [("C++", 39.8), ("C", 16.4), ("Java", 15.4)],               # Topic 3
    3: [("SQL", 37.7), ("Python", 27.4), ("C#", 10.6)],            # Topic 4
    4: [("Python", 31.1), ("C#", 25.1), ("Lua", 15.2)],            # Topic 5
    5: [("Python", 56.6), ("C++", 9.7), ("Bash", 7.0)],            # Topic 6
    6: [("C++", 51.2), ("C", 23.3), ("C#", 5.4)],                  # Topic 7
    7: [("Python", 34.4), ("JavaScript", 8.7), ("Bash", 7.3)],     # Topic 8
    8: [("VBA", 41.5), ("Python", 37.8), ("VB", 6.2)],             # Topic 9
    9: [("Dart", 29.3), ("Java", 24.4), ("XML", 23.0)],            # Topic 10
}

def format_lang_line(lang_list: List[Tuple[str, float]], max_langs: int = 3) -> str:
    """Return 'HTML 39% · Python 32% · JavaScript 27%'."""
    parts = [f"{name} {round(pct)}%" for name, pct in lang_list[:max_langs]]
    return " · ".join(parts)

def plot_from_final_csv_with_labels_and_langs(
    final_csv: str | Path,
    label_map: Optional[Dict[int, str]] = None,
    top_n: int = 10,
    output_png: Optional[str] = None,
    bar_scale: float = 0.90,
) -> None:
    """
    Load final distribution CSV, count conversations per final_cluster_id (0–9),
    and plot top clusters with two-line y-labels.
    """
    df = pd.read_csv(final_csv)
    print(f"Loaded {len(df)} rows from {final_csv}")

    cluster_counts: Dict[int, int] = {}
    id_cols = [c for c in df.columns if re.fullmatch(r"cluster_\d+_id", c)]
    for _, row in df.iterrows():
        seen_clusters = set()
        for col in id_cols:
            val = row[col]
            if pd.notna(val):
                try:
                    cid = int(val)
                    if 0 <= cid <= 9:
                        seen_clusters.add(cid)
                except (ValueError, TypeError):
                    pass
        for cid in seen_clusters:
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    if not cluster_counts:
        print("❌ No valid clusters (IDs 0–9) found.")
        return

    top_series = pd.Series(cluster_counts).sort_values(ascending=False).head(top_n)
    total_convs = int(df["conversation_id"].nunique())

    # Build display labels: "Topic i: <name>" in the sorted order
    display_label_map: Dict[int, str] = {}
    lm = label_map or {}
    for display_idx, real_cluster in enumerate(top_series.index, start=1):
        real_label = lm.get(real_cluster, "Unknown")
        display_label_map[real_cluster] = f"Topic {display_idx}: {real_label}"

    print("\n✅ Final Labels, Counts, Percentages for Plotting (Clusters 0–9 Only):")
    for cluster_id, count in top_series.items():
        pct = round(count / total_convs * 100, 1)
        print(f"{display_label_map[cluster_id]} - Count: {count}, Percentage: {pct}%")

    plot_rearranged_clusters_with_labels_and_langs(
        top_counts=top_series,
        total_conversations=total_convs,
        label_map=display_label_map,
        lang_stats=lang_stats,
        top_n=top_n,
        output_png=output_png or "rq2_top_topics_with_langs.png",
        topic_fontsize=16,
        langs_fontsize=14,
        bar_scale=0.60,  # ← pass through
        left_margin_inch=9.0,  # Left margin in inches
    )
 
def plot_rearranged_clusters_with_labels_and_langs(
    top_counts: pd.Series,
    total_conversations: int,
    label_map: Dict[int, str],
    lang_stats: Dict[int, List[Tuple[str, float]]],
    top_n: int = 10,
    output_png: Optional[str] = None,
    topic_fontsize: int = 16,
    langs_fontsize: int = 12,
    bar_scale: float = 0.75,
    left_margin_inch: float = 8.0,  # Left margin in inches
) -> None:
    """
    Corrected version with proper topic ordering (1 to 10 top to bottom)
    and aligned language labels.
    """
    # Sort and select top topics (already in correct 1-10 order)
    top_counts = top_counts.sort_values(ascending=False).head(top_n)
    counts = top_counts.values  # Don't reverse for plotting
    
    # Build labels in correct order (Topic 1 first)
    y_labels = []
    for cid in top_counts.index:
        topic_label = label_map.get(cid, f"Topic {cid}: Unknown")
        langs = lang_stats.get(cid, [])
        # lang_line = " · ".join([f"{lang} {pct}%" for lang, pct in langs[:3]])
        lang_line = "(" + " · ".join([f"{lang} {pct}%" for lang, pct in langs[:3]]) + ")"
        y_labels.append((topic_label, lang_line))
    
    percentages = (counts / total_conversations * 100).round(1)

    # Layout
    n_bars = len(counts)
    # fig, ax = plt.subplots(figsize=(12, max(6, 0.9 * n_bars)))
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # print(max(6, 0.9 * n_bars), "rows in figure..........")
    bar_height = 1.3
    spacing = 1.55
    y_pos = np.arange(n_bars) * (bar_height * spacing)
    
    # Plot bars (in correct order)
    # bars = ax.barh(y_pos, counts*0.5, height=bar_height, color="dimgray")
    bars = ax.barh(y_pos, counts*bar_scale, height=bar_height, color="dimgray")
    
    for x in np.linspace(0, counts.max()*bar_scale, 4):
        ax.axvline(x, ymin=0.08, ymax=0.95, linestyle="--", alpha=0.35, color="dimgray")
        
    # Right-side percentages
    max_count = counts.max()
    right_pad = max_count * 0.22
    for bar, pct in zip(bars, percentages):
        ax.text(
            # max_count*bar_scale + right_pad*1.25,
            max_count*bar_scale*1.25 + right_pad *1.25,
            bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}%",
            va="center",
            ha="right",
            fontsize=20,
            color="#111111"
        )

    # Left-side labels - now in correct order
    left_margin = max_count * 0.3
    for y, (topic, langs), count in zip(y_pos, y_labels, counts):
        ax.text(
            -left_margin * 0.1,
            y - bar_height * 0.35,
            topic,
            ha="right",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="#111111"
        )
        ax.text(
            -left_margin * 0.1,
            y + bar_height * 0.25,  # Adjusted language label position
            langs,
            ha="right",
            va="center",
            fontsize=16,
            color="#111111"
        )

    # Correct axis limits
    ax.set_xlim(-left_margin, max_count + right_pad * 1.5)
    ax.set_ylim(-bar_height, y_pos[-1] + bar_height * 1.6)
    ax.invert_yaxis()  # This flips to show Topic 1 at top
    
    ax.axis("off")
    plt.tight_layout()
    fig_width_inch = 12
    # left_margin_inch = 8.0
    plt.subplots_adjust(left=left_margin_inch/fig_width_inch)
    
    plt.show()
       
# ──────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cluster_map_csv = Path("../rq2_results/1_fullturn/2_analyzing/0_6final_cluster_map.csv")
    topics_info_csv = Path("../rq2_results/1_fullturn/1_training/topics_info.csv")
    output_csv = Path("../rq2_results/1_fullturn/2_analyzing/0_9enhanced_cluster_map_with_names.csv")
    label_map = enhance_cluster_map_with_labels(cluster_map_csv, topics_info_csv, output_csv)

    final_distribution_csv = Path("../rq2_results/1_fullturn/2_analyzing/0_7final_cluster_distribution.csv")
    output_png = None  # or set a path
    plot_from_final_csv_with_labels_and_langs(
        final_distribution_csv,
        label_map=label_map,
        top_n=10,
        output_png=output_png,
        bar_scale=1.0,  # change to 0.10 if you want very short bars
    )
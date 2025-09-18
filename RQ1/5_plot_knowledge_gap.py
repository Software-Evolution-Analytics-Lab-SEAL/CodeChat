import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CATEGORY_MAP = {
    1: "Missing Specifications",
    2: "Different Use Cases",
    3: "Incremental Problem Solving",
    4: "Exploring Alternative Approaches",
    5: "Wordy Response",
    6: "Additional Functionality",
    7: "Erroneous Response",
    8: "Missing Context",
    9: "Clarity of Generated Response",
    10: "Inaccurate/Untrustworthy Response",
    11: "Miscellaneous"
}

def load_data(gap_csv, turns_csv):
    df_gap = pd.read_csv(gap_csv).dropna(subset=['predicted_category_number','sz_number'])
    df_gap['predicted_category_number'] = df_gap['predicted_category_number'].astype(int)
    df_turns = pd.read_csv(turns_csv)
    return df_gap, df_turns

def summarize_gap(df):
    total = len(df)
    summ = df.groupby('predicted_category_number').size().reset_index(name='Count')
    summ['Percentage'] = (summ['Count']/total*100).round(1)
    all_cats = pd.DataFrame({'predicted_category_number': range(1,12)})
    summ = pd.merge(all_cats, summ, on='predicted_category_number', how='left').fillna(0)
    summ['Category'] = summ['predicted_category_number'].map(CATEGORY_MAP)
    summ = summ.sort_values(by='Count', ascending=False).reset_index(drop=True)

    # Prepend blank row to shift bars down
    empty_row = pd.DataFrame([{'predicted_category_number': 0, 'Count': 0, 'Percentage': 0, 'Category': ''}])
    summ = pd.concat([empty_row, summ], ignore_index=True)
    return summ

def summarize_turns(df, top_n=10):
    total_conv = df['conversation_id'].nunique()
    tc = df['turn'].value_counts().reset_index()
    tc.columns = ['turn','Count']
    tc['Percentage'] = tc['Count']/total_conv*100
    tc = tc.sort_values(by='turn')
    tc = tc.sort_values(by='Percentage', ascending=False).head(top_n)
    others_pct = 100 - tc['Percentage'].sum()
    tc = tc.reset_index(drop=True)
    tc = pd.concat([tc, pd.DataFrame([{'turn':'10+','Count':None,'Percentage':others_pct}])]).reset_index(drop=True)
    return tc

def plot_combined(gap, turns):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})
    bar_h = 0.5
    bar_gap = bar_h * 1.3

    yA = np.arange(len(turns)) * bar_gap
    yB = np.arange(len(gap)) * bar_gap

    # --- Plot A: Turns Distribution ---
    cmap = LinearSegmentedColormap.from_list("custom", ["dimgray", "dimgray"])
    colorsA = cmap(np.linspace(0, 1, len(turns)))
    barsA = axA.barh(yA, turns['Percentage'], height=bar_h, color=colorsA)
    axA.invert_yaxis()
    axA.set_yticks(yA)
    axA.set_yticklabels(turns['turn'], fontsize=16, weight='bold')
    axA.set_xlim(0, 40)
    # axA.set_title("a) Conversation Turns Distribution", fontsize=20, fontweight='bold')
    axA.set_xticks(np.linspace(0, 40, 5))
    # axA.grid(axis='x', linestyle='--', color='gray', alpha=0.7)
    # Add 4 evenly spaced dashed lines with start and end control
    # max_x = axA.get_xlim()[1]
    max_bar_xa = max(bar.get_width() for bar in barsA)
    for x in np.linspace(0, max_bar_xa, 5):
        axA.axvline(
            x=x,
            ymin=0.05,  # 5% from bottom
            ymax=0.95,  # 95% to top
            linestyle='--',
            color='gray',
            alpha=0.7,
            linewidth=1
        )

    axA.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axA.set_ylabel("Number of Turns", fontsize=18)#, weight='bold')  # <-- Add this line
    for spine in axA.spines.values():
        spine.set_visible(False)

    max_width_A = max(turns['Percentage'])
    for bar, pct in zip(barsA, turns['Percentage']):
        axA.text(max_width_A + 1, bar.get_y() + bar.get_height() / 2, f"{pct:.0f}%", va='center', ha='left', fontsize=18)

    # --- Plot B: Knowledge Gap Categories (with shifted empty top row) ---
    barsB = axB.barh(yB, gap['Count'], height=bar_h, color="#5f5f5f")
    axB.invert_yaxis()
    axB.set_yticks(yB)
    axB.set_yticklabels(gap['Category'], fontsize=16, weight='bold')
    axB.set_xlim(0, gap['Count'].max() * 1.3)
    axB.set_xticks(np.linspace(0, gap['Count'].max() * 1.2, 5))
    # Add 4 evenly spaced dashed vertical lines with y-range control in axB
    # max_x = axB.get_xlim()[1]
    max_bar_x = max(bar.get_width() for bar in barsB)
    for x in np.linspace(0, max_bar_x, 5):
        axB.axvline(
            x=x,
            ymin=0.05,  # 5% from bottom of axB
            ymax=0.85,  # 95% to top of axB
            linestyle='--',
            color='gray',
            alpha=0.7,
            linewidth=1
        )
    axB.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    for spine in axB.spines.values():
        spine.set_visible(False)

    max_width_B = gap['Count'][1:].max()  # skip the empty bar
    for bar, pct in zip(barsB[1:], gap['Percentage'][1:]):
        axB.text(max_width_B + 3, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va='center', ha='left', fontsize=18)
        # axB.text(max_width_B + 10, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va='center', ha='left', fontsize=18)

    # --- Red alignment line: from end of bar 2 in left to start of bar 1 in right ---
    if len(turns) >= 2 and len(gap) >= 2:
        y_data = yA[0] - bar_h + bar_gap / 16  # Center line between first two bars
        xA = 0
        xB = 0
        xA_fig, y_fig = axA.transData.transform((xA, y_data))
        xB_fig, _     = axB.transData.transform((xB, y_data))
        xA_fig, y_fig = fig.transFigure.inverted().transform((xA_fig, y_fig))
        xB_fig, _     = fig.transFigure.inverted().transform((xB_fig, y_fig))
        xA_fig -= 0.04
        xB_fig += 0.40
        line = plt.Line2D([xA_fig, xB_fig], [y_fig, y_fig],
                          transform=fig.transFigure,
                          color='blue', linestyle='--', linewidth=2)
        fig.add_artist(line)
    
    axA.text(0.5, -0.01, "(a)", transform=axA.transAxes,
            ha='center', va='center', fontsize=14)

    axB.text(0.5, -0.01, "(b)", transform=axB.transAxes,
             ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    df_gap, df_turns = load_data(
        './rq1_results/RQ1_labelled.csv',
        './rq1_results/3_1conversation_turns_times.csv'
    )
    gap_summary = summarize_gap(df_gap)
    turns_summary = summarize_turns(df_turns, top_n=10)
    plot_combined(gap_summary, turns_summary)

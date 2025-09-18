import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines 
from collections import Counter
sns.set(style="whitegrid")

def get_top_msg_types_from_csv(input_csv, msg_type_col, turn_col, conversation_col, top_n=5, exclude_types=None):
    """
    Get top message types from turn 1 with flexible column mapping.
    """
    exclude_types = exclude_types or []
    df = pd.read_csv(input_csv)
    df[msg_type_col] = df[msg_type_col].str.strip()

    df_turn1 = df[df[turn_col] == 1]
    msg_type_counts = Counter(df_turn1[msg_type_col])

    filtered_counts = [
        (msg, count) for msg, count in msg_type_counts.most_common()
        if msg not in exclude_types
    ]

    top_msg_types = [msg for msg, _ in filtered_counts[:top_n]]
    print(f"[✓] Top {top_n} message types (excluding {exclude_types}): {top_msg_types}")
    return top_msg_types

def compute_msg_type_presence_percentage(input_csv, msg_type_col, turn_col, conversation_col, top_msg_types, max_turn=5):
    """
    Compute percentage of conversations per turn containing a msg_type with flexible column mapping.
    """
    df = pd.read_csv(input_csv)
    df[msg_type_col] = df[msg_type_col].str.strip()

    turn_counts = df.groupby(conversation_col)[turn_col].max()
    results = []

    for msg_type in top_msg_types:
        for turn in range(1, max_turn + 1):
            eligible_convos = turn_counts[turn_counts >= turn].index
            total_eligible = len(eligible_convos)

            df_turn = df[
                (df[turn_col] == turn) &
                (df[conversation_col].isin(eligible_convos))
            ]

            affected_convos = df_turn[df_turn[msg_type_col] == msg_type][conversation_col].nunique()
            percent = (affected_convos / total_eligible * 100) if total_eligible > 0 else 0.0

            results.append({
                'msg_type': msg_type,
                'turn_number': turn,
                'total_eligible_conversations': total_eligible,
                'conversations_with_msg_type': affected_convos,
                'percent_conversations_with_msg_type': round(percent, 2)
            })

    return pd.DataFrame(results)

def plot_msg_type_presence_percentage_sub(ax, df_result, title, max_turn=5, line_width=2.8, marker_size=9, x_padding=0.1):
    label_mapping = {
        "trailing-whitespace": "TrailingWhitespace",
        "invalid-name": "InvalidName",
        "undefined-variable": "UndefVar",
        "import-error": "ImportError",
        "no-undef": "VarUndef",
        "no-unused-vars": "UnusedVars",
        "missingIncludeSystem": "MissingIncludeSys",
        "unusedFunction": "UnusedFunction",
        "syntaxError": "SyntaxError",
        "CommentRequired": "CommentRequired",
        "LocalVariableCouldBeFinal": "LocalVarFinal",
        "MethodArgumentCouldBeFinal": "MethodArgFinal",
        "CS0246": "CS0246",
        "CS0103": "CS0103",
        # "CS0246": "CS0246:NamespaceUnfound",
        # "CS0103": "CS0103:NameIdMissing",
        "EnableGenerateDocumentationFile": "EnableDocGen"
    }

    df_result['msg_type_display'] = df_result['msg_type'].apply(lambda x: label_mapping.get(x, x))
    df_result['turn_number'] = df_result['turn_number'].astype(int)

    # Panel label
    language = title.split(" - ")[0]
    language_map = {
        "Python": "(a) Python", "JavaScript": "(b) JavaScript",
        "C++": "(c) C++", "Java": "(d) Java", "C#": "(e) C#"
    }
    panel_label = language_map.get(language, "")
    ax.text(0.5, -0.32, panel_label, transform=ax.transAxes,
            ha='center', va='center', fontsize=18, style='italic')

    # Plot
    sns.lineplot(
        data=df_result,
        x='turn_number',
        y='percent_conversations_with_msg_type',
        hue='msg_type_display',
        linewidth=line_width,
        marker='o',
        markersize=marker_size,
        ax=ax
    )
    # Axes
    ax.set_xlabel("")
    # ax.set_ylabel("% of Convo", fontsize=16)
    ax.set_ylabel("")
    # ax.set_ylabel("% of Convo", fontsize=16)
    
    ax.set_ylim(0,120)
    ax.set_xlim(1 - x_padding, max_turn + x_padding)
    ax.set_xticks(range(1, max_turn + 1))
    ax.set_xticklabels([f"{x}" for x in range(1, max_turn + 1)], fontsize=16)
    ax.set_xlabel("Turn", fontsize=16)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{y}%" for y in range(0, 101, 20)], fontsize=16)#, weight='bold')
    # ax.set_yticklabels([f"{y}" for y in range(0, 101, 20)], fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(alpha=0.2, linestyle='--')

    if language == "Python":
        handles, labels = ax.get_legend_handles_labels()

        if len(handles) == 3:
            # Dummy invisible handle to insert vertical space
            spacer_handle = mlines.Line2D([], [], linestyle='None', color='None')

            custom_handles = [handles[0], spacer_handle, handles[1], handles[2]]
            custom_labels  = [labels[0], "", labels[1], labels[2]]

            legend = ax.legend(
                handles=custom_handles,
                labels=custom_labels,
                title="",
                loc='upper right',
                bbox_to_anchor=(1.0, 1.05),  # move legend up
                frameon=False,
                ncol=1,                      # <- one column
                columnspacing=0.6,
                handletextpad=0.4,
                labelspacing=0.6,           # <- more space between entries
                borderaxespad=0.3,
                prop={'weight': 'bold', 'size': 14}
            )
    else:
        legend = ax.legend(title="", 
                           loc='upper right', 
                           bbox_to_anchor=(1.07, 1.08),
                           fontsize=14,
                           frameon=False,
                           prop={'weight': 'bold', 'size': 14})

    # Style
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")
        
def process_and_plot_language(ax, config, top_n=3, max_turn=5):
    """
    Process a dataset and plot onto a subplot, using flexible config.
    Also prints per-turn percentages per message type.
    """
    top_msg_types = get_top_msg_types_from_csv(
        config['input_csv'],
        config['msg_type_col'],
        config['turn_col'],
        config['conversation_col'],
        top_n=top_n,
        exclude_types=config.get('exclude_types', [])
    )

    df_percent = compute_msg_type_presence_percentage(
        config['input_csv'],
        config['msg_type_col'],
        config['turn_col'],
        config['conversation_col'],
        top_msg_types,
        max_turn=max_turn
    )

    language_name = config['language_name']
    print(f"\n[✓] Percentage Breakdown for {language_name}:")
    
    for msg_type in top_msg_types:
        print(f"  Issue: {msg_type}")
        df_issue = df_percent[df_percent['msg_type'] == msg_type].sort_values('turn_number')
        
        for _, row in df_issue.iterrows():
            turn = row['turn_number']
            percent = row['percent_conversations_with_msg_type']
            total = row['total_eligible_conversations']
            affected = row['conversations_with_msg_type']
            print(f"    Turn {turn}: {percent}% ({affected}/{total} conversations)")

    plot_msg_type_presence_percentage_sub(
        ax,
        df_percent,
        f"{language_name} - % Convo with Issues",
        max_turn=max_turn
    )

def multi_language_msg_type_comparison(language_configs, top_n=3, max_turn=5):
    """
    Plot multi-language msg_type trends in a 2x3 grid layout.
    
    Args:
        language_configs: List of per-language configs
        top_n: Top N message types per language
        max_turn: Max turn number to analyze
    """
    num_languages = len(language_configs)
    rows, cols = 1,5 #2, 3  # Fixed 2x3 layout

    # fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharey=False)
    # plt.subplots_adjust(hspace=-0.4)  # Default is ~0.2, increase for more gap

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), sharey=True, constrained_layout=True)
    # fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), constrained_layout=True)
    print("....................", cols * 4)
    # Flatten axes for easy iteration
    axes = axes.flatten()

    for idx, config in enumerate(language_configs):
        if idx >= len(axes):
            break  # Prevent overflow if fewer than 6 axes

        ax = axes[idx]
        process_and_plot_language(ax, config, top_n=top_n, max_turn=max_turn)

    # Hide unused subplots if < 6 languages
    for j in range(num_languages, len(axes)):
        fig.delaxes(axes[j])
        
    # fig.text(-0.05, 0.5, "% of Convo", va='center', rotation='vertical', fontsize=12)#, weight='bold')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    language_configs = [
        {
            'language_name': 'Python',
            'input_csv': './rq3_results/1_py_detailed_c4_followupturn.csv',
            'msg_type_col': 'msg_type',
            'turn_col': 'turn_number',
            'conversation_col': 'conversation_id',
            'exclude_types': ['trailing-whitespace'] #,'import-error'] #,'undefined-variable']
        },
        {
            'language_name': 'JavaScript',
            'input_csv': './rq3_results/2_js_detailed_c4_followupturn.csv',
            'msg_type_col': 'issue_id',
            'turn_col': 'turn_number',
            'conversation_col': 'conversation_id',
            'exclude_types': []  # Optional
        },
        {
            'language_name': 'C++',
            'input_csv': './rq3_results/3_cpp_detailed_c4_followupturn.csv',
            'msg_type_col': 'issue_id',
            'turn_col': 'turn_number',
            'conversation_col': 'conversation_id',
            'exclude_types': []
        },
        {
            'language_name': 'Java',
            'input_csv': './rq3_results/4_java_detailed_c4_followupturn.csv',
            'msg_type_col': 'Rule',  # Unique identifier in Java CSV
            'turn_col': 'turn_number',
            'conversation_col': 'conversation_id',
            'exclude_types': []
        },
        {
            'language_name': 'C#',
            'input_csv': './rq3_results/5_csharp_detailed_c4_followupturn.csv',
            'msg_type_col': 'rule_id',  # Unique identifier in C# CSV
            'turn_col': 'turn_number',
            'conversation_col': 'conversation_id',
            'exclude_types': []
        }
    ]

    multi_language_msg_type_comparison(language_configs, top_n=3, max_turn=5)
